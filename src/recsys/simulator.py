"""The synthetic world, used as ground truth: lessons, learners and how learners react.

Nothing here is learned. Every rule is written down so the effects in docs/results.md can be
traced back to a line of code. The effects were planted on purpose:

1. **Interest:** each learner has hidden topic interests (a sparse Dirichlet draw). Lessons in a
   liked topic are chosen more often and finished more often. Collaborative filtering can learn
   this from co-occurrence, because learners with similar interests start similar lessons.
2. **Difficulty gap:** each learner has a hidden skill on the same 1-5 scale as lesson levels.
   A lesson far above the learner's skill, or with an unmet prerequisite, is usually abandoned.
   A lesson far below it is finished but boring.
3. **Coming back:** a hidden engagement value scales the chance of being active each day.
   Finishing a lesson in the "flow zone" raises it, abandoning lowers it, boredom lowers it a
   little. So what you recommend today changes whether the learner returns tomorrow.
4. **Learning:** finishing a lesson near your level raises your skill a little.
5. **Cold start:** 30% of learners arrive after day 0, so some have no history at a given time.

The logging policy (how learners found lessons before any recommender existed) is a mix of
"next lesson in the same track", a global popular shelf, and level-aware browsing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .domain import EVENT_COLUMNS, Catalog, LearnerHistory

TOPICS = (
    "grammar",
    "vocabulary",
    "listening",
    "speaking",
    "reading",
    "writing",
    "pronunciation",
    "ielts",
    "business",
    "travel",
)
N_LEVELS = 5
MINUTE = 1.0 / 1440.0

# A policy sees only what a real system could see: the learner's history and the time.
Policy = Callable[[LearnerHistory, float], Sequence[int]]


@dataclass(frozen=True)
class SimConfig:
    # Logging policy (organic discovery)
    continue_prob: float = 0.35  # open the next lesson in the same topic track
    shelf_prob: float = 0.25  # pick from the global "popular" shelf (level-blind)
    shelf_size: int = 30
    browse_level_weight: float = 0.7  # learners can see level labels while browsing
    # Reaction to a recommendation slate
    follow_prob: float = 0.7  # chance to pick from the slate instead of browsing
    position_weight: float = 0.3  # mild position bias inside the slate
    # Completion model: logit = base + w_i*log(interest) + w_a*appeal - slope*max(0, gap - margin)
    base_logit: float = 1.5
    interest_weight: float = 0.5
    appeal_weight: float = 0.5
    hard_slope: float = 2.5
    hard_margin: float = 0.3
    prereq_gap: float = 1.0  # an unmet prerequisite makes a lesson feel one level harder
    # Learning and engagement
    learn_rate: float = 0.02
    easy_gap: float = -0.8  # gap below this is "too easy"
    flow_boost: float = 0.05
    bored_drop: float = 0.03
    abandon_drop: float = 0.06
    engagement_home: float = 0.9
    engagement_reversion: float = 0.05
    engagement_bounds: tuple[float, float] = (0.1, 1.3)
    new_share: float = 0.3  # learners who arrive after day 0


@dataclass(frozen=True)
class Population:
    """Hidden learner parameters. Models never see these."""

    learner_ids: tuple[str, ...]
    skill0: np.ndarray
    interest: np.ndarray  # (n_learners, n_topics), rows sum to 1
    activity: np.ndarray  # base chance of a study day
    pace: np.ndarray  # mean number of extra lessons per study session
    arrival: np.ndarray  # first possible study day (int)


@dataclass(frozen=True)
class World:
    catalog: Catalog
    appeal: np.ndarray  # hidden lesson attractiveness
    population: Population
    config: SimConfig = field(default_factory=SimConfig)


@dataclass
class State:
    """Everything that changes while the world runs."""

    skill: np.ndarray
    engagement: np.ndarray
    histories: list[LearnerHistory]

    def copy(self) -> State:
        return deepcopy(self)


def make_catalog(seed: int = 0, per_topic: int = 30) -> tuple[Catalog, np.ndarray]:
    """300 lessons: 10 topics x 30, levels 1-5 in order, each lesson above level 1 needs one or
    two lessons one level lower in the same topic. Returns the catalogue and hidden appeal."""
    rng = np.random.default_rng(seed)
    per_level = per_topic // N_LEVELS
    topic: list[int] = []
    level: list[int] = []
    seq: list[int] = []
    prereqs: list[tuple[int, ...]] = []
    for t in range(len(TOPICS)):
        levels = np.repeat(np.arange(1, N_LEVELS + 1), per_level)
        base = len(topic)
        for s, lv in enumerate(levels):
            topic.append(t)
            level.append(int(lv))
            seq.append(s)
            if lv == 1:
                prereqs.append(())
                continue
            lower = base + np.flatnonzero(levels == lv - 1)
            n_req = 2 if rng.random() < 0.3 else 1
            prereqs.append(tuple(sorted(int(x) for x in rng.choice(lower, n_req, replace=False))))
    ids = tuple(f"{TOPICS[t][:4]}_{s:02d}" for t, s in zip(topic, seq, strict=True))
    catalog = Catalog(TOPICS, np.array(topic), np.array(level), np.array(seq), tuple(prereqs), ids)
    appeal = rng.normal(0.0, 0.6, len(topic))
    return catalog, appeal


def make_population(n: int, n_days: int, seed: int, cfg: SimConfig) -> Population:
    """Beginners are the largest group, as in most language apps."""
    rng = np.random.default_rng(seed)
    group = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
    centre = np.array([1.6, 2.8, 4.0])[group]
    skill = np.clip(rng.normal(centre, 0.4), 0.8, 5.2)
    interest = rng.dirichlet(np.full(len(TOPICS), 0.4), size=n)
    activity = rng.beta(2.0, 3.0, size=n)
    pace = rng.gamma(2.0, 0.6, size=n)
    late = rng.random(n) < cfg.new_share
    arrival = np.where(late, rng.integers(1, max(n_days, 2), size=n), 0)
    ids = tuple(f"L{u:04d}" for u in range(n))
    return Population(ids, skill, interest, activity, pace, arrival)


def make_world(n_learners: int, n_days: int, seed: int = 0) -> World:
    cfg = SimConfig()
    catalog, appeal = make_catalog(seed)
    return World(catalog, appeal, make_population(n_learners, n_days, seed + 1, cfg), cfg)


def initial_state(world: World) -> State:
    n = len(world.population.learner_ids)
    return State(
        skill=world.population.skill0.copy(),
        engagement=np.full(n, world.config.engagement_home),
        histories=[LearnerHistory() for _ in range(n)],
    )


def difficulty_gap(world: World, skill: float, history: LearnerHistory, lesson: int) -> float:
    """Level minus skill, plus a penalty when a prerequisite is neither done nor clearly mastered.

    A learner whose skill is a full level above a prerequisite counts as knowing it already
    (placement), so advanced newcomers are not forced through beginner lessons.
    """
    cat = world.catalog
    unmet = any(
        p not in history.completed and skill < cat.level[p] + 1 for p in cat.prereqs[lesson]
    )
    return float(cat.level[lesson] - skill + world.config.prereq_gap * unmet)


def p_complete(world: World, u: int, skill: float, history: LearnerHistory, lesson: int) -> float:
    cfg = world.config
    gap = difficulty_gap(world, skill, history, lesson)
    interest = _interest_match(world, u)[world.catalog.topic[lesson]]
    logit = (
        cfg.base_logit
        + cfg.interest_weight * np.log(interest)
        + cfg.appeal_weight * world.appeal[lesson]
        - cfg.hard_slope * max(0.0, gap - cfg.hard_margin)
    )
    return float(1.0 / (1.0 + np.exp(-logit)))


def _interest_match(world: World, u: int) -> np.ndarray:
    """Interest relative to uniform (1 = average), clipped so log-interest stays bounded."""
    pop = world.population
    return np.clip(pop.interest[u] * pop.interest.shape[1], 0.1, 5.0)


def _gumbel_pick(rng: np.random.Generator, util: np.ndarray, cand: np.ndarray) -> int:
    """Softmax choice via the Gumbel-max trick (one draw per candidate)."""
    noisy = util + rng.gumbel(size=len(util))
    return int(cand[int(np.argmax(noisy))])


def _organic_choice(world: World, state: State, u: int, rng: np.random.Generator) -> int:
    cat, cfg = world.catalog, world.config
    h = state.histories[u]
    avail = np.ones(len(cat), dtype=bool)
    avail[list(h.started)] = False
    if not avail.any():
        return -1
    last = h.last_lesson
    if last >= 0 and rng.random() < cfg.continue_prob:
        same = np.flatnonzero(avail & (cat.topic == cat.topic[last]))
        if len(same):
            return int(same[np.argmin(cat.seq[same])])
    log_m = np.log(_interest_match(world, u))[cat.topic]
    if rng.random() < cfg.shelf_prob:
        shelf = np.argsort(-world.appeal)[: cfg.shelf_size]
        cand = shelf[avail[shelf]]
        if len(cand):
            return _gumbel_pick(rng, 1.2 * log_m[cand], cand)
    cand = np.flatnonzero(avail)
    util = (
        1.2 * log_m[cand]
        + 0.6 * world.appeal[cand]
        - cfg.browse_level_weight * np.abs(cat.level[cand] - state.skill[u] - 0.3)
    )
    return _gumbel_pick(rng, util, cand)


def _slate_choice(
    world: World, state: State, u: int, slate: Sequence[int], rng: np.random.Generator
) -> int:
    """Pick from the slate by interest and appeal (difficulty is only known after starting)."""
    h = state.histories[u]
    cand = np.array([i for i in slate if i not in h.started], dtype=int)
    if len(cand) == 0 or rng.random() >= world.config.follow_prob:
        return _organic_choice(world, state, u, rng)
    log_m = np.log(_interest_match(world, u))[world.catalog.topic[cand]]
    util = (
        1.2 * log_m + 0.6 * world.appeal[cand] - world.config.position_weight * np.arange(len(cand))
    )
    return _gumbel_pick(rng, util, cand)


def _react(world: World, state: State, u: int, lesson: int, rng: np.random.Generator) -> bool:
    """Start the lesson; return True if completed. Updates skill and engagement."""
    cfg = world.config
    h = state.histories[u]
    skill = float(state.skill[u])
    gap = difficulty_gap(world, skill, h, lesson)
    done = rng.random() < p_complete(world, u, skill, h, lesson)
    if done and cfg.easy_gap <= gap <= 1.0:
        state.skill[u] += cfg.learn_rate
        state.engagement[u] += cfg.flow_boost
    elif done and gap < cfg.easy_gap:
        state.engagement[u] -= cfg.bored_drop
    elif not done:
        state.engagement[u] -= cfg.abandon_drop
    low, high = cfg.engagement_bounds
    state.engagement[u] = float(np.clip(state.engagement[u], low, high))
    return bool(done)


def simulate(
    world: World,
    state: State,
    start_day: int,
    n_days: int,
    rng: np.random.Generator,
    policies: Sequence[Policy | None] | None = None,
) -> pd.DataFrame:
    """Run the world for `n_days` from `start_day`, mutating `state`; return the event log.

    `policies[u]` is the recommender shown to learner u (None = no recommender, organic only).
    Each study day a learner tries 1-6 lessons (1 + Poisson(pace)); after an abandon they
    stop with probability 0.5.
    Every attempt logs a `view` and then a `complete` or `abandon` 8-25 minutes later.
    """
    pop, cfg = world.population, world.config
    rows: list[tuple[str, str, float, str]] = []
    ids = world.catalog.lesson_ids
    n = len(pop.learner_ids)
    for day in range(start_day, start_day + n_days):
        p_active = np.clip(pop.activity * state.engagement, 0.0, 0.95)
        active = (pop.arrival <= day) & (rng.random(n) < p_active)
        for u in np.flatnonzero(active).tolist():
            t = day + rng.uniform(0.3, 0.9)
            policy = policies[u] if policies is not None else None
            for _ in range(1 + min(int(rng.poisson(pop.pace[u])), 5)):
                h = state.histories[u]
                if policy is None:
                    lesson = _organic_choice(world, state, u, rng)
                else:
                    lesson = _slate_choice(world, state, u, policy(h, t), rng)
                if lesson < 0:
                    break
                done = _react(world, state, u, lesson, rng)
                t_end = t + rng.uniform(8, 25) * MINUTE
                outcome = "complete" if done else "abandon"
                h.add(t, lesson, "view")
                h.add(t_end, lesson, outcome)
                rows.append((pop.learner_ids[u], ids[lesson], t, "view"))
                rows.append((pop.learner_ids[u], ids[lesson], t_end, outcome))
                t = t_end + 2 * MINUTE
                if not done and rng.random() < 0.5:
                    break
        state.engagement += cfg.engagement_reversion * (cfg.engagement_home - state.engagement)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def generate_history(world: World, n_days: int, seed: int) -> tuple[pd.DataFrame, State]:
    """The pre-recommender era: organic discovery only. Returns the log and the final state."""
    state = initial_state(world)
    events = simulate(world, state, 0, n_days, np.random.default_rng(seed))
    return events, state

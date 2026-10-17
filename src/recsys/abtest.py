"""Simulated online A/B test: deploy two policies in the simulator and measure what happens.

Setup: the world has run for the history period with no recommender. On the next day every
learner is randomised 50/50 to policy A (most popular) or B (ALS + learning-aware re-rank +
cold-start fallback). Both are trained on the full history and then frozen. For N days each
study session shows a slate of k lessons from the learner's policy; the learner picks from it
(or browses, see `SimConfig.follow_prob`).

Metrics, all per randomised learner (so learners who never come back count as zero):

- **Primary:** lessons completed per learner per week.
- **Guardrail:** abandonment rate = abandoned starts / all starts (a ratio of sums, so its
  standard error uses the delta method).
- **Secondary:** active days per learner per week (coming back), and true skill gain, which only
  a simulator can measure.

CUPED (Deng et al. 2013): Y_adj = Y - theta * (X - mean(X)) with X = lessons completed per
week in the pre-period and theta = cov(X, Y) / var(X) pooled over both arms. X was fixed before
randomisation, so the adjustment cannot bias the effect; it only removes the part of Y's
variance that pre-period activity already explains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .policy import Ranker
from .simulator import Policy, State, World, simulate

DAYS_PER_WEEK = 7.0


@dataclass(frozen=True)
class ABConfig:
    n_days: int = 28
    pre_days: int = 28
    k: int = 5
    seed: int = 2026
    alpha: float = 0.05


@dataclass(frozen=True)
class Effect:
    control: float
    treatment: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def rel(self) -> float:
        return self.diff / self.control if self.control else float("nan")


def as_policy(ranker: Ranker, k: int) -> Policy:
    return lambda history, now: ranker.recommend(history, k).items


def per_learner(
    events: pd.DataFrame, learner_ids: tuple[str, ...], start: float, end: float
) -> pd.DataFrame:
    """Completions, starts, abandons and active days per learner in [start, end)."""
    df = events[(events["t"] >= start) & (events["t"] < end)]
    weeks = (end - start) / DAYS_PER_WEEK
    out = pd.DataFrame(index=pd.Index(learner_ids, name="learner_id"))
    for name in ("complete", "abandon", "view"):
        counts = df[df["event"] == name].groupby("learner_id").size()
        out[name] = counts.reindex(out.index, fill_value=0).astype(float)
    days = df.assign(day=df["t"].astype(int)).groupby("learner_id")["day"].nunique()
    out["active_days_per_week"] = days.reindex(out.index, fill_value=0).astype(float) / weeks
    out["completed_per_week"] = out["complete"] / weeks
    return out.rename(columns={"view": "starts", "abandon": "abandons"})


def run_ab(
    world: World,
    state: State,
    history: pd.DataFrame,
    start_day: int,
    policy_a: Ranker,
    policy_b: Ranker,
    cfg: ABConfig | None = None,
) -> pd.DataFrame:
    """Randomise, run the world for `cfg.n_days`, return one row per learner."""
    cfg = cfg or ABConfig()
    rng = np.random.default_rng(cfg.seed)
    ids = world.population.learner_ids
    arm = np.where(rng.permutation(len(ids)) % 2 == 0, "A", "B")
    pa, pb = as_policy(policy_a, cfg.k), as_policy(policy_b, cfg.k)
    policies: list[Policy | None] = [pa if a == "A" else pb for a in arm]
    sim_state = state.copy()
    skill_before = sim_state.skill.copy()
    events = simulate(world, sim_state, start_day, cfg.n_days, rng, policies)
    during = per_learner(events, ids, start_day, start_day + cfg.n_days)
    pre = per_learner(history, ids, start_day - cfg.pre_days, start_day)
    during["arm"] = arm
    during["pre_completed_per_week"] = pre["completed_per_week"].to_numpy()
    during["skill_gain"] = sim_state.skill - skill_before
    return during


def welch_effect(y_b: np.ndarray, y_a: np.ndarray, alpha: float = 0.05) -> Effect:
    """Difference in means B - A with a Welch t-interval."""
    va, vb = y_a.var(ddof=1) / len(y_a), y_b.var(ddof=1) / len(y_b)
    se = float(np.sqrt(va + vb))
    df = (va + vb) ** 2 / (va**2 / (len(y_a) - 1) + vb**2 / (len(y_b) - 1))
    diff = float(y_b.mean() - y_a.mean())
    q = float(stats.t.ppf(1 - alpha / 2, df))
    p = float(2 * stats.t.sf(abs(diff) / se, df)) if se > 0 else 1.0
    return Effect(float(y_a.mean()), float(y_b.mean()), diff, diff - q * se, diff + q * se, p)


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the CUPED-adjusted outcome and theta (pooled over arms)."""
    var_x = float(x.var(ddof=1))
    theta = float(np.cov(x, y, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0
    return y - theta * (x - x.mean()), theta


def ratio_effect(
    num_b: np.ndarray,
    den_b: np.ndarray,
    num_a: np.ndarray,
    den_a: np.ndarray,
    alpha: float = 0.05,
) -> Effect:
    """Difference of two ratio metrics sum(num)/sum(den), with delta-method standard errors.

    Learners are the randomisation unit but starts are not independent (one learner's abandons
    are correlated), so a naive binomial SE on starts would be too small.
    """

    def ratio_and_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
        n, mx, my = len(num), den.mean(), num.mean()
        r = my / mx
        cov = np.cov(num, den, ddof=1)
        var = (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1]) / (n * mx**2)
        return float(r), float(var)

    ra, va = ratio_and_var(num_a, den_a)
    rb, vb = ratio_and_var(num_b, den_b)
    se = float(np.sqrt(va + vb))
    z = float(stats.norm.ppf(1 - alpha / 2))
    diff = rb - ra
    p = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else 1.0
    return Effect(ra, rb, diff, diff - z * se, diff + z * se, p)


@dataclass(frozen=True)
class ABSummary:
    n_a: int
    n_b: int
    primary: Effect
    primary_cuped: Effect
    theta: float
    variance_reduction: float
    abandonment: Effect
    active_days: Effect
    skill_gain: Effect
    srm_p: float  # sample ratio mismatch check (chi-square on arm sizes)


def analyse(df: pd.DataFrame, alpha: float = 0.05) -> ABSummary:
    a, b = df[df["arm"] == "A"], df[df["arm"] == "B"]

    def col(frame: pd.DataFrame, name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=float)

    y = col(df, "completed_per_week")
    x = col(df, "pre_completed_per_week")
    y_adj, theta = cuped(y, x)
    is_b = (df["arm"] == "B").to_numpy()
    raw = welch_effect(col(b, "completed_per_week"), col(a, "completed_per_week"), alpha)
    adj = welch_effect(y_adj[is_b], y_adj[~is_b], alpha)
    # Report the CUPED effect around the raw arm means, so "control" and "treatment" stay readable.
    adj = Effect(
        raw.control, raw.control + adj.diff, adj.diff, adj.ci_low, adj.ci_high, adj.p_value
    )
    var_red = 1.0 - float(y_adj.var(ddof=1) / y.var(ddof=1))
    srm = float(stats.chisquare([len(a), len(b)]).pvalue)
    return ABSummary(
        n_a=len(a),
        n_b=len(b),
        primary=raw,
        primary_cuped=adj,
        theta=theta,
        variance_reduction=var_red,
        abandonment=ratio_effect(
            col(b, "abandons"), col(b, "starts"), col(a, "abandons"), col(a, "starts"), alpha
        ),
        active_days=welch_effect(
            col(b, "active_days_per_week"), col(a, "active_days_per_week"), alpha
        ),
        skill_gain=welch_effect(col(b, "skill_gain"), col(a, "skill_gain"), alpha),
        srm_p=srm,
    )

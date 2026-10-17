"""`recsys report`: run the whole study and write docs/results.md plus two charts.

1. Simulate the pre-recommender era (organic discovery) and keep the true skill at the cutoff.
2. Offline evaluation: fit on events before the cutoff, score what learners completed after it.
3. Train the deployed policy on the full history and run the simulated A/B test after it.

Every number in docs/results.md comes from this file; nothing is typed in by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .abtest import ABConfig, ABSummary, Effect, analyse, run_ab
from .domain import LearnerHistory, histories_from_events
from .evaluate import FALLBACK, HYBRID, OfflineConfig, OfflineResult, build_rankers, run_offline
from .features import build_feature_table, parity_check
from .simulator import State, World, difficulty_gap, initial_state, make_world, simulate

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#b7b7b2"
DATA_LABEL = "Synthetic data"
POLICY_A = "Most popular"


@dataclass(frozen=True)
class ReportConfig:
    n_learners: int = 2000
    n_days: int = 56
    cutoff_day: int = 42
    world_seed: int = 0
    history_seed: int = 1
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    ab: ABConfig = field(default_factory=ABConfig)


@dataclass
class Study:
    config: ReportConfig
    world: World
    events: pd.DataFrame
    offline: OfflineResult
    ab_rows: pd.DataFrame
    ab: ABSummary
    level_corr: float
    level_mae: float
    parity: float


def simulate_history(cfg: ReportConfig) -> tuple[World, pd.DataFrame, np.ndarray, State]:
    """Run the world to the cutoff, snapshot true skill, then run to the end of the history."""
    world = make_world(cfg.n_learners, cfg.n_days, cfg.world_seed)
    state = initial_state(world)
    rng = np.random.default_rng(cfg.history_seed)
    before = simulate(world, state, 0, cfg.cutoff_day, rng)
    skill_at_cutoff = state.skill.copy()
    after = simulate(world, state, cfg.cutoff_day, cfg.n_days - cfg.cutoff_day, rng)
    return world, pd.concat([before, after], ignore_index=True), skill_at_cutoff, state


def run_study(cfg: ReportConfig) -> Study:
    world, events, skill_at_cutoff, state = simulate_history(cfg)
    pos = {lid: u for u, lid in enumerate(world.population.learner_ids)}

    def gap(learner: str, history: LearnerHistory, lesson: int) -> float:
        return difficulty_gap(world, float(skill_at_cutoff[pos[learner]]), history, lesson)

    offline = run_offline(events, world.catalog, float(cfg.cutoff_day), cfg.offline, gap)

    end = float(cfg.n_days)
    histories = histories_from_events(events, world.catalog)
    rankers, _ = build_rankers(world.catalog, histories, cfg.offline)
    ab_rows = run_ab(world, state, events, cfg.n_days, rankers[POLICY_A], rankers[HYBRID], cfg.ab)

    learners, _ = build_feature_table(events, world.catalog, end)
    warm = learners[learners["n_started"] > cfg.offline.cold_start_max]
    true = np.array([state.skill[pos[lid]] for lid in warm.index])
    est = warm["est_level"].to_numpy(dtype=float)
    return Study(
        config=cfg,
        world=world,
        events=events,
        offline=offline,
        ab_rows=ab_rows,
        ab=analyse(ab_rows, cfg.ab.alpha),
        level_corr=float(np.corrcoef(est, true)[0, 1]),
        level_mae=float(np.abs(est - true).mean()),
        parity=parity_check(events, world.catalog, end),
    )


def cell(table: pd.DataFrame, model: str, segment: str, col: str) -> float:
    """One number from the (model, segment) x metric table, typed for mypy."""
    mask = (table.index.get_level_values("model") == model) & (
        table.index.get_level_values("segment") == segment
    )
    return float(table[col].to_numpy(dtype=float)[mask][0])


def _effect_row(name: str, e: Effect, digits: int = 3, pct: bool = False) -> str:
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{digits}f}")
    d = (lambda v: f"{100 * v:+.1f} pts") if pct else (lambda v: f"{v:+.{digits}f}")
    return (
        f"| {name} | {f(e.control)} | {f(e.treatment)} | {d(e.diff)} | "
        f"{d(e.ci_low)} to {d(e.ci_high)} | {e.p_value:.1g} |"
    )


def results_markdown(study: Study) -> str:
    cfg, off, ab = study.config, study.offline, study.ab
    ev = study.events
    k = cfg.offline.k
    t = off.table
    n_all = int(cell(t, HYBRID, "all", "n_learners"))
    n_cold = int(cell(t, HYBRID, "cold start", "n_learners"))
    starts = int((ev["event"] == "view").sum())
    lines = [
        "# Results",
        "",
        f"Data: **{DATA_LABEL}** from `recsys.simulator` (seeded). All numbers below come from "
        "`recsys report`; rerun it to reproduce them.",
        "",
        "## 1. The simulated history",
        "",
        f"{cfg.n_learners:,} learners, {len(study.world.catalog)} lessons in "
        f"{len(study.world.catalog.topics)} topics, {cfg.n_days} days with no recommender "
        f"(organic discovery). {len(ev):,} events: {starts:,} lesson starts, "
        f"{int((ev['event'] == 'complete').sum()):,} completions, "
        f"{int((ev['event'] == 'abandon').sum()):,} abandons.",
        "",
        "## 2. Offline evaluation (time-based split)",
        "",
        f"Models fit on {off.n_train_events:,} events before day {off.cutoff:.0f}. Scored on "
        f"{n_all:,} learners who completed at least one new lesson after it; {n_cold} of them are "
        f"cold-start (at most {cfg.offline.cold_start_max} lessons started before the cutoff). "
        f"Relevant = lessons completed after the cutoff and not started before it. k = {k}.",
        "",
        "`too hard` uses the simulator's ground truth: the share of recommended lessons more than "
        "one level above the learner's true skill at the cutoff (an unmet prerequisite adds one "
        "level). A real log cannot measure this.",
        "",
    ]
    cols = [f"recall@{k}", f"ndcg@{k}", f"map@{k}", "coverage", "novelty", "too_hard"]
    for segment in ("all", "warm", "cold start"):
        seg = t.xs(segment, level="segment")
        n = int(seg["n_learners"].iloc[0])
        lines += [
            f"**{segment.capitalize()} learners** ({n:,})",
            "",
            "| Model | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
            "| --- |" + " --- |" * len(cols),
        ]
        for name, row in seg.iterrows():
            cells = [f"{row[c]:.3f}" if c != "novelty" else f"{row[c]:.2f}" for c in cols]
            label = f"**{name}**" if name == HYBRID else str(name)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    loss = off.als.loss_history
    lines += [
        f"ALS loss (Hu et al. objective) by iteration: {loss[0]:,.0f} -> {loss[1]:,.0f} -> ... "
        f"-> {loss[-1]:,.0f} (never increases). The two-tower model was not run here "
        "(torch is not installed in this environment).",
        "",
        f"Level estimate check (learners with more than {cfg.offline.cold_start_max} lessons at "
        f"day {cfg.n_days}): correlation with the hidden true skill {study.level_corr:.2f}, mean "
        f"absolute error {study.level_mae:.2f} levels.",
        "",
        f"Feature parity (batch table vs serving path, all learners): max difference "
        f"{study.parity:g}.",
        "",
        "## 3. Simulated online A/B test",
        "",
        f"From day {cfg.n_days}, every learner is randomised 50/50: A = most popular, "
        f"B = {HYBRID}. Both trained on the full history and frozen. {cfg.ab.n_days} days, "
        f"slates of {cfg.ab.k} lessons. {ab.n_a:,} learners in A, {ab.n_b:,} in B "
        f"(sample-ratio check p = {ab.srm_p:.2f}). 95% intervals.",
        "",
        "| Metric | A (popular) | B (ALS + re-rank) | B - A | 95% CI | p |",
        "| --- | --- | --- | --- | --- | --- |",
        _effect_row("Lessons completed per learner per week (primary)", ab.primary),
        _effect_row("Same, CUPED-adjusted", ab.primary_cuped),
        _effect_row("Abandonment rate (guardrail)", ab.abandonment, pct=True),
        _effect_row("Active days per learner per week", ab.active_days),
        _effect_row("True skill gain (simulator only)", ab.skill_gain),
        "",
        f"Relative lift on the primary metric: {100 * ab.primary.rel:+.1f}% "
        f"(CUPED CI {100 * ab.primary_cuped.ci_low / ab.primary.control:+.1f}% to "
        f"{100 * ab.primary_cuped.ci_high / ab.primary.control:+.1f}% of the control mean). "
        f"CUPED covariate: lessons completed per week in the {cfg.ab.pre_days} days before the "
        f"test; theta = {ab.theta:.3f}, variance reduction {100 * ab.variance_reduction:.1f}%, "
        f"CI width {_width(ab.primary):.3f} -> {_width(ab.primary_cuped):.3f}.",
        "",
        "## 4. Offline vs online",
        "",
    ]
    lines += _mismatch_lines(study)
    return "\n".join(lines) + "\n"


def _width(e: Effect) -> float:
    return e.ci_high - e.ci_low


def _mismatch_lines(study: Study) -> list[str]:
    t, k, ab = study.offline.table, study.config.offline.k, study.ab
    r = f"recall@{k}"

    def get(model: str, col: str, seg: str = "all") -> float:
        return cell(t, model, seg, col)

    seq = "Next in sequence"
    m = f"map@{k}"
    best_map = str(t.xs("all", level="segment")[m].idxmax())
    rr_change = get("ALS + re-rank", r) - get("ALS (implicit)", r)
    exposure = (
        f"- **{seq}** has the highest offline MAP@{k} ({get(seq, m):.3f}) even though its "
        f"recall@{k} is only {get(seq, r):.3f}, and {get(seq, 'too_hard'):.0%} of its picks are "
        "too hard."
        if best_map == seq
        else f"- **{seq}** reaches MAP@{k} {get(seq, m):.3f} with recall@{k} {get(seq, r):.3f}."
    )
    return [
        f"- Offline, B's recall@{k} is {get(HYBRID, r) / get(POLICY_A, r):.1f}x that of "
        f"popularity ({get(HYBRID, r):.3f} vs {get(POLICY_A, r):.3f}). Online, B lifts "
        f"completions per learner per week by {100 * ab.primary.rel:.0f}%. A large offline "
        "ratio does not translate one-to-one into an online lift: learners also browse on their "
        "own, and a recommendation they would have found anyway adds nothing.",
        f"- Re-ranking {'raises' if rr_change >= 0 else 'lowers'} offline recall by "
        f"{abs(rr_change):.3f} (ALS {get('ALS (implicit)', r):.3f} -> ALS + re-rank "
        f"{get('ALS + re-rank', r):.3f}) and cuts too-hard recommendations "
        f"from {get('ALS (implicit)', 'too_hard'):.1%} to {get('ALS + re-rank', 'too_hard'):.1%}. "
        'Recall only asks "did they finish it anyway"; the simulator\'s abandonment and return '
        "rules also punish every too-hard lesson that was shown and dropped. Logged data cannot "
        "show what happens after a recommendation the logging policy never made.",
        exposure
        + " The logging policy often opened the next lesson in the same track, so the test "
        "window rewards copying the logging policy (exposure bias of offline evaluation).",
        f"- On cold-start learners the content/popularity fallback "
        f"({get(FALLBACK, r, 'cold start'):.3f}) beats plain ALS "
        f"({get('ALS (implicit)', r, 'cold start'):.3f}); with 0-3 lessons a fold-in user "
        "vector is mostly noise, so the deployed policy switches to the fallback.",
        "- All effects were planted in the simulator on purpose (see docs/how-it-works.md). The "
        "A/B result shows the code detects and measures them; it does not predict the size of "
        "the effect on real learners.",
    ]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.9, subtitle, fontsize=9.5, color=INK_2, va="top")


def plot_offline(study: Study, path: Path) -> None:
    import matplotlib.pyplot as plt

    k = study.config.offline.k
    col = f"recall@{k}"
    t = study.offline.table
    warm = t.xs("warm", level="segment")[col].sort_values()
    cold = t.xs("cold start", level="segment")[col].reindex(warm.index)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150, sharey=True)
    fig.set_facecolor(SURFACE)
    for ax, series, label in zip(
        axes, (warm, cold), ("Warm learners", "Cold-start learners (0-3 lessons)"), strict=True
    ):
        colors = [BLUE if n == HYBRID else GREY for n in series.index]
        ax.barh(series.index, series.to_numpy(), color=colors, height=0.62)
        for y, v in enumerate(series.to_numpy()):
            ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.6)
        ax.set_title(label, fontsize=9.5, color=INK_2, loc="left")
        ax.set_xlabel(f"Recall@{k} after the cutoff (higher is better)", fontsize=9, color=INK_2)
    n_warm = int(cell(t, HYBRID, "warm", "n_learners"))
    n_cold = int(cell(t, HYBRID, "cold start", "n_learners"))
    _header(
        fig,
        f"Deployed policy: recall@{k} {warm[HYBRID]:.2f} on warm learners vs "
        f"{warm[POLICY_A]:.2f} for popularity; fallback lifts cold start to {cold[HYBRID]:.2f}",
        f"Time-based split at day {study.offline.cutoff:.0f}: {n_warm:,} warm and {n_cold} "
        f"cold-start learners. Blue = deployed policy. {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_ab(study: Study, path: Path) -> None:
    import matplotlib.pyplot as plt

    ab = study.ab
    rows = [
        ("Lessons completed / learner / week\n(primary)", ab.primary, True),
        ("Same, CUPED-adjusted", ab.primary_cuped, True),
        ("Active days / learner / week", ab.active_days, False),
        ("Abandonment rate (guardrail)", ab.abandonment, False),
    ][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=150)
    fig.set_facecolor(SURFACE)
    for y, (_, e, primary) in enumerate(rows):
        color = BLUE if primary else GREY
        lo, hi, mid = (100 * v / e.control for v in (e.ci_low, e.ci_high, e.diff))
        ax.plot([lo, hi], [y, y], color=color, linewidth=3, solid_capstyle="round")
        ax.plot(mid, y, "o", markersize=9, color=color)
        ax.text(hi + 1.2, y, f"{mid:+.1f}%", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    _style(ax)
    ax.grid(axis="y", visible=False)
    span = max(abs(100 * r[1].ci_low / r[1].control) for r in rows)
    span = max(span, max(abs(100 * r[1].ci_high / r[1].control) for r in rows)) + 8
    ax.set_xlim(-span, span)
    ax.set_xlabel("B vs A, relative change with 95% CI (% of the A mean)", fontsize=9, color=INK_2)
    cfg = study.config.ab
    _header(
        fig,
        f"ALS + re-rank: {100 * ab.primary.rel:+.0f}% lessons completed per week, "
        f"abandonment {100 * ab.abandonment.control:.0f}% -> {100 * ab.abandonment.treatment:.0f}%",
        f"Simulated A/B test, {cfg.n_days} days, {ab.n_a + ab.n_b:,} learners randomised 50/50. "
        f"A = most popular, B = ALS + re-rank + fallback. {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(out_dir: Path, cfg: ReportConfig | None = None) -> Study:
    study = run_study(cfg or ReportConfig())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(results_markdown(study))
    plot_offline(study, out_dir / "img" / "offline_recall.png")
    plot_ab(study, out_dir / "img" / "ab_effect.png")
    return study

# How it works

This page follows the data from the simulator to a recommendation, and explains the maths on
the way. File references point to `src/recsys/`.

## 1. The world: a learner simulator (`simulator.py`)

The simulator is the ground truth. Models never see its hidden state, only the event log.

**Lessons.** 300 lessons in 10 English-learning topics (grammar, vocabulary, listening,
speaking, reading, writing, pronunciation, IELTS, business, travel). Each topic is a track of
30 lessons: six at each level 1 to 5, in order. Every lesson above level 1 has one or two
prerequisites one level lower in the same topic. Each lesson also has a hidden `appeal`
(how attractive the title and thumbnail are), drawn from `N(0, 0.6)`.

**Learners.** 2,000 learners. Each has hidden values:

| Hidden value | Distribution | What it does |
| --- | --- | --- |
| `skill` | 45% beginners `N(1.6, 0.4)`, 35% intermediate `N(2.8, 0.4)`, 20% advanced `N(4.0, 0.4)` | same scale as lesson levels |
| `interest` | `Dirichlet(0.4)` over the 10 topics | sparse: most learners like 2-3 topics |
| `activity` | `Beta(2, 3)` | base chance of studying on a given day |
| `pace` | `Gamma(2, 0.6)` | extra lessons per study session |
| `arrival` | 70% day 0, 30% a random later day | late arrivals are the cold-start learners |
| `engagement` | starts at 0.9, moves | scales the daily chance of coming back |

**Finishing a lesson.** When learner `u` starts lesson `i`:

    gap = level_i - skill_u + 1.0 * [a prerequisite is neither completed nor clearly mastered]
    logit P(complete) = 1.5 + 0.5 * log(interest match) + 0.5 * appeal_i - 2.5 * max(0, gap - 0.3)

"Interest match" is the learner's interest in the topic relative to uniform (1 = average),
clipped to [0.1, 5]. "Clearly mastered" means the learner's skill is a full level above the
prerequisite. So a lesson well above your level, or one you lack the basics for, is usually
abandoned.

**Coming back.** Each day a learner studies with probability `activity * engagement`.
Engagement changes after every lesson:

- finished and `-0.8 <= gap <= 1.0` (the "flow zone"): `+0.05`, and skill `+0.02` (learning);
- finished but `gap < -0.8` (too easy, boring): `-0.03`;
- abandoned: `-0.06`;
- each night engagement moves 5% of the way back to 0.9.

These are the effects planted on purpose: interest drives choice and completion, the
difficulty gap drives abandonment, and both drive whether learners return.

**How learners find lessons without a recommender** (the logging policy, `_organic_choice`):

- 35%: open the next lesson not yet started in the topic of the last lesson;
- 25%: pick from a global "popular" shelf (the 30 most appealing lessons, level-blind);
- otherwise browse: softmax over `1.2 log(interest) + 0.6 appeal - 0.7 |level - skill - 0.3|`.

**With a recommender** (`_slate_choice`): with probability 0.7 the learner picks from the slate
of k lessons, by interest and appeal with a mild position bias. They cannot see difficulty
before starting, so a bad slate leads to abandonment. Otherwise they browse as above.

**The log.** Every attempt writes a `view` and then `complete` or `abandon` 8-25 minutes later:
`learner_id, lesson_id, t (days), event`. 56 days of history give 96,764 events.

## 2. Shared types (`domain.py`)

- `Catalog`: public lesson data only (topic, level, track position, prerequisites, ids). No
  appeal: that is hidden.
- `LearnerHistory`: one learner's events in time order with running sets (`started`,
  `completed`, `abandoned`) and implicit-feedback weights `r_ui`: a view counts 1, a completion
  adds 2, an abandon adds 0. So a finished lesson has `r = 3`, an abandoned one `r = 1`.
- `histories_from_events` (batch: whole table, groupby) and `EventStore` (serving: one learner
  at a time). Both filter strictly before an `as_of` time.

## 3. The time-based split (`data.py`)

`time_split(events, T)` puts every event with `t < T` in train and every event with `t >= T` in
test. T = day 42 of 56. A learner's test targets are the lessons they **completed** after T and
had **not started** before T (`heldout_targets`). An attempt that starts before T and finishes
after it stays out of the targets, so no label crosses the line.

Cold-start learners: at most 3 lessons started before T (this includes learners who arrive
after T and have no history at all).

The leakage test rewrites every event after T (other lessons, all marked complete) and checks
that the fitted ALS factors and the feature tables do not change at all.

## 4. Features (`features.py`)

`learner_features(history, catalog, as_of)` is the only definition of learner features:

| Feature | Meaning |
| --- | --- |
| `n_started`, `n_completed`, `n_abandoned` | counts of distinct lessons |
| `completion_rate` | `(completed + 1) / (started + 2)` (Laplace smoothing) |
| `est_level` | estimated level, see below |
| `days_since_last` | time since the last event (-1 with no history) |
| `active_days` | distinct days with an event |
| `top_topic_share`, `n_topics` | how focused the learner is |

**Level estimate.** A shrunk average of "votes": a completed lesson at level L votes `L + 0.5`,
an abandoned one votes `L - 1`, and two pseudo-votes sit at 1.5 (a new learner is assumed to be
a beginner). On the synthetic data it correlates 0.81 with the hidden skill for learners with
more than three lessons (mean absolute error 0.63 levels).

`item_features` gives lesson popularity (learners who started it), smoothed completion rate and
novelty bits `-log2(share of learners who started it)`.

**Parity.** `build_feature_table` (batch, used for training) and `serving_learner_features`
(one learner from the `EventStore`, used by the API) both call `learner_features`.
`parity_check` compares them for every learner; `recsys train` refuses to save if any value
differs. On the synthetic data the largest difference is exactly 0.

## 5. Models

All scorers share one interface: `fit(histories, n_items)` and `score(history) -> array`.

**Most popular** (`baselines.py`): lessons started by the most learners.

**Next in sequence**: the earliest not-yet-started lessons in the learner's most recent topic,
then the topic before, then popularity. This is "continue where you left off".

**Item kNN**: cosine similarity between lesson columns of the binary learner x lesson matrix,
keeping the 50 strongest neighbours per lesson. `score(i) = sum_{j started} sim(i, j)`.

**Implicit ALS** (`als.py`, Hu, Koren and Volinsky 2008). With `r_ui` from above:

    p_ui = 1 if r_ui > 0 else 0          (preference)
    c_ui = 1 + alpha * r_ui              (confidence)
    L(X, Y) = sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (sum_u |x_u|^2 + sum_i |y_i|^2)

All cells count, zeros too: an unopened lesson is weak evidence of no interest. With Y fixed,
each `x_u` is a ridge regression with a closed form:

    x_u = (Y^T C_u Y + reg I)^-1 Y^T C_u p_u
    Y^T C_u Y = Y^T Y + Y_u^T (C_u - I) Y_u

`Y^T Y` is computed once per half-step; the correction only involves the lessons u touched, so
each solve costs `O(f^2 n_u + f^3)` instead of `O(f^2 n_items)`. The lesson step is the same
with X fixed. Each half-step solves its sub-problem exactly, so the loss never goes up. A test
checks this; on the synthetic data it falls from 185,884 to 86,295 in 12 iterations.

**Fold-in.** At serving time a learner's vector is solved from their current history against
the frozen lesson factors, with the same `_solve_one`. Training ends with a user step, so for a
training learner fold-in returns exactly the stored factor (tested). New events are used at
once, without retraining.

Settings: 16 factors, `reg = 30`, `alpha = 2`, 12 iterations, chosen once on a separate tuning
world (seed 7), not on the reported data.

**Two-tower** (`two_tower.py`, optional `[deep]` extra): lesson tower from id, topic and level
embeddings; learner tower = mean of started-lesson embeddings through a small MLP; full softmax
over 300 lessons with leave-one-out targets. Not run here: torch is not installed in this
environment, and the test skips without it.

## 6. Learning-aware re-ranking (`rerank.py`)

Take the 60 best unseen lessons from the model. Scale their scores to [0, 1] (`rel`), then:

    final_i = rel_i - 1.0 * max(0, level_i - est_level - 0.5)     (too hard)
                    - 0.3 * max(0, est_level - level_i - 1.5)     (too easy)

Lessons with an unmet prerequisite are removed. A prerequisite counts as met if it was
completed, or if `est_level` is a full level above it (placement). Blocked lessons are only
used to fill a list that would otherwise be shorter than k, and they go last.

Optional MMR (`mmr_lambda = 0.7` in the evaluation): pick greedily the lesson that maximises
`lambda * final_i - (1 - lambda) * [same topic as a lesson already picked]`.

## 7. Policies (`policy.py`)

- `PlainRanker`: a scorer as is.
- `LearningRecommender`: the deployed policy. Estimate the level; if the learner has started
  at most 3 lessons use the content + popularity fallback (popularity, plus a bonus for topics
  they touched); otherwise ALS fold-in. Then re-rank.
- `save`/`load`: JSON + NPZ, no pickle.

## 8. Offline evaluation (`evaluate.py`, `metrics.py`)

Every model is fit on events before day 42. For each of the 1,708 learners with at least one
new completion after day 42, the model recommends 10 lessons from the history before day 42.

| Metric | Definition |
| --- | --- |
| recall@10 | hits / number of relevant lessons |
| NDCG@10 | binary-gain DCG / ideal DCG |
| MAP@10 | mean over learners of AP@10 = sum of precision@i at hits / min(relevant, 10) |
| coverage | share of the catalogue recommended to at least one learner |
| novelty | mean `-log2(share of learners who started the lesson)` over recommended lessons |
| too hard | (simulator only) share of recommended lessons with true gap > 1 at the cutoff |

Results are reported for all, warm and cold-start learners.

## 9. Simulated A/B test (`abtest.py`)

After the 56-day history, all 2,000 learners are randomised 50/50. Arm A sees most-popular
slates, arm B the deployed policy, both trained on the full history and frozen. The world runs
28 more days. Per learner: lessons completed per week (primary), abandoned starts / all starts
(guardrail), active days per week, true skill gain.

- Primary effect: difference in means with a Welch t-interval.
- **CUPED:** `Y_adj = Y - theta (X - mean X)`, `theta = cov(X, Y) / var(X)` over both arms,
  `X` = lessons completed per week in the 28 days before the test. Because X was measured
  before randomisation, it is independent of the arm, so the adjusted difference has the same
  expected value; its variance is smaller by a factor `1 - corr(X, Y)^2`.
- **Guardrail:** abandonment is a ratio of sums over learners. Its standard error uses the
  delta method with the learner as the unit:
  `Var(R) ~ (Var(num) - 2R Cov(num, den) + R^2 Var(den)) / (n mean(den)^2)`.
- A chi-square sample-ratio-mismatch check on the arm sizes.

## 10. Serving (`artifacts.py`, `api.py`)

`recsys train` fits the deployed policy on the full history, runs the parity check and writes
`artifacts/` (catalogue, lesson factors, popularity, config, event store, feature snapshots,
metadata). `recsys serve` loads them. `GET /recommend?learner_id=&k=` reads the learner's
history from the store, computes features with `learner_features`, and returns lessons with
topic, level and whether prerequisites are met. Unknown ids get the cold-start fallback.

## 11. The report (`report.py`)

`recsys report` runs everything above with fixed seeds and writes `docs/results.md` and the two
charts. No number is typed by hand. It takes about 20 seconds.

# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. A simulator as the ground truth

- **Decision:** a seeded world with hidden skill, interests, engagement and lesson appeal; real
  data only through loaders.
- **Why:** CI has no internet, and no public dataset can answer "what would have happened if we
  had shown something else?". A simulator can: it runs an online A/B test, and it knows each
  learner's true skill, so "too hard" recommendations can be counted.
- **Cost:** the world was built with the effects the re-ranker exploits (difficulty gap,
  prerequisites, engagement). The A/B lift shows the code finds and measures them, not how big
  they are for real learners. The README says so next to the results.

## 2. Completions as the relevance label

- **Decision:** offline, a hit is a lesson completed after the cutoff and not started before.
- **Why:** the business question is "keep learning and keep coming back". A click on a lesson
  that is abandoned two minutes later is not success.
- **Cost:** fewer positives per learner than with views, so metrics are noisier on small data.

## 3. One global time cutoff

- **Decision:** train strictly before day 42, test from day 42; a test that rewrites the future
  and checks that models and features do not change.
- **Why:** a deployed model only knows the past. A random split lets a model learn a learner's
  future taste while predicting their past.
- **Cost:** learners who arrive after the cutoff are only in test. That is realistic (they are
  the cold-start case) and they are reported separately.

## 4. ALS written in NumPy, direct solve per row

- **Decision:** implement Hu, Koren and Volinsky (2008) with the `Y^T Y` trick and
  `np.linalg.solve` for each learner and each lesson.
- **Why:** the maths is short and testable (monotone loss, planted blocks, fold-in = trained
  factor). With 16 factors and 300 lessons a direct solve is fast; conjugate gradient pays off
  at hundreds of factors or millions of rows.
- **Cost:** a Python loop over rows: about a second per fit here, too slow for millions of
  learners. The `implicit` library or Spark ALS would be the production choice.

## 5. Implicit weights: view 1, complete +2, abandon 0

- **Decision:** `r_ui` = 1 for a started lesson, 3 if finished. Abandon is not negative.
- **Why:** choosing a lesson shows interest even if it was too hard. Treating abandonment as a
  negative would teach the model "this learner dislikes grammar" when the truth is "this
  grammar lesson was too hard". Difficulty is handled by the re-ranker instead.
- **Cost:** the model alone keeps recommending appealing lessons that are too hard (23.2% of
  plain ALS picks are too hard by the simulator's ground truth). That is the re-ranker's job.

## 6. Hyperparameters chosen on a separate world

- **Decision:** a small grid (factors, reg, alpha) on a world with seed 7; the reported world
  uses seed 0. Picked: 16 factors, reg 30, alpha 2.
- **Why:** tuning on the reported test window would inflate the reported numbers.
- **Cost:** the reg value is not scaled with the data size, so a much smaller log (as in the
  tests) gets over-regularised factors. Retune when the data changes.

## 7. Re-ranking as explicit rules, not a learned model

- **Decision:** level penalties around an estimated level, a prerequisite filter, optional MMR.
- **Why:** a teacher can read and argue with each rule, and the rules use catalogue data the
  collaborative model does not see. They also work for a learner with no history.
- **Cost:** the weights (1.0, 0.3, tolerance 0.5) are hand-set. A learned ranker on completion
  outcomes is listed as a next step.

## 8. Placement counts as a met prerequisite

- **Decision:** a prerequisite is met if completed, or if the estimated level is a full level
  above it. The simulator uses the same rule with the true skill.
- **Why:** otherwise advanced newcomers would be forced through beginner lessons and get bored.
- **Cost:** a wrong level estimate can unlock a lesson too early. The level estimate is
  shrunk toward beginner level to make this rare.

## 9. Cold-start fallback at 3 lessons

- **Decision:** learners with 0-3 started lessons get content + popularity (topic boost), then
  the same re-ranking.
- **Why:** a fold-in vector from one to three lessons is mostly noise; on cold-start learners
  the fallback gets recall@10 0.361 vs 0.232 for plain ALS.
- **Cost:** the threshold is fixed; a smooth blend by history length could be better.

## 10. One feature function, two paths, a parity gate

- **Decision:** `learner_features` is called by the batch table and by the serving path;
  `recsys train` fails if they differ.
- **Why:** training/serving skew is one of the most common silent bugs in recommenders.
- **Cost:** per-learner Python, not a vectorised SQL job. Fine for thousands of learners.

## 11. Simulated A/B test with CUPED and a delta-method guardrail

- **Decision:** randomise learners 50/50, primary metric completions per learner per week
  (zero for learners who never return), CUPED with pre-period completions, abandonment rate
  with delta-method standard errors, a sample-ratio check.
- **Why:** these are the tools a real experiment would use, and the simulator checks they run
  end to end. Averaging over randomised learners (not over active ones) avoids survivorship
  bias.
- **Cost:** CUPED only removed 10.9% of the variance here; pre-period completions are a weak
  predictor because study habits change with engagement. A real app with longer histories
  would likely do better.

## 12. Frozen policies in the A/B test

- **Decision:** both arms are trained once on the history and not retrained during the test.
  B still uses each learner's newest events through fold-in.
- **Why:** simple and fair: neither arm gets fresh lesson popularity or factors.
- **Cost:** a real system would retrain daily, which could help either arm.

## 13. JSON + NPZ artefacts and an event store

- **Decision:** no pickle; the API reads histories from a saved event log.
- **Why:** readable, safe to load, and the API can answer by learner id like a real service.
- **Cost:** the store is in memory and static; production would use a key-value store fed by
  the event stream.

## 14. Two-tower only as an optional extra

- **Decision:** `two_tower.py` needs `pip install -e ".[deep]"`; its test skips without torch
  and the file is left out of coverage.
- **Why:** torch would make CI slow, and the catalogue is small enough for ALS.
- **Cost:** it was not run in the environment where this repo was built, so it has no
  reported numbers. Run `pytest tests/test_two_tower.py` after installing the extra.

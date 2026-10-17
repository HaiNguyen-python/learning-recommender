# Results

Data: **Synthetic data** from `recsys.simulator` (seeded). All numbers below come from `recsys report`; rerun it to reproduce them.

## 1. The simulated history

2,000 learners, 300 lessons in 10 topics, 56 days with no recommender (organic discovery). 96,764 events: 48,382 lesson starts, 27,680 completions, 20,702 abandons.

## 2. Offline evaluation (time-based split)

Models fit on 71,780 events before day 42. Scored on 1,708 learners who completed at least one new lesson after it; 220 of them are cold-start (at most 3 lessons started before the cutoff). Relevant = lessons completed after the cutoff and not started before it. k = 10.

`too hard` uses the simulator's ground truth: the share of recommended lessons more than one level above the learner's true skill at the cutoff (an unmet prerequisite adds one level). A real log cannot measure this.

**All learners** (1,708)

| Model | recall@10 | ndcg@10 | map@10 | coverage | novelty | too hard |
| --- | --- | --- | --- | --- | --- | --- |
| Most popular | 0.202 | 0.138 | 0.071 | 0.107 | 2.01 | 0.004 |
| Next in sequence | 0.321 | 0.333 | 0.258 | 0.943 | 3.80 | 0.230 |
| Item kNN (cosine) | 0.342 | 0.274 | 0.174 | 0.743 | 2.63 | 0.147 |
| ALS (implicit) | 0.402 | 0.328 | 0.219 | 0.753 | 3.21 | 0.232 |
| ALS + re-rank | 0.435 | 0.358 | 0.245 | 0.577 | 3.22 | 0.016 |
| ALS + re-rank + MMR | 0.417 | 0.348 | 0.231 | 0.553 | 2.88 | 0.013 |
| Content/popularity + re-rank | 0.432 | 0.347 | 0.238 | 0.633 | 3.43 | 0.034 |
| **ALS + re-rank + fallback** | 0.447 | 0.365 | 0.248 | 0.577 | 3.14 | 0.016 |

**Warm learners** (1,488)

| Model | recall@10 | ndcg@10 | map@10 | coverage | novelty | too hard |
| --- | --- | --- | --- | --- | --- | --- |
| Most popular | 0.186 | 0.130 | 0.067 | 0.107 | 2.03 | 0.005 |
| Next in sequence | 0.320 | 0.343 | 0.268 | 0.943 | 3.98 | 0.254 |
| Item kNN (cosine) | 0.357 | 0.282 | 0.176 | 0.710 | 2.54 | 0.140 |
| ALS (implicit) | 0.427 | 0.346 | 0.227 | 0.753 | 3.20 | 0.239 |
| ALS + re-rank | 0.460 | 0.377 | 0.256 | 0.577 | 3.27 | 0.016 |
| ALS + re-rank + MMR | 0.440 | 0.364 | 0.239 | 0.553 | 2.89 | 0.014 |
| Content/popularity + re-rank | 0.442 | 0.357 | 0.245 | 0.633 | 3.60 | 0.037 |
| **ALS + re-rank + fallback** | 0.460 | 0.377 | 0.256 | 0.577 | 3.27 | 0.016 |

**Cold start learners** (220)

| Model | recall@10 | ndcg@10 | map@10 | coverage | novelty | too hard |
| --- | --- | --- | --- | --- | --- | --- |
| Most popular | 0.305 | 0.187 | 0.096 | 0.040 | 1.89 | 0.000 |
| Next in sequence | 0.330 | 0.271 | 0.192 | 0.397 | 2.55 | 0.064 |
| Item kNN (cosine) | 0.241 | 0.219 | 0.167 | 0.533 | 3.23 | 0.188 |
| ALS (implicit) | 0.232 | 0.213 | 0.165 | 0.493 | 3.26 | 0.185 |
| ALS + re-rank | 0.263 | 0.231 | 0.175 | 0.387 | 2.88 | 0.012 |
| ALS + re-rank + MMR | 0.263 | 0.236 | 0.179 | 0.323 | 2.75 | 0.009 |
| Content/popularity + re-rank | 0.361 | 0.280 | 0.193 | 0.407 | 2.31 | 0.015 |
| **ALS + re-rank + fallback** | 0.361 | 0.280 | 0.193 | 0.407 | 2.31 | 0.015 |

ALS loss (Hu et al. objective) by iteration: 185,884 -> 94,770 -> ... -> 86,295 (never increases). The two-tower model was not run here (torch is not installed in this environment).

Level estimate check (learners with more than 3 lessons at day 56): correlation with the hidden true skill 0.81, mean absolute error 0.63 levels.

Feature parity (batch table vs serving path, all learners): max difference 0.

## 3. Simulated online A/B test

From day 56, every learner is randomised 50/50: A = most popular, B = ALS + re-rank + fallback. Both trained on the full history and frozen. 28 days, slates of 5 lessons. 1,000 learners in A, 1,000 in B (sample-ratio check p = 1.00). 95% intervals.

| Metric | A (popular) | B (ALS + re-rank) | B - A | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Lessons completed per learner per week (primary) | 2.015 | 2.757 | +0.742 | +0.615 to +0.869 | 2e-29 |
| Same, CUPED-adjusted | 2.015 | 2.760 | +0.745 | +0.626 to +0.864 | 3e-33 |
| Abandonment rate (guardrail) | 36.6% | 25.0% | -11.5 pts | -12.8 pts to -10.2 pts | 4e-67 |
| Active days per learner per week | 1.765 | 1.948 | +0.183 | +0.097 to +0.270 | 3e-05 |
| True skill gain (simulator only) | 0.034 | 0.081 | +0.047 | +0.040 to +0.055 | 7e-36 |

Relative lift on the primary metric: +36.8% (CUPED CI +31.1% to +42.9% of the control mean). CUPED covariate: lessons completed per week in the 28 days before the test; theta = 0.396, variance reduction 10.9%, CI width 0.254 -> 0.238.

## 4. Offline vs online

- Offline, B's recall@10 is 2.2x that of popularity (0.447 vs 0.202). Online, B lifts completions per learner per week by 37%. A large offline ratio does not translate one-to-one into an online lift: learners also browse on their own, and a recommendation they would have found anyway adds nothing.
- Re-ranking raises offline recall by 0.033 (ALS 0.402 -> ALS + re-rank 0.435) and cuts too-hard recommendations from 23.2% to 1.6%. Recall only asks "did they finish it anyway"; the simulator's abandonment and return rules also punish every too-hard lesson that was shown and dropped. Logged data cannot show what happens after a recommendation the logging policy never made.
- **Next in sequence** has the highest offline MAP@10 (0.258) even though its recall@10 is only 0.321, and 23% of its picks are too hard. The logging policy often opened the next lesson in the same track, so the test window rewards copying the logging policy (exposure bias of offline evaluation).
- On cold-start learners the content/popularity fallback (0.361) beats plain ALS (0.232); with 0-3 lessons a fold-in user vector is mostly noise, so the deployed policy switches to the fallback.
- All effects were planted in the simulator on purpose (see docs/how-it-works.md). The A/B result shows the code detects and measures them; it does not predict the size of the effect on real learners.

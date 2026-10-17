# Learning guide (for me, Hai)

This file is my study guide for this repo. Goal: explain every part in an interview, change it
with confidence, and answer "why?" three levels deep. Work through it before pinning the repo.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough and the maths) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 0. Why this matters to me as a teacher

In my classes and at HaiEduTech, students rarely left because we had too little material. They
left after a run of lessons that did not fit: a B2 reading text for an A2 learner, a grammar
point that needed something they had not learned yet, or a fifth lesson in a row on a topic
they did not care about. A good teacher picks the next lesson with three questions in mind:
does it interest this student, can they do it, and does it build on what they know?

This repo turns those three questions into code:

- **Interest** is what collaborative filtering learns: learners who opened the same lessons
  tend to like the same topics. ALS learns it from starts and completions.
- **Can they do it** is the level estimate and the "too hard" penalty. In class this is the
  zone of proximal development: a little above the current level, not far above.
- **Builds on what they know** is the prerequisite filter. The placement rule ("you are a full
  level above it") is what I did with placement tests for new students.
- **Keep coming back** is the engagement rule in the simulator: a lesson that was too hard
  makes a student less likely to open the app tomorrow. That is why the A/B test measures
  completions per learner per week and abandonment, not clicks.

What I must be careful about when I tell this story: the simulator was built with these effects,
so the A/B test can only show that the code detects and measures them. It does not prove the
size of the gain with real learners. Only an experiment with real students can.

## 1. Run it and watch each step (1 hour)

```bash
make install
make check                         # lint, types, tests with coverage
recsys generate --learners 200     # look at data/events.csv and data/lessons.csv in a notebook
recsys evaluate --learners 500     # the offline table on a smaller world
recsys train                       # artifacts/: open meta.json and config.json
recsys report                      # docs/results.md and the two charts
recsys serve                       # http://127.0.0.1:8000/docs, try /recommend
```

Then change one thing and predict the effect first: in `simulator.py` set `hard_slope` to 0
(difficulty no longer causes abandonment) and rerun `recsys report`. What should happen to the
A/B lift and to the "too hard" column? Then try `follow_prob = 0.3`.

## 2. Read the code in this order

| File | What to understand | Question to answer yourself |
| --- | --- | --- |
| `simulator.py` | hidden skill, interest, engagement; completion logit; logging policy | Why can a simulator measure "too hard" when a real log cannot? |
| `domain.py` | `LearnerHistory`, implicit weights, `EventStore` | Why is abandon weight 0 and not negative? |
| `data.py` | time split, held-out targets, loaders | Why exclude lessons started before the cutoff from the targets? |
| `features.py` | level estimate, one function for both paths, parity check | What kind of bug would the parity check catch? |
| `baselines.py` | popular, next in sequence, kNN, fallback | Why is "next in sequence" so strong on MAP but bad online? |
| `als.py` | confidence, closed-form solve, `Y^T Y` trick, fold-in | Derive `x_u` by setting the gradient to zero. Why does the loss never go up? |
| `rerank.py` | penalties, prerequisite filter, placement, MMR | What happens to an advanced newcomer without the placement rule? |
| `policy.py` | cold-start switch, save/load | Why not use ALS for a learner with one lesson? |
| `metrics.py` | recall, NDCG, MAP, coverage, novelty | Compute NDCG for `[7, 1, 8]` with relevant `{7, 8}` by hand. |
| `evaluate.py` | segments, ground-truth "too hard" | Why report cold start separately? |
| `abtest.py` | randomisation, Welch, CUPED, delta method, SRM | Why can CUPED not bias the effect? Why use the delta method for abandonment? |
| `api.py`, `artifacts.py` | serving from artefacts, unknown learners | How would the event store work in production? |
| `report.py` | every number comes from code | Which sentences in results.md are computed, and which are fixed text? |

## 3. Key concepts in my own words (fill these in)

- **Implicit vs explicit feedback:** ...
- **Confidence weighting (c = 1 + alpha r):** ...
- **Alternating least squares and why each step is a ridge regression:** ...
- **Fold-in:** ...
- **Time-based split and data leakage in recommenders:** ...
- **Recall@k vs NDCG@k vs MAP@k; coverage and novelty:** ...
- **Cold start:** ...
- **MMR (maximal marginal relevance):** ...
- **Exposure bias / logging policy bias in offline evaluation:** ...
- **A/B test: randomisation unit, primary vs guardrail metric:** ...
- **CUPED:** ...
- **Delta method for ratio metrics:** ...
- **Training/serving skew and feature parity:** ...

## 4. Interview questions this repo answers

1. *How does implicit ALS differ from ALS for ratings?* Every cell counts with a confidence;
   zeros are weak negatives. Show the `Y^T Y + Y_u^T (C_u - I) Y_u` trick.
2. *How do you recommend for a user who was not in training?* Fold-in with frozen item
   factors; for 0-3 interactions, a content/popularity fallback (0.361 vs 0.232 recall on cold
   start here).
3. *How do you evaluate a recommender offline without leakage?* One global time cutoff,
   completions after it as labels, already-started lessons excluded, a test that rewrites the
   future and checks nothing changes.
4. *Why did your offline and online results disagree?* Exposure bias (next-in-sequence wins
   MAP), and offline recall cannot see the cost of a too-hard lesson. The re-ranker barely
   moves recall but cuts too-hard picks from 23.2% to 1.6%.
5. *How would you run the A/B test?* Learner-level randomisation, completions per learner per
   week as primary (zeros included), abandonment as guardrail with delta-method CI, CUPED, SRM
   check, a fixed duration decided in advance.
6. *What would you put in production first?* The fallback and the prerequisite filter: cheap,
   explainable, and they protect new learners.
7. *How do you avoid training/serving skew?* One feature function, two data paths, a parity
   gate before saving artefacts.

## 5. Make it mine (do at least three)

- [ ] Download EdNet KT1 (or export a MOOCCube user-video log), run `recsys evaluate --data ...`
      and compare popularity, kNN and ALS on real data.
- [ ] Add a third A/B arm: ALS without re-ranking. How much of B's lift comes from the rules?
- [ ] Replace the hand-set penalties with a logistic model of P(complete | gap, prerequisites,
      ALS score) fitted on the history.
- [ ] Add inverse propensity scoring to estimate the online effect from logged data.
- [ ] Use real HaiEduTech lesson names and levels for the catalogue.
- [ ] Install `.[deep]`, run the two-tower test, and add the model to the report.

## 6. Quality checks I should be able to explain

- `make check` = `ruff check`, `ruff format --check`, `mypy src`, `pytest --cov-fail-under=80`.
- Why the leakage test rewrites the *future* and expects the *fitted model* to be identical.
- Why the ALS test checks the loss after every iteration, not only at the end.
- Why the parity check compares *all* learners and requires an exact 0.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] Every number in the README is from `docs/results.md`, which I regenerated myself.
- [ ] The README says the data is synthetic and that the simulator was built with the effects
      the re-ranker uses.
- [ ] The two-tower model is described as optional and not run; I run it before claiming any
      result.
- [ ] Anything I have not built is under "What I'd do next".

Note: the Docker image was not built in the environment this repo was first made in (no Docker
daemon there). CI builds it; run `docker build -t learning-recommender .` once yourself before a
demo.

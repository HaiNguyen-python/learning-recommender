"""FastAPI service: `/recommend?learner_id=&k=` and `/health`.

The service loads the artefacts written by `recsys train`: the policy bundle and an event store.
For each request it reads the learner's history from the store, computes features with the
same `learner_features` function as the training table, and calls the same
`LearningRecommender.recommend` that the offline evaluation and the simulated A/B test used.
An unknown learner id is a new learner: they get the cold-start fallback, not an error.
"""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from .artifacts import Served
from .features import learner_features
from .rerank import prereqs_met


class Lesson(BaseModel):
    lesson_id: str
    topic: str
    level: int
    prereqs_met: bool


class RecommendResponse(BaseModel):
    learner_id: str
    known_learner: bool
    strategy: str
    est_level: float
    features: dict[str, float]
    items: list[Lesson]


def recommend(served: Served, learner_id: str, k: int) -> RecommendResponse:
    rec, cat = served.recommender, served.recommender.catalog
    history = served.store.history(learner_id, served.as_of)
    result = rec.recommend(history, k)
    ids = cat.lesson_ids
    return RecommendResponse(
        learner_id=learner_id,
        known_learner=learner_id in served.store,
        strategy=result.strategy,
        est_level=round(result.est_level, 3),
        features=learner_features(history, cat, served.as_of),
        items=[
            Lesson(
                lesson_id=ids[i],
                topic=cat.topics[cat.topic[i]],
                level=int(cat.level[i]),
                prereqs_met=prereqs_met(i, history, cat, result.est_level),
            )
            for i in result.items
        ],
    )


def create_app(served: Served) -> FastAPI:
    app = FastAPI(title="recsys", version="0.1.0")
    app.state.served = served

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        s: Served = request.app.state.served
        return {"status": "ok", "lessons": len(s.recommender.catalog), "learners": len(s.store)}

    @app.get("/recommend", response_model=RecommendResponse)
    def recommend_endpoint(
        request: Request,
        learner_id: str = Query(min_length=1, max_length=64),
        k: int = Query(10, ge=1, le=50),
    ) -> RecommendResponse:
        return recommend(request.app.state.served, learner_id, k)

    return app

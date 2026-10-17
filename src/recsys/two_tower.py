"""Optional two-tower model (needs `pip install -e ".[deep]"`, i.e. torch).

Not run in CI or in the environment where this repo was built (torch could not be installed
there), so it has no reported numbers. It is here to show how the same `Scorer` interface takes
a neural retrieval model.

- Lesson tower: embedding(lesson) + embedding(topic) + embedding(level) -> linear layer.
- Learner tower: mean of lesson embeddings over the learner's started lessons -> small MLP.
- Score: dot product of the two towers.
- Training: leave-one-out. For every learner and every started lesson, the other started
  lessons are the context and that lesson is the target. The catalogue has only ~300 lessons,
  so a full softmax over all lessons is cheap and exact (no negative sampling needed).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .domain import Catalog, LearnerHistory


class TwoTower:
    name = "Two-tower (torch)"

    def __init__(
        self, catalog: Catalog, dim: int = 32, epochs: int = 10, lr: float = 1e-2, seed: int = 0
    ) -> None:
        self.catalog, self.dim, self.epochs, self.lr, self.seed = catalog, dim, epochs, lr, seed
        self._model: Any = None

    def _build(self, n_items: int) -> Any:
        import torch
        from torch import nn

        cat = self.catalog
        dim = self.dim

        class Towers(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.item = nn.Embedding(n_items, dim)
                self.topic = nn.Embedding(len(cat.topics), dim)
                self.level = nn.Embedding(int(cat.level.max()) + 1, dim)
                self.item_out = nn.Linear(dim, dim)
                self.bag = nn.EmbeddingBag(n_items, dim, mode="mean")
                self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                self.register_buffer("topic_ids", torch.as_tensor(cat.topic, dtype=torch.long))
                self.register_buffer("level_ids", torch.as_tensor(cat.level, dtype=torch.long))

            def items(self) -> torch.Tensor:
                ids = torch.arange(n_items)
                x = self.item(ids) + self.topic(self.topic_ids) + self.level(self.level_ids)
                return self.item_out(x)

            def users(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
                return self.user_mlp(self.bag(flat, offsets))

        return Towers()

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> TwoTower:
        import torch

        torch.manual_seed(self.seed)
        contexts, targets = [], []
        for h in histories.values():
            started = sorted(h.started)
            if len(started) < 2:
                continue
            for target in started:
                contexts.append([i for i in started if i != target])
                targets.append(target)
        model = self._build(n_items)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(targets)), max(len(targets) // 256, 1)):
                ctx = [contexts[b] for b in batch]
                flat = torch.as_tensor([i for c in ctx for i in c], dtype=torch.long)
                offsets = torch.as_tensor(np.cumsum([0] + [len(c) for c in ctx[:-1]]))
                logits = model.users(flat, offsets) @ model.items().T
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.as_tensor([targets[b] for b in batch], dtype=torch.long)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model = model
        return self

    def score(self, history: LearnerHistory) -> np.ndarray:
        import torch

        model = self._model
        with torch.no_grad():
            items = model.items()
            if not history.started:
                return np.zeros(items.shape[0])
            flat = torch.as_tensor(sorted(history.started), dtype=torch.long)
            user = model.users(flat, torch.zeros(1, dtype=torch.long))
            return (user @ items.T).numpy()[0]

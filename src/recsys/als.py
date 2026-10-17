"""Implicit-feedback matrix factorisation: ALS with confidence weights (Hu, Koren, Volinsky 2008).

Data: r_ui >= 0 is how strongly learner u interacted with lesson i (see `EVENT_WEIGHT`).

    preference  p_ui = 1 if r_ui > 0 else 0
    confidence  c_ui = 1 + alpha * r_ui

Every (u, i) cell counts, including the zeros: a lesson never opened is weak evidence of no
interest (confidence 1), a finished lesson strong evidence of interest. The loss is

    L(X, Y) = sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg * (sum_u |x_u|^2 + sum_i |y_i|^2)

With Y fixed, L is a ridge regression for each x_u with a closed form:

    x_u = (Y^T C_u Y + reg I)^-1 Y^T C_u p_u

C_u is diagonal, and only the lessons u touched differ from 1, so
Y^T C_u Y = Y^T Y + Y_u^T (C_u - I) Y_u is cheap: one shared f x f product plus a small
correction from u's own lessons. Lessons are solved the same way with X fixed. Each half-step
is an exact minimisation, so L never goes up (a test checks this).

The user step is also how a learner is scored at serving time ("fold-in"): their factor is
solved from their current history against the frozen lesson factors. Training ends with a
user step, so a training learner's stored factor and their fold-in factor are identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .domain import LearnerHistory


@dataclass
class ImplicitALS:
    factors: int = 32
    reg: float = 1.0
    alpha: float = 10.0
    iterations: int = 12
    seed: int = 0
    name: str = "ALS (implicit)"
    item_factors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    user_factors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    loss_history: list[float] = field(default_factory=list)

    def fit_matrix(self, r: sparse.csr_matrix) -> ImplicitALS:
        """Alternate exact item and user solves; record the loss after each full iteration."""
        rng = np.random.default_rng(self.seed)
        n_users, n_items = r.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))
        r_t = sparse.csr_matrix(r.T)
        self.loss_history = []
        for _ in range(self.iterations):
            self.item_factors = self._solve_all(r_t, self.user_factors)
            self.user_factors = self._solve_all(r, self.item_factors)
            self.loss_history.append(self.loss(r))
        return self

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> ImplicitALS:
        return self.fit_matrix(interaction_matrix(list(histories.values()), n_items))

    def _solve_all(self, r: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        gram = fixed.T @ fixed
        out = np.zeros((r.shape[0], self.factors))
        for row in range(r.shape[0]):
            start, end = r.indptr[row], r.indptr[row + 1]
            out[row] = self._solve_one(gram, fixed, r.indices[start:end], r.data[start:end])
        return out

    def _solve_one(
        self, gram: np.ndarray, fixed: np.ndarray, idx: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """x = (F^T F + F_s^T (C_s - I) F_s + reg I)^-1 F_s^T C_s 1, over the touched rows s."""
        conf = 1.0 + self.alpha * weights
        f_s = fixed[idx]
        a = gram + (f_s.T * (conf - 1.0)) @ f_s + self.reg * np.eye(self.factors)
        b = f_s.T @ conf
        return np.linalg.solve(a, b)

    def user_vector(self, history: LearnerHistory) -> np.ndarray:
        """Fold-in: solve the learner's factor from their history against frozen lesson factors."""
        pairs = [(i, w) for i, w in history.weight.items() if w > 0]
        if not pairs:
            return np.zeros(self.factors)
        idx = np.array([i for i, _ in pairs], dtype=int)
        w = np.array([w for _, w in pairs], dtype=float)
        gram = self.item_factors.T @ self.item_factors
        return self._solve_one(gram, self.item_factors, idx, w)

    def score(self, history: LearnerHistory) -> np.ndarray:
        return self.item_factors @ self.user_vector(history)

    def loss(self, r: sparse.csr_matrix) -> float:
        """The full objective L(X, Y), zeros included (dense: fine at this catalogue size)."""
        pred = self.user_factors @ self.item_factors.T
        total = float((pred**2).sum())  # every cell as if p=0, c=1
        coo = r.tocoo()
        s = pred[coo.row, coo.col]
        conf = 1.0 + self.alpha * coo.data
        total += float((conf * (1.0 - s) ** 2 - s**2).sum())  # fix the observed cells
        reg = (self.user_factors**2).sum() + (self.item_factors**2).sum()
        return total + self.reg * float(reg)


def interaction_matrix(histories: list[LearnerHistory], n_items: int) -> sparse.csr_matrix:
    """Learners x lessons matrix of r_ui (rows in the order given)."""
    rows, cols, vals = [], [], []
    for u, h in enumerate(histories):
        for lesson, w in h.weight.items():
            if w <= 0:  # an outcome without its view carries no preference signal
                continue
            rows.append(u)
            cols.append(lesson)
            vals.append(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(histories), n_items), dtype=float)

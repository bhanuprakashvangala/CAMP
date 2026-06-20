"""Shared zoo helpers — kept in a real module so pickled estimators can
resolve their qualified class names regardless of which entry point loads
the bucket dict."""
from __future__ import annotations
import numpy as np


class NGBLogMu:
    """Adapter so model.predict(X) returns mu_log (NGBoost LogNormal location).

    This keeps the deployment-time interface uniform across the zoo: every
    retained estimator emits a log-space mean, regardless of whether the
    underlying class natively predicts log-space (LR, kNN, MLP, RF, LightGBM)
    or raw-space with a LogNormal head (NGBoost)."""
    __slots__ = ("ngb",)

    def __init__(self, ngb):
        self.ngb = ngb

    def predict(self, X):
        dist = self.ngb.pred_dist(X)
        return np.asarray(dist.loc, dtype=float)

    def __getstate__(self):
        return {"ngb": self.ngb}

    def __setstate__(self, state):
        self.ngb = state["ngb"]

"""
MiniRocket class definitions — required for joblib unpickling of friend's
.joblib model files. The pickled objects reference these classes by their
fully-qualified name (`src.models.minirocket.MultilabelMiniRocket`), so
this module path must exist in the Python search path at load time.

These classes are copied verbatim from the friend's training code. We
don't need to fit anything here — just make the class definitions
importable so the saved estimators can be reconstructed.

Requires: pip install sktime
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.multioutput import MultiOutputClassifier

try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
except ImportError:
    from sktime.transformations.panel.rocket._minirocket_multivariate import (
        MiniRocketMultivariate,
    )


def _to_sktime_format(X_wave: np.ndarray) -> np.ndarray:
    """(N, seq_len, n_channels) -> (N, n_channels, seq_len)."""
    return np.transpose(X_wave, (0, 2, 1)).astype(np.float32)


class MultilabelMiniRocket:
    """Multi-label MINIROCKET for fault detection."""

    def __init__(self, num_kernels: int = 5000, random_state: int = 42):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.transformer = None
        self.classifier = None

    def fit(self, X_wave, y):
        X = _to_sktime_format(X_wave)
        self.transformer = MiniRocketMultivariate(
            num_kernels=self.num_kernels,
            random_state=self.random_state,
            n_jobs=-1,
        )
        feats = self.transformer.fit_transform(X).astype(np.float32)
        self.classifier = MultiOutputClassifier(
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        )
        self.classifier.fit(feats, y.astype(int))
        return self

    def predict(self, X_wave):
        X = _to_sktime_format(X_wave)
        feats = self.transformer.transform(X).astype(np.float32)
        return self.classifier.predict(feats)

    def predict_proba(self, X_wave):
        X = _to_sktime_format(X_wave)
        feats = self.transformer.transform(X).astype(np.float32)
        scores = np.column_stack([
            est.decision_function(feats) for est in self.classifier.estimators_
        ])
        return 1.0 / (1.0 + np.exp(-scores))


class MulticlassMiniRocket:
    """Single-label multiclass MINIROCKET for severity."""

    def __init__(self, num_kernels: int = 5000, random_state: int = 42):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.transformer = None
        self.classifier = None
        self.classes_ = None

    def fit(self, X_wave, y):
        X = _to_sktime_format(X_wave)
        self.transformer = MiniRocketMultivariate(
            num_kernels=self.num_kernels,
            random_state=self.random_state,
            n_jobs=-1,
        )
        feats = self.transformer.fit_transform(X).astype(np.float32)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(feats, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X_wave):
        X = _to_sktime_format(X_wave)
        feats = self.transformer.transform(X).astype(np.float32)
        return self.classifier.predict(feats)

    def predict_proba(self, X_wave):
        X = _to_sktime_format(X_wave)
        feats = self.transformer.transform(X).astype(np.float32)
        scores = self.classifier.decision_function(feats)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

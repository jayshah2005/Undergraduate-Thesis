"""The one logistic-regression recipe used for variety and disease, on the leaf features of helpers.data.

StandardScaler, then logistic regression with balanced class weights, fitted on the 3 training rows;
C is chosen from C_GRID by the validation row's log-loss, weighted so each class counts equally
(as the class-weighted losses the CNNs stop on). The validation row only picks C and never fits
the model, as it only picks the checkpoint in the CNNs.
"""
from __future__ import annotations

import numpy as np
# Why scikit-learn: lbfgs logistic regression with a scaler, the model behind every earlier baseline,
# has no drop-in PyTorch equivalent.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def make_model(C: float):
    """Standardise each feature (fit on the training leaves only), then class-balanced logistic regression.

    :param C: Inverse regularization strength.
    :type C: float
    :return: Unfitted scikit-learn pipeline.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=10_000))


def fit_with_validation(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[float, object]:
    """Fit one model per C on the training leaves; keep the one with the lowest balanced validation log-loss.

    n: the number of leaves. d: the number of features.

    :param X: Features, shape (n, d).
    :type X: np.ndarray
    :param y: Class per leaf (0/1, or 0-3 for variety), length n.
    :type y: np.ndarray
    :param tr: Training leaf indices.
    :type tr: np.ndarray
    :param va: Validation leaf indices.
    :type va: np.ndarray
    :return: The chosen C and its fitted model (ties go to the smaller C).
    :rtype: tuple
    """
    labels = np.unique(y[tr])
    weight = compute_sample_weight("balanced", y[va])
    fits = {C: make_model(C).fit(X[tr], y[tr]) for C in C_GRID}
    losses = {C: log_loss(y[va], m.predict_proba(X[va]), labels=labels, sample_weight=weight) for C, m in fits.items()}
    best = min(losses, key=losses.get)
    return best, fits[best]

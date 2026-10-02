"""Agreement metrics between a judge and labels, plus the bias probes."""

from __future__ import annotations

import numpy as np
from scipy import stats


def quadratic_weighted_kappa(
    y_true: list[int], y_pred: list[int], low: int = 1, high: int = 5
) -> float:
    """Cohen's kappa with quadratic weights: 1 = perfect, 0 = chance level."""
    k = high - low + 1
    observed = np.zeros((k, k))
    for t, p in zip(y_true, y_pred, strict=True):
        observed[t - low, p - low] += 1
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    denom = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denom) if denom else 1.0


def agreement(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    t, p = np.array(y_true), np.array(y_pred)
    return {
        "exact": float((t == p).mean()),
        "within_1": float((np.abs(t - p) <= 1).mean()),
        "qwk": quadratic_weighted_kappa(list(t), list(p)),
        "mean_bias": float((p - t).mean()),  # > 0: judge is more generous than the label
    }


def spearman(x: list[float], y: list[float]) -> float:
    return float(stats.spearmanr(x, y).statistic)

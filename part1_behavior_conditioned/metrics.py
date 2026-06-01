"""Metrics for token-level log-TRT regression."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_token_regression_metrics(eval_pred) -> dict[str, float]:
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = eval_pred.label_ids
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    active = np.isfinite(labels) & (labels != -100.0)
    predictions = predictions[active]
    labels = labels[active]
    if labels.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    errors = predictions - labels
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    pearson = _safe_corr(predictions, labels)
    spearman = _safe_corr(_rank(predictions), _rank(labels))
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


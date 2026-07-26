"""Prediction metrics and tidy persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "sensitivity": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "mcc": matthews_corrcoef(y_true, pred),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "negative_log_likelihood": log_loss(y_true, np.column_stack([1 - prob, prob]), labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, prob),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    try:
        metrics["auroc"] = roc_auc_score(y_true, prob)
    except ValueError:
        metrics["auroc"] = np.nan
    try:
        metrics["average_precision"] = average_precision_score(y_true, prob)
    except ValueError:
        metrics["average_precision"] = np.nan
    metrics.update(calibration_errors(y_true, prob))
    return metrics


def calibration_errors(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (prob >= left) & (prob < right if right < 1 else prob <= right)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask] == (prob[mask] >= 0.5)))
        gap = abs(acc - conf)
        ece += gap * float(np.mean(mask))
        mce = max(mce, gap)
    return {"expected_calibration_error": ece, "maximum_calibration_error": mce}


def prediction_frame(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    sample_ids: np.ndarray,
    seed: int,
    model: str,
    dataset: str,
    split: str,
    configuration_id: str,
) -> pd.DataFrame:
    prob = np.asarray(prob)
    return pd.DataFrame(
        {
            "configuration_id": configuration_id,
            "sample_id": sample_ids,
            "seed": seed,
            "model": model,
            "dataset": dataset,
            "split": split,
            "label": y_true.astype(int),
            "probability": prob,
            "predicted_label": (prob >= 0.5).astype(int),
            "logit_or_expectation": np.log(np.clip(prob, 1e-7, 1 - 1e-7) / np.clip(1 - prob, 1e-7, 1)),
        }
    )


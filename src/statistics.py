"""Statistical summaries and comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def bootstrap_metric(y_true: np.ndarray, prob: np.ndarray, metric: str, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, pr = y_true[idx], prob[idx]
        pred = pr >= 0.5
        if metric == "accuracy":
            value = accuracy_score(yt, pred)
        elif metric == "f1":
            value = f1_score(yt, pred, average="macro", zero_division=0)
        elif metric == "auroc":
            value = roc_auc_score(yt, pr) if len(np.unique(yt)) == 2 else np.nan
        else:
            value = float(np.mean(pr))
        values.append(value)
    return pd.DataFrame({"metric": metric, "value": values})


def mcnemar_errors(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    n = b01 + b10
    p = float(binomtest(min(b01, b10), n, 0.5).pvalue) if n else 1.0
    return {"mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": p, "effect_discordance": abs(b01 - b10) / (n + 1e-12)}


def wilcoxon_signed(values_a: list[float], values_b: list[float]) -> dict[str, float]:
    if len(values_a) < 2:
        return {"wilcoxon_p": np.nan, "effect_median_difference": np.nan}
    stat = wilcoxon(values_a, values_b, zero_method="zsplit")
    return {"wilcoxon_p": float(stat.pvalue), "effect_median_difference": float(np.median(np.asarray(values_a) - np.asarray(values_b)))}


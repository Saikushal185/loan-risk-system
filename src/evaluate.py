"""Extra evaluation metrics: ROC-AUC, PR-AUC and a KS statistic."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov separation between good and bad applicants."""
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def evaluate(y_true, y_score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
    }

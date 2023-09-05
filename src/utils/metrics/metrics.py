from typing import Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


def accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute accuracy.

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        Accuracy on data
    """
    return balanced_accuracy_score(targets, preds)


def auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute ROC-AUC curve.

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        ROC-AUC curve on data
    """
    return roc_auc_score(targets, preds)


def f1(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute F1-score.

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        F1-score on data
    """
    return f1_score(targets, preds)


def recall(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute recall

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        Recall on data
    """
    return recall_score(targets, preds)


def precision(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute precision

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        Precision on data
    """
    return precision_score(targets, preds)


def compute_metrics(targets: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute all pertinent metrics on data.

    Computes accuracy, ROC-AUC, F1-score, recall, and precision on provided data.

    Args:
        targets: Ground-truth labels
        preds: Model predictions

    Returns:
        Tuple of accuracy, ROC-AUC, f1-score, recall, and precision
    """
    acc_val = accuracy(targets, preds)
    auc_val = auc(targets, preds)
    f1_val = f1(targets, preds)
    recall_val = recall(targets, preds)
    precision_val = precision(targets, preds)

    return acc_val, auc_val, f1_val, recall_val, precision_val

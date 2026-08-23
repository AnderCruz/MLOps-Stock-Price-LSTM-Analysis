from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true,
    probabilities,
) -> dict:
    """
    Evaluate a binary classifier from true labels
    and positive-class probabilities.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.

    probabilities:
        Predicted probability for the positive class.

    Returns
    -------
    dict
        Classification metrics and predictions.

    Notes
    -----
    Predictions use a 0.5 probability threshold.
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    # ==================================================
    # VALIDATION
    # ==================================================

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must "
            "have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "y_true and probabilities cannot be empty."
        )

    unique_targets = np.unique(
        y_true
    )

    if not np.all(
        np.isin(
            unique_targets,
            [0, 1],
        )
    ):
        raise ValueError(
            "y_true must contain binary "
            "labels: 0 and 1."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            "probabilities must contain "
            "only finite values."
        )

    if np.any(
        probabilities < 0
    ) or np.any(
        probabilities > 1
    ):
        raise ValueError(
            "probabilities must be "
            "between 0 and 1."
        )

    # ==================================================
    # PREDICTIONS
    # ==================================================

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ==================================================
    # CLASSIFICATION METRICS
    # ==================================================

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # ==================================================
    # ROC-AUC
    # ==================================================

    unique_classes = np.unique(
        y_true
    )

    if len(unique_classes) == 2:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        roc_auc = None

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    # ==================================================
    # RESULT
    # ==================================================

    return {
        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            None
            if roc_auc is None
            else float(roc_auc)
        ),

        "confusion_matrix": (
            cm.astype(int).tolist()
        ),

        "predictions": (
            predictions.astype(int).tolist()
        ),
    }

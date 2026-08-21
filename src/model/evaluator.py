from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


@dataclass
class EvaluationResult:
    """Evaluation metrics for a forecasting model."""

    mae: float
    rmse: float
    mape_percent: float
    direction_accuracy_percent: float

    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a serializable dictionary."""

        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape_percent": self.mape_percent,
            "direction_accuracy_percent": (
                self.direction_accuracy_percent
            ),
            "n_samples": self.n_samples,
        }


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Zero-valued targets are excluded because MAPE is undefined
    for zero denominators.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                / y_true[mask]
            )
        )
        * 100
    )


def calculate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate directional accuracy.

    Direction is based on the change between consecutive
    actual/predicted prices.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) < 2:
        return 0.0

    actual_direction = np.sign(
        np.diff(y_true)
    )

    predicted_direction = np.sign(
        np.diff(y_pred)
    )

    return float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    """
    Evaluate model predictions against actual values.
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same number of samples."
        )

    if not np.all(
        np.isfinite(y_true)
    ):
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError(
            "y_pred contains non-finite values."
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mape = calculate_mape(
        y_true,
        y_pred,
    )

    direction_accuracy = (
        calculate_direction_accuracy(
            y_true,
            y_pred,
        )
    )

    return EvaluationResult(
        mae=float(mae),
        rmse=rmse,
        mape_percent=float(mape),
        direction_accuracy_percent=(
            direction_accuracy
        ),
        n_samples=len(y_true),
    )

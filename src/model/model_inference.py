from __future__ import annotations

from typing import Any

import numpy as np


class ModelInferenceError(Exception):
    """Raised when model inference cannot be performed."""


class ModelInference:
    """
    Lightweight inference service for the directional LSTM.

    Expected input:

        (sequence_length, n_features)

    The service:

        1. validates the input shape;
        2. applies the fitted scaler;
        3. creates the batch dimension;
        4. obtains the model probability;
        5. converts the probability into UP/DOWN.
    """

    def __init__(
        self,
        *,
        model: Any,
        scaler: Any,
        sequence_length: int,
        feature_columns: list[str],
        threshold: float = 0.50,
    ) -> None:

        if sequence_length <= 0:
            raise ModelInferenceError(
                "sequence_length must be positive."
            )

        if not feature_columns:
            raise ModelInferenceError(
                "feature_columns cannot be empty."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ModelInferenceError(
                "threshold must be between 0 and 1."
            )

        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.feature_columns = list(
            feature_columns
        )
        self.threshold = threshold

    def predict(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """
        Generate a directional prediction.

        Parameters
        ----------
        features:
            Array with shape:

                (sequence_length, n_features)

        Returns
        -------
        dict
            Contains:

                probability
                direction
        """

        X = np.asarray(
            features,
            dtype=float,
        )

        expected_features = len(
            self.feature_columns
        )

        if X.ndim != 2:
            raise ModelInferenceError(
                "Features must be a 2-dimensional array."
            )

        if X.shape[0] != self.sequence_length:
            raise ModelInferenceError(
                "Invalid sequence length: "
                f"expected {self.sequence_length}, "
                f"received {X.shape[0]}."
            )

        if X.shape[1] != expected_features:
            raise ModelInferenceError(
                "Invalid number of features: "
                f"expected {expected_features}, "
                f"received {X.shape[1]}."
            )

        if not np.isfinite(X).all():
            raise ModelInferenceError(
                "Features contain non-finite values."
            )

        try:
            X_scaled = self.scaler.transform(
                X
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Failed to transform features "
                "with the scaler."
            ) from exc

        X_batch = np.asarray(
            X_scaled,
            dtype=float,
        ).reshape(
            1,
            self.sequence_length,
            expected_features,
        )

        try:
            prediction = self.model.predict(
                X_batch,
                verbose=0,
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Model prediction failed."
            ) from exc

        probabilities = np.asarray(
            prediction,
            dtype=float,
        ).reshape(-1)

        if probabilities.size != 1:
            raise ModelInferenceError(
                "Model must return exactly "
                "one probability."
            )

        probability = float(
            probabilities[0]
        )

        if not np.isfinite(probability):
            raise ModelInferenceError(
                "Model returned a non-finite "
                "probability."
            )

        if not 0.0 <= probability <= 1.0:
            raise ModelInferenceError(
                "Model returned an invalid "
                "probability."
            )

        direction = (
            "UP"
            if probability >= self.threshold
            else "DOWN"
        )

        return {
            "probability": probability,
            "direction": direction,
        }

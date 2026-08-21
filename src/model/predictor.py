from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model.lstm_model import rmse


class ModelPredictor:
    """
    Load a versioned LSTM model artifact and perform inference.
    """

    def __init__(
        self,
        model_dir: str,
    ) -> None:

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: "
                f"{self.model_dir}"
            )

        self.model_path = (
            self.model_dir / "model.keras"
        )

        self.scaler_path = (
            self.model_dir / "scaler.pkl"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: "
                f"{self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found: "
                f"{self.scaler_path}"
            )

        # --------------------------------------------------
        # Load TensorFlow model
        # --------------------------------------------------
        #
        # rmse must be imported before loading because the
        # serialized Keras model contains this custom metric.
        #

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "rmse": rmse,
                "StockPriceLSTM>rmse": rmse,
            },
        )

        # --------------------------------------------------
        # Load scaler
        # --------------------------------------------------

        with self.scaler_path.open(
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

    def predict_scaled(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return predictions in scaled model space.
        """

        if X is None or len(X) == 0:
            raise ValueError(
                "Input sequence is empty."
            )

        predictions = self.model.predict(
            X,
            verbose=0,
        )

        return np.asarray(
            predictions
        ).reshape(-1)

    def predict_price(
        self,
        X: np.ndarray,
        target_index: int = 0,
    ) -> np.ndarray:
        """
        Return predictions in original price scale.
        """

        predictions_scaled = (
            self.predict_scaled(X)
        )

        n_samples = len(
            predictions_scaled
        )

        reconstructed = np.zeros(
            (
                n_samples,
                self.scaler.n_features_in_,
            )
        )

        reconstructed[
            :,
            target_index,
        ] = predictions_scaled

        inverted = (
            self.scaler.inverse_transform(
                reconstructed
            )
        )

        return inverted[
            :,
            target_index,
        ]

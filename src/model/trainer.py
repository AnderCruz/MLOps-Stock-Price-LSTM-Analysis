from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf


@dataclass
class TrainingResult:
    """Result of an LSTM training run."""

    model: tf.keras.Model
    history: dict[str, list[float]]
    epochs_completed: int
    best_epoch: int
    best_val_loss: float


def train_lstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 10,
) -> TrainingResult:
    """
    Train an LSTM model using chronological training data.

    Parameters
    ----------
    model:
        Compiled TensorFlow/Keras model.

    X_train:
        Training sequences.

    y_train:
        Training targets.

    epochs:
        Maximum number of training epochs.

    batch_size:
        Number of samples per batch.

    validation_split:
        Fraction of training data used for validation.

    patience:
        Number of epochs without improvement before
        early stopping.

    Returns
    -------
    TrainingResult
        Trained model and training metadata.
    """

    if X_train is None or len(X_train) == 0:
        raise ValueError(
            "X_train is empty."
        )

    if y_train is None or len(y_train) == 0:
        raise ValueError(
            "y_train is empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain "
            "the same number of samples."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if not 0 < validation_split < 1:
        raise ValueError(
            "validation_split must be between 0 and 1."
        )

    if patience < 0:
        raise ValueError(
            "patience cannot be negative."
        )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
    )

    history_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    val_losses = history_dict.get(
        "val_loss",
        [],
    )

    if val_losses:
        best_epoch = int(
            np.argmin(val_losses)
        )

        best_val_loss = float(
            val_losses[best_epoch]
        )

    else:
        best_epoch = (
            len(history_dict.get("loss", [])) - 1
        )

        best_val_loss = float("nan")

    epochs_completed = len(
        history_dict.get("loss", [])
    )

    return TrainingResult(
        model=model,
        history=history_dict,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch + 1,
        best_val_loss=best_val_loss,
    )

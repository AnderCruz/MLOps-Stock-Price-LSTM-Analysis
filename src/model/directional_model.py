from __future__ import annotations

import tensorflow as tf

from keras.saving import register_keras_serializable
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.optimizers import Adam


@register_keras_serializable(
    package="StockDirectionLSTM"
)
def build_directional_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_1_units: int = 72,
    lstm_2_units: int = 48,
    dropout_1: float = 0.2,
    dropout_2: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile an LSTM model for binary
    next-day market direction classification.

    Output:
        0 -> Down / Non-positive return
        1 -> Up / Positive return
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if n_features <= 0:
        raise ValueError(
            "n_features must be greater than zero."
        )

    model = Sequential(
        name="stock_direction_lstm"
    )

    model.add(
        Input(
            shape=(
                sequence_length,
                n_features,
            )
        )
    )

    model.add(
        LSTM(
            units=lstm_1_units,
            activation="tanh",
            return_sequences=True,
        )
    )

    model.add(
        Dropout(
            dropout_1
        )
    )

    model.add(
        LSTM(
            units=lstm_2_units,
            activation="tanh",
            return_sequences=False,
        )
    )

    model.add(
        Dropout(
            dropout_2
        )
    )

    # Binary classification:
    # 0 = Down
    # 1 = Up
    model.add(
        Dense(
            units=1,
            activation="sigmoid",
        )
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model

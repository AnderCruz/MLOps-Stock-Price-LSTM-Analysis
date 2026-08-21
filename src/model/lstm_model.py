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
    package="StockPriceLSTM"
)
def rmse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Root Mean Squared Error."""

    return tf.sqrt(
        tf.reduce_mean(
            tf.square(
                y_true - y_pred
            )
        )
    )


def build_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_1_units: int = 72,
    lstm_2_units: int = 48,
    dropout_1: float = 0.2,
    dropout_2: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile the stock-price LSTM model.
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
        name="stock_price_lstm"
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

    model.add(
        Dense(
            units=1,
            activation="linear",
        )
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=[
            "mae",
            "mape",
            rmse,
        ],
    )

    return model

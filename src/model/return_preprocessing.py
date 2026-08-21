from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model.preprocessing import chronological_train_test_split
from src.model.sequence_builder import build_sequences


@dataclass
class PreparedReturnData:
    X_train: np.ndarray
    y_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    scaler: StandardScaler

    train_data: pd.DataFrame
    test_data: pd.DataFrame

    train_size: int
    test_size: int

    target_column: str
    sequence_length: int
    n_features: int

    def inverse_transform_target(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert scaled target values back to the
        original return scale.

        The StandardScaler was fitted on all model
        features, so the target column must be placed
        into a full feature matrix before inverse
        transformation.
        """

        values = np.asarray(
            values
        ).reshape(-1)

        reconstructed = np.zeros(
            (
                len(values),
                self.n_features,
            ),
            dtype=np.float64,
        )

        target_index = (
            0
        )

        reconstructed[
            :,
            target_index,
        ] = values

        inverse = (
            self.scaler.inverse_transform(
                reconstructed
            )
        )

        return inverse[
            :,
            target_index,
        ]



def prepare_return_dataset(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    test_ratio: float = 0.20,
) -> PreparedReturnData:

    if df is None or df.empty:
        raise ValueError(
            "Cannot prepare return data from an empty DataFrame."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    train_df, test_df = (
        chronological_train_test_split(
            df=df,
            test_ratio=test_ratio,
        )
    )

    if len(train_df) <= sequence_length:
        raise ValueError(
            "Training data does not contain enough "
            "observations for the sequence length."
        )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_df
    )

    test_scaled = scaler.transform(
        test_df
    )

    train_scaled_df = pd.DataFrame(
        train_scaled,
        index=train_df.index,
        columns=train_df.columns,
    )

    test_scaled_df = pd.DataFrame(
        test_scaled,
        index=test_df.index,
        columns=test_df.columns,
    )

    X_train, y_train = build_sequences(
        df=train_scaled_df,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    history = train_scaled_df.tail(
        sequence_length
    )

    test_context = pd.concat(
        [
            history,
            test_scaled_df,
        ]
    )

    X_test, y_test = build_sequences(
        df=test_context,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    expected_train_samples = (
        len(train_df) - sequence_length
    )

    expected_test_samples = len(test_df)

    if len(X_train) != expected_train_samples:
        raise RuntimeError(
            "Unexpected training sequence count."
        )

    if len(X_test) != expected_test_samples:
        raise RuntimeError(
            "Unexpected test sequence count."
        )

    return PreparedReturnData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        train_data=train_df,
        test_data=test_df,
        train_size=len(train_df),
        test_size=len(test_df),
        target_column=target_column,
        sequence_length=sequence_length,
        n_features=len(df.columns),
    )
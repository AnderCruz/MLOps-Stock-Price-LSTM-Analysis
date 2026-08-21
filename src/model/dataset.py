from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.model.preprocessing import (
    chronological_train_test_split,
)
from src.model.sequence_builder import (
    build_sequences,
)


@dataclass
class PreparedModelData:
    """
    Fully prepared dataset for LSTM training and evaluation.

    The scaler is fitted exclusively on the training data.
    """

    X_train: np.ndarray
    y_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    scaler: MinMaxScaler

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
        Convert scaled target values back to the original
        target scale.

        The scaler was fitted on all model features, so we
        reconstruct the full feature matrix before applying
        inverse_transform().
        """

        values = np.asarray(
            values
        ).reshape(-1)

        reconstructed = np.zeros(
            (
                len(values),
                self.n_features,
            )
        )

        reconstructed[:, 0] = values

        inverse = self.scaler.inverse_transform(
            reconstructed
        )

        return inverse[:, 0]


def prepare_model_dataset(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    test_ratio: float = 0.2,
) -> PreparedModelData:
    """
    Prepare chronological train/test sequences for LSTM.

    Processing order:

    1. Validate input.
    2. Chronologically split train/test.
    3. Fit scaler ONLY on training data.
    4. Transform train and test.
    5. Add training history to the test period.
    6. Build LSTM sequences.

    No future test information is used to fit preprocessing.
    """

    # ==================================================
    # 1. INPUT VALIDATION
    # ==================================================

    if df is None or df.empty:
        raise ValueError(
            "Cannot prepare model data from an empty DataFrame."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in DataFrame."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    if len(df) <= sequence_length:
        raise ValueError(
            "Not enough observations for the "
            "requested sequence length."
        )

    # ==================================================
    # 2. CHRONOLOGICAL TRAIN / TEST SPLIT
    # ==================================================

    train_df, test_df = chronological_train_test_split(
        df=df,
        test_ratio=test_ratio,
    )

    if len(train_df) <= sequence_length:
        raise ValueError(
            "Training dataset does not contain enough "
            "observations for the sequence length."
        )

    # ==================================================
    # 3. FIT SCALER ONLY ON TRAINING DATA
    # ==================================================

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        train_df
    )

    test_scaled = scaler.transform(
        test_df
    )

    # ==================================================
    # 4. CONVERT BACK TO DATAFRAMES
    #
    # Preserve:
    # - dates
    # - column names
    # ==================================================

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

    # ==================================================
    # 5. BUILD TRAINING SEQUENCES
    # ==================================================

    X_train, y_train = build_sequences(
        df=train_scaled_df,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    # ==================================================
    # 6. BUILD TEST SEQUENCES
    #
    # The first test prediction needs the final
    # sequence_length observations from TRAIN.
    #
    # Example:
    #
    # TRAIN:
    #       ...
    #       [last 60 rows]
    #
    # TEST:
    #       [first test row]
    #       [second test row]
    #       ...
    #
    # Therefore:
    #
    # last 60 TRAIN rows
    #          +
    #      TEST DATA
    #          ↓
    #   test_with_context
    #          ↓
    #      X_test
    # ==================================================

    historical_context = train_scaled_df.tail(
        sequence_length
    )

    test_with_context = pd.concat(
        [
            historical_context,
            test_scaled_df,
        ]
    )

    X_test, y_test = build_sequences(
        df=test_with_context,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    # ==================================================
    # 7. STRUCTURAL VALIDATION
    # ==================================================

    expected_train_samples = (
        len(train_df) - sequence_length
    )

    expected_test_samples = len(test_df)

    if len(X_train) != expected_train_samples:
        raise RuntimeError(
            "Unexpected number of training sequences. "
            f"Expected {expected_train_samples}, "
            f"received {len(X_train)}."
        )

    if len(X_test) != expected_test_samples:
        raise RuntimeError(
            "Unexpected number of test sequences. "
            f"Expected {expected_test_samples}, "
            f"received {len(X_test)}."
        )

    if len(y_train) != expected_train_samples:
        raise RuntimeError(
            "Unexpected number of training targets. "
            f"Expected {expected_train_samples}, "
            f"received {len(y_train)}."
        )

    if len(y_test) != expected_test_samples:
        raise RuntimeError(
            "Unexpected number of test targets. "
            f"Expected {expected_test_samples}, "
            f"received {len(y_test)}."
        )

    # ==================================================
    # 8. FINAL OBJECT
    # ==================================================

    return PreparedModelData(
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

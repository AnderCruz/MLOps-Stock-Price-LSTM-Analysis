from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model.preprocessing import chronological_train_test_split
from src.model.sequence_builder import build_sequences


@dataclass
class PreparedDirectionData:
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

    def inverse_transform_features(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Restore feature values to original scale.
        """

        values = np.asarray(values)

        if values.ndim != 2:
            raise ValueError(
                "values must be a 2-dimensional array."
            )

        if values.shape[1] != self.n_features:
            raise ValueError(
                "Feature dimension does not match scaler."
            )

        return self.scaler.inverse_transform(
            values
        )


def prepare_direction_dataset(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    test_ratio: float = 0.20,
) -> PreparedDirectionData:

    if df is None or df.empty:
        raise ValueError(
            "Cannot prepare direction data "
            "from an empty DataFrame."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    # --------------------------------------------------
    # Validate binary target
    # --------------------------------------------------

    target_values = (
        df[target_column]
        .dropna()
        .unique()
    )

    if not set(target_values).issubset(
        {0, 1}
    ):
        raise ValueError(
            "Direction target must contain "
            "only 0 and 1."
        )

    if df[target_column].isna().any():
        raise ValueError(
            "Direction target contains NaN values."
        )

    # --------------------------------------------------
    # Feature columns
    # --------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns available."
        )

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Fit scaler ONLY on training features
    # --------------------------------------------------

    scaler = StandardScaler()

    train_features = train_df[
        feature_columns
    ]

    test_features = test_df[
        feature_columns
    ]

    train_scaled = scaler.fit_transform(
        train_features
    )

    test_scaled = scaler.transform(
        test_features
    )

    train_scaled_df = pd.DataFrame(
        train_scaled,
        index=train_df.index,
        columns=feature_columns,
    )

    test_scaled_df = pd.DataFrame(
        test_scaled,
        index=test_df.index,
        columns=feature_columns,
    )

    # --------------------------------------------------
    # Build sequence DataFrames
    #
    # build_sequences() expects the target column
    # to be present in the DataFrame, but it includes
    # every column in X.
    #
    # Therefore we deliberately append the target as
    # the LAST column and remove that column from X
    # immediately after sequence construction.
    # --------------------------------------------------

    train_sequence_df = train_scaled_df.copy()

    train_sequence_df[
        target_column
    ] = train_df[
        target_column
    ].to_numpy()

    test_sequence_df = test_scaled_df.copy()

    test_sequence_df[
        target_column
    ] = test_df[
        target_column
    ].to_numpy()

    ordered_columns = (
        feature_columns
        + [target_column]
    )

    train_sequence_df = (
        train_sequence_df[
            ordered_columns
        ]
    )

    test_sequence_df = (
        test_sequence_df[
            ordered_columns
        ]
    )

    # --------------------------------------------------
    # Training sequences
    # --------------------------------------------------

    X_train_with_target, y_train = (
        build_sequences(
            df=train_sequence_df,
            target_column=target_column,
            sequence_length=sequence_length,
        )
    )

    # Remove target from model inputs.
    X_train = X_train_with_target[
        :, :, :len(feature_columns)
    ]

    # --------------------------------------------------
    # Test sequences
    # --------------------------------------------------

    history = train_sequence_df.tail(
        sequence_length
    )

    test_context = pd.concat(
        [
            history,
            test_sequence_df,
        ]
    )

    X_test_with_target, y_test = (
        build_sequences(
            df=test_context,
            target_column=target_column,
            sequence_length=sequence_length,
        )
    )

    # Remove target from model inputs.
    X_test = X_test_with_target[
        :, :, :len(feature_columns)
    ]

    # --------------------------------------------------
    # Structural validation
    # --------------------------------------------------

    expected_train_samples = (
        len(train_df)
        - sequence_length
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

    if X_train.shape[2] != len(feature_columns):
        raise RuntimeError(
            "Unexpected training feature dimension."
        )

    if X_test.shape[2] != len(feature_columns):
        raise RuntimeError(
            "Unexpected test feature dimension."
        )

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            "X_train contains non-finite values."
        )

    if not np.isfinite(X_test).all():
        raise RuntimeError(
            "X_test contains non-finite values."
        )

    if not set(
        np.unique(y_train)
    ).issubset({0, 1}):
        raise RuntimeError(
            "y_train contains invalid labels."
        )

    if not set(
        np.unique(y_test)
    ).issubset({0, 1}):
        raise RuntimeError(
            "y_test contains invalid labels."
        )

    return PreparedDirectionData(
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
        n_features=len(feature_columns),
    )

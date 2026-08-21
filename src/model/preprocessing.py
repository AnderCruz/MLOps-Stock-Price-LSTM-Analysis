from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessingResult:
    """Result of leakage-safe time-series preprocessing."""

    train_data: pd.DataFrame
    test_data: pd.DataFrame

    train_scaled: np.ndarray
    test_scaled: np.ndarray

    scaler: MinMaxScaler

    train_size: int
    test_size: int
    split_index: int


def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame chronologically.

    No shuffling is performed.

    Parameters
    ----------
    df:
        Feature DataFrame sorted chronologically.

    test_ratio:
        Proportion of observations assigned to the test set.

    Returns
    -------
    train_df, test_df
    """

    if df is None or df.empty:
        raise ValueError(
            "Cannot split an empty DataFrame."
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    split_index = int(
        len(df) * (1 - test_ratio)
    )

    if split_index <= 0:
        raise ValueError(
            "Training set would be empty."
        )

    if split_index >= len(df):
        raise ValueError(
            "Test set would be empty."
        )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    return train_df, test_df


def fit_transform_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
]:
    """
    Fit MinMaxScaler ONLY on training data.

    The fitted scaler is then used to transform
    both training and test data.
    """

    if train_df is None or train_df.empty:
        raise ValueError(
            "Training data is empty."
        )

    if test_df is None or test_df.empty:
        raise ValueError(
            "Test data is empty."
        )

    if list(train_df.columns) != list(
        test_df.columns
    ):
        raise ValueError(
            "Train and test columns must match."
        )

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        train_df
    )

    test_scaled = scaler.transform(
        test_df
    )

    return (
        train_scaled,
        test_scaled,
        scaler,
    )


def prepare_time_series_data(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> PreprocessingResult:
    """
    Perform chronological splitting and
    leakage-safe scaling.
    """

    train_df, test_df = (
        chronological_train_test_split(
            df=df,
            test_ratio=test_ratio,
        )
    )

    (
        train_scaled,
        test_scaled,
        scaler,
    ) = fit_transform_train_test(
        train_df=train_df,
        test_df=test_df,
    )

    return PreprocessingResult(
        train_data=train_df,
        test_data=test_df,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        scaler=scaler,
        train_size=len(train_df),
        test_size=len(test_df),
        split_index=len(train_df),
    )

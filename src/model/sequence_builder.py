from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised LSTM sequences from a DataFrame.

    Each sample uses `sequence_length` historical observations
    to predict the target at the following observation.
    """

    if df is None or df.empty:
        raise ValueError(
            "Cannot build sequences from an empty DataFrame."
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

    if len(df) <= sequence_length:
        raise ValueError(
            "Not enough observations to build sequences."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    values = df.to_numpy(dtype=np.float32)

    target_index = df.columns.get_loc(
        target_column
    )

    X = []
    y = []

    for i in range(
        sequence_length,
        len(values),
    ):
        X.append(
            values[
                i - sequence_length:i
            ]
        )

        y.append(
            values[i, target_index]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def build_train_test_sequences(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build LSTM sequences for chronological train/test data.

    Training sequences are created exclusively from training data.

    Test sequences use the final `sequence_length` observations
    from training data as historical context for the beginning
    of the test period.

    No future test observations are used to construct inputs.
    """

    if train_df is None or train_df.empty:
        raise ValueError(
            "Training DataFrame is empty."
        )

    if test_df is None or test_df.empty:
        raise ValueError(
            "Test DataFrame is empty."
        )

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in training data."
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in test data."
        )

    if list(train_df.columns) != list(
        test_df.columns
    ):
        raise ValueError(
            "Train and test columns must match."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if len(train_df) <= sequence_length:
        raise ValueError(
            "Training data does not contain enough "
            "observations for the requested sequence length."
        )

    if not train_df.index.is_monotonic_increasing:
        raise ValueError(
            "Training data must be sorted chronologically."
        )

    if not test_df.index.is_monotonic_increasing:
        raise ValueError(
            "Test data must be sorted chronologically."
        )

    # --------------------------------------------------
    # TRAIN SEQUENCES
    # --------------------------------------------------

    X_train, y_train = build_sequences(
        df=train_df,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    # --------------------------------------------------
    # TEST SEQUENCES
    # --------------------------------------------------

    # We need historical context immediately before
    # the test period begins.
    history = train_df.tail(
        sequence_length
    )

    test_context = pd.concat(
        [
            history,
            test_df,
        ]
    )

    X_test, y_test = build_sequences(
        df=test_context,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    # `build_sequences()` produces exactly one target
    # for each observation after the initial history.
    # Because the first `sequence_length` rows are the
    # training history, all resulting targets correspond
    # to test observations.
    assert len(X_test) == len(test_df)
    assert len(y_test) == len(test_df)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )

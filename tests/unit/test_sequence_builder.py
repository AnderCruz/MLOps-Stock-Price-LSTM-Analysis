import numpy as np
import pandas as pd
import pytest

from src.model.sequence_builder import (
    build_sequences,
    build_train_test_sequences,
)


def make_dataframe(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature": np.arange(n, dtype=np.float32),
            "Target": np.arange(n, dtype=np.float32) * 10,
        },
        index=pd.date_range(
            "2026-01-01",
            periods=n,
            freq="D",
        ),
    )


def test_build_sequences_creates_expected_shape():
    df = make_dataframe(6)

    X, y = build_sequences(
        df=df,
        target_column="Target",
        sequence_length=3,
    )

    assert X.shape == (3, 3, 2)
    assert y.shape == (3,)


def test_build_sequences_uses_previous_observations():
    df = make_dataframe(6)

    X, y = build_sequences(
        df=df,
        target_column="Target",
        sequence_length=3,
    )

    np.testing.assert_array_equal(
        X[0],
        df.iloc[0:3].to_numpy(dtype=np.float32),
    )

    assert y[0] == pytest.approx(
        df["Target"].iloc[3]
    )


def test_build_sequences_preserves_temporal_order():
    df = make_dataframe(6)

    X, y = build_sequences(
        df=df,
        target_column="Target",
        sequence_length=3,
    )

    assert X[0, 0, 0] == 0
    assert X[0, 1, 0] == 1
    assert X[0, 2, 0] == 2
    assert y[0] == 30


def test_build_sequences_rejects_invalid_sequence_length():
    df = make_dataframe(6)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_sequences(
            df=df,
            target_column="Target",
            sequence_length=0,
        )


def test_build_sequences_rejects_insufficient_data():
    df = make_dataframe(3)

    with pytest.raises(
        ValueError,
        match="Not enough observations",
    ):
        build_sequences(
            df=df,
            target_column="Target",
            sequence_length=3,
        )


def test_build_sequences_rejects_unsorted_data():
    df = make_dataframe(6)

    df = df.iloc[
        [2, 0, 1, 3, 4, 5]
    ]

    with pytest.raises(
        ValueError,
        match="chronologically",
    ):
        build_sequences(
            df=df,
            target_column="Target",
            sequence_length=3,
        )


def test_build_sequences_rejects_missing_target():
    df = make_dataframe(6)

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        build_sequences(
            df=df,
            target_column="Missing",
            sequence_length=3,
        )


def test_train_test_sequences_have_expected_sizes():
    train_df = make_dataframe(8)
    test_df = make_dataframe(3)

    test_df = pd.DataFrame(
    {
        "Feature": [100, 101, 102],
        "Target": [1000, 1010, 1020],
    },
    index=pd.date_range(
        "2026-01-09",
        periods=3,
        freq="D",
    ),
)


    X_train, y_train, X_test, y_test = (
        build_train_test_sequences(
            train_df=train_df,
            test_df=test_df,
            target_column="Target",
            sequence_length=3,
        )
    )

    assert len(X_train) == 5
    assert len(y_train) == 5

    assert len(X_test) == 3
    assert len(y_test) == 3


def test_test_sequences_use_training_history():
    train_df = make_dataframe(8)

    test_df = pd.DataFrame(
        {
            "Feature": [100, 101, 102],
            "Target": [1000, 1010, 1020],
        },
        index=pd.date_range(
            "2026-01-09",
            periods=3,
            freq="D",
        ),
    )

    X_train, y_train, X_test, y_test = (
        build_train_test_sequences(
            train_df=train_df,
            test_df=test_df,
            target_column="Target",
            sequence_length=3,
        )
    )

    # The first test sequence must use the
    # final three training observations as history.
    np.testing.assert_array_equal(
        X_test[0, :, 0],
        [5, 6, 7],
    )

    assert y_test[0] == 1000


def test_test_sequences_do_not_use_future_test_observations():
    train_df = make_dataframe(8)

    test_df = pd.DataFrame(
        {
            "Feature": [100, 101, 102],
            "Target": [1000, 1010, 1020],
        },
        index=pd.date_range(
            "2026-01-09",
            periods=3,
            freq="D",
        ),
    )

    _, _, X_test, _ = build_train_test_sequences(
        train_df=train_df,
        test_df=test_df,
        target_column="Target",
        sequence_length=3,
    )

    # The first test input contains only training history.
    assert 100 not in X_test[0, :, 0]
    assert 101 not in X_test[0, :, 0]
    assert 102 not in X_test[0, :, 0]


def test_train_and_test_columns_must_match():
    train_df = make_dataframe(8)

    test_df = pd.DataFrame(
        {
            "Feature": [100, 101, 102],
            "Target": [1000, 1010, 1020],
            "Extra": [1, 2, 3],
        },
        index=pd.date_range(
            "2026-01-09",
            periods=3,
            freq="D",
        ),
    )


    with pytest.raises(
        ValueError,
        match="columns must match",
    ):
        build_train_test_sequences(
            train_df=train_df,
            test_df=test_df,
            target_column="Target",
            sequence_length=3,
        )

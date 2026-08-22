import numpy as np
import pandas as pd
import pytest

from src.model.direction_preprocessing import (
    prepare_direction_dataset,
)


def make_direction_dataframe(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature_A": np.arange(
                n,
                dtype=np.float32,
            ),
            "Feature_B": np.arange(
                n,
                dtype=np.float32,
            ) * 10,
            "Direction": (
                np.arange(n) % 2
            ).astype(np.int32),
        },
        index=pd.date_range(
            "2026-01-01",
            periods=n,
            freq="D",
        ),
    )


def test_split_is_chronological():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    assert result.train_size == 16
    assert result.test_size == 4

    assert (
        result.train_data.index[-1]
        < result.test_data.index[0]
    )


def test_test_data_is_not_used_to_fit_scaler():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    train_features = result.train_data[
        ["Feature_A", "Feature_B"]
    ]

    expected_mean = (
        train_features.mean().to_numpy()
    )

    expected_scale = (
        train_features.std(
            ddof=0
        ).to_numpy()
    )

    np.testing.assert_allclose(
        result.scaler.mean_,
        expected_mean,
    )

    np.testing.assert_allclose(
        result.scaler.scale_,
        expected_scale,
    )


def test_test_features_are_transformed_using_training_scaler():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    train_features = result.train_data[
        ["Feature_A", "Feature_B"]
    ].to_numpy()

    expected = (
        train_features
        - result.scaler.mean_
    ) / result.scaler.scale_

    # The first test sequence must use
    # the final three training observations
    # as historical context.
    np.testing.assert_allclose(
        result.X_test[0],
        expected[-3:],
        atol=1e-6,
    )


def test_sequence_dimensions_are_correct():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    assert result.X_train.shape == (
        13,
        3,
        2,
    )

    assert result.X_test.shape == (
        4,
        3,
        2,
    )

    assert result.y_train.shape == (13,)
    assert result.y_test.shape == (4,)

    assert result.n_features == 2
    assert result.sequence_length == 3


def test_targets_are_binary():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    assert set(
        np.unique(result.y_train)
    ).issubset({0, 1})

    assert set(
        np.unique(result.y_test)
    ).issubset({0, 1})


def test_invalid_target_values_are_rejected():
    df = make_direction_dataframe(20)

    df.loc[
        df.index[5],
        "Direction",
    ] = 2

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        prepare_direction_dataset(
            df=df,
            target_column="Direction",
            sequence_length=3,
        )


def test_nan_target_is_rejected():
    df = make_direction_dataframe(20)

    df.loc[
        df.index[5],
        "Direction",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="contains NaN",
    ):
        prepare_direction_dataset(
            df=df,
            target_column="Direction",
            sequence_length=3,
        )


def test_unsorted_dataframe_is_rejected():
    df = make_direction_dataframe(20)

    df = df.iloc[
        [2, 0, 1] + list(range(3, 20))
    ]

    with pytest.raises(
        ValueError,
        match="chronologically",
    ):
        prepare_direction_dataset(
            df=df,
            target_column="Direction",
            sequence_length=3,
        )


def test_missing_target_column_is_rejected():
    df = make_direction_dataframe(20)

    df = df.drop(
        columns=["Direction"]
    )

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        prepare_direction_dataset(
            df=df,
            target_column="Direction",
            sequence_length=3,
        )


def test_inverse_transform_restores_original_features():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
        test_ratio=0.20,
    )

    original = result.train_data[
        ["Feature_A", "Feature_B"]
    ].iloc[:3]

    scaled = result.scaler.transform(
        original
    )

    restored = (
        result.inverse_transform_features(
            scaled
        )
    )

    np.testing.assert_allclose(
        restored,
        original,
        atol=1e-5,
    )


def test_inverse_transform_rejects_wrong_dimensions():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
    )

    with pytest.raises(
        ValueError,
        match="2-dimensional",
    ):
        result.inverse_transform_features(
            np.array([1.0, 2.0])
        )


def test_inverse_transform_rejects_wrong_feature_count():
    df = make_direction_dataframe(20)

    result = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=3,
    )

    with pytest.raises(
        ValueError,
        match="Feature dimension",
    ):
        result.inverse_transform_features(
            np.zeros((2, 99))
        )
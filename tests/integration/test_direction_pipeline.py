import numpy as np
import pandas as pd

from src.model.direction_preprocessing import (
    prepare_direction_dataset,
)
from src.model.direction_target import (
    add_direction_target,
)
from src.model.return_features import (
    add_return_features,
)


def make_price_dataframe(n: int = 40) -> pd.DataFrame:
    prices = np.linspace(
        100.0,
        140.0,
        n,
    )

    return pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(
            "2026-01-01",
            periods=n,
            freq="D",
        ),
    )


def test_direction_data_pipeline_is_leakage_safe():
    prices = make_price_dataframe()

    features = add_return_features(
        prices,
        price_column="Close",
    )

    labelled = add_direction_target(
        features,
        return_column="Return",
        target_column="Direction",
    )

    result = prepare_direction_dataset(
        df=labelled[
            [
                "Return",
                "Return_5D",
                "Return_10D",
                "Direction",
            ]
        ],
        target_column="Direction",
        sequence_length=5,
        test_ratio=0.20,
    )

    assert result.train_size > 0
    assert result.test_size > 0

    assert (
        result.train_data.index[-1]
        < result.test_data.index[0]
    )

    assert result.X_train.ndim == 3
    assert result.X_test.ndim == 3

    assert result.X_train.shape[1] == 5
    assert result.X_test.shape[1] == 5

    assert result.X_train.shape[2] == 3
    assert result.X_test.shape[2] == 3

    assert len(result.X_train) == len(
        result.y_train
    )

    assert len(result.X_test) == len(
        result.y_test
    )

    assert set(
        np.unique(result.y_train)
    ).issubset({0, 1})

    assert set(
        np.unique(result.y_test)
    ).issubset({0, 1})


def test_pipeline_preserves_temporal_boundaries():
    prices = make_price_dataframe()

    features = add_return_features(
        prices,
        price_column="Close",
    )

    labelled = add_direction_target(
        features,
        return_column="Return",
        target_column="Direction",
    )

    data = labelled[
        [
            "Return",
            "Return_5D",
            "Return_10D",
            "Direction",
        ]
    ]

    result = prepare_direction_dataset(
        df=data,
        target_column="Direction",
        sequence_length=5,
        test_ratio=0.20,
    )

    train_end = result.train_data.index[-1]
    test_start = result.test_data.index[0]

    assert train_end < test_start

    # Test inputs must not contain observations
    # from after the corresponding prediction period.
    assert result.X_test.shape[0] == result.test_size
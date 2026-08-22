import pandas as pd
import pytest

from src.model.direction_target import add_direction_target


def test_direction_target_uses_next_day_return():
    df = pd.DataFrame(
        {"Return": [0.01, -0.02, 0.03]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    result = add_direction_target(df)

    assert result["Direction"].tolist() == [0, 1]
    assert len(result) == 2


def test_last_observation_is_removed():
    df = pd.DataFrame(
        {"Return": [0.01, -0.02, 0.03]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    result = add_direction_target(df)

    assert result.index.tolist() == list(
        pd.date_range("2026-01-01", periods=2, freq="D")
    )


def test_direction_target_is_binary_integer():
    df = pd.DataFrame(
        {"Return": [0.01, -0.02, 0.03, 0.00]},
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )

    result = add_direction_target(df)

    assert set(result["Direction"].unique()).issubset({0, 1})
    assert str(result["Direction"].dtype) == "int32"


def test_zero_future_return_is_classified_as_down():
    df = pd.DataFrame(
        {"Return": [0.01, 0.00]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )

    result = add_direction_target(df)

    assert result["Direction"].tolist() == [0]


def test_unsorted_dataframe_is_rejected():
    df = pd.DataFrame(
        {"Return": [0.01, -0.02, 0.03]},
        index=pd.to_datetime(
            ["2026-01-03", "2026-01-01", "2026-01-02"]
        ),
    )

    with pytest.raises(ValueError, match="chronologically"):
        add_direction_target(df)

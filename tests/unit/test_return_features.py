import numpy as np
import pandas as pd
import pytest

from src.model.return_features import add_return_features


def make_price_dataframe(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0 + i for i in range(n)]},
        index=pd.date_range(
            "2026-01-01",
            periods=n,
            freq="D",
        ),
    )


def test_daily_return_is_calculated_correctly():
    prices = [100.0] * 10 + [110.0, 99.0]

    df = pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(
            "2026-01-01",
            periods=len(prices),
            freq="D",
        ),
    )

    result = add_return_features(
        df,
        price_column="Close",
    )

    assert result["Return"].iloc[0] == pytest.approx(0.10)
    assert result["Return"].iloc[1] == pytest.approx(-0.10)


def test_five_day_return_is_calculated_correctly():
    prices = [100.0] * 10 + [110.0, 110.0]

    df = pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(
            "2026-01-01",
            periods=len(prices),
            freq="D",
        ),
    )

    result = add_return_features(
        df,
        price_column="Close",
    )

    # First valid row is 2026-01-11.
    # Close[2026-01-06] = 100
    # Close[2026-01-11] = 110
    # Therefore the 5-period return is +10%.
    assert result["Return_5D"].iloc[0] == pytest.approx(0.10)


def test_ten_day_return_is_calculated_correctly():
    prices = [100.0] * 10 + [120.0]

    df = pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(
            "2026-01-01",
            periods=len(prices),
            freq="D",
        ),
    )

    result = add_return_features(
        df,
        price_column="Close",
    )

    assert result["Return_10D"].iloc[0] == pytest.approx(0.20)


def test_initial_rows_without_history_are_removed():
    df = make_price_dataframe(12)

    result = add_return_features(
        df,
        price_column="Close",
    )

    assert len(result) == 2
    assert result.index[0] == pd.Timestamp("2026-01-11")


def test_unsorted_dataframe_is_rejected():
    df = make_price_dataframe(12)

    df = df.iloc[
        [2, 0, 1] + list(range(3, 12))
    ]

    with pytest.raises(
        ValueError,
        match="chronologically",
    ):
        add_return_features(
            df,
            price_column="Close",
        )


def test_non_finite_price_is_rejected():
    df = make_price_dataframe(12)

    df.loc[
        pd.Timestamp("2026-01-06"),
        "Close",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        add_return_features(
            df,
            price_column="Close",
        )


def test_missing_price_column_is_rejected():
    df = make_price_dataframe(12)

    df = df.rename(
        columns={"Close": "Price"}
    )

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        add_return_features(
            df,
            price_column="Close",
        )

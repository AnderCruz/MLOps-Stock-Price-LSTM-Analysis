from __future__ import annotations

import pandas as pd


def validate_market_data(
    df: pd.DataFrame,
    ticker: str,
    min_records: int = 100,
) -> None:
    """
    Validate the market-data dataset before feature engineering.

    Raises:
        ValueError: if a data-quality rule is violated.
    """

    ticker = ticker.strip().upper()

    if df is None or df.empty:
        raise ValueError("Market dataset is empty.")

    if len(df) < min_records:
        raise ValueError(
            f"Insufficient market records: {len(df)} "
            f"(minimum required: {min_records})."
        )

    expected_columns = [
        f"{ticker}_Close",
        f"{ticker}_Volume",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Market-data index must be a DatetimeIndex."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Market dataset contains duplicate dates."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Market dataset is not sorted chronologically."
        )

    if df[expected_columns].isna().any().any():
        raise ValueError(
            "Market dataset contains missing values."
        )

    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"

    if (df[close_col] <= 0).any():
        raise ValueError(
            f"{close_col} contains non-positive prices."
        )

    if (df[volume_col] < 0).any():
        raise ValueError(
            f"{volume_col} contains negative volume."
        )

    print("Market data validation: PASS")
    print(f"  Ticker: {ticker}")
    print(f"  Records: {len(df)}")
    print(
        f"  Period: "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    print(f"  Columns: {df.columns.tolist()}")

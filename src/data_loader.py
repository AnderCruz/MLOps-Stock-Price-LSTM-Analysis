from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataMetadata:
    """Metadata describing the market dataset actually retrieved."""

    ticker: str
    requested_start_date: str
    actual_start_date: str
    actual_end_date: str
    n_records: int

    def to_dict(self) -> dict:
        """Return metadata as a serializable dictionary."""
        return asdict(self)


def download_market_data(
    ticker: str,
    start_date: str,
) -> Tuple[pd.DataFrame, MarketDataMetadata]:
    """
    Download historical market data for a ticker.

    Returns:
        A tuple containing:
            - DataFrame with {TICKER}_Close and {TICKER}_Volume
            - MarketDataMetadata describing the retrieved dataset
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    print(f"Downloading market data for {ticker}...")

    df = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for ticker '{ticker}'."
        )

    # yfinance may return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            column[0]
            for column in df.columns
        ]

    df.columns = [str(column).title() for column in df.columns]

    required_columns = ["Close", "Volume"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required market-data columns: "
            f"{missing_columns}"
        )

    df = df[required_columns].copy()

    df = df.rename(
        columns={
            "Close": f"{ticker}_Close",
            "Volume": f"{ticker}_Volume",
        }
    )

    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()

    df = df.ffill()
    df = df.dropna()

    if df.empty:
        raise ValueError(
            f"Market data for ticker '{ticker}' "
            "contains no valid records after cleaning."
        )

    metadata = MarketDataMetadata(
        ticker=ticker,
        requested_start_date=str(start_date),
        actual_start_date=df.index.min().date().isoformat(),
        actual_end_date=df.index.max().date().isoformat(),
        n_records=len(df),
    )

    print(
        f"Downloaded {metadata.n_records} records "
        f"for {metadata.ticker} "
        f"from {metadata.actual_start_date} "
        f"to {metadata.actual_end_date}."
    )

    return df, metadata
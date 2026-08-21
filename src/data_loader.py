from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf


@dataclass
class MarketDataMetadata:
    """Metadata describing a downloaded market dataset."""

    ticker: str
    requested_start_date: str
    actual_start_date: str
    actual_end_date: str
    n_records: int


def download_market_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, MarketDataMetadata]:
    """
    Download and normalize historical market data.

    Yahoo Finance is queried with an explicit end date.
    This avoids relying on yfinance's implicit current-date
    handling, which can intermittently return empty data.

    Output schema:

        <TICKER>_Close
        <TICKER>_Volume
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    if not start_date:
        raise ValueError(
            "Start date cannot be empty."
        )

    # ==================================================
    # 1. RESOLVE END DATE
    # ==================================================

    if end_date is None:

        # Yahoo Finance uses an EXCLUSIVE end date.
        # Therefore today's date is sufficient to request
        # all completed market sessions up to yesterday.
        end_date = (
            date.today()
            .isoformat()
        )

    print(
        f"Downloading market data for {ticker}..."
    )

    print(
        f"Date range: "
        f"{start_date} → {end_date}"
    )

    # ==================================================
    # 2. DOWNLOAD
    # ==================================================

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        actions=False,
        threads=False,
    )

    if df is None or df.empty:

        raise ValueError(
            f"No market data returned for ticker "
            f"'{ticker}' for range "
            f"{start_date} → {end_date}."
        )

    # ==================================================
    # 3. HANDLE MULTIINDEX COLUMNS
    # ==================================================

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = [
            column[0]
            for column in df.columns
        ]

    # ==================================================
    # 4. NORMALIZE COLUMN NAMES
    # ==================================================

    df.columns = [
        str(column)
        .strip()
        .title()
        for column in df.columns
    ]

    required_columns = [
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Yahoo Finance data is missing required "
            f"columns: {missing_columns}"
        )

    # ==================================================
    # 5. SELECT REQUIRED MODEL COLUMNS
    # ==================================================

    df = df[
        required_columns
    ].copy()

    # ==================================================
    # 6. PROJECT COLUMN NAMES
    # ==================================================

    df = df.rename(
        columns={
            "Close": f"{ticker}_Close",
            "Volume": f"{ticker}_Volume",
        }
    )

    # ==================================================
    # 7. NORMALIZE INDEX
    # ==================================================

    df.index = pd.to_datetime(
        df.index
    ).normalize()

    df = df.sort_index()

    # ==================================================
    # 8. REMOVE DUPLICATE DATES
    # ==================================================

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # ==================================================
    # 9. HANDLE MISSING VALUES
    # ==================================================

    df = df.ffill()

    df = df.dropna()

    if df.empty:

        raise ValueError(
            f"Market data for '{ticker}' became empty "
            "after normalization."
        )

    # ==================================================
    # 10. METADATA
    # ==================================================

    metadata = MarketDataMetadata(
        ticker=ticker,
        requested_start_date=start_date,
        actual_start_date=(
            df.index.min()
            .date()
            .isoformat()
        ),
        actual_end_date=(
            df.index.max()
            .date()
            .isoformat()
        ),
        n_records=len(df),
    )

    # ==================================================
    # 11. SUMMARY
    # ==================================================

    print(
        f"Downloaded {metadata.n_records} "
        f"records for {ticker} "
        f"from {metadata.actual_start_date} "
        f"to {metadata.actual_end_date}."
    )

    print(
        "Columns:",
        list(df.columns),
    )

    return df, metadata

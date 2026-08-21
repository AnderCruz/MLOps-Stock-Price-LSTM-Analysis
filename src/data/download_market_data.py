from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("artifacts/raw_market_data")


def download_market_data(
    ticker: str = "TSLA",
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> Path:
    """
    Download raw market data from Yahoo Finance.

    This module intentionally contains NO TensorFlow
    or model-related imports.

    The downloaded raw data is persisted locally so
    subsequent training runs do not depend on a live
    Yahoo Finance request.
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("RAW MARKET DATA ACQUISITION")
    print("=" * 70)

    print(
        f"Ticker: {ticker}"
    )

    print(
        f"Start:  {start_date}"
    )

    if end_date is not None:
        print(
            f"End:    {end_date}"
        )

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

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
            f"No market data returned for "
            f"{ticker}."
        )

    # --------------------------------------------------
    # Normalize MultiIndex
    # --------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = [
            column[0]
            for column in df.columns
        ]

    # --------------------------------------------------
    # Keep raw Yahoo columns
    # --------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    df = df.sort_index()

    # --------------------------------------------------
    # Persist raw artifact
    # --------------------------------------------------

    output_path = (
        RAW_DATA_DIR
        / f"{ticker}.csv"
    )

    df.to_csv(
        output_path
    )

    print(
        f"\nDownloaded: {len(df)} records"
    )

    print(
        "Start:",
        df.index.min(),
    )

    print(
        "End:",
        df.index.max(),
    )

    print(
        "Raw artifact:",
        output_path,
    )

    print(
        "\nRAW MARKET DATA: PASS"
    )

    return output_path


if __name__ == "__main__":

    download_market_data(
        ticker="TSLA",
        start_date="2023-01-01",
        end_date="2026-08-16",
    )

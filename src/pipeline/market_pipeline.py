from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.runtime_config import build_runtime_config
from src.feature_engineering import build_model_features
from src.validation.data_validator import validate_market_data
from src.sentiment.news_sentiment_pipeline import (
    apply_sentiment_pipeline,
)
from src.pipeline.run_artifacts import create_run_id


RAW_DATA_DIR = Path(
    "artifacts/raw_market_data"
)


@dataclass
class MarketDataMetadata:
    """Metadata describing persisted market data."""

    ticker: str
    requested_start_date: str
    actual_start_date: str
    actual_end_date: str
    n_records: int


@dataclass
class PipelineResult:
    """
    Result of a complete market-data feature pipeline.
    """

    run_id: str

    ticker: str
    sentiment_enabled: bool

    requested_start_date: str
    actual_start_date: str
    actual_end_date: str
    n_records: int

    market_columns: list[str]
    feature_columns: list[str]

    lineage: dict[str, Any]

    market_data: pd.DataFrame
    features: pd.DataFrame

    def metadata(self) -> dict[str, Any]:
        """
        Return serializable pipeline metadata.
        """

        return {
            "run_id": self.run_id,
            "pipeline": "market_pipeline",
            "ticker": self.ticker,
            "sentiment_enabled": (
                self.sentiment_enabled
            ),
            "requested_start_date": (
                self.requested_start_date
            ),
            "actual_start_date": (
                self.actual_start_date
            ),
            "actual_end_date": (
                self.actual_end_date
            ),
            "n_records": self.n_records,
            "market_columns": (
                self.market_columns
            ),
            "feature_columns": (
                self.feature_columns
            ),
            "lineage": self.lineage,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Return pipeline metadata without DataFrames.
        """

        return self.metadata()


def load_market_data_artifact(
    ticker: str,
    start_date: str,
) -> tuple[
    pd.DataFrame,
    MarketDataMetadata,
]:
    """
    Load persisted raw market data.

    Training must not depend on a live Yahoo Finance
    request. Raw market data is acquired separately and
    persisted under artifacts/raw_market_data/.
    """

    ticker = ticker.strip().upper()

    path = (
        RAW_DATA_DIR
        / f"{ticker}.csv"
    )

    print(
        f"Loading raw market artifact: {path}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw market-data artifact not found: "
            f"{path}\n\n"
            f"Run the acquisition step first:\n"
            f"python -m src.data.download_market_data"
        )

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    if df.empty:
        raise ValueError(
            f"Raw market-data artifact is empty: "
            f"{path}"
        )

    # ==================================================
    # NORMALIZE INDEX
    # ==================================================

    df.index = pd.to_datetime(
        df.index
    ).normalize()

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # ==================================================
    # NORMALIZE COLUMNS
    # ==================================================

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = [
            column[0]
            for column in df.columns
        ]

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

    # The raw CSV should normally contain these
    # columns because it was created directly from
    # Yahoo Finance.

    if not all(
        column in df.columns
        for column in required_columns
    ):

        # Some pandas/yfinance CSV combinations can
        # produce ticker-qualified column names.
        # Try to identify them.

        close_candidates = [
            column
            for column in df.columns
            if str(column).lower()
            == "close"
            or str(column).lower()
            .endswith("_close")
        ]

        volume_candidates = [
            column
            for column in df.columns
            if str(column).lower()
            == "volume"
            or str(column).lower()
            .endswith("_volume")
        ]

        if (
            close_candidates
            and volume_candidates
        ):

            df = df[
                [
                    close_candidates[0],
                    volume_candidates[0],
                ]
            ].copy()

            df.columns = [
                "Close",
                "Volume",
            ]

        else:

            raise ValueError(
                "Raw market artifact is missing "
                "Close and/or Volume columns.\n"
                f"Columns found: {list(df.columns)}"
            )

    else:

        df = df[
            required_columns
        ].copy()

    # ==================================================
    # PROJECT SCHEMA
    # ==================================================

    df = df.rename(
        columns={
            "Close": f"{ticker}_Close",
            "Volume": f"{ticker}_Volume",
        }
    )

    # ==================================================
    # CLEAN DATA
    # ==================================================

    df = df.ffill()

    df = df.dropna()

    if df.empty:
        raise ValueError(
            f"Market data became empty after "
            f"normalization: {path}"
        )

    # ==================================================
    # START-DATE FILTER
    # ==================================================

    requested_start = pd.Timestamp(
        start_date
    )

    df = df[
        df.index >= requested_start
    ]

    if df.empty:
        raise ValueError(
            f"No market records exist on or after "
            f"{start_date}."
        )

    # ==================================================
    # METADATA
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

    print(
        f"Loaded {metadata.n_records} "
        f"records for {ticker} "
        f"from {metadata.actual_start_date} "
        f"to {metadata.actual_end_date}."
    )

    print(
        "Market columns:",
        list(df.columns),
    )

    return df, metadata


def validate_feature_data(
    df: pd.DataFrame,
    ticker: str,
    sentiment_enabled: bool,
) -> None:
    """
    Validate the final feature dataset.
    """

    if df is None or df.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    ticker = ticker.strip().upper()

    required_columns = [
        f"{ticker}_Close",
        f"{ticker}_Volume",
    ]

    if sentiment_enabled:
        required_columns.append(
            "News_Sentiment"
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature dataset is missing required "
            f"columns: {missing_columns}"
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Feature dataset contains duplicate dates."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Feature dataset is not sorted chronologically."
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Feature dataset contains missing values."
        )

    if sentiment_enabled:

        sentiment = df[
            "News_Sentiment"
        ]

        if not sentiment.between(
            -1,
            1,
        ).all():

            raise ValueError(
                "News_Sentiment contains values "
                "outside [-1, 1]."
            )

    print(
        "Feature data validation: PASS"
    )


def run_market_pipeline(
    ticker: str | None = None,
    sentiment_enabled: bool | None = None,
) -> PipelineResult:
    """
    Execute the complete market-data feature pipeline.

    Pipeline stages:

    1. Runtime configuration
    2. Raw market-data artifact loading
    3. Market-data validation
    4. Sentiment enrichment
    5. Feature engineering
    6. Feature validation
    7. Lineage construction
    8. Pipeline result

    IMPORTANT:

    This function does NOT contact Yahoo Finance.

    Raw market data must be acquired separately using:

        python -m src.data.download_market_data
    """

    # ==================================================
    # RUN IDENTIFICATION
    # ==================================================

    run_id = create_run_id()

    # ==================================================
    # RUNTIME CONFIGURATION
    # ==================================================

    config = build_runtime_config(
        ticker=ticker,
        sentiment_enabled=sentiment_enabled,
    )

    runtime_ticker = (
        config["data"]["ticker"]
    )

    runtime_sentiment = (
        config["features"]["sentiment"]["enabled"]
    )

    start_date = (
        config["data"]["start_date"]
    )

    print("\n" + "=" * 70)
    print("MARKET DATA PIPELINE")
    print("=" * 70)

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Ticker: {runtime_ticker}"
    )

    print(
        f"Start date: {start_date}"
    )

    print(
        f"Sentiment: {runtime_sentiment}"
    )

    # ==================================================
    # 1. LOAD PERSISTED MARKET DATA
    # ==================================================

    (
        df_market,
        market_metadata,
    ) = load_market_data_artifact(
        ticker=runtime_ticker,
        start_date=start_date,
    )

    # ==================================================
    # 2. MARKET DATA VALIDATION
    # ==================================================

    validate_market_data(
        df=df_market,
        ticker=runtime_ticker,
    )

    print(
        "Market data validation: PASS"
    )

    print(
        "  Ticker:",
        market_metadata.ticker,
    )

    print(
        "  Records:",
        market_metadata.n_records,
    )

    print(
        "  Period:",
        f"{market_metadata.actual_start_date}"
        f" → "
        f"{market_metadata.actual_end_date}",
    )

    print(
        "  Columns:",
        list(df_market.columns),
    )

    # ==================================================
    # 3. SENTIMENT ENRICHMENT
    # ==================================================

    df_enriched = (
        df_market.copy()
    )

    sentiment_lineage = {
        "enabled": False,
        "source": "disabled",
        "headlines_processed": 0,
        "real_news_days": 0,
        "synthetic_days": 0,
        "fallback_used": False,
        "real_sentiment_coverage_percent": 0.0,
    }

    if runtime_sentiment:

        (
            df_enriched,
            sentiment_lineage,
        ) = apply_sentiment_pipeline(
            df_full=df_enriched,
            ticker=runtime_ticker,
            use_sentiment=True,
        )

    # ==================================================
    # 4. FEATURE ENGINEERING
    # ==================================================

    features = build_model_features(
        df=df_enriched,
        ticker=runtime_ticker,
        sentiment_enabled=runtime_sentiment,
    )

    # ==================================================
    # 5. FEATURE VALIDATION
    # ==================================================

    validate_feature_data(
        df=features,
        ticker=runtime_ticker,
        sentiment_enabled=runtime_sentiment,
    )

    # ==================================================
    # 6. LINEAGE
    # ==================================================

    lineage = {
        "market_data": {
            "source": (
                "persisted_raw_artifact"
            ),
            "artifact": str(
                RAW_DATA_DIR
                / f"{runtime_ticker}.csv"
            ),
            "ticker": runtime_ticker,
            "requested_start_date": (
                market_metadata
                .requested_start_date
            ),
            "actual_start_date": (
                market_metadata
                .actual_start_date
            ),
            "actual_end_date": (
                market_metadata
                .actual_end_date
            ),
            "n_records": (
                market_metadata.n_records
            ),
        },

        "sentiment": sentiment_lineage,

        "feature_engineering": {
            "input_columns": (
                list(df_enriched.columns)
            ),
            "output_columns": (
                list(features.columns)
            ),
        },
    }

    # ==================================================
    # 7. PIPELINE SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Ticker: {runtime_ticker}"
    )

    print(
        f"Period: "
        f"{market_metadata.actual_start_date}"
        f" → "
        f"{market_metadata.actual_end_date}"
    )

    print(
        f"Records: "
        f"{market_metadata.n_records}"
    )

    print(
        f"Sentiment: "
        f"{runtime_sentiment}"
    )

    print(
        f"Market columns: "
        f"{list(df_market.columns)}"
    )

    print(
        f"Feature columns: "
        f"{list(features.columns)}"
    )

    print("\nLineage stages:")

    for stage in lineage:
        print(
            f"  ✓ {stage}"
        )

    # ==================================================
    # 8. RETURN
    # ==================================================

    return PipelineResult(
        run_id=run_id,

        ticker=runtime_ticker,

        sentiment_enabled=(
            runtime_sentiment
        ),

        requested_start_date=(
            market_metadata
            .requested_start_date
        ),

        actual_start_date=(
            market_metadata
            .actual_start_date
        ),

        actual_end_date=(
            market_metadata
            .actual_end_date
        ),

        n_records=(
            market_metadata.n_records
        ),

        market_columns=(
            df_market.columns.tolist()
        ),

        feature_columns=(
            features.columns.tolist()
        ),

        lineage=lineage,

        market_data=df_market,

        features=features,
    )

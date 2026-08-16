from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config.runtime_config import build_runtime_config
from src.data_loader import download_market_data
from src.feature_engineering import build_model_features
from src.validation.data_validator import validate_market_data
from src.sentiment.news_sentiment_pipeline import apply_sentiment_pipeline


@dataclass
class PipelineResult:
    """Result of a complete market-data feature pipeline."""

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
        """Return serializable pipeline metadata."""
        return {
            "ticker": self.ticker,
            "sentiment_enabled": self.sentiment_enabled,
            "requested_start_date": self.requested_start_date,
            "actual_start_date": self.actual_start_date,
            "actual_end_date": self.actual_end_date,
            "n_records": self.n_records,
            "market_columns": self.market_columns,
            "feature_columns": self.feature_columns,
            "lineage": self.lineage,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return pipeline metadata without DataFrames."""
        return self.metadata()


def validate_feature_data(
    df: pd.DataFrame,
    ticker: str,
    sentiment_enabled: bool,
) -> None:
    """
    Validate the final feature dataset before downstream ML processing.
    """

    if df is None or df.empty:
        raise ValueError("Feature dataset is empty.")

    ticker = ticker.strip().upper()

    required_columns = [
        f"{ticker}_Close",
        f"{ticker}_Volume",
    ]

    if sentiment_enabled:
        required_columns.append("News_Sentiment")

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Feature dataset is missing required columns: "
            f"{missing_columns}"
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
        sentiment = df["News_Sentiment"]

        if not sentiment.between(-1, 1).all():
            raise ValueError(
                "News_Sentiment contains values outside [-1, 1]."
            )

    print("Feature data validation: PASS")


def run_market_pipeline(
    ticker: str | None = None,
    sentiment_enabled: bool | None = None,
) -> PipelineResult:
    """
    Execute the complete market-data feature pipeline.

    Runtime overrides are isolated from the base configuration.
    """

    # --------------------------------------------------
    # 1. Runtime configuration
    # --------------------------------------------------

    config = build_runtime_config(
        ticker=ticker,
        sentiment_enabled=sentiment_enabled,
    )

    # Unique identifier for this pipeline execution.
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    runtime_ticker = config["data"]["ticker"]
    runtime_sentiment = config["features"]["sentiment"]["enabled"]
    start_date = config["data"]["start_date"]

    print("\n" + "=" * 70)
    print("MARKET DATA PIPELINE")
    print("=" * 70)

    print(f"Run ID: {run_id}")
    print(f"Ticker: {runtime_ticker}")
    print(f"Start date: {start_date}")
    print(f"Sentiment: {runtime_sentiment}")

    # --------------------------------------------------
    # 2. Market data ingestion
    # --------------------------------------------------

    df_market, metadata = download_market_data(
        ticker=runtime_ticker,
        start_date=start_date,
    )

    # --------------------------------------------------
    # 3. Market-data validation
    # --------------------------------------------------

    validate_market_data(
        df=df_market,
        ticker=runtime_ticker,
    )

    # --------------------------------------------------
    # 4. Optional sentiment enrichment + lineage
    # --------------------------------------------------

    df_enriched = df_market.copy()

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
        df_enriched, sentiment_lineage = apply_sentiment_pipeline(
            df_full=df_enriched,
            ticker=runtime_ticker,
            use_sentiment=True,
        )

    # --------------------------------------------------
    # 5. Feature engineering
    # --------------------------------------------------

    features = build_model_features(
        df=df_enriched,
        ticker=runtime_ticker,
        sentiment_enabled=runtime_sentiment,
    )

    # --------------------------------------------------
    # 6. Feature validation
    # --------------------------------------------------

    validate_feature_data(
        df=features,
        ticker=runtime_ticker,
        sentiment_enabled=runtime_sentiment,
    )

    # --------------------------------------------------
    # 7. Build complete data lineage
    # --------------------------------------------------

    lineage = {
        "run": {
            "run_id": run_id,
        },
        "market_data": {
            "source": "Yahoo Finance",
            "ticker": runtime_ticker,
            "requested_start_date": metadata.requested_start_date,
            "actual_start_date": metadata.actual_start_date,
            "actual_end_date": metadata.actual_end_date,
            "n_records": metadata.n_records,
        },
        "sentiment": sentiment_lineage,
        "feature_engineering": {
            "input_columns": df_enriched.columns.tolist(),
            "output_columns": features.columns.tolist(),
        },
    }

    # --------------------------------------------------
    # 8. Pipeline summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"Run ID: {run_id}")
    print(f"Ticker: {runtime_ticker}")
    print(
        f"Period: "
        f"{metadata.actual_start_date} → "
        f"{metadata.actual_end_date}"
    )
    print(f"Records: {metadata.n_records}")
    print(f"Sentiment: {runtime_sentiment}")
    print(f"Market columns: {list(df_market.columns)}")
    print(f"Feature columns: {list(features.columns)}")

    print("\nSentiment lineage:")
    for key, value in sentiment_lineage.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------
    # 9. Return pipeline result
    # --------------------------------------------------

    return PipelineResult(
        ticker=runtime_ticker,
        sentiment_enabled=runtime_sentiment,
        requested_start_date=metadata.requested_start_date,
        actual_start_date=metadata.actual_start_date,
        actual_end_date=metadata.actual_end_date,
        n_records=metadata.n_records,
        market_columns=df_market.columns.tolist(),
        feature_columns=features.columns.tolist(),
        lineage=lineage,
        market_data=df_market,
        features=features,
    )
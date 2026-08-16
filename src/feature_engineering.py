from __future__ import annotations

import pandas as pd


def build_model_features(
    df: pd.DataFrame,
    ticker: str,
    sentiment_enabled: bool,
) -> pd.DataFrame:
    """
    Build the feature set consumed by the LSTM model.

    Baseline feature contract:

        {TICKER}_Close
        {TICKER}_Volume
        News_Sentiment (optional)

    The feature order is deterministic.
    """

    ticker = ticker.strip().upper()

    close_column = f"{ticker}_Close"
    volume_column = f"{ticker}_Volume"

    required_columns = [
        close_column,
        volume_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    features = df[required_columns].copy()

    if sentiment_enabled:
        if "News_Sentiment" not in df.columns:
            raise ValueError(
                "Sentiment is enabled, but "
                "'News_Sentiment' is missing."
            )

        features["News_Sentiment"] = df["News_Sentiment"]

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    features = features.ffill().dropna()

    return features
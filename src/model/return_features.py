from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(
    df: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """
    Add historical return-based features.

    All features are calculated using information available
    at or before the prediction time.

    Features
    --------
    Return:
        Daily percentage return.

    Return_5D:
        Five-trading-day cumulative return.

    Return_10D:
        Ten-trading-day cumulative return.
    """

    if df is None or df.empty:
        raise ValueError(
            "DataFrame cannot be empty."
        )

    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' not found."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    if not np.isfinite(
        df[price_column].to_numpy()
    ).all():
        raise ValueError(
            "Price column contains non-finite values."
        )

    result = df.copy()

    # --------------------------------------------------
    # 1. DAILY RETURN
    # --------------------------------------------------

    result["Return"] = (
        result[price_column]
        .pct_change()
    )

    # --------------------------------------------------
    # 2. 5-DAY RETURN
    # --------------------------------------------------

    result["Return_5D"] = (
        result[price_column]
        .pct_change(periods=5)
    )

    # --------------------------------------------------
    # 3. 10-DAY RETURN
    # --------------------------------------------------

    result["Return_10D"] = (
        result[price_column]
        .pct_change(periods=10)
    )

    # --------------------------------------------------
    # REMOVE INITIAL ROWS WITHOUT HISTORY
    # --------------------------------------------------

    result = result.dropna()

    if result.empty:
        raise ValueError(
            "No observations remain after creating "
            "return features."
        )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    feature_columns = [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    if not np.isfinite(
        result[feature_columns].to_numpy()
    ).all():
        raise ValueError(
            "Return features contain non-finite values."
        )

    if not result.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    return result

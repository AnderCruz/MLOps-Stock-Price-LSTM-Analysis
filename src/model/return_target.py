from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_target(
    df: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """
    Add next-day return target.

    return[t] = Close[t] / Close[t-1] - 1

    The target represents the return observed at day t
    relative to the previous trading day.
    """

    if df is None or df.empty:
        raise ValueError(
            "DataFrame cannot be empty."
        )

    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' "
            "not found."
        )

    result = df.copy()

    result["Return"] = (
        result[price_column]
        .pct_change()
    )

    result = result.dropna()

    if result.empty:
        raise ValueError(
            "No observations remain after "
            "calculating returns."
        )

    if not np.isfinite(
        result["Return"]
    ).all():
        raise ValueError(
            "Return contains non-finite values."
        )

    return result
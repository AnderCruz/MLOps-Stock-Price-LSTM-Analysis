from __future__ import annotations

import numpy as np
import pandas as pd


def add_direction_target(
    df: pd.DataFrame,
    return_column: str = "Return",
    target_column: str = "Direction",
) -> pd.DataFrame:
    """
    Add a next-day directional classification target.

    Direction[t] represents the sign of the return observed
    at t+1.

        Direction[t] = 1 if Return[t+1] > 0
        Direction[t] = 0 if Return[t+1] <= 0

    The target is shifted backward so that all features at
    observation t are available before the target event at t+1.
    """

    if df is None or df.empty:
        raise ValueError(
            "DataFrame cannot be empty."
        )

    if return_column not in df.columns:
        raise ValueError(
            f"Return column '{return_column}' "
            "not found."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically."
        )

    result = df.copy()

    future_return = (
        result[return_column]
        .shift(-1)
    )

    result[target_column] = (
        future_return
        .gt(0)
        .astype(float)
    )

    # Last observation has no future return.
    result = result.iloc[:-1].copy()

    if result.empty:
        raise ValueError(
            "No observations remain after "
            "creating direction target."
        )

    result[target_column] = (
        result[target_column]
        .astype(np.int32)
    )

    if not np.isfinite(
        result[return_column]
    ).all():
        raise ValueError(
            "Return contains non-finite values."
        )

    if not result[target_column].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Direction target must contain only "
            "0 and 1."
        )

    return result

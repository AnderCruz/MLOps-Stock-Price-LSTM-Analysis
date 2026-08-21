from __future__ import annotations

import hashlib

import pandas as pd


def fingerprint_dataframe(
    df: pd.DataFrame,
) -> str:
    """
    Generate a deterministic SHA-256 fingerprint
    for a DataFrame.

    The fingerprint changes when the data, columns,
    index, or ordering changes.
    """

    if df is None:
        raise ValueError(
            "Cannot fingerprint a None DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot fingerprint an empty DataFrame."
        )

    normalized = df.copy()

    # Ensure deterministic column ordering.
    normalized = normalized.reindex(
        sorted(normalized.columns),
        axis=1,
    )

    # Ensure deterministic index representation.
    normalized.index = normalized.index.astype(str)

    payload = pd.util.hash_pandas_object(
        normalized,
        index=True,
    ).values.tobytes()

    schema = "|".join(
        f"{column}:{normalized[column].dtype}"
        for column in normalized.columns
    ).encode("utf-8")

    combined = schema + b"|" + payload

    return hashlib.sha256(
        combined
    ).hexdigest()

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint_config(
    config: dict[str, Any],
) -> str:
    """
    Generate a deterministic SHA-256 fingerprint
    for a runtime configuration.
    """

    if not isinstance(config, dict):
        raise TypeError(
            "Configuration must be a dictionary."
        )

    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

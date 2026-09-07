from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.api.prediction_api import create_prediction_app
from src.model.model_registry import ModelRegistry
from src.model.production_inference import ProductionModelInference


DEFAULT_REGISTRY_PATH = (
    Path("artifacts")
    / "registry"
    / "registry.json"
)

DEFAULT_MODEL_NAME = "stock-direction-lstm"
DEFAULT_SEQUENCE_LENGTH = 60

DEFAULT_FEATURE_COLUMNS = [
    "Return",
    "Return_5D",
    "Return_10D",
]


def create_production_app(
    *,
    registry: ModelRegistry | None = None,
    inference: Any | None = None,
):
    """Create the FastAPI application backed by the production model."""

    if inference is None:

        if registry is None:

            registry_path = Path(
                os.getenv(
                    "MODEL_REGISTRY_PATH",
                    str(DEFAULT_REGISTRY_PATH),
                )
            )

            registry = ModelRegistry(
                registry_path=registry_path
            )

        model_name = os.getenv(
            "MODEL_NAME",
            DEFAULT_MODEL_NAME,
        )

        sequence_length = int(
            os.getenv(
                "MODEL_SEQUENCE_LENGTH",
                str(DEFAULT_SEQUENCE_LENGTH),
            )
        )

        inference = ProductionModelInference(
            registry=registry,
            model_name=model_name,
            sequence_length=sequence_length,
            feature_columns=DEFAULT_FEATURE_COLUMNS,
        )

    return create_prediction_app(
        inference=inference
    )



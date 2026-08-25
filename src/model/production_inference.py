from __future__ import annotations

from pathlib import Path
from typing import Any

from src.model.artifact_manager import (
    load_model_artifact,
    load_scaler_artifact,
)

# Local aliases keep the artifact-loading boundary easy to mock.
load_model = load_model_artifact
load_scaler = load_scaler_artifact
from src.model.model_inference import (
    ModelInference,
    ModelInferenceError,
)
from src.model.model_registry import (
    ModelNotFoundError,
    ModelRegistry,
)


class ProductionModelInference:
    """
    Inference service backed by the model currently
    registered as production.
    """

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        model_name: str,
        sequence_length: int,
        feature_columns: list[str],
        threshold: float = 0.50,
    ) -> None:

        self.registry = registry
        self.model_name = model_name

        model_record = self._get_production_model()

        artifact_path = Path(
            model_record["artifact_path"]
        )

        try:
            model = load_model(
                artifact_path
            )
            scaler = load_scaler(
                artifact_path
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Failed to load production model "
                "artifacts."
            ) from exc

        self._metadata = {
            "model_name": model_record[
                "model_name"
            ],
            "model_version": model_record[
                "model_version"
            ],
            "run_id": model_record[
                "run_id"
            ],
            "artifact_path": model_record[
                "artifact_path"
            ],
            "status": model_record[
                "status"
            ],
        }

        self._inference = ModelInference(
            model=model,
            scaler=scaler,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            threshold=threshold,
        )

    def _get_production_model(
        self,
    ) -> dict[str, Any]:

        models = self.registry.list_models(
            model_name=self.model_name
        )

        production_models = [
            model
            for model in models
            if model.get("status")
            == "production"
        ]

        if not production_models:
            raise ModelInferenceError(
                "No production model is registered "
                f"for '{self.model_name}'."
            )

        if len(production_models) > 1:
            raise ModelInferenceError(
                "Multiple production models are "
                f"registered for '{self.model_name}'."
            )

        return production_models[0]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata of the production model."""

        return dict(self._metadata)

    def predict(
        self,
        features,
    ) -> dict[str, Any]:
        """
        Generate a prediction using the current
        production model.
        """

        result = self._inference.predict(
            features
        )

        return {
            **result,
            "model_name": self._metadata[
                "model_name"
            ],
            "model_version": self._metadata[
                "model_version"
            ],
        }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.model.model_validation import (
    ValidationResult,
)


VALID_STATUSES = {
    "candidate",
    "validated",
    "production",
    "rejected",
}


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""


class ModelAlreadyRegisteredError(
    ModelRegistryError
):
    """Raised when a model version is already registered."""


class ModelNotFoundError(
    ModelRegistryError
):
    """Raised when a registered model cannot be found."""


class InvalidPromotionError(
    ModelRegistryError
):
    """Raised when a model promotion is invalid."""


class ModelRegistry:
    """
    Lightweight filesystem-backed model registry.

    The registry stores model metadata and lifecycle status
    in a JSON file.
    """

    def __init__(
        self,
        registry_path: str | Path,
    ) -> None:

        self.registry_path = Path(
            registry_path
        )

        self._data = self._load()

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def _load(self) -> dict[str, Any]:

        if not self.registry_path.exists():
            return {
                "models": {}
            }

        with self.registry_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ModelRegistryError(
                "Registry file must contain a JSON object."
            )

        data.setdefault(
            "models",
            {}
        )

        return data

    def _save(self) -> None:

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            self.registry_path.with_suffix(
                ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self._data,
                file,
                indent=2,
            )

        temporary_path.replace(
            self.registry_path
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _get_model_versions(
        self,
        model_name: str,
    ) -> dict[str, Any]:

        models = self._data.setdefault(
            "models",
            {}
        )

        return models.setdefault(
            model_name,
            {}
        )

    def _find_model(
        self,
        model_name: str,
        model_version: str,
    ) -> dict[str, Any]:

        models = self._data.get(
            "models",
            {}
        )

        model_versions = models.get(
            model_name
        )

        if not model_versions:
            raise ModelNotFoundError(
                f"Model '{model_name}' "
                f"was not found."
            )

        record = model_versions.get(
            model_version
        )

        if record is None:
            raise ModelNotFoundError(
                f"Model '{model_name}' "
                f"version '{model_version}' "
                f"was not found."
            )

        return record

    # --------------------------------------------------
    # Registration
    # --------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        model_version: str,
        run_id: str,
        artifact_path: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:

        model_versions = (
            self._get_model_versions(
                model_name
            )
        )

        if model_version in model_versions:
            raise ModelAlreadyRegisteredError(
                f"Model '{model_name}' "
                f"version '{model_version}' "
                f"is already registered."
            )

        record = {
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "status": "candidate",
        }

        model_versions[
            model_version
        ] = record

        self._save()

        return record.copy()

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    def get_model(
        self,
        *,
        model_name: str,
        model_version: str,
    ) -> dict[str, Any]:

        record = self._find_model(
            model_name=model_name,
            model_version=model_version,
        )

        return record.copy()

    def list_models(
        self,
        *,
        model_name: str,
    ) -> list[dict[str, Any]]:

        model_versions = (
            self._data
            .get("models", {})
            .get(model_name, {})
        )

        return [
            record.copy()
            for record in model_versions.values()
        ]

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def record_validation(
        self,
        *,
        model_name: str,
        model_version: str,
        validation: ValidationResult,
    ) -> dict[str, Any]:
        """
        Persist the validation result for a model candidate.

        A successful validation moves the model from
        candidate to validated.

        A failed validation moves the model from
        candidate to rejected.
        """

        record = self._find_model(
            model_name=model_name,
            model_version=model_version,
        )

        current_status = record[
            "status"
        ]

        if current_status != "candidate":
            raise InvalidPromotionError(
                "Model validation is only "
                "allowed for candidate models. "
                f"Current status: "
                f"'{current_status}'."
            )

        record["validation"] = (
            validation.to_dict()
        )

        if validation.passed:
            record["status"] = (
                "validated"
            )
        else:
            record["status"] = (
                "rejected"
            )

        self._save()

        return record.copy()

    # --------------------------------------------------
    # Promotion
    # --------------------------------------------------

    def promote_model(
        self,
        *,
        model_name: str,
        model_version: str,
        target_status: str,
    ) -> dict[str, Any]:

        if target_status not in VALID_STATUSES:
            raise InvalidPromotionError(
                f"Invalid target status: "
                f"'{target_status}'."
            )

        record = self._find_model(
            model_name=model_name,
            model_version=model_version,
        )

        current_status = record[
            "status"
        ]

        # ----------------------------------------------
        # candidate -> validated
        # ----------------------------------------------

        if (
            current_status == "candidate"
            and target_status == "validated"
        ):
            record["status"] = (
                "validated"
            )

            self._save()

            return record.copy()

        # ----------------------------------------------
        # validated -> production
        # ----------------------------------------------

        if (
            current_status == "validated"
            and target_status == "production"
        ):

            model_versions = (
                self._get_model_versions(
                    model_name
                )
            )

            for (
                version,
                other_record,
            ) in model_versions.items():

                if (
                    version != model_version
                    and other_record.get(
                        "status"
                    )
                    == "production"
                ):
                    other_record[
                        "status"
                    ] = "validated"

            record["status"] = (
                "production"
            )

            self._save()

            return record.copy()

        # ----------------------------------------------
        # candidate -> rejected
        # ----------------------------------------------

        if (
            current_status == "candidate"
            and target_status == "rejected"
        ):
            record["status"] = (
                "rejected"
            )

            self._save()

            return record.copy()

        # ----------------------------------------------
        # validated -> rejected
        # ----------------------------------------------

        if (
            current_status == "validated"
            and target_status == "rejected"
        ):
            record["status"] = (
                "rejected"
            )

            self._save()

            return record.copy()

        # ----------------------------------------------
        # Invalid transition
        # ----------------------------------------------

        raise InvalidPromotionError(
            "Invalid model promotion: "
            f"'{current_status}' -> "
            f"'{target_status}'."
        )

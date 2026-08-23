from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_STATUSES = {
    "created",
    "started",
    "completed",
    "failed",
}


@dataclass
class ExperimentRecord:
    """
    Structured record describing a machine-learning experiment.

    The record stores experiment identity, dataset configuration,
    training configuration, evaluation metrics and lifecycle status.
    """

    run_id: str
    experiment_name: str
    model_name: str
    model_version: str

    ticker: str | None = None
    features: list[str] = field(default_factory=list)
    target: str | None = None

    sequence_length: int | None = None
    train_size: int | None = None
    test_size: int | None = None

    training: dict[str, Any] = field(
        default_factory=dict
    )

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "created"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError(
                "run_id must not be empty."
            )

        if not self.experiment_name:
            raise ValueError(
                "experiment_name must not be empty."
            )

        if not self.model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        if not self.model_version:
            raise ValueError(
                "model_version must not be empty."
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Valid statuses are: "
                f"{sorted(VALID_STATUSES)}."
            )

        # Make defensive copies so external mutations
        # do not alter the experiment record.
        self.features = list(self.features)
        self.training = dict(self.training)
        self.metrics = dict(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the experiment record to a serialisable dictionary.
        """

        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "ticker": self.ticker,
            "features": list(self.features),
            "target": self.target,
            "sequence_length": self.sequence_length,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "training": dict(self.training),
            "metrics": dict(self.metrics),
            "status": self.status,
        }

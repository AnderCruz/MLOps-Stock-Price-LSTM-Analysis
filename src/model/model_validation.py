from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MINIMUM_ROC_AUC = 0.55


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of model candidate validation.
    """

    passed: bool
    checks: dict[str, bool]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the validation result into a
        JSON-serialisable dictionary.
        """

        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


def validate_model_candidate(
    *,
    metrics: dict[str, Any],
    majority_baseline: float,
) -> ValidationResult:
    """
    Validate a model candidate against the
    production-readiness policy.

    Validation gates:

    1. Accuracy must beat the majority baseline.
    2. ROC-AUC must meet the minimum threshold.
    """

    accuracy = float(
        metrics["accuracy"]
    )

    roc_auc = float(
        metrics["roc_auc"]
    )

    accuracy_vs_baseline = (
        accuracy > majority_baseline
    )

    minimum_roc_auc = (
        roc_auc >= MINIMUM_ROC_AUC
    )

    checks = {
        "accuracy_vs_baseline":
            accuracy_vs_baseline,
        "minimum_roc_auc":
            minimum_roc_auc,
    }

    reasons: list[str] = []

    if not accuracy_vs_baseline:
        reasons.append(
            "Accuracy does not exceed "
            "the majority baseline."
        )

    if not minimum_roc_auc:
        reasons.append(
            "ROC-AUC is below the minimum "
            "required threshold."
        )

    return ValidationResult(
        passed=(
            accuracy_vs_baseline
            and minimum_roc_auc
        ),
        checks=checks,
        reasons=reasons,
    )

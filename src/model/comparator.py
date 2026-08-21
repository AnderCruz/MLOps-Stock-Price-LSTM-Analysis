from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.model.evaluator import EvaluationResult


@dataclass
class MetricComparison:
    """Comparison between baseline and candidate model."""

    metric: str
    baseline: float
    candidate: float
    delta: float
    improvement_percent: float
    better: bool


@dataclass
class ModelComparison:
    """Complete baseline-versus-candidate comparison."""

    baseline_version: str
    candidate_version: str
    comparisons: list[MetricComparison]

    def to_dict(self) -> dict:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "comparisons": [
                {
                    "metric": item.metric,
                    "baseline": item.baseline,
                    "candidate": item.candidate,
                    "delta": item.delta,
                    "improvement_percent": (
                        item.improvement_percent
                    ),
                    "better": item.better,
                }
                for item in self.comparisons
            ],
        }


def load_baseline_metrics(
    path: str = "baseline/baseline_metrics.json",
) -> dict:
    """Load documented baseline metrics."""

    baseline_path = Path(path)

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found: {path}"
        )

    with baseline_path.open(
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data


def compare_metric(
    metric: str,
    baseline: float,
    candidate: float,
    higher_is_better: bool = False,
) -> MetricComparison:
    """
    Compare one metric.

    For error metrics such as MAE/RMSE/MAPE:
        lower is better.

    For Direction Accuracy:
        higher is better.
    """

    if baseline == 0:
        improvement_percent = 0.0
    elif higher_is_better:
        improvement_percent = (
            (candidate - baseline)
            / abs(baseline)
            * 100.0
        )
    else:
        improvement_percent = (
            (baseline - candidate)
            / abs(baseline)
            * 100.0
        )

    if higher_is_better:
        better = candidate > baseline
    else:
        better = candidate < baseline

    return MetricComparison(
        metric=metric,
        baseline=float(baseline),
        candidate=float(candidate),
        delta=float(candidate - baseline),
        improvement_percent=float(
            improvement_percent
        ),
        better=better,
    )


def compare_against_baseline(
    evaluation: EvaluationResult,
    baseline_path: str = "baseline/baseline_metrics.json",
    candidate_version: str = "refactored-v1",
) -> ModelComparison:
    """
    Compare candidate evaluation metrics against baseline-v0.
    """

    baseline = load_baseline_metrics(
        baseline_path
    )

    baseline_version = baseline[
        "model_version"
    ]

    metrics = baseline[
        "evaluation"
    ]["metrics"]

    comparisons = [
        compare_metric(
            metric="mae",
            baseline=metrics["mae"],
            candidate=evaluation.mae,
            higher_is_better=False,
        ),
        compare_metric(
            metric="rmse",
            baseline=metrics["rmse"],
            candidate=evaluation.rmse,
            higher_is_better=False,
        ),
        compare_metric(
            metric="mape_percent",
            baseline=metrics[
                "mape_percent"
            ],
            candidate=evaluation.mape_percent,
            higher_is_better=False,
        ),
        compare_metric(
            metric="direction_accuracy_percent",
            baseline=metrics[
                "direction_accuracy_percent"
            ],
            candidate=(
                evaluation
                .direction_accuracy_percent
            ),
            higher_is_better=True,
        ),
    ]

    return ModelComparison(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        comparisons=comparisons,
    )

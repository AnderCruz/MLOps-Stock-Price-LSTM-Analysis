from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACT_ROOT = Path("artifacts/runs")


def create_run_id() -> str:
    """Create a UTC run identifier."""

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def create_run_directory(
    run_id: str,
) -> Path:
    """Create and return the artifact directory for a run."""

    run_dir = ARTIFACT_ROOT / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Persist a dictionary as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Persist a DataFrame as CSV."""

    if df is None or df.empty:
        raise ValueError(
            f"Cannot persist empty DataFrame: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=True,
    )


def save_experiment_record(
    experiment: Any,
    run_dir: Path,
) -> Path:
    """
    Persist an ExperimentRecord as experiment.json.

    The experiment object must expose a to_dict() method.
    """

    if experiment is None:
        raise ValueError(
            "experiment cannot be None."
        )

    if not hasattr(
        experiment,
        "to_dict",
    ):
        raise ValueError(
            "experiment must provide a "
            "to_dict() method."
        )

    if run_dir is None:
        raise ValueError(
            "run_dir cannot be None."
        )

    run_dir = Path(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        run_dir
        / "experiment.json"
    )

    save_json(
        experiment.to_dict(),
        path,
    )

    return path


def save_pipeline_run(
    run_id: str,
    metadata: dict[str, Any],
    feature_data: pd.DataFrame | None = None,
    train_data: pd.DataFrame | None = None,
    test_data: pd.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    predictions: pd.DataFrame | None = None,
    experiment: Any | None = None,
) -> Path:
    """
    Persist artifacts associated with a pipeline run.

    Possible artifacts:

        metadata.json
        experiment.json
        feature_data.csv
        train_data.csv
        test_data.csv
        metrics.json
        predictions.csv

    The experiment record is optional to preserve
    backwards compatibility with existing callers.
    """

    run_dir = create_run_directory(
        run_id
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    metadata = dict(metadata)

    metadata["run_id"] = run_id

    save_json(
        metadata,
        run_dir / "metadata.json",
    )

    # --------------------------------------------------
    # Experiment record
    # --------------------------------------------------

    if experiment is not None:

        save_experiment_record(
            experiment=experiment,
            run_dir=run_dir,
        )

    # --------------------------------------------------
    # Complete feature dataset
    # --------------------------------------------------

    if feature_data is not None:

        save_dataframe(
            feature_data,
            run_dir / "feature_data.csv",
        )

    # --------------------------------------------------
    # Training dataset
    # --------------------------------------------------

    if train_data is not None:

        save_dataframe(
            train_data,
            run_dir / "train_data.csv",
        )

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    if test_data is not None:

        save_dataframe(
            test_data,
            run_dir / "test_data.csv",
        )

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    if metrics is not None:

        save_json(
            metrics,
            run_dir / "metrics.json",
        )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    if predictions is not None:

        save_dataframe(
            predictions,
            run_dir / "predictions.csv",
        )

    return run_dir

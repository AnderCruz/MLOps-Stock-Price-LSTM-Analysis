from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import tensorflow as tf


def save_model_artifact(
    model: tf.keras.Model,
    scaler: Any,
    metadata: dict[str, Any],
    model_version: str,
    base_dir: str = "artifacts/models",
) -> Path:
    """
    Persist a trained model and all metadata required to reproduce
    its inference environment.
    """

    model_dir = (
        Path(base_dir)
        / model_version
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Save TensorFlow model
    # --------------------------------------------------

    model_path = (
        model_dir / "model.keras"
    )

    model.save(model_path)

    # --------------------------------------------------
    # 2. Save scaler
    # --------------------------------------------------

    scaler_path = (
        model_dir / "scaler.pkl"
    )

    with scaler_path.open("wb") as f:
        pickle.dump(
            scaler,
            f,
        )

    # --------------------------------------------------
    # 3. Save metadata
    # --------------------------------------------------

    metadata_path = (
        model_dir / "model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("=" * 70)
    print("MODEL ARTIFACT SAVED")
    print("=" * 70)

    print("Directory:", model_dir)
    print("Model:", model_path)
    print("Scaler:", scaler_path)
    print("Metadata:", metadata_path)

    return model_dir


def load_model_artifact(
    model_directory: str | Path,
):
    """
    Load a persisted Keras model from a model artifact directory.
    """

    from tensorflow.keras.models import load_model

    model_directory = Path(
        model_directory
    )

    model_path = (
        model_directory / "model.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return load_model(
        model_path
    )


def load_scaler_artifact(
    model_directory: str | Path,
):
    """
    Load the persisted feature scaler from a model
    artifact directory.
    """

    import pickle

    model_directory = Path(
        model_directory
    )

    scaler_path = (
        model_directory / "scaler.pkl"
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found: {scaler_path}"
        )

    with scaler_path.open(
        "rb"
    ) as file:
        return pickle.load(file)

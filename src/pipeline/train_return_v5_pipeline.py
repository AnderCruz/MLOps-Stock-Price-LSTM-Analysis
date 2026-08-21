from __future__ import annotations

from pathlib import Path

from src.model.artifact_manager import save_model_artifact
from src.model.return_preprocessing import prepare_return_dataset
from src.model.lstm_model import build_lstm_model
from src.model.trainer import train_lstm_model
from src.pipeline.market_pipeline import run_market_pipeline
from src.pipeline.run_artifacts import save_pipeline_run
from src.model.return_target import add_return_target


MODEL_VERSION = "refactored-v5"

SEQUENCE_LENGTH = 60

TEST_RATIO = 0.20

EPOCHS = 60

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.10

PATIENCE = 10


def run_training_pipeline(
    ticker: str = "TSLA",
    sentiment_enabled: bool = True,
):
    """
    Execute the complete reproducible training pipeline.

    Architecture:

        Market Pipeline
              ↓
        Feature Dataset
              ↓
        Train/Test Split
              ↓
        LSTM Training
              ↓
        Model Artifact
              ↓
        Run Artifact
    """

    print("\n" + "=" * 70)
    print("LSTM TRAINING PIPELINE")
    print("=" * 70)

    # ==================================================
    # 1. MARKET / FEATURE PIPELINE
    # ==================================================

    pipeline_result = run_market_pipeline(
        ticker=ticker,
        sentiment_enabled=sentiment_enabled,
    )

    run_id = pipeline_result.run_id

    print(
        f"\nTraining Run ID: {run_id}"
    )

    # ==================================================
    # 2. DATASET PREPARATION
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL DATASET PREPARATION")
    print("=" * 70)

    price_column = (
        f"{pipeline_result.ticker}_Close"
    )

    return_data = add_return_target(
        df=pipeline_result.features,
        price_column=price_column,
    )

    target_column = "Return"

    feature_columns = [
        "Return",
        price_column,
        f"{pipeline_result.ticker}_Volume",
    ]

    return_data = return_data[
        feature_columns
    ]

    prepared = prepare_return_dataset(
        df=return_data,
        target_column=target_column,
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=TEST_RATIO,
    )

    print(
        "Target:",
        prepared.target_column,
    )

    print(
        "Sequence length:",
        prepared.sequence_length,
    )

    print(
        "Features:",
        prepared.n_features,
    )

    print(
        "Training rows:",
        prepared.train_size,
    )

    print(
        "Test rows:",
        prepared.test_size,
    )

    print(
        "X_train:",
        prepared.X_train.shape,
    )

    print(
        "X_test:",
        prepared.X_test.shape,
    )

    # ==================================================
    # 3. SAVE DATASET ARTIFACTS
    # ==================================================

    print("\n" + "=" * 70)
    print("PERSISTING DATASET ARTIFACTS")
    print("=" * 70)

    dataset_metadata = (
        pipeline_result.metadata()
    )

    dataset_metadata.update(
        {
            "status": "training_started",

            "dataset": {
                "target_column": (
                    prepared.target_column
                ),
                "sequence_length": (
                    prepared.sequence_length
                ),
                "n_features": (
                    prepared.n_features
                ),
                "train_size": (
                    prepared.train_size
                ),
                "test_size": (
                    prepared.test_size
                ),
                "X_train_shape": list(
                    prepared.X_train.shape
                ),
                "X_test_shape": list(
                    prepared.X_test.shape
                ),
                "scaler": "StandardScaler",
            },

            "training": {
                "epochs_requested": EPOCHS,
                "batch_size": BATCH_SIZE,
                "validation_split": (
                    VALIDATION_SPLIT
                ),
                "patience": PATIENCE,
            },

            "model_version": MODEL_VERSION,
        }
    )

    run_dir = save_pipeline_run(
        run_id=run_id,
        metadata=dataset_metadata,
        feature_data=pipeline_result.features,
        train_data=prepared.train_data,
        test_data=prepared.test_data,
    )

    print(
        "Run artifact directory:",
        run_dir,
    )

    # ==================================================
    # 4. BUILD MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("LSTM MODEL")
    print("=" * 70)

    model = build_lstm_model(
        sequence_length=prepared.sequence_length,
        n_features=prepared.n_features,
    )

    print(
        "Model:",
        model.name,
    )

    print(
        "Input:",
        model.input_shape,
    )

    print(
        "Output:",
        model.output_shape,
    )

    # ==================================================
    # 5. TRAIN
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    training_result = train_lstm_model(
        model=model,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        patience=PATIENCE,
    )

    # ==================================================
    # 6. MODEL ARTIFACT
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL ARTIFACT")
    print("=" * 70)

    metadata = pipeline_result.metadata()

    metadata.update(
        {
            "status": "success",

            "model_version": MODEL_VERSION,

            "model_name": "stock-return-lstm",

            "framework": "tensorflow",

            "dataset": {
                "target_column": (
                    prepared.target_column
                ),
                "sequence_length": (
                    prepared.sequence_length
                ),
                "n_features": (
                    prepared.n_features
                ),
                "train_size": (
                    prepared.train_size
                ),
                "test_size": (
                    prepared.test_size
                ),
                "X_train_shape": list(
                    prepared.X_train.shape
                ),
                "X_test_shape": list(
                    prepared.X_test.shape
                ),
                "scaler": "StandardScaler",
            },

            "training": {
                "epochs_requested": EPOCHS,
                "epochs_completed": (
                    training_result.epochs_completed
                ),
                "batch_size": BATCH_SIZE,
                "validation_split": (
                    VALIDATION_SPLIT
                ),
                "patience": PATIENCE,
            },
        }
    )

    model_dir = save_model_artifact(
        model=training_result.model,
        scaler=prepared.scaler,
        metadata=metadata,
        model_version=MODEL_VERSION,
    )

    print(
        "Model artifact:",
        model_dir,
    )

    # ==================================================
    # 7. UPDATE RUN METADATA
    # ==================================================

    final_metadata = dict(metadata)

    final_metadata["artifacts"] = {
        "run_directory": str(
            Path(run_dir)
        ),
        "model_directory": str(
            model_dir
        ),
        "test_data": str(
            Path(run_dir)
            / "test_data.csv"
        ),
        "feature_data": str(
            Path(run_dir)
            / "feature_data.csv"
        ),
    }

    final_metadata["status"] = (
        "training_completed"
    )

    save_pipeline_run(
        run_id=run_id,
        metadata=final_metadata,
        feature_data=pipeline_result.features,
        train_data=prepared.train_data,
        test_data=prepared.test_data,
    )


    # ==================================================
    # 8. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Ticker:",
        pipeline_result.ticker,
    )

    print(
        "Model version:",
        MODEL_VERSION,
    )

    print(
        "Training samples:",
        prepared.train_size,
    )

    print(
        "Test samples:",
        prepared.test_size,
    )

    print(
        "Run artifacts:",
        run_dir,
    )

    print(
        "Model artifacts:",
        model_dir,
    )

    return {
        "run_id": run_id,
        "pipeline": pipeline_result,
        "dataset": prepared,
        "training": training_result,
        "model_dir": model_dir,
        "run_dir": run_dir,
    }


if __name__ == "__main__":

    run_training_pipeline(
        ticker="TSLA",
        sentiment_enabled=False,
    )

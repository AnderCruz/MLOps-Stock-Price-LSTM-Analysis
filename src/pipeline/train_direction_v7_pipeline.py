from __future__ import annotations

from pathlib import Path

import numpy as np

from src.model.artifact_manager import (
    save_model_artifact,
)
from src.model.direction_preprocessing import (
    prepare_direction_dataset,
)
from src.model.direction_target import (
    add_direction_target,
)
from src.model.directional_model import (
    build_directional_lstm_model,
)
from src.model.evaluation import (
    evaluate_binary_classifier,
)
from src.model.model_registry import (
    ModelRegistry,
)
from src.model.return_features import (
    add_return_features,
)
from src.model.trainer import (
    train_lstm_model,
)
from src.pipeline.experiment_schema import (
    ExperimentRecord,
)
from src.pipeline.market_pipeline import (
    run_market_pipeline,
)
from src.pipeline.run_artifacts import (
    save_pipeline_run,
)
from src.model.model_registry import (
    ModelRegistry,
)
from src.model.model_validation import (
    validate_model_candidate,
)


MODEL_VERSION = "refactored-v7"

SEQUENCE_LENGTH = 60

TEST_RATIO = 0.20

EPOCHS = 60

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.10

PATIENCE = 10


REGISTRY_PATH = (
    Path("artifacts")
    / "registry"
    / "registry.json"
)


def run_training_pipeline(
    ticker: str = "TSLA",
    sentiment_enabled: bool = False,
):
    """
    Train the V7 directional LSTM model.

    Target:

        Direction[t] = 1
            if Return[t+1] > 0

        Direction[t] = 0
            otherwise.

    Features:

        Return
        Return_5D
        Return_10D
    """

    print("\n" + "=" * 70)
    print("V7 DIRECTIONAL LSTM TRAINING PIPELINE")
    print("=" * 70)

    # ==================================================
    # 1. MARKET DATA
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
    # 2. RETURN FEATURES
    # ==================================================

    print("\n" + "=" * 70)
    print("RETURN FEATURE ENGINEERING")
    print("=" * 70)

    direction_data = add_return_features(
        df=pipeline_result.features,
        price_column=f"{ticker}_Close",
    )

    # ==================================================
    # 3. DIRECTION TARGET
    # ==================================================

    print("\n" + "=" * 70)
    print("DIRECTION TARGET")
    print("=" * 70)

    direction_data = add_direction_target(
        df=direction_data,
        return_column="Return",
    )

    feature_columns = [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    target_column = "Direction"

    direction_data = direction_data[
        feature_columns + [target_column]
    ]

    print(
        "Features:",
        feature_columns,
    )

    print(
        "Target:",
        target_column,
    )

    print(
        "Records:",
        len(direction_data),
    )

    print(
        "UP:",
        int(
            (direction_data[target_column] == 1)
            .sum()
        ),
    )

    print(
        "DOWN:",
        int(
            (direction_data[target_column] == 0)
            .sum()
        ),
    )

    # ==================================================
    # 4. PREPARE DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("DIRECTION DATASET PREPARATION")
    print("=" * 70)

    prepared = prepare_direction_dataset(
        df=direction_data,
        target_column=target_column,
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=TEST_RATIO,
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
        "y_train:",
        prepared.y_train.shape,
    )

    print(
        "X_test:",
        prepared.X_test.shape,
    )

    print(
        "y_test:",
        prepared.y_test.shape,
    )

    # ==================================================
    # 5. EXPERIMENT RECORD
    # ==================================================

    experiment = ExperimentRecord(
        run_id=run_id,
        experiment_name="stock-direction-lstm",
        model_version=MODEL_VERSION,
        model_name="stock-direction-lstm",
        ticker=ticker,
        target="Direction",
        sequence_length=SEQUENCE_LENGTH,
        features=feature_columns,
        training={
            "epochs_requested": EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT,
            "patience": PATIENCE,
            "epochs_completed": None,
        },
    )

    experiment.status = "started"


    # ==================================================
    # 6. PERSIST DATASET ARTIFACTS
    # ==================================================

    print("\n" + "=" * 70)
    print("PERSISTING DATASET ARTIFACTS")
    print("=" * 70)

    run_dir = save_pipeline_run(
        run_id=run_id,
        metadata={
            "model_version": MODEL_VERSION,
            "model_name": experiment.model_name,
            "ticker": ticker,
            "target": target_column,
            "sequence_length": SEQUENCE_LENGTH,
            "features": feature_columns,
            "status": "training_started",
        },
        feature_data=direction_data,
        train_data=prepared.train_data,
        test_data=prepared.test_data,
        experiment=experiment,
    )

    print(
        "Run artifact directory:",
        run_dir,
    )

    # ==================================================
    # 7. BUILD MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("DIRECTIONAL LSTM MODEL")
    print("=" * 70)

    model = build_directional_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
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

    print(
        "Loss:",
        model.loss,
    )

    # ==================================================
    # 8. TRAIN MODEL
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
    # 9. EVALUATE TEST DATA
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    probabilities = (
        training_result.model.predict(
            prepared.X_test,
            verbose=0,
        )
    )

    probabilities = (
        np.asarray(
            probabilities
        ).reshape(-1)
    )

    evaluation = (
        evaluate_binary_classifier(
            y_true=prepared.y_test,
            probabilities=probabilities,
        )
    )

    experiment.metrics = {
        "accuracy": evaluation[
            "accuracy"
        ],
        "precision": evaluation[
            "precision"
        ],
        "recall": evaluation[
            "recall"
        ],
        "f1": evaluation[
            "f1"
        ],
        "roc_auc": evaluation[
            "roc_auc"
        ],
        "confusion_matrix": evaluation[
            "confusion_matrix"
        ],
    }


    # ==================================================
    # MODEL VALIDATION
    # ==================================================

    actual = (
        np.asarray(
            prepared.y_test
        )
        .reshape(-1)
        .astype(int)
    )

    up = int(
        np.sum(actual == 1)
    )

    down = int(
        np.sum(actual == 0)
    )

    majority_baseline = (
        max(up, down)
        / len(actual)
    )

    validation = validate_model_candidate(
        metrics=experiment.metrics,
        majority_baseline=majority_baseline,
    )

    print(
        "Accuracy:",
        f"{experiment.metrics['accuracy']:.4%}",
    )

    print(
        "Precision:",
        f"{experiment.metrics['precision']:.4%}",
    )

    print(
        "Recall:",
        f"{experiment.metrics['recall']:.4%}",
    )

    print(
        "F1:",
        f"{experiment.metrics['f1']:.4%}",
    )

    print(
        "ROC-AUC:",
        f"{experiment.metrics['roc_auc']:.4f}",
    )

    print(
        "Confusion matrix:",
        experiment.metrics[
            "confusion_matrix"
        ],
    )

    # ==================================================
    # 10. MODEL METADATA
    # ==================================================

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": experiment.model_name,
        "ticker": ticker,
        "target": target_column,
        "sequence_length": SEQUENCE_LENGTH,

        "features": feature_columns,

        "dataset": {
            "train_size": prepared.train_size,
            "test_size": prepared.test_size,
            "n_features": prepared.n_features,
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

        "evaluation": dict(
            experiment.metrics
        ),
    }

    # ==================================================
    # 11. SAVE MODEL ARTIFACT
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL ARTIFACT")
    print("=" * 70)

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
    # 12. REGISTER MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("MODEL REGISTRY")
    print("=" * 70)

    registry = ModelRegistry(
        registry_path=REGISTRY_PATH
    )

    registry_record = (
        registry.register_model(
            model_name=experiment.model_name,
            model_version=experiment.model_version,
            run_id=experiment.run_id,
            artifact_path=str(model_dir),
            metrics=dict(
                experiment.metrics
            ),
        )
    )

    registry_record = registry.record_validation(
        model_name=experiment.model_name,
        model_version=experiment.model_version,
        validation=validation,
    )


    print(
        "Registered model:",
        registry_record[
            "model_name"
        ],
    )

    print(
        "Model version:",
        registry_record[
            "model_version"
        ],
    )

    print(
        "Registry status:",
        registry_record[
            "status"
        ],
    )

    # ==================================================
    # 13. UPDATE EXPERIMENT RECORD
    # ==================================================

    experiment.training[
        "epochs_completed"
    ] = training_result.epochs_completed

    experiment.status = "completed"

    # ==================================================
    # 14. UPDATE RUN METADATA
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
        "registry": str(
            REGISTRY_PATH
        ),
    }

    final_metadata[
        "model_registry"
    ] = {
        "model_name": registry_record[
            "model_name"
        ],
        "model_version": registry_record[
            "model_version"
        ],
        "status": registry_record[
            "status"
        ],
    }

    final_metadata["status"] = (
        "training_completed"
    )

    save_pipeline_run(
        run_id=run_id,
        metadata=final_metadata,
        feature_data=direction_data,
        train_data=prepared.train_data,
        test_data=prepared.test_data,
        experiment=experiment,
    )

    # ==================================================
    # 15. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("V7 TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Ticker:",
        ticker,
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
        "Model artifacts:",
        model_dir,
    )

    print(
        "Registry:",
        REGISTRY_PATH,
    )

    print(
        "Registry status:",
        registry_record[
            "status"
        ],
    )

    return {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "model_dir": model_dir,
        "registry": registry_record,
        "experiment": experiment,
    }


if __name__ == "__main__":
    run_training_pipeline()

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from model.model_validation import validate_model_candidate
from src.pipeline.market_pipeline import run_market_pipeline
from src.model.return_features import add_return_features
from src.model.direction_target import add_direction_target
from src.model.direction_preprocessing import (
    prepare_direction_dataset,
)
from src.model.predictor import ModelPredictor
from src.model.evaluation import (
    evaluate_binary_classifier,
)


MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "refactored-v7"
)

TICKER = "TSLA"

SEQUENCE_LENGTH = 60

TEST_RATIO = 0.20


def main() -> None:

    print("=" * 70)
    print("V7 DIRECTIONAL MODEL EVALUATION")
    print("=" * 70)

    # ==================================================
    # 1. LOAD MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("1. LOADING PERSISTED MODEL")
    print("=" * 70)

    predictor = ModelPredictor(
        str(MODEL_DIR)
    )

    print("Model:", predictor.model.name)
    print("Input:", predictor.model.input_shape)
    print("Output:", predictor.model.output_shape)
    print(
        "Scaler:",
        type(predictor.scaler).__name__,
    )
    print(
        "Scaler features:",
        predictor.scaler.n_features_in_,
    )

    # ==================================================
    # 2. MARKET DATA
    # ==================================================

    print("\n" + "=" * 70)
    print("2. RECREATING MARKET DATA")
    print("=" * 70)

    result = run_market_pipeline(
        ticker=TICKER,
        sentiment_enabled=False,
    )

    # ==================================================
    # 3. FEATURES
    # ==================================================

    print("\n" + "=" * 70)
    print("3. BUILDING DIRECTION DATASET")
    print("=" * 70)

    df = add_return_features(
        df=result.features,
        price_column=f"{TICKER}_Close",
    )

    df = add_direction_target(
        df=df,
        return_column="Return",
    )

    feature_columns = [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    df = df[
        feature_columns + ["Direction"]
    ]

    print(
        "Features:",
        feature_columns,
    )

    print(
        "Target: Direction"
    )

    print(
        "Records:",
        len(df),
    )

    # ==================================================
    # 4. PREPARE DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("4. PREPARING TEST DATASET")
    print("=" * 70)

    prepared = prepare_direction_dataset(
        df=df,
        target_column="Direction",
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=TEST_RATIO,
    )

    print(
        "Train rows:",
        prepared.train_size,
    )

    print(
        "Test rows:",
        prepared.test_size,
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
    # 5. STRUCTURAL VALIDATION
    # ==================================================

    expected_features = (
        predictor.scaler.n_features_in_
    )

    actual_features = (
        prepared.X_test.shape[2]
    )

    assert actual_features == expected_features

    assert (
        prepared.X_test.shape[1]
        == SEQUENCE_LENGTH
    )

    assert (
        len(prepared.X_test)
        == prepared.test_size
    )

    print(
        "\nStructural validation: PASS"
    )

    # ==================================================
    # 6. PREDICTIONS
    # ==================================================

    print("\n" + "=" * 70)
    print("5. GENERATING DIRECTION PREDICTIONS")
    print("=" * 70)

    probabilities = predictor.model.predict(
        prepared.X_test,
        verbose=0,
    )

    probabilities = (
        np.asarray(probabilities)
        .reshape(-1)
    )

    actual = (
        np.asarray(
            prepared.y_test
        )
        .reshape(-1)
        .astype(int)
    )

    print(
        "Samples:",
        len(probabilities),
    )

    print(
        "Probability min:",
        f"{probabilities.min():.6f}",
    )

    print(
        "Probability max:",
        f"{probabilities.max():.6f}",
    )

    print(
        "Probability mean:",
        f"{probabilities.mean():.6f}",
    )

    # ==================================================
    # 7. CLASS DISTRIBUTION
    # ==================================================

    print("\n" + "=" * 70)
    print("6. TEST DISTRIBUTION")
    print("=" * 70)

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

    print(
        "UP:",
        up,
    )

    print(
        "DOWN:",
        down,
    )

    print(
        "UP ratio:",
        f"{up / len(actual):.4%}",
    )

    print(
        "DOWN ratio:",
        f"{down / len(actual):.4%}",
    )

    print(
        "Majority baseline:",
        f"{majority_baseline:.4%}",
    )

    # ==================================================
    # 8. CLASSIFICATION METRICS
    # ==================================================

    print("\n" + "=" * 70)
    print("7. CLASSIFICATION METRICS")
    print("=" * 70)

    evaluation = evaluate_binary_classifier(
        y_true=actual,
        probabilities=probabilities,
    )

    validation = validate_model_candidate(
        metrics=evaluation,
        majority_baseline=majority_baseline,
    )

    predictions = np.asarray(
        evaluation["predictions"]
    )

    accuracy = evaluation[
        "accuracy"
    ]

    precision = evaluation[
        "precision"
    ]

    recall = evaluation[
        "recall"
    ]

    f1 = evaluation[
        "f1"
    ]

    auc = evaluation[
        "roc_auc"
    ]

    print(
        "Accuracy:",
        f"{accuracy:.4%}",
    )

    print(
        "Precision:",
        f"{precision:.4%}",
    )

    print(
        "Recall:",
        f"{recall:.4%}",
    )

    print(
        "F1:",
        f"{f1:.4%}",
    )

    print(
        "ROC-AUC:",
        f"{auc:.4f}",
    )

    improvement_pp = (
        accuracy
        - majority_baseline
    ) * 100

    print(
        "vs majority:",
        f"{improvement_pp:+.2f} pp",
    )

    # ==================================================
    # 9. CONFUSION MATRIX
    # ==================================================

    print("\n" + "=" * 70)
    print("8. CONFUSION MATRIX")
    print("=" * 70)

    cm = np.asarray(
        evaluation[
            "confusion_matrix"
        ]
    )

    print(
        "                 Pred DOWN   Pred UP"
    )

    print(
        f"Actual DOWN      {cm[0,0]:10d}   {cm[0,1]:7d}"
    )

    print(
        f"Actual UP        {cm[1,0]:10d}   {cm[1,1]:7d}"
    )

    # ==================================================
    # 10. BINOMIAL TEST
    # ==================================================

    print("\n" + "=" * 70)
    print("9. BINOMIAL SIGNIFICANCE TEST")
    print("=" * 70)

    result_test = binomtest(
        k=int(
            np.sum(
                predictions == actual
            )
        ),
        n=len(actual),
        p=majority_baseline,
        alternative="greater",
    )

    p_value = result_test.pvalue

    print(
        "Correct:",
        int(
            np.sum(
                predictions == actual
            )
        ),
    )

    print(
        "Accuracy:",
        f"{accuracy:.4%}",
    )

    print(
        "Null baseline:",
        f"{majority_baseline:.4%}",
    )

    print(
        "p-value:",
        f"{p_value:.6f}",
    )

    # ==================================================
    # 11. SIGNAL VERDICT
    # ==================================================

    print("\n" + "=" * 70)
    print("10. SIGNAL VERDICT")
    print("=" * 70)

    if (
        accuracy > majority_baseline
        and p_value < 0.05
    ):
        verdict = (
            "STATISTICALLY SIGNIFICANT SIGNAL"
        )

    elif accuracy > majority_baseline:
        verdict = (
            "POSITIVE BUT NOT "
            "STATISTICALLY SIGNIFICANT"
        )

    else:
        verdict = (
            "NO OUT-OF-SAMPLE EDGE"
        )

    print(verdict)

    # ==================================================
    # FINAL
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL V7 DIRECTIONAL EVALUATION")
    print("=" * 70)

    print(
        "Model version: refactored-v7"
    )

    print(
        "Ticker:",
        TICKER,
    )

    print(
        "Test samples:",
        len(actual),
    )

    print(
        f"Accuracy: {accuracy:.4%}"
    )

    print(
        f"Majority baseline: "
        f"{majority_baseline:.4%}"
    )

    print(
        f"Improvement: "
        f"{improvement_pp:+.2f} pp"
    )

    print(
        f"F1: {f1:.4%}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"p-value: {p_value:.6f}"
    )

    print(
        "Verdict:",
        verdict,
    )


if __name__ == "__main__":
    main()

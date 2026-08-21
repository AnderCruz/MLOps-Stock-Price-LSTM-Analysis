from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.market_pipeline import run_market_pipeline
from src.model.return_features import add_return_features
from src.model.direction_target import add_direction_target
from src.model.direction_preprocessing import prepare_direction_dataset
from src.model.directional_model import build_directional_lstm_model


TICKER = "TSLA"
SEQUENCE_LENGTH = 60

# Percentual inicial reservado para cada expansão de treino.
TRAIN_RATIOS = [
    0.50,
    0.60,
    0.70,
    0.80,
]

EPOCHS = 60
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.10
PATIENCE = 10


def build_dataset():
    result = run_market_pipeline(
        ticker=TICKER,
        sentiment_enabled=False,
    )

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

    return df[
        feature_columns + ["Direction"]
    ]


def evaluate_fold(
    df: pd.DataFrame,
    train_ratio: float,
    fold_number: int,
):
    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}")
    print("=" * 70)

    split_index = int(
        len(df) * train_ratio
    )

    if split_index >= len(df):
        raise ValueError(
            "Invalid walk-forward split."
        )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # --------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------

    combined = pd.concat(
        [
            train_df,
            test_df,
        ]
    )

    # We deliberately reconstruct the chronological
    # split through the existing preprocessing logic.
    test_ratio = len(test_df) / len(combined)

    prepared = prepare_direction_dataset(
        df=combined,
        target_column="Direction",
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=test_ratio,
    )

    print(
        "X_train:",
        prepared.X_train.shape,
    )

    print(
        "X_test:",
        prepared.X_test.shape,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build_directional_lstm_model(
        sequence_length=prepared.sequence_length,
        n_features=prepared.n_features,
    )

    print(
        "Model:",
        model.name,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=False,
        verbose=0,
    )

    epochs_completed = len(
        history.history["loss"]
    )

    print(
        "Epochs:",
        epochs_completed,
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    probabilities = (
        model.predict(
            prepared.X_test,
            verbose=0,
        )
        .reshape(-1)
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    actual = (
        np.asarray(
            prepared.y_test
        )
        .reshape(-1)
        .astype(int)
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            actual,
            predictions,
        )
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            actual,
            probabilities,
        )
    except ValueError:
        auc = float("nan")

    correct = int(
        np.sum(
            actual == predictions
        )
    )

    n = len(actual)

    majority = max(
        np.mean(actual == 0),
        np.mean(actual == 1),
    )

    p_value = binomtest(
        k=correct,
        n=n,
        p=majority,
        alternative="greater",
    ).pvalue

    print()
    print(
        f"Accuracy:            {accuracy:.4%}"
    )
    print(
        f"Balanced Accuracy:   {balanced_accuracy:.4%}"
    )
    print(
        f"F1:                  {f1:.4%}"
    )
    print(
        f"ROC-AUC:             {auc:.4f}"
    )
    print(
        f"Majority baseline:   {majority:.4%}"
    )
    print(
        f"Improvement:         {(accuracy - majority) * 100:+.2f} pp"
    )
    print(
        f"p-value:             {p_value:.6f}"
    )

    return {
        "fold": fold_number,
        "train_ratio": train_ratio,
        "train_samples": len(prepared.X_train),
        "test_samples": len(prepared.X_test),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "roc_auc": auc,
        "majority_baseline": majority,
        "improvement_pp": (
            accuracy - majority
        ) * 100,
        "p_value": p_value,
        "epochs": epochs_completed,
    }


def main():
    print("=" * 70)
    print("V7 WALK-FORWARD VALIDATION")
    print("=" * 70)

    df = build_dataset()

    print()
    print("Dataset:")
    print("Records:", len(df))
    print("Features:", [
        "Return",
        "Return_5D",
        "Return_10D",
    ])

    results = []

    for fold_number, train_ratio in enumerate(
        TRAIN_RATIOS,
        start=1,
    ):
        results.append(
            evaluate_fold(
                df=df,
                train_ratio=train_ratio,
                fold_number=fold_number,
            )
        )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "fold",
                "train_ratio",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "roc_auc",
                "majority_baseline",
                "improvement_pp",
                "p_value",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print(
        f"Mean Accuracy:          "
        f"{results_df['accuracy'].mean():.4%}"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{results_df['balanced_accuracy'].mean():.4%}"
    )

    print(
        f"Mean F1:                "
        f"{results_df['f1'].mean():.4%}"
    )

    print(
        f"Mean ROC-AUC:           "
        f"{results_df['roc_auc'].mean():.4f}"
    )

    print(
        f"Mean Improvement:       "
        f"{results_df['improvement_pp'].mean():+.2f} pp"
    )

    significant_folds = (
        results_df["p_value"] < 0.05
    ).sum()

    print(
        f"Significant folds:      "
        f"{significant_folds}/{len(results_df)}"
    )

    print("\n" + "=" * 70)

    if (
        results_df["roc_auc"].mean() > 0.55
        and results_df["improvement_pp"].mean() > 0
        and significant_folds >= 2
    ):
        print(
            "WALK-FORWARD SIGNAL: PROMISING"
        )
    else:
        print(
            "WALK-FORWARD SIGNAL: NOT CONFIRMED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()

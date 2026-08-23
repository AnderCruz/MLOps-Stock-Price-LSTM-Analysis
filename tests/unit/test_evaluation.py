import numpy as np
import pytest

from src.model.evaluation import (
    evaluate_binary_classifier,
)


def test_binary_classifier_metrics_are_calculated():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.80, 0.90, 0.20]
    )

    result = evaluate_binary_classifier(
        y_true=y_true,
        probabilities=probabilities,
    )

    assert result["accuracy"] == pytest.approx(
        0.5
    )

    assert result["precision"] == pytest.approx(
        0.5
    )

    assert result["recall"] == pytest.approx(
        0.5
    )

    assert result["f1"] == pytest.approx(
        0.5
    )

    assert result["roc_auc"] == pytest.approx(
        0.75
    )


def test_predictions_use_point_five_threshold():
    y_true = np.array(
        [0, 1, 0, 1]
    )

    probabilities = np.array(
        [0.49, 0.50, 0.51, 0.10]
    )

    result = evaluate_binary_classifier(
        y_true=y_true,
        probabilities=probabilities,
    )

    assert result["predictions"] == [
        0,
        1,
        1,
        0,
    ]


def test_confusion_matrix_is_serializable():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    result = evaluate_binary_classifier(
        y_true=y_true,
        probabilities=probabilities,
    )

    assert result["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]


def test_probability_and_target_lengths_must_match():
    y_true = np.array(
        [0, 1, 1]
    )

    probabilities = np.array(
        [0.2, 0.8]
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_binary_classifier(
            y_true=y_true,
            probabilities=probabilities,
        )


def test_binary_targets_are_required():
    y_true = np.array(
        [0, 1, 2, 1]
    )

    probabilities = np.array(
        [0.1, 0.8, 0.9, 0.2]
    )

    with pytest.raises(
        ValueError,
        match="binary",
    ):
        evaluate_binary_classifier(
            y_true=y_true,
            probabilities=probabilities,
        )


def test_probabilities_must_be_between_zero_and_one():
    y_true = np.array(
        [0, 1, 0, 1]
    )

    probabilities = np.array(
        [0.1, 1.2, 0.3, 0.8]
    )

    with pytest.raises(
        ValueError,
        match="probabilities",
    ):
        evaluate_binary_classifier(
            y_true=y_true,
            probabilities=probabilities,
        )

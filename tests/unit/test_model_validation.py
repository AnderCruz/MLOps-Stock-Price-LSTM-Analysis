from src.model.model_validation import (
    ValidationResult,
    validate_model_candidate,
)


def test_model_passes_validation_when_metrics_meet_policy():
    result = validate_model_candidate(
        metrics={
            "accuracy": 0.60,
            "roc_auc": 0.65,
        },
        majority_baseline=0.51,
    )

    assert isinstance(
        result,
        ValidationResult,
    )

    assert result.passed is True

    assert result.checks[
        "accuracy_vs_baseline"
    ] is True

    assert result.checks[
        "minimum_roc_auc"
    ] is True

    assert result.reasons == []


def test_model_fails_when_accuracy_does_not_beat_baseline():
    result = validate_model_candidate(
        metrics={
            "accuracy": 0.50,
            "roc_auc": 0.65,
        },
        majority_baseline=0.51,
    )

    assert result.passed is False

    assert result.checks[
        "accuracy_vs_baseline"
    ] is False


def test_model_fails_when_roc_auc_is_too_low():
    result = validate_model_candidate(
        metrics={
            "accuracy": 0.60,
            "roc_auc": 0.51,
        },
        majority_baseline=0.51,
    )

    assert result.passed is False

    assert result.checks[
        "minimum_roc_auc"
    ] is False


def test_model_fails_when_both_checks_fail():
    result = validate_model_candidate(
        metrics={
            "accuracy": 0.50,
            "roc_auc": 0.51,
        },
        majority_baseline=0.51,
    )

    assert result.passed is False

    assert (
        len(result.reasons) == 2
    )


def test_validation_result_is_serializable():
    result = validate_model_candidate(
        metrics={
            "accuracy": 0.60,
            "roc_auc": 0.65,
        },
        majority_baseline=0.51,
    )

    data = result.to_dict()

    assert data["passed"] is True
    assert "checks" in data
    assert "reasons" in data

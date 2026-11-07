
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.drift.drift_detector import (
    calculate_psi,
    calculate_ks,
    calculate_wasserstein,
    calculate_js,
    normalize_categories,
    classify_result,
    save_results,
)

from src.drift.drift_detector import (
    calculate_psi,
    calculate_ks,
    calculate_wasserstein,
    calculate_js,
    normalize_categories,
    classify_result,
)


# --------------------------------------------------
# PSI TESTS
# --------------------------------------------------



def test_psi_normalizes_distributions_after_smoothing():
    reference = pd.Series(list(range(100)))
    current = pd.Series(list(range(90)) + [200] * 10)

    result = calculate_psi(reference, current)

    # Independently calculate the expected normalized PSI.
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, 11))
    )
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 0.001

    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    expected = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    assert result == pytest.approx(expected)


def test_psi_identical_distributions():
    reference = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    current = reference.copy()

    result = calculate_psi(reference, current)

    assert result == pytest.approx(0.0)


def test_psi_detects_distribution_shift():
    reference = pd.Series(np.arange(100))
    current = pd.Series(np.arange(100) + 50)

    result = calculate_psi(reference, current)

    assert result > 0


def test_psi_returns_nan_for_empty_reference():
    reference = pd.Series([], dtype=float)
    current = pd.Series([1, 2, 3])

    result = calculate_psi(reference, current)

    assert np.isnan(result)


def test_psi_returns_nan_for_empty_current():
    reference = pd.Series([1, 2, 3])
    current = pd.Series([], dtype=float)

    result = calculate_psi(reference, current)

    assert np.isnan(result)


def test_psi_handles_constant_identical_distributions():
    reference = pd.Series([5, 5, 5, 5])
    current = pd.Series([5, 5, 5, 5])

    result = calculate_psi(reference, current)

    assert result == pytest.approx(0.0)


def test_psi_detects_shift_from_constant_reference():
    reference = pd.Series([5, 5, 5, 5])
    current = pd.Series([10, 10, 10, 10])

    result = calculate_psi(reference, current)

    assert result > 0


# --------------------------------------------------
# KS TESTS
# --------------------------------------------------

def test_ks_identical_distributions():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = reference.copy()

    result = calculate_ks(reference, current)

    assert result == pytest.approx(1.0)


def test_ks_detects_distribution_difference():
    reference = pd.Series(np.arange(100))
    current = pd.Series(np.arange(100) + 1000)

    result = calculate_ks(reference, current)

    assert result < 0.05


def test_ks_returns_nan_for_empty_data():
    reference = pd.Series([], dtype=float)
    current = pd.Series([1, 2, 3])

    result = calculate_ks(reference, current)

    assert np.isnan(result)


def test_ks_ignores_invalid_numeric_values():
    reference = pd.Series(["1", "2", "invalid", "3"])
    current = pd.Series(["1", "2", "3"])

    result = calculate_ks(reference, current)

    assert result == pytest.approx(1.0)


# --------------------------------------------------
# WASSERSTEIN TESTS
# --------------------------------------------------

def test_wasserstein_identical_distributions():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = reference.copy()

    result = calculate_wasserstein(reference, current)

    assert result == pytest.approx(0.0)


def test_wasserstein_detects_shift():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = pd.Series([11, 12, 13, 14, 15])

    result = calculate_wasserstein(reference, current)

    assert result > 0


def test_wasserstein_returns_nan_for_empty_data():
    reference = pd.Series([], dtype=float)
    current = pd.Series([1, 2, 3])

    result = calculate_wasserstein(reference, current)

    assert np.isnan(result)


def test_wasserstein_returns_nan_for_constant_reference():
    reference = pd.Series([10, 10, 10, 10, 10])
    current = pd.Series([10, 11, 12, 13, 14])

    result = calculate_wasserstein(reference, current)

    assert np.isnan(result)


# --------------------------------------------------
# JENSEN-SHANNON TESTS
# --------------------------------------------------

def test_js_identical_categorical_distributions():
    reference = pd.Series(["cash", "card", "cash", "card"])
    current = reference.copy()

    result = calculate_js(reference, current)

    assert result == pytest.approx(0.0)


def test_js_detects_categorical_distribution_shift():
    reference = pd.Series(["cash"] * 100)
    current = pd.Series(["card"] * 100)

    result = calculate_js(reference, current)

    assert result == pytest.approx(1.0)


def test_js_handles_categories_missing_from_one_distribution():
    reference = pd.Series(["cash", "cash", "cash", "card"])
    current = pd.Series(["cash", "cash", "card", "other"])

    result = calculate_js(reference, current)

    assert np.isfinite(result)
    assert 0 <= result <= 1


def test_js_returns_nan_for_empty_distributions():
    reference = pd.Series([], dtype=object)
    current = pd.Series([], dtype=object)

    result = calculate_js(reference, current)

    assert np.isnan(result)


def test_js_returns_nan_when_one_distribution_is_empty():
    reference = pd.Series(["cash", "card"])
    current = pd.Series([], dtype=object)

    result = calculate_js(reference, current)

    assert np.isnan(result)


def test_normalize_categories_handles_numeric_formats():
    series = pd.Series([1, 1.0, "2", " card ", None])

    result = normalize_categories(series)

    assert result.tolist() == [
        "1",
        "1",
        "2",
        "card",
        "MISSING",
    ]


# --------------------------------------------------
# SEVERITY CLASSIFICATION TESTS
# --------------------------------------------------

@pytest.mark.parametrize(
    "metric,value,expected_detected,expected_severity",
    [
        ("PSI", 0.05, False, "low"),
        ("PSI", 0.15, False, "moderate"),
        ("PSI", 0.20, True, "high"),
        ("KS_PVALUE", 0.01, True, "moderate"),
        ("KS_PVALUE", 0.05, False, "low"),
        ("WASSERSTEIN", 0.02, False, "low"),
        ("WASSERSTEIN", 0.08, False, "moderate"),
        ("WASSERSTEIN", 0.10, True, "high"),
        ("JS_DIVERGENCE", 0.02, False, "low"),
        ("JS_DIVERGENCE", 0.07, False, "moderate"),
        ("JS_DIVERGENCE", 0.10, True, "high"),
    ],
)
def test_classify_result(
    metric,
    value,
    expected_detected,
    expected_severity,
):
    detected, severity = classify_result(metric, value)

    assert detected == expected_detected
    assert severity == expected_severity


def test_classify_nan_as_unknown():
    detected, severity = classify_result("PSI", np.nan)

    assert detected is False
    assert severity == "unknown"


def test_unknown_metric_raises_error():
    with pytest.raises(ValueError, match="Unknown metric"):
        classify_result("INVALID_METRIC", 0.5)

# --------------------------------------------------
# SAVE RESULTS TESTS
# --------------------------------------------------

def create_drift_result(**overrides):
    result = {
        "reference_period": "2025-01-01",
        "current_period": "2025-02-01",
        "reference_source": "January 2025 random sample (seed=42)",
        "current_batch_id": 3,
        "feature_name": "fare_amount",
        "metric_name": "PSI",
        "metric_value": 0.25,
        "threshold": 0.20,
        "drift_detected": True,
        "severity": "high",
        "reference_count": 10000,
        "current_count": 10000,
    }

    result.update(overrides)
    return result


def test_save_results_deletes_before_inserting():
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value

    results = [
        create_drift_result(),
        create_drift_result(
            feature_name="trip_distance",
            metric_name="KS_PVALUE",
        ),
    ]

    save_results(engine, results)

    assert connection.execute.call_count == 2

    delete_call, insert_call = connection.execute.call_args_list

    assert "DELETE FROM drift_results" in str(delete_call.args[0])
    assert "INSERT INTO drift_results" in str(insert_call.args[0])

    assert delete_call.args[1]["current_batch_id"] == 3
    assert insert_call.args[1] == results


def test_save_results_uses_database_transaction():
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value

    save_results(engine, [create_drift_result()])

    engine.begin.assert_called_once()
    assert connection.execute.call_count == 2


def test_save_results_rejects_mixed_batches():
    engine = MagicMock()

    results = [
        create_drift_result(),
        create_drift_result(current_batch_id=4),
    ]

    with pytest.raises(ValueError, match="exactly one batch"):
        save_results(engine, results)

    engine.begin.assert_not_called()


def test_save_results_returns_for_empty_results():
    engine = MagicMock()

    save_results(engine, [])

    engine.begin.assert_not_called()

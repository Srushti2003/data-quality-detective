import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from src.evaluation import incident_evaluator as evaluator


# --------------------------------------------------
# calculate_metrics() tests
# --------------------------------------------------

def test_perfect_detection():
    actual = {1, 2, 3}
    predicted = {1, 2, 3}
    all_ids = {1, 2, 3, 4, 5}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 3
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2
    assert result["precision_score"] == 1.0
    assert result["recall_score"] == 1.0
    assert result["f1_score"] == 1.0


def test_partial_detection():
    actual = {1, 2, 3}
    predicted = {1, 2, 4}
    all_ids = {1, 2, 3, 4, 5}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision_score"] == pytest.approx(2 / 3)
    assert result["recall_score"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)


def test_no_predictions():
    actual = {1, 2}
    predicted = set()
    all_ids = {1, 2, 3, 4}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 2
    assert result["true_negative"] == 2
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_no_actual_incidents():
    actual = set()
    predicted = {1, 2}
    all_ids = {1, 2, 3, 4}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 2
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_no_actual_incidents_and_no_predictions():
    actual = set()
    predicted = set()
    all_ids = {1, 2, 3}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 3
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_all_predictions_are_wrong():
    actual = {1, 2}
    predicted = {3, 4}
    all_ids = {1, 2, 3, 4, 5}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 2
    assert result["false_negative"] == 2
    assert result["true_negative"] == 1
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_all_records_are_actual_incidents():
    actual = {1, 2, 3}
    predicted = {1, 2}
    all_ids = {1, 2, 3}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 1
    assert result["true_negative"] == 0
    assert result["precision_score"] == 1.0
    assert result["recall_score"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(0.8)


def test_empty_dataset():
    actual = set()
    predicted = set()
    all_ids = set()

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 0
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_duplicate_ids_do_not_change_set_metrics():
    actual = {1, 1, 2}
    predicted = {1, 1, 3}
    all_ids = {1, 2, 3, 4}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1


def test_prediction_outside_dataset_is_counted_as_false_positive():
    actual = {1}
    predicted = {1, 999}
    all_ids = {1, 2, 3}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    # Documents the current implementation:
    # predicted IDs are not restricted to all_ids.
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1


def test_actual_incident_outside_dataset_is_counted_as_false_negative():
    actual = {1, 999}
    predicted = {1}
    all_ids = {1, 2, 3}

    result = evaluator.calculate_metrics(
        actual, predicted, all_ids
    )

    # Documents the current implementation:
    # actual IDs are not restricted to all_ids.
    assert result["true_positive"] == 1
    assert result["false_negative"] == 1


# --------------------------------------------------
# Configuration tests
# --------------------------------------------------

def test_expected_detection_methods_are_configured():
    assert evaluator.METHODS == [
        "IQR",
        "Z_SCORE",
        "MODIFIED_Z",
        "ISOLATION_FOREST",
    ]


# --------------------------------------------------
# main() integration tests with mocked database
# --------------------------------------------------

def test_main_loads_data_evaluates_and_saves_results():
    fake_engine = MagicMock()
    fake_connection = MagicMock()
    fake_transaction = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_engine.begin.return_value.__enter__.return_value = (
        fake_transaction
    )

    all_trips = pd.DataFrame({
        "trip_id": [1, 2, 3, 4, 5, 6]
    })

    incidents = pd.DataFrame({
        "incident_type": [
            "EXTREME_DISTANCE",
            "EXTREME_DISTANCE",
            "NEGATIVE_FARE",
        ],
        "trip_id": [1, 2, 3],
    })

    predictions_by_method = {
        "IQR": [1, 3],
        "Z_SCORE": [1, 2],
        "MODIFIED_Z": [3, 4],
        "ISOLATION_FOREST": [5],
    }

    def mock_read_sql(query, connection, params=None):
        query_text = str(query)

        if "FROM taxi_trips" in query_text:
            return all_trips.copy()

        if "FROM injected_incidents" in query_text:
            return incidents.copy()

        if "FROM outlier_results" in query_text:
            method = params["method"]

            return pd.DataFrame({
                "trip_id": predictions_by_method[method]
            })

        raise AssertionError(
            f"Unexpected SQL query: {query_text}"
        )

    with patch.object(
        evaluator, "get_engine", return_value=fake_engine
    ), patch.object(
        evaluator.pd, "read_sql", side_effect=mock_read_sql
    ):
        evaluator.main()

    # The evaluator should delete old results and insert new ones.
    assert fake_engine.begin.return_value.__enter__.return_value.execute.call_count == 2

    execute_calls = (
        fake_engine.begin.return_value.__enter__.return_value.execute.call_args_list
    )

    delete_query = str(execute_calls[0].args[0])
    insert_query = str(execute_calls[1].args[0])

    assert "DELETE FROM incident_evaluation" in delete_query
    assert "INSERT INTO incident_evaluation" in insert_query

    inserted_results = execute_calls[1].args[1]

    # 2 incident categories × 4 detection methods = 8 results.
    assert len(inserted_results) == 8

    assert {
        result["incident_type"]
        for result in inserted_results
    } == {"EXTREME_DISTANCE", "NEGATIVE_FARE"}

    assert {
        result["detection_method"]
        for result in inserted_results
    } == set(evaluator.METHODS)

    assert all(
        result["batch_id"] == evaluator.BATCH_ID
        for result in inserted_results
    )

    fake_engine.dispose.assert_called_once()


def test_main_disposes_engine_when_database_read_fails():
    fake_engine = MagicMock()

    fake_connection = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    with patch.object(
        evaluator, "get_engine", return_value=fake_engine
    ), patch.object(
        evaluator.pd,
        "read_sql",
        side_effect=RuntimeError("Database read failed"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Database read failed",
        ):
            evaluator.main()

    fake_engine.dispose.assert_called_once()
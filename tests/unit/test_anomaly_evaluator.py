
import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch, call

from src.evaluation.anomaly_evaluator import (
    evaluate_method,
    get_engine,
    load_ground_truth,
    load_predictions,
    save_results,
    DETECTION_METHODS,
    BATCH_ID,
)


# --------------------------------------------------
# 1. Basic evaluation
# --------------------------------------------------

def test_evaluate_method_perfect_predictions():
    all_ids = pd.Series([1, 2, 3, 4])
    actual = {1, 2}
    predicted = {1, 2}

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2
    assert result["precision_score"] == 1.0
    assert result["recall_score"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_evaluate_method_partial_predictions():
    all_ids = pd.Series([1, 2, 3, 4, 5])
    actual = {1, 2, 3}
    predicted = {1, 2, 4}

    result = evaluate_method(
        all_ids, actual, predicted, "Z_SCORE"
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1

    assert result["precision_score"] == pytest.approx(2 / 3)
    assert result["recall_score"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 2)


def test_evaluate_method_no_predictions():
    all_ids = pd.Series([1, 2, 3, 4])
    actual = {1, 2}
    predicted = set()

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 2
    assert result["true_negative"] == 2
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0


def test_evaluate_method_no_actual_anomalies():
    all_ids = pd.Series([1, 2, 3, 4])
    actual = set()
    predicted = {1, 2}

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 2
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["false_positive_rate"] == 0.5


def test_evaluate_method_no_actual_or_predicted_anomalies():
    all_ids = pd.Series([1, 2, 3])
    actual = set()
    predicted = set()

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_negative"] == 3
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["false_positive_rate"] == 0.0


# --------------------------------------------------
# 2. Result structure and edge cases
# --------------------------------------------------

def test_evaluate_method_returns_expected_keys():
    result = evaluate_method(
        pd.Series([1, 2, 3]),
        {1},
        {1},
        "IQR",
    )

    expected_keys = {
        "detection_method",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "precision_score",
        "recall_score",
        "f1_score",
        "false_positive_rate",
        "total_records",
    }

    assert set(result.keys()) == expected_keys


def test_evaluate_method_records_method_name():
    result = evaluate_method(
        pd.Series([1, 2]),
        {1},
        {1},
        "MODIFIED_Z",
    )

    assert result["detection_method"] == "MODIFIED_Z"


def test_evaluate_method_records_total_records():
    all_ids = pd.Series([1, 2, 3, 4, 5])

    result = evaluate_method(
        all_ids, {1}, {1}, "IQR"
    )

    assert result["total_records"] == 5


def test_evaluate_method_predictions_outside_dataset_are_ignored():
    all_ids = pd.Series([1, 2, 3])
    actual = {1}
    predicted = {1, 999}

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_positive"] == 1
    assert result["false_positive"] == 0
    assert result["total_records"] == 3


def test_evaluate_method_all_predictions_wrong():
    all_ids = pd.Series([1, 2, 3, 4])
    actual = {1, 2}
    predicted = {3, 4}

    result = evaluate_method(
        all_ids, actual, predicted, "IQR"
    )

    assert result["true_positive"] == 0
    assert result["false_positive"] == 2
    assert result["false_negative"] == 2
    assert result["true_negative"] == 0
    assert result["precision_score"] == 0.0
    assert result["recall_score"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["false_positive_rate"] == 1.0


# --------------------------------------------------
# 3. Detection method configuration
# --------------------------------------------------

def test_all_four_detection_methods_registered():
    assert DETECTION_METHODS == [
        "IQR",
        "Z_SCORE",
        "MODIFIED_Z",
        "ISOLATION_FOREST",
    ]


# --------------------------------------------------
# 4. Database loading functions
# --------------------------------------------------

@patch("src.evaluation.anomaly_evaluator.pd.read_sql")
def test_load_ground_truth(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {"trip_id": [1, 2, 3]}
    )

    connection = MagicMock()

    result = load_ground_truth(connection)

    assert result["trip_id"].tolist() == [1, 2, 3]
    mock_read_sql.assert_called_once()

    kwargs = mock_read_sql.call_args.kwargs
    assert kwargs["params"] == {"batch_id": BATCH_ID}


@patch("src.evaluation.anomaly_evaluator.pd.read_sql")
def test_load_predictions(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {"trip_id": [10, 20]}
    )

    connection = MagicMock()

    result = load_predictions(connection, "IQR")

    assert result["trip_id"].tolist() == [10, 20]

    kwargs = mock_read_sql.call_args.kwargs
    assert kwargs["params"] == {
        "batch_id": BATCH_ID,
        "method": "IQR",
    }


# --------------------------------------------------
# 5. Database saving function
# --------------------------------------------------

def test_save_results_deletes_previous_batch_results():
    connection = MagicMock()

    save_results(connection, [])

    assert connection.execute.call_count == 1

    params = connection.execute.call_args.args[1]
    assert params == {"batch_id": BATCH_ID}


def test_save_results_inserts_each_result():
    connection = MagicMock()

    results = [
        {
            "detection_method": "IQR",
            "true_positive": 2,
            "false_positive": 1,
            "false_negative": 1,
            "true_negative": 6,
            "precision_score": 2 / 3,
            "recall_score": 2 / 3,
            "f1_score": 2 / 3,
            "false_positive_rate": 1 / 7,
            "total_records": 10,
        },
        {
            "detection_method": "Z_SCORE",
            "true_positive": 3,
            "false_positive": 1,
            "false_negative": 0,
            "true_negative": 6,
            "precision_score": 0.75,
            "recall_score": 1.0,
            "f1_score": 6 / 7,
            "false_positive_rate": 1 / 7,
            "total_records": 10,
        },
    ]

    save_results(connection, results)

    # One DELETE + two INSERT operations.
    assert connection.execute.call_count == 3

    inserted_params = [
        item.args[1]
        for item in connection.execute.call_args_list[1:]
    ]

    assert inserted_params[0]["detection_method"] == "IQR"
    assert inserted_params[1]["detection_method"] == "Z_SCORE"

    for params in inserted_params:
        assert params["batch_id"] == BATCH_ID


# --------------------------------------------------
# 6. Database engine configuration
# --------------------------------------------------

@patch("src.evaluation.anomaly_evaluator.create_engine")
@patch("src.evaluation.anomaly_evaluator.URL.create")
@patch("src.evaluation.anomaly_evaluator.os.getenv")
def test_get_engine_uses_environment_configuration(
    mock_getenv,
    mock_url_create,
    mock_create_engine,
):
    mock_getenv.side_effect = lambda key, default=None: {
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_database",
    }.get(key, default)

    mock_url_create.return_value = "mock_database_url"
    mock_create_engine.return_value = "mock_engine"

    result = get_engine()

    assert result == "mock_engine"

    mock_url_create.assert_called_once()
    mock_create_engine.assert_called_once_with(
        "mock_database_url",
        pool_pre_ping=True,
    )

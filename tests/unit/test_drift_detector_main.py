from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

from src.drift import drift_detector as detector
from src.drift.drift_detector import save_results


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def make_engine():
    engine = MagicMock()

    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection

    transaction = MagicMock()
    engine.begin.return_value.__enter__.return_value = transaction

    return engine, connection, transaction


def make_feature_dataframe():
    data = {}

    for feature in (
        detector.NUMERIC_FEATURES
        + detector.CATEGORICAL_FEATURES
    ):
        data[feature] = [1, 2, 3]

    return pd.DataFrame(data)


def make_drift_result(batch_id, metric_name="PSI"):
    return {
        "reference_period": detector.REFERENCE_PERIOD,
        "current_period": "2025-02-01",
        "reference_source": detector.REFERENCE_SOURCE,
        "current_batch_id": batch_id,
        "feature_name": "fare_amount",
        "metric_name": metric_name,
        "metric_value": 0.25,
        "threshold": 0.20,
        "drift_detected": True,
        "severity": "high",
        "reference_count": 100,
        "current_count": 100,
    }


# --------------------------------------------------
# DATABASE ENGINE
# --------------------------------------------------

def test_get_engine_uses_environment_configuration(monkeypatch):
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "test_database")

    url_mock = MagicMock()
    create_engine_mock = MagicMock()

    monkeypatch.setattr(
        detector.URL,
        "create",
        url_mock,
    )
    monkeypatch.setattr(
        detector,
        "create_engine",
        create_engine_mock,
    )

    detector.get_engine()

    url_mock.assert_called_once_with(
        "postgresql+psycopg2",
        username="test_user",
        password="test_password",
        host="localhost",
        port=5433,
        database="test_database",
    )

    create_engine_mock.assert_called_once_with(
        url_mock.return_value,
        pool_pre_ping=True,
    )


def test_get_engine_uses_default_port(monkeypatch):
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "test_database")
    monkeypatch.delenv("DB_PORT", raising=False)

    url_mock = MagicMock()

    monkeypatch.setattr(
        detector.URL,
        "create",
        url_mock,
    )
    monkeypatch.setattr(
        detector,
        "create_engine",
        MagicMock(),
    )

    detector.get_engine()

    assert url_mock.call_args.kwargs["port"] == 5432


# --------------------------------------------------
# REFERENCE DATA LOADING
# --------------------------------------------------

def test_load_reference_data_reads_and_samples_parquet(
    monkeypatch,
    tmp_path,
):
    source_file = tmp_path / "reference.parquet"
    source_file.touch()

    dataframe = pd.DataFrame({
        "VendorID": [1, 2, 1, 2],
        "RatecodeID": [1, 2, 3, 4],
        "Airport_fee": [1.0, 2.0, 3.0, 4.0],
        "fare_amount": [10.0, 20.0, 30.0, 40.0],
    })

    read_mock = MagicMock(return_value=dataframe)

    monkeypatch.setattr(
        detector,
        "JANUARY_FILE",
        source_file,
    )
    monkeypatch.setattr(
        detector.pd,
        "read_parquet",
        read_mock,
    )

    result = detector.load_reference_data()

    read_mock.assert_called_once_with(source_file)

    assert len(result) == 4
    assert "vendor_id" in result.columns
    assert "rate_code_id" in result.columns
    assert "airport_fee" in result.columns

    assert "VendorID" not in result.columns
    assert "RatecodeID" not in result.columns
    assert "Airport_fee" not in result.columns


def test_load_reference_data_uses_configured_sample_size(
    monkeypatch,
    tmp_path,
):
    source_file = tmp_path / "reference.parquet"
    source_file.touch()

    dataframe = pd.DataFrame({
        "VendorID": range(100),
        "RatecodeID": range(100),
        "Airport_fee": range(100),
    })

    monkeypatch.setattr(
        detector,
        "JANUARY_FILE",
        source_file,
    )
    monkeypatch.setattr(
        detector.pd,
        "read_parquet",
        MagicMock(return_value=dataframe),
    )

    result = detector.load_reference_data()

    assert len(result) == 100


def test_load_reference_data_raises_when_file_missing(
    monkeypatch,
    tmp_path,
):
    missing_file = tmp_path / "missing.parquet"

    monkeypatch.setattr(
        detector,
        "JANUARY_FILE",
        missing_file,
    )

    with pytest.raises(FileNotFoundError):
        detector.load_reference_data()


# --------------------------------------------------
# CURRENT DATA LOADING
# --------------------------------------------------

def test_load_current_data_reads_requested_batch(monkeypatch):
    engine, connection, _ = make_engine()

    dataframe = make_feature_dataframe()

    read_mock = MagicMock(return_value=dataframe)

    monkeypatch.setattr(
        detector.pd,
        "read_sql",
        read_mock,
    )

    result = detector.load_current_data(engine, 3)

    assert len(result) == 3

    read_mock.assert_called_once()

    assert read_mock.call_args.kwargs["params"] == {
        "batch_id": 3
    }


def test_load_current_data_rejects_missing_features(monkeypatch):
    engine, _, _ = make_engine()

    dataframe = pd.DataFrame({
        "fare_amount": [10.0],
        "trip_distance": [2.0],
    })

    monkeypatch.setattr(
        detector.pd,
        "read_sql",
        MagicMock(return_value=dataframe),
    )

    with pytest.raises(
        ValueError,
        match="Missing database features",
    ):
        detector.load_current_data(engine, 3)


# --------------------------------------------------
# DATABASE RESULT STORAGE
# --------------------------------------------------

def test_save_results_does_nothing_for_empty_results():
    engine = MagicMock()

    detector.save_results(engine, [])

    engine.begin.assert_not_called()


def test_save_results_inserts_all_results():
    engine, _, transaction = make_engine()

    results = [
        make_drift_result(3),
        make_drift_result(3, "KS_PVALUE"),
    ]

    detector.save_results(engine, results)

    engine.begin.assert_called_once()

    assert transaction.execute.call_count == 2

    # First operation deletes previous results.
    delete_call = transaction.execute.call_args_list[0]
    assert "DELETE FROM drift_results" in str(delete_call.args[0])

    # Second operation inserts the new results.
    insert_call = transaction.execute.call_args_list[1]
    assert "INSERT INTO drift_results" in str(insert_call.args[0])

    assert insert_call.args[1] == results


# --------------------------------------------------
# DRIFT DETECTION ORCHESTRATION
# --------------------------------------------------

def test_detect_drift_processes_all_features(monkeypatch):
    reference = make_feature_dataframe()
    current = make_feature_dataframe()

    monkeypatch.setattr(
        detector,
        "calculate_psi",
        MagicMock(return_value=0.25),
    )
    monkeypatch.setattr(
        detector,
        "calculate_ks",
        MagicMock(return_value=0.01),
    )
    monkeypatch.setattr(
        detector,
        "calculate_wasserstein",
        MagicMock(return_value=0.15),
    )
    monkeypatch.setattr(
        detector,
        "calculate_js",
        MagicMock(return_value=0.20),
    )

    results = detector.detect_drift(
        reference,
        current,
        batch_id=3,
        current_period="2025-02-01",
    )

    expected_count = (
        len(detector.NUMERIC_FEATURES) * 3
        + len(detector.CATEGORICAL_FEATURES)
    )

    assert len(results) == expected_count

    assert all(
        result["current_batch_id"] == 3
        for result in results
    )

    assert all(
        result["current_period"] == "2025-02-01"
        for result in results
    )

    assert all(
        result["reference_period"] == detector.REFERENCE_PERIOD
        for result in results
    )

    assert all(
        result["reference_source"] == detector.REFERENCE_SOURCE
        for result in results
    )


def test_detect_drift_generates_expected_metric_types(monkeypatch):
    reference = make_feature_dataframe()
    current = make_feature_dataframe()

    monkeypatch.setattr(
        detector,
        "calculate_psi",
        MagicMock(return_value=0.25),
    )
    monkeypatch.setattr(
        detector,
        "calculate_ks",
        MagicMock(return_value=0.01),
    )
    monkeypatch.setattr(
        detector,
        "calculate_wasserstein",
        MagicMock(return_value=0.15),
    )
    monkeypatch.setattr(
        detector,
        "calculate_js",
        MagicMock(return_value=0.20),
    )

    results = detector.detect_drift(
        reference,
        current,
        batch_id=4,
        current_period="2025-03-01",
    )

    numeric_results = [
        result
        for result in results
        if result["metric_name"] != "JS_DIVERGENCE"
    ]

    categorical_results = [
        result
        for result in results
        if result["metric_name"] == "JS_DIVERGENCE"
    ]

    assert len(numeric_results) == len(detector.NUMERIC_FEATURES) * 3
    assert len(categorical_results) == len(detector.CATEGORICAL_FEATURES)

    assert {
        result["metric_name"]
        for result in numeric_results
    } == {
        "PSI",
        "KS_PVALUE",
        "WASSERSTEIN",
    }


def test_detect_drift_records_non_null_numeric_counts(monkeypatch):
    reference = make_feature_dataframe()
    current = make_feature_dataframe()

    reference.loc[0, "fare_amount"] = np.nan
    current.loc[0, "fare_amount"] = np.nan

    monkeypatch.setattr(
        detector,
        "calculate_psi",
        MagicMock(return_value=0.0),
    )
    monkeypatch.setattr(
        detector,
        "calculate_ks",
        MagicMock(return_value=1.0),
    )
    monkeypatch.setattr(
        detector,
        "calculate_wasserstein",
        MagicMock(return_value=0.0),
    )
    monkeypatch.setattr(
        detector,
        "calculate_js",
        MagicMock(return_value=0.0),
    )

    results = detector.detect_drift(
        reference,
        current,
        batch_id=3,
        current_period="2025-02-01",
    )

    fare_results = [
        result
        for result in results
        if result["feature_name"] == "fare_amount"
    ]

    assert len(fare_results) == 3

    for result in fare_results:
        assert result["reference_count"] == 2
        assert result["current_count"] == 2


# --------------------------------------------------
# MAIN WORKFLOW
# --------------------------------------------------

def test_main_processes_all_current_batches(
    monkeypatch,
    tmp_path,
):
    engine = MagicMock()

    reference = make_feature_dataframe()

    batch_3_data = make_feature_dataframe()
    batch_4_data = make_feature_dataframe()

    batch_3_results = [make_drift_result(3)]
    batch_4_results = [make_drift_result(4)]

    load_current_mock = MagicMock(
        side_effect=[batch_3_data, batch_4_data]
    )

    detect_mock = MagicMock(
        side_effect=[batch_3_results, batch_4_results]
    )

    save_mock = MagicMock()

    monkeypatch.setattr(
        detector,
        "get_engine",
        MagicMock(return_value=engine),
    )
    monkeypatch.setattr(
        detector,
        "load_reference_data",
        MagicMock(return_value=reference),
    )
    monkeypatch.setattr(
        detector,
        "load_current_data",
        load_current_mock,
    )
    monkeypatch.setattr(
        detector,
        "detect_drift",
        detect_mock,
    )
    monkeypatch.setattr(
        detector,
        "save_results",
        save_mock,
    )

    monkeypatch.chdir(tmp_path)

    detector.main()

    assert load_current_mock.call_count == 2
    assert detect_mock.call_count == 2
    assert save_mock.call_count == 2

    assert load_current_mock.call_args_list == [
        call(engine, 3),
        call(engine, 4),
    ]

    assert save_mock.call_args_list == [
        call(engine, batch_3_results),
        call(engine, batch_4_results),
    ]

    assert engine.dispose.called

    report_file = (
        tmp_path
        / "reports"
        / "drift"
        / "drift_results.csv"
    )

    assert report_file.exists()

    report = pd.read_csv(report_file)

    assert len(report) == 2
    assert set(report["current_batch_id"]) == {3, 4}


def test_main_disposes_engine_when_reference_loading_fails(
    monkeypatch,
):
    engine = MagicMock()

    monkeypatch.setattr(
        detector,
        "get_engine",
        MagicMock(return_value=engine),
    )
    monkeypatch.setattr(
        detector,
        "load_reference_data",
        MagicMock(
            side_effect=FileNotFoundError("Reference file missing")
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Reference file missing",
    ):
        detector.main()

    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_current_batch_loading_fails(
    monkeypatch,
):
    engine = MagicMock()

    monkeypatch.setattr(
        detector,
        "get_engine",
        MagicMock(return_value=engine),
    )
    monkeypatch.setattr(
        detector,
        "load_reference_data",
        MagicMock(return_value=make_feature_dataframe()),
    )
    monkeypatch.setattr(
        detector,
        "load_current_data",
        MagicMock(
            side_effect=RuntimeError("Database read failed")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Database read failed",
    ):
        detector.main()

    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_result_saving_fails(
    monkeypatch,
):
    engine = MagicMock()

    monkeypatch.setattr(
        detector,
        "get_engine",
        MagicMock(return_value=engine),
    )
    monkeypatch.setattr(
        detector,
        "load_reference_data",
        MagicMock(return_value=make_feature_dataframe()),
    )
    monkeypatch.setattr(
        detector,
        "load_current_data",
        MagicMock(return_value=make_feature_dataframe()),
    )
    monkeypatch.setattr(
        detector,
        "detect_drift",
        MagicMock(return_value=[make_drift_result(3)]),
    )
    monkeypatch.setattr(
        detector,
        "save_results",
        MagicMock(
            side_effect=RuntimeError("Database write failed")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Database write failed",
    ):
        detector.main()

    engine.dispose.assert_called_once()

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

    assert delete_call.args[1] == {
        "current_batch_id": 3,
        "current_period": "2025-02-01",
        "reference_period": "2025-01-01",
        "reference_source": "January 2025 random sample (seed=42)",
    }

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
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.quality import validation_engine as validator


def make_engine(batch=None):
    """Create a mocked SQLAlchemy engine and its connections."""
    engine = MagicMock()

    read_connection = MagicMock()
    transaction_connection = MagicMock()

    engine.connect.return_value.__enter__.return_value = read_connection
    engine.begin.return_value.__enter__.return_value = transaction_connection

    batch_result = MagicMock()
    batch_result.mappings.return_value.first.return_value = batch

    read_connection.execute.return_value = batch_result

    return engine, read_connection, transaction_connection


def make_valid_dataframe():
    """Return representative taxi records for all quality rules."""
    return pd.DataFrame({
        "pickup_datetime": pd.to_datetime([
            "2025-01-01 10:00:00",
            None,
            "2025-01-01 12:00:00",
        ]),
        "dropoff_datetime": pd.to_datetime([
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00",
            "2025-01-01 11:00:00",
        ]),
        "passenger_count": [1, 0, 2],
        "trip_distance": [2.5, -10, 3.0],
        "fare_amount": [15.0, -50.0, None],
        "extra": [1.0, -30.0, 0.0],
        "mta_tax": [0.5, -5.0, 0.0],
        "tip_amount": [1.0, -10.0, 0.0],
        "tolls_amount": [0.0, -2.0, 0.0],
        "improvement_surcharge": [0.5, -1.0, 0.0],
        "total_amount": [18.0, -100.0, 20.0],
        "congestion_surcharge": [0.0, -1.0, 0.0],
        "airport_fee": [0.0, -1.0, 0.0],
        "cbd_congestion_fee": [0.0, 0.0, 0.0],
    })

# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

def test_get_engine_raises_error_when_environment_is_missing(
    monkeypatch,
):
    for key in [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="Missing database settings"):
        validator.get_engine()


def test_get_engine_creates_postgresql_engine(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_PORT", "5433")

    expected_url = object()

    monkeypatch.setattr(
        validator.URL,
        "create",
        MagicMock(return_value=expected_url),
    )

    create_engine_mock = MagicMock(return_value="mock_engine")

    monkeypatch.setattr(
        validator,
        "create_engine",
        create_engine_mock,
    )

    result = validator.get_engine()

    assert result == "mock_engine"

    validator.URL.create.assert_called_once_with(
        "postgresql+psycopg2",
        username="test_user",
        password="test_password",
        host="localhost",
        port=5433,
        database="test_db",
    )

    create_engine_mock.assert_called_once_with(
        expected_url,
        pool_pre_ping=True,
    )


def test_get_engine_uses_default_postgresql_port(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "test_db")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.delenv("DB_PORT", raising=False)

    url_mock = MagicMock()
    create_engine_mock = MagicMock(return_value="mock_engine")

    monkeypatch.setattr(
        validator.URL,
        "create",
        url_mock,
    )
    monkeypatch.setattr(
        validator,
        "create_engine",
        create_engine_mock,
    )

    validator.get_engine()

    assert url_mock.call_args.kwargs["port"] == 5432


# ---------------------------------------------------------
# QUALITY RULE DEFINITIONS
# ---------------------------------------------------------

def test_all_eight_quality_rules_are_registered():
    expected_rules = {
        "missing_pickup_datetime",
        "missing_dropoff_datetime",
        "invalid_trip_duration",
        "negative_trip_distance",
        "negative_fare_amount",
        "negative_total_amount",
        "missing_fare_amount",
        "invalid_passenger_count",
    }

    actual_rules = {
        rule["name"] for rule in validator.RULES
    }

    assert actual_rules == expected_rules


@pytest.mark.parametrize(
    "rule_name, expected_failed_records",
    [
        ("missing_pickup_datetime", 1),
        ("missing_dropoff_datetime", 0),
        ("invalid_trip_duration", 1),
        ("negative_trip_distance", 1),
        ("negative_fare_amount", 1),
        ("negative_total_amount", 1),
        ("missing_fare_amount", 1),
        ("invalid_passenger_count", 1),
    ],
)
def test_quality_rule_detects_expected_records(
    rule_name,
    expected_failed_records,
):
    df = make_valid_dataframe()

    rule = next(
        rule for rule in validator.RULES
        if rule["name"] == rule_name
    )

    result = rule["condition"](df)

    assert result.sum() == expected_failed_records
    assert len(result) == len(df)


def test_numeric_rules_ignore_missing_values():
    df = pd.DataFrame({
        "trip_distance": [None, 2.5],
        "fare_amount": [None, 15.0],
        "total_amount": [None, 18.0],
        "passenger_count": [None, 1],
    })

    numeric_rules = [
        "negative_trip_distance",
        "negative_fare_amount",
        "negative_total_amount",
        "invalid_passenger_count",
    ]

    for rule_name in numeric_rules:
        rule = next(
            rule for rule in validator.RULES
            if rule["name"] == rule_name
        )

        assert rule["condition"](df).sum() == 0


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def test_main_executes_rules_and_saves_results(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(validator, "BATCH_ID", 77)

    engine, read_connection, transaction_connection = (
        make_engine({
            "batch_id": 77,
            "ingestion_status": "success",
        })
    )

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    df = make_valid_dataframe()

    monkeypatch.setattr(
        validator.pd,
        "read_sql",
        MagicMock(return_value=df),
    )

    validator.main()

    # Verify batch lookup.
    read_connection.execute.assert_called_once()


    # One DELETE + sixteen INSERT statements.
    assert transaction_connection.execute.call_count == 17

    # Capture all sixteen INSERT statements (skip the DELETE).
    insert_calls = (
        transaction_connection.execute.call_args_list[1:]
    )

    assert len(insert_calls) == 16

    inserted_rule_names = set()

    for call in insert_calls:
        params = call.args[1]

        assert params["batch_id"] == 77
        assert params["total_records"] == 3
        assert 0 <= params["failure_rate"] <= 100

        inserted_rule_names.add(params["rule_name"])

        details = json.loads(params["details"])
        assert "description" in details

    expected_advanced_rules = {
        "zero_trip_distance",
        "extreme_trip_distance",
        "excessive_passenger_count",
        "extremely_long_trip",
        "extremely_short_trip",
        "unusually_high_fare",
        "invalid_fare_relationship",
        "future_pickup_datetime",
    }

    expected_base_rules = {
        rule["name"] for rule in validator.RULES
    }

    assert inserted_rule_names == (
        expected_base_rules | expected_advanced_rules
    )

    delete_call = transaction_connection.execute.call_args_list[0]

    delete_params = delete_call.args[1]

    assert delete_params["batch_id"] == 77
    assert len(delete_params["rule_names"]) == 16



    # Verify report output.
    output = capsys.readouterr().out

    assert "Quality report for batch 77" in output
    assert "Records checked: 3" in output
    assert "Rules executed: 16" in output
    assert "Results saved to quality_results." in output

    engine.dispose.assert_called_once()


def test_main_rejects_missing_batch(monkeypatch):
    engine, _, _ = make_engine(batch=None)

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(validator, "BATCH_ID", 99)

    with pytest.raises(
        ValueError,
        match="Batch 99 does not exist",
    ):
        validator.main()

    engine.dispose.assert_called_once()


def test_main_rejects_unsuccessful_batch(monkeypatch):
    engine, _, _ = make_engine({
        "batch_id": 77,
        "ingestion_status": "failed",
    })

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(validator, "BATCH_ID", 77)

    with pytest.raises(
        ValueError,
        match="not successfully ingested",
    ):
        validator.main()

    engine.dispose.assert_called_once()


def test_main_rejects_empty_dataframe(monkeypatch):
    engine, _, transaction_connection = make_engine({
        "batch_id": 77,
        "ingestion_status": "success",
    })

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(
        validator.pd,
        "read_sql",
        MagicMock(return_value=pd.DataFrame()),
    )

    monkeypatch.setattr(validator, "BATCH_ID", 77)

    with pytest.raises(
        ValueError,
        match="has no trip records",
    ):
        validator.main()

    transaction_connection.execute.assert_not_called()
    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_database_read_fails(
    monkeypatch,
):
    engine, _, _ = make_engine({
        "batch_id": 77,
        "ingestion_status": "success",
    })

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(
        validator.pd,
        "read_sql",
        MagicMock(side_effect=RuntimeError("Database read failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="Database read failed",
    ):
        validator.main()

    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_rule_execution_fails(
    monkeypatch,
):
    engine, _, _ = make_engine({
        "batch_id": 77,
        "ingestion_status": "success",
    })

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(
        validator.pd,
        "read_sql",
        MagicMock(return_value=make_valid_dataframe()),
    )

    monkeypatch.setattr(
        validator,
        "RULES",
        [{
            "name": "failing_rule",
            "severity": "high",
            "condition": lambda df: 1 / 0,
            "description": "Test failure",
        }],
    )

    with pytest.raises(ZeroDivisionError):
        validator.main()

    engine.dispose.assert_called_once()


def test_main_disposes_engine_when_transaction_fails(
    monkeypatch,
):
    engine, _, transaction_connection = make_engine({
        "batch_id": 77,
        "ingestion_status": "success",
    })

    monkeypatch.setattr(
        validator,
        "get_engine",
        lambda: engine,
    )

    monkeypatch.setattr(
        validator.pd,
        "read_sql",
        MagicMock(return_value=make_valid_dataframe()),
    )

    transaction_connection.execute.side_effect = RuntimeError(
        "Database transaction failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Database transaction failed",
    ):
        validator.main()

    engine.dispose.assert_called_once()

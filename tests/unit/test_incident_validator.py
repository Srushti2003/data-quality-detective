import json
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from src.quality import incident_validator as validator


# --------------------------------------------------
# Test data helpers
# --------------------------------------------------

def make_row():
    return pd.Series({
        "trip_id": 101,
        "pickup_datetime": pd.Timestamp("2025-01-01 10:00:00"),
        "dropoff_datetime": pd.Timestamp("2025-01-01 10:30:00"),
        "passenger_count": 2,
        "trip_distance": 5.5,
        "fare_amount": 20.0,
        "total_amount": 25.0,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 1.0,
        "airport_fee": 0.0,
        "pickup_location_id": 10,
        "dropoff_location_id": 20,
    })


def make_dataframe():
    return pd.DataFrame([
        {
            "trip_id": 101,
            "pickup_datetime": pd.Timestamp("2025-01-01 10:00:00"),
            "dropoff_datetime": pd.Timestamp("2025-01-01 10:30:00"),
            "passenger_count": 2,
            "trip_distance": -5.0,
            "fare_amount": 10.0,
            "total_amount": 15.0,
            "extra": 1.0,
            "mta_tax": 0.5,
            "tip_amount": 2.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.3,
            "congestion_surcharge": 1.0,
            "airport_fee": 0.0,
            "pickup_location_id": 10,
            "dropoff_location_id": 20,
        },
        {
            "trip_id": 102,
            "pickup_datetime": pd.Timestamp("2025-01-01 10:00:00"),
            "dropoff_datetime": pd.Timestamp("2025-01-01 10:30:00"),
            "passenger_count": 2,
            "trip_distance": -5.0,
            "fare_amount": 10.0,
            "total_amount": 15.0,
            "extra": 1.0,
            "mta_tax": 0.5,
            "tip_amount": 2.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.3,
            "congestion_surcharge": 1.0,
            "airport_fee": 0.0,
            "pickup_location_id": 10,
            "dropoff_location_id": 20,
        },
        {
            "trip_id": 103,
            "pickup_datetime": pd.Timestamp("2025-01-01 11:00:00"),
            "dropoff_datetime": pd.Timestamp("2025-01-01 11:45:00"),
            "passenger_count": 1,
            "trip_distance": 8.0,
            "fare_amount": 200.0,
            "total_amount": 220.0,
            "extra": 2.0,
            "mta_tax": 0.5,
            "tip_amount": 10.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.3,
            "congestion_surcharge": 1.0,
            "airport_fee": 0.0,
            "pickup_location_id": 11,
            "dropoff_location_id": 21,
        },
    ])


# --------------------------------------------------
# get_observed_value() tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "rule_name,column",
    [
        ("missing_pickup_datetime", "pickup_datetime"),
        ("missing_dropoff_datetime", "dropoff_datetime"),
        ("invalid_passenger_count", "passenger_count"),
        ("negative_fare_amount", "fare_amount"),
        ("negative_trip_distance", "trip_distance"),
        ("negative_total_amount", "total_amount"),
        ("zero_trip_distance", "trip_distance"),
    ],
)
def test_get_observed_value_returns_mapped_column(
    rule_name, column
):
    row = make_row()

    result = validator.get_observed_value(row, rule_name)

    assert result == str(row[column])


@pytest.mark.parametrize(
    "rule_name,column",
    [
        ("missing_pickup_datetime", "pickup_datetime"),
        ("missing_dropoff_datetime", "dropoff_datetime"),
        ("invalid_passenger_count", "passenger_count"),
        ("negative_fare_amount", "fare_amount"),
        ("negative_trip_distance", "trip_distance"),
        ("negative_total_amount", "total_amount"),
        ("zero_trip_distance", "trip_distance"),
    ],
)
def test_get_observed_value_returns_null_for_missing_values(
    rule_name, column
):
    row = make_row()
    row[column] = None

    result = validator.get_observed_value(row, rule_name)

    assert result == "NULL"


def test_get_observed_value_returns_trip_duration():
    row = make_row()

    result = validator.get_observed_value(
        row,
        "extremely_short_trip",
    )

    assert result == "1800.0"


def test_get_observed_value_returns_fare_relationship_json():
    row = make_row()

    result = validator.get_observed_value(
        row,
        "invalid_fare_relationship",
    )

    parsed = json.loads(result)

    assert parsed["fare_amount"] == 20.0
    assert parsed["extra"] == 1.0
    assert parsed["total_amount"] == 25.0
    assert parsed["tip_amount"] == 2.0


def test_get_observed_value_converts_missing_fare_values_to_null():
    row = make_row()
    row["fare_amount"] = None

    result = validator.get_observed_value(
        row,
        "invalid_fare_relationship",
    )

    parsed = json.loads(result)

    assert parsed["fare_amount"] is None


def test_get_observed_value_returns_duplicate_candidate_json():
    row = make_row()

    result = validator.get_observed_value(
        row,
        "duplicate_trip_candidate",
    )

    parsed = json.loads(result)

    assert parsed["passenger_count"] == "2"
    assert parsed["trip_distance"] == "5.5"
    assert parsed["fare_amount"] == "20.0"
    assert parsed["pickup_location_id"] == "10"


def test_get_observed_value_handles_missing_duplicate_values():
    row = make_row()
    row["trip_distance"] = None

    result = validator.get_observed_value(
        row,
        "duplicate_trip_candidate",
    )

    parsed = json.loads(result)

    assert parsed["trip_distance"] is None


def test_get_observed_value_returns_na_for_unknown_rule():
    row = make_row()

    result = validator.get_observed_value(
        row,
        "unknown_rule",
    )

    assert result == "N/A"


# --------------------------------------------------
# Configuration tests
# --------------------------------------------------

def test_data_columns_contains_expected_trip_fields():
    assert "trip_id" in validator.DATA_COLUMNS
    assert "pickup_datetime" in validator.DATA_COLUMNS
    assert "dropoff_datetime" in validator.DATA_COLUMNS
    assert "trip_distance" in validator.DATA_COLUMNS
    assert "fare_amount" in validator.DATA_COLUMNS
    assert "total_amount" in validator.DATA_COLUMNS


# --------------------------------------------------
# get_engine() test
# --------------------------------------------------

def test_get_engine_uses_environment_configuration():
    environment = {
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "DB_HOST": "localhost",
        "DB_PORT": "5433",
        "DB_NAME": "test_database",
    }

    with patch.dict("os.environ", environment):
        with patch.object(
            validator,
            "create_engine",
        ) as mock_create_engine:

            validator.get_engine()

            mock_create_engine.assert_called_once()

            args, kwargs = mock_create_engine.call_args

            assert args[0].drivername == "postgresql+psycopg2"
            assert args[0].username == "test_user"
            assert args[0].password == "test_password"
            assert args[0].host == "localhost"
            assert args[0].port == 5433
            assert args[0].database == "test_database"
            assert kwargs["pool_pre_ping"] is True


# --------------------------------------------------
# main() successful execution test
# --------------------------------------------------

def test_main_executes_rules_and_saves_results():
    fake_engine = MagicMock()
    fake_connection = MagicMock()
    fake_transaction = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_engine.begin.return_value.__enter__.return_value = (
        fake_transaction
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        "success"
    )

    df = make_dataframe()

    rules = [
        {
            "name": "negative_trip_distance",
            "severity": "high",
            "condition": lambda data: data["trip_distance"] < 0,
            "description": "Trip distance cannot be negative.",
        },
        {
            "name": "high_fare",
            "severity": "medium",
            "condition": pd.Series(
                [False, False, True],
                index=df.index,
            ),
            "description": "Fare exceeds the expected limit.",
        },
    ]

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ), patch.object(
        validator.pd,
        "read_sql",
        return_value=df,
    ), patch.object(
        validator,
        "RULES",
        rules,
    ), patch.object(
        validator,
        "get_advanced_rules",
        return_value=[],
    ):
        validator.main()

    # Two deletes and two inserts should be executed.
    assert fake_transaction.execute.call_count == 4

    calls = fake_transaction.execute.call_args_list

    sql_statements = [
        str(call.args[0])
        for call in calls
    ]

    assert any(
        "DELETE FROM quality_issues" in sql
        for sql in sql_statements
    )

    assert any(
        "DELETE FROM quality_results" in sql
        for sql in sql_statements
    )

    assert any(
        "INSERT INTO quality_results" in sql
        for sql in sql_statements
    )

    assert any(
        "INSERT INTO quality_issues" in sql
        for sql in sql_statements
    )

    # Two custom rules + one automatically added duplicate rule.
    result_insert_call = next(
        call
        for call in calls
        if "INSERT INTO quality_results" in str(call.args[0])
    )

    inserted_results = result_insert_call.args[1]

    assert len(inserted_results) == 3

    rule_names = {
        result["rule_name"]
        for result in inserted_results
    }

    assert rule_names == {
        "negative_trip_distance",
        "high_fare",
        "duplicate_trip_candidate",
    }

    negative_distance_result = next(
        result
        for result in inserted_results
        if result["rule_name"] == "negative_trip_distance"
    )

    assert negative_distance_result["failed_records"] == 2
    assert negative_distance_result["total_records"] == 3
    assert negative_distance_result["failure_rate"] == pytest.approx(
        200 / 3
    )

    issue_insert_call = next(
        call
        for call in calls
        if "INSERT INTO quality_issues" in str(call.args[0])
    )

    inserted_issues = issue_insert_call.args[1]

    # 2 negative distances + 1 high fare + 1 duplicate candidate.
    assert len(inserted_issues) == 4

    assert all(
        issue["batch_id"] == validator.BATCH_ID
        for issue in inserted_issues
    )

    assert all(
        "observed_value" in issue
        for issue in inserted_issues
    )

    assert all(
        "expected_condition" in issue
        for issue in inserted_issues
    )

    duplicate_issue = next(
        issue
        for issue in inserted_issues
        if issue["rule_name"] == "duplicate_trip_candidate"
    )

    duplicate_metadata = json.loads(
        duplicate_issue["issue_metadata"]
    )

    assert duplicate_metadata["rule"] == "duplicate_trip_candidate"

    fake_engine.dispose.assert_called_once()


# --------------------------------------------------
# Batch validation tests
# --------------------------------------------------

def test_main_rejects_missing_or_unsuccessful_batch():
    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        "failed"
    )

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ):
        with pytest.raises(
            ValueError,
            match="missing or not successful",
        ):
            validator.main()

    fake_engine.begin.assert_not_called()
    fake_engine.dispose.assert_called_once()


def test_main_rejects_missing_batch_record():
    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        None
    )

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ):
        with pytest.raises(
            ValueError,
            match="missing or not successful",
        ):
            validator.main()

    fake_engine.dispose.assert_called_once()


def test_main_rejects_empty_dataframe():
    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        "success"
    )

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ), patch.object(
        validator.pd,
        "read_sql",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(
            ValueError,
            match="No records found",
        ):
            validator.main()

    fake_engine.begin.assert_not_called()
    fake_engine.dispose.assert_called_once()


# --------------------------------------------------
# Resource cleanup tests
# --------------------------------------------------

def test_main_disposes_engine_when_database_read_fails():
    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        "success"
    )

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ), patch.object(
        validator.pd,
        "read_sql",
        side_effect=RuntimeError("Database read failed"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Database read failed",
        ):
            validator.main()

    fake_engine.dispose.assert_called_once()


def test_main_disposes_engine_when_rule_execution_fails():
    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.connect.return_value.__enter__.return_value = (
        fake_connection
    )

    fake_connection.execute.return_value.scalar_one_or_none.return_value = (
        "success"
    )

    df = make_dataframe()

    failing_rules = [
        {
            "name": "failing_rule",
            "severity": "high",
            "condition": lambda data: (_ for _ in ()).throw(
                RuntimeError("Rule execution failed")
            ),
            "description": "Test rule failure.",
        }
    ]

    with patch.object(
        validator,
        "get_engine",
        return_value=fake_engine,
    ), patch.object(
        validator.pd,
        "read_sql",
        return_value=df,
    ), patch.object(
        validator,
        "RULES",
        failing_rules,
    ), patch.object(
        validator,
        "get_advanced_rules",
        return_value=[],
    ):
        with pytest.raises(
            RuntimeError,
            match="Rule execution failed",
        ):
            validator.main()

    fake_engine.begin.assert_not_called()
    fake_engine.dispose.assert_called_once()
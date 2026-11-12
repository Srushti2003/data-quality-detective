import json
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.quality import validation_engine as validator


@pytest.fixture
def validation_batch(test_engine):
    """Create an isolated batch with real PostgreSQL trip records."""

    with test_engine.begin() as connection:
        batch_id = connection.execute(
            text("""
                INSERT INTO data_batches (
                    dataset_name,
                    source_file,
                    batch_period,
                    row_count,
                    ingestion_status,
                    is_test_batch
                )
                VALUES (
                    'validation_integration_test',
                    'pytest_validation',
                    '2025-02-01',
                    3,
                    'success',
                    TRUE
                )
                RETURNING batch_id
            """)
        ).scalar_one()

        trips = [
            {
                "batch_id": batch_id,
                "source_row_number": 1,
                "pickup_datetime": datetime(2025, 2, 1, 10, 0),
                "dropoff_datetime": datetime(2025, 2, 1, 10, 30),
                "passenger_count": 1,
                "trip_distance": 2.0,
                "fare_amount": 20.0,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tip_amount": 2.0,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "total_amount": 27.0,
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
            },
            {
                "batch_id": batch_id,
                "source_row_number": 2,
                "pickup_datetime": datetime(2025, 2, 1, 11, 0),
                "dropoff_datetime": datetime(2025, 2, 1, 10, 0),
                "passenger_count": 0,
                "trip_distance": -1.0,
                "fare_amount": -10.0,
                "extra": 0.0,
                "mta_tax": 0.0,
                "tip_amount": 0.0,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.0,
                "total_amount": -10.0,
                "congestion_surcharge": 0.0,
                "airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
            },
            {
                "batch_id": batch_id,
                "source_row_number": 3,
                "pickup_datetime": datetime(2099, 1, 1, 10, 0),
                "dropoff_datetime": datetime(2099, 1, 1, 10, 0, 30),
                "passenger_count": 7,
                "trip_distance": 150.0,
                "fare_amount": 600.0,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tip_amount": 2.0,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "total_amount": 700.0,
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
            },
        ]

        connection.execute(
            text("""
                INSERT INTO taxi_trips (
                    batch_id,
                    source_row_number,
                    pickup_datetime,
                    dropoff_datetime,
                    passenger_count,
                    trip_distance,
                    fare_amount,
                    extra,
                    mta_tax,
                    tip_amount,
                    tolls_amount,
                    improvement_surcharge,
                    total_amount,
                    congestion_surcharge,
                    airport_fee,
                    cbd_congestion_fee
                )
                VALUES (
                    :batch_id,
                    :source_row_number,
                    :pickup_datetime,
                    :dropoff_datetime,
                    :passenger_count,
                    :trip_distance,
                    :fare_amount,
                    :extra,
                    :mta_tax,
                    :tip_amount,
                    :tolls_amount,
                    :improvement_surcharge,
                    :total_amount,
                    :congestion_surcharge,
                    :airport_fee,
                    :cbd_congestion_fee
                )
            """),
            trips,
        )

    yield batch_id

    with test_engine.begin() as connection:
        connection.execute(
            text("""
                DELETE FROM quality_results
                WHERE batch_id = :batch_id
            """),
            {"batch_id": batch_id},
        )

        connection.execute(
            text("""
                DELETE FROM taxi_trips
                WHERE batch_id = :batch_id
            """),
            {"batch_id": batch_id},
        )

        connection.execute(
            text("""
                DELETE FROM data_batches
                WHERE batch_id = :batch_id
                  AND is_test_batch = TRUE
            """),
            {"batch_id": batch_id},
        )


@pytest.fixture
def run_validation(monkeypatch, test_engine, validation_batch):
    """Run the real validation engine against the isolated test database."""

    monkeypatch.setattr(validator, "get_engine", lambda: test_engine)
    monkeypatch.setattr(validator, "BATCH_ID", validation_batch)

    return validator.main


def test_validation_engine_persists_all_rules(
    test_engine,
    validation_batch,
    run_validation,
):
    """Verify all 16 rules execute and persist in PostgreSQL."""

    run_validation()

    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    rule_name,
                    severity,
                    failed_records,
                    total_records,
                    failure_rate,
                    details,
                    jsonb_typeof(details) AS details_type
                FROM quality_results
                WHERE batch_id = :batch_id
                ORDER BY rule_name
            """),
            {"batch_id": validation_batch},
        ).mappings().all()

    assert len(rows) == 16
    assert {row["total_records"] for row in rows} == {3}

    results = {
        row["rule_name"]: row
        for row in rows
    }

    assert results["negative_trip_distance"]["failed_records"] == 1
    assert results["negative_fare_amount"]["failed_records"] == 1
    assert results["negative_total_amount"]["failed_records"] == 1
    assert results["invalid_passenger_count"]["failed_records"] == 1
    assert results["extreme_trip_distance"]["failed_records"] == 1
    assert results["excessive_passenger_count"]["failed_records"] == 1
    assert results["unusually_high_fare"]["failed_records"] == 1
    assert results["future_pickup_datetime"]["failed_records"] == 1
    assert results["extremely_short_trip"]["failed_records"] == 1
    assert results["invalid_fare_relationship"]["failed_records"] == 1

    for row in rows:
        assert 0 <= row["failure_rate"] <= 100
        assert row["details_type"] == "object"
        assert isinstance(row["details"], dict)
        assert "description" in row["details"]


def test_validation_engine_replaces_previous_results(
    test_engine,
    validation_batch,
    run_validation,
):
    """Verify rerunning validation replaces existing rule results."""

    run_validation()

    with test_engine.begin() as connection:
        connection.execute(
            text("""
                UPDATE taxi_trips
                SET trip_distance = 50
                WHERE batch_id = :batch_id
                  AND source_row_number = 3
            """),
            {"batch_id": validation_batch},
        )

    run_validation()

    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT rule_name, failed_records
                FROM quality_results
                WHERE batch_id = :batch_id
            """),
            {"batch_id": validation_batch},
        ).mappings().all()

    results = {
        row["rule_name"]: row["failed_records"]
        for row in rows
    }

    assert len(rows) == 16
    assert results["extreme_trip_distance"] == 0


def test_validation_engine_rolls_back_failed_persistence(
    test_engine,
    validation_batch,
    run_validation,
    monkeypatch,
):
    """Verify failed inserts preserve previously saved results."""

    run_validation()

    with test_engine.connect() as connection:
        original_rows = connection.execute(
            text("""
                SELECT
                    result_id,
                    rule_name,
                    failed_records,
                    severity
                FROM quality_results
                WHERE batch_id = :batch_id
                ORDER BY rule_name
            """),
            {"batch_id": validation_batch},
        ).mappings().all()

    original_snapshot = [dict(row) for row in original_rows]
    assert len(original_snapshot) == 16

    original_get_advanced_rules = validator.get_advanced_rules

    def invalid_advanced_rules(df):
        rules = original_get_advanced_rules(df)
        rules[0]["severity"] = "x" * 21
        return rules

    monkeypatch.setattr(
        validator,
        "get_advanced_rules",
        invalid_advanced_rules,
    )

    with pytest.raises(DBAPIError):
        run_validation()

    with test_engine.connect() as connection:
        current_rows = connection.execute(
            text("""
                SELECT
                    result_id,
                    rule_name,
                    failed_records,
                    severity
                FROM quality_results
                WHERE batch_id = :batch_id
                ORDER BY rule_name
            """),
            {"batch_id": validation_batch},
        ).mappings().all()

    assert [dict(row) for row in current_rows] == original_snapshot

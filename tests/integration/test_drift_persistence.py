
from datetime import date

from sqlalchemy import text

from src.drift.drift_detector import save_results

import pytest

from sqlalchemy.exc import DBAPIError


def create_drift_result(batch_id, **overrides):
    """Build a valid drift result for PostgreSQL persistence tests."""

    result = {
        "reference_period": date(2025, 1, 1),
        "current_period": date(2025, 2, 1),
        "reference_source": "January 2025",
        "current_batch_id": batch_id,
        "feature_name": "fare_amount",
        "metric_name": "PSI",
        "metric_value": 0.35,
        "threshold": 0.20,
        "drift_detected": True,
        "severity": "high",
        "reference_count": 1000,
        "current_count": 1000,
    }

    result.update(overrides)
    return result


def test_save_results_inserts_records(test_engine, test_batch):
    """Verify that save_results persists multiple records in PostgreSQL."""

    results = [
        create_drift_result(test_batch),
        create_drift_result(
            test_batch,
            feature_name="trip_distance",
            metric_name="KS",
            metric_value=0.42,
            severity="medium",
        ),
    ]

    save_results(test_engine, results)

    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    current_batch_id,
                    feature_name,
                    metric_name,
                    metric_value,
                    drift_detected,
                    severity
                FROM drift_results
                WHERE current_batch_id = :batch_id
                ORDER BY feature_name
            """),
            {"batch_id": test_batch},
        ).mappings().all()

    assert len(rows) == 2

    assert rows[0]["feature_name"] == "fare_amount"
    assert rows[0]["metric_name"] == "PSI"
    assert rows[0]["metric_value"] == 0.35
    assert rows[0]["drift_detected"] is True
    assert rows[0]["severity"] == "high"

    assert rows[1]["feature_name"] == "trip_distance"
    assert rows[1]["metric_name"] == "KS"
    assert rows[1]["metric_value"] == 0.42
    assert rows[1]["severity"] == "medium"

    assert all(
        row["current_batch_id"] == test_batch
        for row in rows
    )




def test_save_results_replaces_existing_results(test_engine, test_batch):
    """Verify repeated executions replace previous results."""

    first_results = [
        create_drift_result(test_batch),
        create_drift_result(
            test_batch,
            feature_name="trip_distance",
        ),
    ]

    save_results(test_engine, first_results)

    updated_results = [
        create_drift_result(
            test_batch,
            metric_value=0.55,
            severity="critical",
        ),
        create_drift_result(
            test_batch,
            feature_name="trip_distance",
            metric_value=0.65,
            severity="high",
        ),
    ]

    # Execute the same detection run again.
    save_results(test_engine, updated_results)

    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    feature_name,
                    metric_value,
                    severity
                FROM drift_results
                WHERE current_batch_id = :batch_id
                ORDER BY feature_name
            """),
            {"batch_id": test_batch},
        ).mappings().all()

    # No duplicate records should exist.
    assert len(rows) == 2

    # The updated values should replace the original values.
    assert rows[0]["feature_name"] == "fare_amount"
    assert rows[0]["metric_value"] == 0.55
    assert rows[0]["severity"] == "critical"

    assert rows[1]["feature_name"] == "trip_distance"
    assert rows[1]["metric_value"] == 0.65
    assert rows[1]["severity"] == "high"





def test_save_results_does_not_modify_other_batches(
    test_engine,
    test_batch,
    second_test_batch,
):
    """Verify saving one batch does not affect another batch."""

    batch_a_results = [
        create_drift_result(
            test_batch,
            metric_value=0.30,
        )
    ]

    batch_b_results = [
        create_drift_result(
            second_test_batch,
            metric_value=0.45,
            feature_name="trip_distance",
        )
    ]

    save_results(test_engine, batch_a_results)
    save_results(test_engine, batch_b_results)

    # Rerun Batch A with updated results.
    updated_batch_a_results = [
        create_drift_result(
            test_batch,
            metric_value=0.60,
            severity="critical",
        )
    ]

    save_results(test_engine, updated_batch_a_results)

    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    current_batch_id,
                    feature_name,
                    metric_value,
                    severity
                FROM drift_results
                WHERE current_batch_id IN (
                    :batch_a,
                    :batch_b
                )
                ORDER BY current_batch_id
            """),
            {
                "batch_a": test_batch,
                "batch_b": second_test_batch,
            },
        ).mappings().all()

    # Each batch should still have exactly one record.
    assert len(rows) == 2

    batch_a_row = next(
        row for row in rows
        if row["current_batch_id"] == test_batch
    )

    batch_b_row = next(
        row for row in rows
        if row["current_batch_id"] == second_test_batch
    )

    # Batch A should contain its updated result.
    assert batch_a_row["metric_value"] == 0.60
    assert batch_a_row["severity"] == "critical"

    # Batch B should remain completely unchanged.
    assert batch_b_row["feature_name"] == "trip_distance"
    assert batch_b_row["metric_value"] == 0.45
    assert batch_b_row["severity"] == "high"




def test_save_results_rolls_back_when_insert_fails(
    test_engine,
    test_batch,
):
    """Verify failed inserts restore previously persisted results."""

    original_results = [
        create_drift_result(
            test_batch,
            metric_value=0.35,
            severity="high",
        ),
        create_drift_result(
            test_batch,
            feature_name="trip_distance",
            metric_value=0.42,
            severity="medium",
        ),
    ]

    # First, persist valid records.
    save_results(test_engine, original_results)

    # Create an invalid record exceeding VARCHAR(100).
    invalid_results = [
        create_drift_result(
            test_batch,
            metric_value=0.90,
            severity="critical",
        ),
        create_drift_result(
            test_batch,
            feature_name="x" * 101,
            metric_value=0.95,
            severity="critical",
        ),
    ]

    # PostgreSQL should reject the insert.
    with pytest.raises(DBAPIError):
        save_results(test_engine, invalid_results)

    # Verify that the original records were restored.
    with test_engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    feature_name,
                    metric_value,
                    severity
                FROM drift_results
                WHERE current_batch_id = :batch_id
                ORDER BY feature_name
            """),
            {"batch_id": test_batch},
        ).mappings().all()

    assert len(rows) == 2

    assert rows[0]["feature_name"] == "fare_amount"
    assert rows[0]["metric_value"] == 0.35
    assert rows[0]["severity"] == "high"

    assert rows[1]["feature_name"] == "trip_distance"
    assert rows[1]["metric_value"] == 0.42
    assert rows[1]["severity"] == "medium"

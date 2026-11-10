import pandas as pd
import pytest

from sqlalchemy import text
from pandas.errors import DatabaseError

from src.ingestion import incident_loader


def create_source_files(tmp_path, invalid_column_name=False):
    source = tmp_path / "test_corrupted.parquet"
    labels_source = tmp_path / "test_corrupted_labels.parquet"

    trips = pd.DataFrame({
        "VendorID": [1, 2, 1],
        "tpep_pickup_datetime": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-02 11:00:00",
            "2025-01-03 12:00:00",
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2025-01-01 10:20:00",
            "2025-01-02 11:20:00",
            "2025-01-03 12:20:00",
        ]),
        "passenger_count": [1, 2, 1],
        "trip_distance": [1.5, 2.5, -5.0],
        "fare_amount": [-10.0, 20.0, 30.0],
        "total_amount": [12.0, 22.0, 32.0],
    })

    column_name = "x" * 101 if invalid_column_name else "fare_amount"

    labels = pd.DataFrame({
        "source_row_number": [2, 0],
        "incident_type": [
            "negative_trip_distance",
            "negative_fare_amount",
        ],
        "column_name": [
            "trip_distance",
            column_name,
        ],
    })

    trips.to_parquet(source, index=False)
    labels.to_parquet(labels_source, index=False)

    return source, labels_source


def cleanup_batch(test_engine, batch_id):
    with test_engine.begin() as connection:
        connection.execute(
            text("""
                DELETE FROM injected_incidents
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
            """),
            {"batch_id": batch_id},
        )


def test_incident_loader_inserts_trips_and_labels(
    test_engine,
    tmp_path,
    monkeypatch,
):
    source, labels_source = create_source_files(tmp_path)

    monkeypatch.setattr(incident_loader, "SOURCE", source)
    monkeypatch.setattr(incident_loader, "LABELS_SOURCE", labels_source)
    monkeypatch.setattr(
        incident_loader,
        "get_engine",
        lambda: test_engine,
    )

    batch_id = incident_loader.main()

    try:
        with test_engine.connect() as connection:
            trips = connection.execute(
                text("""
                    SELECT source_row_number, trip_id
                    FROM taxi_trips
                    WHERE batch_id = :batch_id
                    ORDER BY source_row_number
                """),
                {"batch_id": batch_id},
            ).mappings().all()

            labels = connection.execute(
                text("""
                    SELECT
                        i.source_row_number,
                        i.incident_type,
                        i.column_name,
                        i.trip_id
                    FROM (
                        SELECT
                            ti.batch_id,
                            ti.trip_id,
                            ti.incident_type,
                            ti.column_name,
                            tt.source_row_number
                        FROM injected_incidents ti
                        JOIN taxi_trips tt
                            ON tt.trip_id = ti.trip_id
                    ) i
                    WHERE i.batch_id = :batch_id
                    ORDER BY i.source_row_number
                """),
                {"batch_id": batch_id},
            ).mappings().all()

            status = connection.execute(
                text("""
                    SELECT ingestion_status
                    FROM data_batches
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": batch_id},
            ).scalar_one()

        assert len(trips) == 3
        assert len(labels) == 2
        assert status == "success"

        trip_ids = {
            row["source_row_number"]: row["trip_id"]
            for row in trips
        }

        assert labels[0]["source_row_number"] == 0
        assert labels[0]["incident_type"] == "negative_fare_amount"
        assert labels[0]["trip_id"] == trip_ids[0]

        assert labels[1]["source_row_number"] == 2
        assert labels[1]["incident_type"] == "negative_trip_distance"
        assert labels[1]["trip_id"] == trip_ids[2]

    finally:
        cleanup_batch(test_engine, batch_id)


def test_incident_loader_rolls_back_failed_label_insert(
    test_engine,
    tmp_path,
    monkeypatch,
):
    source, labels_source = create_source_files(
        tmp_path,
        invalid_column_name=True,
    )

    monkeypatch.setattr(incident_loader, "SOURCE", source)
    monkeypatch.setattr(incident_loader, "LABELS_SOURCE", labels_source)
    monkeypatch.setattr(
        incident_loader,
        "get_engine",
        lambda: test_engine,
    )

    with pytest.raises(DatabaseError):
        incident_loader.main()

    with test_engine.connect() as connection:
        batch_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM data_batches
                WHERE source_file = :source_file
                  AND dataset_name = 'NYC Yellow Taxi - Incident Test'
            """),
            {"source_file": source.name},
        ).scalar_one()

        trip_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM taxi_trips tt
                JOIN data_batches db
                    ON db.batch_id = tt.batch_id
                WHERE db.source_file = :source_file
                  AND db.dataset_name = 'NYC Yellow Taxi - Incident Test'
            """),
            {"source_file": source.name},
        ).scalar_one()

    assert batch_count == 0
    assert trip_count == 0

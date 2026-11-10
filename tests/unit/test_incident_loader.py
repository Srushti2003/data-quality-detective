import pandas as pd
import pytest

from src.ingestion import incident_loader


def create_source_files(tmp_path, labels):
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
        "trip_distance": [1.5, 2.5, -5.0],
        "fare_amount": [-10.0, 20.0, 30.0],
    })

    trips.to_parquet(source, index=False)
    labels.to_parquet(labels_source, index=False)

    return source, labels_source


def load_with_files(monkeypatch, source, labels_source):
    monkeypatch.setattr(incident_loader, "SOURCE", source)
    monkeypatch.setattr(incident_loader, "LABELS_SOURCE", labels_source)

    return incident_loader.load_source_data()


def test_load_source_data_returns_trips_and_labels(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0],
        "incident_type": ["negative_fare_amount"],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    trips, loaded_labels = load_with_files(
        monkeypatch, source, labels_source
    )

    assert len(trips) == 3
    assert len(loaded_labels) == 1
    assert trips["source_row_number"].tolist() == [0, 1, 2]


def test_load_source_data_rejects_missing_source_file(
    tmp_path, monkeypatch
):
    source = tmp_path / "missing.parquet"
    labels_source = tmp_path / "missing_labels.parquet"

    monkeypatch.setattr(incident_loader, "SOURCE", source)
    monkeypatch.setattr(incident_loader, "LABELS_SOURCE", labels_source)

    with pytest.raises(FileNotFoundError):
        incident_loader.load_source_data()


def test_load_source_data_rejects_missing_labels_file(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0],
        "incident_type": ["negative_fare_amount"],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)
    labels_source.unlink()

    monkeypatch.setattr(incident_loader, "SOURCE", source)
    monkeypatch.setattr(incident_loader, "LABELS_SOURCE", labels_source)

    with pytest.raises(FileNotFoundError):
        incident_loader.load_source_data()


def test_load_source_data_rejects_missing_label_columns(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0],
        "incident_type": ["negative_fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="Missing label columns"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_null_source_row_number(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [None],
        "incident_type": ["negative_fare_amount"],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="null source row numbers"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_non_integer_source_row_number(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [1.5],
        "incident_type": ["negative_fare_amount"],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="must be integers"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_out_of_range_source_row_number(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [10],
        "incident_type": ["negative_fare_amount"],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="outside the dataset"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_null_incident_type(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0],
        "incident_type": [None],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="null incident types"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_empty_incident_type(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0],
        "incident_type": ["   "],
        "column_name": ["fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="empty incident types"):
        load_with_files(monkeypatch, source, labels_source)


def test_load_source_data_rejects_duplicate_incident_labels(
    tmp_path, monkeypatch
):
    labels = pd.DataFrame({
        "source_row_number": [0, 0],
        "incident_type": [
            "negative_fare_amount",
            "negative_fare_amount",
        ],
        "column_name": ["fare_amount", "fare_amount"],
    })

    source, labels_source = create_source_files(tmp_path, labels)

    with pytest.raises(ValueError, match="Duplicate source row"):
        load_with_files(monkeypatch, source, labels_source)

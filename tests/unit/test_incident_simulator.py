import pandas as pd
import pytest

from src.quality import incident_simulator as simulator


@pytest.fixture
def sample_taxi_data():
    """Create 600 rows of sample taxi data."""

    return pd.DataFrame({
        "fare_amount": [20.0] * 600,
        "trip_distance": [5.0] * 600,
        "passenger_count": [2] * 600,
        "tpep_pickup_datetime": pd.date_range(
            "2025-01-01",
            periods=600,
            freq="min"
        ),
        "total_amount": [25.0] * 600,
    })


def test_simulate_incidents_reads_source(
    monkeypatch, sample_taxi_data, tmp_path
):
    read_calls = []

    def fake_read_parquet(source):
        read_calls.append(source)
        return sample_taxi_data.copy()

    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        fake_read_parquet
    )

    output = tmp_path / "processed" / "corrupted.parquet"

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: None
    )

    simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=output
    )

    assert read_calls == [simulator.SOURCE]


def test_simulate_incidents_injects_all_quality_problems(
    monkeypatch, sample_taxi_data, tmp_path
):
    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda source: sample_taxi_data.copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: None
    )

    output = tmp_path / "corrupted.parquet"

    result = simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=output
    )

    assert (result.loc[:99, "fare_amount"] == -50).all()

    assert (result.loc[100:199, "trip_distance"] == -10).all()

    assert (result.loc[200:299, "passenger_count"] == 0).all()

    assert result.loc[300:399, "tpep_pickup_datetime"].isna().all()

    assert (result.loc[400:499, "total_amount"] == -100).all()


def test_simulate_incidents_duplicates_first_100_records(
    monkeypatch, sample_taxi_data, tmp_path
):
    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda source: sample_taxi_data.copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: None
    )

    result = simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=tmp_path / "corrupted.parquet"
    )

    assert len(result) == 700

    pd.testing.assert_frame_equal(
        result.iloc[:100].reset_index(drop=True),
        result.iloc[600:700].reset_index(drop=True)
    )


def test_simulate_incidents_limits_source_to_10000_rows(
    monkeypatch, tmp_path
):
    large_data = pd.DataFrame({
        "fare_amount": [20.0] * 12_000,
        "trip_distance": [5.0] * 12_000,
        "passenger_count": [2] * 12_000,
        "tpep_pickup_datetime": pd.date_range(
            "2025-01-01",
            periods=12_000,
            freq="min"
        ),
        "total_amount": [25.0] * 12_000,
    })

    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda source: large_data.copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: None
    )

    result = simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=tmp_path / "corrupted.parquet"
    )

    assert len(result) == 10_100


def test_simulate_incidents_creates_output_directory(
    monkeypatch, sample_taxi_data, tmp_path
):
    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda source: sample_taxi_data.copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: None
    )

    output = tmp_path / "new_folder" / "corrupted.parquet"

    assert not output.parent.exists()

    simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=output
    )

    assert output.parent.exists()


def test_simulate_incidents_saves_without_dataframe_index(
    monkeypatch, sample_taxi_data, tmp_path
):
    save_calls = []

    def fake_to_parquet(self, path, index):
        save_calls.append({
            "path": path,
            "index": index,
            "rows": len(self),
            "data": self.copy()
        })

    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        lambda source: sample_taxi_data.copy()
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        fake_to_parquet
    )

    output = tmp_path / "corrupted.parquet"

    simulator.simulate_incidents(
        source=simulator.SOURCE,
        output=output
    )

    labels_output = output.with_name(
        f"{output.stem}_labels.parquet"
    )

    assert len(save_calls) == 2

    assert save_calls[0]["path"] == output
    assert save_calls[0]["index"] is False
    assert save_calls[0]["rows"] == 700

    assert save_calls[1]["path"] == labels_output
    assert save_calls[1]["index"] is False
    assert save_calls[1]["rows"] == 700

    labels = save_calls[1]["data"]

    assert set(labels["incident_type"].unique()) == {
        "negative_fare_amount",
        "negative_trip_distance",
        "zero_passenger_count",
        "missing_pickup_datetime",
        "negative_total_amount",
        "duplicate_trip",
    }

    assert labels["incident_type"].value_counts().to_dict() == {
        "negative_fare_amount": 200,
        "negative_trip_distance": 100,
        "zero_passenger_count": 100,
        "missing_pickup_datetime": 100,
        "negative_total_amount": 100,
        "duplicate_trip": 100,
    }

    assert labels["source_row_number"].min() == 0
    assert labels["source_row_number"].max() == 699


def test_simulate_incidents_propagates_read_error(
    monkeypatch, tmp_path
):
    def fake_read_parquet(source):
        raise FileNotFoundError("Source dataset not found")

    monkeypatch.setattr(
        simulator.pd,
        "read_parquet",
        fake_read_parquet
    )

    output = tmp_path / "corrupted.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="Source dataset not found"
    ):
        simulator.simulate_incidents(
            source=simulator.SOURCE,
            output=output
        )

    assert not output.exists()
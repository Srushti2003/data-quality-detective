from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/nyc_taxi/yellow_tripdata_2025-01.parquet")
OUTPUT = Path("data/processed/nyc_taxi_corrupted.parquet")


def simulate_incidents(
    source: Path = SOURCE,
    output: Path = OUTPUT
) -> pd.DataFrame:
    """Generate corrupted taxi data and reproducible ground-truth labels."""

    df = pd.read_parquet(source).head(10_000).copy()
    df = df.reset_index(drop=True)

    original_rows = len(df)

    # Track the original row ranges affected by each incident.
    fare_end = min(100, original_rows)
    distance_start = 100
    distance_end = min(200, original_rows)
    passenger_start = 200
    passenger_end = min(300, original_rows)
    pickup_start = 300
    pickup_end = min(400, original_rows)
    total_start = 400
    total_end = min(500, original_rows)

    # Introduce controlled data-quality problems.
    if fare_end > 0:
        df.loc[:fare_end - 1, "fare_amount"] = -50

    if distance_end > distance_start:
        df.loc[distance_start:distance_end - 1, "trip_distance"] = -10

    if passenger_end > passenger_start:
        df.loc[passenger_start:passenger_end - 1, "passenger_count"] = 0

    if pickup_end > pickup_start:
        df.loc[pickup_start:pickup_end - 1, "tpep_pickup_datetime"] = None

    if total_end > total_start:
        df.loc[total_start:total_end - 1, "total_amount"] = -100

    # Duplicate the first 100 records.
    duplicate_count = min(100, original_rows)

    duplicates = df.iloc[:duplicate_count].copy()

    df = pd.concat(
        [df, duplicates],
        ignore_index=True
    )

    # source_row_number identifies the exact row in the output dataset.
    labels = []

    def add_labels(start, end, incident_type, column_name):
        for row_number in range(start, end):
            labels.append({
                "source_row_number": row_number,
                "incident_type": incident_type,
                "column_name": column_name,
            })

    # Original corrupted records.
    add_labels(0, fare_end, "negative_fare_amount", "fare_amount")

    add_labels(
        distance_start,
        distance_end,
        "negative_trip_distance",
        "trip_distance"
    )

    add_labels(
        passenger_start,
        passenger_end,
        "zero_passenger_count",
        "passenger_count"
    )

    add_labels(
        pickup_start,
        pickup_end,
        "missing_pickup_datetime",
        "pickup_datetime"
    )

    add_labels(
        total_start,
        total_end,
        "negative_total_amount",
        "total_amount"
    )

    # The appended duplicate rows are also negative-fare records.
    duplicate_start = original_rows
    duplicate_end = original_rows + duplicate_count

    add_labels(
        duplicate_start,
        duplicate_end,
        "negative_fare_amount",
        "fare_amount"
    )

    add_labels(
        duplicate_start,
        duplicate_end,
        "duplicate_trip",
        "multiple_columns"
    )

    labels_df = pd.DataFrame(
        labels,
        columns=[
            "source_row_number",
            "incident_type",
            "column_name",
        ],
    )

    labels_output = output.with_name(
        f"{output.stem}_labels.parquet"
    )

    # Create output directory and save both files.
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output, index=False)
    labels_df.to_parquet(labels_output, index=False)

    print("Incident simulation completed.")
    print(f"Original rows: {original_rows:,}")
    print(f"Corrupted rows: {len(df):,}")
    print(f"Ground-truth labels: {len(labels_df):,}")
    print(f"Corrupted dataset: {output}")
    print(f"Ground-truth dataset: {labels_output}")

    return df


def main():
    simulate_incidents()


if __name__ == "__main__":
    main()

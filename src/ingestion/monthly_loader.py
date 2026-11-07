
from pathlib import Path
import os
import re

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

DATA_DIR = Path("data/raw/nyc_taxi")
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42
CHUNK_SIZE = 500

MONTHLY_FILES = [
    "yellow_tripdata_2025-02.parquet",
    "yellow_tripdata_2025-03.parquet",
]

COLUMN_MAP = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "RatecodeID": "rate_code_id",
    "store_and_fwd_flag": "store_and_fwd_flag",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
    "payment_type": "payment_type",
    "fare_amount": "fare_amount",
    "extra": "extra",
    "mta_tax": "mta_tax",
    "tip_amount": "tip_amount",
    "tolls_amount": "tolls_amount",
    "improvement_surcharge": "improvement_surcharge",
    "total_amount": "total_amount",
    "congestion_surcharge": "congestion_surcharge",
    "Airport_fee": "airport_fee",
    "cbd_congestion_fee": "cbd_congestion_fee",
}

DB_COLUMNS = list(COLUMN_MAP.values())

REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
}


def get_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )

    return create_engine(url, pool_pre_ping=True)


def get_month_start(filename):
    match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})", filename)

    if not match:
        raise ValueError(f"Cannot determine month from filename: {filename}")

    year, month = map(int, match.groups())
    return pd.Timestamp(year=year, month=month, day=1).date()


def prepare_sample(source_file):
    file_path = DATA_DIR / source_file

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    print(f"\nReading: {source_file}")

    df = pd.read_parquet(file_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required source columns: {sorted(missing)}"
        )

    print(f"Source records: {len(df):,}")

    sample_count = min(SAMPLE_SIZE, len(df))

    # Reproducible random sample.
    sample = df.sample(
        n=sample_count,
        random_state=RANDOM_STATE,
    ).copy()

    sample = sample.rename(columns=COLUMN_MAP)

    # Keep all available mapped database columns in a fixed order.
    available_columns = [
        column for column in DB_COLUMNS
        if column in sample.columns
    ]

    trips = sample[available_columns].copy()

    for column in ["pickup_datetime", "dropoff_datetime"]:
        trips[column] = pd.to_datetime(
            trips[column],
            errors="coerce",
        )

    # Convert pandas NaN/NaT values into Python None for SQL insertion.
    trips = trips.astype(object).where(pd.notna(trips), None)

    print(f"Sample records: {len(trips):,}")

    return trips


def load_month(source_file, engine):
    batch_period = get_month_start(source_file)

    trips = prepare_sample(source_file)

    with engine.begin() as connection:

        # Prevent accidental duplicate ingestion.
        existing_batch = connection.execute(
            text("""
                SELECT batch_id, ingestion_status
                FROM data_batches
                WHERE source_file = :source_file
                  AND batch_period = :batch_period
                  AND is_test_batch = FALSE
            """),
            {
                "source_file": source_file,
                "batch_period": batch_period,
            },
        ).first()

        if existing_batch:
            raise ValueError(
                f"{source_file} already exists as batch "
                f"{existing_batch.batch_id} "
                f"with status {existing_batch.ingestion_status}. "
                "Skipping to prevent duplicate ingestion."
            )

        # Register the batch.
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
                    'NYC Yellow Taxi',
                    :source_file,
                    :batch_period,
                    :row_count,
                    'loading',
                    FALSE
                )
                RETURNING batch_id
            """),
            {
                "source_file": source_file,
                "batch_period": batch_period,
                "row_count": len(trips),
            },
        ).scalar_one()

        trips.insert(0, "batch_id", batch_id)

        # Insert the sampled taxi records.
        trips.to_sql(
            "taxi_trips",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=CHUNK_SIZE,
            method="multi",
        )

        # Mark the batch successful.
        connection.execute(
            text("""
                UPDATE data_batches
                SET ingestion_status = 'success'
                WHERE batch_id = :batch_id
            """),
            {"batch_id": batch_id},
        )

    print("\nMonthly ingestion successful!")
    print(f"Batch ID: {batch_id}")
    print(f"Source file: {source_file}")
    print(f"Batch period: {batch_period}")
    print(f"Records loaded: {len(trips):,}")


def main():
    engine = get_engine()

    try:
        for source_file in MONTHLY_FILES:
            load_month(source_file, engine)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

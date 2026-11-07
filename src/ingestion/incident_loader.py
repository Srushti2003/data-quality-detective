
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

SOURCE = Path("data/processed/nyc_taxi_corrupted.parquet")
CHUNK_SIZE = 1000

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
    "airport_fee": "airport_fee",
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


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_parquet(SOURCE)
    df = df.rename(columns=COLUMN_MAP)

    required = {
        "pickup_datetime",
        "dropoff_datetime",
        "trip_distance",
        "fare_amount",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    db_columns = [
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "rate_code_id",
        "store_and_fwd_flag",
        "pickup_location_id",
        "dropoff_location_id",
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee",
    ]

    trips = df[
        [col for col in db_columns if col in df.columns]
    ].copy()

    for column in ["pickup_datetime", "dropoff_datetime"]:
        trips[column] = pd.to_datetime(
            trips[column], errors="coerce"
        )

    trips = trips.astype(object).where(pd.notna(trips), None)

    pickup_dates = pd.to_datetime(
        df["pickup_datetime"], errors="coerce"
    ).dropna()

    batch_period = (
        pickup_dates.min().date().replace(day=1)
        if not pickup_dates.empty
        else None
    )

    engine = get_engine()
    batch_id = None

    try:
        with engine.begin() as connection:
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
                        'NYC Yellow Taxi - Incident Test',
                        :source_file,
                        :batch_period,
                        :row_count,
                        'loading',
                        TRUE
                    )
                    RETURNING batch_id
                """),
                {
                    "source_file": SOURCE.name,
                    "batch_period": batch_period,
                    "row_count": len(trips),
                },
            ).scalar_one()

            trips.insert(0, "batch_id", batch_id)

            trips.to_sql(
                "taxi_trips",
                con=connection,
                if_exists="append",
                index=False,
                chunksize=CHUNK_SIZE,
                method="multi",
            )

            connection.execute(
                text("""
                    UPDATE data_batches
                    SET ingestion_status = 'success'
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": batch_id},
            )

        print("Incident dataset ingestion successful!")
        print(f"Batch ID: {batch_id}")
        print(f"Records loaded: {len(trips):,}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

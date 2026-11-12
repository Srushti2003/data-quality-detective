
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

DATA_DIR = Path("data/raw/nyc_taxi")
MAX_ROWS = 10_000
CHUNK_SIZE = 1_000

COLUMN_MAP = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "lpep_pickup_datetime": "pickup_datetime",
    "lpep_dropoff_datetime": "dropoff_datetime",
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
    "cbd_congestion_fee": "cbd_congestion_fee",
}


def get_engine():
    required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError(
            f"Missing database settings in .env: {missing}"
        )

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
    files = sorted(DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {DATA_DIR.resolve()}"
        )

    file_path = files[0]
    print(f"Reading: {file_path.name}")

    df = pd.read_parquet(file_path)
    df = df.head(MAX_ROWS).copy()

    # Standardize column names using the actual source schema.
    df = df.rename(columns=COLUMN_MAP)

    required_columns = {
        "pickup_datetime",
        "dropoff_datetime",
        "trip_distance",
        "fare_amount",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep only fields supported by the current database schema.
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
        "cbd_congestion_fee",
    ]

    available_columns = [
        column for column in db_columns if column in df.columns
    ]
    trips = df[available_columns].copy()

    for column in ["pickup_datetime", "dropoff_datetime"]:
        trips[column] = pd.to_datetime(
            trips[column], errors="coerce"
        )

    # Let PostgreSQL represent missing values as NULL.
    trips = trips.astype(object).where(pd.notna(trips), None)

    pickup_values = pd.to_datetime(
        df["pickup_datetime"], errors="coerce"
    ).dropna()

    batch_period = (
        pickup_values.min().date().replace(day=1)
        if not pickup_values.empty
        else None
    )

    engine = get_engine()
    batch_id = None

    try:
        with engine.begin() as connection:
            result = connection.execute(
                text("""
                    INSERT INTO data_batches (
                        dataset_name,
                        source_file,
                        batch_period,
                        row_count,
                        ingestion_status
                    )
                    VALUES (
                        :dataset_name,
                        :source_file,
                        :batch_period,
                        :row_count,
                        'loading'
                    )
                    RETURNING batch_id
                """),
                {
                    "dataset_name": "NYC Yellow Taxi",
                    "source_file": file_path.name,
                    "batch_period": batch_period,
                    "row_count": len(trips),
                },
            )
            batch_id = result.scalar_one()

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

        print("\nIngestion successful!")
        print(f"Batch ID: {batch_id}")
        print(f"Source file: {file_path.name}")
        print(f"Rows loaded: {len(trips):,}")

    except Exception as exc:
        if batch_id is not None:
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text("""
                            UPDATE data_batches
                            SET ingestion_status = 'failed',
                                error_message = :error
                            WHERE batch_id = :batch_id
                        """),
                        {
                            "batch_id": batch_id,
                            "error": str(exc)[:4000],
                        },
                    )
            except Exception as log_error:
                print(f"Could not update batch status: {log_error}")

        raise

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

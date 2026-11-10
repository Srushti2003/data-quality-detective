from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

SOURCE = Path("data/processed/nyc_taxi_corrupted.parquet")
LABELS_SOURCE = SOURCE.with_name(f"{SOURCE.stem}_labels.parquet")
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
    "cbd_congestion_fee": "cbd_congestion_fee",
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


def load_source_data():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    if not LABELS_SOURCE.is_file():
        raise FileNotFoundError(
            f"Incident labels file not found: {LABELS_SOURCE}"
        )

    df = pd.read_parquet(SOURCE).reset_index(drop=True)
    df.insert(0, "source_row_number", range(len(df)))
    df = df.rename(columns=COLUMN_MAP)

    required_trip_columns = {
        "pickup_datetime",
        "dropoff_datetime",
        "trip_distance",
        "fare_amount",
    }

    missing = required_trip_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing trip columns: {sorted(missing)}")

    labels = pd.read_parquet(LABELS_SOURCE)

    required_label_columns = {
        "source_row_number",
        "incident_type",
        "column_name",
    }

    missing = required_label_columns - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    if labels["source_row_number"].isna().any():
        raise ValueError("Incident labels contain null source row numbers.")

    labels["source_row_number"] = pd.to_numeric(
        labels["source_row_number"], errors="raise"
    )

    if not labels["source_row_number"].map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Incident source row numbers must be integers.")

    labels["source_row_number"] = labels[
        "source_row_number"
    ].astype("int64")

    if labels["incident_type"].isna().any():
        raise ValueError("Incident labels contain null incident types.")

    if labels["incident_type"].astype(str).str.strip().eq("").any():
        raise ValueError("Incident labels contain empty incident types.")

    if labels.duplicated(
        subset=["source_row_number", "incident_type"]
    ).any():
        raise ValueError(
            "Duplicate source row and incident type combinations found."
        )

    invalid_rows = ~labels["source_row_number"].between(
        0, len(df) - 1
    )

    if invalid_rows.any():
        raise ValueError(
            "Incident labels reference source rows outside the dataset."
        )

    return df, labels


def main():
    df, labels = load_source_data()

    db_columns = [
        "source_row_number",
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

    trips = df[
        [column for column in db_columns if column in df.columns]
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

            trip_rows = connection.execute(
                text("""
                    SELECT source_row_number, trip_id
                    FROM taxi_trips
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": batch_id},
            ).mappings().all()

            trip_map = pd.DataFrame(
                trip_rows,
                columns=["source_row_number", "trip_id"],
            )

            if trip_map[["source_row_number", "trip_id"]].isna().any().any():
                raise RuntimeError(
                    "Trip ID mapping contains null source rows or trip IDs."
                )

            if len(trip_map) != len(trips):
                raise RuntimeError(
                    "Could not map every ingested source row to a trip ID."
                )

            if trip_map["source_row_number"].duplicated().any():
                raise RuntimeError(
                    "Multiple trip IDs found for a source row."
                )

            mapped_labels = labels.merge(
                trip_map,
                on="source_row_number",
                how="left",
                validate="many_to_one",
                indicator=True,
            )

            if not mapped_labels["_merge"].eq("both").all():
                raise RuntimeError(
                    "Some incident labels could not be mapped to trip IDs."
                )

            incident_records = mapped_labels[
                ["trip_id", "incident_type", "column_name"]
            ].copy()

            if not incident_records.empty:
                incident_records.insert(0, "batch_id", batch_id)

                incident_records.to_sql(
                    "injected_incidents",
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
        print(f"Trip records loaded: {len(trips):,}")
        print(f"Incident labels loaded: {len(labels):,}")

        return batch_id

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

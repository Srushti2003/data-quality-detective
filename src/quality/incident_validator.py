
import os
import json

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from src.quality.validation_engine import RULES
from src.quality.quality_rules import get_advanced_rules

load_dotenv()

BATCH_ID = int(os.getenv("QUALITY_BATCH_ID", "2"))

DATA_COLUMNS = [
    "trip_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "pickup_location_id",
    "dropoff_location_id",
]


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


def get_observed_value(row, rule_name):
    """
    Return the actual value or relevant values
    associated with a quality rule.
    """

    column_mapping = {
        "missing_pickup_datetime": "pickup_datetime",
        "missing_dropoff_datetime": "dropoff_datetime",
        "invalid_passenger_count": "passenger_count",
        "negative_fare_amount": "fare_amount",
        "negative_trip_distance": "trip_distance",
        "negative_total_amount": "total_amount",
        "zero_trip_distance": "trip_distance",
    }

    if rule_name in column_mapping:
        value = row[column_mapping[rule_name]]

        if pd.isna(value):
            return "NULL"

        return str(value)

    if rule_name == "extremely_short_trip":
        duration = (
            row["dropoff_datetime"] -
            row["pickup_datetime"]
        ).total_seconds()

        return str(duration)

    if rule_name == "invalid_fare_relationship":
        relevant_columns = [
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "total_amount",
        ]

        values = {
            col: (
                None if pd.isna(row[col])
                else float(row[col])
            )
            for col in relevant_columns
        }

        return json.dumps(values)

    if rule_name == "duplicate_trip_candidate":
        relevant_columns = [
            "pickup_datetime",
            "dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "pickup_location_id",
            "dropoff_location_id",
        ]

        values = {
            col: (
                None if pd.isna(row[col])
                else str(row[col])
            )
            for col in relevant_columns
        }

        return json.dumps(values)

    return "N/A"



def main():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            batch = connection.execute(
                text("""
                    SELECT ingestion_status
                    FROM data_batches
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": BATCH_ID},
            ).scalar_one_or_none()

            if batch != "success":
                raise ValueError(
                    f"Batch {BATCH_ID} is missing or not successful."
                )

            df = pd.read_sql(
                text("""
                    SELECT *
                    FROM taxi_trips
                    WHERE batch_id = :batch_id
                    ORDER BY trip_id
                """),
                connection,
                params={"batch_id": BATCH_ID},
            )

        if df.empty:
            raise ValueError("No records found for this batch.")

        rules = RULES + get_advanced_rules(df)

        # Detect duplicate combinations of trip attributes.
        duplicate_columns = [
            "pickup_datetime",
            "dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "pickup_location_id",
            "dropoff_location_id",
        ]

        rules.append({
            "name": "duplicate_trip_candidate",
            "severity": "high",
            "condition": df.duplicated(
                subset=duplicate_columns,
                keep="first",
            ),
            "description": "Trip matches a previous trip on key attributes.",
        })

        issue_records = []
        result_records = []

        for rule in rules:
            condition = rule["condition"]

            # Support both callable rules and precomputed Pandas masks
            if callable(condition):
                mask = condition(df)
            else:
                mask = condition

            mask = mask.fillna(False).astype(bool)

            failed = df.loc[mask]

            count = len(failed)
            total = len(df)

            result_records.append({
                "batch_id": BATCH_ID,
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "failed_records": count,
                "total_records": total,
                "failure_rate": count / total * 100,
                "details": {
                    "description": rule["description"]
                },
            })

            for _, row in failed.iterrows():
                issue_records.append({
                    "batch_id": BATCH_ID,
                    "trip_id": int(row["trip_id"]),
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "observed_value": get_observed_value(
                    row,
                    rule["name"]
                    ),
                    "expected_condition": rule["description"],
                    "issue_metadata": {
                        "source_batch": BATCH_ID,
                        "rule": rule["name"],
                    },
                })

        # Save results and issues atomically.
        with engine.begin() as connection:
            connection.execute(
                text("""
                    DELETE FROM quality_issues
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": BATCH_ID},
            )

            connection.execute(
                text("""
                    DELETE FROM quality_results
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": BATCH_ID},
            )

            if result_records:
                connection.execute(
                    text("""
                        INSERT INTO quality_results (
                            batch_id,
                            rule_name,
                            severity,
                            failed_records,
                            total_records,
                            failure_rate,
                            details
                        )
                        VALUES (
                            :batch_id,
                            :rule_name,
                            :severity,
                            :failed_records,
                            :total_records,
                            :failure_rate,
                            CAST(:details AS JSONB)
                        )
                    """),
                    [
                        {
                            **r,
                            "details": json.dumps(r["details"]),
                        }
                        for r in result_records
                    ],
                )

            if issue_records:
                connection.execute(
                    text("""
                        INSERT INTO quality_issues (
                            batch_id,
                            trip_id,
                            rule_name,
                            severity,
                            observed_value,
                            expected_condition,
                            issue_metadata
                        )
                        VALUES (
                            :batch_id,
                            :trip_id,
                            :rule_name,
                            :severity,
                            :observed_value,
                            :expected_condition,
                            CAST(:issue_metadata AS JSONB)
                        )
                    """),
                    [
                        {
                            **r,
                            "issue_metadata": json.dumps(
                                r["issue_metadata"]
                            ),
                        }
                        for r in issue_records
                    ],
                )

        print(f"\nIncident validation report — Batch {BATCH_ID}")
        print("=" * 75)

        for result in result_records:
            print(
                f"{result['rule_name']:<32}"
                f"Failed: {result['failed_records']:>5}"
                f" | {result['failure_rate']:>6.2f}%"
            )

        print("=" * 75)
        print(f"Records checked: {len(df):,}")
        print(f"Rules executed: {len(rules)}")
        print(f"Individual issues saved: {len(issue_records):,}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

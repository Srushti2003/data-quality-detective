
import json
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from src.quality.quality_rules import get_advanced_rules

load_dotenv()

BATCH_ID = int(os.getenv("QUALITY_BATCH_ID", "1"))

RULES = [
    {
        "name": "missing_pickup_datetime",
        "severity": "critical",
        "condition": lambda df: df["pickup_datetime"].isna(),
        "description": "Pickup timestamp is missing.",
    },
    {
        "name": "missing_dropoff_datetime",
        "severity": "critical",
        "condition": lambda df: df["dropoff_datetime"].isna(),
        "description": "Drop-off timestamp is missing.",
    },
    {
        "name": "invalid_trip_duration",
        "severity": "high",
        "condition": lambda df: (
            df["pickup_datetime"].notna()
            & df["dropoff_datetime"].notna()
            & (
                df["dropoff_datetime"]
                < df["pickup_datetime"]
            )
        ),
        "description": "Drop-off occurs before pickup.",
    },
    {
        "name": "negative_trip_distance",
        "severity": "high",
        "condition": lambda df: (
            df["trip_distance"].notna()
            & (df["trip_distance"] < 0)
        ),
        "description": "Trip distance is negative.",
    },
    {
        "name": "negative_fare_amount",
        "severity": "high",
        "condition": lambda df: (
            df["fare_amount"].notna()
            & (df["fare_amount"] < 0)
        ),
        "description": "Fare amount is negative.",
    },
    {
        "name": "negative_total_amount",
        "severity": "high",
        "condition": lambda df: (
            df["total_amount"].notna()
            & (df["total_amount"] < 0)
        ),
        "description": "Total trip amount is negative.",
    },
    {
        "name": "missing_fare_amount",
        "severity": "medium",
        "condition": lambda df: df["fare_amount"].isna(),
        "description": "Fare amount is missing.",
    },
    {
        "name": "invalid_passenger_count",
        "severity": "medium",
        "condition": lambda df: (
            df["passenger_count"].notna()
            & (df["passenger_count"] <= 0)
        ),
        "description": "Passenger count is zero or negative.",
    },
]


def get_engine():
    required = [
        "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"
    ]
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
    engine = get_engine()

    try:
        with engine.connect() as connection:
            batch = connection.execute(
                text("""
                    SELECT batch_id, ingestion_status
                    FROM data_batches
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": BATCH_ID},
            ).mappings().first()

            if not batch:
                raise ValueError(
                    f"Batch {BATCH_ID} does not exist."
                )

            if batch["ingestion_status"] != "success":
                raise ValueError(
                    f"Batch {BATCH_ID} is not successfully ingested."
                )

            df = pd.read_sql(
                text("""
                    SELECT
                    pickup_datetime,
                    dropoff_datetime,
                    passenger_count,
                    trip_distance,
                    fare_amount,
                    extra,
                    mta_tax,
                    tip_amount,
                    tolls_amount,
                    improvement_surcharge,
                    total_amount,
                    congestion_surcharge,
                    airport_fee,
                    cbd_congestion_fee
                FROM taxi_trips
                WHERE batch_id = :batch_id
                """),
                connection,
                params={"batch_id": BATCH_ID},
            )

        if df.empty:
            raise ValueError(
                f"Batch {BATCH_ID} has no trip records."
            )

        results = []
        total_records = len(df)

        # Existing rules use callable conditions.
        rules_to_execute = [
            {
                **rule,
                "failed_mask": rule["condition"](df),
            }
            for rule in RULES
        ]

        # Advanced rules return precomputed Boolean Series.
        rules_to_execute.extend(
            {
                **rule,
                "failed_mask": rule["condition"],
            }
            for rule in get_advanced_rules(df)
        )

        for rule in rules_to_execute:
            failed_records = int(rule["failed_mask"].sum())

            results.append({
                "batch_id": BATCH_ID,
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "failed_records": failed_records,
                "total_records": total_records,
                "failure_rate": (
                    failed_records / total_records * 100
                ),
                "details": {
                    "description": rule["description"]
                },
            })

        # Replace previous results for these rules and batch,
        # making repeated runs safe from duplicate result rows.
        with engine.begin() as connection:
            connection.execute(
                text("""
                    DELETE FROM quality_results
                    WHERE batch_id = :batch_id
                      AND rule_name = ANY(:rule_names)
                """),
                {
                    "batch_id": BATCH_ID,
                    "rule_names": [r["rule_name"] for r in results],
                },
            )

            for result in results:
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
                    {
                        **result,
                        "details": json.dumps(result["details"]),
                    },
                )

        print(f"\nQuality report for batch {BATCH_ID}")
        print("-" * 75)

        for result in results:
            print(
                f"{result['rule_name']:<30} "
                f"Failed: {result['failed_records']:>5} "
                f"({result['failure_rate']:>6.2f}%) "
                f"[{result['severity']}]"
            )

        print("-" * 75)
        print(f"Records checked: {len(df):,}")
        print(f"Rules executed: {len(results)}")
        print("Results saved to quality_results.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

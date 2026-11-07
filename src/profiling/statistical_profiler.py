
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from scipy.stats import skew, kurtosis

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

BATCH_ID = int(os.getenv("PROFILE_BATCH_ID", "1"))

NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
]

DATETIME_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
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


def numeric_profile(series, total_records):
    values = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    if values.empty:
        return {
            "mean_value": None,
            "median_value": None,
            "std_deviation": None,
            "min_value": None,
            "max_value": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
        }

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return {
        "mean_value": float(values.mean()),
        "median_value": float(values.median()),
        "std_deviation": float(values.std()),
        "min_value": float(values.min()),
        "max_value": float(values.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "skewness": (
            float(skew(values))
            if len(values) >= 3 else None
        ),
        "kurtosis": (
            float(kurtosis(values))
            if len(values) >= 4 else None
        ),
    }


def profile_column(df, column):
    series = df[column]
    total = len(series)
    missing = int(series.isna().sum())

    result = {
        "column_name": column,
        "data_type": str(series.dtype),
        "total_records": total,
        "missing_records": missing,
        "missing_percentage": (
            missing / total * 100 if total else 0
        ),
        "distinct_values": int(series.nunique()),
    }

    if column in NUMERIC_COLUMNS:
        result.update(numeric_profile(series, total))

    elif column in DATETIME_COLUMNS:
        dates = pd.to_datetime(
            series, errors="coerce"
        ).dropna()

        result.update({
            "mean_value": None,
            "median_value": None,
            "std_deviation": None,
            "min_value": (
                float(dates.astype("int64").min() / 1e9)
                if not dates.empty else None
            ),
            "max_value": (
                float(dates.astype("int64").max() / 1e9)
                if not dates.empty else None
            ),
            "q1": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
        })

    else:
        result.update({
            "mean_value": None,
            "median_value": None,
            "std_deviation": None,
            "min_value": None,
            "max_value": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
        })

    return result


def main():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            df = pd.read_sql(
                text("""
                    SELECT *
                    FROM taxi_trips
                    WHERE batch_id = :batch_id
                """),
                connection,
                params={"batch_id": BATCH_ID},
            )

        if df.empty:
            raise ValueError(
                f"No data found for batch {BATCH_ID}"
            )

        columns = NUMERIC_COLUMNS + DATETIME_COLUMNS

        missing_columns = [
            column for column in columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        profiles = [
            profile_column(df, column)
            for column in columns
        ]

        with engine.begin() as connection:
            connection.execute(
                text("""
                    DELETE FROM data_profiles
                    WHERE batch_id = :batch_id
                """),
                {"batch_id": BATCH_ID},
            )

            insert_sql = text("""
                INSERT INTO data_profiles (
                    batch_id,
                    column_name,
                    data_type,
                    total_records,
                    missing_records,
                    missing_percentage,
                    distinct_values,
                    mean_value,
                    median_value,
                    std_deviation,
                    min_value,
                    max_value,
                    q1,
                    q3,
                    iqr,
                    skewness,
                    kurtosis
                )
                VALUES (
                    :batch_id,
                    :column_name,
                    :data_type,
                    :total_records,
                    :missing_records,
                    :missing_percentage,
                    :distinct_values,
                    :mean_value,
                    :median_value,
                    :std_deviation,
                    :min_value,
                    :max_value,
                    :q1,
                    :q3,
                    :iqr,
                    :skewness,
                    :kurtosis
                )
            """)

            for profile in profiles:
                connection.execute(
                    insert_sql,
                    {
                        "batch_id": BATCH_ID,
                        **profile,
                    },
                )

        # Export a readable report.
        report_dir = Path("reports/profiling")
        report_dir.mkdir(parents=True, exist_ok=True)

        report = pd.DataFrame(profiles)

        report.to_csv(
            report_dir / f"batch_{BATCH_ID}_profile.csv",
            index=False,
        )

        print(f"\nStatistical profile — Batch {BATCH_ID}")
        print("=" * 85)

        print(
            report[
                [
                    "column_name",
                    "missing_percentage",
                    "mean_value",
                    "median_value",
                    "std_deviation",
                    "min_value",
                    "max_value",
                    "skewness",
                ]
            ].to_string(index=False)
        )

        print("\nProfiling completed successfully!")
        print(f"Columns profiled: {len(profiles)}")
        print("Results saved to PostgreSQL.")
        print("CSV report saved to reports/profiling/")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

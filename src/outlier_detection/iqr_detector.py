
import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()


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


engine = get_engine()

BASELINE_BATCH_ID = 1
TARGET_BATCH_ID = int(os.getenv("OUTLIER_BATCH_ID", "1"))

NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
]

IQR_MULTIPLIER = 1.5


def load_batch(batch_id):
    query = text("""
        SELECT
            trip_id,
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount,
            tip_amount,
            tolls_amount
        FROM taxi_trips
        WHERE batch_id = :batch_id
    """)

    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={"batch_id": batch_id}
        )


def calculate_iqr_bounds(baseline_df):
    bounds = {}

    for column in NUMERIC_COLUMNS:
        values = baseline_df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # Avoid treating every non-zero value as an outlier
        # when a feature has zero IQR.
        if iqr <= 0:
            print(f"Skipping {column}: IQR is zero.")
            continue

        bounds[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(q1 - IQR_MULTIPLIER * iqr),
            "upper": float(q3 + IQR_MULTIPLIER * iqr),
        }

    return bounds


def detect_outliers(target_df, bounds):
    results = []

    for column, limits in bounds.items():
        lower = limits["lower"]
        upper = limits["upper"]
        iqr = limits["iqr"]

        outlier_rows = target_df[
            (target_df[column] < lower) |
            (target_df[column] > upper)
        ]

        for _, row in outlier_rows.iterrows():
            value = float(row[column])

            distance = max(
                lower - value,
                value - upper,
                0
            )

            anomaly_score = distance / iqr

            if anomaly_score >= 2:
                severity = "high"
            elif anomaly_score >= 1:
                severity = "medium"
            else:
                severity = "low"

            results.append({
                "batch_id": TARGET_BATCH_ID,
                "trip_id": int(row["trip_id"]),
                "column_name": column,
                "detection_method": "IQR",
                "observed_value": value,
                "lower_bound": lower,
                "upper_bound": upper,
                "anomaly_score": float(anomaly_score),
                "severity": severity,
            })

    return results


def save_results(results):
    insert_query = text("""
        INSERT INTO outlier_results (
            batch_id,
            trip_id,
            column_name,
            detection_method,
            observed_value,
            lower_bound,
            upper_bound,
            anomaly_score,
            severity
        )
        VALUES (
            :batch_id,
            :trip_id,
            :column_name,
            :detection_method,
            :observed_value,
            :lower_bound,
            :upper_bound,
            :anomaly_score,
            :severity
        )
        ON CONFLICT (
            batch_id,
            trip_id,
            column_name,
            detection_method
        )
        DO UPDATE SET
            observed_value = EXCLUDED.observed_value,
            lower_bound = EXCLUDED.lower_bound,
            upper_bound = EXCLUDED.upper_bound,
            anomaly_score = EXCLUDED.anomaly_score,
            severity = EXCLUDED.severity,
            detected_at = CURRENT_TIMESTAMP
    """)

    with engine.begin() as connection:
        # Remove stale IQR results for this batch before saving
        connection.execute(
            text("""
                DELETE FROM outlier_results
                WHERE batch_id = :batch_id
                  AND detection_method = 'IQR'
            """),
            {"batch_id": TARGET_BATCH_ID}
        )

        if results:
            connection.execute(insert_query, results)


def main():
    print("=" * 75)
    print("IQR OUTLIER DETECTION")
    print("=" * 75)

    print(f"Baseline batch: {BASELINE_BATCH_ID}")
    print(f"Target batch:   {TARGET_BATCH_ID}")

    baseline_df = load_batch(BASELINE_BATCH_ID)
    target_df = load_batch(TARGET_BATCH_ID)

    if baseline_df.empty:
        raise ValueError("Baseline batch contains no records.")

    if target_df.empty:
        raise ValueError("Target batch contains no records.")

    bounds = calculate_iqr_bounds(baseline_df)

    print("\nBaseline IQR boundaries:")
    for column, limits in bounds.items():
        print(
            f"{column:20s} "
            f"Lower: {limits['lower']:10.3f} "
            f"Upper: {limits['upper']:10.3f}"
        )

    results = detect_outliers(target_df, bounds)

    save_results(results)

    print("\nDetection summary:")
    print(f"Records analyzed: {len(target_df)}")
    print(f"Outlier detections: {len(results)}")

    if results:
        summary = pd.DataFrame(results).groupby(
            ["column_name", "severity"]
        ).size().unstack(fill_value=0)

        print("\nOutliers by column and severity:")
        print(summary)

    print("\nIQR detection completed successfully!")


if __name__ == "__main__":
    main()

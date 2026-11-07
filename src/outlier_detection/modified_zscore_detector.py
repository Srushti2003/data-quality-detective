
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

MODIFIED_Z_THRESHOLD = 3.5
MAD_CONSTANT = 0.6745

NUMERIC_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
]


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


def calculate_baseline_statistics(df):
    statistics = {}

    for column in NUMERIC_COLUMNS:
        values = df[column].dropna()

        if values.empty:
            continue

        median = values.median()
        mad = (values - median).abs().median()

        if pd.isna(mad) or mad == 0:
            print(f"Skipping {column}: MAD is zero.")
            continue

        # Convert the modified Z-score threshold
        # into lower and upper raw-value boundaries.
        margin = (
            MODIFIED_Z_THRESHOLD * mad / MAD_CONSTANT
        )

        statistics[column] = {
            "median": float(median),
            "mad": float(mad),
            "lower": float(median - margin),
            "upper": float(median + margin),
        }

    return statistics


def detect_outliers(target_df, statistics):
    results = []

    for column, stats in statistics.items():

        modified_z_scores = (
            MAD_CONSTANT
            * (target_df[column] - stats["median"])
            / stats["mad"]
        )

        outlier_mask = (
            modified_z_scores.abs() > MODIFIED_Z_THRESHOLD
        )

        outlier_rows = target_df[outlier_mask]

        for index, row in outlier_rows.iterrows():

            score = float(abs(modified_z_scores.loc[index]))

            if score >= 7:
                severity = "high"
            else:
                severity = "medium"

            results.append({
                "batch_id": TARGET_BATCH_ID,
                "trip_id": int(row["trip_id"]),
                "column_name": column,
                "detection_method": "MODIFIED_Z",
                "observed_value": float(row[column]),
                "lower_bound": stats["lower"],
                "upper_bound": stats["upper"],
                "anomaly_score": score,
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

        connection.execute(
            text("""
                DELETE FROM outlier_results
                WHERE batch_id = :batch_id
                  AND detection_method = 'MODIFIED_Z'
            """),
            {"batch_id": TARGET_BATCH_ID}
        )

        if results:
            connection.execute(insert_query, results)


def main():

    print("=" * 75)
    print("MODIFIED Z-SCORE OUTLIER DETECTION")
    print("=" * 75)

    print(f"Baseline batch: {BASELINE_BATCH_ID}")
    print(f"Target batch: {TARGET_BATCH_ID}")

    baseline_df = load_batch(BASELINE_BATCH_ID)
    target_df = load_batch(TARGET_BATCH_ID)

    if baseline_df.empty or target_df.empty:
        raise ValueError("Baseline or target batch contains no records.")

    statistics = calculate_baseline_statistics(baseline_df)

    print("\nBaseline statistics:")

    for column, stats in statistics.items():
        print(
            f"{column:20s} "
            f"Median: {stats['median']:10.3f} "
            f"MAD: {stats['mad']:10.3f} "
            f"Lower: {stats['lower']:10.3f} "
            f"Upper: {stats['upper']:10.3f}"
        )

    results = detect_outliers(target_df, statistics)

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

    print("\nModified Z-Score detection completed successfully!")


if __name__ == "__main__":
    main()

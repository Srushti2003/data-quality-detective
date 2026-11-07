
import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

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

CONTAMINATION = 0.02
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
]


def load_batch(batch_id):
    columns = ", ".join(
        ["trip_id"] + FEATURE_COLUMNS
    )

    query = text(f"""
        SELECT {columns}
        FROM taxi_trips
        WHERE batch_id = :batch_id
    """)

    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={"batch_id": batch_id}
        )


def prepare_features(df):
    return df[FEATURE_COLUMNS].replace(
        [float("inf"), float("-inf")],
        float("nan")
    )


def detect_outliers(baseline_df, target_df):

    # Fit preprocessing only on the baseline dataset.
    imputer = SimpleImputer(strategy="median")

    baseline_imputed = imputer.fit_transform(
        prepare_features(baseline_df)
    )

    target_imputed = imputer.transform(
        prepare_features(target_df)
    )

    scaler = RobustScaler()

    baseline_scaled = scaler.fit_transform(
        baseline_imputed
    )

    target_scaled = scaler.transform(
        target_imputed
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train exclusively on the baseline dataset.
    model.fit(baseline_scaled)

    # -1 indicates an anomaly; 1 indicates normal.
    predictions = model.predict(target_scaled)

    # Higher values indicate more unusual observations.
    anomaly_scores = -model.score_samples(target_scaled)

    # Use the baseline's 99th percentile as a severity reference.
    baseline_scores = -model.score_samples(baseline_scaled)

    high_severity_threshold = float(
        pd.Series(baseline_scores).quantile(0.99)
    )

    results = []

    for index, prediction in enumerate(predictions):

        if prediction != -1:
            continue

        score = float(anomaly_scores[index])

        severity = (
            "high"
            if score >= high_severity_threshold
            else "medium"
        )

        row = target_df.iloc[index]

        results.append({
            "batch_id": TARGET_BATCH_ID,
            "trip_id": int(row["trip_id"]),
            "column_name": "multivariate",
            "detection_method": "ISOLATION_FOREST",
            "observed_value": None,
            "lower_bound": None,
            "upper_bound": None,
            "anomaly_score": score,
            "severity": severity,
        })

    return results, predictions, anomaly_scores


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
            anomaly_score = EXCLUDED.anomaly_score,
            severity = EXCLUDED.severity,
            detected_at = CURRENT_TIMESTAMP
    """)

    with engine.begin() as connection:

        connection.execute(
            text("""
                DELETE FROM outlier_results
                WHERE batch_id = :batch_id
                  AND detection_method = 'ISOLATION_FOREST'
            """),
            {"batch_id": TARGET_BATCH_ID}
        )

        if results:
            connection.execute(insert_query, results)


def main():

    print("=" * 75)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 75)

    print(f"Baseline batch: {BASELINE_BATCH_ID}")
    print(f"Target batch: {TARGET_BATCH_ID}")
    print(f"Contamination: {CONTAMINATION}")

    baseline_df = load_batch(BASELINE_BATCH_ID)
    target_df = load_batch(TARGET_BATCH_ID)

    if baseline_df.empty or target_df.empty:
        raise ValueError("Baseline or target batch contains no records.")

    results, predictions, scores = detect_outliers(
        baseline_df,
        target_df
    )

    save_results(results)

    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())

    print("\nDetection summary:")
    print(f"Records analyzed: {len(target_df)}")
    print(f"Normal records: {normal_count}")
    print(f"Anomalous records: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_count / len(target_df) * 100:.2f}%")

    if results:
        summary = pd.DataFrame(results).groupby(
            "severity"
        ).size()

        print("\nAnomalies by severity:")
        print(summary)

    print("\nIsolation Forest detection completed successfully!")


if __name__ == "__main__":
    main()

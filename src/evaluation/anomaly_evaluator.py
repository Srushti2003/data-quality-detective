
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.metrics import confusion_matrix

load_dotenv()

BATCH_ID = int(os.getenv("EVALUATION_BATCH_ID", "2"))

DETECTION_METHODS = [
    "IQR",
    "Z_SCORE",
    "MODIFIED_Z",
    "ISOLATION_FOREST",
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


def load_ground_truth(connection):
    query = text("""
        SELECT DISTINCT trip_id
        FROM injected_incidents
        WHERE batch_id = :batch_id
    """)

    return pd.read_sql(
        query,
        connection,
        params={"batch_id": BATCH_ID},
    )


def load_predictions(connection, method):
    query = text("""
        SELECT DISTINCT trip_id
        FROM outlier_results
        WHERE batch_id = :batch_id
          AND detection_method = :method
    """)

    return pd.read_sql(
        query,
        connection,
        params={
            "batch_id": BATCH_ID,
            "method": method,
        },
    )


def evaluate_method(
    all_trip_ids,
    actual_anomalies,
    predicted_anomalies,
    method,
):
    y_true = all_trip_ids.isin(
        actual_anomalies
    ).astype(int)

    y_pred = all_trip_ids.isin(
        predicted_anomalies
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    return {
        "detection_method": method,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision_score": float(precision),
        "recall_score": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "total_records": int(len(all_trip_ids)),
    }


def save_results(connection, results):

    delete_query = text("""
        DELETE FROM anomaly_evaluation
        WHERE batch_id = :batch_id
    """)

    connection.execute(
        delete_query,
        {"batch_id": BATCH_ID},
    )

    insert_query = text("""
        INSERT INTO anomaly_evaluation (
            batch_id,
            detection_method,
            true_positive,
            false_positive,
            false_negative,
            true_negative,
            precision_score,
            recall_score,
            f1_score,
            false_positive_rate,
            total_records
        )
        VALUES (
            :batch_id,
            :detection_method,
            :true_positive,
            :false_positive,
            :false_negative,
            :true_negative,
            :precision_score,
            :recall_score,
            :f1_score,
            :false_positive_rate,
            :total_records
        )
    """)

    for result in results:
        connection.execute(
            insert_query,
            {
                **result,
                "batch_id": BATCH_ID,
            },
        )


def main():

    engine = get_engine()

    try:
        with engine.connect() as connection:

            print("Loading evaluation data...")

            all_trip_ids = pd.read_sql(
                text("""
                    SELECT trip_id
                    FROM taxi_trips
                    WHERE batch_id = :batch_id
                """),
                connection,
                params={"batch_id": BATCH_ID},
            )["trip_id"]

            actual_anomalies = set(
                load_ground_truth(connection)["trip_id"]
            )

            if all_trip_ids.empty:
                raise ValueError("No records found.")

            if not actual_anomalies:
                raise ValueError("No ground-truth labels found.")

            results = []

            for method in DETECTION_METHODS:

                predicted = set(
                    load_predictions(
                        connection,
                        method,
                    )["trip_id"]
                )

                if not predicted:
                    raise ValueError(
                        f"No predictions found for {method}. "
                        "Run the detector first."
                    )

                result = evaluate_method(
                    all_trip_ids,
                    actual_anomalies,
                    predicted,
                    method,
                )

                results.append(result)

            with engine.begin() as transaction:
                save_results(transaction, results)

            report = pd.DataFrame(results)

            print("\nANOMALY DETECTION EVALUATION")
            print("=" * 90)

            print(
                report[
                    [
                        "detection_method",
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "true_negative",
                        "precision_score",
                        "recall_score",
                        "f1_score",
                        "false_positive_rate",
                    ]
                ].to_string(index=False)
            )

            print("\nEvaluation saved to PostgreSQL.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

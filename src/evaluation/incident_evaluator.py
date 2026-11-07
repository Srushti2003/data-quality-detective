
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

BATCH_ID = int(os.getenv("EVALUATION_BATCH_ID", "2"))

METHODS = [
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


def calculate_metrics(actual, predicted, all_ids):

    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(all_ids - actual - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
    }


def main():

    engine = get_engine()

    try:
        with engine.connect() as connection:

            all_ids = set(
                pd.read_sql(
                    text("""
                        SELECT trip_id
                        FROM taxi_trips
                        WHERE batch_id = :batch_id
                    """),
                    connection,
                    params={"batch_id": BATCH_ID},
                )["trip_id"]
            )

            incidents = pd.read_sql(
                text("""
                    SELECT DISTINCT
                        incident_type,
                        trip_id
                    FROM injected_incidents
                    WHERE batch_id = :batch_id
                """),
                connection,
                params={"batch_id": BATCH_ID},
            )

            predictions = {}

            for method in METHODS:

                predictions[method] = set(
                    pd.read_sql(
                        text("""
                            SELECT DISTINCT trip_id
                            FROM outlier_results
                            WHERE batch_id = :batch_id
                              AND detection_method = :method
                        """),
                        connection,
                        params={
                            "batch_id": BATCH_ID,
                            "method": method,
                        },
                    )["trip_id"]
                )

            results = []

            for incident_type, group in incidents.groupby(
                "incident_type"
            ):

                actual = set(group["trip_id"])

                for method in METHODS:

                    metrics = calculate_metrics(
                        actual,
                        predictions[method],
                        all_ids,
                    )

                    results.append({
                        "batch_id": BATCH_ID,
                        "detection_method": method,
                        "incident_type": incident_type,
                        **metrics,
                    })

            with engine.begin() as connection:

                connection.execute(
                    text("""
                        DELETE FROM incident_evaluation
                        WHERE batch_id = :batch_id
                    """),
                    {"batch_id": BATCH_ID},
                )

                insert_query = text("""
                    INSERT INTO incident_evaluation (
                        batch_id,
                        detection_method,
                        incident_type,
                        true_positive,
                        false_positive,
                        false_negative,
                        true_negative,
                        precision_score,
                        recall_score,
                        f1_score
                    )
                    VALUES (
                        :batch_id,
                        :detection_method,
                        :incident_type,
                        :true_positive,
                        :false_positive,
                        :false_negative,
                        :true_negative,
                        :precision_score,
                        :recall_score,
                        :f1_score
                    )
                """)

                connection.execute(
                    insert_query,
                    results,
                )

            report = pd.DataFrame(results)

            print("\nINCIDENT-SPECIFIC EVALUATION")
            print("=" * 100)

            print(
                report[
                    [
                        "detection_method",
                        "incident_type",
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "precision_score",
                        "recall_score",
                        "f1_score",
                    ]
                ].to_string(index=False)
            )

            print("\nIncident-specific evaluation saved successfully!")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

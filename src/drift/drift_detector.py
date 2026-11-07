
from pathlib import Path
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/raw/nyc_taxi")
JANUARY_FILE = DATA_DIR / "yellow_tripdata_2025-01.parquet"

REFERENCE_PERIOD = "2025-01-01"
REFERENCE_SOURCE = "January 2025 random sample (seed=42)"

CURRENT_BATCHES = {
    3: "2025-02-01",
    4: "2025-03-01",
}

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "payment_type",
    "store_and_fwd_flag",
    "rate_code_id",
]

# Thresholds
PSI_THRESHOLD = 0.20
KS_PVALUE_THRESHOLD = 0.05
WASSERSTEIN_THRESHOLD = 0.10
JS_THRESHOLD = 0.10

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

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


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def load_reference_data():
    if not JANUARY_FILE.exists():
        raise FileNotFoundError(JANUARY_FILE)

    print("Loading January reference dataset...")

    df = pd.read_parquet(JANUARY_FILE)

    df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    ).copy()

    df = df.rename(columns={
        "VendorID": "vendor_id",
        "RatecodeID": "rate_code_id",
        "Airport_fee": "airport_fee",
    })

    return df


def load_current_data(engine, batch_id):
    print(f"Loading current batch: {batch_id}")

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    query = text("""
        SELECT
            vendor_id,
            passenger_count,
            trip_distance,
            rate_code_id,
            store_and_fwd_flag,
            payment_type,
            fare_amount,
            tip_amount,
            tolls_amount,
            total_amount,
            congestion_surcharge,
            airport_fee,
            cbd_congestion_fee
        FROM taxi_trips
        WHERE batch_id = :batch_id
    """)

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"batch_id": batch_id},
        )

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing database features: {sorted(missing)}"
        )

    return df


# --------------------------------------------------
# PSI
# --------------------------------------------------



def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        lower = min(reference.min(), current.min())
        upper = max(reference.max(), current.max())

        if lower == upper:
            return 0.0

        edges = np.linspace(lower, upper, bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    epsilon = 0.001

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    # Smooth zero-frequency bins.
    reference_pct = np.where(
        reference_pct == 0, epsilon, reference_pct
    )
    current_pct = np.where(
        current_pct == 0, epsilon, current_pct
    )

    # Normalize after smoothing so each distribution sums to 1.
    reference_pct = reference_pct / reference_pct.sum()
    current_pct = current_pct / current_pct.sum()

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)




# --------------------------------------------------
# KS TEST
# --------------------------------------------------

def calculate_ks(reference, current):
    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    result = ks_2samp(reference, current)

    return float(result.pvalue)


# --------------------------------------------------
# WASSERSTEIN DISTANCE
# --------------------------------------------------

def calculate_wasserstein(reference, current):
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    distance = wasserstein_distance(reference, current)

    reference_iqr = (
        np.percentile(reference, 75)
        - np.percentile(reference, 25)
    )

    # Normalization is undefined when the reference has no spread.
    if reference_iqr <= 0:
        return np.nan

    return distance / reference_iqr


# --------------------------------------------------
# JENSEN-SHANNON DIVERGENCE
# --------------------------------------------------

def normalize_categories(series):
    def normalize_value(value):
        if pd.isna(value):
            return "MISSING"

        try:
            numeric_value = float(value)

            if np.isfinite(numeric_value):
                if numeric_value.is_integer():
                    return str(int(numeric_value))

                return str(numeric_value)

        except (ValueError, TypeError):
            pass

        return str(value).strip()

    return series.map(normalize_value)


def calculate_js(reference, current):
    reference = normalize_categories(reference)
    current = normalize_categories(current)

    # Avoid division by zero when either distribution is empty.
    if len(reference) == 0 or len(current) == 0:
        return np.nan

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    ref_counts = (
        reference.value_counts()
        .reindex(categories, fill_value=0)
    )

    cur_counts = (
        current.value_counts()
        .reindex(categories, fill_value=0)
    )

    ref_prob = ref_counts / ref_counts.sum()
    cur_prob = cur_counts / cur_counts.sum()

    # Jensen-Shannon distance using natural logarithms.
    midpoint = (ref_prob + cur_prob) / 2

    def kl_divergence(p, q):
        mask = p > 0
        return np.sum(
            p[mask] * np.log(p[mask] / q[mask])
        )

    js_divergence = (
        kl_divergence(ref_prob.to_numpy(), midpoint.to_numpy())
        + kl_divergence(cur_prob.to_numpy(), midpoint.to_numpy())
    ) / 2

    # Normalize the distance to the range [0, 1].
    js_distance = np.sqrt(js_divergence / np.log(2))

    return float(js_distance)



# --------------------------------------------------
# SEVERITY CLASSIFICATION
# --------------------------------------------------

def classify_result(metric_name, value):
    if pd.isna(value):
        return False, "unknown"

    if metric_name == "PSI":
        detected = value >= PSI_THRESHOLD

        if value < 0.10:
            severity = "low"
        elif value < PSI_THRESHOLD:
            severity = "moderate"
        else:
            severity = "high"

    elif metric_name == "KS_PVALUE":
        detected = value < KS_PVALUE_THRESHOLD
        severity = "moderate" if detected else "low"

    elif metric_name == "WASSERSTEIN":
        detected = value >= WASSERSTEIN_THRESHOLD

        if value < 0.05:
            severity = "low"
        elif value < WASSERSTEIN_THRESHOLD:
            severity = "moderate"
        else:
            severity = "high"

    elif metric_name == "JS_DIVERGENCE":
        detected = value >= JS_THRESHOLD

        if value < 0.05:
            severity = "low"
        elif value < JS_THRESHOLD:
            severity = "moderate"
        else:
            severity = "high"

    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    return detected, severity


# --------------------------------------------------
# DATABASE STORAGE
# --------------------------------------------------

def save_results(engine, results):
    if not results:
        return

    insert_query = text("""
        INSERT INTO drift_results (
            reference_period,
            current_period,
            reference_source,
            current_batch_id,
            feature_name,
            metric_name,
            metric_value,
            threshold,
            drift_detected,
            severity,
            reference_count,
            current_count
        )
        VALUES (
            :reference_period,
            :current_period,
            :reference_source,
            :current_batch_id,
            :feature_name,
            :metric_name,
            :metric_value,
            :threshold,
            :drift_detected,
            :severity,
            :reference_count,
            :current_count
        )
    """)

    # Ensure all records belong to the same detection run.
    identity_keys = (
        "current_batch_id",
        "current_period",
        "reference_period",
        "reference_source",
    )

    identity = {
        key: results[0].get(key)
        for key in identity_keys
    }

    if any(value is None for value in identity.values()):
        raise ValueError(
            "Missing detection run metadata."
        )

    for row in results:
        if any(row.get(key) != identity[key] for key in identity_keys):
            raise ValueError(
                "save_results expects results from exactly "
                "one batch and reference period."
            )

    delete_query = text("""
        DELETE FROM drift_results
        WHERE current_batch_id = :current_batch_id
          AND current_period = :current_period
          AND reference_period = :reference_period
          AND reference_source = :reference_source
    """)

    # Delete previous results and insert replacements atomically.
    with engine.begin() as connection:
        connection.execute(delete_query, identity)
        connection.execute(insert_query, results)


# --------------------------------------------------
# DRIFT DETECTION
# --------------------------------------------------

def detect_drift(reference, current, batch_id, current_period):
    results = []

    for feature in NUMERIC_FEATURES:
        print(f"Analyzing numeric feature: {feature}")

        ref_values = reference[feature]
        cur_values = current[feature]

        metrics = {
            "PSI": (
                calculate_psi(ref_values, cur_values),
                PSI_THRESHOLD,
            ),
            "KS_PVALUE": (
                calculate_ks(ref_values, cur_values),
                KS_PVALUE_THRESHOLD,
            ),
            "WASSERSTEIN": (
                calculate_wasserstein(ref_values, cur_values),
                WASSERSTEIN_THRESHOLD,
            ),
        }

        for metric_name, (value, threshold) in metrics.items():
            detected, severity = classify_result(
                metric_name, value
            )

            results.append({
                "reference_period": REFERENCE_PERIOD,
                "current_period": current_period,
                "reference_source": REFERENCE_SOURCE,
                "current_batch_id": batch_id,
                "feature_name": feature,
                "metric_name": metric_name,
                "metric_value": value,
                "threshold": threshold,
                "drift_detected": detected,
                "severity": severity,
                "reference_count": int(ref_values.notna().sum()),
                "current_count": int(cur_values.notna().sum()),
            })

    for feature in CATEGORICAL_FEATURES:
        print(f"Analyzing categorical feature: {feature}")

        value = calculate_js(
            reference[feature],
            current[feature],
        )

        detected, severity = classify_result(
            "JS_DIVERGENCE", value
        )

        results.append({
            "reference_period": REFERENCE_PERIOD,
            "current_period": current_period,
            "reference_source": REFERENCE_SOURCE,
            "current_batch_id": batch_id,
            "feature_name": feature,
            "metric_name": "JS_DIVERGENCE",
            "metric_value": value,
            "threshold": JS_THRESHOLD,
            "drift_detected": detected,
            "severity": severity,
            "reference_count": len(reference),
            "current_count": len(current),
        })

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    engine = get_engine()

    try:
        reference = load_reference_data()

        all_results = []

        for batch_id, current_period in CURRENT_BATCHES.items():
            current = load_current_data(engine, batch_id)

            results = detect_drift(
                reference,
                current,
                batch_id,
                current_period,
            )

            save_results(engine, results)

            all_results.extend(results)

            print(
                f"\nCompleted drift detection for batch {batch_id}"
            )

        report = pd.DataFrame(all_results)

        Path("reports/drift").mkdir(
            parents=True,
            exist_ok=True,
        )

        report.to_csv(
            "reports/drift/drift_results.csv",
            index=False,
        )

        print("\nDrift detection completed successfully!")
        print(f"Total metric results: {len(report)}")
        print("\nDrift summary:")
        print(
            report.groupby(
                ["current_batch_id", "metric_name"]
            )["drift_detected"].sum()
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

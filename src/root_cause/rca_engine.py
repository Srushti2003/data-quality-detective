
from psycopg2.extras import Json
import json
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "data_quality_detective"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )


# ---------------------------------------------------------
# FETCH STATISTICAL DRIFT RESULTS
# ---------------------------------------------------------

def fetch_drift_results(conn):
    query = """
        SELECT
            current_batch_id,
            reference_period,
            current_period,
            feature_name,
            metric_name,
            metric_value,
            threshold,
            severity
        FROM drift_results
        WHERE drift_detected = TRUE
          AND current_batch_id IN (3, 4)
        ORDER BY current_batch_id, feature_name;
    """

    return pd.read_sql_query(query, conn)


# ---------------------------------------------------------
# STATISTICAL FEATURE INVESTIGATION
# ---------------------------------------------------------

def investigate_numeric_feature(conn, batch_id, feature):

    allowed_features = {
        "trip_distance",
        "fare_amount",
        "total_amount",
        "tip_amount",
        "tolls_amount",
        "passenger_count",
        "airport_fee",
        "congestion_surcharge",
        "cbd_congestion_fee"
    }

    if feature not in allowed_features:
        return None

    query = f"""
        SELECT
            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE {feature} IS NULL
            ) AS missing_records,

            MIN({feature}) AS minimum_value,
            MAX({feature}) AS maximum_value,
            AVG({feature}) AS average_value,

            percentile_cont(0.50)
                WITHIN GROUP (ORDER BY {feature})
                AS median_value,

            COUNT(*) FILTER (
                WHERE {feature} < 0
            ) AS negative_records,

            COUNT(*) FILTER (
                WHERE {feature} > 100
            ) AS records_above_100

        FROM taxi_trips
        WHERE batch_id = %s;
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(batch_id,)
    ).iloc[0].to_dict()



# ---------------------------------------------------------
# SEGMENT-LEVEL INVESTIGATION
# ---------------------------------------------------------

def investigate_distance_by_vendor(conn, batch_id):

    query = """
        SELECT
            vendor_id,
            COUNT(*) AS total_trips,
            AVG(trip_distance) AS average_distance,
            percentile_cont(0.50)
                WITHIN GROUP (ORDER BY trip_distance)
                AS median_distance,
            MAX(trip_distance) AS maximum_distance,

            COUNT(*) FILTER (
                WHERE trip_distance > 100
            ) AS extreme_distance_records,

            COUNT(*) FILTER (
                WHERE trip_distance < 0
            ) AS negative_distance_records

        FROM taxi_trips

        WHERE batch_id = %s
          AND trip_distance IS NOT NULL

        GROUP BY vendor_id
        ORDER BY vendor_id;
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(batch_id,)
    )


def generate_vendor_distance_hypothesis(row):

    vendor_id = int(row["vendor_id"])
    trips = int(row["total_trips"])
    average = float(row["average_distance"])
    median = float(row["median_distance"])
    maximum = float(row["maximum_distance"])
    extreme_count = int(row["extreme_distance_records"])

    if extreme_count > 0:

        return {
            "category": "vendor_segment_extreme_distance",

            "description": (
                f"Vendor {vendor_id} has {extreme_count} "
                f"extreme-distance records among {trips} trips. "
                f"Average distance is {average:.2f} miles, "
                f"while median distance is {median:.2f} miles. "
                f"Maximum distance is {maximum:.2f} miles. "
                "The extreme observations may be influencing "
                "the overall distance distribution."
            ),

            "recommendation": (
                "Inspect extreme records for this vendor. "
                "Validate trip duration, distance and location "
                "consistency before determining the cause."
            ),

            "severity": "high",
            "confidence": "high"
        }

    return {
        "category": "vendor_segment_profile",

        "description": (
            f"Vendor {vendor_id} has {trips} trips, with "
            f"average distance {average:.2f} miles and "
            f"median distance {median:.2f} miles. "
            "No trip distances above 100 miles were found "
            "in this segment."
        ),

        "recommendation": (
            "Continue monitoring vendor-level distance "
            "distributions for significant changes."
        ),

        "severity": "low",
        "confidence": "moderate"
    }


def run_vendor_distance_rca(conn):
    reference_query = """
        SELECT batch_period
        FROM data_batches
        WHERE batch_id = 1;
    """

    reference_result = pd.read_sql_query(reference_query, conn)
    reference_period = reference_result.iloc[0]["batch_period"]

    for batch_id in (1, 3, 4):

        period_query = """
            SELECT batch_period
            FROM data_batches
            WHERE batch_id = %s;
        """

        period_result = pd.read_sql_query(
            period_query, conn, params=(batch_id,)
        )

        if period_result.empty:
            print(f"Skipping Batch {batch_id}: period not found.")
            continue

        current_period = period_result.iloc[0]["batch_period"]

        results = investigate_distance_by_vendor(conn, batch_id)

        for _, row in results.iterrows():

            vendor_id = int(row["vendor_id"])
            hypothesis = generate_vendor_distance_hypothesis(row)

            category = (
                f"{hypothesis['category']}_vendor_{vendor_id}"
            )

            evidence = {
                "vendor_id": vendor_id,
                "total_trips": int(row["total_trips"]),
                "average_distance": float(row["average_distance"]),
                "median_distance": float(row["median_distance"]),
                "maximum_distance": float(row["maximum_distance"]),
                "extreme_distance_records": int(
                    row["extreme_distance_records"]
                ),
                "negative_distance_records": int(
                    row["negative_distance_records"]
                )
            }

            insert_query = """
                INSERT INTO root_cause_analysis (
                    batch_id,
                    reference_period,
                    current_period,
                    feature_name,
                    drift_metric,
                    drift_value,
                    drift_threshold,
                    root_cause_category,
                    root_cause_description,
                    supporting_evidence,
                    recommendation,
                    severity,
                    confidence
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (
                    batch_id,
                    feature_name,
                    root_cause_category,
                    drift_metric
                )
                DO UPDATE SET
                    reference_period = EXCLUDED.reference_period,
                    current_period = EXCLUDED.current_period,
                    root_cause_description =
                        EXCLUDED.root_cause_description,
                    supporting_evidence =
                        EXCLUDED.supporting_evidence,
                    recommendation = EXCLUDED.recommendation,
                    severity = EXCLUDED.severity,
                    confidence = EXCLUDED.confidence,
                    investigated_at = CURRENT_TIMESTAMP;
            """

            with conn.cursor() as cursor:
                cursor.execute(
                    insert_query,
                    (
                        batch_id,
                        reference_period,
                        current_period,
                        "trip_distance",
                        "VENDOR_SEGMENT",
                        None,
                        None,
                        category,
                        hypothesis["description"],
                        Json(evidence),
                        hypothesis["recommendation"],
                        hypothesis["severity"],
                        hypothesis["confidence"]
                    )
                )

            print(
                f"Vendor RCA saved | Batch {batch_id} | "
                f"Vendor {vendor_id} | {category}"
            )


def investigate_distance_by_location(conn, batch_id):

    query = """
        SELECT
            pickup_location_id,
            COUNT(*) AS total_trips,

            AVG(trip_distance) AS average_distance,

            percentile_cont(0.50)
                WITHIN GROUP (ORDER BY trip_distance)
                AS median_distance,

            MAX(trip_distance) AS maximum_distance,

            COUNT(*) FILTER (
                WHERE trip_distance > 100
            ) AS extreme_distance_records,

            COUNT(*) FILTER (
                WHERE trip_distance < 0
            ) AS negative_distance_records

        FROM taxi_trips

        WHERE batch_id = %s
          AND pickup_location_id IS NOT NULL
          AND trip_distance IS NOT NULL

        GROUP BY pickup_location_id

        ORDER BY total_trips DESC;
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(batch_id,)
    )


def generate_location_distance_hypothesis(row):

    location_id = int(row["pickup_location_id"])
    trips = int(row["total_trips"])

    average = float(row["average_distance"])
    median = float(row["median_distance"])
    maximum = float(row["maximum_distance"])

    extreme_count = int(row["extreme_distance_records"])
    negative_count = int(row["negative_distance_records"])

    if extreme_count > 0:

        return {
            "category": "location_extreme_distance",

            "description": (
                f"Pickup location {location_id} contains "
                f"{extreme_count} trip records exceeding "
                f"100 miles among {trips} trips. "
                f"Average distance is {average:.2f} miles, "
                f"median distance is {median:.2f} miles, "
                f"and maximum distance is {maximum:.2f} miles. "
                "Extreme observations may be influencing "
                "the location-level distance distribution."
            ),

            "recommendation": (
                "Investigate the extreme trip records. "
                "Validate pickup and dropoff locations, "
                "trip duration, distance and fare consistency."
            ),

            "severity": "high",
            "confidence": "high"
        }

    elif negative_count > 0:

        return {
            "category": "location_negative_distance",

            "description": (
                f"Pickup location {location_id} contains "
                f"{negative_count} negative-distance records "
                f"among {trips} trips."
            ),

            "recommendation": (
                "Inspect affected records and validate "
                "the trip-distance ingestion and calculation logic."
            ),

            "severity": "high",
            "confidence": "high"
        }

    else:

        return {
            "category": "location_distance_profile",

            "description": (
                f"Pickup location {location_id} contains "
                f"{trips} trips. Average distance is "
                f"{average:.2f} miles, median distance is "
                f"{median:.2f} miles, and maximum distance "
                f"is {maximum:.2f} miles. "
                "No distances above 100 miles or negative "
                "distances were identified."
            ),

            "recommendation": (
                "Continue monitoring location-level trip "
                "distance distributions across monthly batches."
            ),

            "severity": "low",
            "confidence": "moderate"
        }


def run_location_distance_rca(conn):

    reference_query = """
        SELECT batch_period
        FROM data_batches
        WHERE batch_id = 1;
    """

    reference_result = pd.read_sql_query(
        reference_query, conn
    )

    reference_period = reference_result.iloc[0]["batch_period"]

    for batch_id in (1, 3, 4):

        period_query = """
            SELECT batch_period
            FROM data_batches
            WHERE batch_id = %s;
        """

        period_result = pd.read_sql_query(
            period_query,
            conn,
            params=(batch_id,)
        )

        if period_result.empty:
            print(f"Skipping Batch {batch_id}")
            continue

        current_period = period_result.iloc[0]["batch_period"]

        results = investigate_distance_by_location(
            conn, batch_id
        )

        for _, row in results.iterrows():

            location_id = int(row["pickup_location_id"])

            hypothesis = generate_location_distance_hypothesis(
                row
            )

            category = (
                f"{hypothesis['category']}_location_{location_id}"
            )

            evidence = {
                "pickup_location_id": location_id,
                "total_trips": int(row["total_trips"]),
                "average_distance": float(row["average_distance"]),
                "median_distance": float(row["median_distance"]),
                "maximum_distance": float(row["maximum_distance"]),
                "extreme_distance_records": int(
                    row["extreme_distance_records"]
                ),
                "negative_distance_records": int(
                    row["negative_distance_records"]
                )
            }

            insert_query = """
                INSERT INTO root_cause_analysis (
                    batch_id,
                    reference_period,
                    current_period,
                    feature_name,
                    drift_metric,
                    drift_value,
                    drift_threshold,
                    root_cause_category,
                    root_cause_description,
                    supporting_evidence,
                    recommendation,
                    severity,
                    confidence
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )

                ON CONFLICT (
                    batch_id,
                    feature_name,
                    root_cause_category,
                    drift_metric
                )

                DO UPDATE SET
                    reference_period = EXCLUDED.reference_period,
                    current_period = EXCLUDED.current_period,
                    root_cause_description =
                        EXCLUDED.root_cause_description,
                    supporting_evidence =
                        EXCLUDED.supporting_evidence,
                    recommendation = EXCLUDED.recommendation,
                    severity = EXCLUDED.severity,
                    confidence = EXCLUDED.confidence,
                    investigated_at = CURRENT_TIMESTAMP;
            """

            with conn.cursor() as cursor:

                cursor.execute(
                    insert_query,
                    (
                        batch_id,
                        reference_period,
                        current_period,
                        "trip_distance",
                        "LOCATION_SEGMENT",
                        None,
                        None,
                        category,
                        hypothesis["description"],
                        Json(evidence),
                        hypothesis["recommendation"],
                        hypothesis["severity"],
                        hypothesis["confidence"]
                    )
                )

            print(
                f"Location RCA saved | Batch {batch_id} | "
                f"Location {location_id} | {category}"
            )


def generate_hypothesis(feature, evidence):

    maximum = evidence["maximum_value"]
    average = evidence["average_value"]
    median = evidence["median_value"]

    negative_count = int(evidence["negative_records"] or 0)
    high_count = int(evidence["records_above_100"] or 0)

    if feature == "trip_distance" and high_count > 0:
        return {
            "category": "extreme_value_anomaly",
            "description": (
                f"{high_count} records have trip distances "
                "greater than 100 miles. These extreme "
                "observations may influence the observed "
                "distribution drift. The underlying cause "
                "has not been established."
            ),
            "recommendation": (
                "Inspect extreme trip records, validate "
                "pickup and dropoff locations, and check "
                "trip duration against distance."
            ),
            "severity": "high",
            "confidence": "high"
        }

    if feature in {"fare_amount", "total_amount"} and negative_count > 0:
        return {
            "category": "negative_monetary_values",
            "description": (
                f"{negative_count} records contain negative "
                f"{feature} values. These observations may "
                "contribute to the distribution change."
            ),
            "recommendation": (
                "Investigate negative monetary transactions "
                "by payment type and compare fare components "
                "before classifying them as invalid."
            ),
            "severity": "moderate",
            "confidence": "moderate"
        }

    if (
        maximum is not None
        and median is not None
        and average is not None
        and median > 0
        and average > median * 3
    ):
        return {
            "category": "distribution_skew",
            "description": (
                f"The average {feature} is substantially "
                "higher than its median, suggesting "
                "a right-skewed distribution."
            ),
            "recommendation": (
                "Inspect upper-tail observations and compare "
                "the feature across business segments."
            ),
            "severity": "moderate",
            "confidence": "moderate"
        }

    return {
        "category": "distribution_change",
        "description": (
            f"Statistical drift was detected in {feature}, "
            "but this investigation did not establish "
            "a specific underlying cause."
        ),
        "recommendation": (
            "Investigate the feature by vendor, payment "
            "type and pickup location."
        ),
        "severity": "moderate",
        "confidence": "low"
    }


# ---------------------------------------------------------
# BUSINESS-RULE: FARE RECONCILIATION
# ---------------------------------------------------------

def investigate_fare_reconciliation(conn, batch_id):

    query = """
        WITH reconciliation AS (
            SELECT
                CASE
                    WHEN fare_amount < 0
                        THEN 'negative_fare'
                    WHEN fare_amount = 0
                        THEN 'zero_fare'
                    ELSE 'positive_fare'
                END AS fare_category,

                total_amount - (
                    COALESCE(fare_amount, 0)
                    + COALESCE(extra, 0)
                    + COALESCE(mta_tax, 0)
                    + COALESCE(tip_amount, 0)
                    + COALESCE(tolls_amount, 0)
                    + COALESCE(improvement_surcharge, 0)
                    + COALESCE(congestion_surcharge, 0)
                    + COALESCE(airport_fee, 0)
                    + COALESCE(cbd_congestion_fee, 0)
                ) AS difference

            FROM taxi_trips

            WHERE batch_id = %s
              AND payment_type = 0
              AND total_amount IS NOT NULL
        )

        SELECT
            fare_category,
            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE ABS(difference) > 0.01
            ) AS mismatch_records,

            AVG(difference) AS average_difference,
            MIN(difference) AS minimum_difference,
            MAX(difference) AS maximum_difference,

            COUNT(*) FILTER (
                WHERE ABS(difference - 2.50) < 0.01
            ) AS records_with_2_50_difference

        FROM reconciliation

        GROUP BY fare_category
        ORDER BY fare_category;
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(batch_id,)
    )


def generate_reconciliation_hypothesis(row, batch_id):

    category = row["fare_category"]

    total = int(row["total_records"])
    mismatches = int(row["mismatch_records"])

    average_difference = float(row["average_difference"])
    minimum_difference = float(row["minimum_difference"])
    maximum_difference = float(row["maximum_difference"])

    records_2_50 = int(row["records_with_2_50_difference"])

    mismatch_percentage = (
        mismatches / total * 100 if total else 0
    )

    return {
        "category": "payment_type_reconciliation_pattern",

        "description": (
            f"Payment type 0 contains {mismatches} "
            f"reconciliation mismatches among {total} "
            f"{category.replace('_', ' ')} records "
            f"({mismatch_percentage:.2f}%). "
            f"The average reconciliation difference is "
            f"{average_difference:.2f}, with differences "
            f"ranging from {minimum_difference:.2f} "
            f"to {maximum_difference:.2f}. "
            f"{records_2_50} records have a difference "
            "of approximately 2.50. This is an observed "
            "pattern; the underlying cause is unconfirmed."
        ),

        "recommendation": (
            "Validate payment-type-specific fare calculations "
            "against the applicable TLC data dictionary and "
            "fare rules. Investigate negative-fare transactions "
            "separately and monitor reconciliation differences "
            "by fare category."
        ),

        "severity": (
            "high" if mismatch_percentage >= 50 else "moderate"
        ),

        "confidence": "high"
    }


# ---------------------------------------------------------
# BUSINESS-RULE: MISSING FEE COMPONENTS
# ---------------------------------------------------------

def investigate_missing_components(conn, batch_id):

    query = """
        SELECT
            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE congestion_surcharge IS NULL
            ) AS missing_congestion_surcharge,

            COUNT(*) FILTER (
                WHERE airport_fee IS NULL
            ) AS missing_airport_fee,

            COUNT(*) FILTER (
                WHERE cbd_congestion_fee IS NULL
            ) AS missing_cbd_fee

        FROM taxi_trips
        WHERE batch_id = %s
          AND payment_type = 0;
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(batch_id,)
    ).iloc[0].to_dict()


def generate_missing_component_hypothesis(evidence):

    total = int(evidence["total_records"])

    missing_congestion = int(
        evidence["missing_congestion_surcharge"]
    )

    missing_airport = int(
        evidence["missing_airport_fee"]
    )

    missing_cbd = int(
        evidence["missing_cbd_fee"]
    )

    return {
        "category": "payment_type_missing_components",

        "description": (
            f"Among {total} payment-type-0 records, "
            f"{missing_congestion} have missing congestion "
            f"surcharge values, {missing_airport} have "
            f"missing airport fee values, and {missing_cbd} "
            "have missing CBD congestion fee values. "
            "Missing values are not automatically invalid; "
            "their applicability must be verified."
        ),

        "recommendation": (
            "Validate component applicability using trip date, "
            "location and the applicable TLC fare rules. "
            "Distinguish expected NULL values from missing "
            "required values before raising data-quality alerts."
        ),

        "severity": "moderate",
        "confidence": "high"
    }


# ---------------------------------------------------------
# SAVE RCA RESULTS
# ---------------------------------------------------------

def save_rca(conn, drift, evidence, hypothesis):

    query = """
        INSERT INTO root_cause_analysis (
            batch_id,
            reference_period,
            current_period,
            feature_name,
            drift_metric,
            drift_value,
            drift_threshold,
            root_cause_category,
            root_cause_description,
            supporting_evidence,
            recommendation,
            severity,
            confidence
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )

        ON CONFLICT (
            batch_id,
            feature_name,
            root_cause_category,
            drift_metric
        )

        DO UPDATE SET
            reference_period = EXCLUDED.reference_period,
            current_period = EXCLUDED.current_period,
            drift_value = EXCLUDED.drift_value,
            drift_threshold = EXCLUDED.drift_threshold,
            root_cause_description =
                EXCLUDED.root_cause_description,
            supporting_evidence =
                EXCLUDED.supporting_evidence,
            recommendation = EXCLUDED.recommendation,
            severity = EXCLUDED.severity,
            confidence = EXCLUDED.confidence,
            investigated_at = CURRENT_TIMESTAMP;
    """

    values = (
        int(drift["batch_id"]),
        drift["reference_period"],
        drift["current_period"],
        drift["feature_name"],
        drift["drift_metric"],
        float(drift["drift_value"]),
        float(drift["drift_threshold"]),
        hypothesis["category"],
        hypothesis["description"],
        json.dumps(evidence, default=str),
        hypothesis["recommendation"],
        hypothesis["severity"],
        hypothesis["confidence"]
    )

    with conn.cursor() as cursor:
        cursor.execute(query, values)


# ---------------------------------------------------------
# STATISTICAL RCA PIPELINE
# ---------------------------------------------------------

def run_statistical_rca(conn):

    drift_results = fetch_drift_results(conn)

    if drift_results.empty:
        print("No detected drift results found.")
        return

    for _, drift in drift_results.iterrows():

        batch_id = int(drift["current_batch_id"])
        feature = drift["feature_name"]

        evidence = investigate_numeric_feature(
            conn,
            batch_id,
            feature
        )

        if evidence is None:
            continue

        hypothesis = generate_hypothesis(
            feature,
            evidence
        )

        save_rca(
            conn,
            {
                "batch_id": batch_id,
                "reference_period": drift["reference_period"],
                "current_period": drift["current_period"],
                "feature_name": feature,
                "drift_metric": drift["metric_name"],
                "drift_value": drift["metric_value"],
                "drift_threshold": drift["threshold"]
            },
            evidence,
            hypothesis
        )

        print(
            f"Statistical RCA | Batch {batch_id} | "
            f"{feature} | {hypothesis['category']}"
        )


# ---------------------------------------------------------
# BUSINESS RCA PIPELINE
# ---------------------------------------------------------

def run_business_rca(conn):

    batch_periods = {
        3: ("2025-01-01", "2025-02-01"),
        4: ("2025-01-01", "2025-03-01")
    }

    for batch_id, periods in batch_periods.items():

        reference_period, current_period = periods

        # Fare reconciliation investigation
        reconciliation = investigate_fare_reconciliation(
            conn,
            batch_id
        )

        for _, row in reconciliation.iterrows():

            hypothesis = generate_reconciliation_hypothesis(
                row,
                batch_id
            )

            category = row["fare_category"]

            drift = {
                "batch_id": batch_id,
                "reference_period": reference_period,
                "current_period": current_period,
                "feature_name": (
                    f"payment_type_0_{category}_reconciliation"
                ),
                "drift_metric": "BUSINESS_RULE",
                "drift_value": float(row["average_difference"]),
                "drift_threshold": 0.01
            }

            save_rca(
                conn,
                drift,
                row.to_dict(),
                hypothesis
            )

            print(
                f"Business RCA | Batch {batch_id} | "
                f"{category} | Reconciliation"
            )

        # Missing fee component investigation
        missing_evidence = investigate_missing_components(
            conn,
            batch_id
        )

        missing_hypothesis = generate_missing_component_hypothesis(
            missing_evidence
        )

        missing_drift = {
            "batch_id": batch_id,
            "reference_period": reference_period,
            "current_period": current_period,
            "feature_name": "payment_type_0_missing_components",
            "drift_metric": "BUSINESS_RULE",
            "drift_value": float(
                missing_evidence["missing_congestion_surcharge"]
                + missing_evidence["missing_airport_fee"]
            ),
            "drift_threshold": 0.0
        }

        save_rca(
            conn,
            missing_drift,
            missing_evidence,
            missing_hypothesis
        )

        print(
            f"Business RCA | Batch {batch_id} | "
            "Missing fee components"
        )


# ---------------------------------------------------------
# MAIN RCA EXECUTION
# ---------------------------------------------------------

def run_rca():

    conn = get_connection()

    try:

        print("\nDATA QUALITY DETECTIVE")
        print("ROOT CAUSE ANALYSIS")
        print("=" * 65)

        print("\n[1] Statistical RCA")
        run_statistical_rca(conn)

        print("\n[2] Business-rule RCA")
        run_business_rca(conn)

        print("\n[3] Vendor-level RCA")
        run_vendor_distance_rca(conn)

        print("\n[4] Location-level RCA")
        run_location_distance_rca(conn)

        conn.commit()

        print("\n" + "=" * 65)
        print("RCA RESULTS SAVED SUCCESSFULLY.")

    except Exception:

        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    run_rca()


import pandas as pd


def get_advanced_rules(df):
    duration = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60

    fare_component_columns = [
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
    ]

    fare_components_available = df[
        fare_component_columns
    ].notna().all(axis=1)

    fare_components = df[
        fare_component_columns
    ].sum(axis=1)

    return [
        {
            "name": "zero_trip_distance",
            "severity": "medium",
            "condition": (
                df["trip_distance"].notna()
                & (df["trip_distance"] == 0)
            ),
            "description": "Trip distance is zero.",
        },
        {
            "name": "extreme_trip_distance",
            "severity": "high",
            "condition": (
                df["trip_distance"].notna()
                & (df["trip_distance"] > 100)
            ),
            "description": (
                "Trip distance exceeds 100 miles."
            ),
        },

        {
            "name": "excessive_passenger_count",
            "severity": "medium",
            "condition": (
                df["passenger_count"].notna()
                & (df["passenger_count"] > 6)
            ),
            "description": (
                "Passenger count exceeds 6."
            ),
        },

        {
            "name": "extremely_long_trip",
            "severity": "high",
            "condition": duration > 1440,
            "description": "Trip duration exceeds 24 hours.",
        },
        {
            "name": "extremely_short_trip",
            "severity": "medium",
            "condition": (
                duration.notna()
                & (duration >= 0)
                & (duration < 1)
            ),
            "description": "Trip duration is below one minute.",
        },
        {
            "name": "unusually_high_fare",
            "severity": "medium",
            "condition": df["fare_amount"] > 500,
            "description": "Fare amount exceeds $500.",
        },
        {
            "name": "invalid_fare_relationship",
            "severity": "medium",
            "condition": (
                df["total_amount"].notna()
                & fare_components_available
                & (
                    (df["total_amount"] - fare_components).abs()
                    > 0.05 + 1e-9
                )
            ),
            "description": (
                "Total amount differs from the sum of "
                "fare components by more than $0.05."
            ),
        },
        {
            "name": "future_pickup_datetime",
            "severity": "high",
            "condition": (
                df["pickup_datetime"].notna()
                & (
                    df["pickup_datetime"]
                    > pd.Timestamp.now()
                )
            ),
            "description": "Pickup timestamp is in the future.",
        },
    ]

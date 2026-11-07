
import pandas as pd


def get_advanced_rules(df):
    duration = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60

    fare_components = (
        df["fare_amount"].fillna(0)
        + df["extra"].fillna(0)
        + df["mta_tax"].fillna(0)
        + df["tip_amount"].fillna(0)
        + df["tolls_amount"].fillna(0)
        + df["improvement_surcharge"].fillna(0)
        + df["congestion_surcharge"].fillna(0)
        + df["airport_fee"].fillna(0)
    )

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
                & df["fare_amount"].notna()
                & (
                    (df["total_amount"] - fare_components).abs()
                    > 0.05
                )
            ),
            "description": (
                "Total amount differs from the sum of "
                "available fare components by more than $0.05."
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

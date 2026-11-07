
import pandas as pd
import pytest

from src.quality.validation_engine import RULES


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "pickup_datetime": [
            pd.Timestamp("2025-01-01 10:00:00")
        ],
        "dropoff_datetime": [
            pd.Timestamp("2025-01-01 10:30:00")
        ],
        "passenger_count": [2],
        "trip_distance": [5.0],
        "fare_amount": [20.0],
        "total_amount": [28.0],
    })


def get_rule(rule_name):
    return next(
        rule for rule in RULES
        if rule["name"] == rule_name
    )


def test_all_eight_rules_are_registered():
    expected_rules = {
        "missing_pickup_datetime",
        "missing_dropoff_datetime",
        "invalid_trip_duration",
        "negative_trip_distance",
        "negative_fare_amount",
        "negative_total_amount",
        "missing_fare_amount",
        "invalid_passenger_count",
    }

    actual_rules = {rule["name"] for rule in RULES}

    assert actual_rules == expected_rules


@pytest.mark.parametrize(
    "rule_name,column",
    [
        ("missing_pickup_datetime", "pickup_datetime"),
        ("missing_dropoff_datetime", "dropoff_datetime"),
        ("missing_fare_amount", "fare_amount"),
    ],
)
def test_missing_values_detected(sample_data, rule_name, column):
    sample_data.loc[0, column] = None

    result = get_rule(rule_name)["condition"](sample_data)

    assert result.iloc[0]


def test_invalid_trip_duration_detected(sample_data):
    sample_data.loc[0, "dropoff_datetime"] = (
        sample_data.loc[0, "pickup_datetime"]
        - pd.Timedelta(minutes=5)
    )

    result = get_rule("invalid_trip_duration")["condition"](
        sample_data
    )

    assert result.iloc[0]


@pytest.mark.parametrize(
    "rule_name,column,value",
    [
        ("negative_trip_distance", "trip_distance", -10),
        ("negative_fare_amount", "fare_amount", -50),
        ("negative_total_amount", "total_amount", -100),
    ],
)
def test_negative_values_detected(
    sample_data, rule_name, column, value
):
    sample_data.loc[0, column] = value

    result = get_rule(rule_name)["condition"](sample_data)

    assert result.iloc[0]


@pytest.mark.parametrize("passenger_count", [0, -1])
def test_invalid_passenger_count_detected(
    sample_data, passenger_count
):
    sample_data.loc[0, "passenger_count"] = passenger_count

    result = get_rule("invalid_passenger_count")["condition"](
        sample_data
    )

    assert result.iloc[0]


def test_valid_record_passes_all_rules(sample_data):
    for rule in RULES:
        result = rule["condition"](sample_data)

        assert not result.iloc[0], (
            f"Valid record failed rule: {rule['name']}"
        )


@pytest.mark.parametrize(
    "column",
    ["trip_distance", "fare_amount", "total_amount",
     "passenger_count"],
)
def test_missing_numeric_values_do_not_trigger_numeric_rules(
    sample_data, column
):
    sample_data.loc[0, column] = None

    numeric_rules = {
        "negative_trip_distance",
        "negative_fare_amount",
        "negative_total_amount",
        "invalid_passenger_count",
    }

    for rule in RULES:
        if rule["name"] in numeric_rules:
            result = rule["condition"](sample_data)

            assert not result.iloc[0], (
                f"Unexpected failure for {rule['name']}"
            )

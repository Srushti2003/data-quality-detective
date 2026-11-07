
from sqlalchemy import inspect


EXPECTED_TABLES = {
    "data_batches",
    "taxi_trips",
    "quality_results",
    "drift_results",
}


def test_all_required_tables_exist(test_engine):
    inspector = inspect(test_engine)

    actual_tables = set(inspector.get_table_names())

    assert EXPECTED_TABLES.issubset(actual_tables)


def test_required_columns_exist(test_engine):
    inspector = inspect(test_engine)

    expected_columns = {
        "data_batches": {
            "batch_id",
            "dataset_name",
            "source_file",
            "batch_period",
            "row_count",
            "ingestion_status",
            "is_test_batch",
        },
        "taxi_trips": {
            "trip_id",
            "batch_id",
            "vendor_id",
            "pickup_datetime",
            "dropoff_datetime",
            "fare_amount",
            "trip_distance",
            "total_amount",
        },
        "quality_results": {
            "result_id",
            "batch_id",
            "rule_name",
            "severity",
            "failed_records",
            "failure_rate",
            "details",
        },
        "drift_results": {
            "drift_id",
            "reference_period",
            "current_period",
            "current_batch_id",
            "feature_name",
            "metric_name",
            "metric_value",
            "drift_detected",
            "severity",
        },
    }

    for table_name, required_columns in expected_columns.items():
        actual_columns = {
            column["name"]
            for column in inspector.get_columns(table_name)
        }

        assert required_columns.issubset(actual_columns), (
            f"Missing columns in {table_name}: "
            f"{required_columns - actual_columns}"
        )


def test_primary_keys_are_configured(test_engine):
    inspector = inspect(test_engine)

    expected_primary_keys = {
        "data_batches": "batch_id",
        "taxi_trips": "trip_id",
        "quality_results": "result_id",
        "drift_results": "drift_id",
    }

    for table_name, expected_column in expected_primary_keys.items():
        primary_key = inspector.get_pk_constraint(table_name)

        assert primary_key["constrained_columns"] == [
            expected_column
        ]


def test_foreign_keys_are_configured(test_engine):
    inspector = inspect(test_engine)

    expected_foreign_keys = {
        "taxi_trips": ("batch_id", "data_batches", "batch_id"),
        "quality_results": ("batch_id", "data_batches", "batch_id"),
        "drift_results": (
            "current_batch_id",
            "data_batches",
            "batch_id",
        ),
    }

    for table_name, expected in expected_foreign_keys.items():
        foreign_keys = inspector.get_foreign_keys(table_name)

        actual_relationships = {
            (
                fk["constrained_columns"][0],
                fk["referred_table"],
                fk["referred_columns"][0],
            )
            for fk in foreign_keys
        }

        assert expected in actual_relationships, (
            f"Expected foreign key {expected} "
            f"not found in {table_name}"
        )

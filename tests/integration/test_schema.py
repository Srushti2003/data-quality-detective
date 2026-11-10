
from sqlalchemy import inspect


EXPECTED_TABLES = {
    "data_batches",
    "taxi_trips",
    "injected_incidents",
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
            "source_row_number",
            "vendor_id",
            "pickup_datetime",
            "dropoff_datetime",
            "fare_amount",
            "trip_distance",
            "total_amount",
            "cbd_congestion_fee",
        },
        "injected_incidents": {
            "incident_id",
            "batch_id",
            "trip_id",
            "incident_type",
            "column_name",
            "created_at",
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
        "injected_incidents": "incident_id",
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
        "taxi_trips": {
            ("batch_id", "data_batches", "batch_id"),
        },
        "injected_incidents": {
            ("batch_id", "data_batches", "batch_id"),
            ("trip_id", "taxi_trips", "trip_id"),
        },
        "quality_results": {
            ("batch_id", "data_batches", "batch_id"),
        },
        "drift_results": {
            ("current_batch_id", "data_batches", "batch_id"),
        },
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

        assert expected.issubset(actual_relationships), (
            f"Missing foreign keys in {table_name}: "
            f"{expected - actual_relationships}"
        )


def test_unique_constraints_are_configured(test_engine):
    inspector = inspect(test_engine)

    expected_constraints = {
        "taxi_trips": {
            ("batch_id", "source_row_number"),
        },
        "injected_incidents": {
            ("batch_id", "trip_id", "incident_type"),
        },
    }

    for table_name, expected in expected_constraints.items():
        constraints = inspector.get_unique_constraints(table_name)

        actual = {
            tuple(constraint["column_names"])
            for constraint in constraints
        }

        assert expected.issubset(actual), (
            f"Missing unique constraints in {table_name}: "
            f"{expected - actual}"
        )

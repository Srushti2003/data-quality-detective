
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env.test")

EXPECTED_DATABASE = "data_quality_detective_test"
SCHEMA_FILE = PROJECT_ROOT / "src" / "database" / "schema.sql"


@pytest.fixture(scope="session")
def test_engine():
    """Create a connection exclusively to the dedicated test database."""

    required_variables = [
        "TEST_DB_HOST",
        "TEST_DB_PORT",
        "TEST_DB_NAME",
        "TEST_DB_USER",
        "TEST_DB_PASSWORD",
    ]

    missing = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            f"Missing test database configuration: {missing}"
        )

    database_name = os.getenv("TEST_DB_NAME")

    if database_name != EXPECTED_DATABASE:
        raise RuntimeError(
            f"Unsafe database configuration: expected "
            f"{EXPECTED_DATABASE}, received {database_name}"
        )

    database_url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("TEST_DB_USER"),
        password=os.getenv("TEST_DB_PASSWORD"),
        host=os.getenv("TEST_DB_HOST"),
        port=int(os.getenv("TEST_DB_PORT")),
        database=database_name,
    )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as connection:
            actual_database = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()

            if actual_database != EXPECTED_DATABASE:
                raise RuntimeError(
                    "Database safety check failed. "
                    f"Connected to: {actual_database}"
                )

        if not SCHEMA_FILE.is_file():
            raise FileNotFoundError(
                f"Database schema file not found: {SCHEMA_FILE}"
            )

        schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")

        with engine.begin() as connection:
            for statement in schema_sql.split(";"):
                statement = statement.strip()

                if statement:
                    connection.exec_driver_sql(statement)

        yield engine

    finally:
        engine.dispose()


@pytest.fixture
def test_batch(test_engine):
    """Create and clean up one test-owned batch."""

    with test_engine.begin() as connection:
        batch_id = connection.execute(
            text("""
                INSERT INTO data_batches (
                    dataset_name,
                    source_file,
                    batch_period,
                    row_count,
                    ingestion_status,
                    is_test_batch
                )
                VALUES (
                    'integration_test',
                    'pytest',
                    '2025-02-01',
                    100,
                    'completed',
                    TRUE
                )
                RETURNING batch_id
            """)
        ).scalar_one()

    yield batch_id

    with test_engine.begin() as connection:
        connection.execute(
            text("""
                DELETE FROM drift_results
                WHERE current_batch_id = :batch_id
            """),
            {"batch_id": batch_id},
        )

        connection.execute(
            text("""
                DELETE FROM data_batches
                WHERE batch_id = :batch_id
                  AND is_test_batch = TRUE
            """),
            {"batch_id": batch_id},
        )


@pytest.fixture
def second_test_batch(test_engine):
    """Create and clean up a second independent test batch."""

    with test_engine.begin() as connection:
        batch_id = connection.execute(
            text("""
                INSERT INTO data_batches (
                    dataset_name,
                    source_file,
                    batch_period,
                    row_count,
                    ingestion_status,
                    is_test_batch
                )
                VALUES (
                    'integration_test',
                    'pytest_second_batch',
                    '2025-03-01',
                    100,
                    'completed',
                    TRUE
                )
                RETURNING batch_id
            """)
        ).scalar_one()

    yield batch_id

    with test_engine.begin() as connection:
        connection.execute(
            text("""
                DELETE FROM drift_results
                WHERE current_batch_id = :batch_id
            """),
            {"batch_id": batch_id},
        )

        connection.execute(
            text("""
                DELETE FROM data_batches
                WHERE batch_id = :batch_id
                  AND is_test_batch = TRUE
            """),
            {"batch_id": batch_id},
        )

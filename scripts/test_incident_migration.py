
"""
Destructive migration verification utility.

WARNING:
This script modifies the database schema to simulate a legacy state.
It must only be run against the dedicated
data_quality_detective_test database.

Never run this script against the production database.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env.test")

EXPECTED_DATABASE = "data_quality_detective_test"
MIGRATION_FILE = (
    PROJECT_ROOT
    / "src"
    / "database"
    / "migrations"
    / "001_add_incident_loader_constraints.sql"
)

required = [
    "TEST_DB_HOST",
    "TEST_DB_PORT",
    "TEST_DB_NAME",
    "TEST_DB_USER",
    "TEST_DB_PASSWORD",
]

missing = [key for key in required if not os.getenv(key)]
if missing:
    raise RuntimeError(f"Missing test DB settings: {missing}")

if os.getenv("TEST_DB_NAME") != EXPECTED_DATABASE:
    raise RuntimeError("Safety check failed: incorrect test database.")

if not MIGRATION_FILE.is_file():
    raise FileNotFoundError(MIGRATION_FILE)

url = URL.create(
    "postgresql+psycopg2",
    username=os.getenv("TEST_DB_USER"),
    password=os.getenv("TEST_DB_PASSWORD"),
    host=os.getenv("TEST_DB_HOST"),
    port=int(os.getenv("TEST_DB_PORT")),
    database=os.getenv("TEST_DB_NAME"),
)

engine = create_engine(url, pool_pre_ping=True)
connection = engine.raw_connection()
cursor = connection.cursor()

batch_id = None
trip_id = None
incident_id = None

try:
    cursor.execute("SELECT current_database()")
    actual_database = cursor.fetchone()[0]

    if actual_database != EXPECTED_DATABASE:
        raise RuntimeError(
            f"Connected to unexpected database: {actual_database}"
        )

    print(f"Database safety check passed: {actual_database}")

    # Simulate the older schema in the dedicated test database.
    cursor.execute("""
        ALTER TABLE injected_incidents
        DROP CONSTRAINT IF EXISTS injected_incidents_trip_id_fkey
    """)

    cursor.execute("""
        ALTER TABLE taxi_trips
        DROP CONSTRAINT IF EXISTS uq_taxi_batch_source_row
    """)

    cursor.execute("""
        ALTER TABLE taxi_trips
        DROP COLUMN IF EXISTS source_row_number
    """)

    # Insert temporary legacy records to verify preservation.
    cursor.execute("""
        INSERT INTO data_batches (
            dataset_name,
            source_file,
            ingestion_status,
            is_test_batch
        )
        VALUES (
            'migration_verification',
            'temporary_migration_test',
            'completed',
            TRUE
        )
        RETURNING batch_id
    """)
    batch_id = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO taxi_trips (batch_id)
        VALUES (%s)
        RETURNING trip_id
    """, (batch_id,))
    trip_id = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO injected_incidents (
            batch_id,
            trip_id,
            incident_type,
            column_name
        )
        VALUES (%s, %s, %s, %s)
        RETURNING incident_id
    """, (
        batch_id,
        trip_id,
        "migration_verification_incident",
        "test_column",
    ))
    incident_id = cursor.fetchone()[0]

    connection.commit()
    print("Legacy schema simulation and temporary records created.")

    # Apply the actual migration.
    migration_sql = MIGRATION_FILE.read_text(encoding="utf-8")
    cursor.execute(migration_sql)
    connection.commit()

    print("Migration executed successfully.")

    # Verify the new column and its type.
    cursor.execute("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'taxi_trips'
          AND column_name = 'source_row_number'
    """)
    column_result = cursor.fetchone()

    if not column_result or column_result[0] != "bigint":
        raise AssertionError("source_row_number column verification failed.")

    print("PASS: source_row_number BIGINT column exists.")

    # Verify the unique constraint.
    cursor.execute("""
        SELECT pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'taxi_trips'::regclass
          AND conname = 'uq_taxi_batch_source_row'
    """)
    unique_result = cursor.fetchone()

    if not unique_result or unique_result[0] != (
        "UNIQUE (batch_id, source_row_number)"
    ):
        raise AssertionError("Unique constraint verification failed.")

    print("PASS: batch/source-row unique constraint exists.")

    # Verify the foreign key and that it is validated.
    cursor.execute("""
        SELECT
            pg_get_constraintdef(oid),
            convalidated
        FROM pg_constraint
        WHERE conrelid = 'injected_incidents'::regclass
          AND conname = 'injected_incidents_trip_id_fkey'
    """)
    fk_result = cursor.fetchone()

    if not fk_result:
        raise AssertionError("Incident-to-trip foreign key is missing.")

    if fk_result[0] != (
        "FOREIGN KEY (trip_id) REFERENCES taxi_trips(trip_id)"
    ):
        raise AssertionError("Foreign key definition is unexpected.")

    if not fk_result[1]:
        raise AssertionError("Foreign key has not been validated.")

    print("PASS: incident-to-trip foreign key exists and is validated.")

    # Verify the original temporary records survived migration.
    cursor.execute("""
        SELECT COUNT(*)
        FROM injected_incidents i
        JOIN taxi_trips t ON t.trip_id = i.trip_id
        WHERE i.incident_id = %s
          AND i.batch_id = %s
          AND t.batch_id = %s
    """, (incident_id, batch_id, batch_id))

    preserved_count = cursor.fetchone()[0]

    if preserved_count != 1:
        raise AssertionError("Temporary incident record was not preserved.")

    print("PASS: temporary batch, trip, and incident record were preserved.")

    print("\nMIGRATION VERIFICATION PASSED.")

finally:
    # Clear any failed or unfinished transaction before cleanup.
    connection.rollback()

    # Remove only the records created by this verification script.
    try:
        if incident_id is not None:
            cursor.execute(
                "DELETE FROM injected_incidents WHERE incident_id = %s",
                (incident_id,),
            )

        if trip_id is not None:
            cursor.execute(
                "DELETE FROM taxi_trips WHERE trip_id = %s",
                (trip_id,),
            )

        if batch_id is not None:
            cursor.execute(
                "DELETE FROM data_batches WHERE batch_id = %s",
                (batch_id,),
            )

        connection.commit()

        if batch_id is not None:
            print("Temporary verification records cleaned up.")

    except Exception:
        connection.rollback()
        print("WARNING: Temporary-record cleanup failed.")

    cursor.close()
    connection.close()
    engine.dispose()

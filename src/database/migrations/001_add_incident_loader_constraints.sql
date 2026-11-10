-- Migration: 001_add_incident_loader_constraints.sql
-- Purpose:
--   1. Add source_row_number to taxi_trips.
--   2. Enforce unique source-row numbers within each batch.
--   3. Add and validate the incident-to-trip foreign key.
--
-- Existing records are preserved. Historical source_row_number values
-- remain NULL because their original row positions are not reconstructed.

BEGIN;

-- Prevent concurrent writes while schema constraints are migrated.
LOCK TABLE taxi_trips, injected_incidents
IN SHARE ROW EXCLUSIVE MODE;


-- 1. Add the missing source-row column.
ALTER TABLE taxi_trips
ADD COLUMN IF NOT EXISTS source_row_number BIGINT;

-- 2. Add the batch/source-row unique constraint.
DO $$
DECLARE
    existing_definition TEXT;
BEGIN
    SELECT pg_get_constraintdef(oid)
    INTO existing_definition
    FROM pg_constraint
    WHERE conrelid = 'taxi_trips'::regclass
      AND conname = 'uq_taxi_batch_source_row';

    IF existing_definition IS NOT NULL THEN
        IF existing_definition <> 'UNIQUE (batch_id, source_row_number)' THEN
            RAISE EXCEPTION
                'Existing constraint uq_taxi_batch_source_row has an unexpected definition: %',
                existing_definition;
        END IF;
    ELSE
        ALTER TABLE taxi_trips
        ADD CONSTRAINT uq_taxi_batch_source_row
        UNIQUE (batch_id, source_row_number);
    END IF;
END $$;

-- 3. Recheck incident references before adding the foreign key.
DO $$
DECLARE
    orphan_count BIGINT;
    batch_mismatch_count BIGINT;
BEGIN
    SELECT COUNT(*)
    INTO orphan_count
    FROM injected_incidents i
    LEFT JOIN taxi_trips t ON t.trip_id = i.trip_id
    WHERE t.trip_id IS NULL;

    IF orphan_count > 0 THEN
        RAISE EXCEPTION
            'Migration stopped: % orphan incident labels found.',
            orphan_count;
    END IF;

    SELECT COUNT(*)
    INTO batch_mismatch_count
    FROM injected_incidents i
    JOIN taxi_trips t ON t.trip_id = i.trip_id
    WHERE i.batch_id <> t.batch_id;

    IF batch_mismatch_count > 0 THEN
        RAISE EXCEPTION
            'Migration stopped: % incident labels have mismatched batches.',
            batch_mismatch_count;
    END IF;
END $$;

-- 4. Add the missing trip foreign key without immediately scanning
--    the existing table. New writes are still checked.
DO $$
DECLARE
    existing_definition TEXT;
BEGIN
    SELECT pg_get_constraintdef(oid)
    INTO existing_definition
    FROM pg_constraint
    WHERE conrelid = 'injected_incidents'::regclass
      AND conname = 'injected_incidents_trip_id_fkey';

    IF existing_definition IS NOT NULL THEN
        IF existing_definition <>
            'FOREIGN KEY (trip_id) REFERENCES taxi_trips(trip_id)' THEN
            RAISE EXCEPTION
                'Existing injected_incidents_trip_id_fkey has an unexpected definition: %',
                existing_definition;
        END IF;
    ELSE
        ALTER TABLE injected_incidents
        ADD CONSTRAINT injected_incidents_trip_id_fkey
        FOREIGN KEY (trip_id)
        REFERENCES taxi_trips(trip_id)
        NOT VALID;
    END IF;
END $$;

-- 5. Validate the foreign key against all existing records.
ALTER TABLE injected_incidents
VALIDATE CONSTRAINT injected_incidents_trip_id_fkey;

COMMIT;

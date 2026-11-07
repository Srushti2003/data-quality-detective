
CREATE TABLE IF NOT EXISTS data_batches (
    batch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    batch_period DATE,
    row_count BIGINT,
    ingestion_status VARCHAR(30) NOT NULL DEFAULT 'pending',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    is_test_batch BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS taxi_trips (
    trip_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES data_batches(batch_id),

    vendor_id INTEGER,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count DOUBLE PRECISION,
    trip_distance DOUBLE PRECISION,

    rate_code_id DOUBLE PRECISION,
    store_and_fwd_flag VARCHAR(10),
    pickup_location_id INTEGER,
    dropoff_location_id INTEGER,

    payment_type DOUBLE PRECISION,
    fare_amount DOUBLE PRECISION,
    extra DOUBLE PRECISION,
    mta_tax DOUBLE PRECISION,
    tip_amount DOUBLE PRECISION,
    tolls_amount DOUBLE PRECISION,
    improvement_surcharge DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    congestion_surcharge DOUBLE PRECISION,
    airport_fee DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS quality_results (
    result_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES data_batches(batch_id),
    rule_name VARCHAR(150) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    failed_records BIGINT NOT NULL DEFAULT 0,
    total_records BIGINT NOT NULL DEFAULT 0,
    failure_rate DOUBLE PRECISION,
    details JSONB,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS drift_results (
    drift_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    reference_period DATE NOT NULL,
    current_period DATE NOT NULL,
    reference_source VARCHAR(255) NOT NULL,

    current_batch_id BIGINT NOT NULL
        REFERENCES data_batches(batch_id),

    feature_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,

    metric_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,

    drift_detected BOOLEAN NOT NULL,
    severity VARCHAR(20) NOT NULL,

    reference_count INTEGER NOT NULL,
    current_count INTEGER NOT NULL,

    evaluated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
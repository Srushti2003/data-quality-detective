from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/nyc_taxi/yellow_tripdata_2025-01.parquet")
OUTPUT = Path("data/processed/nyc_taxi_corrupted.parquet")


def simulate_incidents(
    source: Path = SOURCE,
    output: Path = OUTPUT
) -> pd.DataFrame:
    """Generate a dataset containing controlled data-quality incidents."""

    df = pd.read_parquet(source).head(10_000).copy()

    original_rows = len(df)

    # Introduce controlled data-quality problems.
    df.loc[df.index[:100], "fare_amount"] = -50

    df.loc[df.index[100:200], "trip_distance"] = -10

    df.loc[df.index[200:300], "passenger_count"] = 0

    df.loc[df.index[300:400], "tpep_pickup_datetime"] = None

    df.loc[df.index[400:500], "total_amount"] = -100

    # Duplicate the first 100 records.
    df = pd.concat(
        [df, df.iloc[:100]],
        ignore_index=True
    )

    # Create the output directory and save the corrupted dataset.
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output, index=False)

    print("Incident simulation completed.")
    print(f"Original rows: {original_rows:,}")
    print(f"Corrupted rows: {len(df):,}")
    print(f"Output: {output}")

    return df


def main():
    simulate_incidents()


if __name__ == "__main__":
    main()
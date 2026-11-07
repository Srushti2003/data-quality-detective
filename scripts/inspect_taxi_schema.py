
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/nyc_taxi")

files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {DATA_DIR.resolve()}"
    )

for file in files:
    df = pd.read_parquet(file)

    print(f"\n{'=' * 60}")
    print(f"File: {file.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names and data types:")
    print(df.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

from pathlib import Path
import pandas as pd

data_dir = Path("data/raw/nyc_taxi")

for file in sorted(data_dir.glob("*.parquet")):
    df = pd.read_parquet(file)

    print(f"\nFile: {file.name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nSample records:")
    print(df.head(3))
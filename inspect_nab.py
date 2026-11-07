from pathlib import Path
import pandas as pd

files = [
    Path("data/nab/nyc_taxi.csv"),
    Path("data/nab/ambient_temperature_system_failure.csv"),
    Path("data/nab/machine_temperature_system_failure.csv"),
]

for file in files:
    if not file.exists():
        print(f"MISSING: {file}")
        continue

    df = pd.read_csv(file)

    print(f"\n{file.name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))
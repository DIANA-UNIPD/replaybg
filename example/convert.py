import os
import pandas as pd

folder = "."  # ← change this to your folder path

csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

for fname in csv_files:
    csv_path = os.path.join(folder, fname)
    parquet_path = os.path.join(folder, fname.replace(".csv", ".parquet"))

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)

    print(f"✓ {fname} → {fname.replace('.csv', '.parquet')} ({len(df)} rows, {len(df.columns)} cols)")

print("Done!")
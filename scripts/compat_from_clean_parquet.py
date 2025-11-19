import pandas as pd
from pathlib import Path

SRC = Path("data/processed/clean_zori.parquet")  # Original Parquet file
DST = Path("data/processed/zori_smoothed_seasonal.parquet")  # SARIMAx input file

df = pd.read_parquet(SRC)

# Rename to new schema
out = (
    df[["ZIP", "State", "Date", "Rent"]]
      .rename(columns={
          "ZIP": "zip",
          "State": "state",
          "Date": "date",
          "Rent": "zori_smoothed_seasonal",
      })
      .copy()
)

# Types + ordering
out["zip"] = out["zip"].astype(str).str.zfill(5)      # keep leading zeros (e.g., 01002)
out["date"] = pd.to_datetime(out["date"])
out = out.sort_values(["zip", "date"])

DST.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(DST, index=False)

print(out.head(3))
print(
    "rows", len(out),
    "zips", out["zip"].nunique(),
    "date_min", out["date"].min(),
    "date_max", out["date"].max(),
)
print(f"Wrote {DST}")

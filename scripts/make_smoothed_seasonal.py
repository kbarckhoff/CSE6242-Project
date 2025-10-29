# TODO: This file reads the raw data [Zillow's wide format which has one column per month] and crates
#  a parquet file [a long format which one row per month]

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import re

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

def is_date_label(s: str) -> bool:
    """
    Treat a column name as a date if pandas can parse it,
    e.g., '1/31/2015', '2015-01-31', 'Jan-2015', etc.
    """
    try:
        pd.to_datetime(s)
        return True
    except Exception:
        return False

def melt_wide_to_long(df: pd.DataFrame,
                      value_name: str = "zori_smoothed_seasonal") -> pd.DataFrame:
    """
    Convert Zillow wide format (one column per month) into long format:
      columns: date, zip, state, <value_name>
    ZIPs live in 'RegionName' and state code in 'State' if present.
    """
    # Identify ID columns and date columns
    id_candidates = ["RegionID", "RegionName", "RegionType", "StateName",
                     "State", "City", "Metro", "CountyName"]
    id_cols = [c for c in id_candidates if c in df.columns]
    date_cols = [c for c in df.columns if c not in id_cols and is_date_label(c)]

    if not date_cols:
        raise ValueError("Could not find any date columns in the CSV header.")

    long = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )

    # TODO: Parse fields
    long["date"] = pd.to_datetime(long["date"], errors="coerce")  # change from string to datetime
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")  # change from string to number

    # TODO: Standardize keys: zip, state
    if "RegionName" in long.columns:
        # Keep only clean 5-digit ZIPs
        z = long["RegionName"].astype(str).str.extract(r"(\d{5})", expand=False)
        long["zip"] = z
    elif "zip" not in long.columns:
        long["zip"] = pd.NA

    if "State" in long.columns:
        long["state"] = long["State"].astype(str).str.upper()
    elif "state" not in long.columns:
        long["state"] = pd.NA

    # TODO: Drop Null values
    long = long.dropna(subset=["date", value_name, "zip"])
    long = long[long["zip"].str.fullmatch(r"\d{5}")]

    # TODO: Optional filtering based on date, zip and state
    keep = ["date", "zip", "state", value_name]
    keep = [c for c in keep if c in long.columns]
    long = long[keep].sort_values(["zip", "date"]).reset_index(drop=True)

    return long

def main():
    p = argparse.ArgumentParser(
        description="Convert Zillow smoothed seasonality CSV (wide) to long Parquet."
    )
    p.add_argument("--csv", required=True,
                   help="Path to the Zillow smoothed seasonality CSV (zip granularity).")
    p.add_argument("--out", default=str(PROCESSED / "zori_smoothed_seasonal.parquet"),
                   help="Output Parquet path.")
    p.add_argument("--subset-states", nargs="*", default=None,
                   help="Optional list of state codes to keep (e.g., GA TX CA).")
    args = p.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    print(f"Reading CSV -> {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    long = melt_wide_to_long(df)

    if args.subset_states:
        keep = {s.upper() for s in args.subset_states}
        long = long[long["state"].isin(keep)]
        print(f"Filtered to states: {sorted(keep)}; rows={len(long):,}")

    print(f"Writing Parquet -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_path, index=False)

    print("Done.")

if __name__ == "__main__":
    main()

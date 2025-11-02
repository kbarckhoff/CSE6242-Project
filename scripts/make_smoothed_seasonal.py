#!/usr/bin/env python
"""
Create a tidy parquet from Zillow's wide ZIP CSV.

Input (wide):
    RegionID, SizeRank, RegionName, RegionType, StateName, State, City, Metro, CountyName,
    2015-01-31, 2015-02-28, ... <many monthly date columns>

Output (long parquet):
    zip (str, zero-padded 5)
    state (str)
    date (datetime64[ns], normalized to month-start "MS")
    zori_smoothed_seasonal (float)

Usage (examples):
    python scripts/make_smoothed_seasonal.py \
      --csv "data/raw/zori_all_homes_smoothed_seasonal_zip.csv" \
      --out "data/processed/zori_smoothed_seasonal.parquet"

    # restrict to GA and TX only (optional)
    python scripts/make_smoothed_seasonal.py \
      --csv "data/raw/zori_all_homes_smoothed_seasonal_zip.csv" \
      --out "data/processed/zori_smoothed_seasonal.parquet" \
      --subset-states GA TX
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tidy parquet from Zillow wide ZIP CSV")
    p.add_argument("--csv", required=True, type=Path, help="Path to Zillow wide ZIP CSV")
    p.add_argument("--out", required=True, type=Path, help="Output parquet path")
    p.add_argument(
        "--subset-states",
        nargs="*",
        default=None,
        help="Optional list of 2-letter state codes to keep (e.g., GA TX)",
    )
    p.add_argument(
        "--value-name",
        default="zori_smoothed_seasonal",
        help="Name of the value column in the long output (default: zori_smoothed_seasonal)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Read minimally but safely: keep RegionType to filter ZIP rows; RegionName is the ZIP.
    # Force RegionName and State to string to avoid dtype surprises.
    print(f">> reading CSV: {args.csv}")
    df = pd.read_csv(
        args.csv,
        dtype={"RegionName": "string", "State": "string", "RegionType": "string"},
        low_memory=False,
    )

    # Keep only ZIP rows
    if "RegionType" in df.columns:
        df = df.loc[df["RegionType"].str.lower() == "zip"].copy()
    else:
        print("!! RegionType column missing; assuming all rows are ZIPs")

    # Optional state filtering
    if args.subset_states:
        keep = {s.strip().upper() for s in args.subset_states}
        if "State" in df.columns:
            before = len(df)
            df = df.loc[df["State"].str.upper().isin(keep)].copy()
            print(f">> subset states {sorted(keep)}: kept {len(df):,} of {before:,} rows")
        else:
            print("!! State column not found; cannot subset by state")

    # Identify date columns (YYYY-MM-DD)
    date_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
    if not date_cols:
        raise ValueError("No YYYY-MM-DD date columns found in CSV")

    id_vars = []
    # 'RegionName' holds the ZIP code; keep State for convenience
    for c in ["RegionName", "State"]:
        if c in df.columns:
            id_vars.append(c)

    # Melt wide → long
    print(">> melting wide date columns to long…")
    long = df.melt(
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="date",
        value_name=args.value_name,
    )

    # Rename and clean
    long = long.rename(columns={"RegionName": "zip", "State": "state"})

    # Ensure ZIP is 5-char string with leading zeros if needed
    long["zip"] = long["zip"].astype("string").str.zfill(5)

    # Parse and normalize date to month start
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date"])

    long["date"] = long["date"].dt.to_period("M").dt.to_timestamp(how="start")

    # Coerce values to float and drop NA rows
    long[args.value_name] = pd.to_numeric(long[args.value_name], errors="coerce")
    long = long.dropna(subset=[args.value_name])

    # Keep only needed columns, sorted for nice parquet layout
    keep_cols = ["zip", "state", "date", args.value_name]
    long = long[keep_cols].sort_values(["zip", "date"]).reset_index(drop=True)

    # Write parquet
    print(
        f">> writing parquet: {args.out}  "
        f"(rows={len(long):,}, zips={long['zip'].nunique():,}, months≈{long['date'].nunique():,})"
    )
    long.to_parquet(args.out, index=False)

    # A tiny preview
    print(">> sample:")
    print(long.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

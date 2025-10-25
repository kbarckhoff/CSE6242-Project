# scripts/ingest_clean_zori.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def find_first(patterns: list[str], search_dirs: list[Path]) -> Path | None:
    """
        Return the first file that matches any of the given glob patterns in the given
        search directories (checked in order).

        Parameters
        ----------
        patterns : list[str]
            Glob patterns to try, e.g., "zori_all_homes_smoothed_zip*.csv".
        search_dirs : list[pathlib.Path]
            Directories to search. Each directory is searched with each pattern.

        Returns
        -------
        pathlib.Path | None
            The first matching path if found; otherwise None.

        """
    for d in search_dirs:
        for pat in patterns:
            hits = list(d.glob(pat))
            if hits:
                return hits[0]
    return None

def melt_zori_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Zillow ZORI 'wide' format (one column per month) into a tidy 'long'
    DataFrame with one row per (state, zip, date) and a numeric `zori` value.

    Steps
    -----
    - Keep identifier columns (RegionID, SizeRank, RegionName, RegionType, StateName,
      State, City, Metro, CountyName).
    - Treat all remaining date-like columns (e.g., 'YYYY-MM-31') as values.
    - Melt to columns: id vars + ['date', 'zori'].
    - Coerce `date` to datetime and `zori` to numeric (invalids -> NaN).
    - Sort rows by state, zip, date for stable downstream processing.

    Returns
    -------
    pd.DataFrame
        Tidy long DataFrame with columns including: state, zip, date, zori, and
        the original ID columns.
    """

    id_cols = [
        "RegionID", "SizeRank", "RegionName", "RegionType", "StateName",
        "State", "City", "Metro", "CountyName"
    ]
    date_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(
        id_vars=[c for c in id_cols if c in df.columns],
        value_vars=date_cols,
        var_name="date",
        value_name="zori"
    )
    # Coerce date and numeric value
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["zori"] = pd.to_numeric(long["zori"], errors="coerce")
    # Standardize key fields
    if "RegionName" in long.columns:
        long = long.rename(columns={"RegionName": "zip"})
    if "StateName" in long.columns:
        long = long.rename(columns={"StateName": "state_name"})
    if "State" in long.columns:
        long = long.rename(columns={"State": "state"})
    # Keep only rows with a 5-digit ZIP
    long = long[long["zip"].astype(str).str.fullmatch(r"\d{5}", na=False)]
    return long

def small_subset_for_commit(df: pd.DataFrame, states=("GA", "TX"), last_n_months=24, max_rows=20000) -> pd.DataFrame:
    """
       Create a small, commit-safe subset for code review and CI.

       Filtering rules
       ---------------
       - Keep only the most recent `last_n_months- 2 years by default` of data.
       - Keep only rows whose `state` is in the provided `states`.
       - If the subset still exceeds `max_rows`, downsample deterministically
         (`random_state=42`) and sort by state, zip, date.

       Returns
       -------
       pd.DataFrame
           A compact slice of the long DataFrame
       """
    recent_cutoff = df["date"].dropna().max() - pd.offsets.MonthEnd(last_n_months)
    df_small = df[(df["date"] >= recent_cutoff) & (df["state"].isin(states))].copy()
    if len(df_small) > max_rows:
        df_small = df_small.sample(max_rows, random_state=42).sort_values(["state","zip","date"])
    return df_small

def basic_eda_summaries(df_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build a set of tiny EDA summary tables from the tidy long ZORI DataFrame.

    Returns
    -------
    dict[str, pd.DataFrame]

    """

    out = {}
    # Overview counts
    overview = pd.DataFrame({
        "n_rows": [len(df_long)],
        "n_zips": [df_long["zip"].nunique()],
        "n_states": [df_long["state"].nunique()],
        "date_min": [df_long["date"].min()],
        "date_max": [df_long["date"].max()],
        "missing_rate": [df_long["zori"].isna().mean()]
    })
    out["overview"] = overview

    # Zip counts by state
    by_state = (df_long.dropna(subset=["zori"])
                        .groupby("state")["zip"].nunique()
                        .reset_index(name="n_zips"))
    out["by_state"] = by_state.sort_values("n_zips", ascending=False)

    # Missingness by state
    miss_by_state = (df_long.assign(is_na=df_long["zori"].isna())
                             .groupby("state")["is_na"]
                             .mean()
                             .reset_index(name="missing_rate"))
    out["missing_by_state"] = miss_by_state.sort_values("missing_rate", ascending=False)

    # Count of months per ZIP (coverage)
    months_per_zip = (df_long.groupby("zip")["date"].nunique()
                              .reset_index(name="n_months"))
    out["months_per_zip"] = months_per_zip.sort_values("n_months", ascending=False)

    return out

def main(args) -> None:
    """
    End-to-end ingest pipeline:

    1) Locate the raw Zillow CSV in `data/raw/` (or `--input_dir`) using common
       filename patterns.
    2) Read the CSV, convert to tidy long format using `melt_zori_wide_to_long` function.
    3) Produce a small, commit-safe sample using `small_subset_for_commit` function and write:
         - `data/samples/zori_long_small.csv` (CSV)
         - `data/processed/zori_long_small.parquet` (Parquet for fast local use)
    4) Generate tiny EDA tables using `basic_eda_summaries` function and write them under
       `data/samples/` for quick review in PRs.



    Notes
    -----
    This function writes only small artifacts to version control. Full raw inputs
    and large intermediate files should live in Google cloud.
    """

    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    raw_dir = args.input_dir if args.input_dir else data_dir / "raw"
    raw_dir = Path(raw_dir)

    patterns = [
        "zori_all_homes_smoothed_zip*.csv",
        "zori_all_homes_smoothed_zip.csv",
        "zori_all_homes_smoothed*.csv",
    ]
    csv_path = find_first(patterns, [raw_dir])
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find Zillow ZORI CSV in {raw_dir}. "
            f"Expected one of: {patterns}"
        )

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    long = melt_zori_wide_to_long(df)

    # Create small subset safe to commit
    small = small_subset_for_commit(long, states=tuple(args.states), last_n_months=args.last_n_months)
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_small = samples_dir / "zori_long_small.csv"
    small.to_csv(out_small, index=False)
    print(f"Wrote small tidy sample -> {out_small.resolve()}  (rows={len(small):,})")

    # Save small processed Parquet too (fast for later notebooks)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "zori_long_small.parquet").write_bytes(small.to_parquet(index=False))

    # Tiny EDA outputs (commit these)
    eda = basic_eda_summaries(small)
    for name, t in eda.items():
        path = samples_dir / f"eda_{name}.csv"
        t.to_csv(path, index=False)
        print(f"Wrote {name} -> {path.resolve()} (rows={len(t):,})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest Zillow ZORI CSV, clean to tidy long, and write small commit-safe outputs.")
    p.add_argument("--input_dir", type=str, default=None, help="Folder where raw Zillow CSV lives (default: data/raw).")
    p.add_argument("--states", nargs="+", default=["GA", "TX"], help="States to include in the small commit-safe subset.")
    p.add_argument("--last_n_months", type=int, default=24, help="Recent months to keep in the subset.")
    main(p.parse_args())

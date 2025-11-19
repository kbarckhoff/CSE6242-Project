# TODO: Create a library file that contains the functions used by the
#  make_time_series.py, make_forecasts.py, and volatility_index.py

from __future__ import annotations
from pathlib import Path
import os
import polars as pl
import pandas as pd

# Path read-Local or GCS
def scan_parquet_anywhere(path: str | Path) -> pl.LazyFrame:
    """
    Returns a Polars LazyFrame scanning a parquet dataset from local or GCS.
    Works with a single parquet file or a directory of partitioned parquet.
    """
    path = str(path)
    return pl.scan_parquet(path)

def to_pandas_if_small(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas(use_pyarrow_extension_array=True)

# ---- Geo discovery ----
def list_geos(path: str | Path, geo_col: str) -> list[str]:
    """
    Return sorted unique values of geo_col without loading entire table.
    """
    lf = scan_parquet_anywhere(path).select(pl.col(geo_col)).unique()
    vals = lf.collect().get_column(geo_col).drop_nans().drop_nulls().to_list()
    return sorted(x for x in vals if x)

# ---- Monthly aggregation per geo ----
def monthly_series_for_geo(source: str | Path | pd.DataFrame,
                           geo_type: str,
                           geo: str,
                           value_col: str) -> pd.Series:
    """
    Return a clean monthly pd.Series for a single geo (zip/state/metro).
    Handles duplicate rows per month (e.g., state-level) by aggregating.
    """
    # 1) Load
    if isinstance(source, (str, Path)):
        df = pd.read_parquet(source)
    else:
        df = source

    # 2) Which column holds the geo id?
    geo_col_map = {"state": "state", "zip": "zip", "metro": "RegionName"}
    geo_col = geo_col_map.get(geo_type, geo_type)

    # 3) Filter and keep only date + value
    sub = df.loc[df[geo_col] == geo, ["date", value_col]].copy()

    # 4) Normalize dates to **month start** and aggregate to ONE row per month
    sub["date"] = (
        pd.to_datetime(sub["date"])
          .dt.to_period("M")
          .dt.to_timestamp(how="start")     # <-- key fix (no "MS" here)
    )
    sub = sub.groupby("date", as_index=False)[value_col].mean()

    # 5) Build series and make it a regular monthly index
    s = sub.set_index("date")[value_col].astype("float64").sort_index()
    s = s.resample("MS").mean()  # monthly start index; safe if some months missing

    s.name = value_col
    return s



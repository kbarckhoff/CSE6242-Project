# TODO: Create a library file that contains the functions used by the
#  time_series.py, make_forecasts.py, and volatility_index.py

# scripts/utils_series.py
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
    # If this fails to find the path, we can mount GCS (gcsfuse) or use pandas+gcsfs fallback.
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
    Return a monthly pd.Series for a single geo, reading from a parquet PATH
    or from an already-loaded DataFrame.
    """
    # Load if a path is passed
    if isinstance(source, (str, Path)):
        df = pd.read_parquet(source)
    else:
        df = source

    # column name for the geo type
    geo_col_map = {
        "state": "state",
        "zip": "zip",
        "metro": "RegionName",
    }
    geo_col = geo_col_map.get(geo_type, geo_type)

    # filter + sort + cast to a clean monthly series
    sub = df.loc[df[geo_col] == geo, ["date", value_col]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub.sort_values("date", inplace=True)

    s = sub.set_index("date")[value_col].astype("float64")

    # Ensure monthly data [per month inputs]
    try:
        s = s.asfreq("MS")  # month start; use "M" if your dates are month-end
    except Exception:
        pass

    s.name = value_col
    return s

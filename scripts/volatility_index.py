from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

VALUE_COL = "zori_smoothed_seasonal"  # column in parquet

# ---------- helpers ----------
def _coerce_monthly(y: pd.Series) -> pd.Series:
    """Ensure numeric values, month-start DatetimeIndex, and MS freq."""
    if y.empty:
        return y
    y = pd.Series(pd.to_numeric(y.values, errors="coerce"),
                  index=pd.to_datetime(y.index)).dropna()
    if y.empty:
        return y
    # was: y.index = y.index.to_period("M").dt.to_timestamp(how="start")
    y.index = y.index.to_period("M").to_timestamp(how="start")
    y = y.asfreq("MS")
    return y


def future_index_from_last(y: pd.Series, steps: int) -> pd.DatetimeIndex:
    last = y.index[-1]
    start = (last.to_period("M") + 1).to_timestamp(how="start")
    return pd.date_range(start, periods=steps, freq="MS")

def volatility_from_series(y: pd.Series, window: int = 12) -> pd.Series:
    """
    Monthly % changes -> rolling std (percentage points).
    """
    y = _coerce_monthly(y)
    if y.empty:
        return y
    mret = y.pct_change() * 100.0
    vi = mret.rolling(window=window, min_periods=max(3, window // 2)).std()
    vi.name = "volatility_index"
    return vi.dropna()

def _append_forecast(hist_df: pd.DataFrame, geo_type: str, geo_id: str) -> pd.DataFrame:
    """
    If a forecast file exists, append it. Expect columns: date, rent_forecast.
    """
    fpath = Path("data/processed/forecasts") / f"{geo_type}={geo_id}" / "forecast.csv"
    if not fpath.exists():
        return hist_df
    f = pd.read_csv(fpath)
    if "rent_forecast" not in f.columns or "date" not in f.columns:
        return hist_df
    f = f[["date", "rent_forecast"]].rename(columns={"rent_forecast": VALUE_COL})
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    out = (pd.concat([hist_df, f], ignore_index=True)
             .drop_duplicates(subset=["date"], keep="last")
             .sort_values("date")
             .reset_index(drop=True))
    return out

def compute_for_geo(df: pd.DataFrame, geo_type: str, geo_id: str,
                    window: int, include_forecast: bool) -> pd.DataFrame:
    sub = df.loc[df[geo_type].astype(str) == str(geo_id), ["date", VALUE_COL]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "volatility_index"])
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)

    if include_forecast:
        sub = _append_forecast(sub, geo_type, geo_id)

    ts = pd.Series(sub[VALUE_COL].to_numpy(), index=sub["date"])
    vi = volatility_from_series(ts, window=window)
    return vi.reset_index().rename(columns={"index": "date"})

def list_geos(df: pd.DataFrame, geo_type: str) -> List[str]:
    return sorted(df[geo_type].astype(str).unique().tolist())

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True,
                   help="Path to zori_smoothed_seasonal.parquet")
    p.add_argument("--geo-type", default="zip", choices=["zip", "state"])
    p.add_argument("--geos", nargs="*", default=None,
                   help="List of geo ids. If omitted, process all for geo-type.")
    p.add_argument("--window", type=int, default=12,
                   help="Rolling window (months) for volatility.")
    p.add_argument("--include-forecast", action="store_true",
                   help="Append forecast.csv to extend volatility past history.")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/volatility"))
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    # Expect columns like: ['zip','state','date', VALUE_COL]
    need = {"date", VALUE_COL, args.geo_type}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    if not args.geos:
        geos = list_geos(df, args.geo_type)
    else:
        geos = [str(g) for g in args.geos]

    for g in geos:
        out = compute_for_geo(df, args.geo_type, g, args.window, args.include_forecast)
        out_dir = args.out_dir / f"{args.geo_type}={g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "volatility.csv"
        out.to_csv(out_csv, index=False)
        print(f"[wrote] {out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    main()

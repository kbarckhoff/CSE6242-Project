# TODO: Compute the volatility index for the states provided as arguments
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from utils_series import list_geos, monthly_series_for_geo

def volatility_from_forecast_csv(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    # expected columns: mean, ci95_lo, ci95_hi
    vi = (df["ci95_hi"] - df["ci95_lo"]).abs() / df["mean"].abs()
    vi.name = "volatility_index"
    return vi

def residual_volatility(y: pd.Series, window: int = 12) -> pd.Series:
    y_fit = y.rolling(window, min_periods=max(3, window//2)).mean()
    resid = (y - y_fit).dropna()
    vi = resid.rolling(window, min_periods=max(3, window//2)).std() / y_fit
    vi = vi.dropna()
    vi.name = "volatility_index"
    return vi

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--geo-type", default="state")
    p.add_argument("--geos", nargs="*", help="List of geos; if omitted or 'ALL', process all")
    p.add_argument("--value-col", default="zori_smoothed_seasonal")
    p.add_argument("--forecast-root", type=Path, default=Path("data/processed/forecasts"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/volatility"))
    p.add_argument("--mode", choices=["forecast", "residual"], default="forecast")
    p.add_argument("--window", type=int, default=12)
    args = p.parse_args()

    geos = args.geos
    if not geos or (len(geos) == 1 and geos[0].upper() == "ALL"):
        geos = list_geos(args.parquet, args.geo_type)

    for g in geos:
        if args.mode == "forecast":
            fc_csv = args.forecast_root / f"state={g}" / "forecast.csv"
            vi = volatility_from_forecast_csv(fc_csv)
        else:
            y = monthly_series_for_geo(args.parquet, args.geo_type, g, args.value_col)
            vi = residual_volatility(y, window=args.window)

        out_csv = args.out_dir / f"state={g}" / "volatility.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        vi.to_csv(out_csv, index=True)

if __name__ == "__main__":
    main()

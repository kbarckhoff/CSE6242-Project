# scripts/export_forecast_table.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- helpers---
def _coerce_monthly(y: pd.Series) -> pd.Series:
    idx = pd.to_datetime(y.index)
    y = pd.Series(pd.to_numeric(y.values, errors="coerce"), index=idx).dropna()
    if y.empty:
        return y
    y.index = y.index.to_period("M").to_timestamp(how="start")
    y = y.asfreq("MS")
    return y

def _sarimax_forecast(y: pd.Series, steps: int = 12):
    # conservative SARIMAX
    y = _coerce_monthly(y)
    if y.empty:
        # fallback: no data
        mean = pd.Series([np.nan] * steps,
                         index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1),
                                             periods=steps, freq="MS")) if len(y) else pd.Series([], dtype=float)
        return mean
    model = SARIMAX(
        y, order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
    )
    res = model.fit(disp=False)
    mean = res.get_forecast(steps=steps).predicted_mean.astype(float)
    return mean

def _latest_volatility(y: pd.Series, window: int = 12) -> float:
    """
    12-month rolling std of monthly percent change; return the last available value.
    """
    y = _coerce_monthly(y)
    if len(y) < window + 1:
        return np.nan
    pct = y.pct_change()
    vol = pct.rolling(window=window).std()
    return float(vol.iloc[-1])

# --- main export ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to zori_smoothed_seasonal.parquet")
    ap.add_argument("--geo-type", choices=["zip", "state"], default="zip")
    ap.add_argument("--geos", nargs="*", help="Optional list of geos (e.g., ZIPs) to include")
    ap.add_argument("--horizons", default="3,6,9,12", help="Comma-separated months ahead")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    df = pd.read_parquet(args.parquet)

    # expected cols: ['zip','state','date','zori_smoothed_seasonal'] from the parquet maker
    df["date"] = pd.to_datetime(df["date"])
    value_col = "zori_smoothed_seasonal"

    # choose key col by geo-type
    key = "zip" if args.geo_type == "zip" else "state"

    if args.geos:
        allowed = set(args.geos)
        df = df[df[key].astype(str).isin(allowed)]

    out_rows = []

    for g, gdf in df.groupby(key, sort=False):
        y = gdf.sort_values("date").set_index("date")[value_col]
        mean = _sarimax_forecast(y, steps=max(horizons))
        vol = _latest_volatility(y, window=12)

        # map months-ahead to the appropriate forecast index

        for h in horizons:
            try:
                f = float(mean.iloc[h - 1])
            except Exception:
                f = np.nan
            out_rows.append({
                "ZipCode" if args.geo_type == "zip" else "State": g,
                "Forecast_Month": h,
                "Forecast_Rent": f,
                "Volatility": vol
            })

    out = pd.DataFrame(out_rows)

    # If geo-type=zip, the column must be 'ZipCode'
    if args.geo_type == "zip":
        cols = ["ZipCode", "Forecast_Month", "Forecast_Rent", "Volatility"]
    else:
        cols = ["State", "Forecast_Month", "Forecast_Rent", "Volatility"]

    out = out[cols].sort_values(cols[:2]).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")

if __name__ == "__main__":
    main()

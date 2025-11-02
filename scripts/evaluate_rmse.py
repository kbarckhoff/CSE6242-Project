import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------- helpers ----------

def _coerce_monthly(y: pd.Series) -> pd.Series:
    """Ensure float values, month-start DateTimeIndex, no NaNs, MS freq."""
    if y.empty:
        return y
    idx = pd.to_datetime(y.index)
    y = pd.Series(pd.to_numeric(y.values, errors="coerce"), index=idx).dropna()
    if y.empty:
        return y
    # normalize to month start and set monthly-start frequency
    y.index = y.index.to_period("M").to_timestamp(how="start")
    y = y.asfreq("MS")
    return y

def _fit_forecast(y_train: pd.Series, steps: int) -> pd.Series:
    """Fit the SARIMAX and forecast `steps` ahead."""
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean.astype(float)
    return mean

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae  = float(np.mean(np.abs(err)))
    # prevent dividing by zero for MAPE
    denom = np.where(y_true.values == 0, np.nan, np.abs(y_true.values))
    mape = float(np.nanmean(np.abs(err.values) / denom) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}

# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Path to zori_smoothed_seasonal.parquet")
    p.add_argument("--geo-type", choices=["zip", "state"], default="zip")
    p.add_argument("--geos", nargs="+", required=True, help="List of geos to evaluate (e.g., 30309 30305)")
    p.add_argument("--value-col", default="zori_smoothed_seasonal", help="Value column in parquet")
    p.add_argument("--horizon", type=int, default=9, help="Hold-out months and forecast horizon")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/metrics"))
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    # Expecting columns: ['zip','state','date', value_col]
    df["date"] = pd.to_datetime(df["date"])

    out_rows = []
    for g in args.geos:
        # filter
        sub = df.loc[df[args.geo_type].astype(str) == str(g), ["date", args.value_col]].sort_values("date")
        if sub.empty:
            print(f"[skip] {g}: no data")
            continue

        y = pd.Series(sub[args.value_col].values, index=sub["date"].values, name="zori").astype(float)
        y = _coerce_monthly(y)

        if y.size < args.horizon + 24:  # need some history to fit; tweak if needed
            print(f"[skip] {g}: not enough history ({y.size} pts)")
            continue

        # hold-out last H months for evaluation
        H = args.horizon
        y_train = y.iloc[:-H]
        y_test  = y.iloc[-H:]

        y_pred = _fit_forecast(y_train, steps=H)
        # create a proper forecast index matching the test set
        future_idx = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
        y_pred.index = future_idx

        m = _metrics(y_test, y_pred)
        row = {
            "geo_type": args.geo_type,
            "geo": g,
            "n_train": int(y_train.size),
            "n_test": int(y_test.size),
            "horizon": H,
            **m,
        }
        out_rows.append(row)

        print(f"{args.geo_type}={g}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, MAPE={m['mape']:.2f}%")

    if out_rows:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"metrics_{args.geo_type}.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"→ wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()

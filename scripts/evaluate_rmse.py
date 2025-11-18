from __future__ import annotations
import argparse, random, warnings, pathlib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

def coerce_monthly(y: pd.Series) -> pd.Series:
    """Numeric, monthly-start index, no gaps (MS freq)."""
    y = pd.Series(pd.to_numeric(y.values, errors="coerce"),
                  index=pd.to_datetime(y.index)).dropna()
    y.index = y.index.to_period("M").to_timestamp(how="start")
    y = y.asfreq("MS")
    return y

def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b)**2)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=pathlib.Path,
                   default=pathlib.Path("data/processed/zori_smoothed_seasonal.parquet"))
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("data/processed/metrics/rmse_sarimax_sample100.csv"))
    p.add_argument("--n", type=int, default=100, help="number of ZIPs to sample")
    p.add_argument("--h", type=int, default=12, help="forecast horizon for backtest")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)  # expects: ['zip','state','date','zori_smoothed_seasonal']
    df = df.rename(columns={"zori_smoothed_seasonal": "rent"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["zip", "date"])

    # Only ZIPs with enough history (>= 36 months)
    eligible = (df.groupby("zip")["date"].nunique() >= 36)
    zips_all = sorted(eligible[eligible].index.tolist())
    if not zips_all:
        raise SystemExit("No ZIPs with >=36 months of history in parquet.")

    random.seed(args.seed)
    sample = random.sample(zips_all, k=min(args.n, len(zips_all)))

    rows, fails = [], []
    for i, z in enumerate(sample, 1):
        sub = df.loc[df["zip"] == z, ["date", "rent"]].set_index("date").sort_index()
        y = coerce_monthly(sub["rent"])
        if len(y) < args.h + 12:
            # too short for a 12-mo backtest — skip without error
            continue

        train, test = y.iloc[:-args.h], y.iloc[-args.h:]

        try:
            model = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=True,
                enforce_invertibility=True,
                simple_differencing=False,
            )
            fit = model.fit(disp=False)
            fc = fit.get_forecast(steps=args.h).predicted_mean
            score = rmse(test.values, fc.values)
            rows.append({"ZipCode": str(z).zfill(5), "RMSE_SARIMAX": score})
        except Exception as e:
            fails.append((z, repr(e)))

        if i % 10 == 0:
            print(f"[{i}/{len(sample)}] processed...")

    out = pd.DataFrame(rows).sort_values("ZipCode").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Wrote {args.out}  rows={len(out)}  (attempted={len(sample)}, fails={len(fails)})")
    if len(out):
        print(f"Mean RMSE (SARIMAX): {out['RMSE_SARIMAX'].mean():.2f}")

if __name__ == "__main__":
    main()

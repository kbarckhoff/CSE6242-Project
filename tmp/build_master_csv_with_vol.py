import pandas as pd, numpy as np, pathlib

# 1) Collect all per-ZIP forecasts that were produced
base = pathlib.Path("data/processed/forecasts")
rows = []
for d in base.glob("zip-*"):
    f = d / "forecast.csv"
    if f.exists():
        t = pd.read_csv(f)
        t["zip"] = d.name.split("-")[1]
        rows.append(t)

if not rows:
    raise SystemExit("No forecast CSVs found under data/processed/forecasts")

df = pd.concat(rows, ignore_index=True)
df["zip"]  = df["zip"].astype(str).str.zfill(5)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["zip", "date"])

# Keep horizons 3, 6, 9, 12
df["Forecast_Month"] = df.groupby("zip").cumcount() + 1
df = df[df["Forecast_Month"].isin([3, 6, 9, 12])]

# 2) Build Volatility from the long parquet used for modeling
parq = pathlib.Path("data/processed/zori_smoothed_seasonal.parquet")
if not parq.exists():
    raise SystemExit(f"Missing {parq}; cannot compute Volatility.")

long = pd.read_parquet(parq)
# Normalize column names that our earlier compat script produced
# Expecting: zip, date, zori_smoothed_seasonal
long = long.rename(columns={c: c.lower() for c in long.columns})
assert {"zip","date"}.issubset(long.columns), "Parquet must contain zip and date columns"
val_col = "zori_smoothed_seasonal" if "zori_smoothed_seasonal" in long.columns else "rent"
if val_col not in long.columns:
    raise SystemExit("Parquet must contain value column 'zori_smoothed_seasonal' or 'rent'")

long["zip"]  = long["zip"].astype(str).str.zfill(5)
long["date"] = pd.to_datetime(long["date"])

def sigma_last12(g):
    y = pd.to_numeric(g[val_col], errors="coerce")
    y.index = pd.to_datetime(g["date"])
    y = y.sort_index()
    deltas = y.diff().dropna()
    if len(deltas) == 0:
        return np.nan
    return float(np.nanstd(deltas.tail(12)))

vol = (long.groupby("zip")
            .apply(sigma_last12)
            .rename("Volatility")
            .reset_index())

# 3) Merge volatility into the forecast rows
df = df.merge(vol, on="zip", how="left")

# 4) Final selection/rename and write
out = (df.rename(columns={"zip":"ZipCode", "rent_forecast":"Forecast_Rent"})
         [["ZipCode","Forecast_Month","Forecast_Rent","Volatility"]]
         .reset_index(drop=True))

dest = pathlib.Path("data/processed/metrics/forecasts_all_zip_l12m.csv")
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print(f"? Wrote {dest}  rows = {len(out)}  (with Volatility)")

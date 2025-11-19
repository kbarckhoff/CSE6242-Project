import pandas as pd, pathlib

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
df = df.sort_values(["zip","date"])

# Horizon (1..12); keep 3, 6, 9, 12
df["Forecast_Month"] = df.groupby("zip").cumcount() + 1
df = df[df["Forecast_Month"].isin([3, 6, 9, 12])]

# Try to join Volatility (best-effort from common locations)
vol = None
for p in [
    "data/processed/volatility/volatility_index.csv",
    "data/processed/metrics/metrics_zip.csv",
    "data/processed/metrics_zip.csv",
]:
    P = pathlib.Path(p)
    if P.exists():
        t = pd.read_csv(P)
        zcols = [c for c in t.columns if c.lower() in ("zip","zipcode","zip_code","geoid","geo")]
        vcols = [c for c in t.columns if "volat" in c.lower()]
        if zcols and vcols:
            vol = t[[zcols[0], vcols[0]]].rename(columns={zcols[0]:"zip", vcols[0]:"Volatility"})
            vol["zip"] = vol["zip"].astype(str).str.zfill(5)
            break

if vol is not None:
    df = df.merge(vol, on="zip", how="left")
else:
    df["Volatility"] = ""

out = (
    df.rename(columns={"zip":"ZipCode", "rent_forecast":"Forecast_Rent"})
      [["ZipCode","Forecast_Month","Forecast_Rent","Volatility"]]
      .reset_index(drop=True)
)

dest = pathlib.Path("data/processed/metrics/forecasts_all_zip_l12m.csv")
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print(f"? Wrote {dest}  rows = {len(out)}")

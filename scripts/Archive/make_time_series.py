from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_history(parquet: Path, geo_type: str, geo: str, value_col: str = "zori_smoothed_seasonal") -> pd.Series:
    df = pd.read_parquet(parquet)
    df = df[(df[geo_type] == geo)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df[value_col].values, index=df["date"]).asfreq("MS")
    return s

def load_forecast(out_root: Path, geo_type: str, geo: str) -> pd.DataFrame:
    fcsv = out_root / f"{geo_type}={geo}" / "forecast.csv"
    return pd.read_csv(fcsv, parse_dates=["date"])

def load_volatility(vol_root: Path, geo_type: str, geo: str) -> pd.DataFrame:
    vcsv = vol_root / f"{geo_type}={geo}" / "volatility.csv"
    return pd.read_csv(vcsv, parse_dates=["date"])

def plot_one(zip_code: str, hist: pd.Series, fc: pd.DataFrame, vol: pd.DataFrame, out_png: Path, state: str | None = None):
    # Align and combine for plotting
    fc = fc.sort_values("date")
    vol = vol.sort_values("date")

    # Create figure
    plt.figure(figsize=(10, 6))
    gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.25)

    # --- Top: rent history + forecast + CI
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(hist.index, hist.values, label="history", linewidth=1.8)

    ax1.plot(fc["date"], fc["rent_forecast"], label="forecast", linewidth=2)
    if {"ci95_lo","ci95_hi"}.issubset(fc.columns):
        ax1.fill_between(fc["date"], fc["ci95_lo"], fc["ci95_hi"], alpha=0.2, label="95% CI")

    ax1.set_title(f"{state + ': ' if state else ''}{zip_code} — Rent history & forecast")
    ax1.set_ylabel("ZORI (index)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    # --- Bottom: volatility index
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    ax2.plot(vol["date"], vol["volatility_index"], linewidth=1.8)
    ax2.set_title("Volatility index (12-month rolling std of monthly % change)")
    ax2.set_ylabel("volatility")
    ax2.grid(True, alpha=0.25)

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[time_series] wrote {out_png}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="processed parquet (zori_smoothed_seasonal.parquet)")
    p.add_argument("--geo-type", default="zip", choices=["zip","state"])
    p.add_argument("--geos", nargs="+", required=True, help="ZIPs or states (match geo_type)")
    p.add_argument("--forecasts-root", default="data/processed/forecasts")
    p.add_argument("--volatility-root", default="data/processed/volatility")
    p.add_argument("--out-dir", default="figures/time_series")
    p.add_argument("--value-col", default="zori_smoothed_seasonal")
    args = p.parse_args()

    parquet = Path(args.parquet)
    forecasts_root = Path(args.forecasts_root)
    volatility_root = Path(args.volatility_root)
    out_dir = Path(args.out_dir)

    for g in args.geos:
        hist = load_history(parquet, args.geo_type, g, value_col=args.value_col)
        # Optional: if you stored state alongside ZIP, you can pass state to title. Else leave None.
        try:
            fc = load_forecast(forecasts_root, args.geo_type, g)
        except FileNotFoundError:
            print(f"[skip] forecast missing for {g}")
            continue
        try:
            vol = load_volatility(volatility_root, args.geo_type, g)
        except FileNotFoundError:
            print(f"[skip] volatility missing for {g}")
            continue

        out_png = out_dir / f"{args.geo_type}={g}.png"
        plot_one(g, hist, fc, vol, out_png)

if __name__ == "__main__":
    main()

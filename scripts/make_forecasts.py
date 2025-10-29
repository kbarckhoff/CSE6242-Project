# TODO: Make forecats using the parquet file
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from utils_series import list_geos, monthly_series_for_geo
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def sarimax_forecast(y: pd.Series, steps: int = 9):
    # light auto-ish spec; tune later
    model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)  # 95%
    ci.columns = ["ci95_lo", "ci95_hi"]
    return mean, ci

def export_and_plot(state: str, y: pd.Series, mean, ci, out_csv: Path, out_png: Path | None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([mean.rename("mean"), ci], axis=1)
    df.index = pd.to_datetime(df.index)      # ensure datetime index
    df.index.name = "date"
    df.to_csv(out_csv, index=True)
    ...


    if out_png:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(y.index, y.values, label="history", linewidth=2)
        ax.plot(mean.index, mean.values, label="forecast", linewidth=2)
        ax.fill_between(mean.index, ci["ci95_lo"], ci["ci95_hi"], alpha=0.2, label="95% CI")
        ax.set_title(f"{state}: SARIMAX forecast")
        ax.set_xlabel("Date"); ax.set_ylabel("ZORI (index)")
        ax.legend(loc="upper left")
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--geo-type", default="state")
    p.add_argument("--geos", nargs="*", help="List of geos; if omitted or 'ALL', process all")
    p.add_argument("--value-col", default="zori_smoothed_seasonal")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/forecasts"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures/forecasts"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    geos = args.geos
    if not geos or (len(geos) == 1 and geos[0].upper() == "ALL"):
        geos = list_geos(args.parquet, args.geo_type)

    for g in geos:
        y = monthly_series_for_geo(args.parquet, args.geo_type, g, args.value_col)
        mean, ci = sarimax_forecast(y, steps=args.steps)
        export_and_plot(
            g, y, mean, ci,
            out_csv=args.out_dir / f"state={g}" / "forecast.csv",
            out_png = None if args.no_figures else args.fig_dir / f"forecast_{g}.png"


        )

if __name__ == "__main__":
    main()

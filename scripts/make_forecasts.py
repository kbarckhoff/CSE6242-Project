from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# utils in your repo
from utils_series import list_geos, monthly_series_for_geo


# ---------- helpers ----------

def _coerce_monthly(y: pd.Series) -> pd.Series:
    """
    Ensure y is numeric, has a DatetimeIndex at month-start (MS), and no NaNs.
    """
    # numeric values with a real datetime index
    y = pd.Series(pd.to_numeric(y.values, errors="coerce"),
                  index=pd.to_datetime(y.index)).dropna()

    if y.empty:
        return y

    # normalize index to the first day of each month
    # (no 'MS' freq here; that's what caused the error)
    y.index = y.index.to_period("M").to_timestamp(how="start")

    # and declare the freq as monthly-start
    y = y.asfreq("MS")
    return y


def _future_index_from_last(y: pd.Series, steps: int) -> pd.DatetimeIndex:
    """
    Build a monthly-start future index beginning the month after the last observed.
    """
    last = y.index[-1]
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=steps, freq="MS")


def sarimax_forecast(y: pd.Series, steps: int = 9) -> tuple[pd.Series, pd.DataFrame]:
    """
    Forecast with a conservative SARIMAX. If CI comes back non-finite, fall back to residual sigma.
    If too little data, use a persistence baseline with sigma from recent deltas.
    Returns:
        mean (pd.Series), ci (pd.DataFrame with ['ci95_lo','ci95_hi'])
    """
    y = _coerce_monthly(y)
    n = len(y.dropna())

    # if there isn't enough data, use a simple baseline
    if n < 18:
        # repeat last value; CI from recent month-to-month deltas
        mean = pd.Series([y.iloc[-1]] * steps, index=_future_index_from_last(y, steps))
        deltas = y.diff().dropna()
        if len(deltas) >= 6:
            sigma = float(np.nanstd(deltas[-12:]))  # use up to last 12 deltas
        else:
            sigma = float(np.nanstd(deltas)) if len(deltas) else 0.0
        ci = pd.DataFrame(
            {"ci95_lo": mean - 1.96 * sigma, "ci95_hi": mean + 1.96 * sigma},
            index=mean.index,
        )
        return mean.astype(float), ci.astype(float)

    # SARIMAX model (better than ARIMA for short term forecasts)
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
    )
    res = model.fit(disp=False)

    # Forecast
    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean.astype(float)
    ci = pred.conf_int(alpha=0.05).copy()

    # Normalize CI column names
    if ci.shape[1] == 2:
        ci.columns = ["ci95_lo", "ci95_hi"]

    # Fallback if any non-finite slips in
    if not np.isfinite(ci.values).all():
        sigma = float(np.nanstd(res.resid, ddof=1))
        ci = pd.DataFrame(
            {"ci95_lo": mean - 1.96 * sigma, "ci95_hi": mean + 1.96 * sigma},
            index=mean.index,
        )

    return mean.astype(float), ci.astype(float)


def export_and_plot(
    state: str,
    y: pd.Series,
    mean: pd.Series,
    ci: pd.DataFrame,
    out_csv: Path,
    out_png: Path | None,
) -> None:
    """
    Write CSV and plot the forecast.
    """
    # Future index aligned to monthly start
    future_idx = _future_index_from_last(y, len(mean))
    mean = mean.reindex(future_idx)
    ci = ci.reindex(future_idx)

    df = pd.DataFrame(
        {
            "date": future_idx,
            "rent_forecast": mean.values.astype(float),
            "ci95_lo": ci["ci95_lo"].values.astype(float),
            "ci95_hi": ci["ci95_hi"].values.astype(float),
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if out_png is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y.index, y.values, label="history", linewidth=2)
        ax.plot(mean.index, mean.values, label="forecast", linewidth=2)
        ax.fill_between(mean.index, ci["ci95_lo"], ci["ci95_hi"], alpha=0.2, label="95% CI")
        ax.set_title(f"{state}: SARIMAX forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("ZORI (index)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160)
        plt.close(fig)


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, type=Path, help="Parquet with long ZORI series")
    p.add_argument("--geo-type", default="state", choices=["state", "zip"], help="Geo granularity")
    p.add_argument("--geos", nargs="*", help="Specific geos; omit or use ALL for all available")
    p.add_argument("--value-col", default="zori_smoothed_seasonal", help="Value column in parquet")
    p.add_argument("--steps", type=int, default=12, help="Forecast horizon in months")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/forecasts"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures/forecasts"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    # Resolve geos list
    geos: list[str] | None = args.geos
    if (not geos) or (len(geos) == 1 and (geos[0] or "").upper() == "ALL"):
        geos = list_geos(args.parquet, args.geo_type)

    for g in geos:
        # Pull a clean monthly series for this geo
        y = monthly_series_for_geo(args.parquet, args.geo_type, g, args.value_col)
        y = _coerce_monthly(y)
        if y.empty:
            print(f"[skip] {g}: no data")
            continue

        mean, ci = sarimax_forecast(y, steps=args.steps)

        out_csv = args.out_dir / f"{args.geo_type}={g}" / "forecast.csv"
        out_png = None if args.no_figures else (args.fig_dir / f"forecast_{g}.png")

        export_and_plot(state=g, y=y, mean=mean, ci=ci, out_csv=out_csv, out_png=out_png)

        print(f"[ok] wrote {out_csv} {'(no figure)' if out_png is None else f'& {out_png}'}")


if __name__ == "__main__":
    main()

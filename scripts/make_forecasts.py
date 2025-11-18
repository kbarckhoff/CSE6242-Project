
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# local utils
from utils_series import list_geos, monthly_series_for_geo


# ---------------- helpers ----------------

def _coerce_monthly(y: pd.Series) -> pd.Series:
    """
    Ensure y is numeric, monthly-start indexed (MS), sorted, and has no NaNs.
    Returns an empty Series (len 0) if nothing usable remains.
    """
    if y is None or len(y) == 0:
        return pd.Series(dtype=float)

    y = pd.Series(
        pd.to_numeric(y.values, errors="coerce"),
        index=pd.to_datetime(y.index),
        dtype="float64",
    ).dropna()

    if y.empty:
        return y

    # Month-start normalized index, sorted
    y = y.sort_index()
    y.index = y.index.to_period("M").to_timestamp(how="start")
    y = y.asfreq("MS")
    return y


def _future_index_from_last(y: pd.Series, steps: int) -> pd.DatetimeIndex:
    last = y.index[-1]
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=steps, freq="MS")


def sarimax_forecast(y: pd.Series, steps: int = 12) -> tuple[pd.Series, pd.DataFrame]:
    """
    Conservative SARIMAX -> (mean, ci). Falls back to a persistence baseline
    if the SARIMAX fit or its CI become non-finite.
    """
    y = _coerce_monthly(y)
    n = len(y.dropna())

    # --- Short series fallback ---
    if n < 18:
        future_idx = _future_index_from_last(y, steps)
        mean = pd.Series(y.iloc[-1], index=future_idx, dtype=float)

        deltas = y.diff().dropna()
        sigma = float(np.nanstd(deltas.iloc[-12:])) if len(deltas) else 0.0
        ci = pd.DataFrame(
            {
                "ci95_lo": mean - 1.96 * sigma,
                "ci95_hi": mean + 1.96 * sigma,
            },
            index=mean.index,
        )
        return mean.astype(float), ci.astype(float)

    # --- SARIMAX model ---
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
    )
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=steps)
    mean = pred.predicted_mean.astype(float)
    ci = pred.conf_int(alpha=0.05).copy()

    # Normalize CI names
    if ci.shape[1] == 2:
        ci.columns = ["ci95_lo", "ci95_hi"]

    # Fallback if CI has any non-finite values
    if not np.isfinite(ci.values).all():
        sigma = float(np.nanstd(res.resid, ddof=1))
        ci = pd.DataFrame(
            {
                "ci95_lo": mean - 1.96 * sigma,
                "ci95_hi": mean + 1.96 * sigma,
            },
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
    Write CSV and optionally save a forecast plot.
    """
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


# ---------------- CLI ----------------

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

    total = len(geos)
    made = 0
    skipped = 0
    failed = 0

    for g in geos:
        try:
            # Pull a clean monthly series for this geo
            y = monthly_series_for_geo(args.parquet, args.geo_type, g, args.value_col)
            y = _coerce_monthly(y)

            # Skip if no usable data
            if y.empty or y.dropna().shape[0] < 2:
                print(f"[skip] {g}: no data")
                skipped += 1
                continue

            # Forecast
            mean, ci = sarimax_forecast(y, steps=args.steps)

            # Outputs
            out_csv = args.out_dir / f"{args.geo_type}-{g}" / "forecast.csv"
            out_png = None if args.no_figures else (args.fig_dir / f"forecast_{g}.png")

            export_and_plot(state=g, y=y, mean=mean, ci=ci, out_csv=out_csv, out_png=out_png)
            print(f"[ok] wrote {out_csv} {'(no figure)' if out_png is None else f'& {out_png}'}")
            made += 1

        except Exception as e:
            print(f"[err] {g}: {type(e).__name__}: {e}")
            failed += 1
            continue

    print(f"\nSummary: total={total}, made={made}, skipped(no data)={skipped}, failed={failed}")


if __name__ == "__main__":
    main()

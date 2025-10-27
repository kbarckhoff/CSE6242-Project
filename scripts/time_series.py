# scripts/time_series.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import matplotlib.pyplot as plt


def load_long(parquet_path: Path) -> pd.DataFrame:
    """
    Load the long/tidy ZORI sample  and return a clean DataFrame.


    """
    df = pd.read_parquet(parquet_path)

    # required columns
    req = {"state", "zip", "date", "zori"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing cols in {parquet_path}: {missing}")

    # Coerce and clean
    df["zori"] = pd.to_numeric(df["zori"], errors="coerce")  # converts from string to floating pt.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep valid rows only
    df = df.dropna(subset=["zori", "date"]).copy()  # drop null values

    # Standardize to month-end. This is already the case but just to guarantee there are no multiple days
    # for a given month
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    # Sort for rolling/pct_change to work correctly
    df = df.sort_values(["state", "zip", "date"])
    return df


def monthly_series_by_state(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Aggregate to monthly mean ZORI for a single state.
    Returns a DataFrame indexed by date with columns: ['zori', 'roll12', 'yoy']
    """
    df_state = df.loc[df["state"] == state, ["date", "zori"]].copy()
    if df_state.empty:
        raise ValueError(f"No rows found for state '{state}'. "
                         f"Available: {sorted(df['state'].unique().tolist())}")

    # Monthly mean ZORI across all zips in the state
    ts = (
        df_state.groupby("date", as_index=True)["zori"]
        .mean()
        .sort_index()
        .to_frame()
    )

    # 12-month rolling mean
    ts["roll12"] = ts["zori"].rolling(window=12, min_periods=12).mean()

    # YoY percentage (12-month change)
    ts["yoy"] = ts["zori"].pct_change(12)

    # Drop rows where everything is NA (early months)
    ts = ts.dropna(how="all")
    return ts


def plot_state_series(ts: pd.DataFrame, state: str, out_png: Path) -> None:
    """
    Plot ZORI, 12m rolling mean, and YoY% for a state.

    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # x-axis: use 'date' column if present, otherwise the index
    x = ts["date"] if "date" in ts.columns else ts.index

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- main axis: ZORI + 12m rolling mean (distinct colors) ---
    lines, labels = [], []

    (l1,) = ax.plot(x, ts["zori"], label="ZORI", color="tab:blue", linewidth=2)
    lines.append(l1); labels.append(l1.get_label())

    if "roll12" in ts.columns and ts["roll12"].notna().any():
        (l2,) = ax.plot(x, ts["roll12"], label="12m rolling mean", color="tab:green", linewidth=2)
        lines.append(l2); labels.append(l2.get_label())

    ax.set_title(f"{state}: ZORI & 12m rolling mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("ZORI (index)")

    # --- secondary axis: YoY % (fixed color, separate axis) ---
    ax2 = ax.twinx()
    if "yoy" in ts.columns and ts["yoy"].notna().any():
        (l3,) = ax2.plot(x, ts["yoy"], label="YoY %", color="tab:orange", linewidth=2, alpha=0.9)
        y = ts["yoy"].dropna()
        if not y.empty:
            ymin = y.min() - 0.01
            ymax = y.max() + 0.01
            if ymin == ymax:
                ymin -= 0.01; ymax += 0.01
            ax2.set_ylim(ymin, ymax)
        ax2.set_ylabel("YoY change (%)")
        lines.append(l3); labels.append(l3.get_label())

    # Optional: match your other chart’s cleaner frame
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # one combined legend
    ax.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")



def main(args=None):
    p = argparse.ArgumentParser(description="Plot state time-series from small ZORI parquet.")
    p.add_argument("--parquet", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "zori_long_small.parquet"),
                   help="Path to the small tidy parquet created by the ingest step.")
    p.add_argument("--states", nargs="+", default=["TX", "GA"],
                   help="State codes to plot.")
    p.add_argument("--out_dir", type=str, default="figures",
                   help="Directory to write figures (png).")
    a = p.parse_args(args)

    parquet_path = Path(a.parquet).expanduser()
    out_dir = Path(a.out_dir)

    df = load_long(parquet_path)

    for st in a.states:
        ts = monthly_series_by_state(df, st)
        plot_state_series(ts, st, out_dir / f"time_series_{st}.png")


if __name__ == "__main__":
    main()

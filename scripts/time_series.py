# TODO: Create time series plots using the parquet file
from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from utils_series import list_geos, monthly_series_for_geo


def plot_state_series(ts, state: str, out_png: Path | None, make_figure: bool = True) -> None:
    """Plot ZORI + 12m rolling mean for one geo."""
    if not make_figure:
        return

    ts = ts.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts.index, ts.values, label="ZORI", linewidth=2)

    rm = ts.rolling(12, min_periods=6).mean()
    ax.plot(rm.index, rm.values, label="12m rolling mean", linewidth=2)

    ax.set_title(f"{state}: ZORI & 12m rolling mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("ZORI (index)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160)

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--geo-type", default="state")
    p.add_argument(
        "--geos",
        nargs="*",
        help="List of geos; omit or pass ALL to process all geos in the parquet",
    )
    p.add_argument("--value-col", default="zori_smoothed_seasonal")
    p.add_argument("--out-dir", type=Path, default=Path("figures/time_series"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    geos = args.geos
    if not geos or (len(geos) == 1 and geos[0].upper() == "ALL"):
        geos = list_geos(args.parquet, args.geo_type)

    for g in geos:
        ts = monthly_series_for_geo(args.parquet, args.geo_type, g, args.value_col)
        out_png = None if args.no_figures else args.out_dir / f"time_series_{g}.png"
        plot_state_series(ts, g, out_png, make_figure=not args.no_figures)

        if out_png:
            print(f"[time_series] wrote {out_png}")


if __name__ == "__main__":
    main()

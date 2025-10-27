# scripts/make_figures.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_zip_counts_by_state(by_state_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(by_state_csv)
    df = df.sort_values("n_zips", ascending=False)

    ax = df.plot.bar(
        x="state", y="n_zips", legend=False, figsize=(7, 4),
        title="ZIP coverage by state (sample)"
    )
    ax.set_xlabel("State")
    ax.set_ylabel("# ZIPs in sample")

    # 🔹 remove the top border line
    ax.spines["top"].set_visible(False)
    # (optional: also remove right border for a cleaner look)
    # ax.spines["right"].set_visible(False)

    # headroom + labels (keep from earlier)
    ymax = float(df["n_zips"].max())
    ax.set_ylim(0, ymax * 1.20)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h)}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center", va="bottom",
            fontsize=9, xytext=(0, 3), textcoords="offset points"
        )

    ax.figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out_png, dpi=150)
    print(f"Wrote figure -> {out_png}")


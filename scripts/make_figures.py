# scripts/make_figures.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_zip_counts_by_state(by_state_csv: Path, out_png: Path) -> None:
    """
    Read the eda_by_state.csv (columns: state, n_zips) and write a simple bar chart.

    - Sorts states by n_zips descending
    - Annotates bar values
    """
    df = pd.read_csv(by_state_csv)
    if not {"state", "n_zips"}.issubset(df.columns):
        raise ValueError(
            f"{by_state_csv} must have columns: 'state', 'n_zips' (got {list(df.columns)})"
        )

    # Sort (descending)
    df = df.sort_values("n_zips", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["state"], df["n_zips"])
    ax.set_title("ZIP coverage by state (sample)")
    ax.set_xlabel("State")
    ax.set_ylabel("# ZIPs in sample")

    # annotate counts on top
    for x, y in zip(df["state"], df["n_zips"]):
        ax.text(x, y + max(df["n_zips"]) * 0.02, str(int(y)), ha="center", va="bottom")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote figure -> {out_png}")


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    by_state_csv = Path(args.by_state_csv).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "zip_counts_by_state.png"

    plot_zip_counts_by_state(by_state_csv, out_png)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make simple figures from sample EDA CSVs.")
    p.add_argument(
        "--by_state_csv",
        default="data/samples/eda_by_state.csv",
        help="Path to eda_by_state.csv (from ingest step).",
    )
    p.add_argument(
        "--out_dir",
        default="figures",
        help="Directory to write figures (png).",
    )
    main(p.parse_args())

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from typing import Iterable, List


def run(cmd: list[str]) -> None:
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def geo_col_name(geo_type: str) -> str:
    return {"zip": "zip", "state": "state", "metro": "RegionName"}[geo_type]


def chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="main.py")

    # I/O
    p.add_argument("--raw-csv", required=True)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--forecasts-dir", default="data/processed/forecasts")
    p.add_argument("--volatility-dir", default="data/processed/volatility")
    p.add_argument("--metrics-dir", default="data/processed/metrics")
    p.add_argument("--final-table", default="data/processed/forecasts/Rental_forecasts_sarimax_all.csv")

    # Scope/options
    p.add_argument("--subset-states", nargs="+", default=None)
    p.add_argument("--geo-type", choices=["zip", "state", "metro"], default="zip")
    p.add_argument("--geos", nargs="+", default=["ALL"])
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 9, 12])
    p.add_argument("--vol-window", type=int, default=12)
    p.add_argument("--include-forecast", action="store_true")

    # Skips (when testing)
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--no-figures", action="store_true")

    # Windows safety
    p.add_argument("--chunk-size", type=int, default=400,
                   help="Max geos passed per subprocess call to avoid Windows command length limits.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    forecasts_dir = Path(args.forecasts_dir)
    volatility_dir = Path(args.volatility_dir)
    metrics_dir = Path(args.metrics_dir)
    final_table = Path(args.final_table)

    processed_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    volatility_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_metrics:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    final_table.parent.mkdir(parents=True, exist_ok=True)

    # 1) CSV -> parquet
    parquet_path = processed_dir / "zori_smoothed_seasonal.parquet"
    cmd = [
        sys.executable, "scripts/make_smoothed_seasonal.py",
        "--csv", args.raw_csv,
        "--out", str(parquet_path),
    ]
    if args.subset_states:
        cmd += ["--subset-states", *args.subset_states]
    run(cmd)

    # Determine geos
    if args.geos == ["ALL"]:
        geo_col = geo_col_name(args.geo_type)
        uniques = pd.read_parquet(parquet_path, columns=[geo_col])[geo_col].dropna().astype(str).unique()
        target_geos = sorted(uniques.tolist())
    else:
        target_geos = [str(g) for g in args.geos]

    # 2) Forecasts
    for batch in chunks(target_geos, args.chunk_size):
        cmd = [
            sys.executable, "scripts/make_forecasts.py",
            "--parquet", str(parquet_path),
            "--geo-type", args.geo_type,
            "--geos", *batch,
            "--out-dir", str(forecasts_dir),
        ]
        if args.skip_plots or args.no_figures:
            cmd.append("--no-figures")
        run(cmd)

    # 3) Volatility
    for batch in chunks(target_geos, args.chunk_size):
        cmd = [
            sys.executable, "scripts/volatility_index.py",
            "--parquet", str(parquet_path),
            "--geo-type", args.geo_type,
            "--geos", *batch,
            "--window", str(args.vol_window),
            "--out-dir", str(volatility_dir),
        ]
        if args.include_forecast:
            cmd.append("--include-forecast")
        run(cmd)

    # 4) Metrics
    if not args.skip_metrics:
        for batch in chunks(target_geos, args.chunk_size):
            cmd = [
                sys.executable, "scripts/evaluate_rmse.py",
                "--parquet", str(parquet_path),
                "--geo-type", args.geo_type,
                "--geos", *batch,
                "--out-dir", str(metrics_dir),
            ]
            run(cmd)

    # 5) Final csv export table
    cmd = [
        sys.executable, "scripts/export_forecast_table.py",
        "--forecasts-dir", str(forecasts_dir),
        "--volatility-dir", str(volatility_dir),
        "--geo-type", args.geo_type,
        "--horizons", *[str(h) for h in args.horizons],
        "--out-csv", str(final_table),
    ]
    run(cmd)

    print("\n Pipeline complete.")
    print(f"   Final table: {final_table}")
    if not args.skip_metrics:
        print(f"   Metrics:     {metrics_dir}")


if __name__ == "__main__":
    main()

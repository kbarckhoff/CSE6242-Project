# TODO: Scripts
import argparse
import glob
from pathlib import Path

import pandas as pd


def main(args):
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_csv = Path(args.output_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csvs = sorted(glob.glob(str(in_dir / "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {in_dir}")

    print(f"Reading: {csvs[0]}")
    df = pd.read_csv(csvs[0], low_memory=False)

    # Small, deterministic sample of n rows from the big zillow data [to run locally and check for trends]
    n = min(len(df), args.nrows)
    sample = df.sample(n=n, random_state=42)

    sample.to_csv(out_csv, index=False)
    print(f"Wrote {len(sample):,} rows -> {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make a small sample from a big Zillow CSV.")
    p.add_argument(
        "--input_dir",
        required=True,
        help="Folder that contains the raw Zillow CSV(s).",
    )
    p.add_argument(
        "--output_csv",
        default="data/samples/zori_sample.csv",
        help="Path to write the sample CSV (inside repo).",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=1000,
        help="Number of rows to sample (default: 1000).",
    )
    main(p.parse_args())

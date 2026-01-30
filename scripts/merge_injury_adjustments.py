from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge injury adjustment CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths.")
    parser.add_argument("--output", default="data/injuries_combined.csv", help="Output CSV path.")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
    merged = merged.dropna(subset=["date", "player_name", "minutes_multiplier"])
    merged = merged.drop_duplicates(subset=["date", "player_name"], keep="last")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

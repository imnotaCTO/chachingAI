from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from sgp_engine.matching import normalize_player_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer historical injury minute multipliers from Kaggle logs.")
    parser.add_argument("--kaggle-path", default="PlayerStatistics.csv", help="Path to Kaggle PlayerStatistics.csv.")
    parser.add_argument("--start", default="2023-05-03", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--window", type=int, default=10, help="Rolling window for baseline minutes.")
    parser.add_argument("--min-games", type=int, default=5, help="Minimum games to compute baseline.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Only emit multipliers below this threshold (or minutes=0).",
    )
    parser.add_argument("--output", default="data/injuries_inferred.csv", help="Output CSV path.")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    usecols = [
        "firstName",
        "lastName",
        "gameDateTimeEst",
        "numMinutes",
        "playerteamCity",
        "playerteamName",
        "opponentteamCity",
        "opponentteamName",
    ]
    df = pd.read_csv(args.kaggle_path, usecols=usecols)
    df["date"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    if end_date:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    else:
        df = df[df["date"] >= start_date]

    df["player_name"] = (
        df["firstName"].fillna("").astype(str).str.strip()
        + " "
        + df["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["team"] = (
        df["playerteamCity"].fillna("").astype(str).str.strip()
        + " "
        + df["playerteamName"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["opponent"] = (
        df["opponentteamCity"].fillna("").astype(str).str.strip()
        + " "
        + df["opponentteamName"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["player_norm"] = df["player_name"].map(normalize_player_name)
    df["minutes"] = pd.to_numeric(df["numMinutes"], errors="coerce")
    df = df.dropna(subset=["player_norm", "minutes"])

    rows = []
    for player, group in df.groupby("player_norm"):
        group = group.sort_values("date")
        minutes = group["minutes"].astype(float)
        baseline = minutes.shift(1).rolling(args.window, min_periods=args.min_games).mean()
        for idx, row in group.iterrows():
            base = baseline.loc[idx]
            if pd.isna(base) or base <= 0:
                continue
            multiplier = float(row["minutes"]) / float(base)
            if row["minutes"] == 0 or multiplier < args.threshold:
                rows.append(
                    {
                        "date": row["date"],
                        "player_name": row["player_name"],
                        "minutes_multiplier": max(0.0, min(1.0, multiplier)),
                        "status": "inferred",
                        "source": "kaggle",
                        "team": row.get("team"),
                        "opponent": row.get("opponent"),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "player_name", "minutes_multiplier", "status", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} inferred injury rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sgp_engine.ingestion.balldontlie import BallDontLieClient


DEFAULT_COLUMNS = [
    "game_id",
    "date",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "location",
    "minutes",
    "points",
    "rebounds",
    "assists",
]


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_minutes(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "DNP", "DND"}:
        return None
    if ":" in text:
        mins, secs = text.split(":", 1)
        try:
            return float(mins) + float(secs) / 60.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_stats_range(
    client: BallDontLieClient,
    season: int,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    rows: list[dict] = []
    cursor: int | None = None
    while True:
        params: dict[str, object] = {
            "seasons[]": season,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor
        payload = client.get_stats(params)
        data = payload.get("data", [])
        if not data:
            break
        for row in data:
            player = row.get("player", {})
            game = row.get("game", {})
            team = row.get("team", {})
            team_id = team.get("id")
            home_team = game.get("home_team", {})
            visitor_team = game.get("visitor_team", {})
            opponent = None
            location = None
            if team_id and game:
                if team_id == home_team.get("id"):
                    opponent = visitor_team.get("abbreviation")
                    location = "Home"
                elif team_id == visitor_team.get("id"):
                    opponent = home_team.get("abbreviation")
                    location = "Away"
            rows.append(
                {
                    "game_id": game.get("id"),
                    "date": pd.to_datetime(game.get("date")) if game.get("date") else pd.NaT,
                    "player_id": str(player.get("id")) if player.get("id") else None,
                    "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                    "team": team.get("abbreviation") or team.get("full_name"),
                    "opponent": opponent,
                    "location": location,
                    "minutes": parse_minutes(row.get("min")),
                    "points": row.get("pts"),
                    "rebounds": row.get("reb"),
                    "assists": row.get("ast"),
                }
            )
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = int(next_cursor)
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df
    return df.dropna(subset=["player_id", "date"])


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if "game_id" in df.columns and df["game_id"].notna().any():
        return df.drop_duplicates(subset=["player_id", "game_id"])
    return df.drop_duplicates(subset=["player_id", "date", "team", "opponent"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync BallDontLie stats to a local cache file.")
    parser.add_argument("--season", type=int, required=True, help="Season end year (e.g., 2026).")
    parser.add_argument(
        "--output",
        default="data/bdl_stats.parquet",
        help="Output file (parquet or csv).",
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="daily",
        help="Backfill full season or append recent days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to fetch in daily mode.",
    )
    parser.add_argument("--start-date", default=None, help="Optional backfill start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Optional backfill end date (YYYY-MM-DD).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise SystemExit("Missing BALLDONTLIE_API_KEY.")
    auth_scheme = os.environ.get("BALLDONTLIE_AUTH_SCHEME")
    base_url = os.environ.get("BALLDONTLIE_BASE_URL") or "https://api.balldontlie.io/v1"

    client = BallDontLieClient(api_key=api_key, base_url=base_url, auth_scheme=auth_scheme)

    output_path = Path(args.output)
    existing = load_existing(output_path)

    if args.mode == "backfill":
        if args.start_date:
            start = datetime.fromisoformat(args.start_date)
        else:
            start = datetime(args.season - 1, 10, 1, tzinfo=timezone.utc)
        if args.end_date:
            end = datetime.fromisoformat(args.end_date)
        else:
            end = datetime(args.season, 6, 30, tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        end = now
        start = now - timedelta(days=args.days)

    fetched = fetch_stats_range(client, args.season, start, end)
    if fetched.empty:
        print("No stats returned for requested range.")
        return 0

    combined = pd.concat([existing, fetched], ignore_index=True)
    combined = dedupe(combined)
    combined = combined.sort_values(["date", "player_name"])
    save_df(combined, output_path)
    print(f"Wrote {len(combined)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

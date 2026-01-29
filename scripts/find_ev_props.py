from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from sgp_engine.ingestion import (
    OddsAPIClient,
    extract_player_props,
    fetch_player_game_logs_by_name,
    fetch_player_game_logs_by_name_kaggle,
    fetch_player_game_logs_by_name_nba_api,
)
from sgp_engine.matching import normalize_player_name
from sgp_engine.modeling import fit_lognormal_params
from sgp_engine.odds import expected_value, prob_to_american

DEFAULT_STAT_ORDER = ("points", "rebounds", "assists")
DEFAULT_MARKET_MAP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
}


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


def parse_date(value: str) -> datetime.date:
    if value.lower() == "today":
        return datetime.now().date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def event_in_date(event: dict, target_date: datetime.date) -> bool:
    commence = event.get("commence_time")
    if not commence:
        return False
    try:
        dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except ValueError:
        return False
    local_dt = dt.astimezone()
    return local_dt.date() == target_date


def _format_event(event: dict, index: int) -> str:
    home = event.get("home_team") or "Home"
    away = event.get("away_team") or "Away"
    start = event.get("commence_time") or "TBD"
    return f"[{index}] {away} @ {home} | {start} | {event.get('id')}"


def _choose_event(events: list[dict]) -> dict:
    print("Available events:")
    for idx, event in enumerate(events):
        print(_format_event(event, idx))
    choice = input("Select event number: ").strip()
    if not choice.isdigit():
        raise SystemExit("Invalid selection.")
    index = int(choice)
    if index < 0 or index >= len(events):
        raise SystemExit("Selection out of range.")
    return events[index]


def _cache_filename(name: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")
    return safe or "unknown"


def _load_cached_logs(
    cache_dir: Path,
    source: str,
    season: int,
    player_name: str,
    ttl_hours: float,
) -> pd.DataFrame | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if ttl_hours >= 0 and age_hours > ttl_hours:
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _save_cached_logs(cache_dir: Path, source: str, season: int, player_name: str, logs: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    logs.to_pickle(path)


def lognormal_hit_probability(mu: float, sigma: float, line: float, direction: str) -> float:
    if line <= -1:
        raise ValueError("line must be greater than -1 for log1p")
    z = np.log1p(line)
    cdf = norm.cdf(z, loc=mu, scale=sigma)
    if direction.lower() == "over":
        return float(1 - cdf)
    if direction.lower() == "under":
        return float(cdf)
    raise ValueError(f"Unsupported direction: {direction}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find positive EV NBA player props for a slate.")
    parser.add_argument("--date", default="today", help="Date for slate (YYYY-MM-DD or 'today').")
    parser.add_argument("--season", type=int, default=2024, help="NBA season year for logs.")
    parser.add_argument(
        "--stats-source",
        default="nba_api",
        choices=["nba_api", "balldontlie", "kaggle"],
        help="Source for player game logs.",
    )
    parser.add_argument(
        "--kaggle-path",
        default="PlayerStatistics.csv",
        help="Path to Kaggle PlayerStatistics.csv.",
    )
    parser.add_argument("--min-games", type=int, default=15, help="Minimum games required.")
    parser.add_argument("--min-minutes", type=float, default=20.0, help="Minimum minutes per game.")
    parser.add_argument("--min-ev", type=float, default=0.03, help="Minimum EV threshold.")
    parser.add_argument(
        "--stats",
        nargs="+",
        default=list(DEFAULT_STAT_ORDER),
        help="Stats to include (points rebounds assists).",
    )
    parser.add_argument("--max-results", type=int, default=50, help="Max rows to print.")
    parser.add_argument("--output-csv", default=None, help="CSV output path.")
    parser.add_argument("--regions", default="us", help="The Odds API regions.")
    parser.add_argument("--sportsbook", default="DraftKings", help="Filter props to a sportsbook title.")
    parser.add_argument("--event-id", default=None, help="Optional Odds API event id to target.")
    parser.add_argument("--cache-dir", default=".cache/ev_props", help="Directory for player log cache.")
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=12.0,
        help="Cache TTL in hours. Use -1 to never expire.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N player log fetches.",
    )
    args = parser.parse_args()

    load_dotenv()
    odds_api_key = os.environ.get("ODDS_API_KEY")
    if not odds_api_key:
        raise SystemExit("Missing ODDS_API_KEY. Set it in the environment or .env file.")
    bdl_api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if args.stats_source == "balldontlie" and not bdl_api_key:
        raise SystemExit("Missing BALLDONTLIE_API_KEY for balldontlie stats source.")

    target_date = parse_date(args.date)
    stats = [stat.lower() for stat in args.stats]
    market_map = DEFAULT_MARKET_MAP

    client = OddsAPIClient(api_key=odds_api_key)
    events = client.get_events()
    events = [event for event in events if event_in_date(event, target_date)]
    if not events:
        raise SystemExit(f"No events found for {target_date.isoformat()}.")
    if args.event_id:
        selected = [event for event in events if event.get("id") == args.event_id]
        if not selected:
            raise SystemExit("Event id not found in today's slate.")
        events = selected
    else:
        events = [_choose_event(events)]

    results: list[dict] = []
    player_cache: dict[str, pd.DataFrame] = {}
    cache_dir = Path(args.cache_dir)
    fetched_logs = 0

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        print(f"Fetching props for {event.get('away_team')} @ {event.get('home_team')} ({event_id})")
        odds_payload = client.get_event_odds(event_id, regions=args.regions)
        if isinstance(odds_payload, dict):
            odds_payload = [odds_payload]
        props = extract_player_props(odds_payload)
        if args.sportsbook:
            target_book = str(args.sportsbook).strip().lower()
            props = [
                prop
                for prop in props
                if str(prop.get("sportsbook", "")).strip().lower() == target_book
            ]
        if not props:
            continue
        unique_players = {
            normalize_player_name(str(prop.get("player_name", "")))
            for prop in props
            if prop.get("player_name")
        }
        if unique_players:
            print(f"Found {len(unique_players)} players with props.")
        for prop in props:
            market = prop.get("market_type")
            if market not in market_map:
                continue
            stat = market_map[market]
            if stat not in stats:
                continue
            player_name = prop.get("player_name")
            line = prop.get("line")
            odds = prop.get("odds")
            direction = prop.get("direction") or "over"
            sportsbook = prop.get("sportsbook")
            if not player_name or line is None or odds is None:
                continue

            normalized = normalize_player_name(str(player_name))
            if normalized not in player_cache:
                try:
                    cached = _load_cached_logs(
                        cache_dir,
                        args.stats_source,
                        args.season,
                        str(player_name),
                        args.cache_ttl_hours,
                    )
                    if cached is not None:
                        logs = cached
                    else:
                        if args.stats_source == "nba_api":
                            logs = fetch_player_game_logs_by_name_nba_api(
                                player_name=str(player_name),
                                season=args.season,
                            )
                        elif args.stats_source == "kaggle":
                            logs = fetch_player_game_logs_by_name_kaggle(
                                player_name=str(player_name),
                                season_end_year=args.season,
                                data_path=args.kaggle_path,
                            )
                        else:
                            logs = fetch_player_game_logs_by_name(
                                player_name=str(player_name),
                                season=args.season,
                                api_key=bdl_api_key,
                            )
                        _save_cached_logs(
                            cache_dir,
                            args.stats_source,
                            args.season,
                            str(player_name),
                            logs,
                        )
                except Exception:
                    continue
                fetched_logs += 1
                if args.progress_every > 0 and fetched_logs % args.progress_every == 0:
                    print(f"Loaded logs for {fetched_logs} players...")
                player_cache[normalized] = logs
            logs = player_cache[normalized]
            if logs.empty:
                continue
            filtered = logs[logs["minutes"] >= args.min_minutes]
            if filtered.shape[0] < args.min_games:
                continue

            samples = filtered[stat].dropna().to_numpy()
            if samples.size < args.min_games:
                continue
            mu, sigma = fit_lognormal_params(samples.astype(float))
            probability = lognormal_hit_probability(mu, sigma, float(line), str(direction))
            fair_odds = prob_to_american(min(max(probability, 1e-6), 1 - 1e-6))
            ev = expected_value(probability, float(odds))
            if ev < args.min_ev:
                continue

            results.append(
                {
                    "game": f"{event.get('away_team')} @ {event.get('home_team')}",
                    "player": player_name,
                    "stat": stat,
                    "line": line,
                    "direction": direction,
                    "odds": odds,
                    "sportsbook": sportsbook,
                    "model_prob": probability,
                    "fair_odds": fair_odds,
                    "ev": ev,
                    "sample_size": int(samples.size),
                    "avg_minutes": float(filtered["minutes"].mean()),
                    "event_id": event_id,
                }
            )

    if not results:
        raise SystemExit("No props met the EV threshold.")

    df = pd.DataFrame(results)
    df = df.sort_values(by="ev", ascending=False)

    # Keep the best EV per player/stat/line/direction
    df = df.sort_values(by="ev", ascending=False).drop_duplicates(
        subset=["player", "stat", "line", "direction"], keep="first"
    )

    output_path = args.output_csv
    if output_path is None:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ev_props_{target_date.strftime('%Y%m%d')}.csv"

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    print(df.head(args.max_results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

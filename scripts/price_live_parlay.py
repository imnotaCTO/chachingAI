from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Iterable

import pandas as pd

from sgp_engine.ingestion import (
    OddsAPIClient,
    extract_player_props,
    fetch_player_game_logs_by_name,
    fetch_player_game_logs_by_name_kaggle,
    fetch_player_game_logs_by_name_nba_api,
)
from sgp_engine.ingestion.basketball_reference import (
    fetch_player_game_logs_by_name as fetch_player_game_logs_by_name_bbr,
)
from sgp_engine.matching import match_props_to_player_ids, normalize_player_name, props_to_leg_specs
from sgp_engine.pipeline import price_parlay_from_samples

STAT_COLUMN_CANDIDATES = {
    "points": ("points", "pts"),
    "rebounds": ("rebounds", "reb"),
    "assists": ("assists", "ast"),
}


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


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def resolve_stat_columns(player_logs: pd.DataFrame, stats: Iterable[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for stat in stats:
        candidates = STAT_COLUMN_CANDIDATES.get(stat, ())
        column = pick_column(player_logs.columns, candidates)
        if column is None:
            raise ValueError(f"Missing stat column for {stat}. Available columns: {list(player_logs.columns)}")
        resolved[stat] = column
    return resolved


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


def _filter_events_by_team(events: list[dict], team_query: str) -> list[dict]:
    query = team_query.strip().lower()
    if not query:
        return events
    filtered = []
    for event in events:
        home = str(event.get("home_team", "")).lower()
        away = str(event.get("away_team", "")).lower()
        if query in home or query in away:
            filtered.append(event)
    return filtered


def _select_player_from_props(props: list[dict]) -> str | None:
    counts: dict[str, int] = {}
    for prop in props:
        name = prop.get("player_name")
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def main() -> int:
    parser = argparse.ArgumentParser(description="Price a live parlay from Odds API props.")
    parser.add_argument("--player", default=None, help="Player name to price (e.g., Nikola Jokic).")
    parser.add_argument("--player-id", type=int, default=None, help="Optional BallDontLie player id override.")
    parser.add_argument("--season", type=int, default=2024, help="NBA season year for logs.")
    parser.add_argument(
        "--stats-source",
        default="balldontlie",
        choices=["balldontlie", "nba_api", "bbr", "kaggle"],
        help="Source for player game logs.",
    )
    parser.add_argument(
        "--kaggle-path",
        default="PlayerStatistics.csv",
        help="Path to Kaggle PlayerStatistics.csv.",
    )
    parser.add_argument("--sportsbook-odds", type=float, required=True, help="Parlay odds from sportsbook.")
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["points", "rebounds", "assists"],
        help="Stats to include in order (e.g., points assists).",
    )
    parser.add_argument(
        "--auto-player",
        action="store_true",
        help="Auto-select a player from the selected event if --player is omitted or missing.",
    )
    parser.add_argument(
        "--team",
        default=None,
        help="Filter events by team name (e.g., Nuggets). Defaults to player's team when available.",
    )
    parser.add_argument(
        "--direction",
        default="over",
        help="Direction to use when selecting props (over or under).",
    )
    parser.add_argument("--event-id", default=None, help="Optional Odds API event id to target.")
    parser.add_argument(
        "--list-players",
        action="store_true",
        help="List player names available in props for the selected event and exit.",
    )
    parser.add_argument("--allow-last-name-match", action="store_true", help="Enable conservative fuzzy matching.")
    parser.add_argument("--bdl-base-url", default=None, help="Override BallDontLie base URL.")
    parser.add_argument(
        "--bdl-auth-scheme",
        default=None,
        help="Override BallDontLie auth scheme (e.g., Bearer).",
    )
    parser.add_argument(
        "--allow-scrape-fallback",
        action="store_true",
        help="Fall back to scraping Basketball Reference if BallDontLie fails.",
    )
    parser.add_argument(
        "--bbr-player-id",
        default=None,
        help="Basketball Reference player id (e.g., jokicni01) for scrape fallback.",
    )
    parser.add_argument(
        "--bbr-html-path",
        default=None,
        help="Optional path to a saved Basketball Reference game log HTML file.",
    )
    args = parser.parse_args()

    load_dotenv()
    odds_api_key = os.environ.get("ODDS_API_KEY")
    if not odds_api_key:
        raise SystemExit("Missing ODDS_API_KEY. Set it in the environment or .env file.")
    bdl_api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if args.stats_source == "balldontlie" and not bdl_api_key and not args.allow_scrape_fallback:
        raise SystemExit("Missing BALLDONTLIE_API_KEY. Set it in the environment or .env file.")
    bdl_base_url = args.bdl_base_url or os.environ.get("BALLDONTLIE_BASE_URL")
    bdl_auth_scheme = args.bdl_auth_scheme or os.environ.get("BALLDONTLIE_AUTH_SCHEME")

    player_logs = None

    stat_order = [stat.lower() for stat in args.stats]
    stat_columns = resolve_stat_columns(player_logs, stat_order)
    available = player_rows.dropna(subset=list(stat_columns.values()))
    if available.shape[0] < 2:
        raise SystemExit("Not enough historical samples after filtering for required stats.")

    samples = available[[stat_columns[stat] for stat in stat_order]].to_numpy()

    client = OddsAPIClient(api_key=odds_api_key)
    if args.event_id:
        event_id = args.event_id
        event = None
    else:
        events = client.get_events()
        if not events:
            raise SystemExit("No events returned from The Odds API.")
        team_query = args.team
        if team_query:
            filtered_events = _filter_events_by_team(events, team_query)
            if filtered_events:
                events = filtered_events
            else:
                print(f"No events matched team filter '{team_query}'. Showing all events.")
        event = _choose_event(events)
        event_id = event.get("id")
    odds_payload = client.get_event_odds(event_id)
    if isinstance(odds_payload, dict):
        odds_payload = [odds_payload]
    props = extract_player_props(odds_payload)
    if args.list_players:
        names = sorted({prop.get("player_name") for prop in props if prop.get("player_name")})
        print("Players with props for this event:")
        for name in names:
            print(f"- {name}")
        return 0
    auto_player = args.auto_player or args.player is None
    selected_player = args.player
    if selected_player is None:
        selected_player = _select_player_from_props(props)
        if selected_player is None:
            raise SystemExit("No player props available for this event.")
        print(f"Auto-selected player: {selected_player}")
    normalized_target = normalize_player_name(selected_player)
    prop_names = {normalize_player_name(str(prop.get('player_name', ''))) for prop in props}
    if normalized_target not in prop_names:
        names = sorted({prop.get("player_name") for prop in props if prop.get("player_name")})
        print("Selected player not found in this event. Available players:")
        for name in names:
            print(f"- {name}")
        raise SystemExit("No matching props found for the player.")

    if args.stats_source == "balldontlie" and not bdl_api_key and not args.allow_scrape_fallback:
        raise SystemExit("Missing BALLDONTLIE_API_KEY. Set it in the environment or .env file.")

    if args.stats_source == "nba_api":
        player_logs = fetch_player_game_logs_by_name_nba_api(
            player_name=selected_player,
            season=args.season,
        )
    elif args.stats_source == "kaggle":
        player_logs = fetch_player_game_logs_by_name_kaggle(
            player_name=selected_player,
            season_end_year=args.season,
            data_path=args.kaggle_path,
        )
    elif args.stats_source == "bbr":
        try:
            html_text = None
            if args.bbr_html_path:
                with open(args.bbr_html_path, "r", encoding="utf-8") as handle:
                    html_text = handle.read()
            player_logs = fetch_player_game_logs_by_name_bbr(
                player_name=selected_player,
                season=args.season,
                player_id=args.bbr_player_id,
                html_text=html_text,
            )
        except ValueError as exc:
            if args.bbr_player_id:
                raise
            raise SystemExit(
                f"{exc}. Provide --bbr-player-id (e.g., jokicni01) to bypass lookup."
            ) from exc
    else:
        if bdl_api_key:
            try:
                player_logs = fetch_player_game_logs_by_name(
                    player_name=selected_player,
                    season=args.season,
                    api_key=bdl_api_key,
                    player_id=args.player_id,
                    allow_last_name_match=args.allow_last_name_match,
                    base_url=bdl_base_url or "https://api.balldontlie.io/v1",
                    auth_scheme=bdl_auth_scheme,
                )
            except Exception as exc:
                if not args.allow_scrape_fallback:
                    raise
                print(f"BallDontLie failed ({exc}); falling back to Basketball Reference scrape.")

        if player_logs is None:
            try:
                html_text = None
                if args.bbr_html_path:
                    with open(args.bbr_html_path, "r", encoding="utf-8") as handle:
                        html_text = handle.read()
                player_logs = fetch_player_game_logs_by_name_bbr(
                    player_name=selected_player,
                    season=args.season,
                    player_id=args.bbr_player_id,
                    html_text=html_text,
                )
            except ValueError as exc:
                if args.bbr_player_id:
                    raise
                raise SystemExit(
                    f"{exc}. Provide --bbr-player-id (e.g., jokicni01) to bypass lookup."
                ) from exc
    player_rows = player_logs.copy()
    matched = match_props_to_player_ids(
        props,
        player_logs,
        allow_last_name_match=args.allow_last_name_match,
    )

    target_id = None
    if "player_id" in player_rows.columns and not player_rows["player_id"].isna().all():
        target_id = str(player_rows["player_id"].iloc[0])

    filtered = []
    for prop in matched:
        if target_id and prop.get("player_id") == target_id:
            filtered.append(prop)
        elif not target_id and normalize_player_name(str(prop.get("player_name", ""))) == normalized_target:
            filtered.append(prop)

    if args.event_id:
        filtered = [prop for prop in filtered if prop.get("event_id") == args.event_id]

    if not filtered:
        raise SystemExit("No matching props found for the player.")

    if not args.event_id:
        counts = Counter(prop.get("event_id") for prop in filtered)
        event_id = counts.most_common(1)[0][0]
        filtered = [prop for prop in filtered if prop.get("event_id") == event_id]

    desired_direction = args.direction.lower()
    filtered = [
        prop
        for prop in filtered
        if str(prop.get("direction", "")).lower() == desired_direction
    ]

    if not filtered:
        raise SystemExit("No props found matching the requested direction.")

    legs = props_to_leg_specs(
        filtered,
        stat_order=stat_order,
        market_map=DEFAULT_MARKET_MAP,
    )
    if not legs:
        raise SystemExit("No legs generated from props. Check stat list or markets.")

    game = None
    if filtered:
        home = filtered[0].get("game")
        away = filtered[0].get("opponent")
        if home and away:
            game = f"{home} vs {away}"

    recommendation = price_parlay_from_samples(
        samples,
        legs,
        sportsbook_odds=args.sportsbook_odds,
        game=game,
    )

    print("Game:", recommendation.game)
    print("Joint probability:", recommendation.joint_probability)
    print("Model fair odds:", recommendation.model_fair_odds)
    print("Expected value:", recommendation.expected_value)
    print("Legs:")
    for leg in recommendation.legs:
        print(
            f"  {leg.player} {leg.stat} {leg.direction} {leg.line} -> {leg.probability:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Literal

import pandas as pd

SeasonType = Literal["Regular Season", "Playoffs"]


class NBAStatsImportError(ImportError):
    """Raised when nba_api is unavailable."""


def _require_nba_api():
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players
    except ImportError as exc:
        raise NBAStatsImportError(
            "nba_api is required. Install with `pip install nba_api`."
        ) from exc
    return playergamelog, players


def _format_season(season: int) -> str:
    start_year = season - 1
    end_year = season
    return f"{start_year}-{str(end_year)[-2:]}"


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _find_player_id(player_name: str) -> int:
    playergamelog, players = _require_nba_api()
    normalized = _normalize_name(player_name)
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"Player not found via nba_api: {player_name}")

    exact = [p for p in matches if _normalize_name(p.get("full_name", "")) == normalized]
    if exact:
        active = [p for p in exact if p.get("is_active")]
        return int(active[0]["id"] if active else exact[0]["id"])

    active = [p for p in matches if p.get("is_active")]
    if len(active) == 1:
        return int(active[0]["id"])
    if len(matches) == 1:
        return int(matches[0]["id"])
    names = ", ".join(p.get("full_name", "") for p in matches)
    raise ValueError(f"Multiple players matched '{player_name}': {names}")


def fetch_player_game_logs_by_name(
    player_name: str,
    season: int,
    season_type: SeasonType = "Regular Season",
) -> pd.DataFrame:
    """Fetch player game logs using nba_api."""
    playergamelog, _ = _require_nba_api()
    player_id = _find_player_id(player_name)
    season_str = _format_season(season)
    logs = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season_str,
        season_type_all_star=season_type,
    ).get_data_frames()[0]

    if logs.empty:
        raise ValueError("No game logs returned for player.")

    matchup = logs.get("MATCHUP")
    opponent = None
    location = None
    if matchup is not None:
        matchup = matchup.astype(str)
        opponent = matchup.str.split(" vs ").str[-1]
        location = matchup.apply(lambda text: "Home" if " vs " in text else "Away" if " @ " in text else None)
        opponent = opponent.where(matchup.str.contains(" vs "), matchup.str.split(" @ ").str[-1])

    logs = logs.assign(
        player_name=player_name,
        player_id=str(player_id),
        date=pd.to_datetime(logs.get("GAME_DATE")),
        team=logs.get("TEAM_ABBREVIATION"),
        opponent=opponent,
        location=location,
        minutes=pd.to_numeric(logs.get("MIN"), errors="coerce"),
        points=pd.to_numeric(logs.get("PTS"), errors="coerce"),
        rebounds=pd.to_numeric(logs.get("REB"), errors="coerce"),
        assists=pd.to_numeric(logs.get("AST"), errors="coerce"),
    )
    logs = logs.dropna(subset=["minutes", "points", "rebounds", "assists"])
    return logs[
        [
            "player_name",
            "player_id",
            "date",
            "team",
            "opponent",
            "location",
            "minutes",
            "points",
            "rebounds",
            "assists",
        ]
    ]

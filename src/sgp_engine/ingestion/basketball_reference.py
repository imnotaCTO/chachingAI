from __future__ import annotations

from typing import Literal

import pandas as pd

SeasonType = Literal["Regular Season", "Playoffs"]


class BasketballReferenceImportError(ImportError):
    """Raised when basketball_reference_scraper is unavailable."""


def _require_scraper():
    try:
        from basketball_reference_scraper import box_scores, player_game_logs, team_game_logs
    except ImportError as exc:  # pragma: no cover - exercised by runtime import
        raise BasketballReferenceImportError(
            "basketball_reference_scraper is required. Install with `pip install basketball-reference-scraper`."
        ) from exc
    return box_scores, player_game_logs, team_game_logs


def fetch_player_game_logs(season: int, season_type: SeasonType = "Regular Season") -> pd.DataFrame:
    """Fetch player game logs from Basketball Reference via the scraper."""
    _, player_game_logs, _ = _require_scraper()
    return player_game_logs.get_game_logs(season=season, season_type=season_type)


def fetch_team_game_logs(season: int, season_type: SeasonType = "Regular Season") -> pd.DataFrame:
    """Fetch team game logs from Basketball Reference via the scraper."""
    _, _, team_game_logs = _require_scraper()
    return team_game_logs.get_game_logs(season=season, season_type=season_type)


def fetch_box_scores(date: str) -> pd.DataFrame:
    """Fetch box scores for a given date (YYYY-MM-DD)."""
    box_scores, _, _ = _require_scraper()
    return box_scores.get_box_scores(date)

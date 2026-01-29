"""Data ingestion helpers for Basketball Reference and The Odds API."""

from .basketball_reference import (
    fetch_box_scores,
    fetch_player_game_logs,
    fetch_team_game_logs,
)
from .balldontlie import BallDontLieClient, fetch_player_game_logs_by_name
from .kaggle_nba import fetch_player_game_logs_by_name as fetch_player_game_logs_by_name_kaggle
from .nba_stats import fetch_player_game_logs_by_name as fetch_player_game_logs_by_name_nba_api
from .odds_api import OddsAPIClient, extract_player_props

__all__ = [
    "fetch_player_game_logs",
    "BallDontLieClient",
    "fetch_player_game_logs_by_name",
    "fetch_player_game_logs_by_name_kaggle",
    "fetch_player_game_logs_by_name_nba_api",
    "fetch_team_game_logs",
    "fetch_box_scores",
    "OddsAPIClient",
    "extract_player_props",
]

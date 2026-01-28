"""Core utilities for the Intelligent NBA Same Game Parlay engine."""

from .ingestion import (
    OddsAPIClient,
    extract_player_props,
    fetch_box_scores,
    fetch_player_game_logs,
    fetch_team_game_logs,
)
from .modeling import fit_lognormal_params, lognormal_correlations, lognormal_mean
from .odds import american_to_prob, expected_value, prob_to_american
from .simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula

__all__ = [
    "fit_lognormal_params",
    "lognormal_mean",
    "lognormal_correlations",
    "simulate_lognormal_copula",
    "evaluate_legs",
    "ParlayLeg",
    "american_to_prob",
    "prob_to_american",
    "expected_value",
    "fetch_player_game_logs",
    "fetch_team_game_logs",
    "fetch_box_scores",
    "OddsAPIClient",
    "extract_player_props",
]

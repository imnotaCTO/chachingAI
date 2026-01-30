"""Core utilities for the Intelligent NBA Same Game Parlay engine."""

from .feature_engineering import (
    add_is_home_flag,
    add_minutes_ema,
    add_rolling_stats,
    apply_stability_filters,
    build_player_feature_dataset,
    merge_team_context,
)
from .ingestion import (
    OddsAPIClient,
    BallDontLieClient,
    extract_player_props,
    fetch_player_game_logs_by_name,
    fetch_player_game_logs_by_name_kaggle,
)
from .modeling import (
    fit_lognormal_params,
    lognormal_correlations,
    lognormal_mean,
    make_psd_correlation,
)
from .matching import (
    build_player_name_index,
    match_props_to_player_ids,
    normalize_player_name,
    props_to_leg_specs,
)
from .odds import american_to_prob, expected_value, prob_to_american
from .pace import TeamPaceLookup, normalize_team_name
from .pipeline import LegResult, LegSpec, ParlayRecommendation, price_parlay_from_samples
from .simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula

__all__ = [
    "fit_lognormal_params",
    "lognormal_mean",
    "lognormal_correlations",
    "make_psd_correlation",
    "simulate_lognormal_copula",
    "evaluate_legs",
    "ParlayLeg",
    "american_to_prob",
    "prob_to_american",
    "expected_value",
    "TeamPaceLookup",
    "normalize_team_name",
    "normalize_player_name",
    "build_player_name_index",
    "match_props_to_player_ids",
    "props_to_leg_specs",
    "LegSpec",
    "LegResult",
    "ParlayRecommendation",
    "price_parlay_from_samples",
    "fetch_player_game_logs",
    "fetch_player_game_logs_by_name",
    "fetch_player_game_logs_by_name_kaggle",
    "OddsAPIClient",
    "BallDontLieClient",
    "extract_player_props",
    "add_is_home_flag",
    "add_minutes_ema",
    "add_rolling_stats",
    "apply_stability_filters",
    "build_player_feature_dataset",
    "merge_team_context",
]

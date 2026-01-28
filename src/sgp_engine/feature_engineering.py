from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_STATS = ("points", "rebounds", "assists")


def add_is_home_flag(
    data: pd.DataFrame,
    location_col: str = "location",
    output_col: str = "is_home",
) -> pd.DataFrame:
    """Add a home/away indicator if the location column exists."""
    if location_col not in data.columns:
        return data
    result = data.copy()
    home_values = {"H", "Home", "HOME", "home"}
    result[output_col] = result[location_col].isin(home_values).astype(int)
    return result


def add_rolling_stats(
    data: pd.DataFrame,
    window: int,
    stats: Sequence[str] = DEFAULT_STATS,
    player_id_col: str = "player_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add rolling mean/std features for each stat using prior games."""
    if player_id_col not in data.columns:
        raise ValueError(f"Missing required column: {player_id_col}")
    if date_col not in data.columns:
        raise ValueError(f"Missing required column: {date_col}")
    for stat in stats:
        if stat not in data.columns:
            raise ValueError(f"Missing required stat column: {stat}")

    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([player_id_col, date_col])
    grouped = result.groupby(player_id_col, group_keys=False)

    for stat in stats:
        shifted = grouped[stat].shift(1)
        result[f"{stat}_mean_{window}"] = shifted.rolling(window).mean().reset_index(level=0, drop=True)
        result[f"{stat}_std_{window}"] = shifted.rolling(window).std(ddof=1).reset_index(level=0, drop=True)

    return result


def add_minutes_ema(
    data: pd.DataFrame,
    span: int,
    minutes_col: str = "minutes",
    player_id_col: str = "player_id",
    date_col: str = "date",
    output_col: str = "minutes_ema",
) -> pd.DataFrame:
    """Add an exponential moving average of minutes using prior games."""
    if minutes_col not in data.columns:
        raise ValueError(f"Missing required column: {minutes_col}")
    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([player_id_col, date_col])
    grouped = result.groupby(player_id_col, group_keys=False)
    result[output_col] = (
        grouped[minutes_col]
        .shift(1)
        .ewm(span=span, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return result


def apply_stability_filters(
    data: pd.DataFrame,
    min_games: int,
    min_minutes: float,
    minutes_col: str = "minutes",
    player_id_col: str = "player_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter out rows that do not meet games-played or minutes thresholds."""
    if minutes_col not in data.columns:
        raise ValueError(f"Missing required column: {minutes_col}")
    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([player_id_col, date_col])
    grouped = result.groupby(player_id_col, group_keys=False)
    result["prior_games"] = grouped.cumcount()
    filtered = result[(result["prior_games"] >= min_games) & (result[minutes_col] >= min_minutes)]
    return filtered.drop(columns=["prior_games"])


def merge_team_context(
    data: pd.DataFrame,
    team_logs: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "date",
    pace_col: str = "pace",
) -> pd.DataFrame:
    """Merge team-level pace proxies onto player data when available."""
    if pace_col not in team_logs.columns:
        raise ValueError(f"Missing required pace column: {pace_col}")
    if team_col not in team_logs.columns:
        raise ValueError(f"Missing required team column: {team_col}")
    if date_col not in team_logs.columns:
        raise ValueError(f"Missing required date column: {date_col}")
    team_logs = team_logs[[team_col, date_col, pace_col]].copy()
    team_logs[date_col] = pd.to_datetime(team_logs[date_col])
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    return data.merge(team_logs, on=[team_col, date_col], how="left")


def build_player_feature_dataset(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame | None = None,
    window: int = 10,
    ema_span: int = 5,
    min_games: int = 10,
    min_minutes: float = 10.0,
    stats: Sequence[str] = DEFAULT_STATS,
    player_id_col: str = "player_id",
    date_col: str = "date",
    minutes_col: str = "minutes",
    location_col: str = "location",
    team_col: str = "team",
    pace_col: str = "pace",
) -> pd.DataFrame:
    """Build a feature dataset from historical player logs."""
    features = add_rolling_stats(
        player_logs,
        window=window,
        stats=stats,
        player_id_col=player_id_col,
        date_col=date_col,
    )
    features = add_minutes_ema(
        features,
        span=ema_span,
        minutes_col=minutes_col,
        player_id_col=player_id_col,
        date_col=date_col,
    )
    features = add_is_home_flag(features, location_col=location_col)
    features = apply_stability_filters(
        features,
        min_games=min_games,
        min_minutes=min_minutes,
        minutes_col=minutes_col,
        player_id_col=player_id_col,
        date_col=date_col,
    )
    if team_logs is not None:
        features = merge_team_context(
            features,
            team_logs=team_logs,
            team_col=team_col,
            date_col=date_col,
            pace_col=pace_col,
        )
    return features

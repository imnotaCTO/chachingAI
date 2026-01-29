from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..matching import normalize_player_name


def _season_window(season_end_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=season_end_year - 1, month=10, day=1)
    end = pd.Timestamp(year=season_end_year, month=6, day=30)
    return start, end


@lru_cache(maxsize=2)
def _load_kaggle_player_stats(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle PlayerStatistics.csv not found at {csv_path}")

    usecols = [
        "firstName",
        "lastName",
        "personId",
        "gameDateTimeEst",
        "playerteamCity",
        "playerteamName",
        "opponentteamCity",
        "opponentteamName",
        "home",
        "numMinutes",
        "points",
        "assists",
        "reboundsTotal",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df["player_name"] = (
        df["firstName"].fillna("").astype(str).str.strip()
        + " "
        + df["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["_normalized_name"] = df["player_name"].map(normalize_player_name)
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
    df["location"] = df["home"].apply(lambda value: "Home" if value == 1 else "Away")
    df = df.rename(
        columns={
            "personId": "player_id",
            "gameDateTimeEst": "date",
            "numMinutes": "minutes",
            "reboundsTotal": "rebounds",
        }
    )
    return df


def fetch_player_game_logs_by_name(
    player_name: str,
    season_end_year: int,
    data_path: str = "PlayerStatistics.csv",
) -> pd.DataFrame:
    """Fetch player game logs from the Kaggle-style dataset."""
    df = _load_kaggle_player_stats(data_path)
    normalized = normalize_player_name(player_name)
    start, end = _season_window(season_end_year)
    subset = df[
        (df["_normalized_name"] == normalized)
        & (df["date"] >= start)
        & (df["date"] <= end)
    ].copy()
    if subset.empty:
        raise ValueError(f"No Kaggle logs found for {player_name} in season {season_end_year}.")
    return subset[
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


def fetch_player_game_logs_from_cache(
    player_name: str,
    data_path: str,
    *,
    season_end_year: int | None = None,
) -> pd.DataFrame:
    """Fetch player game logs from a normalized local cache file."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Local cache not found at {csv_path}")
    df = pd.read_parquet(csv_path) if csv_path.suffix.lower() == ".parquet" else pd.read_csv(csv_path)
    normalized = normalize_player_name(player_name)
    df["_normalized_name"] = df["player_name"].map(normalize_player_name)
    subset = df[df["_normalized_name"] == normalized].copy()
    if season_end_year is not None and "date" in subset.columns:
        start, end = _season_window(season_end_year)
        subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
        subset = subset[(subset["date"] >= start) & (subset["date"] <= end)]
    if subset.empty:
        raise ValueError(f"No cache logs found for {player_name}.")
    return subset[
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

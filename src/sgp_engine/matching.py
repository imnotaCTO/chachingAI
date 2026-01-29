from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from .pipeline import LegSpec

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_NON_NAME = re.compile(r"[^a-z0-9\\s]")
_MULTI_SPACE = re.compile(r"\\s+")

DEFAULT_STAT_ORDER = ("points", "rebounds", "assists")
DEFAULT_MARKET_MAP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
}


def normalize_player_name(name: str) -> str:
    """Normalize player names for matching across data sources."""
    lowered = name.strip().lower()
    lowered = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    cleaned = _NON_NAME.sub(" ", lowered)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    tokens = [token for token in cleaned.split(" ") if token and token not in _SUFFIXES]
    return " ".join(tokens)


def build_player_name_index(
    player_logs: pd.DataFrame,
    *,
    name_cols: Sequence[str] = ("player_name", "name", "player"),
    player_id_col: str = "player_id",
) -> dict[str, str]:
    """Build a normalized name index to player_id (or name if id is missing)."""
    name_col = next((col for col in name_cols if col in player_logs.columns), None)
    if name_col is None:
        raise ValueError("No player name column found in player_logs")

    if player_id_col in player_logs.columns:
        ids = player_logs[[name_col, player_id_col]].dropna()
        grouped = ids.drop_duplicates(subset=[name_col, player_id_col])
        return {
            normalize_player_name(row[name_col]): str(row[player_id_col])
            for _, row in grouped.iterrows()
        }

    unique_names = player_logs[name_col].dropna().unique()
    return {normalize_player_name(name): str(name) for name in unique_names}


def match_props_to_player_ids(
    props: Iterable[Mapping[str, object]],
    player_logs: pd.DataFrame,
    *,
    player_name_key: str = "player_name",
    name_cols: Sequence[str] = ("player_name", "name", "player"),
    player_id_col: str = "player_id",
    alias_map: Mapping[str, str] | None = None,
    allow_last_name_match: bool = False,
) -> list[dict]:
    """Attach player_id to each prop row using normalized name matching."""
    index = build_player_name_index(
        player_logs,
        name_cols=name_cols,
        player_id_col=player_id_col,
    )

    normalized_to_id = index
    alias_normalized = {
        normalize_player_name(key): value for key, value in (alias_map or {}).items()
    }

    results: list[dict] = []
    for prop in props:
        player_name = prop.get(player_name_key)
        if not player_name:
            results.append({**prop, "player_id": None, "match_status": "missing"})
            continue

        raw = str(player_name)
        normalized = normalize_player_name(raw)
        if normalized in alias_normalized:
            results.append(
                {
                    **prop,
                    "player_id": alias_normalized[normalized],
                    "match_status": "alias",
                }
            )
            continue

        if normalized in normalized_to_id:
            results.append(
                {
                    **prop,
                    "player_id": normalized_to_id[normalized],
                    "match_status": "exact",
                }
            )
            continue

        if allow_last_name_match and normalized:
            tokens = normalized.split(" ")
            if len(tokens) >= 2:
                first_initial = tokens[0][0]
                last_name = tokens[-1]
                candidates = [
                    name
                    for name in normalized_to_id
                    if name.endswith(last_name) and name[0] == first_initial
                ]
                if len(candidates) == 1:
                    match = candidates[0]
                    results.append(
                        {
                            **prop,
                            "player_id": normalized_to_id[match],
                            "match_status": "fuzzy",
                        }
                    )
                    continue

        results.append({**prop, "player_id": None, "match_status": "unmatched"})

    return results


def props_to_leg_specs(
    props: Iterable[Mapping[str, object]],
    *,
    stat_order: Sequence[str] = DEFAULT_STAT_ORDER,
    market_map: Mapping[str, str] = DEFAULT_MARKET_MAP,
    market_key: str = "market_type",
    line_key: str = "line",
    direction_key: str = "direction",
    player_key: str = "player_name",
) -> list[LegSpec]:
    """Convert prop rows to LegSpec entries based on stat order mapping."""
    order_map = {stat: idx for idx, stat in enumerate(stat_order)}
    legs: list[LegSpec] = []

    for prop in props:
        market = prop.get(market_key)
        if not market:
            continue
        stat = market_map.get(str(market))
        if stat is None or stat not in order_map:
            continue
        line = prop.get(line_key)
        if line is None:
            continue
        direction = prop.get(direction_key) or "over"
        legs.append(
            LegSpec(
                index=order_map[stat],
                line=float(line),
                direction=str(direction),
                player=str(prop.get(player_key)) if prop.get(player_key) else None,
                stat=stat,
            )
        )

    return legs

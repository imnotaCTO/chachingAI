from __future__ import annotations

from dataclasses import dataclass

import re
import unicodedata

import pandas as pd
import requests

BASE_URL = "https://api.balldontlie.io/v1"

_NON_NAME = re.compile(r"[^a-z0-9\\s]")
_MULTI_SPACE = re.compile(r"\\s+")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_player_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    cleaned = _NON_NAME.sub(" ", lowered)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    tokens = [token for token in cleaned.split(" ") if token and token not in _SUFFIXES]
    return " ".join(tokens)


def _parse_minutes(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "DNP", "DND"}:
        return None
    if ":" in text:
        mins, secs = text.split(":", 1)
        try:
            return float(mins) + float(secs) / 60.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class BallDontLieClient:
    api_key: str
    base_url: str = BASE_URL
    auth_scheme: str | None = None

    def _headers(self) -> dict[str, str]:
        if self.auth_scheme:
            return {"Authorization": f"{self.auth_scheme} {self.api_key}"}
        return {"Authorization": self.api_key}

    def _get(self, path: str, params: dict[str, object] | None = None) -> dict:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code == 401:
            raise ValueError(
                "BallDontLie API unauthorized. Check BALLDONTLIE_API_KEY and your plan access. "
                "Free tier plans do not include Game Player Stats."
            )
        response.raise_for_status()
        return response.json()

    def get_players(self, search: str | None = None, per_page: int = 100, cursor: int | None = None) -> dict:
        params: dict[str, object] = {"per_page": per_page}
        if search:
            params["search"] = search
        if cursor is not None:
            params["cursor"] = cursor
        return self._get("/players", params=params)

    def get_stats(self, params: dict[str, object]) -> dict:
        return self._get("/stats", params=params)


def _match_candidates(candidates: list[dict], normalized: str) -> int | None:
    for player in candidates:
        full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        if _normalize_player_name(full_name) == normalized:
            return int(player["id"])
    return None


def find_player_id(client: BallDontLieClient, player_name: str, allow_last_name_match: bool = True) -> int:
    normalized = _normalize_player_name(player_name)
    if not normalized:
        raise ValueError("player_name must be non-empty")

    payload = client.get_players(search=player_name, per_page=100)
    matches = payload.get("data", [])
    match_id = _match_candidates(matches, normalized)
    if match_id is not None:
        return match_id

    last_name = normalized.split(" ")[-1]
    payload = client.get_players(search=last_name, per_page=100)
    matches = payload.get("data", [])
    match_id = _match_candidates(matches, normalized)
    if match_id is not None:
        return match_id

    if allow_last_name_match:
        first_initial = normalized[0]
        candidates = []
        for player in matches:
            full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            candidate = _normalize_player_name(full_name)
            if candidate.endswith(f" {last_name}") and candidate.startswith(first_initial):
                candidates.append(player)
        if len(candidates) == 1:
            return int(candidates[0]["id"])
    raise ValueError(f"Player not found via BallDontLie search: {player_name}")


def fetch_player_game_logs_by_name(
    player_name: str,
    season: int,
    api_key: str,
    postseason: bool = False,
    player_id: int | None = None,
    allow_last_name_match: bool = True,
    base_url: str = BASE_URL,
    auth_scheme: str | None = None,
) -> pd.DataFrame:
    """Fetch game logs for a single player using BallDontLie stats endpoint."""
    client = BallDontLieClient(api_key=api_key, base_url=base_url, auth_scheme=auth_scheme)
    if player_id is None:
        player_id = find_player_id(client, player_name, allow_last_name_match=allow_last_name_match)

    stats_rows: list[dict] = []
    cursor: int | None = None
    while True:
        params: dict[str, object] = {
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": 100,
        }
        if postseason:
            params["postseason"] = "true"
        if cursor is not None:
            params["cursor"] = cursor
        payload = client.get_stats(params)
        stats_rows.extend(payload.get("data", []))
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = int(next_cursor)

    if not stats_rows:
        raise ValueError("No stats returned. Check season and API access tier.")

    records = []
    for row in stats_rows:
        player = row.get("player", {})
        game = row.get("game", {})
        team = row.get("team", {})
        team_id = team.get("id")
        team_abbr = team.get("abbreviation")
        opponent_abbr = None
        location = None
        if team_id and game:
            home_team = game.get("home_team", {})
            visitor_team = game.get("visitor_team", {})
            if team_id == home_team.get("id"):
                opponent_abbr = visitor_team.get("abbreviation")
                location = "Home"
            elif team_id == visitor_team.get("id"):
                opponent_abbr = home_team.get("abbreviation")
                location = "Away"

        records.append(
            {
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "player_id": str(player_id),
                "date": pd.to_datetime(game.get("date")) if game.get("date") else pd.NaT,
                "team": team_abbr or team.get("full_name"),
                "opponent": opponent_abbr,
                "location": location,
                "minutes": _parse_minutes(row.get("min")),
                "points": row.get("pts"),
                "rebounds": row.get("reb"),
                "assists": row.get("ast"),
            }
        )

    logs = pd.DataFrame.from_records(records)
    logs = logs.dropna(subset=["minutes", "points", "rebounds", "assists"])
    return logs

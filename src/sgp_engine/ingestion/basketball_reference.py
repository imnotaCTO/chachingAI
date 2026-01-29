from __future__ import annotations

from typing import Literal

from io import StringIO

import html
import re
import unicodedata

import pandas as pd
import requests

SeasonType = Literal["Regular Season", "Playoffs"]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

class BasketballReferenceImportError(ImportError):
    """Raised when basketball_reference_scraper is unavailable."""


def _require_scraper():
    try:
        from basketball_reference_scraper import box_scores, player_game_logs, team_game_logs
    except ImportError as exc:  # pragma: no cover - exercised by runtime import
        raise BasketballReferenceImportError(
            "basketball_reference_scraper is required and must expose player_game_logs. "
            "If unavailable, use fetch_player_game_logs_by_name instead."
        ) from exc
    return box_scores, player_game_logs, team_game_logs


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


def _find_player_id_from_link(page: str, player_name: str) -> str | None:
    pattern = re.compile(
        r'href="/players/[a-z]/([^"]+)\\.html"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    normalized_target = _normalize_player_name(player_name)
    for player_id, name in pattern.findall(page):
        if _normalize_player_name(html.unescape(name)) == normalized_target:
            return player_id
    return None


def _find_player_id(player_name: str) -> str:
    normalized = _normalize_player_name(player_name)
    if not normalized:
        raise ValueError("player_name must be non-empty")
    last_name_initial = normalized.split(" ")[-1][0]
    url = f"https://www.basketball-reference.com/players/{last_name_initial}/"
    response = requests.get(url, timeout=30, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    matches = pd.read_html(response.text, attrs={"id": "players"}) if "players" in response.text else []
    if matches:
        table = matches[0].copy()
        if "Player" in table.columns and "player_id" not in table.columns:
            table["player_id"] = table["Player"].astype(str).apply(
                lambda name: _find_player_id_from_link(response.text, name)
            )
        for _, row in table.iterrows():
            name = row.get("Player") or row.get("Player Name") or ""
            player_id = row.get("player_id")
            if not player_id:
                continue
            candidate = _normalize_player_name(html.unescape(str(name)))
            if candidate == normalized:
                return str(player_id)
        last = normalized.split(" ")[-1]
        first_initial = normalized[0]
        candidates = [
            str(row.get("player_id"))
            for _, row in table.iterrows()
            if _normalize_player_name(str(row.get("Player") or "")).endswith(f" {last}")
            and _normalize_player_name(str(row.get("Player") or "")).startswith(first_initial)
        ]
        if len(candidates) == 1:
            return candidates[0]

    pattern = re.compile(
        r'data-append-csv="([^"]+)"[^>]*>\\s*<a[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for player_id, name in pattern.findall(response.text):
        candidate = _normalize_player_name(html.unescape(name))
        if candidate == normalized:
            return player_id

    if pattern.findall(response.text):
        last = normalized.split(" ")[-1]
        first_initial = normalized[0]
        candidates = [
            (player_id, name)
            for player_id, name in pattern.findall(response.text)
            if _normalize_player_name(html.unescape(name)).endswith(f" {last}")
            and _normalize_player_name(html.unescape(name)).startswith(first_initial)
        ]
        if len(candidates) == 1:
            return candidates[0][0]
    raise ValueError(f"Player not found on Basketball Reference index: {player_name}")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _extract_commented_table(page: str, table_id: str) -> str | None:
    pattern = re.compile(r"<!--(.*?)-->", re.DOTALL)
    for comment in pattern.findall(page):
        if f'id="{table_id}"' in comment or f"id='{table_id}'" in comment:
            return comment
    return None


def _parse_game_log_html(
    html_text: str,
    player_name: str,
    player_id: str,
) -> pd.DataFrame:
    tables = []
    try:
        tables = pd.read_html(StringIO(html_text), attrs={"id": "pgl_basic"})
    except ValueError:
        tables = []

    if not tables:
        commented = _extract_commented_table(html_text, "pgl_basic")
        if commented:
            try:
                tables = pd.read_html(StringIO(commented))
            except ValueError:
                tables = []

    if not tables:
        table_match = re.search(
            r'<table[^>]*id=["\\\']pgl_basic["\\\'][^>]*>.*?</table>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
        if table_match:
            tables = pd.read_html(StringIO(table_match.group(0)))

    if not tables and commented:
        table_match = re.search(
            r'<table[^>]*id=["\\\']pgl_basic["\\\'][^>]*>.*?</table>',
            commented,
            re.IGNORECASE | re.DOTALL,
        )
        if table_match:
            tables = pd.read_html(StringIO(table_match.group(0)))

    if not tables:
        raise ValueError("Could not locate game log table.")
    logs = tables[0].copy()
    if "Rk" in logs.columns:
        logs = logs[logs["Rk"] != "Rk"]

    date_col = _find_column(list(logs.columns), ["date"])
    team_col = _find_column(list(logs.columns), ["tm", "team"])
    opp_col = _find_column(list(logs.columns), ["opp", "opponent"])
    location_col = _find_column(list(logs.columns), ["home/away", "unnamed: 5"])
    minutes_col = _find_column(list(logs.columns), ["mp", "minutes"])
    points_col = _find_column(list(logs.columns), ["pts", "points"])
    rebounds_col = _find_column(list(logs.columns), ["trb", "rebounds"])
    assists_col = _find_column(list(logs.columns), ["ast", "assists"])

    if minutes_col is None or points_col is None or rebounds_col is None or assists_col is None:
        raise ValueError("Missing core stat columns in Basketball Reference logs.")

    logs = logs.dropna(subset=[minutes_col, points_col, rebounds_col, assists_col])
    logs = logs.assign(
        player_name=player_name,
        player_id=player_id,
        date=pd.to_datetime(logs[date_col]) if date_col else pd.NaT,
        team=logs[team_col] if team_col else None,
        opponent=logs[opp_col] if opp_col else None,
        location=logs[location_col] if location_col else None,
        minutes=pd.to_numeric(logs[minutes_col], errors="coerce"),
        points=pd.to_numeric(logs[points_col], errors="coerce"),
        rebounds=pd.to_numeric(logs[rebounds_col], errors="coerce"),
        assists=pd.to_numeric(logs[assists_col], errors="coerce"),
    )
    if location_col:
        logs["location"] = logs["location"].apply(lambda x: "Away" if str(x).strip() == "@" else "Home")
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


def fetch_player_game_logs_by_name(
    player_name: str,
    season: int,
    season_type: SeasonType = "Regular Season",
    player_id: str | None = None,
    html_text: str | None = None,
) -> pd.DataFrame:
    """Fetch game logs for a single player using Basketball Reference."""
    player_id = player_id or _find_player_id(player_name)
    if html_text is not None:
        return _parse_game_log_html(html_text, player_name, player_id)

    initial = player_id[0]
    if season_type == "Playoffs":
        base_url = f"https://www.basketball-reference.com/players/{initial}/{player_id}/gamelog-playoffs"
    else:
        base_url = f"https://www.basketball-reference.com/players/{initial}/{player_id}/gamelog/{season}"

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    response = session.get(base_url, timeout=30)
    response.raise_for_status()
    html_text = response.text
    try:
        return _parse_game_log_html(html_text, player_name, player_id)
    except ValueError:
        alt_url = f"{base_url}/"
        if alt_url != base_url:
            response = session.get(alt_url, timeout=30)
            response.raise_for_status()
            return _parse_game_log_html(response.text, player_name, player_id)
        raise


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

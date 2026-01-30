from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from sgp_engine.ingestion import (
    OddsAPIClient,
    extract_player_props,
    fetch_player_game_logs_by_name,
    fetch_player_game_logs_by_name_kaggle,
    fetch_player_game_logs_from_cache,
    fetch_multi_player_game_logs_by_name,
    fetch_multi_player_game_logs_from_cache,
)
from sgp_engine.matching import normalize_player_name
from sgp_engine.modeling import fit_lognormal_params, lognormal_correlations, lognormal_mean
from sgp_engine.simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula
from sgp_engine.pace import TeamPaceLookup, normalize_team_name

DEFAULT_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_points_alternate",
    "player_rebounds_alternate",
    "player_assists_alternate",
)
MARKET_TO_STAT = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_points_alternate": "points",
    "player_rebounds_alternate": "rebounds",
    "player_assists_alternate": "assists",
}


def _normalize_book(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_injury_adjustments(path: str) -> dict[tuple[datetime.date, str], float]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    required = {"date", "player_name", "minutes_multiplier"}
    if not required.issubset(df.columns):
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "player_name", "minutes_multiplier"])
    df["player_norm"] = df["player_name"].map(normalize_player_name)
    df["minutes_multiplier"] = pd.to_numeric(df["minutes_multiplier"], errors="coerce")
    df = df.dropna(subset=["minutes_multiplier", "player_norm"])
    adjustments: dict[tuple[datetime.date, str], float] = {}
    for row in df.itertuples(index=False):
        adjustments[(row.date, row.player_norm)] = float(row.minutes_multiplier)
    return adjustments


def load_team_injury_context(path: str) -> dict[tuple[datetime.date, str], dict[str, float | int | None]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "date" not in df.columns or "minutes_multiplier" not in df.columns:
        return {}
    team_cols = [col for col in ("team", "team_name", "team_full", "team_abbr") if col in df.columns]
    if not team_cols:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["minutes_multiplier"] = pd.to_numeric(df["minutes_multiplier"], errors="coerce")
    df = df.dropna(subset=["date", "minutes_multiplier"])

    context: dict[tuple[datetime.date, str], dict[str, float | int | None]] = {}
    for row in df.itertuples(index=False):
        labels: set[str] = set()
        for col in team_cols:
            value = getattr(row, col, None)
            if isinstance(value, str) and value.strip():
                labels.add(value.strip())
        if not labels:
            continue
        multiplier = float(row.minutes_multiplier)
        for label in labels:
            key = (row.date, normalize_team_name(label))
            entry = context.setdefault(
                key,
                {
                    "injury_count": 0,
                    "out_count": 0,
                    "severity_sum": 0.0,
                    "multiplier_sum": 0.0,
                    "avg_multiplier": None,
                },
            )
            if multiplier < 1:
                entry["injury_count"] = int(entry["injury_count"]) + 1
                entry["severity_sum"] = float(entry["severity_sum"]) + max(0.0, 1 - multiplier)
                entry["multiplier_sum"] = float(entry["multiplier_sum"]) + multiplier
                if multiplier <= 0:
                    entry["out_count"] = int(entry["out_count"]) + 1

    for entry in context.values():
        count = int(entry.get("injury_count") or 0)
        if count > 0:
            entry["avg_multiplier"] = float(entry["multiplier_sum"]) / count
        else:
            entry["avg_multiplier"] = None
        entry.pop("multiplier_sum", None)
    return context


def load_injury_context(
    path: str,
) -> tuple[dict[tuple[datetime.date, str], float], dict[tuple[datetime.date, str], dict[str, float | int | None]]]:
    if not os.path.exists(path):
        return {}, {}
    return load_injury_adjustments(path), load_team_injury_context(path)


def _cache_filename(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return safe.strip("_") or "unknown"


def _infer_player_team(
    logs: pd.DataFrame,
    event_date: datetime.date | None,
    home_team: str | None,
    away_team: str | None,
) -> tuple[str | None, str | None]:
    if "team" not in logs.columns:
        return None, None
    team_value = None
    if "date" in logs.columns:
        dated = logs.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce").dt.date
        if event_date is not None:
            dated = dated[dated["date"] <= event_date]
        dated = dated.dropna(subset=["team"])
        if not dated.empty:
            team_value = str(dated["team"].iloc[-1]).strip()
    if team_value is None:
        series = logs["team"].dropna()
        if not series.empty:
            team_value = str(series.iloc[-1]).strip()
    if not team_value:
        return None, None

    if home_team and away_team:
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)
        team_norm = normalize_team_name(team_value)
        if team_norm == home_norm:
            return home_team, away_team
        if team_norm == away_norm:
            return away_team, home_team
    return team_value, None


def _lookup_team_context(
    context: dict[tuple[datetime.date, str], dict[str, float | int | None]],
    event_date: datetime.date | None,
    team: str | None,
) -> dict[str, float | int | None] | None:
    if not context or not event_date or not team:
        return None
    return context.get((event_date, normalize_team_name(team)))


def _load_cached_logs(cache_dir: Path, source: str, season: int, player_name: str, ttl_hours: float) -> pd.DataFrame | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if ttl_hours >= 0 and age_hours > ttl_hours:
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _save_cached_logs(cache_dir: Path, source: str, season: int, player_name: str, logs: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    logs.to_pickle(path)


def lognormal_hit_probability(mu: float, sigma: float, line: float, direction: str) -> float:
    if line <= -1:
        raise ValueError("line must be greater than -1 for log1p")
    z = np.log1p(line)
    cdf = norm.cdf(z, loc=mu, scale=sigma)
    if direction.lower() == "over":
        return float(1 - cdf)
    if direction.lower() == "under":
        return float(cdf)
    raise ValueError(f"Unsupported direction: {direction}")


def _predict_minutes(
    logs: pd.DataFrame,
    *,
    window: int,
    use_ema: bool,
    ema_span: int,
    min_games: int,
) -> float | None:
    if "minutes" not in logs.columns:
        return None
    minutes = logs["minutes"].dropna().astype(float)
    minutes = minutes[minutes > 0]
    if "date" in logs.columns:
        logs = logs.copy()
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        logs = logs.sort_values("date")
        minutes = logs["minutes"].dropna().astype(float)
        minutes = minutes[minutes > 0]
    if window and window > 0:
        minutes = minutes.tail(window)
    if minutes.size < max(2, min_games):
        return None
    if use_ema:
        span = max(2, int(ema_span))
        alpha = 2 / (span + 1)
        n = minutes.size
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return None
        return float((weights * minutes.to_numpy()).sum() / weights_sum)
    return float(minutes.mean())


def event_in_date(event: dict, target_date: datetime.date) -> bool:
    commence = event.get("commence_time")
    if not commence:
        return False
    try:
        dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except ValueError:
        return False
    local_dt = dt.astimezone()
    return local_dt.date() == target_date


def _build_game_key(logs: pd.DataFrame) -> pd.Series:
    if "game_id" in logs.columns:
        return logs["game_id"].astype(str)
    if "date" not in logs.columns or "team" not in logs.columns or "opponent" not in logs.columns:
        return pd.Series([None] * len(logs), index=logs.index)
    dates = pd.to_datetime(logs["date"], errors="coerce").dt.date
    teams = logs["team"].fillna("").astype(str)
    opponents = logs["opponent"].fillna("").astype(str)
    matchup = pd.Series(
        [
            "|".join(sorted([team, opponent])) if team and opponent else None
            for team, opponent in zip(teams, opponents)
        ],
        index=logs.index,
    )
    date_str = dates.astype(str)
    key = date_str + ":" + matchup
    invalid = dates.isna() | matchup.isna()
    return key.where(~invalid)


def _apply_home_advantage(
    samples: np.ndarray,
    locations: pd.Series | None,
    context: str | None,
) -> tuple[np.ndarray, float | None, bool]:
    if locations is None or context not in {"home", "away"} or samples.size == 0:
        return samples, None, False
    loc = locations.astype(str).str.lower()
    if context == "home":
        mask = loc == "home"
    else:
        mask = loc == "away"
    if not mask.any():
        return samples, None, False
    overall_mean = float(np.mean(samples))
    target_mean = float(np.mean(samples[mask.to_numpy()]))
    if overall_mean <= 0 or target_mean <= 0:
        return samples, None, False
    factor = target_mean / overall_mean
    return samples * factor, factor, True


class APIServer(BaseHTTPRequestHandler):
    cache_dir = Path(".cache/api")
    cache_ttl_hours = 12.0
    cache_ttl_seconds = 600
    max_players = 30
    odds_api_key: str | None = None
    bdl_api_key: str | None = None
    kaggle_path: str = "PlayerStatistics.csv"
    cache_path: str | None = None
    team_stats_path: str = "TeamStatistics.csv"
    pace_lookup: TeamPaceLookup | None = None
    injury_adjustments: dict[tuple[datetime.date, str], float] = {}
    injury_team_context: dict[tuple[datetime.date, str], dict[str, float | int | None]] = {}
    player_stat_cache: dict[tuple, dict[str, float | int | None]] = {}
    sportsbook_cache: dict[str, list[str]] = {}
    events_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    props_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    event_lookup: dict[str, dict[str, Any]] = {}

    def _set_headers(self, status: int = HTTPStatus.OK, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            return self._send_json({"status": "ok", "version": "0.2.0"})

        if path == "/api/sportsbooks":
            return self._handle_sportsbooks()

        if path == "/api/events":
            return self._handle_events(query)

        if path.startswith("/api/events/") and path.endswith("/props"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                event_id = parts[2]
                return self._handle_props(event_id, query)

        self._set_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(b'{"error":"not_found"}')

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/parlay/price":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST)
            return self._handle_price(payload)

        self._set_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(b'{"error":"not_found"}')

    def _send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self._set_headers(status=status)
        self.wfile.write(body)

    def _handle_sportsbooks(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        date_str = query.get("date", [None])[0]
        markets = query.get("markets", [",".join(DEFAULT_MARKETS)])[0].split(",")
        cache_key = f"{date_str or 'default'}:{','.join(markets)}"
        cached = self.sportsbook_cache.get(cache_key)
        if cached:
            return self._send_json({"sportsbooks": cached})

        if not self.odds_api_key:
            return self._send_json({"sportsbooks": ["DraftKings"]})

        client = OddsAPIClient(api_key=self.odds_api_key)
        try:
            odds_payload = client.get_odds(markets=markets)
        except Exception:
            return self._send_json({"sportsbooks": ["DraftKings"]})

        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                odds_payload = [event for event in odds_payload if event_in_date(event, target_date)]
            except ValueError:
                pass

        books = set()
        for event in odds_payload:
            for bookmaker in event.get("bookmakers", []):
                title = bookmaker.get("title")
                if title:
                    books.add(title)
        sportsbooks = sorted(books) if books else ["DraftKings"]
        self.sportsbook_cache[cache_key] = sportsbooks
        return self._send_json({"sportsbooks": sportsbooks})

    def _handle_events(self, query: dict[str, list[str]]) -> None:
        if not self.odds_api_key:
            return self._send_json({"error": "missing_odds_api_key"}, status=HTTPStatus.BAD_REQUEST)

        date_str = query.get("date", [None])[0]
        target_date = datetime.now().date() if not date_str else datetime.strptime(date_str, "%Y-%m-%d").date()
        cache_key = target_date.isoformat()
        cached = self.events_cache.get(cache_key)
        if cached and time.time() - cached[0] < self.cache_ttl_seconds:
            return self._send_json({"events": cached[1]})

        client = OddsAPIClient(api_key=self.odds_api_key)
        try:
            events = client.get_events()
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        events = [event for event in events if event_in_date(event, target_date)]
        slim = [
            {
                "event_id": event.get("id"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
            }
            for event in events
            if event.get("id")
        ]
        for event in slim:
            event_id = event.get("event_id")
            if event_id:
                self.event_lookup[str(event_id)] = event
        self.events_cache[cache_key] = (time.time(), slim)
        return self._send_json({"events": slim})

    def _fetch_player_logs(self, source: str, player_name: str, season: int) -> pd.DataFrame:
        cached = _load_cached_logs(self.cache_dir, source, season, player_name, self.cache_ttl_hours)
        if cached is not None:
            return cached

        if source == "kaggle":
            logs = fetch_player_game_logs_by_name_kaggle(
                player_name=player_name, season_end_year=season, data_path=self.kaggle_path
            )
        elif source == "cache":
            if not self.cache_path:
                raise ValueError("missing_cache_path")
            logs = fetch_player_game_logs_from_cache(
                player_name=player_name,
                data_path=self.cache_path,
                season_end_year=season,
            )
        else:
            if not self.bdl_api_key:
                raise ValueError("missing_bdl_api_key")
            logs = fetch_player_game_logs_by_name(
                player_name=player_name, season=season, api_key=self.bdl_api_key
            )
        _save_cached_logs(self.cache_dir, source, season, player_name, logs)
        return logs

    def _fetch_multi_player_logs(self, source: str, player_names: list[str], season: int) -> pd.DataFrame:
        if source == "kaggle":
            return fetch_multi_player_game_logs_by_name(
                player_names=player_names, season_end_year=season, data_path=self.kaggle_path
            )
        if source == "cache":
            if not self.cache_path:
                raise ValueError("missing_cache_path")
            return fetch_multi_player_game_logs_from_cache(
                player_names=player_names,
                data_path=self.cache_path,
                season_end_year=season,
            )
        raise ValueError("multi_player_source_unsupported")

    def _player_stat_params(
        self,
        player_name: str,
        source: str,
        season: int,
        stat: str,
        *,
        window: int,
        min_games: int,
        min_minutes: float,
        use_ema: bool,
        ema_span: int,
        use_minutes_model: bool,
        use_pace_model: bool,
        expected_pace: float | None,
        event_date: datetime.date | None,
        event_home_team: str | None,
        event_away_team: str | None,
        use_home_advantage: bool,
    ) -> dict[str, float | int | None] | None:
        pace_key = round(expected_pace, 3) if expected_pace is not None else None
        home_norm = normalize_team_name(event_home_team) if event_home_team else None
        away_norm = normalize_team_name(event_away_team) if event_away_team else None
        cache_key = (
            player_name,
            source,
            season,
            stat,
            window,
            min_games,
            min_minutes,
            use_ema,
            ema_span,
            use_minutes_model,
            use_pace_model,
            pace_key,
            event_date,
            home_norm,
            away_norm,
            use_home_advantage,
        )
        cached = self.player_stat_cache.get(cache_key)
        if cached:
            return cached

        logs = self._fetch_player_logs(source, player_name, season)
        if stat not in logs.columns:
            return None
        logs = logs.copy()
        if "date" in logs.columns:
            logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
            logs = logs.sort_values("date")
        if use_pace_model and self.pace_lookup is not None:
            logs = self.pace_lookup.attach_pace(logs)
        minutes_pred = None
        if use_minutes_model:
            minutes_pred = _predict_minutes(
                logs,
                window=window,
                use_ema=use_ema,
                ema_span=ema_span,
                min_games=min_games,
            )
        injury_multiplier = None
        if event_date and self.injury_adjustments:
            key = (event_date, normalize_player_name(player_name))
            injury_multiplier = self.injury_adjustments.get(key)
            if minutes_pred is not None and injury_multiplier is not None:
                minutes_pred = minutes_pred * injury_multiplier
        player_team, opponent_team = _infer_player_team(logs, event_date, event_home_team, event_away_team)
        team_ctx = _lookup_team_context(self.injury_team_context, event_date, player_team)
        opp_ctx = _lookup_team_context(self.injury_team_context, event_date, opponent_team)
        if "minutes" in logs.columns and min_minutes > 0:
            logs = logs[logs["minutes"] >= min_minutes]
        season_samples = logs[stat].dropna().astype(float).to_numpy()
        season_avg = float(season_samples.mean()) if season_samples.size > 0 else None
        samples_cols = [stat, "minutes"]
        if "pace" in logs.columns:
            samples_cols.append("pace")
        samples_df = logs[samples_cols].dropna()
        samples_df = samples_df[samples_df["minutes"] > 0]
        if window and window > 0:
            samples_df = samples_df.tail(window)
        samples = samples_df[stat].astype(float).to_numpy()
        minutes_adjusted = False
        pace_adjusted = False
        home_adjusted = False
        home_factor = None
        home_context = None
        if use_minutes_model and minutes_pred is not None and samples.size:
            minutes = samples_df["minutes"].astype(float).to_numpy()
            samples = samples / minutes * minutes_pred
            minutes_adjusted = True
        if use_pace_model and expected_pace is not None and "pace" in samples_df.columns and samples.size:
            pace_series = samples_df["pace"].astype(float).to_numpy()
            pace_series = np.where(pace_series == 0, np.nan, pace_series)
            if np.isfinite(pace_series).any():
                samples = samples / pace_series * expected_pace
                pace_adjusted = True
        if use_home_advantage and samples.size and "location" in logs.columns:
            if player_team and event_home_team and event_away_team:
                player_norm = normalize_team_name(player_team)
                home_norm = normalize_team_name(event_home_team)
                away_norm = normalize_team_name(event_away_team)
                if player_norm == home_norm:
                    home_context = "home"
                elif player_norm == away_norm:
                    home_context = "away"
            location_series = logs.loc[samples_df.index, "location"] if "location" in logs.columns else None
            samples, home_factor, home_adjusted = _apply_home_advantage(samples, location_series, home_context)
        if samples.size < max(2, min_games):
            result = {
                "mu": None,
                "sigma": None,
                "sample_size": int(samples.size),
                "season_avg": season_avg,
                "model_mean": None,
                "predicted_minutes": minutes_pred,
                "minutes_adjusted": minutes_adjusted,
                "predicted_pace": expected_pace,
                "pace_adjusted": pace_adjusted,
                "injury_multiplier": injury_multiplier,
                "player_team": player_team,
                "opponent_team": opponent_team,
                "team_injury_count": team_ctx.get("injury_count") if team_ctx else None,
                "team_injury_out_count": team_ctx.get("out_count") if team_ctx else None,
                "team_injury_severity": team_ctx.get("severity_sum") if team_ctx else None,
                "team_injury_avg_multiplier": team_ctx.get("avg_multiplier") if team_ctx else None,
                "opp_injury_count": opp_ctx.get("injury_count") if opp_ctx else None,
                "opp_injury_out_count": opp_ctx.get("out_count") if opp_ctx else None,
                "opp_injury_severity": opp_ctx.get("severity_sum") if opp_ctx else None,
                "opp_injury_avg_multiplier": opp_ctx.get("avg_multiplier") if opp_ctx else None,
                "home_context": home_context,
                "home_factor": home_factor,
                "home_adjusted": home_adjusted,
            }
            self.player_stat_cache[cache_key] = result
            return result
        if use_ema:
            span = max(2, int(ema_span))
            alpha = 2 / (span + 1)
            n = samples.size
            weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
            weights_sum = weights.sum()
            if weights_sum <= 0:
                return None
            log_samples = np.log1p(samples)
            mu = float((weights * log_samples).sum() / weights_sum)
            var = float((weights * (log_samples - mu) ** 2).sum() / weights_sum)
            sigma = float(np.sqrt(max(var, 1e-12)))
        else:
            mu, sigma = fit_lognormal_params(samples)
        model_mean = float(lognormal_mean(mu, sigma))
        result = {
            "mu": float(mu),
            "sigma": float(sigma),
            "sample_size": int(samples.size),
            "season_avg": season_avg,
            "model_mean": model_mean,
            "predicted_minutes": minutes_pred,
            "minutes_adjusted": minutes_adjusted,
            "predicted_pace": expected_pace,
            "pace_adjusted": pace_adjusted,
            "injury_multiplier": injury_multiplier,
            "player_team": player_team,
            "opponent_team": opponent_team,
            "team_injury_count": team_ctx.get("injury_count") if team_ctx else None,
            "team_injury_out_count": team_ctx.get("out_count") if team_ctx else None,
            "team_injury_severity": team_ctx.get("severity_sum") if team_ctx else None,
            "team_injury_avg_multiplier": team_ctx.get("avg_multiplier") if team_ctx else None,
            "opp_injury_count": opp_ctx.get("injury_count") if opp_ctx else None,
            "opp_injury_out_count": opp_ctx.get("out_count") if opp_ctx else None,
            "opp_injury_severity": opp_ctx.get("severity_sum") if opp_ctx else None,
            "opp_injury_avg_multiplier": opp_ctx.get("avg_multiplier") if opp_ctx else None,
            "home_context": home_context,
            "home_factor": home_factor,
            "home_adjusted": home_adjusted,
        }
        self.player_stat_cache[cache_key] = result
        return result

    def _handle_props(self, event_id: str, query: dict[str, list[str]]) -> None:
        if not self.odds_api_key:
            return self._send_json({"error": "missing_odds_api_key"}, status=HTTPStatus.BAD_REQUEST)

        sportsbook = (query.get("sportsbook", [None])[0] or "").strip()
        stats_source = (query.get("stats_source", ["kaggle"])[0] or "kaggle").strip()
        season = int(query.get("season", [datetime.now().year])[0])
        max_players = int(query.get("max_players", [self.max_players])[0])
        markets = query.get("markets", [",".join(DEFAULT_MARKETS)])[0].split(",")
        window = int(query.get("window", [0])[0] or 0)
        min_games = int(query.get("min_games", [15])[0] or 15)
        min_minutes = float(query.get("min_minutes", [0])[0] or 0)
        use_ema = query.get("use_ema", ["0"])[0] in {"1", "true", "True"}
        ema_span = int(query.get("ema_span", [10])[0] or 10)
        use_minutes_model = query.get("use_minutes_model", ["0"])[0] in {"1", "true", "True"}
        use_pace_model = query.get("use_pace_model", ["0"])[0] in {"1", "true", "True"}
        use_home_advantage = query.get("use_home_advantage", ["1"])[0] in {"1", "true", "True"}
        cache_key = (
            f"{event_id}:{sportsbook}:{stats_source}:{season}:{max_players}:{','.join(markets)}:"
            f"{window}:{min_games}:{min_minutes}:{int(use_ema)}:{ema_span}:{int(use_minutes_model)}:"
            f"{int(use_pace_model)}:{int(use_home_advantage)}"
        )
        cached = self.props_cache.get(cache_key)
        if cached and time.time() - cached[0] < self.cache_ttl_seconds:
            return self._send_json(cached[1])

        client = OddsAPIClient(api_key=self.odds_api_key)
        try:
            odds_payload = client.get_event_odds(event_id, markets=markets)
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        if isinstance(odds_payload, dict):
            odds_payload = [odds_payload]
        expected_pace = None
        event_date = None
        event_home_team = None
        event_away_team = None
        if odds_payload:
            event_info = odds_payload[0]
            commence = event_info.get("commence_time")
            home_team = event_info.get("home_team")
            away_team = event_info.get("away_team")
            event_home_team = str(home_team) if home_team else None
            event_away_team = str(away_team) if away_team else None
            dt = None
            if commence:
                try:
                    dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                    event_date = dt.astimezone().date()
                except ValueError:
                    event_date = None
            if event_id:
                self.event_lookup[str(event_id)] = {
                    "event_id": event_id,
                    "home_team": event_home_team,
                    "away_team": event_away_team,
                    "commence_time": commence,
                }
            if use_pace_model and self.pace_lookup is not None and home_team and away_team and dt:
                try:
                    expected_pace = self.pace_lookup.expected_pace(
                        str(home_team),
                        str(away_team),
                        dt.astimezone(),
                        window,
                    )
                except ValueError:
                    expected_pace = None
        available_markets: set[str] = set()
        bookmakers: set[str] = set()
        for event in odds_payload:
            for bookmaker in event.get("bookmakers", []):
                title = bookmaker.get("title")
                if title:
                    bookmakers.add(title)
                for market in bookmaker.get("markets", []):
                    key = market.get("key")
                    if key:
                        available_markets.add(str(key))
        props = extract_player_props(odds_payload, markets=markets)
        sportsbook_warning = None
        if sportsbook:
            target = _normalize_book(sportsbook)
            filtered = []
            for prop in props:
                book = str(prop.get("sportsbook", "")).strip()
                normalized = _normalize_book(book)
                if not normalized:
                    continue
                if normalized == target or target in normalized or normalized in target:
                    filtered.append(prop)
            if filtered:
                props = filtered
            else:
                sportsbook_warning = "No matching sportsbook found; returning all props."

        players_seen: set[str] = set()
        enriched: list[dict[str, Any]] = []
        last_update = None

        for prop in props:
            player_name = prop.get("player_name")
            if not player_name:
                continue
            if player_name not in players_seen:
                players_seen.add(player_name)
            if len(players_seen) > max_players:
                break

            market = prop.get("market_type")
            stat = MARKET_TO_STAT.get(str(market))
            if not stat:
                continue

            if last_update is None:
                last_update = prop.get("timestamp")

            model_probability = None
            sample_size = None
            season_avg = None
            model_mean = None
            params = None
            try:
                params = self._player_stat_params(
                    player_name,
                    stats_source,
                    season,
                    stat,
                    window=window,
                    min_games=min_games,
                    min_minutes=min_minutes,
                    use_ema=use_ema,
                    ema_span=ema_span,
                    use_minutes_model=use_minutes_model,
                    use_pace_model=use_pace_model,
                    expected_pace=expected_pace,
                    event_date=event_date,
                    event_home_team=event_home_team,
                    event_away_team=event_away_team,
                    use_home_advantage=use_home_advantage,
                )
                if params:
                    mu = params.get("mu")
                    sigma = params.get("sigma")
                    sample_size = params.get("sample_size")
                    season_avg = params.get("season_avg")
                    model_mean = params.get("model_mean")
                    line = float(prop.get("line")) if prop.get("line") is not None else None
                    direction = str(prop.get("direction") or "over")
                    if line is not None and mu is not None and sigma is not None:
                        model_probability = lognormal_hit_probability(mu, sigma, line, direction)
            except Exception:
                model_probability = None
                sample_size = None
                season_avg = None
                model_mean = None

            enriched.append(
                {
                    "event_id": prop.get("event_id"),
                    "sportsbook": prop.get("sportsbook"),
                    "player_name": player_name,
                    "market_type": market,
                    "stat": stat,
                    "line": prop.get("line"),
                    "direction": prop.get("direction") or "over",
                    "odds": prop.get("odds"),
                    "model_probability": model_probability,
                    "sample_size": sample_size,
                    "season_avg": season_avg,
                    "model_mean": model_mean,
                    "predicted_minutes": params.get("predicted_minutes") if params else None,
                    "minutes_adjusted": params.get("minutes_adjusted") if params else None,
                    "predicted_pace": params.get("predicted_pace") if params else None,
                    "pace_adjusted": params.get("pace_adjusted") if params else None,
                    "injury_multiplier": params.get("injury_multiplier") if params else None,
                    "player_team": params.get("player_team") if params else None,
                    "opponent_team": params.get("opponent_team") if params else None,
                    "team_injury_count": params.get("team_injury_count") if params else None,
                    "team_injury_out_count": params.get("team_injury_out_count") if params else None,
                    "team_injury_severity": params.get("team_injury_severity") if params else None,
                    "team_injury_avg_multiplier": params.get("team_injury_avg_multiplier") if params else None,
                    "opp_injury_count": params.get("opp_injury_count") if params else None,
                    "opp_injury_out_count": params.get("opp_injury_out_count") if params else None,
                    "opp_injury_severity": params.get("opp_injury_severity") if params else None,
                    "opp_injury_avg_multiplier": params.get("opp_injury_avg_multiplier") if params else None,
                    "home_context": params.get("home_context") if params else None,
                    "home_factor": params.get("home_factor") if params else None,
                    "home_adjusted": params.get("home_adjusted") if params else None,
                }
            )

        response = {
            "event_id": event_id,
            "sportsbook": sportsbook,
            "stats_source": stats_source,
            "season": season,
            "last_update": last_update,
            "props": enriched,
            "available_markets": sorted(available_markets),
            "bookmakers": sorted(bookmakers),
            "use_home_advantage": use_home_advantage,
        }
        if sportsbook_warning:
            response["warning"] = sportsbook_warning
        self.props_cache[cache_key] = (time.time(), response)
        return self._send_json(response)

    def _handle_price(self, payload: dict[str, Any]) -> None:
        legs = payload.get("legs", [])
        stats_source = payload.get("stats_source", "balldontlie")
        season = int(payload.get("season", datetime.now().year))
        window = int(payload.get("window", 0) or 0)
        min_games = int(payload.get("min_games", 15) or 15)
        min_minutes = float(payload.get("min_minutes", 0) or 0)
        use_ema = payload.get("use_ema", False)
        if isinstance(use_ema, str):
            use_ema = use_ema.lower() in {"1", "true", "yes", "y"}
        use_ema = bool(use_ema)
        ema_span = int(payload.get("ema_span", 10) or 10)
        use_minutes_model = payload.get("use_minutes_model", False)
        if isinstance(use_minutes_model, str):
            use_minutes_model = use_minutes_model.lower() in {"1", "true", "yes", "y"}
        use_minutes_model = bool(use_minutes_model)
        use_home_advantage = payload.get("use_home_advantage", True)
        if isinstance(use_home_advantage, str):
            use_home_advantage = use_home_advantage.lower() in {"1", "true", "yes", "y"}
        use_home_advantage = bool(use_home_advantage)
        if not legs:
            return self._send_json({"error": "legs_required"}, status=HTTPStatus.BAD_REQUEST)

        leg_results = []
        sample_sizes = []
        for leg in legs:
            player_name = leg.get("player_name")
            stat = leg.get("stat")
            line = leg.get("line")
            direction = leg.get("direction") or "over"
            odds = leg.get("odds")

            model_probability = None
            sample_size = None
            if player_name and stat and line is not None:
                try:
                    event_home = None
                    event_away = None
                    event_date = None
                    event_id = leg.get("event_id")
                    if event_id:
                        event_info = self.event_lookup.get(str(event_id))
                        if event_info:
                            event_home = event_info.get("home_team")
                            event_away = event_info.get("away_team")
                            commence = event_info.get("commence_time")
                            if commence:
                                try:
                                    dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                                    event_date = dt.astimezone().date()
                                except ValueError:
                                    event_date = None
                    params = self._player_stat_params(
                        player_name,
                        stats_source,
                        season,
                        stat,
                        window=window,
                        min_games=min_games,
                        min_minutes=min_minutes,
                        use_ema=use_ema,
                        ema_span=ema_span,
                        use_minutes_model=use_minutes_model,
                        use_pace_model=False,
                        expected_pace=None,
                        event_date=event_date,
                        event_home_team=event_home,
                        event_away_team=event_away,
                        use_home_advantage=use_home_advantage,
                    )
                    if params and params.get("mu") is not None and params.get("sigma") is not None:
                        mu = float(params["mu"])
                        sigma = float(params["sigma"])
                        sample_size = int(params["sample_size"]) if params.get("sample_size") is not None else None
                        model_probability = lognormal_hit_probability(mu, sigma, float(line), direction)
                        if sample_size is not None:
                            sample_sizes.append(sample_size)
                except Exception:
                    model_probability = None
            if model_probability is None:
                model_probability = 0.5
            leg_results.append(
                {
                    "player_name": player_name,
                    "stat": stat,
                    "line": line,
                    "direction": direction,
                    "model_probability": model_probability,
                    "odds": odds,
                    "event_id": leg.get("event_id"),
                    "sample_size": sample_size,
                }
            )

        def correlated_group(group_legs: list[dict[str, Any]]) -> tuple[float, list[float], int] | None:
            if stats_source not in {"kaggle", "cache"}:
                return None
            try:
                event_home = None
                event_away = None
                event_date = None
                event_id = group_legs[0].get("event_id") if group_legs else None
                if event_id:
                    event_info = self.event_lookup.get(str(event_id))
                    if event_info:
                        event_home = event_info.get("home_team")
                        event_away = event_info.get("away_team")
                        commence = event_info.get("commence_time")
                        if commence:
                            try:
                                dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                                event_date = dt.astimezone().date()
                            except ValueError:
                                event_date = None
                player_names = [str(leg["player_name"]) for leg in group_legs]
                logs = self._fetch_multi_player_logs(stats_source, player_names, season)
                logs = logs.copy()
                if "_normalized_name" not in logs.columns:
                    logs["_normalized_name"] = logs["player_name"].map(normalize_player_name)
                if "date" in logs.columns:
                    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
                if "minutes" in logs.columns and min_minutes > 0:
                    logs = logs[logs["minutes"] >= min_minutes]
                logs["game_key"] = _build_game_key(logs)
                logs = logs.dropna(subset=["game_key"])
                if logs.empty:
                    return None

                series_list = []
                for leg in group_legs:
                    normalized = normalize_player_name(str(leg["player_name"]))
                    player_logs = logs[logs["_normalized_name"] == normalized]
                    if player_logs.empty or leg["stat"] not in player_logs.columns:
                        return None
                    player_logs = player_logs.copy()
                    player_logs = player_logs[player_logs["minutes"].notna()]
                    player_logs = player_logs[player_logs["minutes"] > 0]
                    player_logs = player_logs.set_index("game_key")
                    series = player_logs[leg["stat"]].dropna().astype(float)
                    if use_minutes_model:
                        minutes_pred = _predict_minutes(
                            player_logs.reset_index(),
                            window=window,
                            use_ema=use_ema,
                            ema_span=ema_span,
                            min_games=min_games,
                        )
                        if minutes_pred is not None and "minutes" in player_logs.columns:
                            minutes_series = player_logs["minutes"].astype(float)
                            series = series / minutes_series.loc[series.index] * minutes_pred
                    if window and window > 0:
                        series = series.tail(window)
                    if use_home_advantage and event_home and event_away and event_date:
                        player_team, _ = _infer_player_team(
                            player_logs.reset_index(),
                            event_date,
                            event_home,
                            event_away,
                        )
                        home_context = None
                        if player_team:
                            player_norm = normalize_team_name(player_team)
                            home_norm = normalize_team_name(str(event_home))
                            away_norm = normalize_team_name(str(event_away))
                            if player_norm == home_norm:
                                home_context = "home"
                            elif player_norm == away_norm:
                                home_context = "away"
                        if home_context and "location" in player_logs.columns:
                            location_series = player_logs.loc[series.index, "location"]
                            adjusted, _, adjusted_flag = _apply_home_advantage(
                                series.to_numpy(),
                                location_series,
                                home_context,
                            )
                            if adjusted_flag:
                                series = pd.Series(adjusted, index=series.index)
                    series_list.append(series)

                common_index = series_list[0].index
                for series in series_list[1:]:
                    common_index = common_index.intersection(series.index)
                if common_index.size < max(2, min_games):
                    return None

                key_dates = (
                    logs.dropna(subset=["game_key", "date"])
                    .drop_duplicates(subset=["game_key"])
                    .set_index("game_key")["date"]
                )
                common_index = key_dates.loc[common_index].sort_values().index
                samples = np.column_stack([series.loc[common_index].to_numpy() for series in series_list])
                if use_ema:
                    span = max(2, int(ema_span))
                    alpha = 2 / (span + 1)
                    n = samples.shape[0]
                    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
                    weights_sum = weights.sum()
                    if weights_sum <= 0:
                        return None
                    log_samples = np.log1p(samples)
                    mu = (weights[:, None] * log_samples).sum(axis=0) / weights_sum
                    var = (weights[:, None] * (log_samples - mu) ** 2).sum(axis=0) / weights_sum
                    sigma = np.sqrt(np.maximum(var, 1e-12))
                else:
                    mu = []
                    sigma = []
                    for col in range(samples.shape[1]):
                        col_mu, col_sigma = fit_lognormal_params(samples[:, col])
                        mu.append(col_mu)
                        sigma.append(col_sigma)
                mu = np.array(mu, dtype=float)
                sigma = np.array(sigma, dtype=float)
                correlation = lognormal_correlations(samples)
                if not np.isfinite(correlation).all():
                    correlation = np.eye(samples.shape[1])
                generated = simulate_lognormal_copula(
                    mu=mu,
                    sigma=sigma,
                    correlation=correlation,
                    simulations=20000,
                )
                parlay_legs = [
                    ParlayLeg(index=i, line=float(group_legs[i]["line"]), direction=group_legs[i]["direction"])
                    for i in range(len(group_legs))
                ]
                result = evaluate_legs(generated, parlay_legs)
                return (
                    float(result.joint_probability),
                    [float(p) for p in result.leg_probabilities],
                    int(samples.shape[0]),
                )
            except Exception:
                return None

        grouped: dict[str, list[int]] = {}
        for idx, leg in enumerate(leg_results):
            group_id = leg.get("event_id") or f"leg-{idx}"
            grouped.setdefault(str(group_id), []).append(idx)

        joint_probability = 1.0
        correlated_groups = 0
        correlated_legs: set[int] = set()
        correlated_sample_sizes: list[int] = []

        for indices in grouped.values():
            if len(indices) < 2:
                for idx in indices:
                    joint_probability *= leg_results[idx]["model_probability"]
                continue

            group_legs = [leg_results[idx] for idx in indices]
            if any(not leg.get("player_name") or not leg.get("stat") or leg.get("line") is None for leg in group_legs):
                for idx in indices:
                    joint_probability *= leg_results[idx]["model_probability"]
                continue

            correlated = correlated_group(group_legs)
            if correlated is None:
                for idx in indices:
                    joint_probability *= leg_results[idx]["model_probability"]
                continue

            group_joint, leg_probs, aligned_size = correlated
            joint_probability *= group_joint
            correlated_groups += 1
            correlated_sample_sizes.append(aligned_size)
            for local_idx, global_idx in enumerate(indices):
                leg_results[global_idx]["model_probability"] = leg_probs[local_idx]
                leg_results[global_idx]["sample_size"] = aligned_size
                correlated_legs.add(global_idx)

        def american_to_prob(odds: float) -> float:
            if odds == 0:
                return 0.0
            if odds > 0:
                return 100 / (odds + 100)
            return -odds / (-odds + 100)

        implied = 1.0
        for leg in leg_results:
            odds = leg.get("odds")
            if odds is None:
                continue
            implied *= american_to_prob(float(odds))

        def prob_to_american(probability: float) -> float:
            if probability >= 0.5:
                return -100 * probability / (1 - probability)
            return 100 * (1 - probability) / probability

        def expected_value(probability: float, odds: float) -> float:
            payout = odds / 100 if odds > 0 else 100 / abs(odds)
            return probability * payout - (1 - probability)

        fair_odds = prob_to_american(min(max(joint_probability, 1e-6), 1 - 1e-6))
        implied_odds = prob_to_american(min(max(implied, 1e-6), 1 - 1e-6))
        ev = expected_value(joint_probability, implied_odds)
        if correlated_groups == 0:
            correlation_mode = "independent"
        elif len(correlated_legs) == len(leg_results):
            correlation_mode = "same_game"
        else:
            correlation_mode = "mixed"

        return self._send_json(
            {
                "legs": leg_results,
                "joint_probability": joint_probability,
                "sportsbook_implied_probability": implied,
                "model_fair_odds": fair_odds,
                "expected_value": ev,
                "diagnostics": {
                    "sample_size": min(sample_sizes) if sample_sizes else None,
                    "correlated_sample_size": min(correlated_sample_sizes) if correlated_sample_sizes else None,
                    "correlation_mode": correlation_mode,
                    "correlated_groups": correlated_groups,
                    "correlated_legs": len(correlated_legs),
                },
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Local API server for the parlay UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--cache-ttl-hours", type=float, default=12.0, help="Cache TTL in hours.")
    parser.add_argument("--max-players", type=int, default=30, help="Max players to enrich per request.")
    parser.add_argument("--kaggle-path", default="PlayerStatistics.csv", help="Path to Kaggle stats CSV.")
    parser.add_argument("--cache-path", default=None, help="Path to normalized BallDontLie cache (parquet or csv).")
    parser.add_argument("--team-stats-path", default="TeamStatistics.csv", help="Path to TeamStatistics.csv.")
    parser.add_argument("--injuries-path", default=None, help="Path to injury adjustments CSV.")
    args = parser.parse_args()

    load_dotenv()
    odds_api_key = os.environ.get("ODDS_API_KEY")
    bdl_api_key = os.environ.get("BALLDONTLIE_API_KEY")

    server = ThreadingHTTPServer((args.host, args.port), APIServer)
    server.RequestHandlerClass.odds_api_key = odds_api_key
    server.RequestHandlerClass.bdl_api_key = bdl_api_key
    server.RequestHandlerClass.cache_ttl_hours = args.cache_ttl_hours
    server.RequestHandlerClass.max_players = args.max_players
    server.RequestHandlerClass.kaggle_path = args.kaggle_path
    server.RequestHandlerClass.cache_path = args.cache_path
    server.RequestHandlerClass.team_stats_path = args.team_stats_path
    try:
        server.RequestHandlerClass.pace_lookup = TeamPaceLookup.from_csv(args.team_stats_path)
    except Exception:
        server.RequestHandlerClass.pace_lookup = None
    if args.injuries_path:
        adjustments, team_context = load_injury_context(args.injuries_path)
        server.RequestHandlerClass.injury_adjustments = adjustments
        server.RequestHandlerClass.injury_team_context = team_context

    print(f"API server running on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
)
from sgp_engine.modeling import fit_lognormal_params

DEFAULT_MARKETS = ("player_points", "player_rebounds", "player_assists")
MARKET_TO_STAT = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
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


def _cache_filename(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return safe.strip("_") or "unknown"


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


class APIServer(BaseHTTPRequestHandler):
    cache_dir = Path(".cache/api")
    cache_ttl_hours = 12.0
    cache_ttl_seconds = 600
    max_players = 30
    odds_api_key: str | None = None
    bdl_api_key: str | None = None
    kaggle_path: str = "PlayerStatistics.csv"
    player_stat_cache: dict[tuple[str, str, int], dict[str, tuple[float, float, int]]] = {}
    sportsbook_cache: dict[str, list[str]] = {}
    events_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    props_cache: dict[str, tuple[float, dict[str, Any]]] = {}

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
        cache_key = "default"
        cached = self.sportsbook_cache.get(cache_key)
        if cached:
            return self._send_json({"sportsbooks": cached})

        if not self.odds_api_key:
            return self._send_json({"sportsbooks": ["DraftKings"]})

        client = OddsAPIClient(api_key=self.odds_api_key)
        try:
            odds_payload = client.get_odds(markets=("player_points",))
        except Exception:
            return self._send_json({"sportsbooks": ["DraftKings"]})

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
        else:
            if not self.bdl_api_key:
                raise ValueError("missing_bdl_api_key")
            logs = fetch_player_game_logs_by_name(
                player_name=player_name, season=season, api_key=self.bdl_api_key
            )
        _save_cached_logs(self.cache_dir, source, season, player_name, logs)
        return logs

    def _player_stat_params(
        self, player_name: str, source: str, season: int, stat: str
    ) -> tuple[float, float, int] | None:
        cache_key = (player_name, source, season)
        cached = self.player_stat_cache.get(cache_key, {})
        if stat in cached:
            return cached[stat]

        logs = self._fetch_player_logs(source, player_name, season)
        if stat not in logs.columns:
            return None
        samples = logs[stat].dropna().astype(float).to_numpy()
        if samples.size < 2:
            return None
        mu, sigma = fit_lognormal_params(samples)
        result = (mu, sigma, int(samples.size))
        cached[stat] = result
        self.player_stat_cache[cache_key] = cached
        return result

    def _handle_props(self, event_id: str, query: dict[str, list[str]]) -> None:
        if not self.odds_api_key:
            return self._send_json({"error": "missing_odds_api_key"}, status=HTTPStatus.BAD_REQUEST)

        sportsbook = (query.get("sportsbook", [None])[0] or "").strip()
        stats_source = (query.get("stats_source", ["balldontlie"])[0] or "balldontlie").strip()
        season = int(query.get("season", [datetime.now().year])[0])
        max_players = int(query.get("max_players", [self.max_players])[0])
        markets = query.get("markets", [",".join(DEFAULT_MARKETS)])[0].split(",")
        cache_key = f"{event_id}:{sportsbook}:{stats_source}:{season}:{max_players}:{','.join(markets)}"
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
            try:
                params = self._player_stat_params(player_name, stats_source, season, stat)
                if params:
                    mu, sigma, sample_size = params
                    line = float(prop.get("line")) if prop.get("line") is not None else None
                    direction = str(prop.get("direction") or "over")
                    if line is not None:
                        model_probability = lognormal_hit_probability(mu, sigma, line, direction)
            except Exception:
                model_probability = None
                sample_size = None

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
                }
            )

        response = {
            "event_id": event_id,
            "sportsbook": sportsbook,
            "stats_source": stats_source,
            "season": season,
            "last_update": last_update,
            "props": enriched,
        }
        if sportsbook_warning:
            response["warning"] = sportsbook_warning
        self.props_cache[cache_key] = (time.time(), response)
        return self._send_json(response)

    def _handle_price(self, payload: dict[str, Any]) -> None:
        legs = payload.get("legs", [])
        stats_source = payload.get("stats_source", "balldontlie")
        season = int(payload.get("season", datetime.now().year))
        if not legs:
            return self._send_json({"error": "legs_required"}, status=HTTPStatus.BAD_REQUEST)

        leg_results = []
        joint_probability = 1.0
        sample_sizes = []
        for leg in legs:
            player_name = leg.get("player_name")
            stat = leg.get("stat")
            line = leg.get("line")
            direction = leg.get("direction") or "over"
            odds = leg.get("odds")

            model_probability = None
            if player_name and stat and line is not None:
                try:
                    params = self._player_stat_params(player_name, stats_source, season, stat)
                    if params:
                        mu, sigma, sample_size = params
                        model_probability = lognormal_hit_probability(mu, sigma, float(line), direction)
                        sample_sizes.append(sample_size)
                except Exception:
                    model_probability = None
            if model_probability is None:
                model_probability = 0.5
            joint_probability *= model_probability
            leg_results.append(
                {
                    "player_name": player_name,
                    "stat": stat,
                    "line": line,
                    "direction": direction,
                    "model_probability": model_probability,
                    "odds": odds,
                }
            )

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

        return self._send_json(
            {
                "legs": leg_results,
                "joint_probability": joint_probability,
                "sportsbook_implied_probability": implied,
                "model_fair_odds": fair_odds,
                "expected_value": ev,
                "diagnostics": {
                    "sample_size": min(sample_sizes) if sample_sizes else None,
                    "correlation_mode": "independent",
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

    print(f"API server running on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

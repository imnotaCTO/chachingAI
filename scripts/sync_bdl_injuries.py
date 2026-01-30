from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_STATUS_MAP = {
    "out": 0.0,
    "doubtful": 0.15,
    "questionable": 0.6,
    "probable": 0.85,
    "day-to-day": 0.7,
    "gtd": 0.6,
}


DEFAULT_STATUS_MAP_PATH = os.path.join("config", "injury_status_map.json")


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


def fetch_all_injuries(api_key: str, base_url: str, per_page: int) -> list[dict[str, Any]]:
    injuries: list[dict[str, Any]] = []
    cursor = None
    headers = {"Authorization": api_key}
    while True:
        params = {"per_page": per_page}
        if cursor is not None:
            params["cursor"] = cursor
        response = requests.get(f"{base_url}/player_injuries", headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        injuries.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return injuries


def fetch_all_teams(api_key: str, base_url: str, per_page: int) -> dict[int, dict[str, str]]:
    teams: list[dict[str, Any]] = []
    cursor = None
    headers = {"Authorization": api_key}
    while True:
        params = {"per_page": per_page}
        if cursor is not None:
            params["cursor"] = cursor
        response = requests.get(f"{base_url}/teams", headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        teams.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    team_map: dict[int, dict[str, str]] = {}
    for team in teams:
        try:
            team_id = int(team.get("id"))
        except (TypeError, ValueError):
            continue
        team_map[team_id] = {
            "team": str(team.get("full_name") or team.get("name") or "").strip(),
            "team_abbr": str(team.get("abbreviation") or "").strip(),
        }
    return team_map


def _load_status_map(value: str | None) -> dict[str, float]:
    if not value:
        return DEFAULT_STATUS_MAP.copy()
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as handle:
            return {**DEFAULT_STATUS_MAP, **json.load(handle)}
    return {**DEFAULT_STATUS_MAP, **json.loads(value)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync BallDontLie player injuries to a local CSV cache.")
    parser.add_argument("--output", default="data/injuries_bdl.csv", help="Output CSV file.")
    parser.add_argument("--base-url", default="https://api.balldontlie.io/v1", help="BallDontLie base URL.")
    parser.add_argument("--per-page", type=int, default=100, help="Results per page.")
    parser.add_argument(
        "--status-map",
        default=DEFAULT_STATUS_MAP_PATH if os.path.exists(DEFAULT_STATUS_MAP_PATH) else None,
        help="Optional JSON file or inline JSON mapping statuses to minutes multipliers.",
    )
    parser.add_argument("--snapshot-date", default=None, help="Override snapshot date (YYYY-MM-DD).")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting.")
    parser.add_argument("--snapshot-dir", default=None, help="Optional directory to write per-day snapshots.")
    parser.add_argument(
        "--team-map",
        default=None,
        help="Optional JSON file or inline JSON mapping team_id to name/abbreviation.",
    )
    parser.add_argument(
        "--skip-team-lookup",
        action="store_true",
        help="Skip fetching team names from BallDontLie.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise SystemExit("Missing BALLDONTLIE_API_KEY. Set it in the environment or .env file.")

    status_map = _load_status_map(args.status_map)

    snapshot_date = args.snapshot_date or datetime.now().strftime("%Y-%m-%d")
    injuries = fetch_all_injuries(api_key, args.base_url, args.per_page)
    team_map: dict[int, dict[str, str]] = {}
    if args.team_map:
        if os.path.exists(args.team_map):
            with open(args.team_map, "r", encoding="utf-8") as handle:
                team_map = json.load(handle)
        else:
            team_map = json.loads(args.team_map)
        converted: dict[int, dict[str, str]] = {}
        for key, value in (team_map or {}).items():
            try:
                key_int = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                converted[key_int] = value
        team_map = converted
    elif not args.skip_team_lookup:
        try:
            team_map = fetch_all_teams(api_key, args.base_url, args.per_page)
        except Exception:
            team_map = {}

    rows: list[dict[str, Any]] = []
    for injury in injuries:
        player = injury.get("player") or {}
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        status = str(injury.get("status") or "").strip()
        status_key = status.lower()
        minutes_multiplier = status_map.get(status_key)
        team_id = player.get("team_id")
        team_name = None
        team_abbr = None
        if isinstance(team_id, (int, str)):
            try:
                team_id_int = int(team_id)
            except (TypeError, ValueError):
                team_id_int = None
            else:
                team_meta = team_map.get(team_id_int, {})
                team_name = team_meta.get("team") or None
                team_abbr = team_meta.get("team_abbr") or None
        rows.append(
            {
                "date": snapshot_date,
                "player_name": name,
                "status": status,
                "return_date": injury.get("return_date"),
                "description": injury.get("description"),
                "minutes_multiplier": minutes_multiplier,
                "source": "balldontlie",
                "player_id": player.get("id"),
                "team_id": team_id,
                "team": team_name,
                "team_abbr": team_abbr,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.append and output_path.exists():
        existing_keys: set[tuple[str, str]] = set()
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                existing_keys.add((row.get("date", ""), row.get("player_name", "")))
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["date"])
            if output_path.stat().st_size == 0:
                writer.writeheader()
            for row in rows:
                key = (row.get("date", ""), row.get("player_name", ""))
                if key in existing_keys:
                    continue
                writer.writerow(row)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["date"])
            writer.writeheader()
            writer.writerows(rows)

    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{snapshot_date}.csv"
        with snapshot_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["date"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} injury rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

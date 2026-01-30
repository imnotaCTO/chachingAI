from __future__ import annotations

import argparse
import calendar
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from zoneinfo import ZoneInfo


DEFAULT_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_points_alternate",
    "player_rebounds_alternate",
    "player_assists_alternate",
)
PROPS_START_DATE = date(2023, 5, 3)


@dataclass(frozen=True)
class OddsAPI:
    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"

    def _get(self, path: str, params: dict[str, Any]) -> requests.Response:
        return requests.get(f"{self.base_url}{path}", params=params, timeout=30)

    def historical_events(self, sport: str, snapshot_iso: str) -> list[dict]:
        params = {"apiKey": self.api_key, "date": snapshot_iso}
        response = self._get(f"/historical/sports/{sport}/events", params)
        response.raise_for_status()
        return response.json()

    def historical_event_odds(
        self,
        sport: str,
        event_id: str,
        snapshot_iso: str,
        regions: str,
        markets: tuple[str, ...],
        odds_format: str,
    ) -> dict:
        params = {
            "apiKey": self.api_key,
            "date": snapshot_iso,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        response = self._get(f"/historical/sports/{sport}/events/{event_id}/odds", params)
        response.raise_for_status()
        return response.json()


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


def _daterange(start: datetime, end: datetime) -> list[datetime]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _parse_month(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError("month must be in YYYY-MM format")
    year = int(parts[0])
    month = int(parts[1])
    if month < 1 or month > 12:
        raise ValueError("month must be between 01 and 12")
    return year, month


def _month_start_end(year: int, month: int, tz: ZoneInfo) -> tuple[datetime, datetime]:
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, tzinfo=tz)
    end = datetime(year, month, last_day, tzinfo=tz)
    return start, end


def _iter_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    months = []
    year = start_year
    month = start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _retry_get(
    func,
    *args,
    max_retries: int = 3,
    base_sleep: float = 2.0,
    **kwargs,
):
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as exc:
            response = exc.response
            if response is None:
                raise
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else base_sleep * (attempt + 1)
                time.sleep(sleep_for)
                continue
            if response.status_code >= 500 and attempt < max_retries:
                time.sleep(base_sleep * (attempt + 1))
                continue
            raise


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_write(path: Path, payload: dict[str, Any], force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return True


def _write_error(path: Path, event: dict[str, Any], snapshot: str, error: str, force: bool) -> None:
    payload = {"snapshot": snapshot, "event": event, "error": error}
    _safe_write(path, payload, force)


def _run_for_range(
    start_date: datetime,
    end_date: datetime,
    *,
    tz: ZoneInfo,
    args: argparse.Namespace,
    client: OddsAPI,
    markets: tuple[str, ...],
    output_dir: Path,
) -> None:
    for day in _daterange(start_date, end_date):
        if day.date() < PROPS_START_DATE:
            print(f"[skip] {day.strftime('%Y-%m-%d')} before props availability ({PROPS_START_DATE})")
            continue
        day_label = day.strftime("%Y-%m-%d")
        snapshot_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        snapshot_start_utc = snapshot_start.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")

        if args.dry_run:
            print(f"[dry-run] events {day_label} @ {snapshot_start_utc}")
            continue

        try:
            events = _retry_get(client.historical_events, args.sport, snapshot_start_utc)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            print(f"[warn] failed events for {day_label} (status {status})")
            continue
        except requests.RequestException as exc:
            print(f"[warn] failed events for {day_label} ({exc.__class__.__name__})")
            continue

        if isinstance(events, dict):
            events = events.get("data", [])

        events_path = output_dir / day_label / "events.json"
        _safe_write(events_path, {"snapshot": snapshot_start_utc, "events": events}, args.force)

        for event in events:
            commence = event.get("commence_time")
            if not commence:
                continue
            commence_dt = _parse_datetime(str(commence)).astimezone(tz)
            if commence_dt.date() != day.date():
                continue
            event_id = event.get("id")
            if not event_id:
                continue
            snapshot_time = (commence_dt - timedelta(minutes=args.snapshot_minutes)).astimezone(
                ZoneInfo("UTC")
            )
            snapshot_iso = snapshot_time.isoformat().replace("+00:00", "Z")
            output_path = output_dir / day_label / f"{event_id}.json"
            if output_path.exists() and not args.force:
                continue

            if args.dry_run:
                print(f"[dry-run] odds {event_id} @ {snapshot_iso}")
                continue

            try:
                odds = _retry_get(
                    client.historical_event_odds,
                    args.sport,
                    event_id,
                    snapshot_iso,
                    args.regions,
                    markets,
                    args.odds_format,
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                error_text = ""
                if exc.response is not None:
                    try:
                        error_json = exc.response.json()
                        error_text = json.dumps(error_json, ensure_ascii=True)
                    except ValueError:
                        error_text = exc.response.text or ""
                print(f"[warn] failed odds {event_id} on {day_label} (status {status})")
                error_path = output_dir / day_label / f"{event_id}.error.json"
                _write_error(error_path, event, snapshot_iso, error_text, args.force)
                continue
            except requests.RequestException as exc:
                print(f"[warn] failed odds {event_id} on {day_label} ({exc.__class__.__name__})")
                error_path = output_dir / day_label / f"{event_id}.error.json"
                _write_error(error_path, event, snapshot_iso, exc.__class__.__name__, args.force)
                continue

            payload = {
                "snapshot": snapshot_iso,
                "event": event,
                "odds": odds,
            }
            _safe_write(output_path, payload, args.force)
            time.sleep(args.sleep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical Odds API player props.")
    parser.add_argument("--start", default="2023-05-03", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--month", default=None, help="Month to download (YYYY-MM).")
    parser.add_argument("--month-end", default=None, help="Optional end month (YYYY-MM).")
    parser.add_argument("--timezone", default="America/New_York", help="Local timezone for date grouping.")
    parser.add_argument("--snapshot-minutes", type=int, default=60, help="Minutes before tipoff.")
    parser.add_argument("--sport", default="basketball_nba", help="Sport key.")
    parser.add_argument("--regions", default="us", help="Odds API regions.")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated markets.")
    parser.add_argument("--odds-format", default="american", help="Odds format.")
    parser.add_argument("--output-dir", default="data/odds_history", help="Output directory.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned requests only.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ODDS_API_KEY. Set it in the environment or .env file.")

    tz = ZoneInfo(args.timezone)
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=tz)
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=tz)
        if args.end
        else datetime.now(tz)
    )
    output_dir = Path(args.output_dir)
    markets = tuple(token.strip() for token in args.markets.split(",") if token.strip())

    client = OddsAPI(api_key=api_key)

    if args.month:
        start_year, start_month = _parse_month(args.month)
        if args.month_end:
            end_year, end_month = _parse_month(args.month_end)
        else:
            end_year, end_month = start_year, start_month
        for year, month in _iter_months(start_year, start_month, end_year, end_month):
            month_start, month_end = _month_start_end(year, month, tz)
            _run_for_range(
                month_start,
                month_end,
                tz=tz,
                args=args,
                client=client,
                markets=markets,
                output_dir=output_dir,
            )
    else:
        _run_for_range(
            start_date,
            end_date,
            tz=tz,
            args=args,
            client=client,
            markets=markets,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

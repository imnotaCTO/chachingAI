from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import requests

DEFAULT_MARKETS = ("player_points", "player_rebounds", "player_assists")


@dataclass(frozen=True)
class OddsAPIClient:
    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"

    def get_odds(
        self,
        sport: str = "basketball_nba",
        regions: str = "us",
        markets: Sequence[str] = DEFAULT_MARKETS,
        odds_format: str = "american",
        date_format: str = "iso",
    ) -> list[dict]:
        """Fetch odds payloads for the given markets."""
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        url = f"{self.base_url}/sports/{sport}/odds"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


def extract_player_props(
    odds_payload: Iterable[dict],
    markets: Sequence[str] = DEFAULT_MARKETS,
) -> list[dict]:
    """Normalize Odds API payloads into flat player prop rows."""
    extracted: list[dict] = []
    for event in odds_payload:
        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker.get("title")
            timestamp = bookmaker.get("last_update")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                if market_key not in markets:
                    continue
                for outcome in market.get("outcomes", []):
                    extracted.append(
                        {
                            "player_name": outcome.get("description") or outcome.get("name"),
                            "market_type": market_key,
                            "line": outcome.get("point"),
                            "odds": outcome.get("price"),
                            "sportsbook": sportsbook,
                            "timestamp": timestamp,
                            "event_id": event.get("id"),
                            "game": event.get("home_team"),
                            "opponent": event.get("away_team"),
                        }
                    )
    return extracted

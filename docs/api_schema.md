# API Schema (Draft)

This is a minimal API contract to support the UI flows. Names are aligned with existing modules.

## Endpoints

### GET /api/health
Response:
```json
{ "status": "ok", "version": "0.2.0" }
```

### GET /api/sportsbooks
Response:
```json
{ "sportsbooks": ["DraftKings", "FanDuel"] }
```

### GET /api/events
Query:
- date: YYYY-MM-DD
- sport: default "basketball_nba"
- team: optional filter
Response:
```json
{
  "events": [
    {
      "event_id": "abcd",
      "home_team": "Denver Nuggets",
      "away_team": "Los Angeles Lakers",
      "commence_time": "2026-01-29T01:00:00Z"
    }
  ]
}
```

### GET /api/events/{event_id}/props
Query:
- sportsbook: name
- markets: comma list, default "player_points,player_rebounds,player_assists"
  - alternate markets supported:
    - `player_points_alternate`
    - `player_rebounds_alternate`
    - `player_assists_alternate`
Response:
```json
{
  "event_id": "abcd",
  "sportsbook": "DraftKings",
  "props": [
    {
      "player_name": "Nikola Jokic",
      "market_type": "player_points",
      "line": 25.5,
      "odds": -110,
      "direction": "over"
    }
  ],
  "last_update": "2026-01-29T00:15:00Z"
}
```

### GET /api/players/search
Query:
- q: player name or fragment
Response:
```json
{
  "players": [
    { "player_id": "237", "player_name": "Nikola Jokic" }
  ]
}
```

### GET /api/players/{player_id}/logs
Query:
- source: "balldontlie" | "kaggle"
- season: year
Response (minimal):
```json
{
  "player_id": "237",
  "season": 2024,
  "rows": [
    { "date": "2024-11-01", "points": 25, "rebounds": 10, "assists": 8, "minutes": 34 }
  ]
}
```

### POST /api/parlay/price
Request:
```json
{
  "sportsbook": "DraftKings",
  "events": ["abcd"],
  "stats_source": "balldontlie",
  "season": 2024,
  "assumptions": {
    "correlation_mode": "copula",
    "cross_game_mode": "independent"
  },
  "legs": [
    {
      "event_id": "abcd",
      "player_id": "237",
      "player_name": "Nikola Jokic",
      "stat": "points",
      "line": 25.5,
      "direction": "over",
      "odds": -110
    },
    {
      "event_id": "abcd",
      "player_id": "237",
      "player_name": "Nikola Jokic",
      "stat": "assists",
      "line": 8.5,
      "direction": "over",
      "odds": -105
    }
  ],
  "simulations": 20000
}
```

Response:
```json
{
  "legs": [
    {
      "player_name": "Nikola Jokic",
      "stat": "points",
      "line": 25.5,
      "direction": "over",
      "model_probability": 0.58,
      "implied_probability": 0.524
    }
  ],
  "joint_probability": 0.41,
  "sportsbook_implied_probability": 0.312,
  "model_fair_odds": 144,
  "expected_value": 0.18,
  "diagnostics": {
    "sample_size": 72,
    "correlation_matrix": [[1.0, 0.22], [0.22, 1.0]],
    "data_source": "balldontlie"
  }
}
```

## Notes
- The API can be backed by `OddsAPIClient`, `fetch_player_game_logs_by_name_*`, and `price_parlay_from_samples`.
- For multi-game parlays, `events` contains multiple ids and `cross_game_mode` defaults to "independent".

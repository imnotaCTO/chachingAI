# Intelligent NBA Same Game Parlay Engine

This repository bootstraps the modeling utilities described in the product spec. The initial focus is on
lognormal distribution fitting, correlation estimation, and Gaussian copula simulation to price NBA
Same Game Parlays for player props.

## Modules

- `sgp_engine.modeling` provides lognormal parameter fitting and correlation helpers.
- `sgp_engine.simulation` simulates correlated lognormal outcomes and evaluates parlay legs.
- `sgp_engine.odds` converts American odds and computes expected value.
- `sgp_engine.ingestion` wraps Basketball Reference and The Odds API ingestion.
- `sgp_engine.feature_engineering` builds rolling feature pipelines from historical logs.

## Quickstart

```python
import numpy as np
from sgp_engine.modeling import fit_lognormal_params, lognormal_correlations
from sgp_engine.simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula
from sgp_engine.odds import expected_value

# Example historical samples for PTS/REB/AST
samples = np.array([
    [25, 8, 6],
    [18, 10, 4],
    [32, 7, 9],
    [22, 11, 5],
])

mu = []
sigma = []
for col in range(samples.shape[1]):
    col_mu, col_sigma = fit_lognormal_params(samples[:, col])
    mu.append(col_mu)
    sigma.append(col_sigma)

correlation = lognormal_correlations(samples)

generated = simulate_lognormal_copula(
    mu=np.array(mu),
    sigma=np.array(sigma),
    correlation=correlation,
    simulations=20000,
)

legs = [
    ParlayLeg(index=0, line=24.5, direction="over"),
    ParlayLeg(index=2, line=5.5, direction="over"),
]

result = evaluate_legs(generated, legs)
print(result.leg_probabilities, result.joint_probability)
print("EV:", expected_value(result.joint_probability, odds=220))
```

## Pricing a Parlay (Offline)

```python
import numpy as np
from sgp_engine.pipeline import LegSpec, price_parlay_from_samples

samples = np.array([
    [25, 8, 6],
    [18, 10, 4],
    [32, 7, 9],
    [22, 11, 5],
])

legs = [
    LegSpec(index=0, line=24.5, direction="over", player="Player A", stat="PTS"),
    LegSpec(index=2, line=5.5, direction="over", player="Player A", stat="AST"),
]

recommendation = price_parlay_from_samples(
    samples,
    legs,
    sportsbook_odds=220,
    game="LAL vs DEN",
)
print(recommendation.joint_probability, recommendation.model_fair_odds, recommendation.expected_value)
```

## Data Ingestion

```python
from sgp_engine.ingestion import OddsAPIClient, extract_player_props, fetch_player_game_logs

# Basketball Reference
player_logs = fetch_player_game_logs(season=2024)

# The Odds API
client = OddsAPIClient(api_key="YOUR_API_KEY")
odds_payload = client.get_odds()
props = extract_player_props(odds_payload)
```

## Line Matching

```python
from sgp_engine.ingestion import OddsAPIClient, extract_player_props, fetch_player_game_logs_by_name
from sgp_engine.matching import match_props_to_player_ids, props_to_leg_specs

player_logs = fetch_player_game_logs_by_name(player_name="Nikola Jokic", season=2024, api_key="YOUR_BALLDONTLIE_KEY")
client = OddsAPIClient(api_key="YOUR_API_KEY")
props = extract_player_props(client.get_odds())
matched = match_props_to_player_ids(props, player_logs)
legs = props_to_leg_specs(matched)
```

## Feature Engineering

```python
from sgp_engine.feature_engineering import build_player_feature_dataset
from sgp_engine.ingestion import fetch_player_game_logs, fetch_team_game_logs

player_logs = fetch_player_game_logs(season=2024)
team_logs = fetch_team_game_logs(season=2024)

features = build_player_feature_dataset(
    player_logs,
    team_logs=team_logs,
    window=10,
    ema_span=5,
    min_games=10,
    min_minutes=10.0,
)
```

## Live Parlay Pricing Script

```powershell
python scripts/price_live_parlay.py --player "Nikola Jokic" --season 2024 --sportsbook-odds 220 --stats points assists
```

If name matching fails, you can supply a BallDontLie player id:

```powershell
python scripts/price_live_parlay.py --player "Nikola Jokic" --player-id 237 --season 2024 --sportsbook-odds 220 --stats points assists
```

Requires:
- `ODDS_API_KEY` for The Odds API
- `BALLDONTLIE_API_KEY` for BallDontLie (Game Player Stats endpoint access; free tier does not include stats)

Optional BallDontLie overrides:
- `BALLDONTLIE_BASE_URL` (default `https://api.balldontlie.io/v1`)
- `BALLDONTLIE_AUTH_SCHEME` (set to `Bearer` if your key requires it)

Scrape fallback (local/prototyping use only):
- Pass `--allow-scrape-fallback` to fall back to Basketball Reference if BallDontLie fails.
- If name lookup fails, pass `--bbr-player-id` (e.g., `jokicni01`).
- If Basketball Reference blocks requests, save the game log HTML and pass `--bbr-html-path`.

To use nba_api instead of BallDontLie:

```powershell
python scripts/price_live_parlay.py --stats-source nba_api --player "Nikola Jokic" --season 2024 --sportsbook-odds 220 --stats points assists
```

To use BallDontLie for daily EV scan:

```powershell
python scripts/find_ev_props.py --stats-source balldontlie --date today --season 2024 --min-games 15 --min-minutes 20 --min-ev 0.03 --stats points rebounds assists
```

The script will list upcoming events from The Odds API and prompt you to choose a game. You can also skip the prompt by passing `--event-id`.
You can also filter the list with `--team`.

To list available player props for a selected event:

```powershell
python scripts/price_live_parlay.py --stats-source nba_api --player "Nikola Jokic" --season 2024 --sportsbook-odds 220 --stats points assists --list-players
```

To auto-select a player from the chosen event:

```powershell
python scripts/price_live_parlay.py --auto-player --stats-source nba_api --season 2024 --sportsbook-odds 220 --stats points assists
```

To use the Kaggle dataset (PlayerStatistics.csv in repo root):

```powershell
python scripts/price_live_parlay.py --stats-source kaggle --player "Nikola Jokic" --season 2026 --sportsbook-odds 220 --stats points assists
```

## Find +EV Props (Daily Slate)

```powershell
python scripts/find_ev_props.py --date today --season 2024 --min-games 15 --min-minutes 20 --min-ev 0.03 --stats points rebounds assists
```

This writes a CSV to `outputs/` and prints the top results to the console.

To use the Kaggle dataset for daily EV scan:

```powershell
python scripts/find_ev_props.py --stats-source kaggle --kaggle-path PlayerStatistics.csv --date today --season 2026 --min-games 15 --min-minutes 20 --min-ev 0.03 --stats points rebounds assists
```

The EV scan now prompts you to select a single event (use `--event-id` to skip).
Caching is enabled by default in `.cache/ev_props` with a 12-hour TTL.

## Parlay Builder

Build a 10-leg parlay from an EV CSV, enforcing:
- min EV per leg (default 0.10)
- max 2 legs per player
- max 3 legs per game
- min joint probability

```powershell
python scripts/build_parlay.py --ev-csv outputs/ev_props_20260129.csv --season 2026 --stats-source kaggle --kaggle-path PlayerStatistics.csv --min-ev 0.10 --min-joint-prob 0.01 --legs 10
```

The parlay EV is computed against a book-implied parlay price (product of leg implied probabilities).
Teammate correlation is estimated by joining player logs on team+date and correlating log1p stats, optionally filtered by opponent + home/away context (min shared games default 8).

## Next Steps

1. Build matching logic for live prop lines.
2. Expand correlation modeling across teammates and minutes trends.

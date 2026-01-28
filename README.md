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

## Next Steps

1. Build matching logic for live prop lines.
2. Expand correlation modeling across teammates and minutes trends.

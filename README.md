# Intelligent NBA Same Game Parlay Engine

This repository bootstraps the modeling utilities described in the product spec. The initial focus is on
lognormal distribution fitting, correlation estimation, and Gaussian copula simulation to price NBA
Same Game Parlays for player props.

## Modules

- `sgp_engine.modeling` provides lognormal parameter fitting and correlation helpers.
- `sgp_engine.simulation` simulates correlated lognormal outcomes and evaluates parlay legs.
- `sgp_engine.odds` converts American odds and computes expected value.

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

## Next Steps

1. Add data ingestion for Basketball Reference and The Odds API.
2. Build matching logic for live prop lines.
3. Expand correlation modeling across teammates and minutes trends.

# Intelligent NBA Same Game Parlay Engine

## Product Overview

This system is an AI-driven **Same Game Parlay (SGP) modeling engine** for NBA player props (Points, Rebounds, Assists). The goal is to generate **true joint probabilities** for correlated player outcomes within a single game and identify **positive expected value (EV)** parlays relative to sportsbook pricing.

The system explicitly **does not assume independence** between legs. Instead, it models player-level stat distributions and their correlations using historical data, then prices parlays via simulation.

---

## Core Objectives

1. Estimate per-player probability distributions for PTS / REB / AST
2. Model intra-game correlations (same player & teammates)
3. Simulate Same Game Parlays using correlated distributions
4. Compare model probabilities to sportsbook odds
5. Surface +EV SGPs with explainability

---

## Data Sources

### 1. Basketball Reference (Historical Stats)
**Source:** https://github.com/vishaalagartha/basketball_reference_scraper/

Used for:
- Player game logs (PTS, REB, AST, MIN)
- Team-level pace proxies
- Opponent context

Key tables extracted:
- `player_game_logs`
- `team_game_logs`
- `box_scores`

Primary fields:
- `player_id`
- `game_id`
- `date`
- `team`
- `opponent`
- `minutes`
- `points`
- `rebounds`
- `assists`

---

### 2. The Odds API (Market Data)
**Source:** https://the-odds-api.com/

Used for:
- Player prop lines (PTS / REB / AST)
- Over / Under odds
- Same Game Parlay prices

Key endpoints:
- `/sports/basketball_nba/odds`
- Markets:
  - `player_points`
  - `player_rebounds`
  - `player_assists`

Captured fields:
- `player_name`
- `market_type`
- `line`
- `odds`
- `sportsbook`
- `timestamp`

---

## System Architecture

```
Basketball Reference Scraper ──┐
                               ├── Feature Engineering ── Distribution Models
The Odds API ──────────────────┘                                 │
                                                                  ↓
                                                     Correlated Simulation Engine
                                                                  ↓
                                                     Same Game Parlay Evaluator
                                                                  ↓
                                                      +EV SGP Recommendations
```

---

## Feature Engineering

### Player-Level Features

Rolling windows (last N games):
- Mean PTS / REB / AST
- Std dev PTS / REB / AST
- Minutes trend (EMA)

Game context:
- Home vs away
- Opponent team
- Team pace proxy

Stability filters:
- Minimum minutes threshold
- Games played threshold

---

## Statistical Modeling

### Distributional Assumption

Each stat is modeled as **lognormal**:

```
ln(X) ~ Normal(μ, σ²)
```

Where X ∈ {PTS, REB, AST}

Reasoning:
- Non-negative
- Right-skewed
- Heavy tails for ceiling games

---

### Parameter Estimation

For historical samples x₁...xₙ:

```
yᵢ = ln(xᵢ + 1)
μ = mean(y)
σ² = variance(y)
```

Notes:
- `+1` avoids log(0)
- Distribution is treated as shifted lognormal

Approximate mean:
```
E[X] ≈ exp(μ + σ² / 2) - 1
```

---

## Correlation Modeling

### Same Player Correlations

Estimated from historical data:
- Corr(PTS, AST)
- Corr(PTS, REB)
- Corr(REB, AST)

Computed on:
```
ln(PTS+1), ln(REB+1), ln(AST+1)
```

### Teammate Correlations (Optional Extension)

- Usage competition (negative correlation)
- Assist chains (positive correlation)

---

## Simulation Engine

### Gaussian Copula Framework

1. Sample Z ~ Multivariate Normal(0, Σ)
2. Transform each Zᵢ → Xᵢ via lognormal inverse CDF
3. Evaluate prop outcomes vs sportsbook lines
4. Repeat N times (10k–100k)

Outputs:
- Probability each leg hits
- Joint probability of entire parlay

---

## Parlay Pricing & EV

### Sportsbook Implied Probability

For American odds:

```
if odds > 0:
  p = 100 / (odds + 100)
else:
  p = -odds / (-odds + 100)
```

### Expected Value

```
EV = P_model * payout - (1 - P_model)
```

Only parlays with:
- EV > threshold
- Probability > minimum confidence
are surfaced.

---

## Output Schema

```json
{
  "game": "LAL vs DEN",
  "legs": [
    {"player": "Nikola Jokic", "stat": "AST", "line": 8.5, "prob": 0.62},
    {"player": "Nikola Jokic", "stat": "PTS", "line": 25.5, "prob": 0.58}
  ],
  "joint_probability": 0.41,
  "sportsbook_odds": +220,
  "model_fair_odds": +144,
  "expected_value": +0.18
}
```

---

## MVP Milestones

**Phase 1 – Offline Backtest**
- Historical stat ingestion
- Distribution fitting
- Single-game parlay simulation

**Phase 2 – Live Odds Integration**
- Real-time The Odds API ingestion
- Line matching & EV detection

**Phase 3 – Model Iteration**
- Minutes prediction layer
- Injury-based adjustments
- Player clustering

---

## Key Risks & Mitigations

| Risk | Mitigation |
|----|----|
| Independence bias | Copula-based simulation |
| Small sample players | Minutes & games threshold |
| Market efficiency | Focus on correlation mispricing |
| Data latency | Odds snapshot versioning |

---

## Success Criteria

- Out-of-sample SGP ROI > 0
- Stable probability calibration
- Clear explainability per parlay

---

## Future Extensions

- Live in-game SGPs
- PRA combined markets
- Reinforcement learning for leg selection
- Bankroll optimization layer

---

*This document defines the full MVP scope for an intelligent NBA Same Game Parlay engine built on Basketball Reference historical data and The Odds API market pricing.*


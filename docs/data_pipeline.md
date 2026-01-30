# Data Pipeline Requirements

## Goals
- Produce aligned samples for legs in a single game to estimate correlation.
- Support single-player and multi-player same-game parlays.
- Allow multi-game parlays with explicit independence assumptions.

## Data Sources
- Props and odds: The Odds API (event, market, line, odds, sportsbook).
- Player logs: BallDontLie or Kaggle.
- Optional team context: pace, home/away, opponent.

## Core Entities
- Event: game on a specific date with home/away teams.
- Player: id + normalized name.
- Game log: per-player, per-game stats with date and opponent.
- Leg: stat, line, direction, odds, event_id, sportsbook.

## Pipeline: Single-Player Same-Game
1) Select event and sportsbook.
2) Pull props for event and sportsbook.
3) Fetch player logs (season) and filter to required stats.
4) Fit lognormal params and simulate copula with correlations across stats.
5) Price the parlay and return diagnostics (sample size, data source).

## Pipeline: Multi-Player Same-Game
1) Select event and sportsbook.
2) Pull props and select multiple players.
3) Fetch logs for each player for the same season.
4) Align samples on game date or game id to form a joint matrix.
   - If a player missed a game (DNP), drop that row or impute conservatively.
5) Build a correlation matrix across all selected stats and players.
6) Project to PSD if needed, then simulate and price.

## Pipeline: Multi-Game Parlays
1) Select multiple events and sportsbook.
2) Build per-event joint samples (same-game process).
3) Combine event-level probabilities.
   - Default: assume independence across events.
   - Optional: cross-game correlation if a model exists.
4) Surface independence warning in the UI when enabled.

## Data Quality Rules
- Minimum sample size per stat (suggested: 10-15 games).
- Minimum minutes threshold.
- Log1p transform to handle zeros.
- Validate that correlation matrices are PSD before Cholesky.
- Optional home/away adjustment: scale samples by the player’s home vs away split for the event context.

## Caching and Performance
- Cache player logs by (source, season, player).
- Cache props payloads by event_id and sportsbook.
- Persist priced parlay requests for reproducibility.

## Injury Snapshots (Daily)
- Take a daily snapshot of BallDontLie injuries so historical backtests can join injuries by date.
- Recommended schedule: once per day at 9:00 AM ET (and optionally again mid-day).
- Use a status-to-multiplier config (`config/injury_status_map.json`) to keep the mapping consistent.

Example command (append to history + write per-day snapshot):

```powershell
python scripts/sync_bdl_injuries.py --append --output data/injuries_bdl.csv --snapshot-dir data/injuries/bdl
```

Team/opponent injury context is derived from the `team` / `team_abbr` columns in the injuries file and is exposed
as aggregate counts + severity in the props API + backtest outputs.

## Known Gaps
- Multi-player same-game alignment is not implemented in code.
- Cross-game correlations are not modeled yet.
- Injury/minutes projections are not integrated.

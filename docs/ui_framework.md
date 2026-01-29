# UI Framework - Parlay Builder

## End Goal
- Allow a user to build a parlay leg by leg for a single sportsbook.
- Support a single game or multiple games.
- Show model probability vs sportsbook implied probability, plus expected value.

## Product Phases
### Phase 1 (MVP)
- Single game, single sportsbook, single player multi-stat legs.
- Fast leg selection from live props.
- Clear summary: joint probability, implied probability, fair odds, EV.

### Phase 2
- Same-game multi-player legs.
- Correlation diagnostics (heatmap + confidence warnings).
- Saved parlays and sharing.

### Phase 3
- Multi-game parlays.
- Explicit independence assumption toggle and warnings.
- Optional cross-game correlation modeling if data supports it.

## Primary User Flows
### Flow A: Single-Game, Single-Player
1) Select sportsbook.
2) Select date + game.
3) Pick player.
4) Add legs from live prop list (line, direction, odds).
5) View model probability and EV in summary panel.

### Flow B: Same-Game, Multi-Player
1) Select sportsbook and game.
2) Add legs across multiple players.
3) Review correlation diagnostics and sample sizes.
4) Confirm parlay summary and EV.

### Flow C: Multi-Game Parlay
1) Select sportsbook and date range.
2) Add legs from multiple games.
3) Review independence assumption or cross-game model status.
4) Confirm parlay summary and EV.

## Page Structure (Single Game)
- Header: date, sportsbook selector, refresh timestamp.
- Left rail: games list and filters (team, player, stat, line range).
- Main: parlay builder with leg cards and quick-add from props.
- Right panel: parlay summary + diagnostics.

## Key Components
- SportsbookPicker
- GameList + GameCard
- PlayerSearch
- PropList + PropRow (line, odds, direction)
- ParlayLegCard
- ParlaySummary (joint prob, implied prob, EV, fair odds)
- CorrelationDiagnostics (Phase 2+)
- DataQualityBadge (sample size, source)

## UX Rules
- One sportsbook per parlay (hard constraint for pricing consistency).
- Show implied probability next to sportsbook odds.
- Always display sample size and data source for transparency.
- Warn on low sample sizes or high model uncertainty.
- Multi-game: show independence assumption by default unless cross-game model exists.

## State Model (Minimal)
- selected_sportsbook
- selected_date
- selected_events (1..n)
- selected_player (optional)
- legs[]: {player_id, player_name, stat, line, direction, odds, event_id}
- model_result: {leg_probabilities, joint_probability, implied_probability, fair_odds, ev}
- diagnostics: {sample_sizes, correlation_matrix, data_source}

## Success Metrics
- Time to first priced parlay.
- Drop-off between game selection and leg addition.
- User comprehension of model vs implied probabilities.


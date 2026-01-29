from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from sgp_engine.ingestion import (
    fetch_player_game_logs_by_name,
    fetch_player_game_logs_by_name_kaggle,
    fetch_player_game_logs_by_name_nba_api,
)
from sgp_engine.matching import normalize_player_name
from sgp_engine.modeling import fit_lognormal_params, lognormal_correlations, make_psd_correlation
from sgp_engine.odds import american_to_prob, expected_value, prob_to_american
from sgp_engine.simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula

STAT_ORDER = ("points", "rebounds", "assists")


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


def _cache_filename(name: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")
    return safe or "unknown"


def _normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_game_teams(game: str) -> tuple[str | None, str | None]:
    if " @ " not in game:
        return None, None
    away, home = game.split(" @ ", 1)
    return away.strip(), home.strip()


def _resolve_game_context(leg: CandidateLeg, team_map: dict[str, str | None]) -> tuple[str | None, str | None]:
    away, home = _parse_game_teams(leg.game)
    if not away or not home:
        return None, None
    team = leg.team or team_map.get(leg.player_key)
    if not team:
        return None, None
    team_norm = _normalize_team_name(team)
    away_norm = _normalize_team_name(away)
    home_norm = _normalize_team_name(home)
    if team_norm and home_norm and team_norm in home_norm:
        return away, "Home"
    if team_norm and away_norm and team_norm in away_norm:
        return home, "Away"
    return None, None


def _load_cached_logs(
    cache_dir: Path,
    source: str,
    season: int,
    player_name: str,
    ttl_hours: float,
) -> pd.DataFrame | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if ttl_hours >= 0 and age_hours > ttl_hours:
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _save_cached_logs(cache_dir: Path, source: str, season: int, player_name: str, logs: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{season}_{_cache_filename(player_name)}.pkl"
    path = cache_dir / filename
    logs.to_pickle(path)


def _fetch_logs(
    player_name: str,
    source: str,
    season: int,
    kaggle_path: str,
    bdl_api_key: str | None,
) -> pd.DataFrame:
    if source == "nba_api":
        return fetch_player_game_logs_by_name_nba_api(player_name=player_name, season=season)
    if source == "kaggle":
        return fetch_player_game_logs_by_name_kaggle(
            player_name=player_name,
            season_end_year=season,
            data_path=kaggle_path,
        )
    if not bdl_api_key:
        raise SystemExit("Missing BALLDONTLIE_API_KEY for balldontlie source.")
    return fetch_player_game_logs_by_name(
        player_name=player_name,
        season=season,
        api_key=bdl_api_key,
    )


@dataclass(frozen=True)
class CandidateLeg:
    player: str
    player_key: str
    stat: str
    line: float
    direction: str
    odds: float
    model_prob: float
    team: str | None
    game: str
    event_id: str | None


def _build_candidate_pool(df: pd.DataFrame, min_ev: float, max_candidates: int) -> list[CandidateLeg]:
    df = df.copy()
    df = df[df["ev"] >= min_ev]
    df = df.sort_values(by="ev", ascending=False)
    df = df.drop_duplicates(subset=["player", "stat", "line", "direction"], keep="first")
    if max_candidates > 0:
        df = df.head(max_candidates)
    candidates: list[CandidateLeg] = []
    for _, row in df.iterrows():
        player = str(row["player"])
        candidates.append(
            CandidateLeg(
                player=player,
                player_key=normalize_player_name(player),
                stat=str(row["stat"]).lower(),
                line=float(row["line"]),
                direction=str(row.get("direction", "over")),
                odds=float(row["odds"]),
                model_prob=float(row["model_prob"]),
                team=str(row["team"]) if "team" in row and pd.notna(row["team"]) else None,
                game=str(row.get("game", "")),
                event_id=str(row.get("event_id")) if "event_id" in row and pd.notna(row["event_id"]) else None,
            )
        )
    return candidates


def _get_player_stat_params(
    logs: pd.DataFrame,
    stat: str,
    min_minutes: float,
    min_games: int,
) -> tuple[float, float] | None:
    if stat not in logs.columns:
        return None
    filtered = logs[logs["minutes"] >= min_minutes]
    samples = filtered[stat].dropna().to_numpy()
    if samples.size < min_games:
        return None
    return fit_lognormal_params(samples.astype(float))


def _get_player_correlation(logs: pd.DataFrame, min_minutes: float) -> np.ndarray | None:
    missing = [stat for stat in STAT_ORDER if stat not in logs.columns]
    if missing:
        return None
    filtered = logs[logs["minutes"] >= min_minutes]
    samples = filtered[list(STAT_ORDER)].dropna().to_numpy()
    if samples.shape[0] < 2:
        return None
    return lognormal_correlations(samples.astype(float))


def _corr_for_pair(
    player_corr: dict[str, np.ndarray],
    player_key: str,
    stat_a: str,
    stat_b: str,
) -> float:
    if player_key not in player_corr:
        return 0.0
    matrix = player_corr[player_key]
    index = {stat: idx for idx, stat in enumerate(STAT_ORDER)}
    if stat_a not in index or stat_b not in index:
        return 0.0
    return float(matrix[index[stat_a], index[stat_b]])


def _teammate_correlation(
    logs_map: dict[str, pd.DataFrame],
    player_key_a: str,
    player_key_b: str,
    stat_a: str,
    stat_b: str,
    min_shared_games: int,
    cache: dict[tuple[str, ...], float],
    opponent: str | None = None,
    location: str | None = None,
) -> float:
    key = tuple(sorted([player_key_a, player_key_b])) + (stat_a, stat_b)
    if opponent:
        key = key + (str(opponent),)
    if location:
        key = key + (str(location),)
    if key in cache:
        return cache[key]
    logs_a = logs_map.get(player_key_a)
    logs_b = logs_map.get(player_key_b)
    if logs_a is None or logs_b is None:
        cache[key] = 0.0
        return 0.0
    needed_cols = {"date", "team", stat_a}
    if not needed_cols.issubset(logs_a.columns) or not {"date", "team", stat_b}.issubset(logs_b.columns):
        cache[key] = 0.0
        return 0.0
    required_a = {"date", "team", stat_a}
    required_b = {"date", "team", stat_b}
    if opponent and ("opponent" not in logs_a.columns or "opponent" not in logs_b.columns):
        cache[key] = 0.0
        return 0.0
    if location and ("location" not in logs_a.columns or "location" not in logs_b.columns):
        cache[key] = 0.0
        return 0.0
    cols_a = list(required_a)
    cols_b = list(required_b)
    for col in ("opponent", "location"):
        if col in logs_a.columns and col not in cols_a:
            cols_a.append(col)
        if col in logs_b.columns and col not in cols_b:
            cols_b.append(col)
    logs_a_use = logs_a[cols_a].copy()
    logs_b_use = logs_b[cols_b].copy()
    if opponent:
        opp_norm = _normalize_team_name(opponent)
        logs_a_use = logs_a_use[
            logs_a_use["opponent"].astype(str).map(_normalize_team_name) == opp_norm
        ]
        logs_b_use = logs_b_use[
            logs_b_use["opponent"].astype(str).map(_normalize_team_name) == opp_norm
        ]
    if location:
        loc_norm = str(location).lower()
        logs_a_use = logs_a_use[logs_a_use["location"].astype(str).str.lower() == loc_norm]
        logs_b_use = logs_b_use[logs_b_use["location"].astype(str).str.lower() == loc_norm]
    merged = logs_a_use[["date", "team", stat_a]].merge(
        logs_b_use[["date", "team", stat_b]],
        on=["date", "team"],
        how="inner",
        suffixes=("_a", "_b"),
    ).dropna()
    if merged.shape[0] < min_shared_games:
        cache[key] = 0.0
        return 0.0
    x = np.log1p(merged[stat_a].astype(float).to_numpy())
    y = np.log1p(merged[stat_b].astype(float).to_numpy())
    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    cache[key] = corr
    return corr


def _simulate_joint_probability(
    legs: list[CandidateLeg],
    params: dict[tuple[str, str], tuple[float, float]],
    player_corr: dict[str, np.ndarray],
    logs_map: dict[str, pd.DataFrame],
    team_map: dict[str, str | None],
    min_shared_games: int,
    simulations: int,
) -> float:
    mu = []
    sigma = []
    for leg in legs:
        key = (leg.player_key, leg.stat)
        if key not in params:
            raise ValueError(f"Missing parameters for {leg.player} {leg.stat}")
        leg_mu, leg_sigma = params[key]
        mu.append(leg_mu)
        sigma.append(leg_sigma)
    mu_arr = np.array(mu)
    sigma_arr = np.array(sigma)
    size = len(legs)
    corr = np.eye(size)
    teammate_cache: dict[tuple[str, ...], float] = {}
    context: list[tuple[str | None, str | None]] = []
    for leg in legs:
        context.append(_resolve_game_context(leg, team_map))
    for i in range(size):
        for j in range(i + 1, size):
            if legs[i].player_key == legs[j].player_key:
                corr[i, j] = _corr_for_pair(
                    player_corr,
                    legs[i].player_key,
                    legs[i].stat,
                    legs[j].stat,
                )
                corr[j, i] = corr[i, j]
                continue
            same_game = (legs[i].event_id and legs[j].event_id and legs[i].event_id == legs[j].event_id)
            if not same_game and legs[i].game and legs[j].game and legs[i].game == legs[j].game:
                same_game = True
            team_a = team_map.get(legs[i].player_key) or legs[i].team
            team_b = team_map.get(legs[j].player_key) or legs[j].team
            if same_game and team_a and team_b and team_a == team_b:
                opponent, location = context[i]
                if opponent and location:
                    corr[i, j] = _teammate_correlation(
                        logs_map,
                        legs[i].player_key,
                        legs[j].player_key,
                        legs[i].stat,
                        legs[j].stat,
                        min_shared_games,
                        teammate_cache,
                        opponent=opponent,
                        location=location,
                    )
                else:
                    corr[i, j] = _teammate_correlation(
                        logs_map,
                        legs[i].player_key,
                        legs[j].player_key,
                        legs[i].stat,
                        legs[j].stat,
                        min_shared_games,
                        teammate_cache,
                    )
                corr[j, i] = corr[i, j]
    corr = make_psd_correlation(corr)
    sims = simulate_lognormal_copula(
        mu=mu_arr,
        sigma=sigma_arr,
        correlation=corr,
        simulations=simulations,
    )
    parlay_legs = [
        ParlayLeg(index=i, line=leg.line, direction=leg.direction) for i, leg in enumerate(legs)
    ]
    result = evaluate_legs(sims, parlay_legs)
    return float(result.joint_probability)


@dataclass
class ParlayState:
    legs: list[CandidateLeg]
    player_counts: Counter
    game_counts: Counter
    approx_prob: float
    book_prob: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a parlay from an EV CSV.")
    parser.add_argument("--ev-csv", required=True, help="Path to EV props CSV.")
    parser.add_argument("--season", type=int, default=2024, help="NBA season year for logs.")
    parser.add_argument(
        "--stats-source",
        default="nba_api",
        choices=["nba_api", "balldontlie", "kaggle"],
        help="Source for player game logs.",
    )
    parser.add_argument("--kaggle-path", default="PlayerStatistics.csv", help="Path to Kaggle PlayerStatistics.csv.")
    parser.add_argument("--min-games", type=int, default=15, help="Minimum games required.")
    parser.add_argument("--min-minutes", type=float, default=20.0, help="Minimum minutes per game.")
    parser.add_argument("--min-ev", type=float, default=0.10, help="Minimum EV threshold for candidate legs.")
    parser.add_argument("--min-joint-prob", type=float, default=0.01, help="Minimum joint probability.")
    parser.add_argument("--legs", type=int, default=10, help="Number of legs to build.")
    parser.add_argument("--max-legs-per-player", type=int, default=2, help="Max legs per player.")
    parser.add_argument("--max-legs-per-game", type=int, default=3, help="Max legs per game.")
    parser.add_argument("--beam-width", type=int, default=100, help="Beam width.")
    parser.add_argument("--max-candidates", type=int, default=150, help="Max candidate legs.")
    parser.add_argument("--simulations", type=int, default=20000, help="Simulations per parlay.")
    parser.add_argument("--min-shared-games", type=int, default=8, help="Min shared games for teammate correlation.")
    parser.add_argument("--cache-dir", default=".cache/ev_props", help="Directory for player log cache.")
    parser.add_argument("--cache-ttl-hours", type=float, default=12.0, help="Cache TTL in hours.")
    parser.add_argument("--progress-every", type=int, default=10, help="Progress every N player log loads.")
    parser.add_argument("--output-csv", default=None, help="Output CSV for parlays.")
    args = parser.parse_args()

    load_dotenv()
    bdl_api_key = os.environ.get("BALLDONTLIE_API_KEY")

    ev_df = pd.read_csv(args.ev_csv)
    required = {"player", "stat", "line", "odds", "direction", "model_prob", "ev"}
    missing = required - set(ev_df.columns)
    if missing:
        raise SystemExit(f"Missing columns in EV CSV: {sorted(missing)}")

    candidates = _build_candidate_pool(ev_df, args.min_ev, args.max_candidates)
    if len(candidates) < args.legs:
        raise SystemExit("Not enough candidate legs after filtering.")

    cache_dir = Path(args.cache_dir)
    player_logs_cache: dict[str, pd.DataFrame] = {}
    params: dict[tuple[str, str], tuple[float, float]] = {}
    player_corr: dict[str, np.ndarray] = {}
    team_map: dict[str, str | None] = {}
    loaded = 0

    for leg in candidates:
        if leg.player_key in player_logs_cache:
            continue
        cached = _load_cached_logs(cache_dir, args.stats_source, args.season, leg.player, args.cache_ttl_hours)
        if cached is None:
            logs = _fetch_logs(leg.player, args.stats_source, args.season, args.kaggle_path, bdl_api_key)
            _save_cached_logs(cache_dir, args.stats_source, args.season, leg.player, logs)
        else:
            logs = cached
        player_logs_cache[leg.player_key] = logs
        if "team" in logs.columns:
            non_null = logs["team"].dropna().astype(str)
            team_map[leg.player_key] = non_null.iloc[-1] if not non_null.empty else None
        loaded += 1
        if args.progress_every > 0 and loaded % args.progress_every == 0:
            print(f"Loaded logs for {loaded} players...")

    for leg in candidates:
        key = (leg.player_key, leg.stat)
        if key in params:
            continue
        logs = player_logs_cache.get(leg.player_key)
        if logs is None:
            continue
        stat_params = _get_player_stat_params(logs, leg.stat, args.min_minutes, args.min_games)
        if stat_params:
            params[key] = stat_params
        if leg.player_key not in player_corr:
            corr = _get_player_correlation(logs, args.min_minutes)
            if corr is not None:
                player_corr[leg.player_key] = corr

    filtered_candidates = []
    for leg in candidates:
        if (leg.player_key, leg.stat) in params:
            filtered_candidates.append(leg)
    candidates = filtered_candidates
    if len(candidates) < args.legs:
        raise SystemExit("Not enough candidates with valid parameters.")

    beam = [
        ParlayState(
            legs=[],
            player_counts=Counter(),
            game_counts=Counter(),
            approx_prob=1.0,
            book_prob=1.0,
        )
    ]

    for _ in range(args.legs):
        next_beam: list[ParlayState] = []
        for state in beam:
            for cand in candidates:
                if cand in state.legs:
                    continue
                if state.player_counts[cand.player_key] >= args.max_legs_per_player:
                    continue
                game_key = cand.event_id or cand.game
                if state.game_counts[game_key] >= args.max_legs_per_game:
                    continue
                new_legs = state.legs + [cand]
                new_player_counts = state.player_counts.copy()
                new_player_counts[cand.player_key] += 1
                new_game_counts = state.game_counts.copy()
                new_game_counts[game_key] += 1
                new_state = ParlayState(
                    legs=new_legs,
                    player_counts=new_player_counts,
                    game_counts=new_game_counts,
                    approx_prob=state.approx_prob * cand.model_prob,
                    book_prob=state.book_prob * american_to_prob(cand.odds),
                )
                next_beam.append(new_state)
        if not next_beam:
            raise SystemExit("No valid parlays found with the given constraints.")
        next_beam.sort(key=lambda x: x.approx_prob, reverse=True)
        beam = next_beam[: args.beam_width]

    results: list[dict] = []
    for idx, state in enumerate(beam, start=1):
        joint_prob = _simulate_joint_probability(
            state.legs,
            params,
            player_corr,
            player_logs_cache,
            team_map,
            args.min_shared_games,
            args.simulations,
        )
        if joint_prob < args.min_joint_prob:
            continue
        book_prob = max(min(state.book_prob, 1 - 1e-6), 1e-6)
        book_odds = prob_to_american(book_prob)
        ev = expected_value(joint_prob, book_odds)
        results.append(
            {
                "rank": idx,
                "legs": len(state.legs),
                "joint_probability": joint_prob,
                "book_implied_probability": book_prob,
                "book_implied_odds": book_odds,
                "expected_value": ev,
                "legs_detail": "; ".join(
                    f"{leg.player} {leg.stat} {leg.direction} {leg.line} ({leg.odds})"
                    for leg in state.legs
                ),
            }
        )

    if not results:
        raise SystemExit("No parlays met the minimum joint probability.")

    out_df = pd.DataFrame(results).sort_values(by="expected_value", ascending=False)
    output_path = args.output_csv or "outputs/parlays.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} parlays to {output_path}")
    print(out_df.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

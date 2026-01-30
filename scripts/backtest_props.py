from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
from scipy.optimize import minimize
from zoneinfo import ZoneInfo

from sgp_engine.ingestion import (
    extract_player_props,
    fetch_player_game_logs_by_name_kaggle,
    fetch_player_game_logs_from_cache,
)
from sgp_engine.matching import DEFAULT_MARKET_MAP, normalize_player_name
from sgp_engine.modeling import fit_lognormal_params, lognormal_correlations, lognormal_mean
from sgp_engine.odds import american_to_prob, expected_value
from sgp_engine.simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula
from sgp_engine.pace import TeamPaceLookup, normalize_team_name


MAIN_MARKETS = ("player_points", "player_rebounds", "player_assists")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _season_end_year(dt: datetime) -> int:
    return dt.year + 1 if dt.month >= 10 else dt.year


def _lognormal_hit_probability(mu: float, sigma: float, line: float, direction: str) -> float:
    if line <= -1:
        raise ValueError("line must be greater than -1 for log1p")
    z = np.log1p(line)
    cdf = norm.cdf(z, loc=mu, scale=sigma)
    if direction.lower() == "over":
        return float(1 - cdf)
    if direction.lower() == "under":
        return float(cdf)
    raise ValueError(f"Unsupported direction: {direction}")


def _iter_event_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in data_dir.rglob("*.json"):
        if path.name == "events.json" or path.name.endswith(".error.json"):
            continue
        files.append(path)
    return sorted(files)


def _load_event_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "odds" in payload and "event" in payload:
        event = payload["event"]
        odds = payload["odds"]
        snapshot = payload.get("snapshot")
        return event, odds, snapshot
    return payload, payload, payload.get("snapshot")


def _match_game_row(
    logs: pd.DataFrame,
    event_date: datetime.date,
    home_team: str | None,
    away_team: str | None,
) -> pd.Series | None:
    if "date" not in logs.columns:
        return None
    logs = logs.copy()
    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
    logs = logs.dropna(subset=["date"])
    candidates = logs[logs["date"].dt.date == event_date]
    if candidates.empty:
        return None
    if home_team and away_team:
        home = home_team.lower().strip()
        away = away_team.lower().strip()
        for _, row in candidates.iterrows():
            team = str(row.get("team", "")).lower()
            opponent = str(row.get("opponent", "")).lower()
            if team in {home, away} or opponent in {home, away}:
                return row
    return candidates.iloc[0]


def _filter_logs_before(
    logs: pd.DataFrame,
    cutoff: datetime,
    min_minutes: float,
) -> pd.DataFrame:
    if "date" in logs.columns:
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        logs = logs[logs["date"] < cutoff]
    if min_minutes > 0 and "minutes" in logs.columns:
        logs = logs[logs["minutes"] >= min_minutes]
    return logs


def _compute_model_stats(
    samples: np.ndarray,
    *,
    use_ema: bool,
    ema_span: int,
) -> tuple[float, float]:
    if use_ema:
        span = max(2, int(ema_span))
        alpha = 2 / (span + 1)
        n = samples.size
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            raise ValueError("ema_weights_invalid")
        log_samples = np.log1p(samples)
        mu = float((weights * log_samples).sum() / weights_sum)
        var = float((weights * (log_samples - mu) ** 2).sum() / weights_sum)
        sigma = float(np.sqrt(max(var, 1e-12)))
        return mu, sigma
    return fit_lognormal_params(samples)


def _predict_minutes(
    logs: pd.DataFrame,
    *,
    cutoff: datetime,
    window: int,
    use_ema: bool,
    ema_span: int,
    min_games: int,
) -> float | None:
    if "minutes" not in logs.columns:
        return None
    logs = logs.copy()
    if "date" in logs.columns:
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        logs = logs[logs["date"] < cutoff]
        logs = logs.sort_values("date")
    minutes = logs["minutes"].dropna().astype(float)
    minutes = minutes[minutes > 0]
    if window and window > 0:
        minutes = minutes.tail(window)
    if minutes.size < max(2, min_games):
        return None
    if use_ema:
        span = max(2, int(ema_span))
        alpha = 2 / (span + 1)
        n = minutes.size
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return None
        return float((weights * minutes.to_numpy()).sum() / weights_sum)
    return float(minutes.mean())


def _apply_home_advantage(
    samples: np.ndarray,
    locations: pd.Series | None,
    context: str | None,
) -> tuple[np.ndarray, float | None, bool]:
    if locations is None or context not in {"home", "away"} or samples.size == 0:
        return samples, None, False
    loc = locations.astype(str).str.lower()
    mask = loc == ("home" if context == "home" else "away")
    if not mask.any():
        return samples, None, False
    overall_mean = float(np.mean(samples))
    target_mean = float(np.mean(samples[mask.to_numpy()]))
    if overall_mean <= 0 or target_mean <= 0:
        return samples, None, False
    factor = target_mean / overall_mean
    return samples * factor, factor, True


def _load_injury_adjustments(path: str) -> dict[tuple[datetime.date, str], float]:
    df = pd.read_csv(path)
    required = {"date", "player_name", "minutes_multiplier"}
    if not required.issubset(df.columns):
        raise ValueError("injuries file must include date, player_name, minutes_multiplier")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "player_name", "minutes_multiplier"])
    df["player_norm"] = df["player_name"].map(normalize_player_name)
    df["minutes_multiplier"] = pd.to_numeric(df["minutes_multiplier"], errors="coerce")
    df = df.dropna(subset=["minutes_multiplier", "player_norm"])
    adjustments: dict[tuple[datetime.date, str], float] = {}
    for row in df.itertuples(index=False):
        adjustments[(row.date, row.player_norm)] = float(row.minutes_multiplier)
    return adjustments


def _load_team_injury_context(path: str) -> dict[tuple[datetime.date, str], dict[str, float | int | None]]:
    df = pd.read_csv(path)
    if "date" not in df.columns or "minutes_multiplier" not in df.columns:
        return {}
    team_cols = [col for col in ("team", "team_name", "team_full", "team_abbr") if col in df.columns]
    if not team_cols:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["minutes_multiplier"] = pd.to_numeric(df["minutes_multiplier"], errors="coerce")
    df = df.dropna(subset=["date", "minutes_multiplier"])

    context: dict[tuple[datetime.date, str], dict[str, float | int | None]] = {}
    for row in df.itertuples(index=False):
        labels: set[str] = set()
        for col in team_cols:
            value = getattr(row, col, None)
            if isinstance(value, str) and value.strip():
                labels.add(value.strip())
        if not labels:
            continue
        multiplier = float(row.minutes_multiplier)
        for label in labels:
            key = (row.date, normalize_team_name(label))
            entry = context.setdefault(
                key,
                {
                    "injury_count": 0,
                    "out_count": 0,
                    "severity_sum": 0.0,
                    "multiplier_sum": 0.0,
                    "avg_multiplier": None,
                },
            )
            if multiplier < 1:
                entry["injury_count"] = int(entry["injury_count"]) + 1
                entry["severity_sum"] = float(entry["severity_sum"]) + max(0.0, 1 - multiplier)
                entry["multiplier_sum"] = float(entry["multiplier_sum"]) + multiplier
                if multiplier <= 0:
                    entry["out_count"] = int(entry["out_count"]) + 1

    for entry in context.values():
        count = int(entry.get("injury_count") or 0)
        if count > 0:
            entry["avg_multiplier"] = float(entry["multiplier_sum"]) / count
        else:
            entry["avg_multiplier"] = None
        entry.pop("multiplier_sum", None)
    return context


def _load_injury_context(
    path: str,
) -> tuple[dict[tuple[datetime.date, str], float], dict[tuple[datetime.date, str], dict[str, float | int | None]]]:
    return _load_injury_adjustments(path), _load_team_injury_context(path)


def _infer_player_team(
    logs: pd.DataFrame,
    event_date: datetime.date,
    home_team: str | None,
    away_team: str | None,
) -> tuple[str | None, str | None]:
    if "team" not in logs.columns:
        return None, None
    team_value = None
    if "date" in logs.columns:
        dated = logs.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce").dt.date
        dated = dated[dated["date"] <= event_date]
        dated = dated.dropna(subset=["team"])
        if not dated.empty:
            team_value = str(dated["team"].iloc[-1]).strip()
    if team_value is None:
        series = logs["team"].dropna()
        if not series.empty:
            team_value = str(series.iloc[-1]).strip()
    if not team_value:
        return None, None

    if home_team and away_team:
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)
        team_norm = normalize_team_name(team_value)
        if team_norm == home_norm:
            return home_team, away_team
        if team_norm == away_norm:
            return away_team, home_team
    return team_value, None


def _lookup_team_context(
    context: dict[tuple[datetime.date, str], dict[str, float | int | None]],
    event_date: datetime.date,
    team: str | None,
) -> dict[str, float | int | None] | None:
    if not context or not team:
        return None
    return context.get((event_date, normalize_team_name(team)))


def _fit_platt_scaling(probs: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    scores = np.log(clipped / (1 - clipped))
    y = outcomes.astype(float)

    def nll(params: np.ndarray) -> float:
        a, b = params
        logits = a * scores + b
        p = 1 / (1 + np.exp(-logits))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    result = minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B")
    if not result.success:
        raise ValueError("platt_fit_failed")
    a, b = result.x
    return float(a), float(b)


def _apply_platt(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    clipped = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    scores = np.log(clipped / (1 - clipped))
    logits = a * scores + b
    return 1 / (1 + np.exp(-logits))


def _model_prob_from_summary(
    lines: np.ndarray,
    directions: pd.Series,
    dist: str,
    log_mean: np.ndarray | None,
    log_std: np.ndarray | None,
    sample_mean: np.ndarray | None,
    sample_var: np.ndarray | None,
) -> np.ndarray:
    probs = np.full(lines.shape, np.nan, dtype=float)
    direction_lower = directions.astype(str).str.lower().to_numpy()
    over_mask = direction_lower == "over"
    under_mask = direction_lower == "under"
    if dist == "lognormal":
        if log_mean is None or log_std is None:
            return probs
        sigma = log_std.astype(float)
        mu = log_mean.astype(float)
        valid = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(lines) & (lines > -1)
        if valid.any():
            z = np.log1p(lines[valid])
            cdf = np.full(lines.shape, np.nan, dtype=float)
            cdf[valid] = norm.cdf(z, loc=mu[valid], scale=sigma[valid])
            probs[over_mask] = 1 - cdf[over_mask]
            probs[under_mask] = cdf[under_mask]
        return probs
    if dist == "gamma":
        if sample_mean is None or sample_var is None:
            return probs
        mean = sample_mean.astype(float)
        var = sample_var.astype(float)
        valid = np.isfinite(mean) & np.isfinite(var) & (mean > 0) & (var > 0) & np.isfinite(lines)
        if valid.any():
            shape = (mean[valid] ** 2) / var[valid]
            scale = var[valid] / mean[valid]
            cdf = np.full(lines.shape, np.nan, dtype=float)
            cdf[valid] = gamma.cdf(lines[valid], a=shape, scale=scale)
            probs[over_mask] = 1 - cdf[over_mask]
            probs[under_mask] = cdf[under_mask]
        return probs
    if dist == "normal":
        if sample_mean is None or sample_var is None:
            return probs
        mean = sample_mean.astype(float)
        var = sample_var.astype(float)
        sigma = np.sqrt(var)
        valid = np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0) & np.isfinite(lines)
        if valid.any():
            cdf = np.full(lines.shape, np.nan, dtype=float)
            cdf[valid] = norm.cdf(lines[valid], loc=mean[valid], scale=sigma[valid])
            probs[over_mask] = 1 - cdf[over_mask]
            probs[under_mask] = cdf[under_mask]
        return probs
    return probs


class _LRUCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(0, int(max_items))
        self._data: OrderedDict[tuple, pd.DataFrame | None] = OrderedDict()

    def get(self, key: tuple) -> pd.DataFrame | None:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: tuple, value: pd.DataFrame | None) -> None:
        if self.max_items <= 0:
            return
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        if len(self._data) > self.max_items:
            self._data.popitem(last=False)


def analyze_outputs(
    props_path: Path,
    args: argparse.Namespace,
    edge_thresholds: list[float],
) -> None:
    summary_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    selection_stats: dict[tuple[str, str], dict[str, float]] = {}
    calibration_stats: dict[tuple[str, str], dict[str, float]] = {}
    calibrated_summary_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    calibrated_selection_stats: dict[tuple[str, str], dict[str, float]] = {}
    calibrated_calibration_stats: dict[tuple[str, str], dict[str, float]] = {}

    def _init_stats() -> dict[str, float]:
        return {
            "count": 0,
            "notable_count": 0,
            "hit_sum": 0,
            "hit_count": 0,
            "model_prob_sum": 0,
            "model_prob_count": 0,
            "implied_prob_sum": 0,
            "implied_prob_count": 0,
            "ev_sum": 0,
            "ev_count": 0,
            "profit_sum": 0,
            "profit_count": 0,
            "notable_profit_sum": 0,
            "notable_profit_count": 0,
            "kelly_profit_sum": 0,
            "kelly_profit_count": 0,
            "kelly_cap_profit_sum": 0,
            "kelly_cap_profit_count": 0,
            "brier_sum": 0,
            "brier_count": 0,
            "logloss_sum": 0,
            "logloss_count": 0,
        }

    def _update(stats: dict[str, float], frame: pd.DataFrame) -> None:
        stats["count"] += int(frame.shape[0])
        stats["notable_count"] += int(frame["notable_bet"].sum())
        if frame["actual_hit"].notna().any():
            stats["hit_sum"] += float(frame["actual_hit"].sum())
            stats["hit_count"] += int(frame["actual_hit"].notna().sum())
        if frame["model_prob"].notna().any():
            stats["model_prob_sum"] += float(frame["model_prob"].sum())
            stats["model_prob_count"] += int(frame["model_prob"].notna().sum())
        if frame["implied_prob"].notna().any():
            stats["implied_prob_sum"] += float(frame["implied_prob"].sum())
            stats["implied_prob_count"] += int(frame["implied_prob"].notna().sum())
        if frame["ev"].notna().any():
            stats["ev_sum"] += float(frame["ev"].sum())
            stats["ev_count"] += int(frame["ev"].notna().sum())
        if frame["profit"].notna().any():
            stats["profit_sum"] += float(frame["profit"].sum())
            stats["profit_count"] += int(frame["profit"].notna().sum())
        notable_profit = frame.loc[frame["notable_bet"], "profit"]
        if notable_profit.notna().any():
            stats["notable_profit_sum"] += float(notable_profit.sum())
            stats["notable_profit_count"] += int(notable_profit.notna().sum())
        if frame["kelly_profit"].notna().any():
            stats["kelly_profit_sum"] += float(frame["kelly_profit"].sum())
            stats["kelly_profit_count"] += int(frame["kelly_profit"].notna().sum())
        if frame["kelly_profit_capped"].notna().any():
            stats["kelly_cap_profit_sum"] += float(frame["kelly_profit_capped"].sum())
            stats["kelly_cap_profit_count"] += int(frame["kelly_profit_capped"].notna().sum())
        if frame["brier"].notna().any():
            stats["brier_sum"] += float(frame["brier"].sum())
            stats["brier_count"] += int(frame["brier"].notna().sum())
        if frame["logloss"].notna().any():
            stats["logloss_sum"] += float(frame["logloss"].sum())
            stats["logloss_count"] += int(frame["logloss"].notna().sum())

    def _bucket_summary(
        group_key: tuple[str, str, str],
        frame: pd.DataFrame,
    ) -> None:
        stats = summary_stats.setdefault(group_key, _init_stats())
        _update(stats, frame)

    def _selection_summary(group_key: tuple[str, str], frame: pd.DataFrame) -> None:
        stats = selection_stats.setdefault(group_key, _init_stats())
        _update(stats, frame)

    def _calibration_update(
        split: str,
        bucket_label: str,
        frame: pd.DataFrame,
    ) -> None:
        key = (split, bucket_label)
        stats = calibration_stats.setdefault(
            key,
            {
                "count": 0,
                "model_prob_sum": 0,
                "hit_sum": 0,
                "hit_count": 0,
            },
        )
        stats["count"] += int(frame.shape[0])
        stats["model_prob_sum"] += float(frame["model_prob"].sum())
        stats["hit_sum"] += float(frame["actual_hit"].sum())
        stats["hit_count"] += int(frame["actual_hit"].notna().sum())

    def _odds_bucket(odds: float | None) -> str:
        if odds is None or pd.isna(odds):
            return "unknown"
        value = float(odds)
        if value <= -200:
            return "<=-200"
        if value <= -150:
            return "-199 to -150"
        if value <= -120:
            return "-149 to -120"
        if value <= -105:
            return "-119 to -105"
        if value < 100:
            return "-104 to +99"
        if value <= 150:
            return "+100 to +150"
        if value <= 200:
            return "+151 to +200"
        return ">=+201"

    def _sample_bucket(sample: float | None) -> str:
        if sample is None or pd.isna(sample):
            return "unknown"
        value = int(sample)
        if value < 15:
            return "<15"
        if value < 25:
            return "15-24"
        if value < 40:
            return "25-39"
        if value < 60:
            return "40-59"
        return "60+"

    def _mean_edge_bucket(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "unknown"
        edge = float(value)
        if edge <= -3:
            return "<=-3"
        if edge <= -1.5:
            return "-3 to -1.5"
        if edge <= -0.5:
            return "-1.5 to -0.5"
        if edge <= 0.5:
            return "-0.5 to 0.5"
        if edge <= 1.5:
            return "0.5 to 1.5"
        if edge <= 3:
            return "1.5 to 3"
        return ">=3"

    def _edge_bucket(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "unknown"
        edge = float(value)
        if edge <= -0.05:
            return "<=-5%"
        if edge <= -0.02:
            return "-5% to -2%"
        if edge <= 0:
            return "-2% to 0%"
        if edge <= 0.02:
            return "0% to 2%"
        if edge <= 0.05:
            return "2% to 5%"
        return ">=5%"

    bin_edges = np.linspace(0, 1, 11)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]

    calibration_mode = args.calibration
    analysis_dist = getattr(args, "analysis_distribution", "stored")
    if analysis_dist != "stored":
        try:
            header = pd.read_csv(props_path, nrows=1)
            required = {"line", "direction", "sample_mean", "sample_var", "log_mean", "log_std"}
            if not required.issubset(set(header.columns)):
                print("[warn] missing sample summary columns; falling back to stored model_prob.")
                analysis_dist = "stored"
        except Exception:
            print("[warn] failed to inspect props file; falling back to stored model_prob.")
            analysis_dist = "stored"
    platt_params = None
    if calibration_mode == "platt":
        probs_list = []
        hits_list = []
        for chunk in pd.read_csv(
            props_path,
            usecols=[
                "split",
                "model_prob",
                "actual_hit",
                "line",
                "direction",
                "log_mean",
                "log_std",
                "sample_mean",
                "sample_var",
            ],
            chunksize=200000,
        ):
            subset = chunk[chunk["split"] == "val"].dropna(subset=["model_prob", "actual_hit"])
            if analysis_dist != "stored":
                chunk["line"] = pd.to_numeric(chunk["line"], errors="coerce")
                chunk["log_mean"] = pd.to_numeric(chunk["log_mean"], errors="coerce")
                chunk["log_std"] = pd.to_numeric(chunk["log_std"], errors="coerce")
                chunk["sample_mean"] = pd.to_numeric(chunk["sample_mean"], errors="coerce")
                chunk["sample_var"] = pd.to_numeric(chunk["sample_var"], errors="coerce")
                probs = _model_prob_from_summary(
                    chunk["line"].to_numpy(),
                    chunk["direction"],
                    analysis_dist,
                    chunk["log_mean"].to_numpy(),
                    chunk["log_std"].to_numpy(),
                    chunk["sample_mean"].to_numpy(),
                    chunk["sample_var"].to_numpy(),
                )
                chunk["model_prob"] = probs
                subset = chunk[chunk["split"] == "val"].dropna(subset=["model_prob", "actual_hit"])
            if subset.empty:
                continue
            probs_list.append(subset["model_prob"].astype(float).to_numpy())
            hits_list.append(subset["actual_hit"].astype(float).to_numpy())
        if probs_list:
            probs = np.concatenate(probs_list)
            hits = np.concatenate(hits_list)
            platt_params = _fit_platt_scaling(probs, hits)
        else:
            calibration_mode = "none"
            print("[warn] calibration disabled: no validation rows found")

    total_rows = None
    start_time = time.time()
    if getattr(args, "progress", True):
        try:
            with props_path.open("r", encoding="utf-8") as handle:
                total_rows = max(0, sum(1 for _ in handle) - 1)
        except Exception:
            total_rows = None

    processed_rows = 0
    for chunk in pd.read_csv(props_path, chunksize=200000):
        processed_rows += int(chunk.shape[0])
        chunk["actual_hit"] = chunk["actual_hit"].astype("boolean")
        chunk["odds"] = pd.to_numeric(chunk["odds"], errors="coerce")
        chunk["model_prob"] = pd.to_numeric(chunk["model_prob"], errors="coerce")
        chunk["implied_prob"] = pd.to_numeric(chunk["implied_prob"], errors="coerce")
        chunk["ev"] = pd.to_numeric(chunk["ev"], errors="coerce")
        chunk["sample_size"] = pd.to_numeric(chunk["sample_size"], errors="coerce")
        chunk["mean_edge"] = pd.to_numeric(chunk["mean_edge"], errors="coerce")
        chunk["notable_bet"] = chunk["notable_bet"].fillna(False).astype(bool)
        chunk["mean_edge_ok"] = chunk["mean_edge_ok"].fillna(False).astype(bool)
        chunk["ev_positive"] = chunk["ev_positive"].fillna(False).astype(bool)
        chunk["line"] = pd.to_numeric(chunk["line"], errors="coerce")
        chunk["direction"] = chunk["direction"].fillna("over")

        if analysis_dist != "stored":
            chunk["log_mean"] = pd.to_numeric(chunk["log_mean"], errors="coerce")
            chunk["log_std"] = pd.to_numeric(chunk["log_std"], errors="coerce")
            chunk["sample_mean"] = pd.to_numeric(chunk["sample_mean"], errors="coerce")
            chunk["sample_var"] = pd.to_numeric(chunk["sample_var"], errors="coerce")
            probs = _model_prob_from_summary(
                chunk["line"].to_numpy(),
                chunk["direction"],
                analysis_dist,
                chunk["log_mean"].to_numpy(),
                chunk["log_std"].to_numpy(),
                chunk["sample_mean"].to_numpy(),
                chunk["sample_var"].to_numpy(),
            )
            chunk["model_prob"] = probs
            if analysis_dist == "lognormal":
                mu = chunk["log_mean"].to_numpy()
                sigma = chunk["log_std"].to_numpy()
                valid = np.isfinite(mu) & np.isfinite(sigma)
                model_mean = np.full(mu.shape, np.nan, dtype=float)
                model_mean[valid] = np.exp(mu[valid] + (sigma[valid] ** 2) / 2) - 1
            else:
                model_mean = chunk["sample_mean"].to_numpy()
            chunk["model_mean"] = model_mean
            chunk["mean_edge"] = chunk["model_mean"] - chunk["line"]
            direction = chunk["direction"].astype(str).str.lower()
            chunk["mean_edge_ok"] = np.where(
                direction == "over",
                chunk["model_mean"] > chunk["line"],
                chunk["model_mean"] < chunk["line"],
            )
            odds = chunk["odds"].to_numpy()
            p = chunk["model_prob"].to_numpy()
            implied = np.where(
                odds > 0,
                100 / (odds + 100),
                np.where(odds < 0, (-odds) / (-odds + 100), np.nan),
            )
            chunk["implied_prob"] = implied
            valid_ev = np.isfinite(odds) & np.isfinite(p)
            ev = np.full(p.shape, np.nan, dtype=float)
            payout = np.where(odds > 0, odds / 100, np.where(odds < 0, 100 / np.abs(odds), np.nan))
            ev[valid_ev] = p[valid_ev] * payout[valid_ev] - (1 - p[valid_ev])
            chunk["ev"] = ev
            chunk["ev_positive"] = chunk["ev"].fillna(np.nan) > 0
            chunk["notable_bet"] = chunk["ev_positive"] & chunk["mean_edge_ok"]
        if getattr(args, "progress", True):
            elapsed = time.time() - start_time
            rate = processed_rows / elapsed if elapsed > 0 else 0
            if total_rows:
                remaining = max(0, total_rows - processed_rows)
                eta = remaining / rate if rate > 0 else None
                eta_text = f" ETA {eta/60:.1f}m" if eta is not None else ""
                print(
                    f"[analyze] {processed_rows}/{total_rows} rows ({processed_rows/total_rows:.1%})"
                    f" {rate:.0f} r/s{eta_text}",
                    end="\r",
                )
            else:
                print(f"[analyze] {processed_rows} rows {rate:.0f} r/s", end="\r")

        def _profit(row: pd.Series) -> float | None:
            if pd.isna(row["odds"]) or row["actual_hit"] is pd.NA:
                return None
            odds = float(row["odds"])
            payout = odds / 100 if odds > 0 else 100 / abs(odds)
            return payout if row["actual_hit"] else -1.0

        chunk["profit"] = chunk.apply(_profit, axis=1)
        chunk["edge"] = chunk["model_prob"] - chunk["implied_prob"]
        chunk["odds_bucket"] = chunk["odds"].apply(_odds_bucket)
        chunk["sample_bucket"] = chunk["sample_size"].apply(_sample_bucket)
        chunk["mean_edge_bucket"] = chunk["mean_edge"].apply(_mean_edge_bucket)
        chunk["edge_bucket"] = chunk["edge"].apply(_edge_bucket)
        chunk["event_date"] = pd.to_datetime(chunk["event_date"], errors="coerce")
        chunk["recency_bucket"] = chunk["event_date"].dt.to_period("M").astype(str)
        chunk["notable_bucket"] = chunk["notable_bet"].apply(lambda value: "notable" if value else "other")

        def _kelly_fraction(row: pd.Series) -> float | None:
            if pd.isna(row["odds"]) or pd.isna(row["model_prob"]):
                return None
            p = float(row["model_prob"])
            if p <= 0 or p >= 1:
                return None
            odds = float(row["odds"])
            b = odds / 100 if odds > 0 else 100 / abs(odds)
            q = 1 - p
            f = (b * p - q) / b
            return max(0.0, f) / 8.0

        chunk["kelly_fraction"] = chunk.apply(_kelly_fraction, axis=1)
        chunk["kelly_profit"] = chunk.apply(
            lambda row: None
            if row["actual_hit"] is pd.NA or row["kelly_fraction"] is None
            else (
                (row["kelly_fraction"] * (row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"])))
                if row["actual_hit"]
                else -row["kelly_fraction"]
            ),
            axis=1,
        )
        cap = max(0.0, float(args.kelly_cap))
        chunk["kelly_fraction_capped"] = chunk["kelly_fraction"].apply(
            lambda value: min(float(value), cap) if value is not None and not pd.isna(value) else None
        )
        chunk["kelly_profit_capped"] = chunk.apply(
            lambda row: None
            if row["actual_hit"] is pd.NA or row["kelly_fraction_capped"] is None
            else (
                (
                    row["kelly_fraction_capped"]
                    * (row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"]))
                )
                if row["actual_hit"]
                else -row["kelly_fraction_capped"]
            ),
            axis=1,
        )

        mask_probs = chunk["model_prob"].notna() & chunk["actual_hit"].notna()
        if mask_probs.any():
            clipped = np.clip(chunk.loc[mask_probs, "model_prob"].astype(float), 1e-6, 1 - 1e-6)
            y = chunk.loc[mask_probs, "actual_hit"].astype(float)
            chunk.loc[mask_probs, "brier"] = (clipped - y) ** 2
            chunk.loc[mask_probs, "logloss"] = -(
                y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
            )
        else:
            chunk["brier"] = np.nan
            chunk["logloss"] = np.nan

        if calibration_mode == "platt" and platt_params is not None:
            a, b = platt_params
            cal_probs = _apply_platt(chunk["model_prob"].fillna(0.5).to_numpy(), a, b)
            chunk["cal_model_prob"] = cal_probs
            chunk["cal_edge"] = chunk["cal_model_prob"] - chunk["implied_prob"]
            chunk["cal_ev"] = chunk.apply(
                lambda row: expected_value(row["cal_model_prob"], float(row["odds"]))
                if pd.notna(row["odds"]) and pd.notna(row["cal_model_prob"])
                else None,
                axis=1,
            )
            chunk["cal_kelly_fraction"] = chunk.apply(
                lambda row: (
                    max(
                        0.0,
                        ((row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"])) * row["cal_model_prob"]
                         - (1 - row["cal_model_prob"]))
                        / (row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"]))
                    )
                    / 8.0
                )
                if pd.notna(row["odds"]) and pd.notna(row["cal_model_prob"])
                else None,
                axis=1,
            )
            chunk["cal_kelly_profit"] = chunk.apply(
                lambda row: None
                if row["actual_hit"] is pd.NA or row["cal_kelly_fraction"] is None
                else (
                    (row["cal_kelly_fraction"] * (row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"])))
                    if row["actual_hit"]
                    else -row["cal_kelly_fraction"]
                ),
                axis=1,
            )
            cap = max(0.0, float(args.kelly_cap))
            chunk["cal_kelly_fraction_capped"] = chunk["cal_kelly_fraction"].apply(
                lambda value: min(float(value), cap) if value is not None and not pd.isna(value) else None
            )
            chunk["cal_kelly_profit_capped"] = chunk.apply(
                lambda row: None
                if row["actual_hit"] is pd.NA or row["cal_kelly_fraction_capped"] is None
                else (
                    (
                        row["cal_kelly_fraction_capped"]
                        * (row["odds"] / 100 if row["odds"] > 0 else 100 / abs(row["odds"]))
                    )
                    if row["actual_hit"]
                    else -row["cal_kelly_fraction_capped"]
                ),
                axis=1,
            )
            cal_mask = chunk["cal_model_prob"].notna() & chunk["actual_hit"].notna()
            if cal_mask.any():
                clipped = np.clip(chunk.loc[cal_mask, "cal_model_prob"].astype(float), 1e-6, 1 - 1e-6)
                y = chunk.loc[cal_mask, "actual_hit"].astype(float)
                chunk.loc[cal_mask, "cal_brier"] = (clipped - y) ** 2
                chunk.loc[cal_mask, "cal_logloss"] = -(
                    y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
                )
            else:
                chunk["cal_brier"] = np.nan
                chunk["cal_logloss"] = np.nan

        splits = chunk["split"].fillna("all").unique().tolist()
        for split in splits:
            subset = chunk[chunk["split"] == split]
            _bucket_summary((split, "overall", "all"), subset)
            for direction in subset["direction"].dropna().unique():
                _bucket_summary((split, "direction", direction), subset[subset["direction"] == direction])
            for stat in subset["stat"].dropna().unique():
                _bucket_summary((split, "stat", stat), subset[subset["stat"] == stat])
            for bucket in subset["odds_bucket"].dropna().unique():
                _bucket_summary((split, "odds_bucket", bucket), subset[subset["odds_bucket"] == bucket])
            for bucket in subset["sample_bucket"].dropna().unique():
                _bucket_summary((split, "sample_bucket", bucket), subset[subset["sample_bucket"] == bucket])
            for bucket in subset["recency_bucket"].dropna().unique():
                _bucket_summary((split, "recency", bucket), subset[subset["recency_bucket"] == bucket])
            for bucket in subset["notable_bucket"].dropna().unique():
                _bucket_summary((split, "notable", bucket), subset[subset["notable_bucket"] == bucket])
            for bucket in subset["mean_edge_bucket"].dropna().unique():
                _bucket_summary((split, "mean_edge_bucket", bucket), subset[subset["mean_edge_bucket"] == bucket])
            for bucket in subset["edge_bucket"].dropna().unique():
                _bucket_summary((split, "edge_bucket", bucket), subset[subset["edge_bucket"] == bucket])

            _selection_summary((split, "all"), subset)
            notable_subset = subset[subset["notable_bet"]]
            _selection_summary((split, "notable"), notable_subset)
            for threshold in edge_thresholds:
                edge_subset = subset[subset["edge"].notna() & (subset["edge"] >= threshold)]
                _selection_summary((split, f"edge>={threshold:.2f}"), edge_subset)
                notable_edge = edge_subset[edge_subset["notable_bet"]]
                _selection_summary((split, f"notable_edge>={threshold:.2f}"), notable_edge)
                combined = edge_subset[
                    edge_subset["mean_edge_ok"]
                    & edge_subset["ev_positive"]
                    & (edge_subset["sample_size"] >= args.min_games)
                ]
                _selection_summary((split, f"combined_edge>={threshold:.2f}"), combined)

            cal_subset = subset[subset["model_prob"].notna() & subset["actual_hit"].notna()].copy()
            if not cal_subset.empty:
                cal_subset["bucket"] = pd.cut(
                    cal_subset["model_prob"].astype(float),
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True,
                )
                for bucket in cal_subset["bucket"].dropna().unique():
                    bucket_df = cal_subset[cal_subset["bucket"] == bucket]
                    _calibration_update(split, str(bucket), bucket_df)

            if calibration_mode == "platt" and platt_params is not None:
                cal_subset = subset.copy()
                cal_subset["edge"] = cal_subset["cal_edge"]
                cal_subset["ev"] = cal_subset["cal_ev"]
                cal_subset["kelly_profit"] = cal_subset["cal_kelly_profit"]
                cal_subset["kelly_profit_capped"] = cal_subset["cal_kelly_profit_capped"]
                cal_subset["brier"] = cal_subset["cal_brier"]
                cal_subset["logloss"] = cal_subset["cal_logloss"]
                cal_subset["model_prob"] = cal_subset["cal_model_prob"]
                cal_subset["edge_bucket"] = cal_subset["edge"].apply(_edge_bucket)

                stats = calibrated_summary_stats.setdefault((split, "overall", "all"), _init_stats())
                _update(stats, cal_subset)
                for stat in cal_subset["stat"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "stat", stat), _init_stats())
                    _update(stats, cal_subset[cal_subset["stat"] == stat])
                for bucket in cal_subset["odds_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "odds_bucket", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["odds_bucket"] == bucket])
                for bucket in cal_subset["sample_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "sample_bucket", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["sample_bucket"] == bucket])
                for bucket in cal_subset["recency_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "recency", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["recency_bucket"] == bucket])
                for bucket in cal_subset["notable_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "notable", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["notable_bucket"] == bucket])
                for bucket in cal_subset["mean_edge_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "mean_edge_bucket", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["mean_edge_bucket"] == bucket])
                for bucket in cal_subset["edge_bucket"].dropna().unique():
                    stats = calibrated_summary_stats.setdefault((split, "edge_bucket", bucket), _init_stats())
                    _update(stats, cal_subset[cal_subset["edge_bucket"] == bucket])

                sel_stats = calibrated_selection_stats.setdefault((split, "all"), _init_stats())
                _update(sel_stats, cal_subset)
                notable_subset = cal_subset[cal_subset["notable_bet"]]
                sel_stats = calibrated_selection_stats.setdefault((split, "notable"), _init_stats())
                _update(sel_stats, notable_subset)
                for threshold in edge_thresholds:
                    edge_subset = cal_subset[cal_subset["edge"].notna() & (cal_subset["edge"] >= threshold)]
                    sel_stats = calibrated_selection_stats.setdefault(
                        (split, f"edge>={threshold:.2f}"), _init_stats()
                    )
                    _update(sel_stats, edge_subset)
                    notable_edge = edge_subset[edge_subset["notable_bet"]]
                    sel_stats = calibrated_selection_stats.setdefault(
                        (split, f"notable_edge>={threshold:.2f}"), _init_stats()
                    )
                    _update(sel_stats, notable_edge)
                    combined = edge_subset[
                        edge_subset["mean_edge_ok"]
                        & edge_subset["ev_positive"]
                        & (edge_subset["sample_size"] >= args.min_games)
                    ]
                    sel_stats = calibrated_selection_stats.setdefault(
                        (split, f"combined_edge>={threshold:.2f}"), _init_stats()
                    )
                    _update(sel_stats, combined)

                if cal_subset["model_prob"].notna().any() and cal_subset["actual_hit"].notna().any():
                    temp = cal_subset.dropna(subset=["model_prob", "actual_hit"]).copy()
                    temp["bucket"] = pd.cut(
                        temp["model_prob"].astype(float),
                        bins=bin_edges,
                        labels=bin_labels,
                        include_lowest=True,
                    )
                    for bucket in temp["bucket"].dropna().unique():
                        bucket_df = temp[temp["bucket"] == bucket]
                        key = (split, str(bucket))
                        stats = calibrated_calibration_stats.setdefault(
                            key,
                            {
                                "count": 0,
                                "model_prob_sum": 0,
                                "hit_sum": 0,
                                "hit_count": 0,
                            },
                        )
                        stats["count"] += int(bucket_df.shape[0])
                        stats["model_prob_sum"] += float(bucket_df["model_prob"].sum())
                        stats["hit_sum"] += float(bucket_df["actual_hit"].sum())
                        stats["hit_count"] += int(bucket_df["actual_hit"].notna().sum())

    if getattr(args, "progress", True):
        print("")

    summary_rows = []
    for (split, slice_type, slice_value), stats in summary_stats.items():
        summary_rows.append(
            {
                "split": split,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "count": int(stats["count"]),
                "notable_count": int(stats["notable_count"]),
                "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
                "avg_model_prob": stats["model_prob_sum"] / stats["model_prob_count"]
                if stats["model_prob_count"]
                else None,
                "avg_implied_prob": stats["implied_prob_sum"] / stats["implied_prob_count"]
                if stats["implied_prob_count"]
                else None,
                "avg_ev": stats["ev_sum"] / stats["ev_count"] if stats["ev_count"] else None,
                "notable_roi": stats["notable_profit_sum"] / stats["notable_profit_count"]
                if stats["notable_profit_count"]
                else None,
                "roi": stats["profit_sum"] / stats["profit_count"] if stats["profit_count"] else None,
                "kelly_roi": stats["kelly_profit_sum"] / stats["kelly_profit_count"]
                if stats["kelly_profit_count"]
                else None,
                "kelly_cap_roi": stats["kelly_cap_profit_sum"] / stats["kelly_cap_profit_count"]
                if stats["kelly_cap_profit_count"]
                else None,
                "brier": stats["brier_sum"] / stats["brier_count"] if stats["brier_count"] else None,
                "log_loss": stats["logloss_sum"] / stats["logloss_count"]
                if stats["logloss_count"]
                else None,
            }
        )

    if summary_rows:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with summary_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote summary to {summary_out}")

    calibration_rows = []
    for (split, bucket), stats in calibration_stats.items():
        calibration_rows.append(
            {
                "split": split,
                "bucket": bucket,
                "count": int(stats["count"]),
                "avg_model_prob": stats["model_prob_sum"] / stats["count"] if stats["count"] else None,
                "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
            }
        )
    if calibration_rows:
        calibration_out = Path(args.calibration_out)
        calibration_out.parent.mkdir(parents=True, exist_ok=True)
        with calibration_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_rows)
        print(f"Wrote calibration to {calibration_out}")

    selection_rows = []
    for (split, selection), stats in selection_stats.items():
        selection_rows.append(
            {
                "split": split,
                "selection": selection,
                "count": int(stats["count"]),
                "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
                "avg_model_prob": stats["model_prob_sum"] / stats["model_prob_count"]
                if stats["model_prob_count"]
                else None,
                "avg_implied_prob": stats["implied_prob_sum"] / stats["implied_prob_count"]
                if stats["implied_prob_count"]
                else None,
                "avg_ev": stats["ev_sum"] / stats["ev_count"] if stats["ev_count"] else None,
                "roi": stats["profit_sum"] / stats["profit_count"] if stats["profit_count"] else None,
                "kelly_roi": stats["kelly_profit_sum"] / stats["kelly_profit_count"]
                if stats["kelly_profit_count"]
                else None,
                "kelly_cap_roi": stats["kelly_cap_profit_sum"] / stats["kelly_cap_profit_count"]
                if stats["kelly_cap_profit_count"]
                else None,
                "brier": stats["brier_sum"] / stats["brier_count"] if stats["brier_count"] else None,
                "log_loss": stats["logloss_sum"] / stats["logloss_count"]
                if stats["logloss_count"]
                else None,
            }
        )
    if selection_rows:
        selection_out = Path(args.selection_out)
        selection_out.parent.mkdir(parents=True, exist_ok=True)
        with selection_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
            writer.writeheader()
            writer.writerows(selection_rows)
        print(f"Wrote selection summary to {selection_out}")

    if calibrated_summary_stats:
        calibrated_rows = []
        for (split, slice_type, slice_value), stats in calibrated_summary_stats.items():
            calibrated_rows.append(
                {
                    "split": split,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "count": int(stats["count"]),
                    "notable_count": int(stats["notable_count"]),
                    "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
                    "avg_model_prob": stats["model_prob_sum"] / stats["model_prob_count"]
                    if stats["model_prob_count"]
                    else None,
                    "avg_implied_prob": stats["implied_prob_sum"] / stats["implied_prob_count"]
                    if stats["implied_prob_count"]
                    else None,
                    "avg_ev": stats["ev_sum"] / stats["ev_count"] if stats["ev_count"] else None,
                    "notable_roi": stats["notable_profit_sum"] / stats["notable_profit_count"]
                    if stats["notable_profit_count"]
                    else None,
                    "roi": stats["profit_sum"] / stats["profit_count"] if stats["profit_count"] else None,
                    "kelly_roi": stats["kelly_profit_sum"] / stats["kelly_profit_count"]
                    if stats["kelly_profit_count"]
                    else None,
                    "kelly_cap_roi": stats["kelly_cap_profit_sum"] / stats["kelly_cap_profit_count"]
                    if stats["kelly_cap_profit_count"]
                    else None,
                    "brier": stats["brier_sum"] / stats["brier_count"] if stats["brier_count"] else None,
                    "log_loss": stats["logloss_sum"] / stats["logloss_count"]
                    if stats["logloss_count"]
                    else None,
                }
            )
        summary_out = Path(args.summary_out)
        calibrated_out = summary_out.with_name(summary_out.stem + "_calibrated.csv")
        with calibrated_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibrated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibrated_rows)
        print(f"Wrote calibrated summary to {calibrated_out}")

    if calibrated_selection_stats:
        calibrated_rows = []
        for (split, selection), stats in calibrated_selection_stats.items():
            calibrated_rows.append(
                {
                    "split": split,
                    "selection": selection,
                    "count": int(stats["count"]),
                    "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
                    "avg_model_prob": stats["model_prob_sum"] / stats["model_prob_count"]
                    if stats["model_prob_count"]
                    else None,
                    "avg_implied_prob": stats["implied_prob_sum"] / stats["implied_prob_count"]
                    if stats["implied_prob_count"]
                    else None,
                    "avg_ev": stats["ev_sum"] / stats["ev_count"] if stats["ev_count"] else None,
                    "roi": stats["profit_sum"] / stats["profit_count"] if stats["profit_count"] else None,
                    "kelly_roi": stats["kelly_profit_sum"] / stats["kelly_profit_count"]
                    if stats["kelly_profit_count"]
                    else None,
                    "kelly_cap_roi": stats["kelly_cap_profit_sum"] / stats["kelly_cap_profit_count"]
                    if stats["kelly_cap_profit_count"]
                    else None,
                    "brier": stats["brier_sum"] / stats["brier_count"] if stats["brier_count"] else None,
                    "log_loss": stats["logloss_sum"] / stats["logloss_count"]
                    if stats["logloss_count"]
                    else None,
                }
            )
        selection_out = Path(args.selection_out)
        calibrated_out = selection_out.with_name(selection_out.stem + "_calibrated.csv")
        with calibrated_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibrated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibrated_rows)
        print(f"Wrote calibrated selection summary to {calibrated_out}")

    if calibrated_calibration_stats:
        calibrated_rows = []
        for (split, bucket), stats in calibrated_calibration_stats.items():
            calibrated_rows.append(
                {
                    "split": split,
                    "bucket": bucket,
                    "count": int(stats["count"]),
                    "avg_model_prob": stats["model_prob_sum"] / stats["count"] if stats["count"] else None,
                    "hit_rate": stats["hit_sum"] / stats["hit_count"] if stats["hit_count"] else None,
                }
            )
        calibration_out = Path(args.calibration_out)
        calibrated_out = calibration_out.with_name(calibration_out.stem + "_calibrated.csv")
        with calibrated_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(calibrated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibrated_rows)
        print(f"Wrote calibrated calibration to {calibrated_out}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest props from historical Odds API snapshots.")
    parser.add_argument("--data-dir", default="data/odds_history", help="Directory with historical odds JSON.")
    parser.add_argument("--stats-source", default="kaggle", choices=("kaggle", "cache"), help="Stats source.")
    parser.add_argument("--kaggle-path", default="PlayerStatistics.csv", help="Path to Kaggle stats CSV.")
    parser.add_argument("--cache-path", default=None, help="Path to normalized cache for stats_source=cache.")
    parser.add_argument("--team-stats-path", default="TeamStatistics.csv", help="Path to TeamStatistics.csv.")
    parser.add_argument("--injuries-path", default=None, help="Optional CSV of injury adjustments.")
    parser.add_argument("--start", default=None, help="Filter start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Filter end date (YYYY-MM-DD).")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone for event cutoff.")
    parser.add_argument("--sportsbook", default=None, help="Optional sportsbook filter (case-insensitive).")
    parser.add_argument("--markets", default=None, help="Optional comma-separated markets to include.")
    parser.add_argument("--window", type=int, default=25, help="Rolling window of games (0 = all).")
    parser.add_argument("--min-games", type=int, default=15, help="Minimum games required.")
    parser.add_argument("--min-minutes", type=float, default=20, help="Minimum minutes filter.")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA weighting (default uses EMA).",
    )
    parser.add_argument(
        "--no-minutes-model",
        action="store_true",
        help="Disable minutes prediction adjustment (default uses minutes model).",
    )
    parser.add_argument(
        "--no-pace-model",
        action="store_true",
        help="Disable opponent pace adjustment (default uses pace model).",
    )
    parser.add_argument(
        "--no-home-advantage",
        action="store_true",
        help="Disable home/away adjustment (default uses home/away context).",
    )
    parser.add_argument("--ema-span", type=int, default=12, help="EMA span.")
    parser.add_argument("--train-end", default=None, help="Train split end date (YYYY-MM-DD).")
    parser.add_argument("--val-end", default=None, help="Validation split end date (YYYY-MM-DD).")
    parser.add_argument("--output-props", default="outputs/props_backtest.csv", help="CSV output for props.")
    parser.add_argument("--summary-out", default="outputs/props_summary.csv", help="CSV summary output.")
    parser.add_argument(
        "--calibration-out",
        default="outputs/props_calibration.csv",
        help="CSV calibration output.",
    )
    parser.add_argument(
        "--selection-out",
        default="outputs/selection_summary.csv",
        help="CSV selection summary output.",
    )
    parser.add_argument(
        "--edge-thresholds",
        default="0,0.02,0.05",
        help="Comma-separated edge thresholds for selection summaries.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.02,
        help="Max fraction of bankroll per bet for capped Kelly sizing.",
    )
    parser.add_argument(
        "--calibration",
        default="none",
        choices=("none", "platt"),
        help="Calibration mode to apply (validation split only).",
    )
    parser.add_argument(
        "--analysis-distribution",
        default="stored",
        choices=("stored", "lognormal", "gamma", "normal"),
        help="Recompute model probabilities from stored sample stats during analysis.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip prop extraction and only analyze an existing props CSV.",
    )
    parser.add_argument(
        "--input-props",
        default=None,
        help="Props CSV to analyze when --analyze-only is set.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress + ETA during backtest and analysis (default: on).",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--output-joints", default=None, help="Optional CSV output for joint probs.")
    parser.add_argument("--joint-mode", default="none", choices=("none", "same-player"), help="Joint mode.")
    parser.add_argument(
        "--joint-direction",
        default="prefer-over",
        choices=("prefer-over", "over", "under"),
        help="Direction selection for joint legs.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Max event files to process.")
    parser.add_argument(
        "--max-log-cache",
        type=int,
        default=200,
        help="Max player log DataFrames to keep in memory (0 disables cache).",
    )
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    use_ema = not args.no_ema
    use_minutes_model = not args.no_minutes_model
    use_pace_model = not args.no_pace_model
    use_home_advantage = not args.no_home_advantage
    edge_thresholds = []
    for raw in args.edge_thresholds.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            edge_thresholds.append(float(raw))
        except ValueError:
            continue
    if not edge_thresholds:
        edge_thresholds = [0.0, 0.02, 0.05]
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date() if args.train_end else None
    val_end = datetime.strptime(args.val_end, "%Y-%m-%d").date() if args.val_end else None

    if args.analyze_only:
        input_props = Path(args.input_props or args.output_props)
        analyze_outputs(input_props, args, edge_thresholds)
        return 0

    data_dir = Path(args.data_dir)
    markets = None
    if args.markets:
        markets = tuple(token.strip() for token in args.markets.split(",") if token.strip())

    pace_lookup = None
    if use_pace_model:
        try:
            pace_lookup = TeamPaceLookup.from_csv(args.team_stats_path)
        except Exception:
            pace_lookup = None
            use_pace_model = False

    injury_adjustments: dict[tuple[datetime.date, str], float] = {}
    injury_team_context: dict[tuple[datetime.date, str], dict[str, float | int | None]] = {}
    if args.injuries_path:
        try:
            injury_adjustments, injury_team_context = _load_injury_context(args.injuries_path)
        except Exception:
            injury_adjustments = {}
            injury_team_context = {}

    log_cache = _LRUCache(args.max_log_cache)

    def load_player_logs(player_name: str, season_end_year: int) -> pd.DataFrame | None:
        key = (normalize_player_name(player_name), season_end_year)
        cached = log_cache.get(key)
        if cached is not None or (args.max_log_cache <= 0 and key in getattr(load_player_logs, "null_cache", set())):
            return cached
        try:
            if args.stats_source == "kaggle":
                logs = fetch_player_game_logs_by_name_kaggle(
                    player_name=player_name,
                    season_end_year=season_end_year,
                    data_path=args.kaggle_path,
                )
            else:
                if not args.cache_path:
                    raise ValueError("missing_cache_path")
                logs = fetch_player_game_logs_from_cache(
                    player_name=player_name,
                    data_path=args.cache_path,
                    season_end_year=season_end_year,
                )
        except Exception:
            if args.max_log_cache <= 0:
                load_player_logs.null_cache.add(key)
            else:
                log_cache.set(key, None)
            return None
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        if pace_lookup is not None:
            logs = pace_lookup.attach_pace(logs)
        log_cache.set(key, logs)
        return logs

    load_player_logs.null_cache = set()

    output_props = Path(args.output_props)
    output_props.parent.mkdir(parents=True, exist_ok=True)
    props_writer = None
    props_handle = output_props.open("w", newline="", encoding="utf-8")
    prop_rows_written = 0
    joint_rows: list[dict[str, Any]] = []

    files = _iter_event_files(data_dir)
    if args.max_files:
        files = files[: args.max_files]

    total_files = len(files)
    start_time = time.time()
    for idx, path in enumerate(files, start=1):
        if args.progress:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_files - idx) / rate if rate > 0 else None
            eta_text = f" ETA {eta/60:.1f}m" if eta is not None else ""
            print(
                f"[build] {idx}/{total_files} files ({idx/total_files:.1%}) {rate:.2f} f/s{eta_text}",
                end="\r",
            )
        event, odds_payload, snapshot = _load_event_payload(path)
        commence = event.get("commence_time") or odds_payload.get("commence_time")
        if not commence:
            continue
        event_dt = _parse_datetime(str(commence)).astimezone(tz)
        event_date = event_dt.date()
        if start_date and event_date < start_date:
            continue
        if end_date and event_date > end_date:
            continue

        season_end_year = _season_end_year(event_dt)
        event_id = event.get("id") or odds_payload.get("id") or path.stem
        home_team = event.get("home_team") or odds_payload.get("home_team")
        away_team = event.get("away_team") or odds_payload.get("away_team")

        odds_events: list[dict[str, Any]]
        if isinstance(odds_payload, dict) and "data" in odds_payload:
            data = odds_payload.get("data")
            odds_events = [data] if isinstance(data, dict) else list(data or [])
        elif isinstance(odds_payload, dict):
            odds_events = [odds_payload]
        else:
            odds_events = list(odds_payload)

        props = extract_player_props(odds_events, markets=markets or DEFAULT_MARKET_MAP.keys())
        sportsbook_filter = args.sportsbook.lower().strip() if args.sportsbook else None

        joint_candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        for prop in props:
            sportsbook = prop.get("sportsbook")
            if sportsbook_filter and sportsbook and sportsbook_filter not in str(sportsbook).lower():
                continue
            market_type = prop.get("market_type")
            stat = DEFAULT_MARKET_MAP.get(str(market_type))
            if not stat:
                continue
            player_name = prop.get("player_name")
            direction = prop.get("direction") or "over"
            line = prop.get("line")
            odds = prop.get("odds")
            if player_name is None or line is None or odds is None:
                continue

            logs = load_player_logs(str(player_name), season_end_year)
            if logs is None:
                continue

            cutoff = event_dt.replace(tzinfo=None)
            filtered_logs = _filter_logs_before(logs.copy(), cutoff, args.min_minutes)
            predicted_minutes = None
            predicted_pace = None
            injury_multiplier = None
            if use_minutes_model:
                predicted_minutes = _predict_minutes(
                    logs,
                    cutoff=cutoff,
                    window=args.window,
                    use_ema=use_ema,
                    ema_span=args.ema_span,
                    min_games=args.min_games,
                )
            if injury_adjustments:
                injury_multiplier = injury_adjustments.get(
                    (event_date, normalize_player_name(str(player_name)))
                )
                if predicted_minutes is not None and injury_multiplier is not None:
                    predicted_minutes = predicted_minutes * injury_multiplier
            if use_pace_model and pace_lookup is not None:
                predicted_pace = pace_lookup.expected_pace(
                    home_team or "",
                    away_team or "",
                    cutoff,
                    args.window,
                )
            player_team, opponent_team = _infer_player_team(logs, event_date, home_team, away_team)
            team_ctx = _lookup_team_context(injury_team_context, event_date, player_team)
            opp_ctx = _lookup_team_context(injury_team_context, event_date, opponent_team)
            home_context = None
            if use_home_advantage and player_team and home_team and away_team:
                player_norm = normalize_team_name(player_team)
                home_norm = normalize_team_name(str(home_team))
                away_norm = normalize_team_name(str(away_team))
                if player_norm == home_norm:
                    home_context = "home"
                elif player_norm == away_norm:
                    home_context = "away"
            if stat not in filtered_logs.columns:
                continue
            samples_cols = [stat, "minutes"]
            if "pace" in filtered_logs.columns:
                samples_cols.append("pace")
            samples_df = filtered_logs[samples_cols].dropna()
            samples_df = samples_df[samples_df["minutes"] > 0]
            samples = samples_df[stat].astype(float)
            if args.window and args.window > 0:
                samples = samples.tail(args.window)
                samples_df = samples_df.tail(args.window)
            sample_size = int(samples.size)
            avg_mean = None
            empirical_prob = None
            if sample_size > 0:
                if str(direction).lower() == "over":
                    empirical_prob = float((samples > float(line)).mean())
                else:
                    empirical_prob = float((samples < float(line)).mean())

            model_prob = None
            model_mean = None
            minutes_adjusted = False
            pace_adjusted = False
            home_adjusted = False
            home_factor = None
            samples_for_fit = samples.to_numpy()
            if sample_size >= max(2, args.min_games):
                try:
                    if use_minutes_model and predicted_minutes is not None:
                        minutes_series = samples_df["minutes"].astype(float)
                        adjusted = samples.to_numpy() / minutes_series.to_numpy() * predicted_minutes
                        samples_for_fit = adjusted
                        minutes_adjusted = True
                    else:
                        samples_for_fit = samples.to_numpy()
                    if use_pace_model and predicted_pace is not None and "pace" in samples_df.columns:
                        pace_series = samples_df["pace"].astype(float)
                        pace_series = pace_series.replace({0: np.nan}).dropna()
                        if pace_series.size:
                            samples_for_fit = samples_for_fit / pace_series.to_numpy() * predicted_pace
                            pace_adjusted = True
                    if use_home_advantage and home_context and "location" in filtered_logs.columns:
                        location_series = filtered_logs.loc[samples_df.index, "location"]
                        samples_for_fit, home_factor, home_adjusted = _apply_home_advantage(
                            samples_for_fit,
                            location_series,
                            home_context,
                        )
                    mu, sigma = _compute_model_stats(
                        samples_for_fit,
                        use_ema=use_ema,
                        ema_span=args.ema_span,
                    )
                    model_prob = _lognormal_hit_probability(mu, sigma, float(line), str(direction))
                    model_mean = float(lognormal_mean(mu, sigma))
                except Exception:
                    model_prob = None
            else:
                if use_minutes_model and predicted_minutes is not None:
                    minutes_series = samples_df["minutes"].astype(float)
                    samples_for_fit = samples.to_numpy() / minutes_series.to_numpy() * predicted_minutes
                    minutes_adjusted = True
                if use_pace_model and predicted_pace is not None and "pace" in samples_df.columns:
                    pace_series = samples_df["pace"].astype(float)
                    pace_series = pace_series.replace({0: np.nan}).dropna()
                    if pace_series.size:
                        samples_for_fit = samples_for_fit / pace_series.to_numpy() * predicted_pace
                        pace_adjusted = True
                if use_home_advantage and home_context and "location" in filtered_logs.columns:
                    location_series = filtered_logs.loc[samples_df.index, "location"]
                    samples_for_fit, home_factor, home_adjusted = _apply_home_advantage(
                        samples_for_fit,
                        location_series,
                        home_context,
                    )

            sample_mean = float(np.mean(samples_for_fit)) if samples_for_fit.size else None
            sample_var = float(np.var(samples_for_fit, ddof=1)) if samples_for_fit.size > 1 else None
            log_mean = None
            log_std = None
            if samples_for_fit.size > 1 and np.all(samples_for_fit > -1):
                log_samples = np.log1p(samples_for_fit)
                log_mean = float(np.mean(log_samples))
                log_std = float(np.std(log_samples, ddof=1))
            if model_mean is not None:
                avg_mean = model_mean

            implied_prob = None
            ev = None
            if odds is not None:
                try:
                    implied_prob = float(american_to_prob(float(odds)))
                except Exception:
                    implied_prob = None
                if model_prob is not None:
                    try:
                        ev = float(expected_value(model_prob, float(odds)))
                    except Exception:
                        ev = None

            mean_edge = None
            mean_edge_ok = None
            if model_mean is not None:
                mean_edge = float(model_mean - float(line))
                if str(direction).lower() == "over":
                    mean_edge_ok = model_mean > float(line)
                else:
                    mean_edge_ok = model_mean < float(line)
            ev_positive = ev is not None and ev > 0
            notable_bet = bool(ev_positive and mean_edge_ok)

            actual_row = _match_game_row(logs, event_date, home_team, away_team)
            actual_value = float(actual_row[stat]) if actual_row is not None and stat in actual_row else None
            actual_hit = None
            if actual_value is not None:
                if str(direction).lower() == "over":
                    actual_hit = actual_value > float(line)
                else:
                    actual_hit = actual_value < float(line)

            if train_end and event_date <= train_end:
                split = "train"
            elif val_end and event_date <= val_end:
                split = "val"
            elif train_end or val_end:
                split = "test"
            else:
                split = "all"

            row = {
                "event_id": event_id,
                "event_date": event_date.isoformat(),
                "snapshot": snapshot,
                "sportsbook": sportsbook,
                "player_name": player_name,
                "market_type": market_type,
                "stat": stat,
                "line": line,
                "direction": direction,
                "odds": odds,
                "implied_prob": implied_prob,
                "model_prob": model_prob,
                "model_mean": model_mean,
                "avg_mean": avg_mean,
                "mean_edge": mean_edge,
                "mean_edge_ok": mean_edge_ok,
                "empirical_prob": empirical_prob,
                "sample_size": sample_size,
                "actual_value": actual_value,
                "actual_hit": actual_hit,
                "ev": ev,
                "ev_positive": ev_positive,
                "notable_bet": notable_bet,
                "predicted_minutes": predicted_minutes,
                "minutes_adjusted": minutes_adjusted,
                "predicted_pace": predicted_pace,
                "pace_adjusted": pace_adjusted,
                "injury_multiplier": injury_multiplier,
                "home_context": home_context,
                "home_factor": home_factor,
                "home_adjusted": home_adjusted,
                "sample_mean": sample_mean,
                "sample_var": sample_var,
                "log_mean": log_mean,
                "log_std": log_std,
                "player_team": player_team,
                "opponent_team": opponent_team,
                "team_injury_count": team_ctx.get("injury_count") if team_ctx else None,
                "team_injury_out_count": team_ctx.get("out_count") if team_ctx else None,
                "team_injury_severity": team_ctx.get("severity_sum") if team_ctx else None,
                "team_injury_avg_multiplier": team_ctx.get("avg_multiplier") if team_ctx else None,
                "opp_injury_count": opp_ctx.get("injury_count") if opp_ctx else None,
                "opp_injury_out_count": opp_ctx.get("out_count") if opp_ctx else None,
                "opp_injury_severity": opp_ctx.get("severity_sum") if opp_ctx else None,
                "opp_injury_avg_multiplier": opp_ctx.get("avg_multiplier") if opp_ctx else None,
                "split": split,
            }
            if props_writer is None:
                props_writer = csv.DictWriter(props_handle, fieldnames=list(row.keys()))
                props_writer.writeheader()
            props_writer.writerow(row)
            prop_rows_written += 1

            if args.joint_mode == "same-player" and market_type in MAIN_MARKETS:
                joint_key = f"{event_id}:{normalize_player_name(str(player_name))}"
                stats_map = joint_candidates[joint_key]
                direction_lower = str(direction).lower()
                if args.joint_direction == "over" and direction_lower != "over":
                    continue
                if args.joint_direction == "under" and direction_lower != "under":
                    continue
                if stat in stats_map:
                    if args.joint_direction == "prefer-over" and direction_lower == "over":
                        stats_map[stat] = {
                            "line": line,
                            "direction": direction_lower,
                            "player_name": player_name,
                        }
                    continue
                stats_map[stat] = {
                    "line": line,
                    "direction": direction_lower,
                    "player_name": player_name,
                }

        if args.joint_mode == "same-player":
            for joint_key, stats_map in joint_candidates.items():
                if len(stats_map) < 2:
                    continue
                player_name = next(iter(stats_map.values()))["player_name"]
                logs = load_player_logs(str(player_name), season_end_year)
                if logs is None:
                    continue
                cutoff = event_dt.replace(tzinfo=None)
                filtered_logs = _filter_logs_before(logs.copy(), cutoff, args.min_minutes)
                available_stats = [stat for stat in ("points", "rebounds", "assists") if stat in stats_map]
                if not available_stats:
                    continue
                cols = available_stats + (["minutes"] if "minutes" in filtered_logs.columns else [])
                samples_df = filtered_logs[cols].dropna()
                if "minutes" in samples_df.columns:
                    samples_df = samples_df[samples_df["minutes"] > 0]
                if args.window and args.window > 0:
                    samples_df = samples_df.tail(args.window)
                if samples_df.shape[0] < max(2, args.min_games):
                    continue

                minutes_pred = predicted_minutes
                if use_minutes_model and minutes_pred is None:
                    minutes_pred = _predict_minutes(
                        logs,
                        cutoff=cutoff,
                        window=args.window,
                        use_ema=use_ema,
                        ema_span=args.ema_span,
                        min_games=args.min_games,
                    )
                if injury_adjustments:
                    injury_multiplier = injury_adjustments.get(
                        (event_date, normalize_player_name(str(player_name)))
                    )
                    if minutes_pred is not None and injury_multiplier is not None:
                        minutes_pred = minutes_pred * injury_multiplier
                if use_minutes_model and minutes_pred is not None and "minutes" in samples_df.columns:
                    for stat_name in available_stats:
                        samples_df[stat_name] = samples_df[stat_name] / samples_df["minutes"] * minutes_pred
                if use_pace_model and predicted_pace is None and pace_lookup is not None:
                    predicted_pace = pace_lookup.expected_pace(
                        home_team or "",
                        away_team or "",
                        cutoff,
                        args.window,
                    )
                if use_pace_model and predicted_pace is not None and "pace" in samples_df.columns:
                    for stat_name in available_stats:
                        samples_df[stat_name] = samples_df[stat_name] / samples_df["pace"] * predicted_pace

                if "minutes" in samples_df.columns:
                    samples_df = samples_df.drop(columns=["minutes"])
                samples = samples_df.to_numpy(dtype=float)
                if use_ema:
                    span = max(2, int(args.ema_span))
                    alpha = 2 / (span + 1)
                    n = samples.shape[0]
                    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
                    weights_sum = weights.sum()
                    if weights_sum <= 0:
                        continue
                    log_samples = np.log1p(samples)
                    mu = (weights[:, None] * log_samples).sum(axis=0) / weights_sum
                    var = (weights[:, None] * (log_samples - mu) ** 2).sum(axis=0) / weights_sum
                    sigma = np.sqrt(np.maximum(var, 1e-12))
                else:
                    mu = []
                    sigma = []
                    for col in range(samples.shape[1]):
                        col_mu, col_sigma = fit_lognormal_params(samples[:, col])
                        mu.append(col_mu)
                        sigma.append(col_sigma)
                    mu = np.array(mu, dtype=float)
                    sigma = np.array(sigma, dtype=float)

                correlation = lognormal_correlations(samples)
                generated = simulate_lognormal_copula(
                    mu=np.array(mu, dtype=float),
                    sigma=np.array(sigma, dtype=float),
                    correlation=correlation,
                    simulations=20000,
                )
                parlay_legs = []
                leg_meta = []
                for idx, stat in enumerate(available_stats):
                    line = float(stats_map[stat]["line"])
                    direction = stats_map[stat]["direction"]
                    parlay_legs.append(ParlayLeg(index=idx, line=line, direction=direction))
                    leg_meta.append(f"{stat}:{direction}:{line}")
                result = evaluate_legs(generated, parlay_legs)
                joint_prob = float(result.joint_probability)

                actual_row = _match_game_row(logs, event_date, home_team, away_team)
                actual_hit = None
                if actual_row is not None:
                    hits = []
                    for stat in available_stats:
                        value = actual_row.get(stat)
                        if value is None:
                            hits = []
                            break
                        line = float(stats_map[stat]["line"])
                        direction = stats_map[stat]["direction"]
                        if direction == "over":
                            hits.append(float(value) > line)
                        else:
                            hits.append(float(value) < line)
                    if hits:
                        actual_hit = all(hits)

                if train_end and event_date <= train_end:
                    split = "train"
                elif val_end and event_date <= val_end:
                    split = "val"
                elif train_end or val_end:
                    split = "test"
                else:
                    split = "all"

                joint_rows.append(
                    {
                        "event_id": event_id,
                        "event_date": event_date.isoformat(),
                        "snapshot": snapshot,
                        "player_name": player_name,
                        "legs": "|".join(leg_meta),
                        "joint_prob": joint_prob,
                        "sample_size": int(samples.shape[0]),
                        "actual_hit": actual_hit,
                        "split": split,
                    }
                )

    if args.progress:
        print("")

    props_handle.close()

    if args.output_joints and joint_rows:
        output_joints = Path(args.output_joints)
        output_joints.parent.mkdir(parents=True, exist_ok=True)
        with output_joints.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(joint_rows[0].keys()))
            writer.writeheader()
            writer.writerows(joint_rows)

    print(f"Wrote {prop_rows_written} prop rows to {output_props}")
    if args.output_joints:
        print(f"Wrote {len(joint_rows)} joint rows to {args.output_joints}")

    if prop_rows_written:
        analyze_outputs(
            output_props,
            args,
            edge_thresholds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

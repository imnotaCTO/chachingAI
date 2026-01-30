from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_team_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(name))
    return " ".join(cleaned.split())


def compute_possessions(row: pd.Series) -> float | None:
    try:
        fga = float(row["fieldGoalsAttempted"])
        fta = float(row["freeThrowsAttempted"])
        orb = float(row["reboundsOffensive"])
        tov = float(row["turnovers"])
    except (KeyError, TypeError, ValueError):
        return None
    return fga - orb + tov + 0.44 * fta


@dataclass
class TeamPaceLookup:
    table: pd.DataFrame
    team_history: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_csv(cls, path: str) -> TeamPaceLookup:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"TeamStatistics.csv not found at {csv_path}")
        usecols = [
            "gameDateTimeEst",
            "teamCity",
            "teamName",
            "opponentTeamCity",
            "opponentTeamName",
            "fieldGoalsAttempted",
            "freeThrowsAttempted",
            "reboundsOffensive",
            "turnovers",
        ]
        df = pd.read_csv(csv_path, usecols=usecols)
        df["date"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce").dt.date
        df["team"] = (
            df["teamCity"].fillna("").astype(str).str.strip()
            + " "
            + df["teamName"].fillna("").astype(str).str.strip()
        ).str.strip()
        df["opponent"] = (
            df["opponentTeamCity"].fillna("").astype(str).str.strip()
            + " "
            + df["opponentTeamName"].fillna("").astype(str).str.strip()
        ).str.strip()
        df["possessions"] = df.apply(compute_possessions, axis=1)
        df = df.dropna(subset=["date", "team", "opponent", "possessions"])
        df["team_norm"] = df["team"].map(normalize_team_name)
        df["opponent_norm"] = df["opponent"].map(normalize_team_name)
        df = df[["date", "team_norm", "opponent_norm", "possessions"]]

        team_history: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for team, group in df.groupby("team_norm"):
            group = group.sort_values("date")
            dates = np.array(group["date"].tolist(), dtype="datetime64[D]")
            poss = group["possessions"].astype(float).to_numpy()
            team_history[team] = (dates, poss)
        return cls(table=df, team_history=team_history)

    def attach_pace(self, logs: pd.DataFrame) -> pd.DataFrame:
        if logs.empty:
            return logs
        logs = logs.copy()
        logs["date"] = pd.to_datetime(logs["date"], errors="coerce").dt.date
        logs["team_norm"] = logs["team"].map(normalize_team_name)
        logs["opponent_norm"] = logs["opponent"].map(normalize_team_name)
        merged = logs.merge(
            self.table,
            how="left",
            left_on=["date", "team_norm", "opponent_norm"],
            right_on=["date", "team_norm", "opponent_norm"],
        )
        merged = merged.rename(columns={"possessions": "pace"})
        return merged

    def get_team_pace_before(self, team: str, cutoff_date: pd.Timestamp, window: int) -> float | None:
        team_norm = normalize_team_name(team)
        if team_norm not in self.team_history:
            return None
        dates, poss = self.team_history[team_norm]
        cutoff = np.datetime64(cutoff_date.date(), "D")
        idx = np.searchsorted(dates, cutoff, side="left")
        if idx <= 0:
            return None
        history = poss[:idx]
        if window and window > 0:
            history = history[-window:]
        if history.size == 0:
            return None
        return float(history.mean())

    def expected_pace(self, team: str, opponent: str, cutoff_date: pd.Timestamp, window: int) -> float | None:
        team_avg = self.get_team_pace_before(team, cutoff_date, window)
        opp_avg = self.get_team_pace_before(opponent, cutoff_date, window)
        if team_avg is None or opp_avg is None:
            return None
        return float((team_avg + opp_avg) / 2)

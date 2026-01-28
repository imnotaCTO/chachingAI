from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParlayLeg:
    index: int
    line: float
    direction: str


@dataclass(frozen=True)
class ParlayResult:
    leg_probabilities: np.ndarray
    joint_probability: float


def simulate_lognormal_copula(
    mu: np.ndarray,
    sigma: np.ndarray,
    correlation: np.ndarray,
    simulations: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate correlated lognormal outcomes using a Gaussian copula."""
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have the same shape")
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square")
    if correlation.shape[0] != mu.shape[0]:
        raise ValueError("correlation dimension must match mu length")
    rng = rng or np.random.default_rng()
    chol = np.linalg.cholesky(correlation)
    z = rng.standard_normal(size=(simulations, mu.shape[0]))
    correlated = z @ chol.T
    return np.expm1(mu + sigma * correlated)


def evaluate_legs(samples: np.ndarray, legs: Iterable[ParlayLeg]) -> ParlayResult:
    """Evaluate leg hit rates and joint probability from simulated samples."""
    legs_list = list(legs)
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    hits = []
    for leg in legs_list:
        outcomes = samples[:, leg.index]
        if leg.direction.lower() == "over":
            hits.append(outcomes > leg.line)
        elif leg.direction.lower() == "under":
            hits.append(outcomes < leg.line)
        else:
            raise ValueError(f"Unsupported direction: {leg.direction}")
    hits_matrix = np.column_stack(hits)
    leg_probabilities = hits_matrix.mean(axis=0)
    joint_probability = float(np.all(hits_matrix, axis=1).mean())
    return ParlayResult(leg_probabilities=leg_probabilities, joint_probability=joint_probability)

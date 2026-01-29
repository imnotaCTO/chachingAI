from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .modeling import fit_lognormal_params, lognormal_correlations
from .odds import expected_value, prob_to_american
from .simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula


@dataclass(frozen=True)
class LegSpec:
    index: int
    line: float
    direction: str
    player: str | None = None
    stat: str | None = None


@dataclass(frozen=True)
class LegResult:
    player: str | None
    stat: str | None
    line: float
    direction: str
    probability: float


@dataclass(frozen=True)
class ParlayRecommendation:
    game: str | None
    legs: tuple[LegResult, ...]
    joint_probability: float
    sportsbook_odds: float
    model_fair_odds: float
    expected_value: float


def price_parlay_from_samples(
    samples: np.ndarray,
    legs: Sequence[LegSpec],
    sportsbook_odds: float,
    *,
    game: str | None = None,
    correlation: np.ndarray | None = None,
    simulations: int = 20000,
    rng: np.random.Generator | None = None,
    ensure_psd: bool = True,
    min_eigenvalue: float = 1e-6,
) -> ParlayRecommendation:
    """Fit lognormal params, simulate a copula, and price a parlay."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    if len(legs) == 0:
        raise ValueError("legs must be non-empty")

    mu = []
    sigma = []
    for col in range(samples.shape[1]):
        col_mu, col_sigma = fit_lognormal_params(samples[:, col])
        mu.append(col_mu)
        sigma.append(col_sigma)

    if correlation is None:
        correlation = lognormal_correlations(samples)

    generated = simulate_lognormal_copula(
        mu=np.array(mu),
        sigma=np.array(sigma),
        correlation=correlation,
        simulations=simulations,
        rng=rng,
        ensure_psd=ensure_psd,
        min_eigenvalue=min_eigenvalue,
    )

    parlay_legs = [ParlayLeg(index=leg.index, line=leg.line, direction=leg.direction) for leg in legs]
    result = evaluate_legs(generated, parlay_legs)

    joint_probability = float(result.joint_probability)
    clipped = min(max(joint_probability, 1e-6), 1 - 1e-6)
    fair_odds = float(prob_to_american(clipped))
    ev = float(expected_value(joint_probability, sportsbook_odds))

    leg_results = tuple(
        LegResult(
            player=leg.player,
            stat=leg.stat,
            line=leg.line,
            direction=leg.direction,
            probability=float(result.leg_probabilities[idx]),
        )
        for idx, leg in enumerate(legs)
    )
    return ParlayRecommendation(
        game=game,
        legs=leg_results,
        joint_probability=joint_probability,
        sportsbook_odds=sportsbook_odds,
        model_fair_odds=fair_odds,
        expected_value=ev,
    )

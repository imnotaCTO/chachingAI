from __future__ import annotations


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        raise ValueError("odds cannot be zero")
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def prob_to_american(probability: float) -> float:
    """Convert probability to American odds."""
    if not 0 < probability < 1:
        raise ValueError("probability must be between 0 and 1")
    if probability >= 0.5:
        return -100 * probability / (1 - probability)
    return 100 * (1 - probability) / probability


def expected_value(probability: float, odds: float) -> float:
    """Compute expected value per $1 stake for American odds."""
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)
    return probability * payout - (1 - probability)

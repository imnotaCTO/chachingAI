from __future__ import annotations

import numpy as np


def fit_lognormal_params(samples: np.ndarray) -> tuple[float, float]:
    """Estimate lognormal parameters using log1p-transformed samples."""
    if samples.size == 0:
        raise ValueError("samples must be non-empty")
    log_samples = np.log1p(samples)
    mu = float(np.mean(log_samples))
    sigma = float(np.std(log_samples, ddof=1))
    return mu, sigma


def lognormal_mean(mu: float, sigma: float) -> float:
    """Approximate the mean of a shifted lognormal distribution."""
    return float(np.exp(mu + sigma**2 / 2) - 1)


def lognormal_correlations(samples: np.ndarray) -> np.ndarray:
    """Compute correlations on log1p-transformed samples.

    Args:
        samples: Array shaped (n_samples, n_stats).
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    if samples.shape[0] < 2:
        raise ValueError("samples must contain at least two rows")
    log_samples = np.log1p(samples)
    return np.corrcoef(log_samples, rowvar=False)

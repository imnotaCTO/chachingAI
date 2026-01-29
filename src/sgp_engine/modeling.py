from __future__ import annotations

import numpy as np


def fit_lognormal_params(samples: np.ndarray) -> tuple[float, float]:
    """Estimate lognormal parameters using log1p-transformed samples."""
    if samples.size == 0:
        raise ValueError("samples must be non-empty")
    if samples.size < 2:
        raise ValueError("samples must contain at least two values")
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


def make_psd_correlation(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """Project a correlation matrix onto the nearest PSD matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    sym = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.maximum(eigenvalues, min_eigenvalue)
    psd = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    diag = np.diag(psd)
    if np.any(diag <= 0):
        raise ValueError("PSD projection produced non-positive diagonal entries")
    inv_sqrt = 1.0 / np.sqrt(diag)
    return psd * inv_sqrt[:, None] * inv_sqrt[None, :]

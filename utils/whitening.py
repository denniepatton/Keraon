from __future__ import annotations

import numpy as np


def regularize_cov(cov: np.ndarray, shrinkage: float, eig_floor: float) -> np.ndarray:
    """Regularize a symmetric covariance matrix.

    Applies these processes:
    1) sanitize: replace NaN / Inf with 0 so downstream eigh never diverges
    2) shrinkage: Σ' = (1-λ)Σ + λ I
    3) eigenvalue floor: λ_i <- max(λ_i, ε)
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")

    shrinkage = float(shrinkage)
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")
    eig_floor = float(eig_floor)
    if eig_floor <= 0.0:
        raise ValueError("eig_floor must be > 0")

    # Sanitize NaN / Inf entries (can arise from degenerate mixture weights)
    if not np.all(np.isfinite(cov)):
        cov = np.where(np.isfinite(cov), cov, 0.0)

    cov = 0.5 * (cov + cov.T)
    d = cov.shape[0]
    cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(d)

    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        # Fallback: return a well-conditioned diagonal matrix so inference can continue
        return eig_floor * np.eye(d)
    vals = np.maximum(vals, eig_floor)
    return (vecs * vals) @ vecs.T


def inv_sqrt_psd(matrix: np.ndarray, shrinkage: float = 0.02, eig_floor: float = 1e-8) -> np.ndarray:
    """Compute Σ^{-1/2} for a symmetric PSD matrix with deterministic regularization."""
    cov = regularize_cov(matrix, shrinkage=shrinkage, eig_floor=eig_floor)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        d = cov.shape[0]
        return (1.0 / np.sqrt(eig_floor)) * np.eye(d)
    vals = np.maximum(vals, eig_floor)
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    return (vecs * inv_sqrt_vals) @ vecs.T


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Compute sample covariance with ddof=1.  Returns identity for n <= 1."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    if n <= 1:
        return np.eye(X.shape[1])
    Xc = X - np.mean(X, axis=0)
    return (Xc.T @ Xc) / float(max(n - 1, 1))

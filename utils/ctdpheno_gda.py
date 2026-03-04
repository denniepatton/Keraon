from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .whitening import inv_sqrt_psd, regularize_cov, sample_covariance


@dataclass
class CtdPhenoGDAModel:
    feature_columns: List[str]
    classes: List[str]
    mean_global: np.ndarray
    W: np.ndarray

    mu_w: Dict[str, np.ndarray]
    Sigma_w: Dict[str, np.ndarray]

    priors: Dict[str, float]

    # Deterministic regularization settings (must be applied identically at inference)
    cov_shrinkage: float = 0.02
    eig_floor: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "classes": self.classes,
            "mean_global": self.mean_global,
            "W": self.W,
            "mu_w": self.mu_w,
            "Sigma_w": self.Sigma_w,
            "priors": self.priors,
            "cov_shrinkage": self.cov_shrinkage,
            "eig_floor": self.eig_floor,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CtdPhenoGDAModel":
        return CtdPhenoGDAModel(
            feature_columns=list(d["feature_columns"]),
            classes=list(d["classes"]),
            mean_global=np.asarray(d["mean_global"], dtype=float),
            W=np.asarray(d["W"], dtype=float),
            mu_w={k: np.asarray(v, dtype=float) for k, v in d["mu_w"].items()},
            Sigma_w={k: np.asarray(v, dtype=float) for k, v in d["Sigma_w"].items()},
            priors={k: float(v) for k, v in d["priors"].items()},
            cov_shrinkage=float(d.get("cov_shrinkage", 0.02)),
            eig_floor=float(d.get("eig_floor", 1e-8)),
        )


def fit_ctdpheno_gda(
    df_train: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    cov_shrinkage: float = 0.02,
    eig_floor: float = 1e-8,
) -> CtdPhenoGDAModel:
    if "Subtype" not in df_train.columns:
        raise ValueError("df_train must contain 'Subtype' column")

    if feature_columns is None:
        feature_columns = [c for c in df_train.columns if c != "Subtype"]

    X = df_train[feature_columns].to_numpy(dtype=float)
    y = df_train["Subtype"].astype(str).to_numpy()

    classes = sorted(pd.unique(y).tolist())
    if "Healthy" not in classes:
        raise ValueError("Reference must include 'Healthy'")

    # Global centering + shrinkage covariance for whitening
    mean_global = np.mean(X, axis=0)
    Xc = X - mean_global
    Sigma_global = sample_covariance(Xc)
    W = inv_sqrt_psd(Sigma_global, shrinkage=cov_shrinkage, eig_floor=eig_floor)

    mu_w: Dict[str, np.ndarray] = {}
    Sigma_w: Dict[str, np.ndarray] = {}
    priors: Dict[str, float] = {}

    n_total = len(y)
    for cls in classes:
        Xk = X[y == cls]
        priors[cls] = float(len(Xk) / max(n_total, 1))

        muk = np.mean(Xk, axis=0)
        mu_w[cls] = W @ (muk - mean_global)

        # Class covariance in original space, shrink, then whiten
        Xk_c = Xk - muk
        Sig_k = sample_covariance(Xk_c)
        Sigma_w[cls] = W @ Sig_k @ W.T

    return CtdPhenoGDAModel(
        feature_columns=feature_columns,
        classes=classes,
        mean_global=mean_global,
        W=W,
        mu_w=mu_w,
        Sigma_w=Sigma_w,
        priors=priors,
        cov_shrinkage=float(cov_shrinkage),
        eig_floor=float(eig_floor),
    )


def _logpdf_mvnorm(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    return float(multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=False))


def predict_ctdpheno_gda(model: CtdPhenoGDAModel, df_test: pd.DataFrame) -> pd.DataFrame:
    if "TFX" not in df_test.columns:
        raise ValueError("df_test must contain 'TFX' column")

    X = df_test[model.feature_columns].to_numpy(dtype=float)
    tfx = df_test["TFX"].to_numpy(dtype=float)

    # Whitened test vectors
    Y = (model.W @ (X - model.mean_global).T).T

    out = pd.DataFrame(index=df_test.index)
    out["TFX"] = tfx

    logp: Dict[str, np.ndarray] = {cls: np.full(len(df_test), np.nan, dtype=float) for cls in model.classes}

    mu_H = model.mu_w["Healthy"]
    Sigma_H = model.Sigma_w["Healthy"]

    for i in range(len(df_test)):
        ti = float(np.clip(tfx[i], 0.0, 1.0))
        yi = Y[i]

        for cls in model.classes:
            mu_S = model.mu_w[cls]
            Sigma_S = model.Sigma_w[cls]

            mu_mix = (1.0 - ti) * mu_H + ti * mu_S
            # Critical: squared weights
            Sigma_mix = (1.0 - ti) ** 2 * Sigma_H + (ti**2) * Sigma_S

            Sigma_mix = regularize_cov(Sigma_mix, shrinkage=model.cov_shrinkage, eig_floor=model.eig_floor)

            try:
                logp[cls][i] = _logpdf_mvnorm(yi, mu_mix, Sigma_mix)
            except (np.linalg.LinAlgError, ValueError):
                logp[cls][i] = np.nan

    # Generic outputs
    for cls in model.classes:
        out[f"logp_{cls}"] = logp[cls]

    # Posterior over classes (log prior + log likelihood), stable softmax
    log_post = np.vstack([
        logp[cls] + np.log(max(model.priors.get(cls, 0.0), 1e-12)) for cls in model.classes
    ]).T
    m = np.nanmax(log_post, axis=1, keepdims=True)
    ex = np.exp(log_post - m)
    denom = np.nansum(ex, axis=1, keepdims=True)
    post = ex / denom

    for j, cls in enumerate(model.classes):
        out[f"post_{cls}"] = post[:, j]

    out["predicted_class"] = [model.classes[int(np.nanargmax(post[i]))] if np.all(np.isfinite(post[i])) else "" for i in range(len(df_test))]

    return out

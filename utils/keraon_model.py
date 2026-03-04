from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .keraon_helpers import compute_orthonormal_basis, compute_offtarget_basis


@dataclass
class KeraonModel:
    feature_columns: List[str]
    subtypes: List[str]  # includes Healthy
    healthy_mean: np.ndarray
    subtype_means: Dict[str, np.ndarray]

    # Subtype span
    V: np.ndarray  # (n_features, n_subtypes_without_healthy)
    Q_V: np.ndarray
    P: np.ndarray
    P_perp: np.ndarray

    # OffTarget basis in orthogonal complement
    U_off: np.ndarray  # (n_features, n_off)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "subtypes": self.subtypes,
            "healthy_mean": self.healthy_mean,
            "subtype_means": self.subtype_means,
            "V": self.V,
            "Q_V": self.Q_V,
            "P": self.P,
            "P_perp": self.P_perp,
            "U_off": self.U_off,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KeraonModel":
        return KeraonModel(
            feature_columns=list(d["feature_columns"]),
            subtypes=list(d["subtypes"]),
            healthy_mean=np.asarray(d["healthy_mean"], dtype=float),
            subtype_means={k: np.asarray(v, dtype=float) for k, v in d["subtype_means"].items()},
            V=np.asarray(d["V"], dtype=float),
            Q_V=np.asarray(d["Q_V"], dtype=float),
            P=np.asarray(d["P"], dtype=float),
            P_perp=np.asarray(d["P_perp"], dtype=float),
            U_off=np.asarray(d["U_off"], dtype=float),
        )


def fit_keraon_model(df_train: pd.DataFrame, n_offtarget: int = 3) -> KeraonModel:
    if "Subtype" not in df_train.columns:
        raise ValueError("df_train must contain 'Subtype'")

    feature_columns = [c for c in df_train.columns if c != "Subtype"]
    subtypes = sorted(df_train["Subtype"].astype(str).unique().tolist())

    if "Healthy" not in subtypes:
        raise ValueError("Reference must include 'Healthy'")

    subtype_means: Dict[str, np.ndarray] = {}
    for st in subtypes:
        subtype_means[st] = df_train.loc[df_train["Subtype"] == st, feature_columns].mean(axis=0).to_numpy(dtype=float)

    healthy_mean = subtype_means["Healthy"]

    disease_subtypes = [st for st in subtypes if st != "Healthy"]
    basis_vectors = []
    for st in disease_subtypes:
        v = subtype_means[st] - healthy_mean
        v = v / (np.linalg.norm(v) + 1e-12)
        basis_vectors.append(v)

    V = np.column_stack(basis_vectors) if basis_vectors else np.zeros((len(feature_columns), 0))
    Q_V, P, P_perp = compute_orthonormal_basis(V) if V.shape[1] > 0 else (np.zeros((len(feature_columns), 0)), np.zeros((len(feature_columns), len(feature_columns))), np.eye(len(feature_columns)))

    # Reference residual vectors centered from healthy
    ref_vectors = []
    for idx in df_train.index:
        x = df_train.loc[idx, feature_columns].to_numpy(dtype=float) - healthy_mean
        ref_vectors.append(x)

    U_off = compute_offtarget_basis(ref_vectors, V, n_components=n_offtarget)

    return KeraonModel(
        feature_columns=feature_columns,
        subtypes=subtypes,
        healthy_mean=healthy_mean,
        subtype_means=subtype_means,
        V=V,
        Q_V=Q_V,
        P=P,
        P_perp=P_perp,
        U_off=U_off,
    )


def predict_keraon(model: KeraonModel, df_test: pd.DataFrame) -> pd.DataFrame:
    if "TFX" not in df_test.columns:
        raise ValueError("df_test must contain 'TFX'")

    disease_subtypes = [st for st in model.subtypes if st != "Healthy"]
    n_off = int(model.U_off.shape[1])

    out = pd.DataFrame(index=df_test.index)
    out["TFX"] = df_test["TFX"].astype(float)

    # Fractions and burdens
    for st in disease_subtypes:
        out[f"{st}_fraction"] = np.nan
        out[f"{st}_burden"] = np.nan
    out["Healthy_fraction"] = np.nan

    for i in range(n_off):
        out[f"RA{i}_coeff"] = np.nan
        out[f"RA{i}_energy"] = np.nan
        out[f"RA{i}_burden"] = np.nan
        out[f"RA{i}_fraction"] = np.nan

    # QC/diagnostics
    out["energy_subspace"] = np.nan
    out["energy_offtarget"] = np.nan
    out["energy_residual_perp"] = np.nan
    out["residual_perp_fraction"] = np.nan
    out["subtype_cone_misfit"] = np.nan

    # Region label kept for backward-ish compatibility
    out["FS_Region"] = ""

    V = model.V
    P = model.P
    P_perp = model.P_perp
    U = model.U_off

    for sample in df_test.index:
        tfx = float(df_test.loc[sample, "TFX"])
        x_raw = df_test.loc[sample, model.feature_columns].to_numpy(dtype=float)
        x = x_raw - model.healthy_mean

        x_para = P @ x
        x_perp = P_perp @ x

        # OffTarget signed coefficients from orthogonal projection
        if n_off > 0:
            o_signed = U.T @ x_perp
            x_off = U @ o_signed
            residual_perp = x_perp - x_off
        else:
            o_signed = np.zeros((0,), dtype=float)
            x_off = np.zeros_like(x_perp)
            residual_perp = x_perp

        energy_subspace = float(np.dot(x_para, x_para))
        energy_off = float(np.dot(x_off, x_off))
        energy_resid = float(np.dot(residual_perp, residual_perp))

        modeled = energy_subspace + energy_off
        total = modeled + energy_resid

        out.loc[sample, "energy_subspace"] = energy_subspace
        out.loc[sample, "energy_offtarget"] = energy_off
        out.loc[sample, "energy_residual_perp"] = energy_resid
        out.loc[sample, "residual_perp_fraction"] = (energy_resid / total) if total > 0 else 0.0

        if modeled > 0:
            total_subtype_burden = energy_subspace / modeled
            offtarget_burden = energy_off / modeled
        else:
            total_subtype_burden = 0.0
            offtarget_burden = 0.0

        # Interpretive subtype partition by NNLS in subtype span
        if V.shape[1] > 0 and energy_subspace > 0:
            c_nnls, _ = nnls(V, x_para)
            x_fit = V @ c_nnls
            misfit = float(np.dot(x_para - x_fit, x_para - x_fit) / max(energy_subspace, 1e-12))
        else:
            c_nnls = np.zeros((V.shape[1],), dtype=float)
            misfit = np.nan

        out.loc[sample, "subtype_cone_misfit"] = misfit

        # Distribute subtype burden using Gram-weighted weights
        if V.shape[1] > 0 and np.sum(c_nnls) > 0:
            G = V.T @ V
            Gc = G @ c_nnls
            w = np.maximum(c_nnls * Gc, 0.0)
            if float(np.sum(w)) > 1e-12:
                subtype_burdens = (w / np.sum(w)) * total_subtype_burden
            else:
                subtype_burdens = (c_nnls / np.sum(c_nnls)) * total_subtype_burden
        else:
            subtype_burdens = np.zeros((len(disease_subtypes),), dtype=float)

        # Map burdens to subtype names (V columns aligned to disease_subtypes order)
        for j, st in enumerate(disease_subtypes):
            out.loc[sample, f"{st}_burden"] = float(subtype_burdens[j]) if j < len(subtype_burdens) else 0.0

        # RA columns (energy-based burden, signed coeff)
        for i in range(n_off):
            coeff = float(o_signed[i])
            energy_i = float(coeff * coeff)
            burden_i = (energy_i / modeled) if modeled > 0 else 0.0
            out.loc[sample, f"RA{i}_coeff"] = coeff
            out.loc[sample, f"RA{i}_energy"] = energy_i
            out.loc[sample, f"RA{i}_burden"] = burden_i

        # Normalize burdens to sum to 1.0.
        # Guards against NNLS returning zero coefficients while energy_subspace > 0,
        # which would otherwise drop total_subtype_burden and leave fractions < 1.
        if modeled > 0:
            burden_sum = (
                sum(float(out.loc[sample, f"{st}_burden"]) for st in disease_subtypes)
                + sum(float(out.loc[sample, f"RA{i}_burden"]) for i in range(n_off))
            )
            if burden_sum > 1e-12 and abs(burden_sum - 1.0) > 1e-8:
                scale = 1.0 / burden_sum
                for st in disease_subtypes:
                    out.loc[sample, f"{st}_burden"] = float(out.loc[sample, f"{st}_burden"]) * scale
                for i in range(n_off):
                    out.loc[sample, f"RA{i}_burden"] = float(out.loc[sample, f"RA{i}_burden"]) * scale

        # Fractions across all components (Healthy fixed by external TFX)
        out.loc[sample, "Healthy_fraction"] = 1.0 - tfx
        for st in disease_subtypes:
            out.loc[sample, f"{st}_fraction"] = float(tfx) * float(out.loc[sample, f"{st}_burden"])
        for i in range(n_off):
            out.loc[sample, f"RA{i}_fraction"] = float(tfx) * float(out.loc[sample, f"RA{i}_burden"])

        # FS_Region purely diagnostic
        c_sum = float(np.sum(c_nnls))
        out.loc[sample, "FS_Region"] = "Simplex" if c_sum <= 1.0 else "Non-Simplex"

    return out

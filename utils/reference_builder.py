from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.ctdpheno_gda import CtdPhenoGDAModel, fit_ctdpheno_gda, predict_ctdpheno_gda
from utils.keraon_model import KeraonModel, fit_keraon_model, predict_keraon
from utils.keraon_helpers import stability_select_svm_hyperparams
from utils.reference_model import ReferenceModel, save_reference_model


DEFAULT_PARAM_GRID: Dict[str, List[float]] = {
    "LAMBDA_L0": [0.0, 0.1],
    "SCATTER_POW": [1.0, 3.0],
    "MARGIN_ALPHA": [0.0, 0.5],
    "EDGE_BETA": [0.0, 1.0],
    "GAMMA_REDUNDANCY": [0.0, 0.5],
}


def build_reference_model(
    df_train_full: pd.DataFrame,
    scaling_params: Dict[str, Any],
    features: Optional[List[str]] = None,
    run_stability_selection: bool = True,
    stability_param_grid: Optional[Dict[str, List[float]]] = None,
    stability_n_boot: int = 100,
    stability_subsample: float = 0.8,
    stability_freq_threshold: float = 0.10,
    seed: int = 23,
    verbose: bool = True,
) -> ReferenceModel:
    if "Subtype" not in df_train_full.columns:
        raise ValueError("df_train_full must contain 'Subtype'")

    if features is not None and len(features) == 0:
        raise ValueError("features list was provided but empty")

    feature_selection_meta: Dict[str, Any] = {}

    if features is not None:
        selected_features = list(features)
        feature_selection_meta.update({
            "method": "preselected",
            "features": selected_features,
        })
        freq_table = None
        best_params = None
    else:
        if not run_stability_selection:
            raise ValueError("No features provided and stability selection disabled")

        if stability_param_grid is None:
            stability_param_grid = DEFAULT_PARAM_GRID

        best_params, freq_table, selected_features = stability_select_svm_hyperparams(
            ref_df=df_train_full,
            param_grid=stability_param_grid,
            n_boot=stability_n_boot,
            subsample=stability_subsample,
            seed=seed,
            freq_threshold=stability_freq_threshold,
            min_features=max(1, len({s for s in df_train_full["Subtype"].astype(str).unique() if s != "Healthy"})),
            verbose=verbose,
        )

        feature_selection_meta.update({
            "method": "stability_selection_svm",
            "features": selected_features,
            "frozen_hyperparams": best_params,
            "n_boot": stability_n_boot,
            "subsample": stability_subsample,
            "freq_threshold": stability_freq_threshold,
        })
        if freq_table is not None:
            feature_selection_meta["selection_frequencies"] = freq_table

    df_train = df_train_full[["Subtype"] + selected_features].copy()

    # Fit models
    ctd_model = fit_ctdpheno_gda(df_train)
    k_model = fit_keraon_model(df_train, n_offtarget=3)

    return ReferenceModel(
        df_train=df_train,
        scaling_params=scaling_params,
        feature_selection=feature_selection_meta,
        ctdpheno_gda=ctd_model.to_dict(),
        keraon=k_model.to_dict(),
        calibration=None,
    )


def run_inference(model: ReferenceModel, df_test_features: pd.DataFrame, test_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Align feature columns
    feature_cols = [c for c in model.df_train.columns if c != "Subtype"]
    X = df_test_features.reindex(columns=feature_cols)

    # Features absent from the test FM get NaN from reindex; individual samples
    # may also have NaN for features that other samples possess.  After standardization
    # 0.0 is the global mean, which is the most neutral imputation.  Warn so the
    # user knows which features were missing.
    all_nan_cols = X.columns[X.isna().all()]
    partial_nan_cols = X.columns[X.isna().any() & ~X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Warning: {len(all_nan_cols)} model feature(s) absent from ALL test samples; "
              f"imputing with 0 (global mean): {list(all_nan_cols[:5])}"
              + (" ..." if len(all_nan_cols) > 5 else ""))
    if len(partial_nan_cols) > 0:
        print(f"Warning: {len(partial_nan_cols)} feature(s) have partial NaN across test samples; "
              f"imputing with 0: {list(partial_nan_cols[:5])}"
              + (" ..." if len(partial_nan_cols) > 5 else ""))
    if X.isna().any().any():
        X = X.fillna(0.0)

    df_test = pd.concat([test_labels[["TFX"]], X], axis=1)

    ctd = predict_ctdpheno_gda(CtdPhenoGDAModel.from_dict(model.ctdpheno_gda), df_test)
    ker = predict_keraon(KeraonModel.from_dict(model.keraon), df_test)

    # Provide qc metrics on ctd output too
    qc_cols = [
        "residual_perp_fraction",
        "subtype_cone_misfit",
        "energy_subspace",
        "energy_offtarget",
        "energy_residual_perp",
    ]
    ctd = ctd.join(ker[qc_cols], how="left")

    return ctd, ker


def save_model_artifact(model: ReferenceModel, path: str | Path) -> None:
    save_reference_model(model, path)

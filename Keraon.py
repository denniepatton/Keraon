#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v3.0, 3/13/2025

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.calibration import bootstrap_youden_threshold, truth_contains, youden_threshold
from utils.calibration_plots import plot_roc_pdf, plot_score_hist
from utils.keraon_model import KeraonModel
from utils.keraon_plotters import plot_combined_feature_distributions, plot_ctdpheno, plot_keraon, plot_pca
from utils.keraon_utils import load_palette, load_reference_key, load_test_labels, load_triton_fm
from utils.reference_builder import build_reference_model, run_inference
from utils.reference_model import load_reference_model, save_reference_model, write_json, write_tsv

# These Triton features are liable to depth bias / outliers or are intended for larger regions (e.g. gene bodies, not TFBS sites)
drop_features = ['mean-region-depth'] # highly biased by (or are measurements of) depth
limit_features = ['central-depth', 'window-depth']  # most robust and interpretable, used for testing and naive baseline
# limit_features = None

# feature-specific scaling (monotone transforms) to make normal-like (applies to Triton features only)
# Use safe log transform that handles negative values and NaNs gracefully
def safe_log1p(x: float) -> float:
    """Safe log1p transform: log(1 + max(x, 0)), preserving NaN"""
    if pd.isna(x):
        return np.nan
    return np.log1p(max(x, 0.0))

scaling_methods = {
    'central-entropy': safe_log1p,
    'central-gini-simpson': safe_log1p,
    'pn-mean-amplitude': safe_log1p,
    'pn-mean-spacing': safe_log1p,
}


def main():
    parser = argparse.ArgumentParser(description="Keraon: ReferenceModel build / inference / calibration")

    parser.add_argument(
        "-r",
        "--reference_data",
        nargs="*",
        required=True,
        help="Either a single `reference_model.pickle` (preferred) or one or more tidy reference .tsv files.",
    )
    parser.add_argument(
        "-i",
        "--input_data",
        required=False,
        help="Tidy-form test feature matrix .tsv with columns sample/site/feature/value.",
    )
    parser.add_argument(
        "-t",
        "--tfx",
        required=False,
        help="TSV with sample and TFX; optional third column Truth (used for calibration).",
    )
    parser.add_argument(
        "-k",
        "--reference_key",
        default=None,
        help="Reference key TSV (required when building model from reference .tsv files).",
    )
    parser.add_argument(
        "-f",
        "--features",
        default=None,
        help="Optional feature list file (one site_feature per line) to bypass stability selection.",
    )
    parser.add_argument(
        "-p",
        "--palette",
        default=None,
        help="Optional palette TSV mapping subtypes to hex colors.",
    )

    parser.add_argument(
        "--build_reference_model",
        action="store_true",
        default=False,
        help="Build and save `reference_model.pickle` from reference .tsv + key.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Calibration mode: requires Truth labels; computes Youden thresholds.",
    )
    parser.add_argument(
        "--positive_label",
        default=None,
        help="When calibrating, treat Truth containing this label as positive class.",
    )
    parser.add_argument(
        "--model_out",
        default="results/feature_analysis/reference_model.pickle",
        help="Output path for built model (default: results/feature_analysis/reference_model.pickle)",
    )

    args = parser.parse_args()

    print("\n### Keraon ###")

    # Determine mode requirements
    needs_test_inputs = (args.input_data is not None) or (args.tfx is not None) or args.calibrate
    if needs_test_inputs:
        if args.input_data is None or args.tfx is None:
            raise SystemExit("Inference/calibration requires both --input_data and --tfx")

    def infer_unambiguous_positive_label(truth_series: pd.Series) -> Optional[str]:
        """Infer a single positive label from Truth values.

        Ignores sentinel/unlabeled tokens (e.g., Unknown/NA) and drops Healthy.
        Returns the label if exactly one remains, else None.
        """
        ignore = {"healthy", "unknown", "na", "nan", "none", ""}
        labels: set[str] = set()
        for v in truth_series.astype(str).tolist():
            parts = [p.strip() for p in v.split(",") if p.strip()]
            for p in parts:
                if p.strip().lower() in ignore:
                    continue
                labels.add(p)
        return next(iter(labels)) if len(labels) == 1 else None

    # Directories:
    os.makedirs("results", exist_ok=True)
    processing_dir = Path("results/feature_analysis")
    ctd_dir = Path("results/ctdPheno_class-predictions")
    ker_dir = Path("results/keraon_mixture-predictions")
    calib_dir = Path("results/calibration")

    # Always allow feature-analysis outputs (reference build outputs + optional inference analysis plots)
    processing_dir.mkdir(parents=True, exist_ok=True)
    if needs_test_inputs:
        ctd_dir.mkdir(parents=True, exist_ok=True)
        ker_dir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        calib_dir.mkdir(parents=True, exist_ok=True)

    # Load test labels (TFX + optional Truth)
    test_labels = None
    truth_vals = None
    if args.tfx is not None:
        test_labels, truth_vals = load_test_labels(args.tfx)
        if args.calibrate and truth_vals is None:
            raise SystemExit("--calibrate requires Truth labels in the --tfx file (third column)")

    # Palette (optional)
    ref_labels = None
    if args.reference_key is not None and not args.reference_data[0].endswith(".pickle"):
        ref_labels = load_reference_key(args.reference_key)
    palette = load_palette(args.palette, ref_labels) if ref_labels is not None else load_palette(args.palette)

    # Load/build reference model
    model_path = Path(args.model_out)
    if args.reference_data[0].endswith(".pickle"):
        model = load_reference_model(args.reference_data[0])

        if not needs_test_inputs:
            print("Loaded reference model (no --input_data/--tfx provided); nothing else to do.")
            return
    else:
        if not args.build_reference_model:
            raise SystemExit(
                "Reference data provided as .tsv files. Use --build_reference_model (and --reference_key) to build a model artifact first."
            )
        if args.reference_key is None:
            raise SystemExit("Building a reference model from .tsv requires --reference_key")

        print("Loading and scaling reference feature matrix...")
        ref_df, scaling_params = load_triton_fm(
            args.reference_data,
            scaling_methods,
            str(processing_dir),
            palette,
            ref_labels=ref_labels,
            plot_distributions=True,
            limit_features=limit_features,
        )

        # Optional feature dropping (depth-bias)
        if drop_features:
            drop_regex = "|".join([feature_name for feature_name in drop_features])
            cols_to_drop = ref_df.columns[ref_df.columns.str.contains(drop_regex, regex=True)]
            if len(cols_to_drop) > 0:
                ref_df = ref_df.drop(columns=cols_to_drop)

        df_train_full = pd.merge(ref_labels[["Subtype"]], ref_df, left_index=True, right_index=True)

        # Optional pre-selected features
        features = None
        if args.features is not None:
            with open(args.features) as f:
                features = [line.strip() for line in f if line.strip()]

        model = build_reference_model(
            df_train_full=df_train_full,
            scaling_params=scaling_params,
            features=features,
            run_stability_selection=(features is None),
            verbose=True,
        )

        # Persist feature-selection outputs
        if model.feature_selection.get("method") == "stability_selection_svm":
            freq = model.feature_selection.get("selection_frequencies")
            if isinstance(freq, pd.DataFrame):
                write_tsv(processing_dir / "stability_selection.tsv", freq.set_index("feature"))
            write_json(processing_dir / "svm_hyperparams_frozen.json", model.feature_selection.get("frozen_hyperparams", {}))

        save_reference_model(model, model_path)
        print(f"Saved reference model: {model_path}")

        # If user only wanted to build the model, stop here.
        if not needs_test_inputs:
            print("Build complete (no --input_data/--tfx provided).")
            return

    # Save factor loadings for Keraon
    if model.keraon:
        km = KeraonModel.from_dict(model.keraon)
        V = km.V
        U = km.U_off
        disease_subtypes = [st for st in km.subtypes if st != "Healthy"]
        colnames = disease_subtypes + [f"RA{i}" for i in range(U.shape[1])]
        M = np.column_stack([V, U]) if U.shape[1] > 0 else V
        pd.DataFrame(M, index=km.feature_columns, columns=colnames).to_csv(ker_dir / "factor_loadings.tsv", sep="\t")

    # Load and scale test feature matrix using model scaling params
    print("Loading and scaling test feature matrix...")
    test_df_raw, _ = load_triton_fm(
        args.input_data,
        scaling_methods,
        str(processing_dir),
        palette,
        feature_scaling_params=model.scaling_params,
        plot_distributions=True,
    )

    # For plotting, align test features to the model basis
    feature_cols = [c for c in model.df_train.columns if c != "Subtype"]
    X_aligned = test_df_raw.reindex(columns=feature_cols)
    df_test_for_plots = pd.concat([test_labels[["TFX"]], X_aligned], axis=1)
    if truth_vals is not None:
        df_test_for_plots = pd.concat([truth_vals[["Truth"]], df_test_for_plots], axis=1)

    # Inference
    ctd_preds, ker_preds = run_inference(model, test_df_raw, test_labels)

    # Add Truth if present
    if truth_vals is not None:
        ctd_preds = truth_vals.join(ctd_preds, how="left")
        ker_preds = truth_vals.join(ker_preds, how="left")

    # Final-basis PCA + combined feature distributions (optional but useful)
    plot_pca(model.df_train, str(processing_dir) + "/", palette, "PCA_final-basis_wTestSamples", post_df=df_test_for_plots)
    plot_combined_feature_distributions(
        model.df_train,
        df_test_for_plots,
        str(processing_dir / "feature_distributions" / "final-basis_site-features"),
        palette,
    )

    # Inference writes raw model outputs.

    # Save inference outputs
    write_tsv(ctd_dir / "ctdPheno_class-predictions.tsv", ctd_preds)
    write_tsv(ker_dir / "Keraon_mixture-predictions.tsv", ker_preds)

    # Prediction plots (write into their respective prediction folders)
    # Use calibrated thresholds if present; otherwise plot without accuracy annotation.
    plot_positive_label = None
    if args.positive_label is not None:
        plot_positive_label = str(args.positive_label)
    elif truth_vals is not None:
        plot_positive_label = infer_unambiguous_positive_label(truth_vals["Truth"])
    elif model.calibration is not None:
        plot_positive_label = str(model.calibration.get("positive_label"))

    ctd_plot_key = None
    ker_plot_key = None
    ctd_plot_thr = None
    ker_plot_thr = None
    if plot_positive_label:
        ctd_plot_key = f"post_{plot_positive_label}"
        ker_plot_key = f"{plot_positive_label}_fraction"

        if model.calibration is not None:
            try:
                ctd_cal = model.calibration.get("thresholds", {}).get("ctdpheno", {})
                ker_cal = model.calibration.get("thresholds", {}).get("keraon", {})
                if str(ctd_cal.get("score")) == ctd_plot_key:
                    ctd_plot_thr = float(ctd_cal.get("threshold"))
                if str(ker_cal.get("score")) == ker_plot_key:
                    ker_plot_thr = float(ker_cal.get("threshold"))
            except Exception:
                pass

    # Fallback: if we still don't have a label to focus on, pick the first non-Healthy subtype
    # so that the per-sample PDFs are still produced (without accuracy annotation).
    if ctd_plot_key is None or ker_plot_key is None:
        disease = [s for s in model.df_train["Subtype"].astype(str).unique() if s != "Healthy"]
        if disease:
            fallback_label = str(sorted(disease)[0])
            ctd_plot_key = ctd_plot_key or f"post_{fallback_label}"
            ker_plot_key = ker_plot_key or f"{fallback_label}_fraction"

    if ctd_plot_key is not None and ctd_plot_key in ctd_preds.columns:
        plot_ctdpheno(ctd_preds, str(ctd_dir) + "/", ctd_plot_key, ctd_plot_thr)
    if ker_plot_key is not None and ker_plot_key in ker_preds.columns:
        plot_keraon(ker_preds, str(ker_dir) + "/", ker_plot_key, ker_plot_thr, palette=palette)

    # Optional inference ROC plots if truth was provided
    if truth_vals is not None and not args.calibrate:
        positive_label = None
        if args.positive_label is not None:
            positive_label = str(args.positive_label)
        else:
            positive_label = infer_unambiguous_positive_label(truth_vals["Truth"])

        if positive_label is not None:
            y_true = truth_vals["Truth"].apply(lambda x: 1 if truth_contains(x, positive_label) else 0).to_numpy(dtype=int)
            ctd_score_col = f"post_{positive_label}"
            ker_score_col = f"{positive_label}_fraction"
            if ctd_score_col in ctd_preds.columns and ker_score_col in ker_preds.columns:
                ctd_scores = ctd_preds[ctd_score_col].to_numpy(dtype=float)
                ker_scores = ker_preds[ker_score_col].to_numpy(dtype=float)

                plotdir = processing_dir / "inference_plots"
                plot_roc_pdf(y_true, ctd_scores, f"ROC: ctdPheno-GDA ({ctd_score_col})", plotdir / "ROC_ctdPheno.pdf")
                plot_roc_pdf(y_true, ker_scores, f"ROC: Keraon ({ker_score_col})", plotdir / "ROC_Keraon.pdf")
                plot_score_hist(
                    ctd_scores,
                    y_true,
                    "Score distribution: ctdPheno-GDA",
                    ctd_score_col,
                    plotdir / "scores_ctdPheno.pdf",
                    neg_label=f"not-{positive_label}",
                    pos_label=positive_label,
                )
                plot_score_hist(
                    ker_scores,
                    y_true,
                    "Score distribution: Keraon",
                    ker_score_col,
                    plotdir / "scores_Keraon.pdf",
                    neg_label=f"not-{positive_label}",
                    pos_label=positive_label,
                )
            else:
                print("Truth provided but required score columns missing; skipping inference ROC plots.")
        else:
            print("Truth provided but ambiguous; pass --positive_label to enable inference ROC plots.")

    # Calibration mode
    if args.calibrate:
        print("Running calibration...")
        default_positive_label = "NEPC"
        if args.positive_label is not None:
            positive_label = str(args.positive_label)
        else:
            positive_label = infer_unambiguous_positive_label(truth_vals["Truth"])
            if positive_label is None:
                # For mixed Truth (e.g. multiple subtypes and/or MIX,*), use a sensible default
                # so calibration remains one-command for the common NEPC use-case.
                ctd_default_col = f"post_{default_positive_label}"
                ker_default_col = f"{default_positive_label}_fraction"
                if ctd_default_col in ctd_preds.columns and ker_default_col in ker_preds.columns:
                    positive_label = default_positive_label
                    print(
                        f"Truth provided but ambiguous; defaulting --positive_label to '{positive_label}'. "
                        f"(Override with --positive_label if you want a different calibration target.)"
                    )
                else:
                    raise SystemExit(
                        "--calibrate requires --positive_label unless Truth is unambiguous (or NEPC columns are present for defaulting)"
                    )

        y_true = truth_vals["Truth"].apply(lambda x: 1 if truth_contains(x, positive_label) else 0).to_numpy(dtype=int)

        ctd_score_col = f"post_{positive_label}"
        ker_score_col = f"{positive_label}_fraction"
        if ctd_score_col not in ctd_preds.columns:
            raise SystemExit(f"ctdPheno predictions missing required score column: {ctd_score_col}")
        if ker_score_col not in ker_preds.columns:
            raise SystemExit(f"Keraon predictions missing required score column: {ker_score_col}")

        ctd_scores = ctd_preds[ctd_score_col].to_numpy(dtype=float)
        ker_scores = ker_preds[ker_score_col].to_numpy(dtype=float)

        thr1, meta1 = bootstrap_youden_threshold(y_true, ctd_scores)
        thr2, meta2 = bootstrap_youden_threshold(y_true, ker_scores)

        # Also record the single-pass Youden point on the full sample for reference.
        _, meta1_full = youden_threshold(y_true, ctd_scores)
        _, meta2_full = youden_threshold(y_true, ker_scores)

        model.calibration = {
            "positive_label": positive_label,
            "thresholds": {
                "ctdpheno": {"score": ctd_score_col, "threshold": thr1},
                "keraon": {"score": ker_score_col, "threshold": thr2},
            },
            "youden": {"ctdpheno": meta1_full, "keraon": meta2_full},
            "youden_bootstrap": {"ctdpheno": meta1, "keraon": meta2},
        }

        # Build a merged table for calibration outputs without double-joining Truth/TFX.
        # At this point, ctd_preds/ker_preds may already include Truth from earlier logic.
        ctd_for_merge = ctd_preds.copy()
        ker_for_merge = ker_preds.copy()

        overlap_ctd = set(ctd_for_merge.columns).intersection(set(truth_vals.columns))
        if overlap_ctd:
            ctd_for_merge = ctd_for_merge.drop(columns=sorted(overlap_ctd))

        # Drop columns that overlap with truth_vals OR are already in ctd_for_merge
        # to prevent duplicate TFX / TFX_keraon columns
        overlap_ker = set(ker_for_merge.columns).intersection(
            set(truth_vals.columns) | set(ctd_for_merge.columns)
        )
        if overlap_ker:
            ker_for_merge = ker_for_merge.drop(columns=sorted(overlap_ker))

        merged = truth_vals.join(ctd_for_merge, how="left").join(ker_for_merge, how="left", rsuffix="_keraon")
        write_tsv(calib_dir / "calibration_predictions.tsv", merged)
        write_json(calib_dir / "calibration_thresholds.json", model.calibration)

        # Simple report table (one row)
        report = pd.DataFrame(
            {
                "positive_label": [positive_label],
                "ctdpheno_score": [ctd_score_col],
                "ctdpheno_threshold": [thr1],
                "keraon_score": [ker_score_col],
                "keraon_threshold": [thr2],
                "youden_ctdpheno_J": [meta1.get("youden_J_median")],
                "youden_keraon_J": [meta2.get("youden_J_median")],
                "n_samples": [len(merged)],
                "n_pos": [int(np.sum(y_true == 1))],
                "n_neg": [int(np.sum(y_true == 0))],
            },
            index=["calibration"],
        )
        write_tsv(calib_dir / "calibration_report.tsv", report)
        write_json(calib_dir / "calibration_report.json", report.iloc[0].dropna().to_dict())

        # Plots
        plotdir = calib_dir / "calibration_plots"
        plot_roc_pdf(y_true, ctd_scores, f"ROC: ctdPheno-GDA ({ctd_score_col})", plotdir / "ROC_ctdPheno.pdf")
        plot_roc_pdf(y_true, ker_scores, f"ROC: Keraon ({ker_score_col})", plotdir / "ROC_Keraon.pdf")
        plot_score_hist(
            ctd_scores,
            y_true,
            "Score distribution: ctdPheno-GDA",
            ctd_score_col,
            plotdir / "scores_ctdPheno.pdf",
            neg_label=f"not-{positive_label}",
            pos_label=positive_label,
        )
        plot_score_hist(
            ker_scores,
            y_true,
            "Score distribution: Keraon",
            ker_score_col,
            plotdir / "scores_Keraon.pdf",
            neg_label=f"not-{positive_label}",
            pos_label=positive_label,
        )

        # Also generate the per-sample prediction PDFs in the prediction folders using calibrated thresholds.
        plot_ctdpheno(ctd_preds, str(ctd_dir) + "/", ctd_score_col, thr1)
        plot_keraon(ker_preds, str(ker_dir) + "/", ker_score_col, thr2, palette=palette)

        calibrated_path = processing_dir / "reference_model.calibrated.pickle"
        save_reference_model(model, calibrated_path)
        print(f"Saved calibrated model: {calibrated_path}")

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 11/08/2023

"""
This module contains plotting functions for final predictions from Keraon.
"""

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn.decomposition import PCA
import warnings # ignore warnings, e.g. matplotlib deprecation warnings

def plot_pca(
    df: pd.DataFrame,
    direct: str,
    palette: Mapping[str, str] | None,
    name: str,
    post_df: pd.DataFrame | None = None,
) -> None:
    """
    Plot first three (1-2, 2-3) principal components.

    Parameters:
       df (pd.DataFrame): The input dataframe.
       direct (str): The output directory.
       palette (dict): A dictionary mapping categorical names to colors.
       name (str): The name of the plot.
    """
    # Ignore UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    # Separate features from labels
    df_data = df.drop('Subtype', axis=1)
    
    # Check if there are enough features for PCA
    if df_data.shape[1] < 3:  
        print(f'PCA requires at least three features - exiting ({name})')
        return
    df_data_np = df_data.to_numpy()

    # Track which columns are NaN-free so we use the same mask for transform
    valid_col_mask = ~np.isnan(df_data_np).any(axis=0)
    df_data_clean = df_data_np[:, valid_col_mask]

    if df_data_clean.shape[1] < 3:
        print(f'PCA requires at least three non-NaN features - exiting ({name})')
        return
    
    # Perform PCA
    n_components = min(df_data_clean.shape[1], 3)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df_data_clean)
    
    # Create a new dataframe with the PCA results
    pca_df = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
    pca_df['Subtype'] = df['Subtype']
    
    # Calculate the explained variance for each principal component
    explained_var = pca.explained_variance_ratio_
    
    # Create the plot
    _, axs = plt.subplots(min(2, n_components), 1, figsize=(10, 20))

    sns.scatterplot(x="PC1", y="PC2", data=pca_df, palette=palette, ax=axs[0], hue='Subtype', legend=True, s=200, alpha=0.8)
    axs[0].set_title(f'{name} ({df_data_clean.shape[1]} total features)', size=14, y=1.02)
    axs[0].set_xlabel(f'PC1: {round(100 * explained_var[0], 2)}%')
    axs[0].set_ylabel(f'PC2: {round(100 * explained_var[1], 2)}%')

    if n_components > 2:
        sns.scatterplot(x="PC2", y="PC3", data=pca_df, palette=palette, ax=axs[1], hue='Subtype', legend=True, s=200, alpha=0.8)
        axs[1].set_xlabel(f'PC2: {round(100 * explained_var[1], 2)}%')
        axs[1].set_ylabel(f'PC3: {round(100 * explained_var[2], 2)}%')

    if post_df is not None:
        post_data = post_df.drop(columns=[col for col in ['TFX', 'Truth'] if col in post_df.columns]).to_numpy()
        # Use the same column mask as the PCA fit and impute any remaining NaNs with 0
        post_data_clean = post_data[:, valid_col_mask]
        post_data_clean = np.where(np.isfinite(post_data_clean), post_data_clean, 0.0)
        sampleComponents = pca.transform(post_data_clean)
        sample_df = pd.DataFrame(data=sampleComponents, columns=[f'PC{i+1}' for i in range(sampleComponents.shape[1])], index=post_df.index)
        sample_df['TFX'] = post_df['TFX']
        if 'Truth' in post_df.columns:
            sample_df['Truth'] = post_df['Truth']
            sns.scatterplot(x="PC1", y="PC2", data=sample_df, palette=palette, ax=axs[0], legend=True, alpha=0.8, hue="Truth", size="TFX")
            if n_components > 2:
                sns.scatterplot(x="PC2", y="PC3", data=sample_df, palette=palette, ax=axs[1], legend=True, alpha=0.8, hue="Truth", size="TFX")
        else:
            sns.scatterplot(x="PC1", y="PC2", data=sample_df, palette=palette, ax=axs[0], legend=True, alpha=0.8, color='black', size="TFX")
            if n_components > 2:
                sns.scatterplot(x="PC2", y="PC3", data=sample_df, palette=palette, ax=axs[1], legend=True, alpha=0.8, color='black', size="TFX")

    axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if n_components > 2:
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'{direct}{name}.pdf', bbox_inches="tight")


def plot_ctdpheno(
    predictions: pd.DataFrame,
    direct: str,
    key: str,
    threshold: float | None,
    plot_range: Sequence[float] = (0.0, 1.0),
) -> None:
    """
    Plot stick and ball plot of prediction (relative log likelihood) for each subtype.
    
    Parameters:
       predictions (pd.DataFrame): The input dataframe with predictions and Truth column.
       direct (str): The output directory.
       key (str): The name of the subtype/label to base thresholding on.
       threshold (float): The threshold to use for binary classification.
       plot_range (list): The range of values to plot. Default is [0, 1].
    """
    # Calculate accuracy based on Truth column (optional)
    def calculate_accuracy(group_df: pd.DataFrame, truth_label: str, score_col: str, threshold: float | None) -> str:
        if threshold is None or "Truth" not in group_df.columns:
            return "NA"
        if all(str(truth).lower() == "unknown" for truth in group_df["Truth"]):
            return "NA"
        
        # Check if truth_label is in Truth values (allowing for CSV format)
        contains_key = []
        for truth in group_df['Truth']:
            # Split by comma and check if key is in any of the parts
            truth_parts = [part.strip() for part in str(truth).split(',')]
            contains_key.append(truth_label in truth_parts)
        
        # Calculate accuracy
        correct_predictions = 0
        for idx, contains in enumerate(contains_key):
            score = group_df.iloc[idx][score_col]
            if contains and score >= threshold:  # True positive
                correct_predictions += 1
            elif not contains and score < threshold:  # True negative
                correct_predictions += 1
                
        return f"{100 * correct_predictions / len(group_df):.2f}%" if len(group_df) > 0 else "NA"
    
    # Determine label name for Truth matching from score column name.
    truth_label = str(key)
    if truth_label.startswith("post_"):
        truth_label = truth_label[len("post_") :]

    # Group by Truth and order by group size (descending)
    if 'Truth' in predictions.columns:
        truth_groups = predictions.groupby('Truth')
        group_sizes = truth_groups.size().sort_values(ascending=False)
        ordered_truths = group_sizes.index.tolist()
        width_ratios = [group_sizes.loc[t] for t in ordered_truths]
    else:
        # If no Truth column, treat all as one group
        ordered_truths = ["All"]
        width_ratios = [len(predictions)]
        predictions['Truth'] = "All"
        
    # Choose fixed bar width per sample and calculate total width
    bar_width = 0.3
    total_width = sum(width_ratios) * bar_width + 2  # Add margin
    
    # Create figure with subplots arranged by Truth values
    n_groups = len(ordered_truths)
    fig, axs = plt.subplots(1, n_groups, figsize=(total_width, 6), 
                           gridspec_kw={'width_ratios': width_ratios})
    if n_groups == 1:
        axs = [axs]
    
    # Create custom color gradient from gray to red with mix at threshold
    cmap = LinearSegmentedColormap.from_list("GrayToRed", ["dimgray", "darkred"])
    threshold_for_plot = float(threshold) if threshold is not None else 0.5
    norm = TwoSlopeNorm(vmin=float(plot_range[0]), vcenter=threshold_for_plot, vmax=float(plot_range[1]))
    
    # Process each Truth group
    for idx, (ax, truth_val) in enumerate(zip(axs, ordered_truths)):
        # Get data for this group
        group_df = predictions[predictions['Truth'] == truth_val]
        
        # Calculate accuracy for this group
        acc_text = calculate_accuracy(group_df, truth_label, key, threshold)
        
        # Create x positions for bars
        x_positions = np.arange(len(group_df))
        y_values = group_df[key].values
        
        # Make bars start at threshold and go up/down
        bar_heights = y_values - threshold_for_plot
        colors = [cmap(norm(val)) for val in y_values]
        
        # Plot bars
        ax.bar(x_positions, bar_heights, width=0.2, bottom=threshold_for_plot, color=colors)
        
        # Plot scatter points at the values - larger and without outlines
        ax.scatter(x_positions, y_values, s=150, color=colors, zorder=3, edgecolor=None)
        
        # Add horizontal threshold line
        ax.axhline(y=threshold_for_plot, color='black', linestyle='--', lw=1)
        
        # Set y-axis limits
        ax.set_ylim(plot_range)
        
        # X-axis labels and rotation
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_df.index, rotation=90)
        
        # Only the leftmost subplot gets the y-axis label
        if idx == 0:
            ax.set_ylabel(f'{key} score')
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        
        # Set title with Truth value and accuracy
        title = f"Truth: {truth_val}\nAccuracy: {acc_text}" if idx == 0 else f"{truth_val}\n{acc_text}"
        ax.set_title(title, fontsize=14)
    
    # Instead, create a new axes for the colorbar in a fixed position outside the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # First apply tight_layout to properly position the subplots
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  

    # Get the position of the first subplot to match colorbar height
    pos = axs[0].get_position()
    bottom = pos.y0
    height = pos.height

    # Create the colorbar with matching vertical alignment
    cbar_ax = fig.add_axes([0.96, bottom, 0.02, height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([float(plot_range[0]), threshold_for_plot, float(plot_range[1])])
    cbar.set_ticklabels([f"{float(plot_range[0]):.2f}", f"{threshold_for_plot:.2f}", f"{float(plot_range[1]):.2f}"])

    # Add a figure-level title
    fig.suptitle(f"{key} Classification Scores", fontsize=16, y=1.02)

    # Don't call tight_layout again after positioning the colorbar
    plt.savefig(os.path.join(direct, 'ctdPheno_class-predictions.pdf'), bbox_inches="tight")
    plt.close()


def plot_keraon(
    predictions: pd.DataFrame,
    direct: str,
    key: str,
    threshold: float | None,
    palette: Mapping[str, str] | None = None,
) -> None:
    """
    Plot stacked bar plots in 2 rows with N columns (where N = number of test subtype groups).
    
    Top row (2/3 height): Total fraction bar plots (scaled 0-1, includes Healthy component).
    Bottom row (1/3 height): Tumor burden bar plots (scaled 0-1, non-healthy components only).
    
    Each subplot's width is proportional to its number of samples. Samples are sorted by
    TFX (high to low) and aligned vertically between rows. The leftmost subplot title shows 
    "Truth:" and "Accuracy:" along with computed values, while remaining subplots show the 
    truth value and accuracy.
    
    A figure-level title is added that reads, for example:
       "Mixture estimates (positive label based on <score> thresholding)"
    """

    def truth_contains(truth: str, label: str) -> bool:
        parts = [p.strip() for p in str(truth).split(',') if p.strip()]
        return any(p == label for p in parts)

    def _infer_positive_label_from_key(score_key: str) -> str:
        k = str(score_key)
        for suffix in ('_fraction', '_burden'):
            if k.endswith(suffix):
                return k[: -len(suffix)]
        return k
    # Group by Truth and order by group size (descending)
    # If Truth is missing (common in inference), treat all samples as one group.
    if "Truth" in predictions.columns:
        truth_groups = predictions.groupby('Truth')
        group_sizes_series = truth_groups.size().sort_values(ascending=False)
        ordered_truths = group_sizes_series.index.tolist()
    else:
        predictions = predictions.copy()
        predictions["Truth"] = "All"
        truth_groups = predictions.groupby('Truth')
        group_sizes_series = truth_groups.size().sort_values(ascending=False)
        ordered_truths = group_sizes_series.index.tolist()
    
    # Determine each group's sample count and width ratios for the subplots
    width_ratios = [group_sizes_series.loc[t] for t in ordered_truths]
    
    # Choose a fixed bar width (in inches) per sample.
    bar_width = 0.3
    total_width = sum(width_ratios) * bar_width + 1  # add some extra margin
    
    n_groups = len(ordered_truths)
    # Create subplots in 2 rows with variable widths according to width_ratios.
    # Top row (fraction): 3 units, Bottom row (burden): 1 unit (1/3 of top row)
    fig, axs = plt.subplots(2, n_groups, figsize=(total_width, 5), 
                            gridspec_kw={'width_ratios': width_ratios, 'height_ratios': [3, 1]})
    
    # Handle case where n_groups == 1 (axs would be 1D instead of 2D)
    if n_groups == 1:
        axs = axs.reshape(2, 1)
    
    for idx, truth_val in enumerate(ordered_truths):
        # Subset group and order samples by TFX descending
        group_df = predictions[predictions['Truth'] == truth_val].sort_values(by="TFX", ascending=False)
        n_samples = group_df.shape[0]
        
        # Compute accuracy if Truth is present and we have a threshold
        if threshold is not None and truth_val.lower() != "unknown":
            positive_label = _infer_positive_label_from_key(key)
            is_pos = group_df["Truth"].apply(lambda x: truth_contains(x, positive_label))
            correct = np.where(
                is_pos.to_numpy(dtype=bool),
                (group_df[key] > threshold),
                (group_df[key] <= threshold),
            )
            acc = correct.mean() * 100
            acc_text = f"{acc:.2f}%"
        else:
            acc_text = "NA"
        
        # === TOP ROW: Total Fraction (includes Healthy) ===
        ax_top = axs[0, idx]
        
        # Prepare the data for the stacked bar plot (total fractions)
        subset_fraction = group_df.filter(like='_fraction')
        # Exclude QC diagnostic columns that are not cfDNA fraction components
        subset_fraction = subset_fraction.drop(
            columns=[c for c in subset_fraction.columns if 'residual_perp' in c],
            errors='ignore',
        )
        # If the key column is missing, try key + '_fraction'
        if key not in subset_fraction.columns:
            potential_key = f"{key}_fraction"
            if potential_key in subset_fraction.columns:
                key = potential_key
        
        cols = subset_fraction.columns.tolist()
        if 'Healthy_fraction' in cols:
            cols.remove('Healthy_fraction')
        if key in cols:
            cols.remove(key)
        new_cols = [key] + cols
        if 'Healthy_fraction' in group_df.columns:
            new_cols = new_cols + ['Healthy_fraction']
        subset_fraction = subset_fraction[new_cols]
        
        # Plot the stacked bar plot for total fractions
        if palette is not None:
            pal = {f"{k}_fraction": v for k, v in palette.items()}
            pal["Healthy_fraction"] = "#D3D3D3"

            cycle_colors = plt.rcParams.get("axes.prop_cycle", None)
            cycle_colors = (cycle_colors.by_key().get("color", []) if cycle_colors is not None else [])
            cycle_idx = 0
            colors = []
            for col in subset_fraction.columns:
                if col in pal:
                    colors.append(pal[col])
                elif "residual" in str(col).lower():
                    colors.append("#7F7F7F")
                elif cycle_colors:
                    colors.append(cycle_colors[cycle_idx % len(cycle_colors)])
                    cycle_idx += 1
                else:
                    colors.append("#1F77B4")

            subset_fraction.plot(kind='bar', stacked=True, color=colors, width=0.90, legend=False, ax=ax_top)
        else:
            subset_fraction.plot(kind='bar', stacked=True, width=0.90, legend=False, ax=ax_top)
        
        # Only the leftmost subplot gets the y-axis label
        if idx == 0:
            ax_top.set_ylabel('cfDNA fraction', fontsize=12)
        else:
            ax_top.set_ylabel('')
            ax_top.tick_params(labelleft=False)
        if threshold is not None:
            ax_top.axhline(y=threshold, color='black', linestyle='dashed', linewidth=1)
        ax_top.set_ylim(0, 1)
        ax_top.set_xlim(-0.5, n_samples - 0.5)
        
        # Remove x-axis labels and ticks from top row (will share with bottom row)
        ax_top.set_xlabel('')
        ax_top.tick_params(labelbottom=False)
        
        # Only the leftmost gets full labeling
        if idx == 0:
            ax_top.set_title(f"Truth: {truth_val}\nAccuracy: {acc_text}", fontsize=14)
        else:
            ax_top.set_title(f"{truth_val}\n{acc_text}", fontsize=14)
        
        # === BOTTOM ROW: Tumor Burden (non-healthy components, normalized to sum to 1) ===
        ax_bottom = axs[1, idx]
        
        # Prepare the data for tumor burden (exclude Healthy_fraction, use _burden columns)
        subset_burden = group_df.filter(like='_burden')
        
        # Order columns to match the top row (excluding Healthy)
        burden_cols = subset_burden.columns.tolist()
        # Reorder to put key first if it exists
        key_burden = key.replace('_fraction', '_burden')
        if key_burden in burden_cols:
            burden_cols.remove(key_burden)
            burden_cols = [key_burden] + burden_cols
        subset_burden = subset_burden[burden_cols]
        
        # Plot the stacked bar plot for tumor burden
        if palette is not None:
            pal_burden = {f"{k}_burden": v for k, v in palette.items()}

            cycle_colors = plt.rcParams.get("axes.prop_cycle", None)
            cycle_colors = (cycle_colors.by_key().get("color", []) if cycle_colors is not None else [])
            cycle_idx = 0
            colors = []
            for col in subset_burden.columns:
                if col in pal_burden:
                    colors.append(pal_burden[col])
                elif "residual" in str(col).lower():
                    colors.append("#7F7F7F")
                elif cycle_colors:
                    colors.append(cycle_colors[cycle_idx % len(cycle_colors)])
                    cycle_idx += 1
                else:
                    colors.append("#1F77B4")

            subset_burden.plot(kind='bar', stacked=True, color=colors, width=0.90, legend=False, ax=ax_bottom)
        else:
            subset_burden.plot(kind='bar', stacked=True, width=0.90, legend=False, ax=ax_bottom)
        
        # Only the leftmost subplot gets the y-axis label
        if idx == 0:
            ax_bottom.set_ylabel('tumor burden', fontsize=12)
        else:
            ax_bottom.set_ylabel('')
            ax_bottom.tick_params(labelleft=False)
        ax_bottom.set_ylim(0, 1)
        ax_bottom.set_xlim(-0.5, n_samples - 0.5)
        
        # Set x-axis label only for bottom row
        ax_bottom.set_xlabel('')
        
        # Rotate x-axis labels
        ax_bottom.tick_params(axis='x', rotation=90)

    # Get the last subplot from top row to add the legend
    last_ax = axs[0, -1]
    # Create a legend for the last subplot and place it outside, higher and to the right
    handles, labels = last_ax.get_legend_handles_labels()
    # Reorder to match the stacking order (bottom to top in the bars)
    handles = handles[::-1]
    labels = labels[::-1]
    # Remove "_fraction" suffix from labels
    labels = [label.replace('_fraction', '') for label in labels]
    # Place legend outside the rightmost subplot, higher up
    last_ax.legend(handles, labels, title="Subtypes", 
                bbox_to_anchor=(1.15, 1.0), 
                loc='upper left', 
                borderaxespad=0)
    
    # Add a figure-level title.
    # If key ends with '_fraction', remove that for the presence description.
    key_display = key.replace('_fraction', '')
    fig.suptitle(
        f"Mixture estimates (positive label based on {key_display} thresholding)",
        fontsize=16,
        y=0.98,
    )
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(os.path.join(direct, 'Keraon_mixture-predictions.pdf'), bbox_inches="tight")
    plt.close()


def plot_combined_feature_distributions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_directory: str,
    palette: Mapping[str, str],
) -> None:
    """
    Plots and saves combined feature distributions for training and test datasets.

    For each common feature, a PDF is generated showing:
    - Distributions from df_train, separated by 'Subtype' and colored by palette.
    - A single distribution for all samples in df_test, colored for 'Patient'
      (or purple if 'Patient' is not in the palette).

    Args:
        df_train (pd.DataFrame): Training data with a 'Subtype' column and feature columns.
        df_test (pd.DataFrame): Test data with feature columns (e.g., including 'TFX' but
                                 not 'Subtype' for this plotting purpose).
        output_directory (str): Directory where the PDF plots will be saved.
        palette (dict): A dictionary mapping subtype names (and 'Patient') to colors.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Features to plot are columns in df_train excluding 'Subtype'
    feature_columns = [col for col in df_train.columns if col != 'Subtype']
    
    patient_color = palette.get("Patient", "purple") # Default to purple

    print(f"\nGenerating combined feature distribution plots in: {output_directory}")
    for feature_name in feature_columns:
        if feature_name not in df_test.columns:
            print(f"  Warning: Feature '{feature_name}' from training data not found in test data. Skipping this plot.")
            continue

        plt.figure(figsize=(12, 7))
        
        # Plot distributions for df_train by Subtype
        subtypes = df_train['Subtype'].unique()
        for subtype in subtypes:
            subtype_data = df_train[df_train['Subtype'] == subtype][feature_name].dropna()
            if subtype_data.empty:
                print(f"  Note: No data for subtype '{subtype}' for feature '{feature_name}' in training set.")
                continue
            color = palette.get(subtype, "gray") # Default to gray if subtype not in palette
            sns.kdeplot(subtype_data, label=f"Train - {subtype}", color=color, fill=True, alpha=0.4, linewidth=1.5)

        # Plot distribution for all df_test samples
        test_feature_data = df_test[feature_name].dropna()
        if not test_feature_data.empty:
            sns.kdeplot(test_feature_data, label="Test Samples", color=patient_color, fill=True, alpha=0.6, linewidth=2, linestyle='--')
        else:
            print(f"  Note: No data for feature '{feature_name}' in test set.")

        plt.title(f"Distribution of {feature_name}", fontsize=16)
        plt.xlabel(feature_name, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(title="Dataset - Subtype", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        # Sanitize feature_name for filename
        safe_feature_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in feature_name)
        output_pdf_path = os.path.join(output_directory, f"{safe_feature_name}_distribution.pdf")
        
        try:
            plt.savefig(output_pdf_path)
            # print(f"  Saved: {output_pdf_path}")
        except Exception as e:
            print(f"  Error saving plot for feature {feature_name} to {output_pdf_path}: {e}")
        plt.close()
        
    print("Finished generating combined feature distribution plots.")

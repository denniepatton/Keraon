#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 11/08/2023

"""
This module contains plotting functions for final predictions from Keraon.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
import warnings # ignore warnings, e.g. matplotlib deprecation warnings

def plot_pca(df: pd.DataFrame, direct: str, palette: dict, name: str, post_df=None) -> None:
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
    df_data = df_data.to_numpy()
    
    # Perform PCA
    n_components = min(df_data.shape[1], 3)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df_data[:, ~np.isnan(df_data).any(axis=0)])
    
    # Create a new dataframe with the PCA results
    pca_df = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
    pca_df['Subtype'] = df['Subtype']
    
    # Calculate the explained variance for each principal component
    explained_var = pca.explained_variance_ratio_
    
    # Create the plot
    _, axs = plt.subplots(min(2, n_components), 1, figsize=(10, 20))

    sns.scatterplot(x="PC1", y="PC2", data=pca_df, palette=palette, ax=axs[0], hue='Subtype', legend=True, s=200, alpha=0.8)
    axs[0].set_title(f'{name} ({df_data.shape[1]} total features)', size=14, y=1.02)
    axs[0].set_xlabel(f'PC1: {round(100 * explained_var[0], 2)}%')
    axs[0].set_ylabel(f'PC2: {round(100 * explained_var[1], 2)}%')

    if n_components > 2:
        sns.scatterplot(x="PC2", y="PC3", data=pca_df, palette=palette, ax=axs[1], hue='Subtype', legend=True, s=200, alpha=0.8)
        axs[1].set_xlabel(f'PC2: {round(100 * explained_var[1], 2)}%')
        axs[1].set_ylabel(f'PC3: {round(100 * explained_var[2], 2)}%')

    if post_df is not None:
        post_data = post_df.drop(columns=[col for col in ['TFX', 'Truth'] if col in post_df.columns]).to_numpy()
        sampleComponents = pca.transform(post_data)
        sample_df = pd.DataFrame(data=sampleComponents, columns=['PC1', 'PC2', 'PC3'], index=post_df.index)
        sample_df['TFX'] = post_df['TFX']
        if 'Truth' in post_df.columns:
            sample_df['Truth'] = post_df['Truth']
            sns.scatterplot(x="PC1", y="PC2", data=sample_df, palette=palette, ax=axs[0], legend=True, alpha=0.8, hue="Truth", size="TFX")
            sns.scatterplot(x="PC2", y="PC3", data=sample_df, palette=palette, ax=axs[1], legend=True, alpha=0.8, hue="Truth", size="TFX")
        else:
            sns.scatterplot(x="PC1", y="PC2", data=sample_df, palette=palette, ax=axs[0], legend=True, alpha=0.8, color='black', size="TFX")
            sns.scatterplot(x="PC2", y="PC3", data=sample_df, palette=palette, ax=axs[1], legend=True, alpha=0.8, color='black', size="TFX")

    axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'{direct}{name}.pdf', bbox_inches="tight")


def plot_ctdpheno(predictions: pd.DataFrame, direct: str, key: str, threshold: float, plot_range=[0, 1]) -> None:
    """
    Plot stick and ball plot of prediction (relative log likelihood) for each subtype.
    
    Parameters:
       predictions (pd.DataFrame): The input dataframe with predictions and Truth column.
       direct (str): The output directory.
       key (str): The name of the subtype/label to base thresholding on.
       threshold (float): The threshold to use for binary classification.
       plot_range (list): The range of values to plot. Default is [0, 1].
    """
    # Calculate accuracy based on Truth column
    def calculate_accuracy(group_df, key, threshold):
        if 'Truth' not in group_df.columns or all(truth.lower() == "unknown" for truth in group_df['Truth']):
            return "NA"
        
        # Check if key is in Truth values (allowing for CSV format)
        contains_key = []
        for truth in group_df['Truth']:
            # Split by comma and check if key is in any of the parts
            truth_parts = [part.strip() for part in str(truth).split(',')]
            contains_key.append(key in truth_parts)
        
        # Calculate accuracy
        correct_predictions = 0
        for idx, contains in enumerate(contains_key):
            score = group_df.iloc[idx][key]
            if contains and score >= threshold:  # True positive
                correct_predictions += 1
            elif not contains and score < threshold:  # True negative
                correct_predictions += 1
                
        return f"{100 * correct_predictions / len(group_df):.2f}%" if len(group_df) > 0 else "NA"
    
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
    norm = TwoSlopeNorm(vmin=plot_range[0], vcenter=threshold, vmax=plot_range[1])
    
    # Process each Truth group
    for idx, (ax, truth_val) in enumerate(zip(axs, ordered_truths)):
        # Get data for this group
        group_df = predictions[predictions['Truth'] == truth_val]
        
        # Calculate accuracy for this group
        acc_text = calculate_accuracy(group_df, key, threshold)
        
        # Create x positions for bars
        x_positions = np.arange(len(group_df))
        y_values = group_df[key].values
        
        # Make bars start at threshold and go up/down
        bar_heights = y_values - threshold
        colors = [cmap(norm(val)) for val in y_values]
        
        # Plot bars
        bars = ax.bar(x_positions, bar_heights, width=0.2, bottom=threshold, color=colors)
        
        # Plot scatter points at the values - larger and without outlines
        ax.scatter(x_positions, y_values, s=150, color=colors, zorder=3, edgecolor=None)
        
        # Add horizontal threshold line
        ax.axhline(y=threshold, color='black', linestyle='--', lw=1)
        
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
    cbar.set_ticks([0.0, threshold, 1.0])
    cbar.set_ticklabels(['0.0', f'{threshold:.2f}', '1.0'])

    # Add a figure-level title
    fig.suptitle(f"{key} Classification Scores", fontsize=16, y=1.02)

    # Don't call tight_layout again after positioning the colorbar
    plt.savefig(direct + 'ctdPheno_class-predictions.pdf', bbox_inches="tight")
    plt.close()


def plot_keraon(predictions: pd.DataFrame, direct: str, key: str, threshold: float, palette=None) -> None:
    """
    Plot stacked bar plots as subplots arranged horizontally where each subplot’s width
    is proportional to its number of samples. Within each subplot the samples are sorted by
    the "TFX" column (high to low). The leftmost subplot title shows the label "Truth:" 
    and "Accuracy:" along with its computed values, while the remaining subplots simply show
    the truth value and accuracy.
    
    A figure-level title is added that reads, for example:
       "Mixture estimates (NEPC presence based on NEPC_fraction thresholding)"
    """
    # Group by Truth and order by group size (descending)
    truth_groups = predictions.groupby('Truth')
    group_sizes_series = truth_groups.size().sort_values(ascending=False)
    ordered_truths = group_sizes_series.index.tolist()
    
    # Determine each group's sample count and width ratios for the subplots
    width_ratios = [group_sizes_series.loc[t] for t in ordered_truths]
    
    # Choose a fixed bar width (in inches) per sample.
    bar_width = 0.3
    total_width = sum(width_ratios) * bar_width + 1  # add some extra margin
    
    n_groups = len(ordered_truths)
    # Create subplots in one row with variable widths according to width_ratios.
    fig, axs = plt.subplots(1, n_groups, figsize=(total_width, 6), gridspec_kw={'width_ratios': width_ratios})
    if n_groups == 1:
        axs = [axs]
    
    for idx, (ax, truth_val) in enumerate(zip(axs, ordered_truths)):
        # Subset group and order samples by TFX descending
        group_df = predictions[predictions['Truth'] == truth_val].sort_values(by="TFX", ascending=False)
        n_samples = group_df.shape[0]
        
        # Compute accuracy if truth is not "Unknown"
        if truth_val.lower() != "unknown":
            subkey = key.lower().replace('_fraction', '')
            if subkey in truth_val.lower():
                # If the Truth label includes NEPC, patients with NEPC_fraction > threshold are correct.
                correct = (group_df[key] > threshold)
            else:
                # If Truth does NOT include NEPC, any patient with NEPC_fraction > threshold is incorrect.
                correct = (group_df[key] <= threshold)
            acc = correct.mean() * 100
            acc_text = f"{acc:.2f}%"
        else:
            acc_text = "NA"
        
        # Prepare the data for the stacked bar plot:
        subset = group_df.filter(like='_fraction')
        # If the key column is missing, try key + '_fraction'
        if key not in subset.columns:
            potential_key = f"{key}_fraction"
            if potential_key in subset.columns:
                key = potential_key
        
        cols = subset.columns.tolist()
        if 'Healthy_fraction' in cols:
            cols.remove('Healthy_fraction')
        if key in cols:
            cols.remove(key)
        new_cols = [key] + cols
        if 'Healthy_fraction' in group_df.columns:
            new_cols = new_cols + ['Healthy_fraction']
        subset = subset[new_cols]
        
        # Plot the stacked bar plot.
        if palette is not None:
            pal = {f"{k}_fraction": v for k, v in palette.items()}
            pal["Healthy_fraction"] = "#D3D3D3"
            subset.plot(kind='bar', stacked=True, color=pal, width=0.90, legend=False, ax=ax)
        else:
            subset.plot(kind='bar', stacked=True, width=0.90, legend=False, ax=ax)
        
        # Only the leftmost subplot gets the y-axis label; hide it for the rest.
        if idx == 0:
            ax.set_ylabel('total fraction')
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        ax.axhline(y=threshold, color='black', linestyle='dashed')
        ax.set_ylim(0, 1)
        # Set x-axis width based on the number of samples in the group.
        ax.set_xlim(-0.5, n_samples - 0.5)
        
        # Only the leftmost gets full labeling
        if idx == 0:
            ax.set_title(f"Truth: {truth_val}\nAccuracy: {acc_text}", fontsize=14)
        else:
            ax.set_title(f"{truth_val}\n{acc_text}", fontsize=14)

    # Get the last subplot to add the legend to
    last_ax = axs[-1]
    # Create a legend for the last subplot and place it outside, higher and to the right
    handles, labels = last_ax.get_legend_handles_labels()
    # Reorder to match the stacking order (bottom to top in the bars)
    handles = handles[::-1]
    labels = labels[::-1]
    # Place legend outside the rightmost subplot, higher up
    last_ax.legend(handles, labels, title="Subtypes", 
                bbox_to_anchor=(1.15, 1.0), 
                loc='upper left', 
                borderaxespad=0)
    
    # Add a figure-level title.
    # If key ends with '_fraction', remove that for the presence description.
    key_display = key.replace('_fraction', '')
    fig.suptitle(f"Mixture estimates ({key_display} presence based on {key} thresholding)", fontsize=16, y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(direct + 'Keraon_mixture-predictions.pdf', bbox_inches="tight")
    plt.close()


def plot_roc(df: pd.DataFrame, direct: str, key: str) -> list:
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters:
       df (pd.DataFrame): The input dataframe. It should contain a 'Truth' column with class labels and a column with predicted probabilities for the subtype of interest.
                          If multiple subtypes are present, they should be separated by commas, e.g. "Subtype1,Subtype2".
       direct (str): The output directory.
       key (str): The name of the subtype/label to base thresholding on.

    Returns:
       list: A list containing false positive rates, true positive rates, and the area under the ROC curve.
    """
    # Create a new figure
    plt.figure(figsize=(8, 8))

    # Convert multiclass labels into binary labels
    if "fraction" in key:
        df.loc[:, 'Truth'] = df['Truth'].apply(lambda x: 1 if any(label.strip() == key.rsplit('_fraction', 1)[0] for label in x.split(',')) else 0)
    else:
        df.loc[:, 'Truth'] = df['Truth'].apply(lambda x: 1 if any(label.strip() == key for label in x.split(',')) else 0)

    # Sort the dataframe by truth values
    df = df.sort_values(by='Truth')

    # Get the truth values and predicted probabilities
    truths = df['Truth'].to_numpy(dtype=int)
    predictions = df[key].to_numpy(dtype=float)

    # Calculate the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(truths, predictions)

    # Calculate the area under the ROC curve
    auc = metrics.auc(fpr, tpr)

    # Sensitivity = TPR = TP / (TP + FN)
    # Specificity = 1 - FPR = TN / (TN + FP)
    # Youden's J statistic = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1 = TPR - FPR
    J = tpr - fpr

    # The optimal threshold is the one that maximizes the Youden's J statistic
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='Mean ROC (AUC = % 0.2f )' % auc, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle(key + ' binary prediction ROC')
    plt.title('100% TPR threshold: ' + str(round(thresholds[list(tpr).index(1.)], 4)) +
              '\n"Optimal" (Youden) threshold: ' + str(round(optimal_threshold, 4)))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Save the figure
    plt.savefig(direct + key + '_ROC.pdf', bbox_inches="tight")

    # Close the figure
    plt.close()

    return [fpr, tpr, auc, optimal_threshold]


def plot_combined_feature_distributions(df_train: pd.DataFrame, df_test: pd.DataFrame, output_directory: str, palette: dict):
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

#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 11/08/2023

"""
This module contains plotting functions for final predictions from Keraon.
"""
# TODO: impliment a way to pass the subtype of interest to plot_ctdpheno, plot_roc (as opposed to NEPC)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import warnings # ignore warnings, e.g. matplotlib deprecation warnings


def plot_pca(df: pd.DataFrame, direct: str, palette: dict, name: str, post_df=None) -> None:
    """
    Perform standard scaling and plot first two principal components.

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
    
    # Scale the data
    df_data = StandardScaler().fit_transform(df_data.to_numpy())
    
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
        post_data = StandardScaler().fit_transform(post_df.drop(columns=[col for col in ['TFX', 'Truth'] if col in post_df.columns]).to_numpy())
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

    return


def plot_ctdpheno(predictions: pd.DataFrame, direct: str, key: str,  threshold: float, label='sample') -> None:
    """
    Plot stick and ball plot of prediction (relative log liklihood) for each subtype: key, based on truth labels. 

    Parameters:
       predictions (pd.DataFrame): The input dataframe. It should contain predicted probabilities for each subtype.
       direct (str): The output directory.
       subtypes (list): A list of subtype names.
       key (str): The name of the subtype/label to base thresholding on.
       threshold (float): The threshold to use for binary classification.
       label (str): The label for the plot. Default is 'sample'.
    """
    if label == key:
        acc = (predictions[key] >= threshold).mean() * 100
    else:
        acc = (predictions[key] < threshold).mean() * 100
    predictions.loc[:, key] = predictions[key] - threshold
    
    # Create a diverging color map that is centered at zero
    cmap = sns.color_palette("vlag", as_cmap=True)
    norm = TwoSlopeNorm(vmin=-threshold, vcenter=0, vmax=(1-threshold))
    # Convert the colormap to a list of colors
    normalized_data = norm(predictions[key])
    colors = cmap(normalized_data)
    # Create a dictionary mapping each unique value in the `hue` column to a color
    unique_hue_values = predictions[key].unique()
    color_dict = dict(zip(unique_hue_values, colors))

    sns.set_context(rc={'patch.linewidth': 0.0})
    plt.figure(figsize=(int(len(predictions)/2), 3))
    g = sns.barplot(x=predictions.index, y=key, hue=key, data=predictions, palette=color_dict, dodge=False)
    g.legend_.remove()
    g.set_ylim(-threshold, 1-threshold)
    sns.scatterplot(x=predictions.index, y=key, hue=key, data=predictions, palette=color_dict, s=600, legend=False)

    def change_width(ax, new_value):
        for patch in ax.patches:
            current_width = patch.get_width()
            diff = current_width - new_value
            # we change the bar width
            patch.set_width(new_value)
            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    change_width(g, .2)
    for item in g.get_xticklabels():
        item.set_rotation(90)
    plt.axhline(y=0, color='b', linestyle='--', lw=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('relative log likelihood')
    plt.title(label + ' (' + str(round(acc, 2)) + '% accuracy)')
    plt.savefig(direct + label + '_predictions.pdf', bbox_inches="tight")
    plt.close()

def plot_keraon(predictions: pd.DataFrame, direct: str, key: str,  threshold: float, label='sample', palette=None) -> None:
    """
    Plot stacked bar plot (basis components) for each subtype: key, based on truth labels. 

    Parameters:
       predictions (pd.DataFrame): The input dataframe. It should contain predicted probabilities for each subtype.
       direct (str): The output directory.
       subtypes (list): A list of subtype names.
       key (str): The name of the subtype/label to base thresholding on.
       threshold (float): The threshold to use for binary classification.
       label (str): The label for the plot. Default is 'sample'.
    """
    if label + '_fraction' == key:
        acc = (predictions[key] >= threshold).mean() * 100
    else:
        acc = (predictions[key] < threshold).mean() * 100
    palette = {k + '_fraction': v for k, v in palette.items()}
    # Change the color for "Healthy_fraction" to light gray
    palette["Healthy_fraction"] = "#D3D3D3"
    # predictions = predictions.sort_values(by=['Healthy_fraction', key], ascending=[True, False])
    plt.figure()
    predictions = predictions.filter(like='_fraction')
    # Rearrange columns
    cols = predictions.columns.tolist()
    cols.remove('Healthy_fraction')
    cols.remove(key)
    cols = [key] + cols + ['Healthy_fraction']
    predictions = predictions[cols]
    if label == 'Unknown':  # sort by tumor fraction: grouping
        predictions = predictions.sort_index()
    if palette is not None:
        predictions.plot(kind='bar', stacked=True, color=palette, width=0.90, legend=False, figsize=(int(len(predictions)/2), 3))
    else:
        predictions.plot(kind='bar', stacked=True, width=0.90, legend=False, figsize=(int(len(predictions)/2), 3))
    plt.ylabel('total fraction')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.axhline(y=threshold, color='black', linestyle='dashed')
    plt.ylim=(0, 1)
    plt.title(label + ' (' + str(round(acc, 2)) + '% accuracy)')
    plt.savefig(direct + label + '_predictions.pdf', bbox_inches="tight")
    plt.close()

    # Previous code for dividing up the space:
        # if len(df['Subtype'].unique()) == 2:
    #     known_subtypes = ['ARPC', 'NEPC']
    # elif len(df['Subtype'].unique()) == 3:
    #     known_subtypes = ['ARPC', 'NEPC', 'MIX']
    # else:
    #     known_subtypes = ['ARPC', 'NEPC', 'MIX', 'ARPC_crpc']
    # widths = [df.loc[df['Subtype'] == sub].shape[0] / df.loc[df['Subtype'] == known_subtypes[-1]].shape[0] for sub
    #                 in known_subtypes]
    # fig, axes = plt.subplots(nrows=1, ncols=len(known_subtypes), sharey=True, gridspec_kw={'width_ratios': widths},
    #                          figsize=(16, 2))
    # for i, subtype in enumerate(known_subtypes):
    #     df_sub = df.loc[df['Subtype'] == subtype]
    #     if subtype == 'ARPC' or subtype == 'ARPC_crpc':
    #         acc = round(df_sub.loc[df_sub['NEPC_fraction'] < df_sub['thresh']].shape[0] / df_sub.shape[0], 4)
    #         subtitle = 'ARPC (' + str(100 * acc) + '%)'
    #     elif subtype == 'NEPC':
    #         acc = round(df_sub.loc[df_sub['NEPC_fraction'] >= df_sub['thresh']].shape[0] / df_sub.shape[0], 4)
    #         subtitle = 'NEPC (' + str(100 * acc) + '%)'
    #     else:
    #         subtitle = 'MIXED'
    #     df_sub = df_sub[['NEPC_fraction', 'ARPC_fraction', 'Healthy_fraction']]
    #     df_sub.plot(ax=axes[i], kind='bar', stacked=True, color=palette_stacked, width=0.90, legend=False)
    #     axes[i].set_title(subtitle)
    #     if i == 0:
    #         axes[i].set_ylabel('total fraction')
    #     elif i == len(known_subtypes) - 1:
    #         axes[i].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    #     if plot_thresh:
    #         axes[i].axhline(y=thresh, color='black', linestyle='dashed')
    return


def plot_roc(df: pd.DataFrame, direct: str, key: str) -> list:
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters:
       df (pd.DataFrame): The input dataframe. It should contain a 'Truth' column with class labels and a column with predicted probabilities for the subtype of interest.
       direct (str): The output directory.
       key (str): The name of the subtype/label to base thresholding on.

    Returns:
       list: A list containing false positive rates, true positive rates, and the area under the ROC curve.
    """
    # Create a new figure
    plt.figure(figsize=(8, 8))

    # Convert multiclass labels into binary labels
    if "fraction" in key:
        df.loc[:, 'Truth'] = df['Truth'].apply(lambda x: 1 if x == key.rsplit('_fraction', 1)[0] else 0)
    else:
        df.loc[:, 'Truth'] = df['Truth'].apply(lambda x: 1 if x == key else 0)

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
    plt.savefig(direct + 'ROC.pdf', bbox_inches="tight")

    # Close the figure
    plt.close()

    return [fpr, tpr, auc, optimal_threshold]

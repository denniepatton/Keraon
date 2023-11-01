#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 09/19/2023

"""
plotting functions (for finalized output) for Keraon
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'Gray', 'AMPC', 'MIX', '-', '+']
# colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#FFAE42', '#9F009F', '#33BBEE', '#EE7733']
# palette = {targets[i]: colors[i] for i in range(len(targets))}
# # modify this to be the colormap of interest, based on the passed palette (pass a palette path or use defaults)
# cmap = LinearSegmentedColormap.from_list('', ['#0077BB', '#CC3311'])
# matplotlib.colormaps.register(cmap, name="mcm")
# sns.set(font_scale=1.5)
# sns.set_style('ticks')

def plot_ctdpheno(predictions, direct, subtypes):
    """
    only handles the binary case right now
    TODO: add support for multi-class
    TODO: add support for a palette
    """
    if "NEPC" in subtypes:  # use the NEPC threshold from the Cancer Discovery paper
        shift = -0.3314
        predictions['NEPC'] = predictions['NEPC'] + shift
        key_type = 'NEPC'
    else:
        key_type = subtypes[0]
    data = predictions.groupby(predictions[key_type]).size()
    pal = sns.color_palette("vlag", len(data))
    sns.set_context(rc={'patch.linewidth': 0.0})
    plt.figure(figsize=(int(len(predictions)/2), 3))
    g = sns.barplot(x=predictions.index, y=key_type, hue=key_type, data=predictions, palette=pal, dodge=False)
    g.legend_.remove()
    g.set_ylim(-1, 1)
    sns.scatterplot(x=predictions.index, y=key_type, hue=key_type, data=predictions, palette=pal, s=600, legend=False)

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
    plt.savefig(direct + 'predictions.pdf', bbox_inches="tight")
    plt.close()


def plot_roc(df, direct):
    """
    only handles the binary case right now
    TODO: add support for multi-class
    """
    df = df[df['Truth'].isin(['ARPC', 'NEPC'])]
    plt.figure(figsize=(8, 8))
    df = df.sort_values(by='Truth')
    truths = pd.factorize(df['Truth'].values)[0]
    predictions = df['NEPC'].values
    fpr, tpr, thresholds = metrics.roc_curve(truths, predictions)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Mean ROC (AUC = % 0.2f )' % auc, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('NEPC binary prediction ROC')
    plt.title('100% TPR threshold: ' + str(round(thresholds[list(tpr).index(1.)], 4)))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(direct + 'ROC.pdf', bbox_inches="tight")
    plt.close()
    return [fpr, tpr, auc]


def plot_feature_space(df_ref, df_test, axes, palette, label_anchor_points=False, label_test_points=False):
    """
        Plot the (2D) feature space given in axes, with the anchor data as kernels and faded points, and the sample
        data as opaque points scaled by tumor fraction.
            Parameters:
               df_ref: pandas dataframe with anchor data
               df_test: pandas dataframe with test/sample data
               axes ([x, y]): features to plot - must match df column names
               palette ({dict}): colors for each passed subtype
               label_anchor_points (bool): whether to label anchor points with sample names
               label_test_points (bool): whether to label test points with sample names
            Returns:
               fig: matplotlib.pyplot.figure object
        """
    plt.close('all')
    plt.figure(figsize=(8, 8))
    if palette is not None:
        sns.kdeplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', fill=True, palette=palette)
        sns.scatterplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', palette=palette, alpha=0.2)
    else:
        sns.kdeplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', fill=True)
        sns.scatterplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', alpha=0.2)
    if palette is not None and 'Subtype' in df_test and 'TFX' in df_test:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], hue='Subtype', size='TFX', palette=palette)
    elif 'Subtype' in df_test and 'TFX' in df_test:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], hue='Subtype', size='TFX')
    elif 'TFX' in df_test:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], color='Purple', size='TFX')
    else:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], color='Purple')
    if label_anchor_points:
        for x, y, label in zip(df_ref.loc[:, axes[0]], df_ref.loc[:, axes[1]], df_ref.index.values):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=4)
    if label_test_points:
        for x, y, label in zip(df_test.loc[:, axes[0]], df_test.loc[:, axes[1]], df_test.index.values):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=3)
    plt.title('Anchor Distributions and Test Points', size=14, y=1.02)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt.gcf()

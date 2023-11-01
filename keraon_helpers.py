#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 09/19/2023

"""
helper functions (feature selection and analyses) for Keraon
"""

from sys import stdout
import os
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm, shapiro
from scipy.optimize import minimize
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics, preprocessing
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'Gray', 'AMPC', 'MIX', '-', '+']
colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#FFAE42', '#9F009F', '#33BBEE', '#EE7733']
palette = {targets[i]: colors[i] for i in range(len(targets))}
cmap = LinearSegmentedColormap.from_list('', ['#0077BB', '#CC3311'])
matplotlib.colormaps.register(cmap, name="mcm")
sns.set(font_scale=1.5)
sns.set_style('ticks')

def scale_features(df, stdev_dict=None):
    """
    Scale features based on the largest standard deviation in each group to ensure equal weighting and "unitless" comparison between
    feature types. If a standard deviaiton dictionary is supplied, use the values to scale the features (for use in scaling test samples).
        Parameters:
           df: pandas dataframe
           stdev_dict: dictionary of largest anchor standard deviations for each feature
    """
    if stdev_dict is None:
        stdev_dict = {}
        for column in df.columns:
            if column == 'Subtype':
                continue
            for subtype in df['Subtype'].unique():
                stdev = np.nanstd(df.loc[df['Subtype'] == subtype, column].values)
                if column not in stdev_dict or stdev > stdev_dict[column]:
                    stdev_dict[column] = stdev
            df[column] = df[column].apply(lambda x: x / stdev_dict[column])
    else:
        for column in df.columns:
            if column == 'Subtype':
                continue
            df[column] = df[column].apply(lambda x: x / stdev_dict[column])
    return df, stdev_dict



def select_features(df):
    """
    Given a reference data frame containing features and values for select anchoring subtypes, return
    a subset of the dataframe which maximizes the 'difference' between the anchoring subtypes
        Parameters:
           df: pandas dataframe
        Returns:
           out_df: the input df, restricted to the most differentiating features
    """
    n_classes = len(df.Subtype.unique())
    n_features = len(df.columns) - 1
    class_mats = [df.loc[df['Subtype'] == subtype].drop('Subtype', axis=1).to_numpy() for subtype in df.Subtype.unique()]
    print('Running anchor separation maximation on features . . .')
    print("Total classes: " + str(n_classes))
    print("Total features: " + str(n_features))

    def anchor_objective(theta):
        """
        Objective function for anchor selection: find the mean Euclidean distance between all sets of anchor centers
        (assuming multi-variate Gaussian form), and scale (divide by) the mean generalized variance (det of covariance).
        Parameters:
            class_mats: numpy array of 2D arrays, one for each class, with features on the x/1-axis and samples on the y/0-axis
            Returns:
            val: scalar function values
        """
        n_classes = len(class_mats)
        total_edges = n_classes * (n_classes - 1) / 2
        feature_mask = theta.astype(bool)
        class_locs = []
        mean_gv, mean_edge = 0.0, 0.0
        for class_mat in class_mats:
            masked_mat = class_mat[:, feature_mask]
            class_locs.append(np.mean(masked_mat, axis = 0))
            mean_gv += np.linalg.det(np.cov(masked_mat, rowvar=False))
        for idx_a, ele_a in enumerate(class_locs):
            for idx_b, ele_b in enumerate(class_locs):
                if idx_a >= idx_b:
                    continue
                mean_edge += np.linalg.norm(ele_a - ele_b)
        mean_edge /= total_edges
        mean_gv /= n_classes
        score = (-1 * mean_edge**2 / mean_gv)
        return score
    
    # currently brute forcing this MINLP problem - could be improved with a more efficient algorithm
    n = 0
    best_score = 0.0
    total_combinations = math.comb(n_features, n_classes + 1)
    print("Total combinatorial search space: " + str(total_combinations))
    for combo in combinations(range(n_features), n_classes + 1):
        n += 1
        theta = np.zeros(n_features)
        theta[list(combo)] = 1.0
        score = anchor_objective(theta)
        if score < best_score:
            best_score = score
            best_theta = theta
        if n % 10000 == 0:
            stdout.write('\rRunning axes-combinations | completed: [{0}/'.format(n) + str(total_combinations) + ']')
            stdout.flush()
    print('\n')
    df_out = df[df.columns[np.insert(best_theta.astype(bool), 0, [True])]]  # insert True for Subtype column
    return df_out

def diff_exp(df, name, thresh=0.05, sub_name=''):
    print('Conducting differential expression analysis . . .')
    types = list(df.Subtype.unique())
    df_t1 = df.loc[df['Subtype'] == types[0]].drop('Subtype', axis=1)
    df_t2 = df.loc[df['Subtype'] == types[1]].drop('Subtype', axis=1)
    df_lpq = pd.DataFrame(index=df_t1.transpose().index, columns=['ratio', 'p-value'])
    for roi in list(df_t1.columns):
        x, y = df_t1[roi].values, df_t2[roi].values
        if np.count_nonzero(~np.isnan(x)) < 2 or np.count_nonzero(~np.isnan(y)) < 2:
            continue
        df_lpq.at[roi, 'ratio'] = np.mean(x)/np.mean(y)
        df_lpq.at[roi, 'p-value'] = mannwhitneyu(x, y)[1]
    # now calculate p-adjusted (Benjamini-Hochberg corrected p-values)
    df_lpq['p-adjusted'] = fdrcorrection(df_lpq['p-value'])[1]
    df_lpq = df_lpq.sort_values(by=['p-adjusted'])
    df_lpq = df_lpq.infer_objects()
    df_lpq.to_csv(name + '/' + name + sub_name + '_rpq.tsv', sep="\t")
    features = list(df_lpq[(df_lpq['p-adjusted'] < thresh)].index)
    with open(name + '/' + name + sub_name + '_FeatureList.tsv', 'w') as f_output:
        for item in features:
            f_output.write(item + '\n')
    return pd.concat([df.iloc[:, :1], df.loc[:, df.columns.isin(features)]], axis=1, join='inner')


def norm_exp(df, name, thresh=0.05):
    """
    Calculate the Shapiro p-values (likelihood data does NOT belong to a normal distribution) for each subtype in each
    feature and return the dataframe with only features where each subtype is normally distributed
        Parameters:
           df: pandas dataframe
           name (string): name to write results locally
           thresh (float): the minimum p-value limit to return
        Returns:
           out_df: the input df, restricted to normal features
           df_lpq: p-values for each feature (1 - mean value across subtypes)
    """
    print('Conducting normal expression analysis . . .')
    phenotypes = list(df['Subtype'].unique())
    pheno_dfs = [df.loc[df['Subtype'] == phenotype].drop('Subtype', axis=1) for phenotype in phenotypes]
    df_lpq = pd.DataFrame(index=pheno_dfs[0].transpose().index,
                          columns=[phenotype + '_p-value' for phenotype in phenotypes])
    for roi in list(pheno_dfs[0].columns):
        for pheno_index, pheno_df in enumerate(pheno_dfs):
            values = pheno_df[roi].values
            shapiro_test = shapiro(values)
            df_lpq.at[roi, phenotypes[pheno_index] + '_p-value'] = shapiro_test.pvalue
    df_lpq = df_lpq.loc[(df_lpq[df_lpq.columns] > thresh).all(axis=1)]
    df_lpq['p-adjusted'] = df_lpq.mean(axis=1)
    df_lpq.loc[:, 'p-adjusted'] = df_lpq['p-adjusted'].apply(lambda x: 1 - x)
    df_lpq.to_csv(name + 'shapiros.tsv', sep="\t")
    features = list(df_lpq.index)
    return df.loc[:, df.columns.isin(features)], df_lpq


def corr_exp(df, name, thresh=0.95, df_ref_1=None, df_ref_2=None):
    """
    Calculate the correlation matrix for all features and return only one from each group of correlated features
    (those meeting the threshold for correlation). If an additional dataframe with p-values for each feature is
    supplied (e.g. differential expression data) the feature with the lowest p-value in each comparison is
    returned. If there is a tie in p-values another dataframe may be supplied with p-values to break ties (e.g.
    normalcy among classes p-values).
        Parameters:
           df: pandas dataframe
           name (string): name to write results locally
           thresh (float): the maximum correlation limit to return
           df_ref_1: a dataframe with features and 'p-adjusted'
           df_ref_2: a dataframe with features and 'p-adjusted'
        Returns:
           out_df: the input df, restricted to non-correlated features
    """
    print('Conducting correlation analysis . . .')
    correlated_features = set()
    correlation_matrix = df.corr()
    correlation_matrix.to_csv(name + '/' + name + '_correlations.tsv', sep="\t")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > thresh:
                row_name = correlation_matrix.index[i]
                col_name = correlation_matrix.columns[j]
                if df_ref_1 is not None:
                    if df_ref_1.at[col_name, 'p-adjusted'] < df_ref_1.at[row_name, 'p-adjusted']:
                        correlated_features.add(row_name)
                    elif df_ref_1.at[col_name, 'p-adjusted'] > df_ref_1.at[row_name, 'p-adjusted']:
                        correlated_features.add(col_name)
                    elif df_ref_2 is not None:
                        if df_ref_2.at[col_name, 'p-adjusted'] < df_ref_2.at[row_name, 'p-adjusted']:
                            correlated_features.add(row_name)
                        else:
                            correlated_features.add(col_name)
                    else:
                        correlated_features.add(col_name)
                else:
                    correlated_features.add(col_name)
    df_out = df.drop(labels=correlated_features, axis=1)
    return df_out


def metric_analysis(df, direct):
    print('Calculating metric dictionary . . .')
    df = df.dropna(axis=0)
    features = list(df.iloc[:, 1:].columns)
    types = list(df.Subtype.unique())
    mat = {}
    for feature in features:
        sub_df = pd.concat([df.iloc[:, :1], df[[feature]]], axis=1, join='inner')
        mat[feature] = {'Feature': feature}
        for subtype in types:
            mat[feature][subtype + '_Mean'] = np.nanmean(
                sub_df[sub_df['Subtype'] == subtype].iloc[:, 1:].to_numpy().flatten())
            mat[feature][subtype + '_Std'] = np.nanstd(
                sub_df[sub_df['Subtype'] == subtype].iloc[:, 1:].to_numpy().flatten())
    pd.DataFrame(mat).to_csv(direct + 'metric_dict.tsv', sep="\t")
    return mat


def specificity_sensitivity(target, predicted, threshold):
    thresh_preds = np.zeros(len(predicted))
    thresh_preds[predicted > threshold] = 1
    cm = metrics.confusion_matrix(target, thresh_preds)
    return cm[1, 1] / (cm[1, 0] + cm[1, 1]), cm[0, 0] / (cm[0, 0] + cm[0, 1])


def nroc_curve(y_true, predicted, num_thresh=100):
    step = 1/num_thresh
    thresholds = np.arange(0, 1 + step, step)
    fprs, tprs = [], []
    for threshold in thresholds:
        y_pred = np.where(predicted >= threshold, 1, 0)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fprs.append(fp / (fp + tn))
        tprs.append(tp / (tp + fn))
    return fprs, tprs, thresholds

def product_column(a, b):
    ab = []
    for item_a in a:
        for item_b in b:
            ab.append(item_a + '_' + item_b)
    return ab


def subset_data(df, sub_list):
    regions = list(set([item.split('_')[0] for item in list(df.columns) if '_' in item]))
    categories = list(set([item.split('_')[1] for item in list(df.columns) if '_' in item]))
    features = list(set([item.split('_')[2] for item in list(df.columns) if '_' in item]))
    sub_list += [region for region in regions if any(gene + '-' in region for gene in sub_list)]
    sub_list = list(set(sub_list))
    all_features = product_column(categories, features)
    sub_features = product_column(sub_list, all_features)
    sub_df = df[df.columns.intersection(sub_features)]
    return pd.concat([df['Subtype'], sub_df], axis=1, join='inner')

def _get_intersect(points_a, points_b):
    """
        Returns the point of intersection of the line passing through points_a and the line passing through points_b
        (2D) or the line passing through points_a and the plane defined by points_b (3D).
            Parameters:
                points_a [a1, a2]: points defining line a
                points_b [b1, b2, (b3)]: points defining line (plane) b
            Returns:
                out_point [x, y, . . .]: the point of intersection in base coordinates
    """
    if len(points_b) == 2:
        s = np.vstack([points_a[0], points_a[1], points_b[0], points_b[1]])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:  # lines are parallel - SHOULD be impossible given constraints to call this function
            return float('inf'), float('inf')
        return np.array([x/z, y/z])
    else:
        # below is untested
        norm = np.cross(points_b[2] - points_b[0], points_b[1] - points_b[0])
        w = points_a[0] - points_b[0]
        si = -norm.dot(w) / norm.dot(points_a[1] - points_a[0])
        psi = w + si * (points_a[1] - points_a[0]) + points_a[0]
        return psi


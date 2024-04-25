#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 11/08/2023

"""
This module contains helper functions for feature selection and analyses for Keraon.
"""

import itertools
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import shapiro

def is_positive_semi_definite(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= 0)

# def scale_features(df: pd.DataFrame, stdev_dict: dict = None) -> tuple:
#     """
#     Scale features based on the largest standard deviation in each group to ensure equal weighting and "unitless" comparison between
#     feature types. If a standard deviation dictionary is supplied, use the values to scale the features (for use in scaling test samples).
#     Can be thought of as scaling by "the weakest link" without shifting any means.

#     Parameters:
#        df (pd.DataFrame): The dataframe containing the features to scale.
#        stdev_dict (dict): A dictionary of the largest anchor standard deviations for each feature.

#     Returns:
#        df (pd.DataFrame): The dataframe with the scaled features.
#        stdev_dict (dict): The dictionary of the largest anchor standard deviations for each feature.
#     """
#     # If no standard deviation dictionary is supplied, calculate it
#     print('Scaling features "by their weakest link" . . .')
#     if stdev_dict is None:
#         stdev_dict = {}
#         for column in df.columns:
#             if column == 'Subtype':
#                 continue
#             for subtype in df['Subtype'].unique():
#                 stdev = np.nanstd(df.loc[df['Subtype'] == subtype, column].values)
#                 if column not in stdev_dict or stdev > stdev_dict[column]:
#                     stdev_dict[column] = stdev

#     # Scale the features in the dataframe
#     for column in df.columns:
#         if column != 'Subtype' and column in stdev_dict:
#             df[column] /= stdev_dict[column]

#     return df, stdev_dict


# def class_standardize(df: pd.DataFrame, mean_dict: dict = None, stdev_dict: dict = None, ref='All') -> tuple:
#     """
#     Standardize features based on a specific class (e.g., the mean and standard deviation of a reference class).

#     Parameters:
#        df (pd.DataFrame): The dataframe containing the features to scale.
#        stdev_dict (dict): A dictionary of the largest anchor standard deviations for each feature.

#     Returns:
#        df (pd.DataFrame): The dataframe with the scaled features.
#        stdev_dict (dict): The dictionary of the largest anchor standard deviations for each feature.
#     """
#     # If no standard deviation dictionary is supplied, calculate it
#     print('Standardizing features by ' + ref + ' . . .')
#     if mean_dict is None or stdev_dict is None:
#         mean_dict, stdev_dict = {}, {}
#         for column in df.columns:
#             if column == 'Subtype':
#                 continue
#             if ref == 'All':
#                 mean_dict[column] = np.nanmean(df[column].values)
#                 stdev_dict[column] = np.nanstd(df[column].values)
#             else:
#                 mean = np.nanmean(df.loc[df['Subtype'] == ref, column].values)
#                 mean_dict[column] = mean
#                 stdev = np.nanstd(df.loc[df['Subtype'] == ref, column].values)
#                 stdev_dict[column] = stdev

#     # Standardize the features in the dataframe
#     for column in df.columns:
#         if column != 'Subtype' and column in stdev_dict:
#             df[column] -= mean_dict[column]
#             df[column] /= stdev_dict[column]

#     return df, mean_dict, stdev_dict


def minmax_standardize(df: pd.DataFrame, min_dict: dict = None, range_dict: dict = None) -> tuple:
    """

    """
    # If no min/range dictionaries are supplied, calculate them
    print('Standardizing features by min/max . . .')
    if min_dict is None or range_dict is None:
        min_dict, range_dict = {}, {}
        for column in df.columns:
            if column == 'Subtype':
                continue
            min_dict[column] = np.nanmin(df[column].values)
            range_dict[column] = np.nanmax(df[column].values) - min_dict[column]

    # Standardize the features in the dataframe
    for column in df.columns:
        if column != 'Subtype' and column in min_dict:
            df[column] -= min_dict[column]
            df[column] /= range_dict[column]

    return df, min_dict, range_dict


def maximal_simplex_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a reference data frame containing features and values for select anchoring subtypes, return
    a subset of the dataframe which maximizes the simplex volume defined by those anchoring subtypes while
    minimizing the number of features used.
        Parameters:
           df: pandas dataframe
        Returns:
           out_df: the input df, restricted to the most differentiating features
    """
    n_classes = len(df.Subtype.unique())
    n_features = len(df.columns) - 1
    subtypes = df.Subtype.unique()
    class_mats = [df.loc[df['Subtype'] == subtype].drop('Subtype', axis=1).to_numpy() for subtype in subtypes]
    df = df.drop('Subtype', axis=1)
    print('Running anchor maximal simplex volume feature subsetting . . .')
    print("Total classes (simplex order + 1): " + str(n_classes))
    print("Total features (feature space order): " + str(n_features))
    
    def simplex_volume_recursive(vectors):
        """
        Calculate the volume of a simplex in n-dimensional space using recursion and the formula for a simplex volume based on its base and height.
        Parameters:
            vectors (list of np.array): A list of n+1 vectors, each describing a point in n-dimensional space.
        Returns:
            volume (float): The volume of the simplex.
        """
        n = len(vectors)
        if n == 1:
            return vectors[0][0]  # The volume of a 1D simplex (line segment) is its length
        else:
            # Calculate the base of the simplex (the volume of the (n-1)-dimensional simplex formed by the first n-1 vectors)
            base = simplex_volume_recursive(vectors[:-1])

            # Calculate the height of the simplex (the distance from the nth vector to the (n-1)-dimensional simplex formed by the first n-1 vectors)
            height = np.linalg.norm(vectors[-1] - vectors[0])

            # Calculate the volume of the simplex
            volume = base * height / n
            return volume

    def objective(feature_mask):
        """
        Objective function to be maximized.
        Parameters:
            feature_mask (np.array): A binary array indicating which features to include.
            vectors (list of np.array): A list of vectors.
        Returns:
            volume (float): The volume of the simplex formed by the vectors.
        """
        # Apply the feature mask to the vectors
        n_mask = sum(feature_mask)
        masked_mats = [mat[:, feature_mask.astype(bool)] for mat in class_mats]
        masked_vectors = [np.mean(masked_mat, axis=0) for masked_mat in masked_mats]
        masked_cov_mats = [np.cov(masked_mat, rowvar=False) for masked_mat in masked_mats]
        psd_stati = [is_positive_semi_definite(masked_cov_mat) for masked_cov_mat in masked_cov_mats]
        if not all(psd_stati):
            return 0
        masked_dets = [np.linalg.det(cov_matrix) for cov_matrix in masked_cov_mats]
        masked_nstdevs = [det**(1 / n_mask) if det >= 0 else 1000 for det in masked_dets]
        scale_factor = np.mean(masked_nstdevs)
        # Calculate the volume of the simplex, weighted by features used (minimize feature space)
        volume = simplex_volume_recursive(masked_vectors) / scale_factor
        return volume
    
    def greedy_maximize(fun, n, n_min):
        """
        Greedy algorithm to maximize a function.
        Parameters:
            fun (function): The function to be maximized. It takes a binary mask as input.
            n (int): The length of the binary mask.
        Returns:
            best_mask (np.array): The binary mask that maximizes the function.
        """
        # Initialize the best mask and best value
        best_mask = np.zeros(n, dtype=bool)
        best_value = -np.inf
        printed_features = set()

        # Evaluate all possible masks with n_min bits set to 1
        print('Evaluating all possible masks with ' + str(n_min) + ' bits set to 1 . . .')
        for indices in itertools.combinations(range(n), n_min):
            mask = np.zeros(n, dtype=bool)
            mask[list(indices)] = 1
            value = fun(mask)
            if value > best_value:
                best_mask = mask
                best_value = value
        print('Complete. Initial mask features:\n')

        for feature in df.columns[best_mask.astype(bool)]:
            print('-------   ' + feature)
            feature_index = df.columns.get_loc(feature)
            for subtype, mc_df in zip(subtypes, class_mats):
                print(subtype + ': Mean = ' + str(np.mean(mc_df[:, feature_index])) + ', Std Dev = ' + str(np.std(mc_df[:, feature_index])))
            printed_features.add(feature)

        # Use a greedy algorithm to maximize the function
        print('\nRunning greedy maximization . . . (added features:)\n')
        while True:
            # Initialize the best new mask and best new value
            best_new_mask = best_mask
            best_new_value = best_value

            # Iterate over all bits in the mask
            for i in range(n):
                if not best_mask[i]:
                    # Try flipping the bit
                    new_mask = best_mask.copy()
                    new_mask[i] = 1
                    new_value = fun(new_mask)

                    # Update the best new mask and best new value
                    if new_value > best_new_value:
                        best_new_mask = new_mask
                        best_new_value = new_value

            # If no single bit can increase the function value, stop the algorithm
            if best_new_value == best_value:
                break

            # Otherwise, update the best mask and best value
            best_mask = best_new_mask
            best_value = best_new_value
            for feature in df.columns[best_mask.astype(bool)]:
                if feature not in printed_features:
                    print('-------   ' + feature)
                    feature_index = df.columns.get_loc(feature)
                    for subtype, mc_df in zip(subtypes, class_mats):
                        print(subtype + ': Mean = ' + str(np.mean(mc_df[:, feature_index])) + ', Std Dev = ' + str(np.std(mc_df[:, feature_index])))
                    printed_features.add(feature)

        return best_mask

    result = greedy_maximize(lambda x: objective(x), n_features, n_classes-1)
    print('\nMaximal simplex volume feature subset complete.')
    print('Final (weighted) simplex volume: ' + str(objective(result)))
    print('Final number of features: ' + str(sum(result)))
    return df[df.columns[result.astype(bool)]]


def norm_exp(df: pd.DataFrame, direct: str, thresh: float = 0.05):
    """
    Calculate the Shapiro p-values (likelihood data does NOT belong to a normal distribution) for each subtype in each
    feature and return the dataframe with only features where each subtype is normally distributed.

    Parameters:
       df (pandas.DataFrame): The dataframe containing the features to analyze.
       direct (str): The directory to save the results.
       thresh (float): The minimum p-value limit to return.

    Returns:
       out_df (pandas.DataFrame): The input df, restricted to normal features.
       df_lpq (pandas.DataFrame): p-values for each feature ( and p_adj: 1 - mean value across subtypes).
    """
    print('Conducting normal expression analysis . . .')

    # Get the unique phenotypes
    phenotypes = df['Subtype'].unique()

    # Create a dataframe for each phenotype
    pheno_dfs = [df[df['Subtype'] == phenotype].drop('Subtype', axis=1) for phenotype in phenotypes]

    # Initialize the dataframe for the p-values
    df_lpq = pd.DataFrame(index=pheno_dfs[0].columns, columns=[f'{phenotype}_p-value' for phenotype in phenotypes])

    # Calculate the Shapiro p-values for each phenotype and feature
    for roi in pheno_dfs[0].columns:
        for pheno_index, pheno_df in enumerate(pheno_dfs):
            if len(pheno_df[roi]) < 3:
                print(f"Skipping Shapiro-Wilk test for {roi} as it has less than 3 data points.")
                df_lpq.loc[roi, f'{phenotypes[pheno_index]}_p-value'] = 1.0
            else:
                shapiro_test = shapiro(pheno_df[roi])
                df_lpq.loc[roi, f'{phenotypes[pheno_index]}_p-value'] = shapiro_test.pvalue

    # Filter the features based on the threshold
    df_lpq = df_lpq[(df_lpq > thresh).all(axis=1)]

    # Calculate the adjusted p-values
    df_lpq['p-adjusted'] = 1 - df_lpq.mean(axis=1)

    # Save the p-values to a file
    df_lpq.to_csv(f'{direct}shapiros.tsv', sep="\t")

    # Return the filtered dataframe and the p-values
    common_columns = df.columns.intersection(df_lpq.index)
    return df[common_columns], df_lpq


def specificity_sensitivity(target: np.array, predicted: np.array, threshold: float) -> tuple:
    """
    Calculate the specificity and sensitivity for a given threshold.

    Parameters:
       target (np.array): The true labels.
       predicted (np.array): The predicted probabilities.
       threshold (float): The threshold to use for classification.

    Returns:
       sensitivity (float): The sensitivity for the given threshold.
       specificity (float): The specificity for the given threshold.
    """
    # Apply the threshold to the predicted probabilities
    thresh_preds = np.zeros(len(predicted))
    thresh_preds[predicted > threshold] = 1

    # Calculate the confusion matrix
    cm = metrics.confusion_matrix(target, thresh_preds)

    # Calculate the sensitivity and specificity
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return sensitivity, specificity


def nroc_curve(y_true: np.array, predicted: np.array, num_thresh: int = 1000) -> tuple:
    """
    Calculate the false positive rates and true positive rates for different thresholds to generate a ROC curve.

    Parameters:
       y_true (np.array): The true labels.
       predicted (np.array): The predicted probabilities.
       num_thresh (int): The number of thresholds to use.

    Returns:
       fprs (list): The false positive rates for the thresholds.
       tprs (list): The true positive rates for the thresholds.
       thresholds (np.array): The thresholds used.
    """
    step = 1 / num_thresh
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


# def _get_intersect(points_a, points_b):
#     """
#         Returns the point of intersection of the line passing through points_a and the line passing through points_b
#         (2D) or the line passing through points_a and the plane defined by points_b (3D).
#             Parameters:
#                 points_a [a1, a2]: points defining line a
#                 points_b [b1, b2, (b3)]: points defining line (plane) b
#             Returns:
#                 out_point [x, y, . . .]: the point of intersection in base coordinates
#     """
#     if len(points_b) == 2:
#         s = np.vstack([points_a[0], points_a[1], points_b[0], points_b[1]])
#         h = np.hstack((s, np.ones((4, 1))))
#         l1 = np.cross(h[0], h[1])
#         l2 = np.cross(h[2], h[3])
#         x, y, z = np.cross(l1, l2)
#         if z == 0:  # lines are parallel - SHOULD be impossible given constraints to call this function
#             return float('inf'), float('inf')
#         return np.array([x/z, y/z])
#     else:
#         # below is untested
#         norm = np.cross(points_b[2] - points_b[0], points_b[1] - points_b[0])
#         w = points_a[0] - points_b[0]
#         si = -norm.dot(w) / norm.dot(points_a[1] - points_a[0])
#         psi = w + si * (points_a[1] - points_a[0]) + points_a[0]
#         return psi


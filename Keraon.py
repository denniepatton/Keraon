#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.2, 5/2/2024

"""
This newest iteration of Keraon aims to combine ctdPheno and Keraon together, to give dual output predictions
for any appropriately formatted datasets. It will also be made more user-friendly in true script form, with
exmaple runs saved in the accompanying EXAMPLE_RUNS.txt file. I plan to test new feature selection methods, as
well as new methods for Keraon (kept in my own notes for now). Also add simulated run option for both, based on
the references (see PreviousModels tree).
"""

# TODO: ship with binary for use in LuCaPs.

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sys import stdout
from keraon_helpers import *
from keraon_plotters import *
from scipy.special import logsumexp


def ctdpheno(ref_df, df):
    """
    Calculate TFX-shifted multivariate group identity relative log likelihoods (RLLs).

    Parameters:
    ref_df (DataFrame): Reference DataFrame with 'Subtype' column and feature columns.
    df (DataFrame): DataFrame with 'TFX' column and feature columns. Index should be sample identifiers.

    Returns:
    DataFrame: DataFrame with predictions and RLLs for each sample.
    """
    print('Calculating TFX-shifted multivariate group identity RLLs...')

    samples = df.index.tolist()
    subtypes = ref_df['Subtype'].unique().tolist()
    subtypes.remove('Healthy')

    # Initialize DataFrame for storing predictions
    cols = subtypes + ['TFX', 'TFX_shifted', 'Prediction']
    predictions = pd.DataFrame(index=df.index, columns=cols)

    # Set the correct data types for the columns
    predictions[subtypes] = predictions[subtypes].astype(float)
    predictions[['TFX', 'TFX_shifted']] = predictions[['TFX', 'TFX_shifted']].astype(float)
    predictions['Prediction'] = predictions['Prediction'].astype(str)

    # Calculate RLLs for each sample
    for i, sample in enumerate(samples, start=1):
        print(f'\rRunning samples | completed: [{i}/{len(samples)}]', end='')

        tfx = df.loc[sample, 'TFX']
        feature_vals = df.loc[sample].drop('TFX').to_numpy()
        # Calculate mean and covariance for 'Healthy' subtype
        healthy_data = ref_df[ref_df['Subtype'] == 'Healthy'].iloc[:, 1:]
        mu_healthy = healthy_data.mean().to_numpy()
        cov_healthy = np.cov(healthy_data.to_numpy(), rowvar=False)
        # Calculate means for other subtypes
        mu_subs = [ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:].mean().to_numpy() for subtype in subtypes]
        # Calculate RLLs
        log_likelihoods = calculate_log_likelihoods(tfx, feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes)

        # If all log likelihoods are -inf, optimize 'TFX' to maximize total log likelihood
        if np.all(np.isinf(log_likelihoods)) or np.isclose(logsumexp(log_likelihoods), 0):
            tfx_shifted = optimize_tfx(feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes)
            log_likelihoods = calculate_log_likelihoods(tfx_shifted, feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes)
        else:
            tfx_shifted = tfx

        # Calculate weights and update predictions DataFrame
        update_predictions(predictions, sample, tfx, tfx_shifted, log_likelihoods, subtypes)

    print('\nFinished.')
    return predictions


def keraon(ref_df, df):
    """
    Transform the feature space of a dataset into a new basis defined by the mean vectors of different subtypes.

    This function performs a transformation of the feature space of a given dataset into a new basis. 
    The new basis vectors are defined by the mean vectors of different subtypes in the reference dataset. 
    The transformation is performed by subtracting the mean vector of the 'Healthy' subtype from the mean vectors 
    of the other subtypes, and then orthogonalizing the resulting vectors using the Gram-Schmidt process. 
    This results in a set of orthogonal basis vectors, each pointing in the direction of a different subtype.

    After transforming the feature space, the function calculates the components of each sample's feature vector 
    in this new basis. These components represent the sample's "fraction" of each subtype. The function also 
    calculates the "burden" of each subtype, which is the product of the sample's tumor fraction (TFX) and its 
    fraction of the subtype.

    Parameters:
    ref_df (DataFrame): The reference dataframe containing the mean feature values for each subtype.
    df (DataFrame): The input dataframe containing the feature values for each sample.

    Returns:
    DataFrame: A dataframe containing the TFX, fraction and burden of each subtype for each sample, 
               and the region of the feature space where the sample is located.
    """
    # Get initial reference values
    features = list(ref_df.iloc[:, 1:].columns)  # for ordering
    subtypes = list(ref_df['Subtype'].unique())
    mean_vectors = []
    hd_idx = subtypes.index('Healthy')

    # Get anchor/ref vector of means for each subtype
    for subtype in subtypes:
        df_temp = ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:]
        mean_vectors.append(df_temp.mean(axis=0).to_numpy())
        # mean_vectors.append(df_temp.median(axis=0).to_numpy())

    # Check if TFX column exists
    if 'TFX' not in df:
        print('A column TFX with sample tumor fraction estimates is missing from the input dataframe. Exiting.')
        exit()

    # Initialize output dataframe
    out_cols_basis = ['TFX'] + [subtype + '_fraction' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_fraction', 'Healthy_fraction'] + [subtype + '_burden' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_burden', 'FS_Region']
    basis_predictions = pd.DataFrame(index=df.index, columns=out_cols_basis)

    # Set the correct data types for the columns
    basis_predictions[['TFX'] + [subtype + '_fraction' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_fraction', 'Healthy_fraction'] + [subtype + '_burden' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_burden']] = basis_predictions[['TFX'] + [subtype + '_fraction' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_fraction', 'Healthy_fraction'] + [subtype + '_burden' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_burden']].astype(float)
    basis_predictions['FS_Region'] = basis_predictions['FS_Region'].astype(str)

    # Calculate basis vectors
    basis_vectors = [mean_vectors[i] - mean_vectors[hd_idx] for i in range(len(subtypes)) if i != hd_idx]
    basis = gram_schmidt(basis_vectors)  # unit vectors in the direction of each subtype

    # Process each sample
    samples = list(df.index)
    n_complete, n_samples = 0, len(samples)
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        raw_vector = np.asarray(df.loc[sample, features].to_numpy(), dtype=float)
        hd_sample = raw_vector - mean_vectors[hd_idx]  # directional vector from HD to sample
        sample_vector, orthogonal_vector = transform_to_basis(hd_sample, basis)

        # Determine FS region and adjust sample vector if necessary
        if all(sample_vector > 0):
            fs_region = 'Simplex'
        elif all(sample_vector < 0):
            fs_region = 'Contra-Simplex'
            # Project through the origin by negating the vector
            sample_vector = -sample_vector
            # Scale the vector to match TFX
            sample_vector = (sample_vector / np.linalg.norm(sample_vector)) * tfx
        else:
            fs_region = 'Outer-Simplex'
            sample_vector = np.maximum(sample_vector, 0)

        # Calculate fractions and burdens
        projected_length = np.linalg.norm(sample_vector)
        orthogonal_length = np.linalg.norm(orthogonal_vector)
        comp_fraction = np.append(sample_vector / projected_length, orthogonal_length)
        comp_fraction = comp_fraction / np.sum(comp_fraction)
        basis_vals = [tfx] + [tfx * val for val in comp_fraction] + [1 - tfx] + comp_fraction.tolist() + [fs_region]
        basis_predictions.loc[sample] = basis_vals

        # Update progress
        n_complete += 1
        stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
        stdout.flush()

    print('\nFinished\n')
    return basis_predictions


def main():
    parser = argparse.ArgumentParser(description='\n### Keraon.py ###')

    # Required arguments
    parser.add_argument('-i', '--input', required=True, 
                        help='A tidy-form, .tsv feature matrix with test samples. Should contain 4 columns: "sample", "site", "feature", and "value".')
    parser.add_argument('-t', '--tfx', required=True, 
                        help='.tsv file with test sample names and tumor fractions. If a third column with "true" subtypes/categories is passed, additional validation will be performed.')
    parser.add_argument('-r', '--reference', nargs='*', required=True, 
                        help='One or more tidy-form, .tsv feature matrices. Should contain 4 columns: "sample", "site", "feature", and "value".')
    parser.add_argument('-k', '--key', required=True, 
                        help='.tsv file with sample names and subtypes/categories. One subtype must be "Healthy".')

    # Optional arguments
    parser.add_argument('-d', '--doi', default='NEPC', 
                        help='Disease/subtype of interest for plotting and calculating ROCs - must be present in key.')
    parser.add_argument('-x', '--thresholds', nargs=2, type=float, default=(0.5, 0.0311), 
                        help='Tuple containing thresholds for calling the disease of interest.')
    parser.add_argument('-f', '--features', default=None, 
                        help='File with a list of site_feature combinations to restrict to. Sites and features should be separated by an underscore.')
    parser.add_argument('-s', '--svm_selection', action='store_false', default=True, 
                        help='Flag indicating whether to TURN OFF SVM feature selection method.')
    parser.add_argument('-p', '--palette', default=None, 
                        help='tsv file with matched categories/samples and HEX color codes. Subtype/category names must match those passed with the -t and -k options.')

    args = parser.parse_args()

    # Paths
    input_path, tfx_path, ref_path, key_path = args.input, args.tfx, args.reference, args.key
    doi, thresholds, perform_svm = args.doi, args.thresholds, args.svm_selection
    features_path, palette_path = args.features, args.palette

    print('\n### Welcome to Keraon ###\nLoading data . . .')

    # Load palette
    palette = {row[0]: row[1] for row in pd.read_table(palette_path, sep='\t', header=None).itertuples(False, None)} if palette_path else None

    # Load features
    if features_path is not None:
        with open(features_path) as f:
            features = f.read().splitlines()
    else:
        features = None

    # Create necessary directories
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/FeatureSpace/', exist_ok=True)
    os.makedirs('results/ctdPheno/', exist_ok=True)
    os.makedirs('results/Keraon/', exist_ok=True)

    # prepare test dataframe:
    test_df = pd.read_table(input_path, sep='\t', index_col=0, header=0)
    test_df['site'] = test_df['site'].str.replace('_', '-')  # ensure no _ in site names
    test_df['feature'] = test_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
    test_df['cols'] = test_df['site'].astype(str) + '_' + test_df['feature']
    test_df = test_df.pivot_table(index=[test_df.index.values], columns=['cols'], values='value')
    test_labels = pd.read_table(tfx_path, sep='\t', index_col=0, header=None)
    if len(test_labels.columns) == 2:  # truth values supplied
        if not (test_labels[2]=='Unknown').all():
            ordering = test_labels.index
            ordering.names = ['Sample']
        else:
            ordering = None
        truth_vals = test_labels.iloc[:, 1:]
        truth_vals.columns = ['Truth']
        test_labels = test_labels.drop(2, axis=1)
    else:
        ordering = None
        truth_vals = None
    test_labels.columns = ['TFX']

    # prepare reference dataframe:
    override_data_generation = True
    direct = 'results/FeatureSpace/'
    ref_binary = 'results/FeatureSpace/cleaned_anchor_reference.pkl'
    if not os.path.isfile(ref_binary) or override_data_generation:
        print('Creating reference dataframe . . .')
        ref_dfs = [pd.read_table(path, sep='\t', index_col=0, header=0) for path in ref_path]
        ref_df = pd.concat(ref_dfs)
        ref_df['site'] = ref_df['site'].str.replace('_', '-')  # ensure no _ in site names
        ref_df['feature'] = ref_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
        ref_df['cols'] = ref_df['site'].astype(str) + '_' + ref_df['feature']
        if features is not None:
            ref_df = ref_df[ref_df['cols'].isin(features)]
        ref_df = ref_df.pivot_table(index=[ref_df.index.values], columns=['cols'], values='value')
        ref_labels = pd.read_table(key_path, sep='\t', index_col=0, header=0, names=['Subtype'])

        # Get the common columns in both dataframes
        common_columns = ref_df.columns.intersection(test_df.columns)
        # Get columns with no NaNs in either dataframe
        no_nan_columns = ref_df.columns[ref_df.notna().all()].intersection(test_df.columns[test_df.notna().all()])
        # Get the intersection of common_columns and no_nan_columns
        final_columns = common_columns.intersection(no_nan_columns)
        # Restrict both dataframes to the final_columns
        ref_df = ref_df[final_columns]
        test_df = test_df[final_columns]

        # filter features
        print('Filtering features . . .')
        # Get the intersection of ref_df's index and ref_labels's index
        common_index = ref_df.index.intersection(ref_labels.index)
        # Get the labels that are in ref_labels but not in ref_df
        missing_labels = ref_labels.index.difference(ref_df.index)
        # Print the missing labels
        print(f"Missing labels: {missing_labels.tolist()}")
        # Filter ref_labels to only include labels that are in ref_df
        ref_labels = ref_labels.loc[common_index]
        # Now you can safely index ref_df with ref_labels's index
        ref_df = ref_df.loc[ref_labels.index]
        ref_df = ref_df.dropna(axis=1, how='any') # drop any features with missing values
        drop_features = ['mean-depth', 'np-amplitude', 'var-ratio', 'fragment-entropy', 'fragment-diversity', 'plus-minus-ratio',
                         'central-loc', 'minus-one-pos', 'plus-one-pos'] # these Triton features are liable to depth bias / outliers or are intended for larger regions
        drop_regex = '|'.join(drop_features)  # creates a regex that matches any string in drop_features
        ref_df = ref_df.drop(columns=ref_df.filter(regex=drop_regex).columns)

        if perform_svm:
            plot_pca(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct, palette, "PCA-1_initial")
            if len(ref_df.columns) > 10:  # skip this step if few features
                ref_df = norm_exp(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct)[0] # only use nominally "normal" features
                plot_pca(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct, palette, "PCA-2_post-norm-restrict")
            # scale features:
            ref_df, min_dict, range_dict = minmax_standardize(pd.merge(ref_labels, ref_df, left_index=True, right_index=True))
            ref_df = maximal_simplex_volume(ref_df)
            df_train = pd.merge(ref_labels, ref_df, left_index=True, right_index=True)
            plot_pca(df_train, direct, palette, "PCA-3_post-MSV")
            df_train.to_csv(direct + 'final-features.tsv', sep="\t")
        else:
            ref_df, min_dict, range_dict = minmax_standardize(pd.merge(ref_labels, ref_df, left_index=True, right_index=True))
            df_train = ref_df
            df_train.to_csv(direct + 'final-features.tsv', sep="\t")
        print('Finished. Saving reference dataframe . . .')
        with open(ref_binary, 'wb') as f:
            pickle.dump([df_train, min_dict, range_dict], f)
    else:
        print('Loading reference dataframe . . .')
        with open(ref_binary, 'rb') as f:
            df_train, min_dict, range_dict = pickle.load(f)

    # restrict test_df features to those identified in ref_df and anchor feature processing
    test_df = test_df[df_train.drop('Subtype', axis=1).columns]
    # scale features identically to ref_df for consistency in the same space
    test_df = minmax_standardize(test_df, min_dict=min_dict, range_dict=range_dict)[0]
    df_test = pd.merge(test_labels, test_df, left_index=True, right_index=True, how='inner')

    if truth_vals is not None:
        plot_pca(df_train, direct, palette, "PCA-4_post-MSV_wSamples", post_df=pd.merge(truth_vals, df_test, left_index=True, right_index=True))
    else:
        plot_pca(df_train, direct, palette, "PCA-4_post-MSV_wSamples", post_df=df_test)

    # run ctdPheno
    print("\n### Running experiment: classification (ctdPheno)")
    direct = 'results/ctdPheno/'
    subtypes = list(df_train['Subtype'].unique())
    subtypes.remove('Healthy')

    ctdpheno_preds = ctdpheno(df_train, df_test)
    if truth_vals is not None:
        ctdpheno_preds = pd.merge(truth_vals, ctdpheno_preds, left_index=True, right_index=True, how='right')
    if ordering is not None:
        ctdpheno_preds = ctdpheno_preds.reindex(ordering)
    elif truth_vals is not None:
        ctdpheno_preds = ctdpheno_preds.sort_values(['Truth', 'TFX'], ascending=[True, False])
    else:
        ctdpheno_preds = ctdpheno_preds.sort_values(['TFX'], ascending=False)
    # N.B. please remove truth values which are heterogenous mixtures (e.g. MIX) for a proper ROC, as seen below
    if not (ctdpheno_preds['Truth'] == 'Unknown').all():
        threshold = plot_roc(ctdpheno_preds[ctdpheno_preds['Truth'] != 'MIX'], direct, doi)[3]
    else:
        threshold = thresholds[0]
    # Get the unique values in the 'Truth' column
    truths = ctdpheno_preds['Truth'].unique()
    ctdpheno_preds.to_csv(direct + 'ctdPheno_class-predictions.tsv', sep="\t")
    plot_range = [min(ctdpheno_preds[doi]), max(ctdpheno_preds[doi])]
    # Loop over the unique values in the 'Truth' column
    for truth in truths:
        # Subset the data based on the 'Truth' column
        subset = ctdpheno_preds[ctdpheno_preds['Truth'] == truth]
        # Call plot_ctdpheno for the subset of the data
        plot_ctdpheno(subset, direct, doi, threshold, label=truth, plot_range=plot_range)

    # run Keraon
    print("### Running experiment: mixture estimation (Keraon)")
    direct = 'results/Keraon/'
    keraon_preds = keraon(df_train, df_test)
    if truth_vals is not None:
        keraon_preds = pd.merge(truth_vals, keraon_preds, left_index=True, right_index=True)
    if ordering is not None:
        keraon_preds = keraon_preds.reindex(ordering)
    elif truth_vals is not None:
        keraon_preds = keraon_preds.sort_values(['Truth', 'TFX'], ascending=[True, False])
    else:
        keraon_preds = keraon_preds.sort_values(['TFX'], ascending=False)
    keraon_preds.to_csv(direct + 'Keraon_mixture-predictions.tsv', sep="\t")
    # keraon_preds = keraon_preds[keraon_preds['FS_Region'] == 'Contra-Simplex']
    keraon_preds = keraon_preds[~keraon_preds.index.str.contains('CRPC')]
    if not (keraon_preds['Truth'] == 'Unknown').all():
        threshold = plot_roc(keraon_preds[keraon_preds['Truth'] != 'MIX'], direct, doi + '_fraction')[3]
    else:
        threshold = thresholds[1]
    # Get the unique values in the 'Truth' column
    truths = keraon_preds['Truth'].unique()
    # Loop over the unique values in the 'Truth' column
    for truth in truths:
        # Subset the data based on the 'Truth' column
        subset = keraon_preds[keraon_preds['Truth'] == truth]
        # Call plot_ctdpheno for the subset of the data
        plot_keraon(subset, direct, doi + '_fraction', threshold, label=truth, palette=palette)
        # N.B. please remove truth values which are heterogenous mixtures (e.g. MIX) for a proper ROC, as seen below
    print('Now exiting.')


if __name__ == "__main__":
    main()

#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.1, 11/7/2023

"""
This newest iteration of Keraon aims to combine ctdPheno and Keraon together, to give dual output predictions
for any appropriately formatted datasets. It will also be made more user-friendly in true script form, with
exmaple runs saved in the accompanying EXAMPLE_RUNS.txt file. I plan to test new feature selection methods, as
well as new methods for Keraon (kept in my own notes for now). Also add simulated run option for both, based on
the references (see PreviousModels tree).
"""

# TODO: standard scale fatures?
# TODO: add a specific site-feature option, replacing sites/features
# TODO: make feature finding optional
# TODO: adjust how ctdPheno scores are reported? 0-1?

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sys import stdout
from keraon_helpers import *
from keraon_plotters import *
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

def ctdpheno(ref_df, df):
    print('Calculating TFX-shifted multi-variate group identity RLLs . . . ')
    samples = list(df.index)
    subtypes = list(ref_df['Subtype'].unique())
    subtypes.remove('Healthy')
    cols = [subtype for subtype in subtypes] + ['TFX', 'TFX_shifted', 'Prediction']
    predictions = pd.DataFrame(0, index=df.index, columns=cols)

    n_complete, n_samples = 0, len(samples)
    stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
    stdout.flush()
    
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        feature_vals = df.drop(columns='TFX').loc[sample, :].to_numpy()
        mu_healthy = ref_df[ref_df['Subtype'] == 'Healthy'].iloc[:, 1:].mean(axis=0).to_numpy()
        cov_healthy = np.cov(ref_df[ref_df['Subtype'] == 'Healthy'].iloc[:, 1:].to_numpy(), rowvar=False)
        mu_subs = [ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:].mean(axis=0).to_numpy() for subtype in subtypes]
        # cov_subs = [np.cov(ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:].to_numpy(), rowvar=False) for subtype in subtypes]
        
        def get_mvg_lls(tfx):
            if isinstance(tfx, np.ndarray): # for minimize function
                tfx = tfx[0]
            log_likelihoods = []
            for subtype in subtypes:  # this does not include "Healthy"
                subtype_idx = subtypes.index(subtype)
                # caluclate the mean of the mixture distirbution
                mu_mixture = tfx * mu_subs[subtype_idx] + (1 - tfx) * mu_healthy

                # Calculate mixture covariance
                """
                Due to sample imbalances, using the true mixed covariance can cause highly inaccurate results. It has been left implemented below,
                but commented out, for reference. Likewise there is a commented out regularization step, which can be used to tune results.
                Currently covariance matrices are set to the identitiy, which assumes independence of features in each group.
                """
                # cov_mixture = tfx * (cov_subs[subtype_idx] + np.outer(mu_subs[subtype_idx] - mu_mixture, mu_subs[subtype_idx] - mu_mixture)) + \
                #             (1 - tfx) * (cov_healthy + np.outer(mu_healthy - mu_mixture, mu_healthy - mu_mixture))
                # cov_mixture = cov_mixture + alpha * np.eye(cov_mixture.shape[0])  # Add regularization
                # cov_mixture = cov_mixture / np.linalg.det(cov_mixture)
                cov_mixture = np.eye(cov_healthy.shape[0])

                # Calculate log likelihood
                log_likelihood = multivariate_normal.logpdf(feature_vals, mean=mu_mixture, cov=cov_mixture)
                log_likelihoods.append(log_likelihood)
            return log_likelihoods

        log_likelihoods = get_mvg_lls(tfx)
        log_total = logsumexp(log_likelihoods)
        
        # If all log likelihoods are -inf, optimize 'TFX' to maximize total log likelihood
        if all(ll == -np.inf for ll in log_likelihoods) or log_total == 0:
            def neg_total_log_likelihood(tfx):
                log_likelihoods = get_mvg_lls(tfx)
                return -sum(np.exp(log_likelihoods))

            best_tfx = 0
            best_ll = float('inf')

            for try_tfx in np.arange(0, 1.001, 0.001):
                ll = neg_total_log_likelihood(try_tfx)
                if ll < best_ll:
                    best_ll = ll
                    best_tfx = try_tfx

            tfx_shifted = best_tfx
            log_likelihoods = get_mvg_lls(tfx_shifted)
            log_total = logsumexp(log_likelihoods)
        else:
            tfx_shifted = tfx

        if log_total == 0:
            print(' | WARNING: total log likelihood == 0 for sample ' + sample + '. Setting all weights to 0.')
            weights = [0] * len(log_likelihoods)
            log_total = 1
            tfx_shifted = np.nan

        # Calculate relative log likelihoods
        weights = [np.exp(ll - log_total) for ll in log_likelihoods]
        predictions.loc[sample, 'TFX'] = tfx
        predictions.loc[sample, 'TFX_shifted'] = tfx_shifted

        for subtype in subtypes:
            predictions.loc[sample, subtype] = np.round(weights[subtypes.index(subtype)], 4)
            if predictions.loc[sample, subtype] > 0.5:
                predictions.loc[sample, 'Prediction'] = subtypes[subtypes.index(subtype)]

        # Replace 0s with 'NoSolution' in the 'Prediction' column
        predictions['Prediction'] = predictions['Prediction'].replace(0, 'NoSolution')

        n_complete += 1
        stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
        stdout.flush()
    
    print('\nFinished.\n')
    return predictions


def keraon(ref_df, df):
    """
    """
    print('Caluclating subtype-basis components . . . ')
    # get initial reference values:
    features = list(ref_df.iloc[:, 1:].columns)  # simply for ordering
    fd = len(features)  # dimensionality of the feature space
    subtypes = list(ref_df['Subtype'].unique())
    sd = len(subtypes)  # dimensionality of the subtypes
    print('Feature dimensions F = ' + str(fd))
    print('Class dimensions (O(simplex) - 1) K = ' + str(sd))
    mean_vectors = []
    hd_idx = subtypes.index('Healthy')
    # get anchor/ref vector of means for each subtype:
    for subtype in subtypes:
        df_temp = ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:]
        mean_vectors.append(df_temp.mean(axis=0).to_numpy())
    if 'TFX' not in df:
        print('A column TFX with sample tumor fraction estimates is missing from the input dataframe. Exiting.')
        exit()

    samples = list(df.index)
    out_cols_basis = ['TFX'] + [subtype + '_fraction' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_fraction', 'Healthy_fraction'] + [subtype + '_burden' for subtype in subtypes if subtype != 'Healthy']\
                    + ['OffTarget_burden', 'FS_Region']
    print('Output columns: ' + str(out_cols_basis))
    basis_predictions = pd.DataFrame(0, index=df.index, columns=out_cols_basis)
    basis_vectors = [mean_vectors[i] - mean_vectors[hd_idx] for i in range(len(subtypes)) if i != hd_idx]

    def gram_schmidt(vectors):
        basis = []
        for v in vectors:
            w = v - sum(np.dot(v,b)*b for b in basis)
            if (w > 1e-10).any():  
                basis.append(w/np.linalg.norm(w))
            else:
                basis.append(np.zeros_like(w))
        return np.array(basis)

    def transform_to_basis(vector, basis):
        # Project the original vector onto the new basis
        projected_vector = np.dot(vector, basis.T)

        # Project the transformed vector back into the original space
        back_projected_vector = np.dot(projected_vector, basis)

        # Calculate the difference between the original vector and its projection
        difference_vector = vector - back_projected_vector

        # The norm (length) of the difference vector represents the distance
        # the original vector is traveling through the dimensions not represented by the new basis
        return projected_vector, difference_vector
    
    basis = gram_schmidt(basis_vectors)  # AKA unit vectors in the direction of each subtype

    n_complete, n_samples = 0, len(samples)
    stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
    stdout.flush()

    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        raw_vector = np.asarray(df.loc[sample, features].to_numpy(), dtype=float)
        hd_sample = raw_vector - mean_vectors[hd_idx]  # directional vector from HD to sample
        sample_vector, orthogonal_vector = transform_to_basis(hd_sample, basis)

        if all(sample_vector > 0):
            fs_region = 'Simplex'
        elif all(sample_vector < 0):
            fs_region = 'Contra-Simplex'
            # TODO: print warning and no solution, OR project through the origin to match TFX
        else:
            fs_region = 'Outer-Simplex'
            sample_vector = np.maximum(sample_vector, 0)
            
        projected_length = np.linalg.norm(sample_vector)
        orthogonal_length = np.linalg.norm(orthogonal_vector)
        comp_fraction = np.append(sample_vector / projected_length, orthogonal_length)
        comp_fraction = comp_fraction / np.sum(comp_fraction)
        basis_vals = [tfx] + [tfx * val for val in comp_fraction] + [1 - tfx] + comp_fraction.tolist() + [fs_region]
        basis_predictions.loc[sample] = basis_vals
        n_complete += 1
        stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
        stdout.flush()

    print('\nFinished\n')
    return basis_predictions


def main():
    parser = argparse.ArgumentParser(description='\n### Keraon.py ###')
    parser.add_argument('-i', '--input', help='A tidy-form, .tsv feature matrix with test samples, containing 4 '
                                             'columns with the following header: "sample", "site", "feature", and '
                                             '"value". Outputs from Triton (TritonCompositeFM.tsv directly) or Griffin '
                                             'in this format are recommended.', required=True)
    parser.add_argument('-t', '--tfx', help='.tsv file containing matched test sample names (column 1) and '
                                            'tumor fractions (column2), without a header, corresponding to the '
                                            '"samples" column(s) input for testing. If a third, optional column containing '
                                            '"true" subtypes/categories is passed, additional validation will be performed '
                                            'and downstream plotting will use the ORDERING of the sample', required=True)
    parser.add_argument('-r', '--reference', help='One or more tidy-form, .tsv feature matrices containing 4 columns '
                                                  'with the following header: "sample", "site", "feature", and '
                                                  '"value". Outputs from Triton (TritonCompositeFM.tsv directly) or '
                                                  'Griffin in this format are recommended.', nargs='*', required=True)
    parser.add_argument('-k', '--key', help='.tsv file containing matched sample names (column 1) and '
                                            'subtypes/categories (column2), without a header, corresponding to the '
                                            '"samples" column(s) input as reference. One included subtype must be '
                                            '"Healthy" (healthy/normal references).', required=True)
    parser.add_argument('-d', '--doi', help='Disease/subtype of interest (string, found in keys) for plotting and caluclating ROCs', 
                                             required=False, default='NEPC')
    parser.add_argument('-x', '--thresholds', help='Tuple containing thresholds for calling the disease of interest for ctdPheno and Keraon, respectively.', 
                                             nargs=2, type=lambda x: (float(x[0]), float(x[1])), default=(0.5, 0.0311))
    parser.add_argument('-s', '--sites', help='File containing a list (row-wise) of sites to restrict to. This file should '
                                              'NOT contain a header. DEFAULT = None (use all available sites).',
                        required=False, default=None)
    parser.add_argument('-f', '--features', help='File containing a list (row-wise) of features to restrict to. This file '
                                                 'should NOT contain a header. '
                                                 'DEFAULT = None (use all available sites).',
                        required=False, default=None)
    parser.add_argument('-p', '--palette', help='tsv file containing matched categories/samples (column 1) and HEX '
                                                'color codes (column 2, e.g. #0077BB) to specify what color to use '
                                                'for subtypes/categories. Subtype/category names must exactly match '
                                                'category names passed with the -t and -k options. Will error if '
                                                'inputs/categories include labels not present in palette. This file '
                                                'should NOT contain a header. DEFAULT = None (use Seaborn default colors).',
                        required=False, default=None)

    args = parser.parse_args()
    input_path = args.input
    tfx_path = args.tfx
    ref_path = args.reference
    key_path = args.key
    doi = args.doi
    thresholds = args.thresholds
    sites_path = args.sites
    features_path = args.features
    palette_path = args.palette

    print('\n### Welcome to Keraon ###\nLoading data . . .')

    if palette_path is not None:
        palette_path = pd.read_table(palette_path, sep='\t', header=None)
        palette = dict(palette_path.itertuples(False, None))
    else:
        palette = None

    if sites_path is not None:
        with open(sites_path) as f:
            sites = f.read().splitlines()
    else:
        sites = None

    if features_path is not None:
        with open(features_path) as f:
            features = f.read().splitlines()
    else:
        features = None

    if not os.path.exists('results/'): os.makedirs('results/')
    if not os.path.exists('results/FeatureSpace/'): os.makedirs('results/FeatureSpace/')
    if not os.path.exists('results/ctdPheno/'): os.makedirs('results/ctdPheno/')
    if not os.path.exists('results/Keraon/'): os.makedirs('results/Keraon/')

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

    # HA_G lab local exceptions:
    if 'Berchuck' in input_path: # single use case for Berchuck data set
        test_df.index = test_df.index.str.split('_').str[0]
    if 'TAN' in input_path: # single use case for TAN data set
        # Load the renaming file
        rename_df = pd.read_csv('keys/TAN-WGS_naming.tsv', sep='\t', header=None, names=['old', 'new'])
        # Create a dictionary mapping old names to new names
        rename_dict = rename_df.set_index('old')['new'].to_dict()
        # Rename the indices in test_df
        test_df = test_df.rename(index=rename_dict)
        # Restrict to those represented in the file
        test_df = test_df[test_df.index.isin(rename_df['new'])]

    # prepare reference dataframe:
    override_data_generation = False
    direct = 'results/FeatureSpace/'
    ref_binary = 'results/FeatureSpace/cleaned_anchor_reference.pkl'
    ref_binary = 'LuCaP-ATAC_cleaned_anchor_reference.pkl'
    if not os.path.isfile(ref_binary) or override_data_generation:
        print('Creating reference dataframe . . .')
        ref_dfs = [pd.read_table(path, sep='\t', index_col=0, header=0) for path in ref_path]
        ref_df = pd.concat(ref_dfs)
        if sites is not None:
            ref_df = ref_df[ref_df['site'].isin(sites)]
        if features is not None:
            ref_df = ref_df[ref_df['feature'].isin(features)]
        # else:
        #     restrict_features = ['central-depth', 'central-diversity', 'np-score', 'np-period', 'fragment-median']
        #     ref_df = ref_df[ref_df['feature'].isin(restrict_features)]
        ref_df['site'] = ref_df['site'].str.replace('_', '-')  # ensure no _ in site names
        ref_df['feature'] = ref_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
        ref_df['cols'] = ref_df['site'].astype(str) + '_' + ref_df['feature']
        ref_df = ref_df.pivot_table(index=[ref_df.index.values], columns=['cols'], values='value')
        ref_labels = pd.read_table(key_path, sep='\t', index_col=0, header=0, names=['Subtype'])

        # filter features
        print('Filtering features . . .')
        ref_df = ref_df.loc[ref_labels.index]
        ref_df = ref_df.dropna(axis=1, how='any') # drop any features with missing values
        drop_features = ['mean-depth', 'np-amplitude', 'var-ratio', 'fragment-entropy', 'fragment-diversity', 'plus-minus-ratio'] # these features are liable to depth bias / outliers
        drop_regex = '|'.join(drop_features)  # creates a regex that matches any string in drop_features
        ref_df = ref_df.drop(columns=ref_df.filter(regex=drop_regex).columns)

        # HA_G lab local exceptions:
        if 'ATAC' in input_path and 'NEPC' in truth_vals['Truth'].unique():
            ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='PSMA')))]
            ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='top1000')))]
        else: # testing Luminal/Basal
            ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='AD-Exclusive')))]
            ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='NE-Exclusive')))]
            ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='sig-profiles')))]

        plot_pca(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct, palette, "PCA-1_initial")
        if len(ref_df.columns) > 50 and not ref_binary == 'LuCaP-ATAC_PSMA_cleaned_anchor_reference.pkl':  # skip this step if few features
            ref_df = norm_exp(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct)[0] # only use nominally "normal" features
            plot_pca(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct, palette, "PCA-2_post-norm-restrict")
        # scale features:
        ref_df, min_dict, range_dict = minmax_standardize(pd.merge(ref_labels, ref_df, left_index=True, right_index=True))
        ref_df = maximal_simplex_volume(ref_df)
        df_train = pd.merge(ref_labels, ref_df, left_index=True, right_index=True)
        plot_pca(df_train, direct, palette, "PCA-3_post-MSV")
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
    # Loop over the unique values in the 'Truth' column
    for truth in truths:
        # Subset the data based on the 'Truth' column
        subset = ctdpheno_preds[ctdpheno_preds['Truth'] == truth]
        # Call plot_ctdpheno for the subset of the data
        plot_ctdpheno(subset, direct, doi, threshold, label=truth)

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

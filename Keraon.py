#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v3.0, 3/13/2025

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sys import stdout
from scipy.stats import multivariate_normal
from scipy.special import softmax
from utils.keraon_utils import *
from utils.keraon_helpers import *
from utils.keraon_plotters import *

# These Triton features are liable to depth bias / outliers or are intended for larger regions (e.g. gene bodies, not TFBS sites)
drop_features = ['mean-depth', 'fragment-diversity', 'fragment-entropy', 'var-ratio', # highly biased by (or are measurements of) depth
                 'central-loc', 'plus-one-pos', 'minus-one-pos', 'plus-minus-ratio']  # based on peak-calling, which may struggle with low-coverage regions
limit_features = ['central-depth', 'central-diversity']
# limit_features = None

# feature-specific scaling to make normal-like (applies to Triton features only)
scaling_methods = {'fragment-diversity': lambda x: np.log(x+10e-6),
                   'plus-minus-ratio': lambda x: np.log(x+10e-6),
                   'np-period': lambda x: np.log(x+10e-6),
                   'mean-depth': lambda x: np.log(x+10e-6)}


def keraon(ref_df, df):
    """
    Transform the feature space of a dataset into a new basis defined by the mean vectors of different subtypes.

    This function performs a transformation of the feature space of a given dataset into a new basis. 
    The new basis vectors are defined by the mean vectors of different subtypes in the reference dataset. 
    The transformation is performed by subtracting the mean vector of the 'Healthy' subtype from the mean vectors 
    of the other subtypes, and then orthogonalizing the resulting vectors using the Gram-Schmidt process. 
    This results in a set of orthogonal basis vectors, each pointing in the direction of a different subtype.
    If Healthy samples are not all contained within the final (positive) basis, the function will shift the 'Healthy' center/origin
    to ensure that all healthy reference samples are within the simplex.

    After transforming the feature space, the function calculates the components of each sample's feature vector 
    in this new basis. These components represent the sample's "burden" of each subtype. The function also 
    calculates the "fraction" of each subtype, which is the product of the sample's tumor fraction (TFX) and its 
    subtype burden.

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
    raw_basis = gram_schmidt(basis_vectors)  # unit vectors in the direction of each subtype
    basis = raw_basis

    # Shift the 'Healthy' center/origin to ensure all healthy reference samples are within the simplex
    min_coeff = 0.0
    for idx in ref_df.index[ref_df['Subtype'] == 'Healthy']:
        y = (ref_df.loc[idx, features].values - mean_vectors[hd_idx])
        coeffs, _ = transform_to_basis(y, raw_basis)
        min_coeff = min(min_coeff, coeffs.min())  # most negative value

    alpha_global = max(0.0, -min_coeff)

    if alpha_global > 0.0:
        shift_vec            = alpha_global * np.sum(raw_basis, axis=0)
        mean_vectors[hd_idx]    = mean_vectors[hd_idx] - shift_vec

    shifted_basis = [mean_vectors[i] - mean_vectors[hd_idx] for i in range(len(subtypes)) if i != hd_idx]
    basis = gram_schmidt(shifted_basis) # final orthonormal basis

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
    cols = subtypes + ['TFX', 'Prediction']
    predictions = pd.DataFrame(index=df.index, columns=cols)

    # Set the correct data types for the columns
    predictions[subtypes] = predictions[subtypes].astype(float)
    predictions['TFX'] = predictions['TFX'].astype(float)
    predictions['Prediction'] = predictions['Prediction'].astype(str)

    def calculate_log_likelihoods(tfx, feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes):
        """
        Calculate log likelihoods for given TFX value and feature values.
        """
        log_likelihoods = []
        for subtype in subtypes:
            mu_mixture = tfx * mu_subs[subtypes.index(subtype)] + (1 - tfx) * mu_healthy
            cov_subs = [np.cov(ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:].to_numpy(), rowvar=False) for subtype in subtypes]
            cov_mixture = tfx * cov_subs[subtypes.index(subtype)] + (1 - tfx) * cov_healthy
            reg_factor = 1e-6 * np.trace(cov_mixture) / cov_mixture.shape[0]
            cov_mixture += np.eye(cov_mixture.shape[0]) * reg_factor
            # cov_mixture = np.eye(cov_healthy.shape[0])
            log_likelihood = multivariate_normal.logpdf(feature_vals, mean=mu_mixture, cov=cov_mixture)
            log_likelihoods.append(log_likelihood)
        return log_likelihoods

    def update_predictions(predictions, sample, tfx, log_likelihoods, subtypes):
        """
        Update predictions DataFrame with calculated values.
        """
        weights = softmax(log_likelihoods)

        predictions.loc[sample, 'TFX'] = tfx
        max_weight = 0
        max_subtype = 'NoSolution'
        for subtype in subtypes:
            weight = np.round(weights[subtypes.index(subtype)], 4)
            predictions.loc[sample, subtype] = weight
            if weight > max_weight:
                max_weight = weight
                max_subtype = subtype

        predictions.loc[sample, 'Prediction'] = max_subtype

    # Process each sample
    samples = list(df.index)
    n_complete, n_samples = 0, len(samples)
    for sample in samples:
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

        # Calculate weights and update predictions DataFrame
        update_predictions(predictions, sample, tfx, log_likelihoods, subtypes)

        # Update progress
        n_complete += 1
        stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
        stdout.flush()

    print('\nFinished.')
    return predictions


def main():
    parser = argparse.ArgumentParser(description='\n### Keraon.py ###')

    # Required arguments

    parser.add_argument('-r', '--reference_data', nargs='*', required=True, 
                        help="""Either a single, pre-generated reference_simplex.pickle file or one or more tidy-form, .tsv feature matrices (in which case a reference key must also be passed with -k).
                        Tidy files will be used to generate a basis and should contain 4 columns: "sample", "site", "feature", and "value".""")
    parser.add_argument('-i', '--input_data', required=True, 
                        help="""A tidy-form, .tsv feature matrix with test samples. Should contain 4 columns: "sample", "site", "feature", and "value".
                        Sites and features most correspond to those passed with the reference samples or basis""")
    parser.add_argument('-t', '--tfx', required=True, 
                        help=""".tsv file with test sample names and estimated tumor fractions. If a third column with "true" subtypes/categories is passed, additional validation will be performed.
                        If blanks/nans are passed for tfx for any samples, they will be treated as unknowns and tfx will be predicted (less accurate).
                        If multiple subtypes are present, they should be separated by commas, e.g. "ARPC,NEPC,DNPC".""")
    parser.add_argument('-k', '--reference_key', required=True,
                        help='.tsv file with reference sample names, subtypes/categories, and purity. One subtype must be "Healthy" with purity=0.')
    # Optional arguments
    parser.add_argument('-d', '--doi', default=None, 
                        help='Disease/subtype of interest (positive case) for plotting and calculating ROCs. Must be present in key.')
    parser.add_argument('-x', '--thresholds', default=(0.322, 0.025), 
                        help='Thresholds for calling disease of interest / positive class (ctdPheno, Keraon).')
    parser.add_argument('-f', '--features', default=None, 
                        help='File with a list of site_feature combinations to restrict to. Sites and features should be separated by an underscore.')
    parser.add_argument('-s', '--svm_selection', action='store_false', default=True, 
                        help='Flag indicating whether to TURN OFF SVM feature selection method. SVM will not run if a -f/--features file is passed.')
    parser.add_argument('-p', '--palette', default=None, 
                        help='tsv file with matched categories/samples and HEX color codes. Subtype/category names must match those passed with the -t and -k options.')
    args = parser.parse_args()

    # Paths
    input_path, tfx_path, ref_path, key_path = args.input_data, args.tfx, args.reference_data, args.reference_key
    doi, thresholds, palette_path = args.doi, args.thresholds, args.palette
    features_path, perform_svm  = args.features, args.svm_selection

    print('\n### Welcome to Keraon ###\nLoading data . . .\n')
    # Quick sanity check
    """ Keraon can be used in essentially three ways, with regards to the reference data:
    1. Use a pre-generated reference_simplex.pickle file passed with -r, in which case the -k option is ignored/not needed.
    2. Use a tsv feature matrix(ices) passed with -r, in which case a reference key must also be passed with -k, along with
       pre-chosen features passed with -f. No SVM feature selection will be performed. A pickle file will be generated.
    3. Use a tsv feature matrix(ices) passed with -r, in which case a reference key must also be passed with -k. If a feature
       list is not passed with -f, SVM feature selection will be performed. A pickle file will be generated.
    """
    if ref_path[0].endswith('.pickle'):
        print("Loading reference data from a pre-generated pickle file."
              "\nAny passed reference key (-k) will be used for palette only, and features (-f) will be IGNORED.")
        load_features, perform_svm = False, False
    elif features_path is not None and key_path is not None:
        print("Generating a reference basis from the passed tsv feature matrix and key, using the features passed with -f.")
        load_features, perform_svm = True, False
    elif features_path is None and key_path is not None:
        print("Generating a reference basis from the passed tsv feature matrix and key, using SVM feature selection.")
        load_features, perform_svm = False, True
    else:
        print("Error: Invalid combination of input data, reference key, and features. Please check your inputs. Exiting.")
        exit(1)

    # Create necessary directories
    os.makedirs('results/', exist_ok=True)
    processing_dir = 'results/feature_analysis/'
    keraon_dir = 'results/keraon_mixture-predictions/'
    ctdpheno_dir = 'results/ctdPheno_class-predictions/'
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(keraon_dir, exist_ok=True)
    os.makedirs(ctdpheno_dir, exist_ok=True)
    
    # Load reference key
    if key_path is not None:
        print(f"\nLoading reference key from: {key_path}")
        ref_labels = load_reference_key(key_path)
    else:
        print("No reference key provided. Using default labels.")
        ref_labels = None

    # Load test labels (tfx, and truth if provided)
    print(f"\nLoading test labels (tumor fraction) from: {tfx_path}")
    test_labels, truth_vals = load_test_labels(tfx_path)
    if truth_vals is not None:
        print("Loaded truth values for test samples - validation analysis and thresholding will be conducted.")
    else:
        print("No truth values provided for test samples - validation analysis and thresholding will not be conducted.")

    # Load palette
    if ref_labels is not None:
        palette = load_palette(palette_path, ref_labels)
    else:
        palette = load_palette(palette_path)

    # Load features
    if features_path is not None and load_features:
        print(f"\nLoading pre-selected features from: {features_path}")
        print("(SVM feature selection will be skipped)")
        with open(features_path) as f:
            features = f.read().splitlines()
        # Validate feature format
        for i, feature_line in enumerate(features):
            parts = feature_line.split('_')
            if len(parts) != 2:
                print(f"Warning: Feature '{feature_line}' at line {i+1} in '{features_path}' does not follow the 'site_feature' format (exactly one underscore). Exiting.")
                exit(1)
    
    # Load reference data
    print(f"\nLoading reference data from: {ref_path}")
    if ref_path[0].endswith('.pickle'):
        # Load the pre-generated reference_simplex.pickle file
        with open(ref_path[0], 'rb') as f:
            df_train, scaling_params = pickle.load(f)
    else:
        # Load reference data
        ref_df, scaling_params = load_triton_fm(ref_path, scaling_methods, processing_dir, palette, ref_labels=ref_labels, plot_distributions=True, limit_features=limit_features)
        # Drop specified features from the reference dataframe
        if drop_features:
            print(f"\nApplying feature dropping based on `drop_features` list: {drop_features}")
            drop_regex = '|'.join([re.escape(feature_name) for feature_name in drop_features]) # Escape to treat as literal strings
            # Drop from ref_df
            ref_cols_to_drop = ref_df.columns[ref_df.columns.str.contains(drop_regex, regex=True)]
            if not ref_cols_to_drop.empty:
                ref_df = ref_df.drop(columns=ref_cols_to_drop)
        if perform_svm and not load_features:
            plot_pca(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), processing_dir, palette, "PCA_initial")
            df_train = pd.merge(ref_labels['Subtype'], ref_df, left_index=True, right_index=True)
            df_train = maximal_simplex_volume(df_train)
            df_train = pd.merge(ref_labels['Subtype'], df_train, left_index=True, right_index=True)
            plot_pca(df_train, processing_dir, palette, "PCA_post-SVM")
            print('Finished. Saving reference dataframe . . .')
            df_train.to_csv(processing_dir + 'SVM_site_features.tsv', sep="\t")
        elif load_features and not perform_svm:
            print(f"Using pre-selected features.")
            # Ensure that only features present in both the list and the DataFrame are selected
            features_to_keep = [feature for feature in features if feature in ref_df.columns]
            missing_features = [feature for feature in features if feature not in ref_df.columns]
            if missing_features:
                print(f"Warning: The following features from the feature list were not found in the reference data and will be ignored: {missing_features}")
            if not features_to_keep:
                print(f"Error: No features from the provided list were found in the reference data. Exiting.")
                exit(1)
            ref_df = ref_df[features_to_keep]
            df_train = pd.merge(ref_labels['Subtype'], ref_df, left_index=True, right_index=True)
            plot_pca(df_train, processing_dir, palette, "PCA_pre-selected_features")
            print('Finished. Saving reference dataframe . . .')
            df_train.to_csv(processing_dir + 'pre-selected_site_features.tsv', sep="\t")
        else:
            print('No SVM feature selection or pre-selected features were passed. Exiting.')
            exit(1)
        # Save the reference dataframe and scaling parameters to a pickle file
        with open(processing_dir + 'reference_simplex.pickle', 'wb') as f:
            pickle.dump((df_train, scaling_params), f)
        print(f'Reference dataframe and scaling parameters saved as {processing_dir}reference_simplex.pickle')


    # Load test data
    print(f"\nLoading test data from: {input_path}")
    test_df, _ = load_triton_fm(input_path, scaling_methods, processing_dir, palette, feature_scaling_params=scaling_params, plot_distributions=True)
    # Define the required features from the training data
    required_features = df_train.drop('Subtype', axis=1).columns
    # Check for missing features in test_df
    missing_in_test = [feature for feature in required_features if feature not in test_df.columns]
    if missing_in_test:
        print(f"Warning: The test data is missing the following features that are present in the reference/training data: {missing_in_test}")
        print("The test data will be restricted to the features common with the training data.")
    # Restrict test_df features to those identified and present in df_train and also present in the current test_df to avoid KeyErrors
    common_features = [feature for feature in required_features if feature in test_df.columns]
    if not common_features:
        print("Error: No common features found between the test data and the required features from the training data. Exiting.")
        exit(1)
    test_df = test_df[common_features]
    df_test = pd.merge(test_labels, test_df, left_index=True, right_index=True, how='inner')
    if df_test.empty:
        print("Warning: After merging test labels and test data (and aligning features), the resulting df_test is empty.")
        print("This might be due to no common sample IDs between test_labels and test_df after feature alignment, or no common features. Exiting.")
        exit(1) 

    if truth_vals is not None:
        plot_pca(df_train, processing_dir, palette, "PCA_final-basis_wTestSamples", post_df=pd.merge(truth_vals, df_test, left_index=True, right_index=True))
    else:
        plot_pca(df_train, processing_dir, palette, "PCA_final-basis_wTestSamples", post_df=df_test)

    # Print complete feature distributions for selected site_features, showing the test samples against the reference
    plot_combined_feature_distributions(df_train, df_test, processing_dir + 'feature_distributions/final-basis_site-features', palette)

    # run ctdPheno and plot results
    print("\n### Running experiment: classification (ctdPheno)")
    ctdpheno_preds = ctdpheno(df_train, df_test)
    ctdpheno_preds = ctdpheno_preds.sort_values(['TFX'], ascending=False)
    if truth_vals is not None:
        ctdpheno_preds = pd.merge(truth_vals, ctdpheno_preds, left_index=True, right_index=True)
    else:
        ctdpheno_preds['Truth'] = 'Unknown'
    ctdpheno_preds.to_csv(ctdpheno_dir + 'ctdPheno_class-predictions.tsv', sep="\t")
    if not (ctdpheno_preds['Truth'] == 'Unknown').all():
        threshold = plot_roc(ctdpheno_preds.copy(), ctdpheno_dir, doi)[3]
        print(f"Threshold for calling disease of interest / positive class (optimal by maximizing Youden's J): {threshold}")
    else:
        threshold = thresholds[0]
        print(f"Threshold for calling disease of interest / positive class (provided): {threshold}")
    plot_ctdpheno(ctdpheno_preds, ctdpheno_dir, doi, threshold)

    # run Keraon and plot results
    print("\n### Running experiment: mixture estimation (Keraon)")
    keraon_preds = keraon(df_train, df_test)
    keraon_preds = keraon_preds.sort_values(['TFX'], ascending=False)
    if truth_vals is not None:
        keraon_preds = pd.merge(truth_vals, keraon_preds, left_index=True, right_index=True)
    else:
        keraon_preds['Truth'] = 'Unknown'
    keraon_preds.to_csv(keraon_dir + 'Keraon_mixture-predictions.tsv', sep="\t")
    if not (keraon_preds['Truth'] == 'Unknown').all():
        threshold = plot_roc(keraon_preds.copy(), keraon_dir, doi + '_fraction')[3]
        print(f"Threshold for calling disease of interest / positive class (optimal by maximizing Youden's J): {threshold}")
    else:
        threshold = thresholds[1]
        print(f"Threshold for calling disease of interest / positive class (provided): {threshold}")
    plot_keraon(keraon_preds, keraon_dir, doi + '_fraction', threshold, palette=palette)

    print('Now exiting.')


if __name__ == "__main__":
    main()

#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.0, 09/19/2023

"""
This newest iteration of Keraon aims to combine ctdPheno and Keraon together, to give dual output predictions
for any appropriately formatted datasets. It will also be made more user-friendly in true script form, with
exmaple runs saved in the accompanying EXAMPLE_RUNS.txt file. I plan to test new feature selection methods, as
well as new methods for Keraon (kept in my own notes for now). Also add simulated run option for both, based on
the references (see PreviousModels tree).
"""

import os
import argparse
import numpy as np
import pandas as pd
from sys import stdout
from keraon_helpers import *
from keraon_plotters import *
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar


def ctdpheno(ref_dict, df, subtypes):
    """
    beta descent
    ctdPheno: FILL
    """
    print('Running Heterogeneous Beta Predictor  . . . ')
    features = list(ref_dict.keys())
    samples = list(df.index)
    # predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'TFX', 'Prediction', subtypes[0] + '_PLL', subtypes[1] + '_PLL', 'JPLL'])
    cols = [subtype for subtype in subtypes] + ['TFX', 'Prediction', 'JointPLL']
    predictions = pd.DataFrame(0, index=df.index, columns=cols)
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        pdf_set_a, pdf_set_b = [], []
        pdfs = [[]] * len(subtypes)
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
            for subtype in subtypes:
                exp = tfx * ref_dict[feature][subtype + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
                std = np.sqrt(tfx * np.square(ref_dict[feature][subtype + '_Std']) +
                              (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
                pdf = norm.pdf(feature_val, loc=exp, scale=std)
                if np.isfinite(pdf) and pdf != 0:
                    pdfs[subtypes.index(subtype)].append(pdf)
            
            exp_a = tfx * ref_dict[feature][subtypes[0] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_a = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[0] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            exp_b = tfx * ref_dict[feature][subtypes[1] + '_Mean'] + (1 - tfx) * ref_dict[feature]['Healthy_Mean']
            std_b = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[1] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['Healthy_Std']))
            pdf_a = norm.pdf(feature_val, loc=exp_a, scale=std_a)
            pdf_b = norm.pdf(feature_val, loc=exp_b, scale=std_b)
            if np.isfinite(pdf_a) and np.isfinite(pdf_b) and pdf_a != 0 and pdf_b != 0:
                pdf_set_a.append(pdf_a)
                pdf_set_b.append(pdf_b)

        print(pdfs)
        print(pdf_set_a)
        print(pdf_set_b)

        # def objective(theta):
        #     log_likelihood = 0
        #     for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
        #         joint_pdf = theta * val_1 + (1 - theta) * val_2
        #         if joint_pdf > 0:
        #             log_likelihood += np.log(joint_pdf)
        #     return -1 * log_likelihood
        def objective(theta):
            log_likelihood = 0
            for vals, t in zip(pdfs, theta):
                joint_pdf = sum([t * val for val in vals])
                if joint_pdf > 0:
                    log_likelihood += np.log(joint_pdf)
            return -1 * log_likelihood

        # def final_pdf(final_weight):
        #     log_likelihood_a, log_likelihood_b, jpdf = 0, 0, 0
        #     for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
        #         joint_a, joint_b = final_weight * val_1, (1 - final_weight) * val_2
        #         joint_pdf = final_weight * val_1 + (1 - final_weight) * val_2
        #         if joint_a > 0:
        #             log_likelihood_a += np.log(joint_a)
        #         if joint_b > 0:
        #             log_likelihood_b += np.log(joint_b)
        #         if joint_pdf > 0:
        #             jpdf += np.log(joint_pdf)
        #     return log_likelihood_a, log_likelihood_b, jpdf

        # weight_1 = minimize_scalar(objective, bounds=(0, 1), method='bounded').x
        theta_init = [1 / len(subtypes)] * len(subtypes)
        bounds = [(0, 1)] * len(subtypes)
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
        weights = minimize(objective, theta_init, bounds=bounds, constraints=constraints, method='SLSQP').x

        # final_pdf_a, final_pdf_b, final_jpdf = final_pdf(weight_1)
        predictions.loc[sample, 'TFX'] = tfx
        predictions.loc[sample, 'JPLL'] = objective(weights)
        for subtype in subtypes:
            predictions.loc[sample, subtype] = np.round(weights[subtypes.index(subtype)], 4)
            if predictions.loc[sample, subtype] > 0.9:
                predictions.loc[sample, 'Prediction'] = subtypes[subtypes.index(subtype)]
    return predictions


def keraon(ref_df, df, name, palette, benchmarking=False):
    """
        N.B. This is the pulldown from the CD paper
        This model takes in two dataframes: ref_df contains a list of "pure" samples with known subtype 'Subtype'
        including 'Healthy' (for basis anchoring) and *non-correlated* features of interest; df contains test samples of
        unknown (possibly mixed) subtype or mixed subtype and tumor fraction 'TFX'. This model assumes passed features
        are non-correlated, normally distributed within each subtype in the reference, and that feature values scale
        linearly with tumor fraction. The number of subtypes is denoted K (including healthy) and the number of features
        is denoted F below. The algorithm is most accurate when anchor distributions are tightly defined, and subtypes
        are separated from healthy.
        The algorithm works as follows:
        1. Define multi-variate Gaussian distributions for each pure "anchor" in the feature space
        2. If F == 2 (including healthy) plot the surface of the anchor distributions for visualization (optional)
        ~ For each test sample:
        3. Compute basis vectors (the sample's location in class-space, centered at healthy) and components; this step
           translates the sample's coordinates from feature-space to subtype-space
        4. Based on the basis components and TFX establish component-based basis predictions
            Parameters:
                ref_df (dataframe): reference pandas df with anchor values
                df (dataframe): pandas df with test values
                name (string): name to write results locally
                palette ({dict}): colors for each passed subtype - only used for plotting
                benchmarking (bool): if True, benchmarking mode: remove ref_df samples that contain the test sample name
                    from the anchoring distributions before fitting.
                N.B. the ref_df must contain a column 'Subtype' which includes 'Healthy' for some samples; the df
                    must contain a column 'TFX' (if used) with the estimated tumor fraction for each sample; all other
                    columns should be numerical features (scaling has no impact) and indices should be sample names. All
                    features provided in ref_df will be used; df may contain additional columns which will be ignored.
            Returns:
                basis_predictions (dataframe): samples as indices with mixture-level predictions (component-based)

    """
    print('#################################################')
    print('Running Keraon estimation on ' + name)
    if not os.path.exists(name + '/'): os.makedirs(name + '/')
    # get initial reference values:
    ref_df = ref_df.dropna(axis=1)
    features = list(ref_df.iloc[:, 1:].columns)  # simply for ordering
    fd = len(features)  # dimensionality of the feature space
    subtypes = list(ref_df['Subtype'].unique())
    sd = len(subtypes)  # dimensionality of the subtypes
    print('Feature dimensions F = ' + str(fd))
    print(features)
    print('Class dimensions K = ' + str(sd))
    print(subtypes)
    mean_vectors = []
    cov_matrices = []
    hd_idx = subtypes.index('Healthy')
    # get anchor/ref vector of means and covariance matrices:
    for subtype in subtypes:
        df_temp = ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:]
        mean_vectors.append(df_temp.mean(axis=0).to_numpy())
        cov_matrices.append(df_temp.cov().to_numpy())
    print('Covariance determinants:')
    print([np.linalg.det(sigma) for sigma in cov_matrices])
    if fd > sd:  # overdetermined system:
        print('The system is overdetermined: least squares will be used to estimate basis components.')
    if 'TFX' not in df:
        print('A column TFX with sample tumor fraction estimates is missing from the input dataframe. Exiting.')
        exit()

    if fd == 2:  # plot the anchor distributions in feature space as a normalized surface, to show shape
        print('Plotting anchor distributions . . .')
        vectors = mean_vectors

        @np.vectorize
        def anchor_surface(xp, yp, scale=False):
            v = np.array([xp, yp])
            z = 0.0
            for mu, var in zip(vectors, cov_matrices):
                if scale:
                    c = 1
                else:
                    c = 1/np.sqrt((2 * math.pi) ** fd * np.linalg.det(var))
                z += c * math.exp((-1/2) * np.matmul(np.matmul(np.transpose(v - mu), np.linalg.inv(var)), (v - mu)))
            return z
        x_min = min([ref_df[features[0]].min(), df[features[0]].min()])
        x_max = max([ref_df[features[0]].max(), df[features[0]].max()])
        y_min = min([ref_df[features[1]].min(), df[features[1]].min()])
        y_max = max([ref_df[features[1]].max(), df[features[1]].max()])
        x_buffer, y_buffer = 0.3 * (x_max - x_min), 0.3 * (y_max - y_min)
        x_min, x_max = x_min - x_buffer, x_max + x_buffer
        y_min, y_max = y_min - y_buffer, y_max + y_buffer
        x_axis = np.arange(x_min, x_max, 0.01)
        y_axis = np.arange(y_min, y_max, 0.01)
        x, y = np.meshgrid(x_axis, y_axis)
        z_surf = anchor_surface(x, y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z_surf, cmap='viridis', linewidth=0, antialiased=False)
        for i in range(sd):  # plot each anchor maximum point
            ax.scatter(mean_vectors[i][0], mean_vectors[i][1], anchor_surface(mean_vectors[i][0], mean_vectors[i][1]),
                       color=palette[subtypes[i]])
            ax.text(mean_vectors[i][0], mean_vectors[i][1], anchor_surface(mean_vectors[i][0], mean_vectors[i][1]),
                    '%s' % (subtypes[i]), zorder=1, color=palette[subtypes[i]])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel('Probability Density')
        ax.set_title('Feature Space Anchor Density')
        fig.savefig(name + '/' + name + '_AnchorSurface.pdf', bbox_inches='tight')
        plt.close(fig)

    samples = list(df.index)
    out_cols_basis = ['TFX'] + [subtype + '_fraction' for subtype in subtypes if subtype != 'Healthy']\
                    + ['Healthy_fraction'] + [subtype + '_burden' for subtype in subtypes if subtype != 'Healthy']\
                    + ['FS_Region']
    basis_predictions = pd.DataFrame(0, index=df.index, columns=out_cols_basis)
    n_complete, n_samples = 0, len(samples)
    stdout.write('\rRunning samples | completed: [{0}/'.format(n_complete) + str(n_samples) + ']')
    stdout.flush()

    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        sample_vector = np.asarray(df.loc[sample, features].to_numpy(), dtype=float)
        if np.isnan(np.sum(sample_vector)):  # if there are missing values
            print('WARNING: sample ' + sample + ' contained NaN values and will be skipped')
            basis_predictions.drop(sample)
            continue
        if benchmarking:  # re-compute anchors without the sample; only coded for Triplets LuCaP benchmarking data
            # assumed LuCaP Triplets naming convention, e.g. 49_LuCaP_35_LuCaP_NEPC_0.10_TF0.10
            drop_ne = sample.split('_LuCaP_')[0]
            drop_ar = sample.split('_LuCaP_')[1]
            mean_vectors = []
            cov_matrices = []
            hd_idx = subtypes.index('Healthy')
            # get anchor/ref vector of means and covariance matrices:
            for subtype in subtypes:
                df_temp = ref_df[ref_df['Subtype'] == subtype].iloc[:, 1:]
                df_temp = df_temp[~df_temp.index.str.contains(drop_ne)]
                df_temp = df_temp[~df_temp.index.str.contains(drop_ar)]
                mean_vectors.append(df_temp.mean(axis=0).to_numpy())
                cov_matrices.append(df_temp.cov().to_numpy())
        # TFX placement correction and basis component computation:
        basis_vectors = [mean_vectors[i] - mean_vectors[hd_idx] for i in range(len(subtypes)) if i != hd_idx]
        hd_sample = sample_vector - mean_vectors[hd_idx]  # directional vector from HD to sample
        if fd > (sd - 1):  # overdetermined system:
            bc = np.linalg.lstsq(np.column_stack(basis_vectors), hd_sample, rcond=None)[0]
        else:
            bc = np.linalg.solve(np.column_stack(basis_vectors), hd_sample)
        if np.all((bc > 0)) or np.all((bc < 0)):  # if feature lies in anchor volume, or in the contra-space
            if np.all((bc > 0)):  # normalized basis contributions directly used as components
                fs_region = 'KSpace'
                comp_fraction = bc / np.sum(bc)
            else:
                fs_region = 'Contra-KSpace'
                comp_fraction = (bc + 1) / np.sum(bc + 1)  # assume point within contra-space with negative bounds
        else:
            fs_region = 'Extra-KSpace'
            nn_bc = np.where(bc < 0.0, 0.0, bc)  # find comp-fraction using only non-negative basis components
            comp_fraction = nn_bc / np.sum(nn_bc)

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

    # prepare reference dataframe:
    ref_dfs = [pd.read_table(path, sep='\t', index_col=0, header=0) for path in ref_path]
    ref_df = pd.concat(ref_dfs)
    if sites is not None:
        ref_df = ref_df[ref_df['site'].isin(sites)]
    if features is not None:
        ref_df = ref_df[ref_df['feature'].isin(features)]
    ref_df['site'] = ref_df['site'].str.replace('_', '-')  # ensure no _ in site names
    ref_df['feature'] = ref_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
    ref_df['cols'] = ref_df['site'].astype(str) + '_' + ref_df['feature']
    ref_df = ref_df.pivot_table(index=[ref_df.index.values], columns=['cols'], values='value')
    ref_labels = pd.read_table(key_path, sep='\t', index_col=0, header=0, names=['Subtype'])

    # prepare test dataframe:
    test_df = pd.read_table(input_path, sep='\t', index_col=0, header=0)
    if sites is not None:
        test_df = test_df[test_df['site'].isin(sites)]
    if features is not None:
        test_df = test_df[test_df['feature'].isin(features)]
    test_df['site'] = test_df['site'].str.replace('_', '-')  # ensure no _ in site names
    test_df['feature'] = test_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
    test_df['cols'] = test_df['site'].astype(str) + '_' + test_df['feature']
    test_df = test_df.pivot_table(index=[test_df.index.values], columns=['cols'], values='value')
    test_labels = pd.read_table(tfx_path, sep='\t', index_col=0, header=None)
    if len(test_labels.columns) == 2:  # truth values supplied
        ordering = test_labels.index
        ordering.names = ['Sample']
        truth_vals = test_labels.iloc[:, 1:]
        truth_vals.columns = ['Truth']
        test_labels = test_labels.drop(2, axis=1)
    else:
        ordering = None
        truth_vals = None
    test_labels.columns = ['TFX']

    # filter features
    print('Filtering features . . .')
    direct = 'results/FeatureSpace/'
    ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='mean-depth')))] # sampling depth is not a feature
    ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='plus-minus-ratio')))] # highly susceptible to outliers
    ref_df = ref_df[ref_df.columns.drop(list(ref_df.filter(regex='frag-')))] # fragmentomic features tend to NOT be normally distributed
    ref_df = norm_exp(pd.merge(ref_labels, ref_df, left_index=True, right_index=True), direct)[0] # only use nominally "normal" features
    test_df = test_df[ref_df.columns]
    # scale features:
    ref_df, stdev_dict = scale_features(ref_df)
    test_df = scale_features(test_df, stdev_dict=stdev_dict)[0]
    df_test = pd.merge(test_labels, test_df, left_index=True, right_index=True)
    df_train = pd.merge(ref_labels, ref_df, left_index=True, right_index=True)
    df_train = df_train[df_train['Subtype'].notna()]
    df_train = select_features(df_train)
    # TODO: save as binary after to quick load (ahead of train)
    # TODO: plot test data on final feature space vs initial feature space
    
    # run ctdPheno
    print("### Running experiment: classification (formerly ctdPheno)")
    direct = 'results/ctdPheno/'
    subtypes = list(df_train['Subtype'].unique())
    subtypes.remove('Healthy')
    metric_dict = metric_analysis(df_train, direct)
    ctdpheno_preds = ctdpheno(metric_dict, df_test, subtypes)
    # N.B., by default CD-version ctdPheno will only use the features that are significantly different
    # between the non-healthy types, and not the feature selection implimented above
    # df_diff = diff_exp(df_train, name, thresh=1.0)
    # metric_dict = metric_analysis(df_train, direct)
    # ctdpheno_preds = ctdpheno(metric_dict, df_test)
    if truth_vals is not None:
        ctdpheno_preds = pd.merge(truth_vals, ctdpheno_preds, left_index=True, right_index=True)
    if ordering is not None:
        ctdpheno_preds = ctdpheno_preds.reindex(ordering)
    elif truth_vals is not None:
        ctdpheno_preds = ctdpheno_preds.sort_values(['Truth', 'TFX'], ascending=[True, False])
    else:
        ctdpheno_preds = ctdpheno_preds.sort_values(['TFX'], ascending=False)
    ctdpheno_preds.to_csv(direct + 'predictions.tsv', sep="\t")
    plot_ctdpheno(ctdpheno_preds, direct, subtypes)
    plot_roc(ctdpheno_preds, direct)

    exit()

    # run Keraon
    print("### Running experiment: mixture estimation (formerly Keraon)")
    direct = 'results/Keraon/'
    keraon_preds = keraon(df_train, df_test, order=ordering, benchmarking=False)
    # feature_fig = plot_feature_space(df_train, df_test, palette)
    # feature_fig.savefig(name + '/' + name + '_FeatureSpace.pdf', bbox_inches='tight')
    if truth_vals is not None:
        keraon_preds = pd.merge(truth_vals, keraon_preds, left_index=True, right_index=True)
    keraon_preds.to_csv(name + '/' + name + '_basis-predictions.tsv', sep="\t")
    df_train.to_csv(name + '/' + name + '_final-features.tsv', sep="\t")

    print('Finished.')


if __name__ == "__main__":
    main()

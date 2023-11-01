#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 08/25/2022

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from Quix.quix import plot_feature_space, quix


targets = ['Healthy', 'TypeA', 'TypeB']
colors = ['green', 'red', 'blue']
palette = {targets[i]: colors[i] for i in range(len(targets))}


def generate_type_data(n_feats, n_samples, feat_range=(0.0, 1.0), var_range=(0.01, 0.05), pre_mu=None):
    if pre_mu is None:
        mu = np.random.uniform(feat_range[0], feat_range[1], n_feats)
    else: mu = pre_mu
    sigma = np.random.uniform(var_range[0], var_range[1], n_feats)
    return np.vstack([np.random.normal(m, s, n_samples) for m, s in zip(mu, sigma)]).transpose(), mu, sigma


def get_fit_info(df, name):
    pc, pp = pearsonr(df['TypeA_true-fraction'].values, df['TypeA_fraction'].values)
    pp = ['{:.3f}'.format(pval) if pval > 0.001 else '{:.2e}'.format(pval) for pval in [pp]][0]
    print(name + ': Pearson\'s r: ' + str(round(pc, 3)) + ' (p=' + pp + ')')


def main():
    """
    Run example tests on generated data.
    See Quix documentation for more details.
    """
    # N.B. the first column of the reference df should be the category for comparisons: 'Subtype'
    # ALL other columns should be numeric features
    print("Generating anchor data set")
    anchor_dfs = []
    anchor_mus = []
    anchor_vars = []
    set_mus = {'Healthy': [0.10, 0.15],
               'TypeA': [0.05, 0.72],
               'TypeB': [0.50, 0.10]}
    for target in targets:
        data, mus, vars = generate_type_data(len(targets) - 1, 100, pre_mu=set_mus[target])
        df = pd.DataFrame(data)
        df = df.add_prefix('Feature_')
        df.insert(0, 'Subtype', target)
        anchor_dfs.append(df)
        anchor_mus.append(mus)
        anchor_vars.append(vars)
    df_anchor = pd.concat(anchor_dfs)
    print("Generating test data")
    tfxs = [0.01, 0.03, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90]  # % non-Healthy
    ratios = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00]  # ratio of A:B
    n = 20
    temp_dfs = []
    for tfx in tfxs:
        for ratio in ratios:
            mixed_mus = []
            mixed_vars = []
            for i in range(len(targets) - 1):  # for each feature
                mixed_mus.append((1 - tfx) * anchor_mus[0][i] +
                                 tfx * ratio * anchor_mus[1][i] +
                                 tfx * (1 - ratio) * anchor_mus[2][i])
                mixed_vars.append(np.sqrt((1 - tfx) * anchor_vars[0][i]**2 +
                                          tfx * ratio * anchor_vars[1][i]**2 +
                                          tfx * (1 - ratio) * anchor_vars[2][i]**2))
            mix_data = np.vstack([np.random.normal(m, s, n) for m, s in zip(mixed_mus, mixed_vars)]).transpose()
            mix_df = pd.DataFrame(mix_data)
            mix_df = mix_df.add_prefix('Feature_')
            mix_df.insert(0, 'TFX', tfx)
            mix_df.insert(0, 'TypeA_fraction', tfx * ratio)
            temp_dfs.append(mix_df)
    df_test = pd.concat(temp_dfs)
    df_test = df_test.reset_index(drop=True)
    df_test = df_test.rename('sample_{}'.format)
    pheno_labels = df_test[['TypeA_fraction']]
    pheno_labels = pheno_labels.rename({'TypeA_fraction': 'TypeA_true-fraction'}, axis=1)
    df_test = df_test.drop(['TypeA_fraction'], axis=1)
    ####################################################################################################################
    # run experiments
    print("Running experiment!")
    feature_sets = {'SimulatedFeatures': ['Feature_0', 'Feature_1']}
    for identifier, features in feature_sets.items():
        name = identifier
        if not os.path.exists(name + '/'): os.makedirs(name + '/')
        print('### Running Quix framework for ' + identifier + ' ###')
        feature_fig = plot_feature_space(df_anchor, df_test, features[0:2], palette)
        feature_fig.savefig(name + '/' + name + '_FeatureSpace.pdf', bbox_inches='tight')
        basis_preds, quix_preds = quix(df_anchor, df_test, name, palette,
                                       shift_anchors=True, benchmarking=False, basis_mode=False, renormalize=False,
                                       enforce_const=False, direct_mode=False, plot_shifts=False, plot_surfaces=False)
        basis_preds = pd.merge(pheno_labels, basis_preds, left_index=True, right_index=True)
        basis_preds['AbsError'] = (basis_preds['TypeA_true-fraction'] - basis_preds['TypeA_fraction']).abs() /\
                                  basis_preds['TypeA_true-fraction']
        basis_preds.to_csv(name + '/' + name + '_basis-predictions.tsv', sep="\t")
        quix_preds['TypeA_fraction'] = quix_preds['TFX'] * quix_preds['TypeA_burden']
        quix_preds = pd.merge(pheno_labels, quix_preds, left_index=True, right_index=True)
        quix_preds['AbsError'] = (quix_preds['TypeA_true-fraction'] - quix_preds['TypeA_fraction']).abs() /\
                                 quix_preds['TypeA_true-fraction']
        quix_preds.to_csv(name + '/' + name + '_quix-predictions.tsv', sep="\t")
        df_anchor.to_csv(name + '/' + name + '_final-features.tsv', sep="\t")
        get_fit_info(basis_preds, 'Basis Predictions')
        get_fit_info(quix_preds, 'Quix Predictions')


if __name__ == "__main__":
    main()

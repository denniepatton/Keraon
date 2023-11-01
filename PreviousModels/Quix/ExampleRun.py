#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 08/25/2022

import os
import pandas as pd
from Quix.quix import plot_feature_space, quix
# from Quix.quix import diff_exp, norm_exp, corr_exp, plot_feature_space, quix


targets = ['Healthy', 'ARPC', 'NEPC', 'MIX']
colors = ['#009988', '#0077BB', '#CC3311', '#5D3FD3']
palette = {targets[i]: colors[i] for i in range(len(targets))}


def main():
    """
    """
    # load anchor and test data
    # N.B. one column of the reference df should be the category for comparisons: 'Subtype'
    # ALL other columns should be numeric features, with the index as sample names
    print("Loading anchor data sets")
    # LuCaP (pure phenotype) reference data
    lucap_data_path = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    lucap_labels_path = '/fh/fast/ha_g/user/rpatton/references/LuCaP_PCType.txt'
    df_lucap = pd.read_table(lucap_data_path, sep='\t', index_col=0)
    df_lucap['site'] = df_lucap['site'].str.replace('_', '-')
    df_lucap = df_lucap.pivot(columns=['site', 'feature'], values='value').reset_index()
    df_lucap.set_index('sample', inplace=True)
    df_lucap.columns = ['_'.join(col).strip() for col in df_lucap.columns]
    lucap_labels = pd.read_table(lucap_labels_path, sep='\t', header=None)
    lucap_labels = dict(zip(lucap_labels[0], lucap_labels[1]))
    lucap_labels = {k + '_LuCaP': v for k, v in lucap_labels.items()}
    df_lucap['Subtype'] = df_lucap.index.map(lucap_labels)
    # Healthy reference data
    hd_path = '/fh/fast/ha_g/user/rpatton/HD_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    df_hd = pd.read_table(hd_path, sep='\t', index_col=0)
    df_hd['site'] = df_hd['site'].str.replace('_', '-')
    df_hd = df_hd.pivot(columns=['site', 'feature'], values='value').reset_index()
    df_hd.set_index('sample', inplace=True)
    df_hd.columns = ['_'.join(col).strip() for col in df_hd.columns]
    hd_labels = {name: 'Healthy' for name in df_hd.index.unique()}
    df_hd['Subtype'] = df_hd.index.map(hd_labels)
    # Combined anchor set
    df_anchor = pd.concat([df_lucap, df_hd])
    # Define test data
    tests = ['patient-WGS']
    # "patient-WGS" refers to the UW cohort (deep-WGS)
    for test_data in tests:
        if test_data == 'patient-WGS':
            bench = False
            data_path = '/fh/fast/ha_g/user/rpatton/UW-WGS_data/Triton_ATAC/results/TritonCompositeFM.tsv'
            pheno_path = '/fh/fast/ha_g/user/rpatton/scripts/Keraon/data/patient_subtypes.tsv'
            tfx_path = '/fh/fast/ha_g/user/rpatton/scripts/Keraon/data/WGS_TF_hg19.txt'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            df_patient['site'] = df_patient['site'].str.replace('_', '-')
            df_patient = df_patient.pivot(columns=['site', 'feature'], values='value').reset_index()
            df_patient.set_index('sample', inplace=True)
            df_patient.columns = ['_'.join(col).strip() for col in df_patient.columns]
            pheno_labels = pd.read_table(pheno_path, sep='\t', header=None)
            pheno_labels = dict(zip(pheno_labels[0], pheno_labels[1]))
            tfx_labels = pd.read_table(tfx_path, sep='\t', header=None)
            tfx_labels = dict(zip(tfx_labels[0], tfx_labels[1]))
            df_patient['Subtype'] = df_patient.index.map(pheno_labels)
            df_patient['TFX'] = df_patient.index.map(tfx_labels)
        else:
            bench, df_patient, pheno_labels, tfx_labels = None, None, None, None
            print("No test set specified: exiting.")
            exit()
        ################################################################################################################
        # run experiments
        print("Running experiments (" + test_data + ")\n")
        group_sets = {'All-ATAC-TF': ['LongATAC-ADexclusive-10000TFOverlap', 'LongATAC-NEexclusive-10000TFOverlap']}
        feature_sets = ['central-flux', 'dip-width', 'central-depth', 'central-heterogeneity',
                        'np-amplitude', 'np-score', 'fragment-mad']
        for identifier, group in group_sets.items():
            for label in group:
                df_patient[label + '_central-flux'] =\
                    df_patient[label + '_central-heterogeneity'] - df_patient[label + '_central-depth']
                df_patient[label + '_dip-width'] =\
                    df_patient[label + '_plus-one-pos'] - df_patient[label + '_minus-one-pos']
                df_anchor[label + '_central-flux'] = \
                    df_anchor[label + '_central-heterogeneity'] - df_anchor[label + '_central-depth']
                df_anchor[label + '_dip-width'] = \
                    df_anchor[label + '_plus-one-pos'] - df_anchor[label + '_minus-one-pos']
            for feature in feature_sets:
                name = test_data + '_' + identifier + '_' + feature
                features = [site + '_' + feature for site in group]
                print('### Running Keraon framework for ' + test_data + ': ' + identifier + ' ' + feature + ' ###')
                if not os.path.exists(name + '/'): os.makedirs(name + '/')
                df_ref = df_anchor[['Subtype'] + features]
                df_pat = df_patient[['TFX'] + features]
                # Feature cleaning below is unnecessary for pre-picked features, but may be useful for reporting
                # df_ref, df_norm = norm_exp(df_ref, name, thresh=0.5)
                # df_ref, df_lpq = diff_exp(df_ref, name, ['ARPC', 'NEPC'])
                # df_ref = corr_exp(df_ref, name, thresh=0.95, df_ref_1=df_lpq, df_ref_2=df_lpq)
                feature_fig = plot_feature_space(df_ref, df_pat, features, palette)
                feature_fig.savefig(name + '/' + name + '_FeatureSpace.pdf', bbox_inches='tight')
                basis_preds, quix_preds = quix(df_ref, df_pat, name, palette, benchmarking=bench)
                basis_preds['Subtype'] = basis_preds.index.map(pheno_labels)
                quix_preds['Subtype'] = quix_preds.index.map(pheno_labels)
                basis_preds.to_csv(name + '/' + name + '_basis-predictions.tsv', sep="\t")
                quix_preds.to_csv(name + '/' + name + '_quix-predictions.tsv', sep="\t")
                df_ref.to_csv(name + '/' + name + '_final-features.tsv', sep="\t")


if __name__ == "__main__":
    main()

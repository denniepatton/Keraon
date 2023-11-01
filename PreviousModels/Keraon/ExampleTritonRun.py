#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.1, 09/29/2022

import os
import pandas as pd
from Keraon import plot_feature_space, keraon


targets = ['Healthy', 'ARPC', 'NEPC', 'MIX', '-', '+']
colors = ['#009988', '#0077BB', '#CC3311', '#5D3FD3', '#33BBEE', '#EE7733']
palette = {targets[i]: colors[i] for i in range(len(targets))}


def main():
    """
    """
    # load anchor and test data
    # N.B. the first column of the reference df should be the category for comparisons: 'Subtype'
    # ALL other columns should be numeric features
    print("Loading anchor data sets")
    # LuCaP (pure phenotype) reference data
    # lucap_data_path = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    # lucap_labels_path = '/fh/fast/ha_g/user/rpatton/references/LuCaP_PCType.txt'
    lucap_data_path = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Triton_PSMA/results/TritonCompositeFM.tsv'
    lucap_labels_path = '/fh/fast/ha_g/user/rpatton/references/LuCaP/LuCaP_PSMA-status.tsv'
    df_lucap = pd.read_table(lucap_data_path, sep='\t', index_col=0)
    df_lucap['site'] = df_lucap['site'].str.replace('_', '-')
    df_lucap['cols'] = df_lucap['site'].astype(str) + '_' + df_lucap['feature']
    df_lucap = df_lucap.pivot(columns=['cols'], values='value')
    lucap_labels = pd.read_table(lucap_labels_path, sep='\t', header=None)
    lucap_labels = dict(zip(lucap_labels[0], lucap_labels[1]))
    # Healthy reference data
    # hd_path = '/fh/fast/ha_g/user/rpatton/HD_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    hd_path = '/fh/fast/ha_g/user/rpatton/HD_data/Triton_PSMA/results/TritonCompositeFM.tsv'
    df_hd = pd.read_table(hd_path, sep='\t', index_col=0)
    df_hd['site'] = df_hd['site'].str.replace('_', '-')
    df_hd['cols'] = df_hd['site'].astype(str) + '_' + df_hd['feature']
    df_hd = df_hd.pivot(columns=['cols'], values='value')
    hd_labels = {name: 'Healthy' for name in df_hd.index.unique()}
    # Combined anchor set
    df_anchor = pd.concat([df_lucap, df_hd])
    anchor_labels = {**lucap_labels, **hd_labels}
    # Define test data
    # tests = ['patient_WGS', 'patient_Berchuck', 'patient_ULP', 'Triplet25x']
    tests = ['TAN_WGS']
    # "patient_ULP/WGS" refers to DFCI cohort II and UW cohort (ULP/deep-WGS), patient_Berchuck refers to DFCI cohort I,
    # and Triplet25x refers to the 810 LuCaP/healthy admixture benchmarking data
    for test_data in tests:
        if test_data == 'patient_WGS':
            bench = False
            data_path = '/fh/fast/ha_g/user/rpatton/UW-WGS_data/Triton_ATAC/results/TritonCompositeFM.tsv'
            pheno_path = '/fh/fast/ha_g/user/rpatton/scripts/Keraon/data/patient_subtypes.tsv'
            tfx_path = '/fh/fast/ha_g/user/rpatton/scripts/Keraon/data/WGS_TF_hg19.txt'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            pheno_labels = pd.read_table(pheno_path, sep='\t', header=None)
            pheno_labels = dict(zip(pheno_labels[0], pheno_labels[1]))
            tfx_labels = pd.read_table(tfx_path, sep='\t', header=None)
            tfx_labels = dict(zip(tfx_labels[0], tfx_labels[1]))
        elif test_data == 'TAN_WGS':
            bench = False
            data_path = '/fh/fast/ha_g/user/rpatton/TAN_data/Triton_PSMA/results/TritonCompositeFM.tsv'
            labels_path = '/fh/fast/ha_g/user/rpatton/TAN_data/TAN-WGS_TFX_PSMA.tsv'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            df_patient = df_patient[~df_patient.index.str.contains("ULP")]
            df_patient.index = df_patient.index.str.replace("_WGS", "")
            df_patient['site'] = df_patient['site'].str.replace('_', '-')
            df_patient['cols'] = df_patient['site'].astype(str) + '_' + df_patient['feature']
            df_patient = df_patient.pivot(columns=['cols'], values='value')
            sample_renames = {'17330O_1009': '00-147',
                              '22667S_1010': '05-148',
                              '23552H_HA21-26': '05-217',
                              '23736S_1011': '06-064',
                              '23965W_1012': '06-131',
                              '25923L_HA39-40': '19-048',
                              '25981W_HA45-46': '19-028',
                              '26546P_HA73-74': '19-045'}
            df_patient = df_patient.rename(index=sample_renames)
            df_patient.index = df_patient.index.str.slice(0, 6)
            labels = pd.read_table(labels_path, sep='\t', header=None)
            tfx_labels = dict(zip(labels[0], labels[1]))
            pheno_labels = dict(zip(labels[0], labels[2]))
        else:
            bench, df_patient, pheno_labels, tfx_labels = None, None, None, None
            print("No test set specified: exiting.")
            exit()
        ################################################################################################################
        # run experiments
        print("Running experiments (" + test_data + ")\n")
        # group_sets = {'All-ATAC-TF': ['LongATAC_ADexclusive_10000TFOverlap', 'LongATAC_NEexclusive_10000TFOverlap']}
        group_sets = {'PSMA-ATAC': ['PSMA-1000Gain', 'PSMA-1000Loss']}
        feature_sets = ['central-depth', 'central-heterogeneity', 'np-amplitude', 'np-score', 'fragment-mad']
        for identifier, group in group_sets.items():
            for feature in feature_sets:
                name = test_data + '_' + identifier + '_' + feature
                features = [group_sub + '_' + feature for group_sub in group]
                print('### Running Keraon framework for ' + test_data + ': ' + identifier + ' ' + feature + ' ###')
                if not os.path.exists(name + '/'): os.makedirs(name + '/')
                df_ref = df_anchor[features].copy()
                df_pat = df_patient[features].copy()
                df_ref['Subtype'] = df_ref.index.map(anchor_labels)
                # df_pat['Subtype'] = df_pat.index.map(pheno_labels)
                df_pat['TFX'] = df_pat.index.map(tfx_labels)
                df_ref = df_ref.dropna()
                df_pat = df_pat[df_pat['TFX'].notna()]
                # Feature cleaning below is unnecessary for pre-picked features, but may be useful for reporting
                # df_ref, df_norm = norm_exp(df_ref, name, thresh=0.5)
                # df_ref, df_lpq = diff_exp(df_ref, name, ['ARPC', 'NEPC'])
                # df_ref = corr_exp(df_ref, name, thresh=0.95, df_ref_1=df_lpq, df_ref_2=df_lpq)
                # Below has some stupid pandas issue with seaborn thinking "Subtype" shouldn't be a hue category
                feature_fig = plot_feature_space(df_ref, df_pat, features[0:2], palette)
                feature_fig.savefig(name + '/' + name + '_FeatureSpace.pdf', bbox_inches='tight')
                basis_preds = keraon(df_ref, df_pat, name, palette, benchmarking=bench)
                # basis_preds = pd.merge(pheno_labels, basis_preds, left_index=True, right_index=True)
                basis_preds.to_csv(name + '/' + name + '_basis-predictions.tsv', sep="\t")
                df_ref.to_csv(name + '/' + name + '_final-features.tsv', sep="\t")


if __name__ == "__main__":
    main()

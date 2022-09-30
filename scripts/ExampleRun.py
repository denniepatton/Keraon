#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.1, 09/29/2022

import os
import pandas as pd
from Keraon import plot_feature_space, keraon


targets = ['Healthy', 'ARPC', 'NEPC', 'Patient', 'ARlow', 'AMPC', 'MIX']
colors = ['#009988', '#0077BB', '#CC3311', '#EE3377', '#77BB00', '#466F00', '#5D3FD3']
palette = {targets[i]: colors[i] for i in range(len(targets))}


def main():
    """
    Run example tests as reported in DOI: https://doi.org/10.1101/2022.06.21.496879
    See Quix documentation for more details.
    """
    # load anchor and test data
    # N.B. the first column of the reference df should be the category for comparisons: 'Subtype'
    # ALL other columns should be numeric features
    print("Loading anchor data sets")
    # LuCaP (pure phenotype) reference data
    lucap_data_path = '../data/GriffinFeatureMatrix_LuCaP.tsv'
    lucap_labels_path = '../data/LuCaP_subtypes.txt'
    df_lucap = pd.read_table(lucap_data_path, sep='\t', index_col=0)
    lucap_labels = pd.read_table(lucap_labels_path, sep='\t', index_col=0, names=['Subtype', 'PAM50', 'PSMA'])
    lucap_labels = lucap_labels.rename('{}_LuCaP'.format)
    df_lucap = pd.merge(lucap_labels, df_lucap, left_index=True, right_index=True)
    df_lucap = df_lucap[df_lucap['Subtype'] != 'AMPC']
    df_lucap = df_lucap[df_lucap['Subtype'] != 'ARlow']
    df_lucap = df_lucap.drop(['PAM50', 'PSMA'], axis=1)
    # Healthy reference data
    hd_path = '../data/GriffinFeatureMatrix_HealthyDonor.tsv'
    df_hd = pd.read_table(hd_path, sep='\t', index_col=0)
    df_hd.insert(0, 'Subtype', 'Healthy')
    # Combined anchor set
    df_anchor = pd.concat([df_lucap, df_hd])
    # Define test data
    # tests = ['patient_WGS', 'patient_Berchuck', 'patient_ULP', 'Triplet25x']
    tests = ['patient_WGS']
    # "patient_ULP/WGS" refers to DFCI cohort II and UW cohort (ULP/deep-WGS), patient_Berchuck refers to DFCI cohort I,
    # and Triplet25x refers to the 810 LuCaP/healthy admixture benchmarking data
    for test_data in tests:
        if test_data == 'patient_WGS':
            bench = False
            data_path = '../data/GriffinFeatureMatrix_WGS.tsv'
            pheno_path = '../data/patient_subtypes.tsv'
            tfx_path = '../data/WGS_TF_hg19.txt'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            pheno_labels = pd.read_table(pheno_path, sep='\t', index_col=0, names=['Subtype'])
            tfx_labels = pd.read_table(tfx_path, sep='\t', index_col=0, names=['TFX'])
            df_patient = pd.merge(tfx_labels, df_patient, left_index=True, right_index=True)
        elif test_data == 'patient_ULP':
            bench = False
            data_path = '../data/GriffinFeatureMatrix_ULP.tsv'
            pheno_path = '../data/patient_subtypes.tsv'
            tfx_path = '../data/ULP_TF_hg19.txt'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            pheno_labels = pd.read_table(pheno_path, sep='\t', index_col=0, names=['Subtype'])
            tfx_labels = pd.read_table(tfx_path, sep='\t', index_col=0, names=['TFX'])
            df_patient = pd.merge(tfx_labels, df_patient, left_index=True, right_index=True)
            # below ensures WGS and ULP data are identically named
            df_patient = df_patient.rename({'CRPC_161.ctDNA': 'CRPC_161',
                                            'CRPC_1009_4': 'CRPC_248_T1',
                                            'CRPC_1009_5': 'CRPC_248_T2',
                                            'CRPC_1009_6': 'CRPC_248_T6',
                                            'CRPC_22_1': 'CRPC_257',
                                            'CRPC_466.ctDNA_T2': 'CRPC_466_T3',
                                            'CRPC_531_3': 'CRPC_531_T1',
                                            'CRPC_531_1': 'CRPC_531_T2',
                                            'CRPC_531.ctDNA': 'CRPC_531_T3',
                                            'CRPC_554.ctDNA': 'CRPC_554_T1',
                                            'CRPC_554.ctDNA_T2': 'CRPC_554_T2'}, axis=0)
        elif test_data == 'patient_Berchuck':

            def get_prefix(s):
                return s.split('_')[0]

            bench = False
            data_path = '../data/GriffinFeatureMatrix_DFCI1.tsv'
            labels_path = '../data/DFCI1_TFX_Subtype.tsv'
            print("Loading data for " + test_data)
            df_patient = pd.read_table(data_path, sep='\t', index_col=0)
            df_patient.index = df_patient.index.map(get_prefix)
            labels = pd.read_table(labels_path, sep='\t', index_col=0, names=['TFX', 'Subtype'])
            df_patient = pd.merge(labels, df_patient, left_index=True, right_index=True)
            pheno_labels = df_patient[['Subtype']]
            df_patient = df_patient.drop(['Subtype'], axis=1)
        elif test_data == 'Triplet25x':

            def get_tfx(s):
                return float(s.split('_TF')[1])

            def get_nepc(s):
                return float(s.split('_NEPC_')[1].split('_')[0])

            bench = True
            path = '../data/GriffinFeatureMatrix_Triplets.tsv'
            print("Loading " + path)
            df = pd.read_table(path, sep='\t', index_col=0)
            df['name'] = df.index
            df['TFX'] = df['name'].apply(get_tfx)
            df['NEPC_burden'] = df['name'].apply(get_nepc)
            pheno_labels = df[['NEPC_burden']]
            df_patient = df.drop(['name', 'NEPC_burden'], axis=1)
        else:
            bench, df_patient, pheno_labels = None, None, None
            print("No test set specified: exiting.")
            exit()
        ################################################################################################################
        # run experiments
        print("Running experiments (" + test_data + ")\n")
        if test_data == 'Triplet':  # only test on one feature set
            feature_sets = {'All-ATAC-TF': ['AD-ATAC-TF_Central-Mean', 'NE-ATAC-TF_Central-Mean']}
        else:  # predict on all feature sets
            feature_sets = {'All-ATAC-TF': ['AD-ATAC-TF_Central-Mean', 'NE-ATAC-TF_Central-Mean'],
                            '10000-ATAC-TF': ['AD-ATAC-TF-10000_Central-Mean', 'NE-ATAC-TF-10000_Central-Mean'],
                            '1000-ATAC-TF': ['AD-ATAC-TF-1000_Central-Mean', 'NE-ATAC-TF-1000_Central-Mean'],
                            '100-ATAC-TF': ['AD-ATAC-TF-100_Central-Mean', 'NE-ATAC-TF-100_Central-Mean'],
                            '10000-AR-ASCl1': ['AR-10000_Central-Mean', 'ASCL1-10000_Central-Mean'],
                            '1000-AR-ASCL1': ['AR-1000_Central-Mean', 'ASCL1-1000_Central-Mean']}
        for identifier, features in feature_sets.items():
            name = test_data + '_' + identifier
            print('### Running Keraon framework for ' + test_data + ': ' + identifier + ' ###')
            if not os.path.exists(name + '/'): os.makedirs(name + '/')
            df_ref = df_anchor[['Subtype'] + features]
            # Feature cleaning below is unnecessary for pre-picked features, but may be useful for reporting
            # df_ref, df_norm = norm_exp(df_ref, name, thresh=0.5)
            # df_ref, df_lpq = diff_exp(df_ref, name, ['ARPC', 'NEPC'])
            # df_ref = corr_exp(df_ref, name, thresh=0.95, df_ref_1=df_lpq, df_ref_2=df_lpq)
            feature_fig = plot_feature_space(df_ref, df_patient, features[0:2], palette)
            feature_fig.savefig(name + '/' + name + '_FeatureSpace.pdf', bbox_inches='tight')
            basis_preds = keraon(df_ref, df_patient, name, palette, benchmarking=bench)
            basis_preds = pd.merge(pheno_labels, basis_preds, left_index=True, right_index=True)
            basis_preds.to_csv(name + '/' + name + '_basis-predictions.tsv', sep="\t")
            df_ref.to_csv(name + '/' + name + '_final-features.tsv', sep="\t")


if __name__ == "__main__":
    main()

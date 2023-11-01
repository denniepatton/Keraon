#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 09/13/2021

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'Gray', 'AMPC', 'MIX', '-', '+']
colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#FFAE42', '#9F009F', '#33BBEE', '#EE7733']
palette = {targets[i]: colors[i] for i in range(len(targets))}
cmap = LinearSegmentedColormap.from_list('', ['#0077BB', '#CC3311'])
matplotlib.colormaps.register(cmap, name="mcm")
sns.set(font_scale=1.5)
sns.set_style('ticks')


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


def metric_analysis(df, name):
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
    pd.DataFrame(mat).to_csv(name + '/' + name + '_weights.tsv', sep="\t")
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


def beta_descent(ref_dict, df, subtypes, name, eval, order=None, base_df=None, labs=None):
    print('Running Heterogeneous Beta Predictor on ' + name + ' . . . ')
    if not os.path.exists(name + '/'):
        os.makedirs(name + '/')
    features = list(ref_dict.keys())
    cols = subtypes
    cols.append('Prediction')
    samples = list(df.index)
    predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'TFX', 'Prediction',
                                                           subtypes[0] + '_PLL', subtypes[1] + '_PLL', 'JPLL'])
    if labs is None:
        predictions['Subtype'] = df['Subtype']
    else:
        predictions['Subtype'] = labs
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        pdf_set_a, pdf_set_b = [], []
        if base_df is not None:  # recompute reference dictionary without samples
            if eval == 'Triplet':
                sample_comp_1 = sample.split('_')[0] + '_LuCaP'
                sample_comp_2 = sample.split('_')[1] + '_LuCaP'
                ref_dict = metric_analysis(base_df.drop([sample_comp_1, sample_comp_2]), name)
            else:
                sample_comp = sample.split('_')[0] + '_LuCaP'
                ref_dict = metric_analysis(base_df.drop(sample_comp), name)
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
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

        def objective(theta):
            log_likelihood = 0
            for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
                joint_pdf = theta * val_1 + (1 - theta) * val_2
                if joint_pdf > 0:
                    log_likelihood += np.log(joint_pdf)
            return -1 * log_likelihood

        def final_pdf(final_weight):
            log_likelihood_a, log_likelihood_b, jpdf = 0, 0, 0
            for val_1, val_2 in zip(pdf_set_a, pdf_set_b):
                joint_a, joint_b = final_weight * val_1, (1 - final_weight) * val_2
                joint_pdf = final_weight * val_1 + (1 - final_weight) * val_2
                if joint_a > 0:
                    log_likelihood_a += np.log(joint_a)
                if joint_b > 0:
                    log_likelihood_b += np.log(joint_b)
                if joint_pdf > 0:
                    jpdf += np.log(joint_pdf)
            return log_likelihood_a, log_likelihood_b, jpdf

        weight_1 = minimize_scalar(objective, bounds=(0, 1), method='bounded').x

        final_pdf_a, final_pdf_b, final_jpdf = final_pdf(weight_1)
        predictions.loc[sample, 'TFX'] = tfx
        if eval == 'Bar':
            predictions.loc[sample, 'Depth'] = df.loc[sample, 'Depth']
        predictions.loc[sample, 'JPLL'] = final_jpdf
        predictions.loc[sample, subtypes[0]], predictions.loc[sample, subtypes[1]] = np.round(weight_1, 4), np.round(1 - weight_1, 4)
        predictions.loc[sample, subtypes[0] + '_PLL'], predictions.loc[sample, subtypes[1] + '_PLL'] = final_pdf_a, final_pdf_b
        if predictions.loc[sample, subtypes[0]] > 0.9:
            predictions.loc[sample, 'Prediction'] = subtypes[0]
        elif predictions.loc[sample, subtypes[0]] < 0.1:
            predictions.loc[sample, 'Prediction'] = subtypes[1]
        elif predictions.loc[sample, subtypes[0]] > 0.5:
            predictions.loc[sample, 'Prediction'] = 'Mixed_' + subtypes[0]
        else:
            predictions.loc[sample, 'Prediction'] = 'Mixed_' + subtypes[1]
    predictions.to_csv(name + '/' + name + '_beta-predictions.tsv', sep="\t")

    if eval == 'Bar':  # for benchmarking
        depths = ['0.2X', '1X', '25X']
        predictions = predictions[predictions['TFX'] != 0.03]
        for depth in depths:
            df = predictions.loc[predictions['Depth'] == depth]
            plt.figure(figsize=(8, 8))
            sns.swarmplot(x='TFX', y='PSMA+', hue='Subtype', palette=palette, data=df, s=10, alpha=0.8, dodge=False)
            plt.ylabel('PSMA+ Score')
            plt.xlabel('Tumor Fraction')
            plt.title('Benchmarking Scores at ' + depth, size=14, y=1.1)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.savefig(name + '/' + depth + '_BoxPlot.pdf', bbox_inches="tight")
            plt.close()

    if eval == 'SampleBar':
        predictions['NEPC'] = predictions['NEPC'] - 0.3314
        data = predictions.groupby(predictions['NEPC']).size()
        if isinstance(order, str) and order == 'sorted':
            predictions = predictions.sort_index()
        elif order is not None:
            predictions = predictions.reindex(order.index)
        else:
            predictions = predictions.sort_values('NEPC')
        pal = sns.color_palette("mcm", len(data))
        sns.set_context(rc={'patch.linewidth': 0.0})
        plt.figure(figsize=(24, 3))
        g = sns.barplot(x=predictions.index, y='NEPC', hue='NEPC', data=predictions, palette=pal, dodge=False)
        g.legend_.remove()
        sns.scatterplot(x=predictions.index, y='NEPC', hue='NEPC', data=predictions, palette=pal, s=600, legend=False)

        def change_width(ax, new_value):
            for patch in ax.patches:
                current_width = patch.get_width()
                diff = current_width - new_value
                # we change the bar width
                patch.set_width(new_value)
                # we recenter the bar
                patch.set_x(patch.get_x() + diff * .5)

        change_width(g, .2)
        for item in g.get_xticklabels():
            item.set_rotation(90)
        plt.axhline(y=0, color='b', linestyle='--', lw=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(name + '/PredictionBarPlot.pdf', bbox_inches="tight")
        plt.close()


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


def plot_roc(df, feature, out_path, test_name):
    if 'PC-Phenotype' in df.columns:
        df = df.rename(columns={'PC-Phenotype': 'Subtype'})
    df = df[df['Subtype'].isin(['ARPC', 'NEPC'])]
    plt.figure(figsize=(8, 8))
    df = df.sort_values(by='Subtype')
    truths = pd.factorize(df['Subtype'].values)[0]
    predictions = df[feature].values
    fpr, tpr, thresholds = metrics.roc_curve(truths, predictions)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Mean ROC (AUC = % 0.2f )' % auc, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle(feature + '-based NEPC binary prediction ROC for ' + test_name)
    plt.title('100% TPR threshold: ' + str(round(thresholds[list(tpr).index(1.)], 4)))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return [fpr, tpr, auc]


def plot_rocs(metrics, out_path, test_name):
    plt.figure(figsize=(8, 8))
    for metric in metrics:
        plt.plot(metric[1], metric[2], label=(metric[0] + ' : Mean ROC (AUC = % 0.2f )' % metric[3]), lw=2, alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(test_name + ' NEPC binary prediction ROCs')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return


def main():
    # ### disease reference:
    lucap_data_path = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    lucap_labels_path = '/fh/fast/ha_g/user/rpatton/references/LuCaP/LuCaP_AllVars_ARPC-NEPC.tsv'
    df_lucap = pd.read_table(lucap_data_path, sep='\t', index_col=0)
    df_lucap['site'] = df_lucap['site'].str.replace('_', '-')
    df_lucap['cols'] = df_lucap['site'].astype(str) + '_' + df_lucap['feature']
    df_lucap = df_lucap.pivot(columns=['cols'], values='value')
    lucap_labels = pd.read_table(lucap_labels_path, sep='\t', index_col=0, header=0,
                                 names=['Subtype', 'PSMA-status', 'PSMA-mean', 'PAM50', 'CCP-mean', 'depth'])
    df_lucap = pd.merge(lucap_labels, df_lucap, left_index=True, right_index=True)
    # df_lucap = df_lucap[df_lucap['phenotype'] != 'NEPC'] # (for PSMA tests)
    # df_lucap = df_lucap[df_lucap['depth'] >= 3.0] (already applied)
    df_lucap = df_lucap.drop(['PSMA-status', 'PSMA-mean', 'PAM50', 'CCP-mean', 'depth'], axis=1)
    # ### healthy reference:
    hd_path = '/fh/fast/ha_g/user/rpatton/HD_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    hd_labels_path = '/fh/fast/ha_g/user/rpatton/references/LuCaP/Healthy-Ref_subtypes.tsv'
    df_hd = pd.read_table(hd_path, sep='\t', index_col=0)
    df_hd['site'] = df_hd['site'].str.replace('_', '-')
    df_hd['cols'] = df_hd['site'].astype(str) + '_' + df_hd['feature']
    df_hd = df_hd.pivot(columns=['cols'], values='value')
    hd_labels = pd.read_table(hd_labels_path, sep='\t', index_col=0, header=None, names=['Subtype'])
    df_hd = pd.merge(hd_labels, df_hd, left_index=True, right_index=True)
    df_train = pd.concat([df_lucap, df_hd])
    df_train = df_train[df_train['Subtype'].notna()]

    # TAN
    # data_path = '/fh/fast/ha_g/user/rpatton/TAN_data/Triton_PSMA/results/TritonCompositeFM.tsv'
    # labels_path = '/fh/fast/ha_g/user/rpatton/TAN_data/TAN-WGS_TFX_PSMA.tsv'
    # ordering = 'sorted'

    # df_patient = pd.read_table(data_path, sep='\t', index_col=0)
    # labels = pd.read_table(labels_path, sep='\t', index_col=0, names=['TFX', 'Subtype'])
    # df_patient = df_patient[~df_patient.index.str.contains("ULP")]
    # df_patient.index = df_patient.index.str.replace("_WGS", "")
    # df_patient['site'] = df_patient['site'].str.replace('_', '-')
    # df_patient['cols'] = df_patient['site'].astype(str) + '_' + df_patient['feature']
    # df_patient = df_patient.pivot(columns=['cols'], values='value')
    # sample_renames = {'17330O_1009': '00-147',
    #                   '22667S_1010': '05-148',
    #                   '23552H_HA21-26': '05-217',
    #                   '23736S_1011': '06-064',
    #                   '23965W_1012': '06-131',
    #                   '25923L_HA39-40': '19-048',
    #                   '25981W_HA45-46': '19-028',
    #                   '26546P_HA73-74': '19-045'}
    # df_patient = df_patient.rename(index=sample_renames)
    # df_patient.index = df_patient.index.str.slice(0, 6)
    # df_patient = pd.merge(labels, df_patient, left_index=True, right_index=True)
    # pheno_labels = df_patient[['Subtype']]
    # df_patient = df_patient.drop(['Subtype'], axis=1)

    # UW-ULP
    data_path = '/fh/fast/ha_g/user/rpatton/UW-ULP_data/Triton_ATAC/results/TritonCompositeFM.tsv'
    labels_path = '/fh/fast/ha_g/user/rpatton/UW-ULP_data/UW-ULP_TFX_Subtype.tsv'
    ordering = pd.read_table('/fh/fast/ha_g/user/rpatton/UW-ULP_data/ordering.txt', sep='\t', index_col=0, header=None)

    df_patient = pd.read_table(data_path, sep='\t', index_col=0)
    labels = pd.read_table(labels_path, sep='\t', index_col=0, names=['TFX', 'Subtype'])
    df_patient['site'] = df_patient['site'].str.replace('_', '-')
    df_patient['cols'] = df_patient['site'].astype(str) + '_' + df_patient['feature']
    df_patient = df_patient.pivot(columns=['cols'], values='value')
    df_patient = pd.merge(labels, df_patient, left_index=True, right_index=True)
    pheno_labels = df_patient[['Subtype']]
    df_patient = df_patient.drop(['Subtype'], axis=1)

    # run experiments
    print("Running experiments . . .")
    # feature_sets = {'CenterFeats': ['PSMA-1000Gain_central-depth', 'PSMA-1000Loss_central-depth',
    #                                 'PSMA-1000Gain_central-heterogeneity', 'PSMA-1000Loss_central-heterogeneity']}
    feature_sets = {'CenterFeats': ['AD-Exclusive-filtered-sig_central-depth', 'NE-Exclusive-filtered-sig_central-depth',
                                    'AD-Exclusive-filtered-sig_central-diversity', 'NE-Exclusive-filtered-sig_central-diversity'],
                    'CenterDepth': ['AD-Exclusive-filtered-sig_central-depth', 'NE-Exclusive-filtered-sig_central-depth']}

    # roc_metrics = []
    for identifier, features in feature_sets.items():
        name = identifier
        if not os.path.exists(name + '/'): os.makedirs(name + '/')
        df_sub = df_train[['Subtype'] + features]
        df_diff = diff_exp(df_sub, name, thresh=1.0)
        metric_dict = metric_analysis(df_diff, name)
        beta_descent(metric_dict, df_patient, ['NEPC', 'ARPC'], name, eval='SampleBar', labs=pheno_labels, order=ordering)
        ###
        # path = '/fh/fast/ha_g/user/rpatton/ML_testing/ctdPheno/' + name + '/' + name + '_beta-predictions.tsv'
        # df = pd.read_table(path, sep='\t', index_col=0).drop(['Subtype'], axis=1)
        # df = pd.merge(pheno_labels, df, left_index=True, right_index=True)
        # temp_metrics = plot_roc(df, 'NEPC', name + '/' + identifier + '_NEPC-score_ROC.pdf', name)
        # roc_metrics.append([identifier] + temp_metrics)
    # plot_rocs(roc_metrics,  'Berchuck-ctdPheno_AllROCs.pdf', 'Berchuck-ctdPheno_AllROCs')


if __name__ == "__main__":
    main()

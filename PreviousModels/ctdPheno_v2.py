#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.0, 09/13/2021

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, norm
from scipy.optimize import minimize_scalar
from statsmodels.stats.multitest import fdrcorrection
from sklearn import metrics


sns.set(font_scale=1.5)
sns.set_style('ticks')

# def specificity_sensitivity(target, predicted, threshold):
#     thresh_preds = np.zeros(len(predicted))
#     thresh_preds[predicted > threshold] = 1
#     cm = metrics.confusion_matrix(target, thresh_preds)
#     return cm[1, 1] / (cm[1, 0] + cm[1, 1]), cm[0, 0] / (cm[0, 0] + cm[0, 1])
#
#
# def nroc_curve(y_true, predicted, num_thresh=100):
#     step = 1/num_thresh
#     thresholds = np.arange(0, 1 + step, step)
#     fprs, tprs = [], []
#     for threshold in thresholds:
#         y_pred = np.where(predicted >= threshold, 1, 0)
#         fp = np.sum((y_pred == 1) & (y_true == 0))
#         tp = np.sum((y_pred == 1) & (y_true == 1))
#         fn = np.sum((y_pred == 0) & (y_true == 1))
#         tn = np.sum((y_pred == 0) & (y_true == 0))
#         fprs.append(fp / (fp + tn))
#         tprs.append(tp / (tp + fn))
#     return fprs, tprs, thresholds
#
# def plot_roc(df, feature, out_path, test_name):
#     if 'PC-Phenotype' in df.columns:
#         df = df.rename(columns={'PC-Phenotype': 'Subtype'})
#     df = df[df['Subtype'].isin(['ARPC', 'NEPC'])]
#     plt.figure(figsize=(8, 8))
#     df = df.sort_values(by='Subtype')
#     truths = pd.factorize(df['Subtype'].values)[0]
#     predictions = df[feature].values
#     fpr, tpr, thresholds = metrics.roc_curve(truths, predictions)
#     auc = metrics.auc(fpr, tpr)
#     plt.plot(fpr, tpr, label='Mean ROC (AUC = % 0.2f )' % auc, lw=2)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.suptitle(feature + '-based NEPC binary prediction ROC for ' + test_name)
#     plt.title('100% TPR threshold: ' + str(round(thresholds[list(tpr).index(1.)], 4)))
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close()
#     return [fpr, tpr, auc]
#
#
# def plot_rocs(metrics, out_path, test_name):
#     plt.figure(figsize=(8, 8))
#     for metric in metrics:
#         plt.plot(metric[1], metric[2], label=(metric[0] + ' : Mean ROC (AUC = % 0.2f )' % metric[3]), lw=2, alpha=0.7)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(test_name + ' NEPC binary prediction ROCs')
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close()
#     return


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
    pd.DataFrame(mat).to_csv(name + '/' + name + '_weights.tsv', sep="\t", header=False)
    return mat


def beta_descent(ref_dict, df, subtypes, name, order=None):
    print('Running Heterogeneous Beta Predictor  . . . ')
    features = list(ref_dict.keys())
    cols = subtypes
    cols.append('Prediction')
    samples = list(df.index)
    predictions = pd.DataFrame(0, index=df.index, columns=[subtypes[0], subtypes[1], 'TFX', 'Prediction',
                                                           subtypes[0] + '_PLL', subtypes[1] + '_PLL', 'JPLL'])
    for sample in samples:
        tfx = df.loc[sample, 'TFX']
        pdf_set_a, pdf_set_b = [], []
        for feature in features:
            try:
                feature_val = df.loc[sample, feature]
            except KeyError:
                continue
            exp_a = tfx * ref_dict[feature][subtypes[0] + '_Mean'] + (1 - tfx) * ref_dict[feature]['HD_Mean']
            std_a = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[0] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['HD_Std']))
            exp_b = tfx * ref_dict[feature][subtypes[1] + '_Mean'] + (1 - tfx) * ref_dict[feature]['HD_Mean']
            std_b = np.sqrt(tfx * np.square(ref_dict[feature][subtypes[1] + '_Std']) +
                            (1 - tfx) * np.square(ref_dict[feature]['HD_Std']))
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

    # plot
    if "NEPC" in subtypes:  # use the NEPC threshold from the Cancer Discovery paper
        predictions['NEPC'] = predictions['NEPC'] - 0.3314
        key_type = 'NEPC'
    else:
        key_type = subtypes[0]
    data = predictions.groupby(predictions[key_type]).size()
    cmap = LinearSegmentedColormap.from_list('', ['#0077BB', '#CC3311'])
    cm.register_cmap("mycolormap", cmap)
    if order == 'sorted':
        predictions = predictions.sort_index()
    elif order is not None:
        predictions = predictions.reindex(order.index)
    else:
        predictions = predictions.sort_values(key_type)
    pal = sns.color_palette("mycolormap", len(data))
    sns.set_context(rc={'patch.linewidth': 0.0})
    plt.figure(figsize=(24, 3))
    g = sns.barplot(x=predictions.index, y=key_type, hue=key_type, data=predictions, palette=pal, dodge=False)
    g.legend_.remove()
    sns.scatterplot(x=predictions.index, y=key_type, hue=key_type, data=predictions, palette=pal, s=600, legend=False)

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


def main():
    parser = argparse.ArgumentParser(description='\n### ctdPheno_v2.py ###')
    parser.add_argument('-i', '--input', help='A tidy-form, .tsv feature matrices with test samples, containing 4 '
                                             'columns with the following header: "sample", "site", "feature", and '
                                             '"value". Outputs from Triton (TritonCompositeFM.tsv directly) or Griffin '
                                             'in this format are recommended.', required=True)
    parser.add_argument('-t', '--tfx', help='.tsv file containing matched test sample names (column 1) and '
                                            'tumor fractions (column2), without a header, corresponding to the '
                                            '"samples" column(s) input for testing.', required=True)
    parser.add_argument('-r', '--reference', help='One or more tidy-form, .tsv feature matrices containing 4 columns '
                                                  'with the following header: "sample", "site", "feature", and '
                                                  '"value". Outputs from Triton (TritonCompositeFM.tsv directly) or '
                                                  'Griffin in this format are recommended.', nargs='*', required=True)
    parser.add_argument('-k', '--key', help='.tsv file containing matched sample names (column 1) and '
                                            'subtypes/categories (column2), without a header, corresponding to the '
                                            '"samples" column(s) input as reference. One included subtype must be '
                                            '"HD" (healthy/normal references).', required=True)
    parser.add_argument('-s', '--sites', help='File containing a list (row-wise) of sites to use. This file should '
                                              'NOT contain a header. DEFAULT = None (use all available sites).',
                        required=False, default=None)
    parser.add_argument('-f', '--features', help='File containing a list (row-wise) of features to use. This file '
                                                 'should NOT contain a header. '
                                                 'DEFAULT = None (use all available sites).',
                        required=False, default=None)

    args = parser.parse_args()
    input_path = args.input
    tfx_path = args.tfx
    ref_path = args.reference
    key_path = args.key
    sites_path = args.sites
    features_path = args.features

    print('### Welcome to ctdPheno. Loading data . . .')

    ordering = 'sorted'  # for plotting

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

    # prepare reference dataframe:
    ref_dfs = [pd.read_table(path, sep='\t', index_col=0) for path in ref_path]
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
    df_train = pd.merge(ref_labels, ref_df, left_index=True, right_index=True)
    df_train = df_train[df_train['Subtype'].notna()]
    subtypes = list(df_train['Subtype'].unique())
    subtypes.remove('HD')

    # prepare test dataframe:
    test_df = pd.read_table(input_path, sep='\t', index_col=0)
    if sites is not None:
        test_df = test_df[test_df['site'].isin(sites)]
    if features is not None:
        test_df = test_df[test_df['feature'].isin(features)]
    test_df['site'] = test_df['site'].str.replace('_', '-')  # ensure no _ in site names
    test_df['feature'] = test_df['feature'].str.replace('_', '-')  # ensure no _ in feature names
    test_df['cols'] = test_df['site'].astype(str) + '_' + test_df['feature']
    test_df = test_df.pivot_table(index=[test_df.index.values], columns=['cols'], values='value')
    test_labels = pd.read_table(tfx_path, sep='\t', index_col=0, names=['TFX'])
    df_test = pd.merge(test_labels, test_df, left_index=True, right_index=True)

    # run experiments
    print("Running experiments . . .")

    name = "ctdPheno"
    if not os.path.exists(name + '/'): os.makedirs(name + '/')
    df_diff = diff_exp(df_train, name, thresh=1.0)
    # N.B., by default ctdPheno will only use the features that are significantly different between the non-HD types
    metric_dict = metric_analysis(df_diff, name)
    beta_descent(metric_dict, df_test, subtypes, name, order=ordering)

    print('Finished.')


if __name__ == "__main__":
    main()

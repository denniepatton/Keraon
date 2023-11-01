#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v1.5, 08/10/2022

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import metrics
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

targets = ['KSpace', 'Extra-KSpace', 'Contra-KSpace']
colors = ['green', 'blue', 'red']
palette_fs = {targets[i]: colors[i] for i in range(len(targets))}

targets = ['Healthy', 'ARPC', 'ARPC_burden', 'NEPC', 'NEPC_burden', 'Patient', 'MIX']
colors = ['#f5f5f5', '#0077BB', '#0077BB', '#CC3311', '#CC3311', '#EE3377', '#5D3FD3']
palette_samples = {targets[i]: colors[i] for i in range(len(targets))}

targets = ['Healthy_fraction', 'ARPC_fraction', 'NEPC_fraction']
colors = ['#f5f5f5', '#0077BB', '#CC3311']
palette_stacked = {targets[i]: colors[i] for i in range(len(targets))}


def fraction_scatter(df, test_feature, out_path):
    # straight scatter
    plt.figure(figsize=(8, 6))
    test_size = 'Shift_Ratio'
    if test_feature == 'NEPC_weight':
        df = df.rename({'NEPC_weight': 'predicted'}, axis=1)
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    elif test_feature == 'NEPC_burden_predicted':
        df['predicted'] = df['TFX'] * df['NEPC_burden_predicted']
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    elif test_feature == 'TypeA_fraction':
        df = df.rename({'TypeA_fraction': 'predicted'}, axis=1)
        test_size = 'TFX'
        df['true'] = df['TypeA_true-fraction']
    else:
        df = df.rename({'NEPC_fraction': 'predicted'}, axis=1)
        test_size = 'TFX'
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    df.true = df.true.round(decimals=2)
    pc, pp = pearsonr(df['true'].values, df['predicted'].values)
    pp = ['{:.3f}'.format(pval) if pval > 0.001 else '{:.2e}'.format(pval) for pval in [pp]][0]
    model = np.polyfit(df['true'].values, df['predicted'].values, 1)
    label = 'Pearson\'s r: ' + str(round(pc, 3)) + ' (p=' + pp + ')'

    def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None:
            fig, ax = plt.subplots()
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, **kwargs)

        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.set_ylabel('Density')

        return ax

    density_scatter(df.true, df.predicted, bins=[30, 30])
    # sns.scatterplot(x='true', y='predicted', data=df, hue='FS_Region', palette=palette_fs, size=test_size, alpha=0.8)
    # sns.scatterplot(x='true', y='predicted', data=df, palette=palette_fs, size=test_size, color='gray')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.plot(df['true'].values, model[0] * df['true'].values + model[1], lw=2, color='black')
    plt.xlim([0.0, 0.3])
    plt.ylim([0.0, 0.3])
    plt.xlabel('True (total) NEPC fraction')
    plt.ylabel('Predicted (total) NEPC fraction')
    plt.title(label, fontsize=12)
    plt.suptitle('True vs Predicted NEPC fraction')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    # box plot by fraction
    plt.figure(figsize=(8, 7))
    sns.boxplot(x='true', y='predicted', data=df, dodge=False, color='gray')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('True (total) NEPC fraction')
    plt.ylabel('Predicted (total) NEPC fraction')
    plt.title(label, fontsize=12)
    plt.suptitle('True vs Predicted NEPC fraction')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(out_path.replace('.pdf', '_boxes.pdf'), bbox_inches='tight')
    plt.close()
    return


def mae_bar(df, test_feature, out_path):
    plt.figure(figsize=(8, 8))
    if test_feature == 'NEPC_weight':
        df = df.rename({'NEPC_weight': 'predicted'}, axis=1)
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    elif test_feature == 'NEPC_burden_predicted':
        df['predicted'] = df['TFX'] * df['NEPC_burden_predicted']
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    elif test_feature == 'TypeA_fraction':
        df = df.rename({'TypeA_fraction': 'predicted'}, axis=1)
        df['true'] = df['TypeA_true-fraction']
    else:
        df = df.rename({'NEPC_fraction': 'predicted'}, axis=1)
        df['true'] = df['TFX'] * df['NEPC_burden_truth']
    df.true = df.true.round(decimals=2)
    df['ae'] = (df['true'] - df['predicted']).abs()
    total_mae = df['ae'].mean()
    tfx_mae = df.groupby('TFX')['ae'].mean()
    print(tfx_mae)
    label = 'Overall MAE: ' + str(round(total_mae, 3))
    sns.boxplot(x='true', y='ae', data=df, hue='true', showmeans=True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xlabel('True (total) NEPC fraction')
    plt.ylabel('Absolute Error')
    plt.title(label, fontsize=12)
    plt.suptitle('True NEPC fraction vs Absolute Error')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return


def tfx_scatter(df_in, tfxs, out_dir):
    for tfx in tfxs:
        df = df_in[df_in.index.str.contains(tfx)]
        plt.figure(figsize=(8, 8))
        pc, pp = pearsonr(df['NEPC_burden_truth'].values, df['NEPC_burden_predicted'].values)
        pp = ['{:.3f}'.format(pval) if pval > 0.001 else '{:.2e}'.format(pval) for pval in [pp]][0]
        model = np.polyfit(df['NEPC_burden_truth'].values, df['NEPC_burden_predicted'].values, 1)
        label = 'Pearson\'s r: ' + str(round(pc, 3)) + ' (p=' + pp + ')'
        sns.scatterplot(x='NEPC_burden_truth', y='NEPC_burden_predicted', data=df, hue='FS_Region',
                        palette=palette_fs, alpha=0.8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.plot(df['NEPC_burden_truth'].values, model[0] * df['NEPC_burden_truth'].values + model[1],
                 lw=2, color='black')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('True NEPC burden')
        plt.ylabel('Predicted NEPC burden')
        plt.title(label, fontsize=12)
        plt.suptitle('True vs Predicted NEPC burden at ' + tfx)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig(out_dir + tfx.replace('.', '-') + '_NEPC-burden_vs_predicted.pdf', bbox_inches='tight')
        plt.close()
    return


def plot_roc(df, feature, out_path, test_name, tfxs=None):
    if 'PC-Phenotype' in df.columns:
        df = df.rename(columns={'PC-Phenotype': 'Subtype'})
    df = df[df['Subtype'].isin(['ARPC', 'NEPC'])]
    plt.figure(figsize=(8, 8))
    if tfxs is None:
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
    else:
        for i in range(len(tfxs)):
            if i == 0:  # lower end
                label = 'TFX <= ' + str(tfxs[i])
                df_temp = df.loc[df['TFX'] <= tfxs[i]]
            elif i == len(tfxs) + 1:  # higher end
                label = 'TFX > ' + str(tfxs[i-1])
                df_temp = df.loc[df['TFX'] > tfxs[i-1]]
            else:  # mid
                label = str(tfxs[i - 1]) + ' < TFX <= ' + str(tfxs[i])
                df_temp = df.loc[(df['TFX'] > tfxs[i - 1]) & (df['TFX'] <= tfxs[i])]
                df_temp = df_temp.sort_values(by='Subtype')
            truths = pd.factorize(df_temp['Subtype'].values)[0]
            predictions = df_temp[feature].values
            fpr, tpr, _ = metrics.roc_curve(truths, predictions)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=(label + ' : Mean ROC (AUC = % 0.2f )' % auc), lw=2, alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(feature + '-based NEPC binary prediction ROC (TFX-level)')
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


def plot_stack(df, out_path, test_name, order, thresh=None):
    # for each sub-plot, write accuracy for given threshold
    if thresh is None:  # use linear threshold
        plot_thresh = False
        df['thresh'] = df['TFX'] * 0.1982 + 0.0056
    else:
        plot_thresh = True
        df['thresh'] = thresh
    if order is None:
        df = df.sort_values(by=['NEPC_fraction', 'ARPC_fraction'], ascending=[False, False])
    else:
        df = df.reindex(list(order.index))
        df = df.dropna()
    if len(df['Subtype'].unique()) == 2:
        known_subtypes = ['ARPC', 'NEPC']
    elif len(df['Subtype'].unique()) == 3:
        known_subtypes = ['ARPC', 'NEPC', 'MIX']
    else:
        known_subtypes = ['ARPC', 'NEPC', 'MIX', 'ARPC_crpc']
    widths = [df.loc[df['Subtype'] == sub].shape[0] / df.loc[df['Subtype'] == known_subtypes[-1]].shape[0] for sub
                    in known_subtypes]
    fig, axes = plt.subplots(nrows=1, ncols=len(known_subtypes), sharey=True, gridspec_kw={'width_ratios': widths},
                             figsize=(16, 2))
    for i, subtype in enumerate(known_subtypes):
        df_sub = df.loc[df['Subtype'] == subtype]
        if subtype == 'ARPC' or subtype == 'ARPC_crpc':
            acc = round(df_sub.loc[df_sub['NEPC_fraction'] < df_sub['thresh']].shape[0] / df_sub.shape[0], 4)
            subtitle = 'ARPC (' + str(100 * acc) + '%)'
        elif subtype == 'NEPC':
            acc = round(df_sub.loc[df_sub['NEPC_fraction'] >= df_sub['thresh']].shape[0] / df_sub.shape[0], 4)
            subtitle = 'NEPC (' + str(100 * acc) + '%)'
        else:
            subtitle = 'MIXED'
        df_sub = df_sub[['NEPC_fraction', 'ARPC_fraction', 'Healthy_fraction']]
        df_sub.plot(ax=axes[i], kind='bar', stacked=True, color=palette_stacked, width=0.90, legend=False)
        axes[i].set_title(subtitle)
        if i == 0:
            axes[i].set_ylabel('total fraction')
        elif i == len(known_subtypes) - 1:
            axes[i].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        if plot_thresh:
            axes[i].axhline(y=thresh, color='black', linestyle='dashed')
    plt.setp(axes, ylim=(0, 1))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return


def main():
    tests = ['patient-WGS']
    # "patient-WGS" refers to the UW cohort (deep-WGS)
    for test_data in tests:
        roc_metrics = []
        print("Plotting (" + test_data + ")\n")
        group_sets = {'All-ATAC-TF': ['LongATAC-ADexclusive-10000TFOverlap', 'LongATAC-NEexclusive-10000TFOverlap']}
        feature_sets = ['central-flux', 'dip-width', 'central-depth', 'central-heterogeneity', 'np-amplitude', 'np-score', 'fragment-mad']
        if test_data == 'patient-WGS':
            ordering = pd.read_table('/fh/fast/ha_g/user/rpatton/ML_testing/Generative/Samples_WGS.txt',
                                     sep='\t', index_col=0, header=None)
        else:
            ordering = None
        for identifier, group in group_sets.items():
            for feature in feature_sets:
                for pred in ['basis', 'quix']:
                    name = test_data + '_' + identifier + '_' + feature + '_' + pred
                    out_dir = '/fh/fast/ha_g/user/rpatton/scripts/Quix/'\
                              + test_data + '_' + identifier + '_' + feature + '/'
                    path = out_dir + test_data + '_' + identifier + '_' + feature + '_' + pred + '-predictions.tsv'
                    print("Loading " + path)
                    df = pd.read_table(path, sep='\t', index_col=0)
                    df = df.rename({'NEPC_burden_x': 'NEPC_burden_truth',
                                    'NEPC_burden_y': 'NEPC_burden_predicted',
                                    'NEPC_burden': 'NEPC_burden_predicted',
                                    'NEPC_weight': 'NEPC_fraction',
                                    'ARPC_weight': 'ARPC_fraction',
                                    'Healthy_weight': 'Healthy_fraction'}, axis=1)
                    if 'Triplet' in test_data:  # benchmarking analysis
                        # df = df[df['TFX'] >= 0.1]
                        fraction_scatter(df, 'NEPC_fraction', out_dir + 'NEPC-fractions_vs_predicted.pdf')
                        mae_bar(df, 'NEPC_fraction', out_dir + 'NEPC-fractions_vs_predicted_MAE.pdf')
                        # tfx_scatter(df, ['TF0.10', 'TF0.20', 'TF0.30'], out_dir + 'basis_')
                        df = df[df.index.str.contains('|'.join(['NEPC_1.0', 'NEPC_0.0']))]
                        conds = [(df['NEPC_burden_truth'] == 0), (df['NEPC_burden_truth'] > 0)]
                        vals = ['ARPC', 'NEPC']
                        df['Subtype'] = np.select(conds, vals)
                        tfxs = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
                        _ = plot_roc(df, 'NEPC_fraction', out_dir + test_data + '_' + identifier + '_NEPC-fraction_ROC.pdf',
                                     test_data, tfxs=tfxs)
                        _ = plot_roc(df, 'NEPC_burden_predicted',
                                     out_dir + test_data + '_' + identifier + '_NEPC-burden_ROC.pdf', test_data, tfxs=tfxs)
                    elif test_data == 'SimulatedFeatures':  # benchmarking analysis
                        fraction_scatter(df, 'TypeA_fraction', out_dir + 'TypeA-fractions_vs_predicted.pdf')
                        mae_bar(df, 'TypeA_fraction', out_dir + 'TypeA-fractions_vs_predicted_MAE.pdf')
                        # tfx_scatter(df, ['TF0.10', 'TF0.20', 'TF0.30'], out_dir + 'basis_')
                    else:  # general predictive
                        # only want to plot stacked bars for CRPC samples; assign temp subtype so excluded from ROC:
                        df.loc[df.index.str.contains('CRPC'), 'Subtype'] = 'ARPC_crpc'
                        # plot_roc(df, 'NEPC_fraction', out_dir + 'TFX-based_NEPC-fraction_ROC.pdf', 'temp',
                        #          tfxs=[0.1, 0.3])  # this plots separate ROCs by TFX threshold
                        temp_metrics = plot_roc(df, 'NEPC_fraction', out_dir + test_data + '_' + identifier +
                                                '_' + pred + '_NEPC-fraction_ROC.pdf', test_data)
                        _ = plot_roc(df, 'NEPC_burden_predicted', out_dir + test_data + '_' + identifier +
                                     '_' + pred + '_NEPC-burden_ROC.pdf', test_data)
                        plot_stack(df, out_dir + test_data + '_' + identifier + '_' + pred + '_StackedBar.pdf',
                                   test_data, ordering, thresh=0.028)
                        roc_metrics.append([name] + temp_metrics)
            if test_data != 'Triplet' and test_data != 'SimulatedFeatures':
                plot_rocs(roc_metrics, test_data + '_AllROCs.pdf', test_data)


if __name__ == "__main__":
    main()

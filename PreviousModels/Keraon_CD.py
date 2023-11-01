# Robert Patton, rpatton@fredhutch.org
# v3.0, 09/29/2022

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import stdout
from scipy.stats import mannwhitneyu, shapiro
from statsmodels.stats.multitest import fdrcorrection


def diff_exp(df, name, subtypes, thresh=0.05):
    """
    Calculate the FDR-corrected Mann-Whitney U differential p-values for a specific comparison for each feature
        Parameters:
           df: pandas dataframe
           name (string): name to write results locally
           subtypes (list): two variables in category to compute fold change between (1 over 2)
           thresh (float): the maximum p-value limit to return
        Returns:
           out_df: the input df, restricted to significant features
           df_lpq: p-values for each feature
    """
    print('Conducting differential expression analysis . . .')
    df_t1 = df.loc[df['Subtype'] == subtypes[0]].drop('Subtype', axis=1)
    df_t2 = df.loc[df['Subtype'] == subtypes[1]].drop('Subtype', axis=1)
    df_lpq = pd.DataFrame(index=df_t1.transpose().index, columns=['ratio', 'p-value'])
    for roi in list(df_t1.columns):
        x, y = df_t1[roi].values, df_t2[roi].values
        if np.count_nonzero(~np.isnan(x)) < 2 or np.count_nonzero(~np.isnan(y)) < 2:
            continue
        df_lpq.at[roi, 'ratio'] = np.mean(x)/np.mean(y)
        df_lpq.at[roi, 'p-value'] = mannwhitneyu(x, y)[1]
    df_lpq['p-adjusted'] = fdrcorrection(df_lpq['p-value'])[1]
    df_lpq = df_lpq.sort_values(by=['p-adjusted'])
    df_lpq = df_lpq.infer_objects()
    df_lpq.to_csv(name + '/' + name + '_differential.tsv', sep="\t")
    features = list(df_lpq[(df_lpq['p-adjusted'] < thresh)].index)
    return pd.concat([df.iloc[:, :1], df.loc[:, df.columns.isin(features)]], axis=1, join='inner'), df_lpq


def norm_exp(df, name, thresh=0.05):
    """
    Calculate the Shapiro p-values (likelihood data does NOT belong to a normal distribution) for each subtype in each
    feature and return the dataframe with only features where each subtype is normally distributed
        Parameters:
           df: pandas dataframe
           name (string): name to write results locally
           thresh (float): the minimum p-value limit to return
        Returns:
           out_df: the input df, restricted to normal features
           df_lpq: p-values for each feature (1 - mean value across subtypes)
    """
    print('Conducting normal expression analysis . . .')
    phenotypes = list(df['Subtype'].unique())
    pheno_dfs = [df.loc[df['Subtype'] == phenotype].drop('Subtype', axis=1) for phenotype in phenotypes]
    df_lpq = pd.DataFrame(index=pheno_dfs[0].transpose().index,
                          columns=[phenotype + '_p-value' for phenotype in phenotypes])
    for roi in list(pheno_dfs[0].columns):
        for pheno_index, pheno_df in enumerate(pheno_dfs):
            values = pheno_df[roi].values
            shapiro_test = shapiro(values)
            df_lpq.at[roi, phenotypes[pheno_index] + '_p-value'] = shapiro_test.pvalue
    df_lpq = df_lpq.loc[(df_lpq[df_lpq.columns] > thresh).all(axis=1)]
    df_lpq['p-adjusted'] = df_lpq.mean(axis=1)
    df_lpq.loc[:, 'p-adjusted'] = df_lpq['p-adjusted'].apply(lambda x: 1 - x)
    df_lpq.to_csv(name + '/' + name + '_shapiros.tsv', sep="\t")
    features = list(df_lpq.index)
    return pd.concat([df.iloc[:, :1], df.loc[:, df.columns.isin(features)]], axis=1, join='inner'), df_lpq


def corr_exp(df, name, thresh=0.95, df_ref_1=None, df_ref_2=None):
    """
    Calculate the correlation matrix for all features and return only one from each group of correlated features
    (those meeting the threshold for correlation). If an additional dataframe with p-values for each feature is
    supplied (e.g. differential expression data) the feature with the lowest p-value in each comparison is
    returned. If there is a tie in p-values another dataframe may be supplied with p-values to break ties (e.g.
    normalcy among classes p-values).
        Parameters:
           df: pandas dataframe
           name (string): name to write results locally
           thresh (float): the maximum correlation limit to return
           df_ref_1: a dataframe with features and 'p-adjusted'
           df_ref_2: a dataframe with features and 'p-adjusted'
        Returns:
           out_df: the input df, restricted to non-correlated features
    """
    print('Conducting correlation analysis . . .')
    correlated_features = set()
    correlation_matrix = df.corr()
    correlation_matrix.to_csv(name + '/' + name + '_correlations.tsv', sep="\t")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > thresh:
                row_name = correlation_matrix.index[i]
                col_name = correlation_matrix.columns[j]
                if df_ref_1 is not None:
                    if df_ref_1.at[col_name, 'p-adjusted'] < df_ref_1.at[row_name, 'p-adjusted']:
                        correlated_features.add(row_name)
                    elif df_ref_1.at[col_name, 'p-adjusted'] > df_ref_1.at[row_name, 'p-adjusted']:
                        correlated_features.add(col_name)
                    elif df_ref_2 is not None:
                        if df_ref_2.at[col_name, 'p-adjusted'] < df_ref_2.at[row_name, 'p-adjusted']:
                            correlated_features.add(row_name)
                        else:
                            correlated_features.add(col_name)
                    else:
                        correlated_features.add(col_name)
                else:
                    correlated_features.add(col_name)
    df_out = df.drop(labels=correlated_features, axis=1)
    return df_out


def plot_feature_space(df_ref, df_test, axes, palette, label_anchor_points=False, label_test_points=False):
    """
        Plot the (2D) feature space given in axes, with the anchor data as kernels and faded points, and the sample
        data as opaque points scaled by tumor fraction.
            Parameters:
               df_ref: pandas dataframe with anchor data
               df_test: pandas dataframe with test/sample data
               axes ([x, y]): features to plot - must match df column names
               palette ({dict}): colors for each passed subtype
               label_anchor_points (bool): whether to label anchor points with sample names
               label_test_points (bool): whether to label test points with sample names
            Returns:
               fig: matplotlib.pyplot.figure object
        """
    plt.close('all')
    plt.figure(figsize=(8, 8))
    sns.kdeplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', fill=True, palette=palette)
    sns.scatterplot(data=df_ref, x=axes[0], y=axes[1], hue='Subtype', palette=palette, alpha=0.2)
    if 'Subtype' in df_test and 'TFX' in df_test:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], hue='Subtype', size='TFX', palette=palette)
    elif 'TFX' in df_test:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], color='Purple', size='TFX')
    else:
        sns.scatterplot(data=df_test, x=axes[0], y=axes[1], color='Purple')
    if label_anchor_points:
        for x, y, label in zip(df_ref.loc[:, axes[0]], df_ref.loc[:, axes[1]], df_ref.index.values):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=4)
    if label_test_points:
        for x, y, label in zip(df_test.loc[:, axes[0]], df_test.loc[:, axes[1]], df_test.index.values):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=3)
    plt.title('Anchor Distributions and Test Points', size=14, y=1.02)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt.gcf()


def _get_intersect(points_a, points_b):
    """
        Returns the point of intersection of the line passing through points_a and the line passing through points_b
        (2D) or the line passing through points_a and the plane defined by points_b (3D).
            Parameters:
                points_a [a1, a2]: points defining line a
                points_b [b1, b2, (b3)]: points defining line (plane) b
            Returns:
                out_point [x, y, . . .]: the point of intersection in base coordinates
    """
    if len(points_b) == 2:
        s = np.vstack([points_a[0], points_a[1], points_b[0], points_b[1]])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:  # lines are parallel - SHOULD be impossible given constraints to call this function
            return float('inf'), float('inf')
        return np.array([x/z, y/z])
    else:
        # below is untested
        norm = np.cross(points_b[2] - points_b[0], points_b[1] - points_b[0])
        w = points_a[0] - points_b[0]
        si = -norm.dot(w) / norm.dot(points_a[1] - points_a[0])
        psi = w + si * (points_a[1] - points_a[0]) + points_a[0]
        return psi


def keraon(ref_df, df, name, palette, benchmarking=False):
    """
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

# Keraon <img src="misc/logo_v1.png" width="140" align="left">
As a tool for cancer subtype prediction, Keraon uses features derived from cell-free DNA (cfDNA) in conjunction
with PDX reference models to perform both classification and heterogenous phenotype fraction estimation.

_Keraon_ (Ceraon) is named for the Greek god of the ritualistic mixing of wine.  
Like Keraon, this tool knows what went into the mix.
<br/><br/>

## Description
Keraon utilizes features derived from cfDNA WGS to perform cancer phenotype classification (formerly ctdPheno) and heterogeneous/fractional
subtype mixture component estimation. To do this Keraon uses a panel of healthy donor and PDX samples which span the subtypes of interest.
Bioinformatically pure circulating tumor DNA (ctDNA) features garnered from the PDX models, in conjunction with matched features from healthy
donor cfDNA, are used to construct a latent space on which purity-adjusted predictions are based. Keraon yields both categorical classification
scores and mixture estimates, and includes a de novo feature selection option based on simplex volume maximization which works synergistically
with both methods.

### Outputs

When running Keraon, a `results/` directory is generated to store the output files. This directory contains several subfolders, each holding specific results from the analysis:

#### `results/FeatureSpace/`

This folder contains files related to the feature space transformation and preliminary analysis:

- **`cleaned_anchor_reference.pkl`**: A binary file storing the reference dataframe, min, and range dictionaries after preprocessing.
- **`final-features.tsv`**: A tab-separated values file containing the final features selected for the analysis.
- **`PCA-1_initial.pdf`**: A PDF file showing the PCA plot before any feature selection.
- **`PCA-2_post-norm-restrict.pdf`**: A PDF file showing the PCA plot after normal feature restriction (if performed).
- **`PCA-3_post-MSV.pdf`**: A PDF file showing the PCA plot after maximal simplex volume feature selection (if performed).
- **`PCA-4_post-MSV_wSamples.pdf`**: A PDF file showing the PCA plot with sample data included after feature selection.

#### `results/ctdPheno/`

This folder contains results related to the ctdPheno classification analysis:

- **`ctdPheno_class-predictions.tsv`**: A tab-separated values file containing the predicted classifications and relative log likelihoods (RLLs) for each sample.
- **`ROC.pdf`**: A PDF file showing the ROC curve for binary classification performance (if known truth is provided).
- **`<subtype>_predictions.pdf`**: PDF files for each subtype showing stick and ball plots of predictions based on RLLs.

#### `results/Keraon/`

This folder contains results related to the Keraon mixture estimation analysis:

- **`Keraon_mixture-predictions.tsv`**: A tab-separated values file containing the predicted mixture fractions and burdens for each subtype in each sample.
- **`ROC_fraction.pdf`**: A PDF file showing the ROC curve for the mixture fraction classification performance (if known truth is provided).
- **`<subtype>_fraction_predictions.pdf`**: PDF files for each subtype showing stacked bar plots of the mixture fractions.

### Uses

Keraon's primary use case is subtyping late-stage cancers and detecting potential trans-differentiation events. See published results for
classifying and estimating fractions of castration-resistent prostate cancer (CRPC) adenocarcinoma (ARPC) from neuroendocrine-like (NEPC) ([publications](#publications)).

### Publications

[Nucleosome Patterns in Circulating Tumor DNA Reveal Transcriptional Regulation of Advanced Prostate Cancer Phenotypes](https://doi.org/10.1158/2159-8290.CD-22-0692)

## Usage

Keraon can be run on the command line using the following arguments:

### Inputs to Keraon.py:

```
-i, --input           : A tidy-form, .tsv feature matrix with test samples. Should contain 4 columns: "sample", "site", "feature", and "value" (path, required)  
-t, --tfx             : .tsv file with test sample names and tumor fractions. If a third column with "true" subtypes/categories is passed, additional validation will be performed (path, required)  
-r, --reference       : One or more tidy-form, .tsv feature matrices. Should contain 4 columns: "sample", "site", "feature", and "value" (paths, required)  
-k, --key             : .tsv file with sample names and subtypes/categories. One subtype must be "Healthy" (path, required)  
```

### Inputs (extra details):

**input:** The input.tsv file should be a tidy-formatted feature matrix with columns "sample", "site", "feature", and "value". Each row represents a specific feature value for a given sample and site.

**tfx:** The tfx.tsv file should contain test sample names matching input.tsv and their corresponding tumor fractions. If an additional third column with true subtypes/categories is present, it enables additional validation during processing.

**reference:** One or more ref.tsv files formatted similarly to the input file, containing matching reference feature values with the same four columns.

**key:** This key.tsv file must include sample names found in ref.tsv(s) and their corresponding subtypes/categories, with at least one subtype labeled as "Healthy".

### Optional arguments:

```
-d, --doi             : Disease/subtype of interest for plotting and calculating ROCs - must be present in the key (default: 'NEPC')  
-x, --thresholds      : Tuple containing thresholds for calling the disease of interest (default: (0.5, 0.0311))  
-f, --features        : File with a list of site_feature combinations to restrict to. Sites and features should be separated by an underscore (path, optional)  
-s, --svm_selection   : Flag indicating whether to TURN OFF SVM feature selection method (default: True)  
-p, --palette         : .tsv file with matched categories/samples and HEX color codes. Subtype/category names must match those passed with the -t and -k options (path, optional)  
```

### Optional Inputs (extra details):

**features:** This file lists specific site_feature combinations to restrict the analysis to, with sites and features separated by an underscore. Example entries include `AR_central-depth`, `ASCL1_central-depth`, `ATAC-AD-Sig_fragment-mean`, and `ATAC-NE-Sig_fragment-mean`.

**palette:** A .tsv file that provides HEX color codes for categories/samples. The categories/subtype names in this file must match those in the key file.

Examples of correctly formatted feature, key, palette, and tfx files can be found in Keraon/config

### Contained Scripts:

**Keraon.py** | primary script containing both classification and mixture estimation methods
**keraon_helpers.py** | contains helper functions called by Keraon.py  
**keraon_plotters.py** | combines helper functions for plotting outputs of Keraon.py

#### keraon_helpers.py

This script contains functions to perform feature selection, statistical analysis, and classification predictions. Key functionalities include calculating the log likelihoods for subtype prediction using a multivariate normal distribution, optimizing tumor fraction (TFX) to maximize log likelihoods, orthogonalizing feature space vectors using the Gram-Schmidt process, and transforming sample vectors to a new basis. The script also provides methods for standardizing data, evaluating features based on simplex volume maximization, and calculating specificity and sensitivity for classification thresholds. It also includes tools for analyzing normal expression of features and generating ROC curves to assess classification performance.

#### keraon_plotters.py

This script contains plotting functions designed to visualize the final predictions from Keraon. These include generating PCA plots, creating stick and ball plots of subtype predictions from ctdPheno, plotting stacked bar plots to display the fraction and burden of each subtype in a sample for fraction estimation, and constructing ROC curves to evaluate the performance of binary classifiers.

## Methodology

### Simplex Volume Maximization (feature selection)

Simplex volume maximization aims to identify a subset of features within a high-dimensional dataset that maximizes the volume of the simplex formed by the mean vectors of different classes while minimizing the number of features used and the spread at each vertex.
This approach therefore identifies features which are highly differentiating for the given classes in a higher-dimensional space, regardless of whether or not those features provide strong seperation individually. This process is also synergistic with fractional phenotypes estimation performed by Keraon, which uses the simplex to infer class componenets.


As a precursor to the process, if more than 10 features are passed they will be screened for normalcy within each class using the Shapiro-Wilk test. See keraon_helpers.py norm_exp() for specifics. All features, regardless of whether simplex volume maximization is performed, are min-max standardized before any downstream processes.

#### Simplex Volume Calculation

Given \( n \) classes (or subtypes), each represented by a mean vector in a high-dimensional feature space, the volume \( V \) of the simplex formed by these vectors can be calculated recursively using the heights and bases of lower-dimensional simplexes.

For a set of \( n+1 \) vectors \(\mathbf{v}_0, \mathbf{v}_1, \ldots, \mathbf{v}_n \), the volume \( V \) of the simplex can be calculated as:

\[
V = \frac{1}{n} \times \text{Base} \times \text{Height}
\]

where:
- The **Base** is the volume of the \((n-1)\)-dimensional simplex formed by the first \( n \) vectors.
- The **Height** is the perpendicular distance from the \( n \)-th vector to the base simplex.

Mathematically, if \( V_{n-1} \) is the volume of the \((n-1)\)-dimensional simplex, and \( \mathbf{v}_0, \mathbf{v}_1, \ldots, \mathbf{v}_{n-1} \) are the vectors forming the base simplex, then the height \( H \) is given by:

\[
H = \frac{\|\mathbf{v}_n - \mathbf{v}_0\|}{n}
\]

Thus, the volume \( V \) of the \( n \)-dimensional simplex is:

\[
V = \frac{1}{n} \times V_{n-1} \times \frac{\|\mathbf{v}_n - \mathbf{v}_0\|}{n}
\]

#### Objective Function

The objective function to be maximized is the volume of the simplex formed by the mean vectors of the classes, weighted by a penalty factor to account for irregularity and a scaling factor to ensure positive semi-definiteness of covariance matrices. The objective function can be expressed as:

\[
\text{Objective} = \frac{V}{\text{Penalty} \times \text{Scale Factor}}
\]

#### Penalty Calculation

The penalty is introduced to penalize irregular simplices (i.e., those that are not equilateral). It is calculated as the ratio of the maximum to the minimum pairwise distance between the mean vectors. Given the mean vectors \(\mathbf{v}_i\) and \(\mathbf{v}_j\):

\[
\text{Penalty} = \frac{\max(\|\mathbf{v}_i - \mathbf{v}_j\|)}{\min(\|\mathbf{v}_i - \mathbf{v}_j\|)}
\]

where \(\|\mathbf{v}_i - \mathbf{v}_j\|\) denotes the Euclidean distance between pairs of mean vectors.

#### Scale Factor Calculation

The scale factor is used to ensure that the covariance matrices of the selected features are positive semi-definite and to down-weight
simplices which have large variances along edges. It is calculated based on the standard deviations of the projected data onto the edges of the simplex. For each edge \( \mathbf{e}_i \) formed by the mean vectors, the variance of the projected data is computed. The scale factor is the sum of the products of these variances:

1. **Edge Calculation**: For each pair of mean vectors \(\mathbf{v}_i\) and \(\mathbf{v}_j\), compute the edge:

\[
\mathbf{e}_{ij} = \mathbf{v}_i - \mathbf{v}_j
\]

2. **Projection and Variance Calculation**: Project the data onto each edge and compute the variance. For a given edge \(\mathbf{e}_{ij}\), the projection of the data matrix \(\mathbf{X}\) is:

\[
\text{Proj}(\mathbf{X}, \mathbf{e}_{ij}) = \mathbf{X} \cdot \mathbf{e}_{ij}
\]

The variance of the projections is:

\[
\text{Var}(\text{Proj}(\mathbf{X}, \mathbf{e}_{ij}))
\]

3. **Vertex Standard Deviation Volumes**: Calculate the "volume" of the standard deviations at each vertex. For each vertex \(\mathbf{v}_i\), the product of the variances of the edges connected to it is computed:

\[
\text{Vertex StDev Volume}_i = \prod_{\mathbf{e}_{ij}} \text{Var}(\text{Proj}(\mathbf{X}, \mathbf{e}_{ij}))
\]

4. **Scale Factor**: The total scale factor is the sum of the vertex standard deviation volumes:

\[
\text{Scale Factor} = \sum_{i} \text{Vertex StDev Volume}_i
\]

#### Greedy Maximization Algorithm

The greedy maximization algorithm is used to iteratively select features that maximize the objective function. Starting with an empty set of features, the algorithm evaluates all possible feature combinations, selects the base combination that maximizes the objective function, and iteratively adds features until no further improvements can be made.

1. **Initialization**: Start with an empty feature mask and initialize the best value to negative infinity.
2. **Combinatorial Search**: Evaluate all possible masks with a minimal number of features set to 1. This is done to establish an initial subset of features.
3. **Greedy Search**: Iteratively add features that maximize the objective function until no further improvements can be made.

The final result is a subset of features that maximizes the simplex volume while maintaining a balance between feature space complexity and discriminative power.

### Classification ("ctdPheno")

The `ctdpheno` function calculates TFX-shifted multivariate group identity relative log likelihoods (RLLs) for each sample in a dataset. The function uses a reference dataset containing subtype information and feature values to make predictions about the subtype and calculate RLLs for each sample.

#### Multivariate Normal Distribution

The log likelihood of the observed feature values given the TFX and the mean and covariance matrices of the subtypes is calculated using the multivariate normal probability density function (pdf). For a sample \( \mathbf{x} \), the multivariate normal pdf is given by:

\[
\mathcal{L}(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)
\]

where:
- \( \mu \) is the mean vector.
- \( \Sigma \) is the covariance matrix.
- \( k \) is the number of features.

The log of the likelihood (log likelihood) is:

\[
\log \mathcal{L}(\mathbf{x} \mid \mu, \Sigma) = -\frac{1}{2} \left[ (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) + \log |\Sigma| + k \log (2\pi) \right]
\]

#### Mixture Model

For a given sample \( \mathbf{x} \) with tumor fraction TFX, the mean vector of the class mixtures \( \mu_{\text{mixture}} \) are calculated as:

\[
\mu_{\text{mixture}} = \text{TFX} \cdot \mu_{\text{subtype}} + (1 - \text{TFX}) \cdot \mu_{\text{healthy}}
\]

The covariance matrix of the mixture \( \Sigma_{\text{mixture}} \) is simplified as an identity matrix by default:

\[
\Sigma_{\text{mixture}} = I
\]

This is done to account for large disparities in sample size between anchor classes. The log likelihood for each subtype \( i \) is:

\[
\log \mathcal{L}_i = -\frac{1}{2} \left[ (\mathbf{x} - \mu_{\text{mixture}, i})^T I^{-1} (\mathbf{x} - \mu_{\text{mixture}, i}) + \log |I| + k \log (2\pi) \right]
\]

Since \(\Sigma_{\text{mixture}} = I\) and \(\log |I| = 0\), this simplifies to:

\[
\log \mathcal{L}_i = -\frac{1}{2} \left[ (\mathbf{x} - \mu_{\text{mixture}, i})^T (\mathbf{x} - \mu_{\text{mixture}, i}) + k \log (2\pi) \right]
\]

#### Optimizing TFX

If the initial log likelihoods are not real, the function automatically optimizes the TFX to maximize the total log likelihood. This is achieved by directly searching over a range of TFX values and selecting the one that maximizes the likelihood:

\[
\text{TFX}_{\text{optimal}} = \arg \max_{\text{TFX}} \sum_{i=1}^{n} \log \mathcal{L}(\mathbf{x}_i \mid \mu_{\text{mixture}}, \Sigma_{\text{mixture}})
\]

#### Weights and Predictions

The function calculates the weights/scores for each subtype using the softmax function applied to the log likelihoods:

\[
w_i = \frac{e^{\log \mathcal{L}_i}}{\sum_{j} e^{\log \mathcal{L}_j}}
\]

where \( w_i \) is the weight for subtype \( i \).

Barring validation in an additional dataset using an identical reference set of anchors in order to determine an optimal scoring threshold, the prediction for each sample is the subtype with the highest weight.

### Mixture Estimation ("Keraon")

The `keraon` function transforms the feature space of a dataset into a new basis defined by the mean vectors of different subtypes across the selected features, creating a simplex meant to encompass the space connecting healthy, also from the reference, to the subtypes of interest. This transformation enables the calculation of the component fraction of each subtype in a sample's feature vector and thus the "burden" of each subtype, which is the product of the sample's tumor fraction (TFX) and its fraction of the subtype.

#### Basis Vector Calculation

1. **Mean Vectors**: For each subtype \( i \), the mean vector \( \mu_i \) is calculated from the reference data:

\[
\mu_i = \frac{1}{n_i} \sum_{j=1}^{n_i} \mathbf{x}_j^{(i)}
\]

where \( n_i \) is the number of samples in subtype \( i \), and \( \mathbf{x}_j^{(i)} \) is the \( j \)-th sample of subtype \( i \).

2. **Directional Vectors**: Subtract the mean vector of the 'Healthy' subtype \( \mu_{\text{Healthy}} \) from the mean vectors of the other subtypes to get directional vectors from healthy to each subtype:

\[
\mathbf{v}_i = \mu_i - \mu_{\text{Healthy}}
\]

3. **Orthogonal Basis Vectors**: Apply the Gram-Schmidt process to the directional vectors \( \mathbf{v}_i \) to obtain an orthogonal basis, with healthy at the origin and each axis defining a direction along a subtype:

\[
\mathbf{u}_i = \frac{\mathbf{v}_i - \sum_{j=1}^{i-1} \left( \frac{\mathbf{v}_i \cdot \mathbf{u}_j}{\mathbf{u}_j \cdot \mathbf{u}_j} \right) \mathbf{u}_j}{\left\| \mathbf{v}_i - \sum_{j=1}^{i-1} \left( \frac{\mathbf{v}_i \cdot \mathbf{u}_j}{\mathbf{u}_j \cdot \mathbf{u}_j} \right) \mathbf{u}_j \right\|}
\]

#### Sample Transformation

1. **Transform to New Basis**: For each sample vector \( \mathbf{x} \), transform the vector to the new basis by subtracting \( \mu_{\text{Healthy}} \) and projecting onto the orthogonal basis:

\[
\mathbf{y} = \mathbf{x} - \mu_{\text{Healthy}}
\]

\[
\mathbf{p} = \mathbf{y} \cdot \mathbf{U}^T
\]

where \( \mathbf{U} \) is the matrix of orthogonal basis vectors.

2. **Regions of the Feature Space**: Determine the region of the feature space based on the transformed sample vector \( \mathbf{p} \):

- **Simplex**: All components of \( \mathbf{p} \) are positive.
- **Contra-Simplex**: All components of \( \mathbf{p} \) are negative.
- **Outer-Simplex**: Mixed positive and negative components.

3. **Adjust for Contra/Outer-Simplex**: If the sample vector is in the contra-simplex region, negate the vector and scale it to match the TFX (this method is not well validated, so please watch out for samples falling in the contra-simplex space):

\[
\mathbf{p} = -\mathbf{p}
\]

\[
\mathbf{p} = \left( \frac{\mathbf{p}}{\|\mathbf{p}\|} \right) \cdot \text{TFX}
\]

If the sample vector has some, but not all, negative components, those are zeroed out on the assumption that they do not imply any fraction of that subtype.

#### Fraction and Burden Calculation

1. **Projected and Orthogonal Lengths**: Calculate the projected length \( \|\mathbf{p}\| \) of the vector which lies in the feature space defined by the simplex, and the orthogonal length \( \|\mathbf{d}\| \) of the vector which completes the sample vector along an axis orthogonal to the feature space:

\[
\|\mathbf{p}\| = \sqrt{\sum_{i=1}^{k} p_i^2}
\]

\[
\|\mathbf{d}\| = \sqrt{\sum_{i=1}^{k} d_i^2}
\]

where \( \mathbf{d} \) is the difference vector between the original vector and its projection.

2. **Component Fractions**: Normalize the projected and orthogonal components to get the fraction of each subtype and off-target fraction:

\[
\text{comp\_fraction}_i = \frac{p_i}{\|\mathbf{p}\|} \quad \text{for each } i
\]

\[
\text{off\_target\_fraction} = \frac{\|\mathbf{d}\|}{\|\mathbf{p}\| + \|\mathbf{d}\|}
\]

3. **Subtype Burdens**: Calculate the burden of each subtype as the product of TFX and the fraction of the subtype:

\[
\text{burden}_i = \text{TFX} \cdot \text{comp\_fraction}_i
\]

4. **Normalize Fractions**: Ensure the sum of fractions is 1:

\[
\text{comp\_fraction} = \frac{\text{comp\_fraction}}{\sum \text{comp\_fraction}}
\]

## Requirements

Keraon uses mostly standard library imports like NumPy and SciPy and has been tested on Python 3.9 and 3.10.
To create a tested environment using the provided `requirements.txt` file generated by Conda, follow these steps:

1. **Install Conda**: Ensure you have Conda installed. You can download and install Conda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

2. **Download the `requirements.txt` File**: Make sure the `requirements.txt` file is in your current working directory.

3. **Create the Environment**: Use the following command to create a new Conda environment named `<env>` using the dependencies specified in the `requirements.txt` file:

    ```bash
    conda create --name <env> --file requirements.txt
    ```

   Replace `<env>` with your desired environment name.

4. **Activate the Environment**: Once the environment is created, activate it using:

    ```bash
    conda activate <env>
    ```

   Again, replace `<env>` with the name of your environment.

5. **Verify the Installation**: To ensure all packages are installed correctly, you can list the packages in the environment using:

    ```bash
    conda list
    ```

This process will set up a Conda environment with all the necessary packages and versions specified in `requirements.txt`

## Contact
If you have any questions or feedback, please contact me at:  
**Email:** <rpatton@fredhutch.org>

## Acknowledgments
Keraon is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.  

## Software License
Keraon
Copyright (C) 2022 Fred Hutchinson Cancer Center

You should have received a copy of The Clear BSD License along with this program.
If not, see <https://spdx.org/licenses/BSD-3-Clause-Clear.html>.

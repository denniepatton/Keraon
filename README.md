# Keraon <img src="misc/logo_v1.png" width="140" align="left">
As a tool for cancer subtype prediction, Keraon uses features derived from cell-free DNA (cfDNA) in conjunction
with PDX reference models to perform both classification and heterogenous phenotype fraction estimation.

_Keraon_ (Ceraon) is named for the Greek god of the ritualistic mixing of wine.  
Like Keraon, this tool knows what went into the mix.
<br/><br/>

## Description
Keraon utilizes features derived from cfDNA WGS to perform cancer phenotype classification (ctdPheno) and heterogeneous/fractional
phenotype mixture estimation. To do this Keraon uses a panel of healthy donor and PDX samples which span the subtypes of interest as anchors.
Bioinformatically pure circulating tumor DNA (ctDNA) features garnered from the PDX models, in conjunction with matched features from healthy
donor cfDNA, are used to construct a latent space on which purity-adjusted predictions are based. Keraon yields both categorical classification
scores and mixture estimates, and includes a de novo feature selection option based on simplex volume maximization which works synergistically
with both methods.

### Outputs

When running Keraon, a `results/` directory is generated to store the output files. This directory contains several subfolders, each holding specific results from the analysis:

```
results/
├── feature_analysis/
│   ├── reference_simplex.pickle            # constructed reference DF + scaling params
│   ├── pre-selected_site_features.tsv      # final reference features post-scaling used by the model (if -f/--features are provided)
│   ├── SVM_site_features.tsv               # final reference features post-scaling used by the model (chosen by SVM)
│   ├── PCA_pre-selected_features.pdf       # using pre-defined features (if -f/--features are provided)
│   ├── PCA_initial.pdf                     # before feature selection, if using SVM
│   ├── PCA_post‑SVM.pdf                    # after SVM, if using SVM
│   ├── PCA_final-basis_wTestSamples.pdf    # using final feature set, with test samples projected in
│   └── feature_distributions/
│       ├── reference_features/             # per‑feature PDFs post-scaling (reference)
│       ├── test_features/                  # per‑feature PDFs post-scaling (test)
│       └── final-basis_site-features/      # per‑site/per-feature PDFs used by the model (reference + test)
├── ctdPheno_class‑predictions/
│   ├── ctdPheno_class‑predictions.tsv      # RLL-based scoring and class predictions
│   ├── ROC.pdf                             # optional ROC (if truth is provided)
│   └── <subtype>_predictions.pdf           # stick‑and‑ball visualisation
└── keraon_mixture‑predictions/
    ├── Keraon_mixture‑predictions.tsv      # subtype fractions & burdens
    ├── ROC_fraction.pdf                    # optional ROC (if truth is provided)
    └── <subtype>_fraction_predictions.pdf  # stacked‑bar burdens
```

### Uses

Keraon's primary use case is subtyping late-stage cancers and detecting potential trans-differentiation events. See published results for
classifying and estimating fractions of castration-resistent prostate cancer (CRPC) adenocarcinoma (ARPC) from neuroendocrine-like (NEPC) ([publications](#publications)).

### Publications

[Nucleosome Patterns in Circulating Tumor DNA Reveal Transcriptional Regulation of Advanced Prostate Cancer Phenotypes](https://doi.org/10.1158/2159-8290.CD-22-0692)

## Usage

Keraon can be run on the command line using the following arguments (examples of correctly formatted feature, key, palette, and tfx files can be found in Keraon/config):

### Inputs to Keraon.py:

```
-i, --input           : A tidy-form, .tsv feature matrix with test samples. Should contain 4 columns: "sample", "site", "feature", and "value".
                        Sites and features most correspond to those passed with the reference samples or basis
-t, --tfx             : .tsv file with test sample names and estimated tumor fractions. If a third column with "true" subtypes/categories is passed, additional validation will be performed.
                        If blanks/nans are passed for tfx for any samples, they will be treated as unknowns and tfx will be predicted (less accurate).
                        If multiple subtypes are present, they should be separated by commas, e.g. "ARPC,NEPC,DNPC".  
-r, --reference       : Either a single, pre-generated reference_simplex.pickle file or one or more tidy-form, .tsv feature matrices (in which case a reference key must also be passed with -k).
                        Tidy files will be used to generate a basis and should contain 4 columns: "sample", "site", "feature", and "value". 
-k, --key             : .tsv file with reference sample names, subtypes/categories, and purity. One subtype must be "Healthy" with purity=0.
```

### Inputs (extra details):

**input:** The input.tsv file should be a tidy-formatted feature matrix with columns "sample", "site", "feature", and "value". Each row represents a specific feature value for a given sample and site.

**tfx:** The tfx.tsv file should contain test sample names matching input.tsv and their corresponding tumor fractions. If an additional third column with true subtypes/categories is present, it enables additional validation during processing.

**reference:** If not using apre-generated .pickle, one or more ref.tsv files formatted similarly to the input file, containing matching reference feature values with the same four columns.

**key:** This key.tsv file must include sample names found in ref.tsv(s) and their corresponding subtypes/categories, with at least one subtype labeled as "Healthy".

### Optional arguments:

```
-d, --doi             : Disease/subtype of interest (positive case) for plotting and calculating ROCs. Must be present in key.
-x, --thresholds      : Tuple containing thresholds for calling the disease of interest (default: (0.5, 0.0311))  
-f, --features        : File with a list of site_feature combinations to restrict to. Sites and features should be separated by an underscore (path, optional)  
-s, --svm_selection   : Flag indicating whether to TURN OFF SVM feature selection method (default: True)  
-p, --palette         : .tsv file with matched categories/samples and HEX color codes. Subtype/category names must match those passed with the -t and -k options (path, optional)  
```

### Optional Inputs (extra details):

**features:** This file lists specific site_feature combinations to restrict the analysis to, with sites and features separated by an underscore. Example entries include `AR_central-depth`, `ASCL1_central-depth`, `ATAC-AD-Sig_fragment-mean`, and `ATAC-NE-Sig_fragment-mean`.

**palette:** A .tsv file that provides HEX color codes for categories/samples. The categories/subtype names in this file must match those in the key file.

### Contained Scripts:

**Keraon.py** | primary script containing both classification and mixture estimation methods  
**utils/keraon_utils.py** | contains utility functions called by Keraon.py for loading and processing data  
**utils/keraon_helpers.py** | contains helper functions called by Keraon.py for ctdpheno and keraon methods  
**utils/keraon_plotters.py** | combines helper functions for plotting outputs of Keraon.py  

### Methodology

---

### Pre‑processing and Robust Scaling

The raw feature matrix **X****raw**** ∈ ℝ\*\*\*\*n×d** undergoes a robust transformation per feature, across sites, implemented in `load_triton_fm()`.

| symbol | definition                                |
| ------ | ----------------------------------------- |
| μᴴ\_f  | median of *Healthy* anchors for feature *f* |
| IQR\_f | inter‑quartile range of feature *f*       |
| ε      | 10⁻¹², numerical floor                    |

The transformed value is

> x̃ₛ,f = ( xₛ,f − μᴴ\_f ) ⁄ ( IQR\_f + ε ).

Optional per‑feature point transforms (e.g. log₁₀) are applied *before* centering/scaling.  All parameters {μᴴ, IQR} are written to the reference `reference_simplex.pickle` and reused on test data.

---

#### Simplex Volume Maximization (SVM) for feature selection (beta, optional)

This process chooses a set of features which maximize the distances amongst healthy and tumor centroids in some N-dimensional space, by maximizing the volume of the simplex with vertices defined by those centroids.
As Keraon uses an orthonormalized version of the reference simplex defined in the same way to calculate tumor burdens, this method aims to improve those estimates when many features are available.
For any candidate mask of features α (a Boolean vector of length *d* features) the objective is

  Obj(α) = V × S × ρ,

where

- V – simplex volume of the class mean vectors in the masked space
- S – *scale factor* coupling edge length to within‑class scatter
- ρ – *regulatory term* enforcing shape regularity

| quantity                    | formula / description                                       |
| --------------------------- | ----------------------------------------------------------- |
| **V**                       | Cayley–Menger volume of the masked centroid                 |
| Edge set                    | All pairwise Euclidean distances between centroids          |
| Harmonic mean of edges *H*̅  | len(E) ⁄ Σ (1 ⁄ d) , with guard if any d < 10⁻⁹             |
| Regulatory term **ρ**       | min(E) ⁄ max(E) (range ∈ 0…1)                               |
| Scatter per class           | √Σ diag(Σᵢ[α])                                              |
| Mean scatter μₛ              | arithmetic mean over classes (∞ if any Σᵢ ill‑conditioned)  |
| Scale factor **S**          | H̅ ⁄ ( μₛ + 10⁻⁹ )^(3⁄2)                                      |

If any guard condition fails (volume≈0, non‑PSD, H̅≈0, μₛ→∞) the objective returns 0, preventing that mask from being chosen.

The MSV greedy loop iteratively flips the single feature bit that yields the largest positive ΔObj, stopping when ΔObj < 10⁻⁴.

### Greedy Maximization Algorithm

1. **Initial mask** – one feature per tumour subtype based on Mann–Whitney‑U seperation from other classes
2. **Iteration** – add the single unused feature that yields the greatest increase in Obj.
3. **Stopping** – stop when relative Obj gain < 10⁻⁴ or when a user‑defined cap is reached
4. **Output** – the final mask α∗ is written to disk and consumed unchanged by *ctdPheno* and *Keraon*

---

### Classification ("ctdPheno")

The `ctdpheno` function calculates TFX-shifted multivariate group identity relative log likelihoods (RLLs) for each sample in a dataset. The function uses a reference dataset containing subtype information and feature values to make predictions about the subtype and calculate RLLs for each sample.

#### Multivariate Normal Distribution

The log likelihood of the observed feature values given the TFX and the mean and covariance matrices of the subtypes is calculated using the multivariate normal probability density function (pdf). For a sample $\mathbf{x}$, the multivariate normal pdf is given by:

$\mathcal{L}(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)$

where:
- $\mu$ is the mean vector.
- $\Sigma$ is the covariance matrix.
- $k$ is the number of features.

The log of the likelihood (log likelihood) is then:

$\log \mathcal{L}(\mathbf{x} \mid \mu, \Sigma) = -\frac{1}{2} \left[ (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) + \log |\Sigma| + k \log (2\pi) \right]$

For a given sample $\mathbf{x}$ with tumor fraction TFX, the mean vectors of the class mixtures $\mu_{\text{mixture}}$ are calculated as:

$\mu_{\text{mixture}} = \text{TFX} \cdot \mu_{\text{subtype}} + (1 - \text{TFX}) \cdot \mu_{\text{healthy}}$

Covariance matrices are calculated similarly, shifting components based on the provided TFX.

#### Weights and Predictions

The function calculates the weights/scores for each subtype using the softmax function applied to the log likelihoods:

$w_i = \frac{e^{\log \mathcal{L}_i}}{\sum_{j} e^{\log \mathcal{L}_j}}$

where $w_i$ is the weight for subtype $i$.

Barring validation in an additional dataset using an identical reference set of anchors to determine an optimal scoring threshold, the prediction for each sample is the subtype with the highest weight.

---

### Mixture Estimation ("Keraon")

The `keraon` function transforms the feature space of a dataset into a new basis defined by the mean vectors of different subtypes across the selected features, creating a simplex meant to encompass the space connecting healthy, also from the reference, to the subtypes of interest. This transformation enables the direct, geometric calculation of the component fraction of each subtype in a sample's feature vector and thus the "burden" of each subtype.

#### Basis Vector Calculation

1. **Mean Vectors:** For each subtype $i$, the mean vector $\mu_i$ is calculated from the reference data:

$\mu_i = \frac{1}{n_i} \sum_{j=1}^{n_i} \mathbf{x}_j^{(i)}$

where $n_i$ is the number of samples in subtype $i$, and $\mathbf{x}_j^{(i)}$ is the $j$-th sample of subtype $i$.

2. **Directional Vectors:** The 'Healthy' subtype  vector $\mu_{\text{Healthy}}$ is subtracted from the mean vectors of the other subtypes to get directional vectors from healthy to each subtype:

$\mathbf{v}_i = \mu_i - \mu_{Healthy}$

3. **Orthogonal Basis Vectors:** The Gram-Schmidt process is applied to the directional vectors $\mathbf{v}_i$ to obtain an orthogonal basis, with healthy at the origin and each axis defining a direction along a subtype:

$$u_i = \frac{v_i - \sum_{j=1}^{i-1} \left( \frac{v_i \cdot u_j}{u_j \cdot u_j} \right) u_j}
{\left| v_i - \sum_{j=1}^{i-1} \left( \frac{v_i \cdot u_j}{u_j \cdot u_j} \right) u_j \right|}$$

The healthy vertex is then extended equally away from the tumor vertices by the maximum negative displacement amongst healthy reference samples, ensuring all healthy references are enclosed by the simplex. The Gram-Schmidt process is re-applied to produce orthonormality.

#### Sample Transformation

For each sample vector $\mathbf{x}$, the sample is trandformed into the new basis by subtracting $\mu_{\text{Healthy}}$ and projecting onto the orthogonal basis:

$\mathbf{y} = \mathbf{x} - \mu_{\text{Healthy}}$

$\mathbf{p} = \mathbf{y} \cdot \mathbf{U}^T$

where $\mathbf{U}$ is the matrix of orthogonal basis vectors.

#### Fraction and Burden Calculation

The projected length $|\mathbf{p}|$ of the vector that lies in the feature space defined by the simplex, and the orthogonal length $|\mathbf{d}|$ of the vector that completes the sample vector along an axis orthogonal to the feature space are then calculated:

$|\mathbf{p}| = \sqrt{\sum_{i=1}^{k} p_i^2}$

$|\mathbf{d}| = \sqrt{\sum_{i=1}^{k} d_i^2}$

where $\mathbf{d}$ is the difference vector between the original vector and its projection.

These components are then scaled by the provided tumor fraction to get the total fraction of each subtype, including off-target from the orthogonal component:

$w_i = \frac{e^{L_i}}{\sum_{j} e^{L_j}}$

$v_i = \mu_i - \mu_{Healthy}$

---

## Requirements

Keraon uses mostly standard library imports like NumPy and SciPy and has been tested on Python 3.9 and 3.10.
To create a tested environment using the provided `keraon_requirements.yml` file, follow these steps:

1. **Install Micromamba**: Ensure you have Micromamba installed. You can download and install it following the instructions on the [official website](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

2. **Download the `keraon_requirements.yml` File**: Make sure the file is in your current working directory.

3. **Create the Environment**: Use the following command to create a new Micromamba environment named `keraon` using the dependencies specified in the `keraon_requirements.yml` file:

    ```bash
    micromamba create -f keraon_requirements.yml
    ```

4. **Activate the Environment**: Once the environment is created, activate it using:

    ```bash
    micromamba activate keraon
    ```

5. **Verify the Installation**: To ensure all packages are installed correctly, you can list the packages in the environment using:

    ```bash
    micromamba list
    ```

## Contact
If you have any questions or feedback, please contact me here on GitHub or at:  
**Email:** <rpatton@fredhutch.org>

## Acknowledgments
Keraon is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.  

## Software License
Keraon
Copyright (C) 2022 Fred Hutchinson Cancer Center

You should have received a copy of The Clear BSD License along with this program.
If not, see <https://spdx.org/licenses/BSD-3-Clause-Clear.html>.

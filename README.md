# *Keraon*

## Description
Keraon, named for the Greek god of mixing wine, is a tool for estimating the contributing fraction of different cancer phenotypes in (possibly
heterogeneous) cell free tumor DNA from liquid biopsies. The method depends on reference samples of known, "pure" phenotypes (inlcuding healthy) which
define phenotype anchors by which additional samples of unknown phenotype are compared.

## Contact
Robert Patton

Fred Hutchinson Cancer Center

Contact: <rpatton@fredhutch.org>

Date: September 29, 2022

Website: https://github.com/denniepatton

## Requirements
### Software packages or libraries
  - Python 3.8.6
    - Numpy 1.22.3
    - Pandas 1.2.3
    - Scipy 1.8.0
    - Seaborn 0.11.1 (for plotting only)

### Scripts
  - Keraon.py (contains the keraon function and helper tools)
  - ExampleRun.py (example tool use with inlcuded data from DOI: https://doi.org/10.1101/2022.06.21.496879)

### Data
  - GriffinFeatureMatrix_WGS.tsv (*Griffin*-based features for UW cohort and DFCI cohort II - WGS)
  - GriffinFeatureMatrix_ULP.tsv (*Griffin*-based features for UW cohort and DFCI cohort II - ULP)
  - GriffinFeatureMatrix_DFCI1.tsv (*Griffin*-based features for DFCI cohort I - ULP)
  - GriffinFeatureMatrix_Triplets.tsv (*Griffin*-based features for 25x mixed phenotype admixtures)
  - GriffinFeatureMatrix_LuCaP.tsv (*Griffin*-based features for LuCaP [prostate cancer phenotype] anchors)
  - GriffinFeatureMatrix_HealthyDonor.tsv (*Griffin*-based features for LuCaP [prostate cancer phenotype] anchors)
  - DFCI1_TFX_Subtypes.tsv (*IchorCNA* tumor fraction estimates and known histologies for DFCI cohort I)
  - LuCaP_subtypes.tsv (LuCaP PDX phenotypes)
  - patient_subtypes.tsv (known patient histologies for UW cohort and DFCI cochort II)
  - ULP_TF_hg19.txt (*IchorCNA* tumor fraction estimates for UW cohort and DFCI cohort II - ULP)
  - WGS_TF_hg19.txt (*IchorCNA* tumor fraction estimates for UW cohort and DFCI cohort II - WGS)

## Run the analysis
### 1. Run ExampleRun.py to process example data
Make a local copy of the Keraon main folder and navigate to the scripts folder. After loading Python 3.8.6 and required modules
run ExampleRun.py directly to generate results locally. 

### 2. Using the tool as a stand-alone
Functions from Keraon.py may be imported directly into other scripts; see ExampleRun.py for examples on how to format data
and incorporate into a new script.

## Methodology
This model takes in two dataframes: ref_df contains a list of "pure" samples with known subtype 'Subtype'
including 'Healthy' (for basis anchoring) and *non-correlated* features of interest; df contains test samples of
unknown (possibly mixed) subtype or mixed subtype and tumor fraction 'TFX'. This model assumes passed features
are non-correlated, normally distributed within each subtype in the reference, and that feature values scale
linearly with tumor fraction. The number of subtypes is denoted K (including healthy) and the number of features
is denoted F below. The algorithm is most accurate when anchor distributions are tightly defined, and subtypes
are separated from healthy. Features may be evaluated and reduced using the helper functions:
  - diff_exp(), for finding differentially regulated features between anchors
  - norm_exp(), for testing normalcy within anchors
  - corr_exp(), for evaluating and removing inter-feature correlations
  - plot_feature_space(), for visualizing anchors when F = 2 (K = 3)
 
The algorithm works as follows:

### pre-processing
As a pre-processing step, the model first computes the mean vector ***μ_i*** and covariance matrix ***Σ_i*** for each anchor class _i_ in K, under the
assumption that each subtype (including healthy) fits a multivariate Gaussian distribution. Based on model constraints, K - 1 non-correlated features
fully specify the system, and so for example for ARPC:NEPC:Healthy (K = 3) fraction estimation we limited analyses to sets of two features of interest
(F = 2).

### basis-change
Next, for each sample defined by some location in feature space ***v*** and estimated tumor fraction _t_ a change of basis is performed to translate the
sample’s location from feature space to class space, where each (not necessarily orthogonal) axis defines a single phenotype, and the origin represents
pure healthy. If F = K -1, this is accomplished by solving the determined, linear matrix equation for the shifted basis components ***X***:

***BX***=***S***

Where ***B*** = [**μ_(i /= HD)**- **μ_HD**] is the matrix defining all basis vectors from the healthy mean anchor to each phenotype mean anchor,
and ***S*** is the vector from the healthy mean anchor to the sample of interest, ***S*** = ***v*** - ***μ_HD***. If the system is overdetermined
(F > K – 1), least squares is used to estimate the approximate solution. This step allows us to learn where in the class space the sample lies, which
determines how estimates are evaluated:

Anchor Space: if all basis components are positive then the sample lies within the volume of order K – 1 which has vertices defined by the class means.
The relative ratio of basis component magnitudes in the direction of each class are corrected by estimated tumor fraction directly

![image](https://user-images.githubusercontent.com/68241581/193168565-cb7cf21a-9542-4ce6-8f3a-290bbdfb062a.png)

Contra Space: if all basis components are negative then the sample lies within the volume of order K – 1 which forms a reflection of that formed by the
class vertices about healthy. Component fractions for each basis are computed to capture the inverse distance from the healthy anchor, such that

![image](https://user-images.githubusercontent.com/68241581/193168578-cad69cc1-3578-45a8-91e3-b87b95efe58f.png)


Extra Space: if some basis components are positive but others are negative, the sample lies in some space outside of the anchor or contra space. In this
case only positive contributions are considered, such that for all _i_ with ***X_i*** > 0

![image](https://user-images.githubusercontent.com/68241581/193168718-1b2d261c-fdde-4213-9cc5-31a1b1a927e9.png)


### output

The tumor fraction normalized basis component estimates have range [0,1], where values directly correspond to the total fraction of each class in the
sample. Basis-predictions output by Keraon include fractional estimates of each supplied subtype/anchor for each sample, as well as the class space.

## Software License
Keraon Copyright (c) 2022 Fred Hutchinson Cancer Research Center
All rights reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the BSD-3-Clause-Clear license. No licenses are granted to any
patent rights of the Fred Hutchinson Cancer Research Center.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the BSD-3-Clause-Clear license for more details.  

You should have received a copy of the G BSD-3-Clause-Clear license along with this program.
If not, see https://spdx.org/licenses/BSD-3-Clause-Clear.html. 


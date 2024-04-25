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

Under maintenance

### Uses

Keraon's primary use case is subtyping late-stage cancers and detecting potential trans-differentiation events. See published results for
classifying and estimating fractions of castration-resistent prostate cancer (CRPC) adenocarcinoma (ARPC) from neuroendocrine-like (NEPC) ([publications](#publications)).

### Publications

[Nucleosome Patterns in Circulating Tumor DNA Reveal Transcriptional Regulation of Advanced Prostate Cancer Phenotypes](https://doi.org/10.1158/2159-8290.CD-22-0692)

## Usage

Keraon can be run on the command line using the following arguments:

### Inputs to Triton.py:

```
-n, --sample_name		: sample identifier (string, required)  
-i, --input			: input .bam file (path, required)  
-b, --bias			: input-matched .GC_bias file (path, e.g. from Griffin†, required)  
Under Construction
```

### Inputs (extra details):

**input:** input .bam files are assumed to be pre-indexed with matching .bam.bai files in the same directory  

**bias:** sample-matched .GC_bias files can be generated using Griffin's GC correction method

### Contained Scripts:

**Keraon.py** | primary script containing both classification and mixture estimation methods
**keraon_helpers.py** | contains helper functions called by Keraon.py  
**keraon_plotters.py** | combines helper functions for plotting outputs of Keraon.py


#### keraon_helpers.py

Under construction

#### keraon_plotters.py

Under construction

### nc_info



### Methodology

#### Simplex Volume Maximization (feature selection)

Under construction

#### Classification ("ctdPheno")

Under construction

#### Mixture Estimation ("Keraon")

Under construction

## Requirements

On Fred Hutch servers the module Python/3.7.4-foss-2019b-fh1 may be used to run Keraon.
In general, Keraon uses standard libraries supported across many versions, i.e. numpy and scipy.
To see a list of requirements used and tested with Python 3.10 through a conda environment, see requirements.txt

## Contact
If you have any questions or feedback, please contact me at:  
**Email:** <rpatton@fredhutch.org>

## Acknowledgments
Triton is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.  

## Software License
Keraon
Copyright (C) 2022 Fred Hutchinson Cancer Center

You should have received a copy of The Clear BSD License along with this program.
If not, see <https://spdx.org/licenses/BSD-3-Clause-Clear.html>.

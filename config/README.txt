reference_key_example.txt:
On Fred Hutch servers these samples correspond to prostate cancer PDX model ctDNA, prostate cancer patient cfDNA, and "Healthy" cfDNA (N.B. the "Healthy" subtype must always be included in the reference samples key). This key has been restricted to samples with depth >= 5x (WGS) where at least 3 examples of a given phenotype are available (ARPC, NEPC, and Healthy).

tfx_example.txt:
Please note that if multiple subtypes are present in a sample, they should be separated by a comma (and no spaces).

palette_example.txt:
Note that the "Patient" color is used for plotting test points with unavailable subtype information (unknown).

site_features_example.txt:
For restricting to pre-selected site_feature combinations, with identical formatting to output results/FeatureAnalysis/site_features.txt. Remember features and sites containing "_" (underscore) have "_" converted to "-" (dash) in Keraon, so these must have the form some-site_some-feature, one per line.

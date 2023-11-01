This package contains Quix: a tool for estimating mixture proportions for
single data points, given some feature space. In basis_mode, Quix performs
a feature-to-subtype basis conversion and estimates mixtures from their
component values in the subtype space. Basis_mode requires a root subtype
(e.g. healthy) and will estimate the burden of additional subtypes, or, if
the root fraction is known, the total fraction of each subtype. Basis_mode is
a precursor to the Quix, or Quantum-Mixture (superposition) estimator. If root
fraction is known, the feature space is shifted according to the basis_mode
predictions before Quix is applied. Quix uses known pure-subtype feature values
to build a superposition of Gaussian-like wavefunctions, whose relative
proportions are modeled directly by Quix when maximizing the likelihood of a
given sample/data point belonging to it. See GitHub Docs for more info.

"""Helper package for BPSCS -- Penalized Synthetic Control under Spillovers.

Implements the utility-based shrinkage priors (distance-horseshoe ``dhs`` and
distance-spike-and-slab ``ds2``) of Fernandez-Morales, Oganisian & Lee (2026),
which shrink each donor's synthetic-control coefficient by a utility that blends
covariate similarity and spatial distance to the treated unit -- softly
down-weighting donors that are likely contaminated by spillovers.
"""

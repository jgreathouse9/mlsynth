# Vendored: Mani Bayani, Robust PCA Synthetic Control

This directory vendors, verbatim, the reference implementation from Mani
Bayani's dissertation work on Robust PCA Synthetic Control (RPCA-SC), used as the
cross-validation reference for mlsynth's `CLUSTERSC` RPCA-SC family:

- `FPCA.R` — functional-PCA + k-means selection of the donor cluster containing
  the treated unit (West Germany), over the pre-1990 GDP curves.
- `RPCA_2.py` — the RPCA-SC estimator: Robust PCA (Principal Component Pursuit,
  Candès, Li, Ma & Wright 2011) low-rank + sparse decomposition of the donor
  matrix, then a non-negative least-squares fit of the treated unit's pre-period
  path to the low-rank donor components, and the West-German-reunification
  application (the plotting / placebo / robustness code is included as shipped).
- `Data_Germany.csv` — the annual per-capita GDP panel (17 countries,
  1960–2003; the Abadie–Diamond–Hainmueller German reunification data).

Provenance and attribution
--------------------------

These files are the author's own; they are © Mani Bayani and are included here
with the author's permission solely to serve as the executable cross-validation
reference for `benchmarks/cases/clustersc_rpca_germany.py`. They are not part of
the mlsynth library, carry no separate open-source licence, and must not be
relicensed or redistributed independently of this attribution. Direct
reuse questions to the author.

Reference: Bayani, M. *Robust PCA Synthetic Control* (dissertation).

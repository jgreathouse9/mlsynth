DSC — Distributional Synthetic Controls on Dube (2019)
======================================================

.. currentmodule:: mlsynth

Path-A reproduction of the Distributional Synthetic Controls application
(Gunsilius 2023) on the Dube (2019) minimum-wage panel. The authors' reference is
the ``DiSCo`` R package
(`Davidvandijcke/DiSCos <https://github.com/Davidvandijcke/DiSCos>`_), whose
vignette analyses exactly this data.

DSC fits simplex-constrained weights on the **quantile functions** of micro-level
distributions: each ``(unit, time)`` cell is a sample, and the treated unit's
counterfactual quantile function is a weighted average of the donors'
(Agueh-Carlier barycenter / optimal transport).

Data
----

``basedata/dube_minwage.parquet`` -- the ``DiSCo`` package's ``dube`` dataset (Dube
2019; ``adj0contpov`` by state-year), exported from ``dube.rda`` and **subsampled
to 250 observations per state-year cell** (fixed seed) so the micro-panel is
~1 MB rather than 15 MB. 34 states (33 donors) x 7 years (1998-2004); Alaska
(``fips = 2``) treated from 2003, the vignette's ``id_col.target = 2``,
``t0 = 2003``.

Result
------

================================  ==========
Quantity                          DSC
================================  ==========
ATT (mean post QTE)               −0.15
Pre-period 2-Wasserstein fit      0.13
Placebo permutation p (2003)      0.91
Placebo permutation p (2004)      0.32
Donors                            33
================================  ==========

The headline cross-check against the vignette is the placebo-permutation result:
both post-year p-values **exceed 0.05** -- the vignette's stated "no spurious
effect" -- and the small pre-period Wasserstein confirms close distributional
tracking before treatment.

.. note::

   **No live DiSCo cross-validation here.** The ``DiSCo`` R package does not
   install on this environment's R version, and the vignette's weight/QTE numbers
   live in figures rather than text, so a value-for-value run isn't reproducible
   in CI. This case is therefore Path A on the authors' exact dataset and setup,
   with mlsynth's deterministic output pinned and anchored to the one quantitative
   claim the vignette states (``p > 0.05``). The subsampling (250 obs/cell) keeps
   the panel small; it shifts point values slightly from the full-data run but
   preserves the distributional structure and the inference conclusion.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py dsc_dube

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
2019; ``adj0contpov`` by state-year), converted from the package's
``data/dube.rda`` with every column verified bit-identical on round-trip. 652,870
rows, 34 states (33 donors) x 7 years (1998-2004), 2.0 MB as zstd parquet; Alaska
(``fips = 2``) treated from 2003, the vignette's ``id_col.target = 2``,
``t0 = 2003``.

This is the authors' complete analysis dataset, not a sample of it. An earlier
revision used a 250-observations-per-cell subsample retaining 9.1 percent of the
rows. For most estimators that would be an ordinary size-versus-fidelity trade;
for a distributional method it is not. The estimand is the within-cell
distribution, and true cell sizes run from 1,118 to 9,516 — an eight-fold spread
flattened to a constant. Restoring the full data cut the pre-period
2-Wasserstein fit from 0.129 to 0.038.

Result
------

================================  ==========
Quantity                          DSC
================================  ==========
ATT (mean post QTE)               −0.262
Pre-period 2-Wasserstein fit      0.038
Placebo permutation p (2003)      0.500
Placebo permutation p (2004)      0.118
Donors                            33
================================  ==========

These values moved when the full data replaced the subsample (previously −0.15,
0.13, 0.91, 0.32). The permutation p-values are multiples of :math:`1/34`, since
there are 33 donors plus the treated unit.

The headline cross-check against the vignette is the placebo-permutation result:
both post-year p-values **exceed 0.05** -- the vignette's stated "no spurious
effect" -- and the small pre-period Wasserstein confirms close distributional
tracking before treatment.

.. note::

   Two claims previously made here were wrong, and are corrected rather than
   quietly dropped. The ``DiSCo`` R package *does* install in this environment
   (``benchmarks/R/install_discos.sh``), and the vignette's numbers do not "live
   in figures rather than text" — the vignette is built with ``eval=FALSE`` and
   publishes no numbers at all. The conclusion was right, the reasons were not,
   and the wrong reasons made the situation look permanent instead of fixable.

   This case still pins mlsynth's own deterministic output, anchored to the one
   quantitative claim the vignette states (:math:`p > 0.05`). For genuinely
   external validation see :doc:`disco_tenure`, which reproduces the ``disco``
   Stata Journal article's published weights and quantile-effects table.

Reproduce
---------

.. code-block:: bash

   python benchmarks/run_benchmarks.py dsc_dube

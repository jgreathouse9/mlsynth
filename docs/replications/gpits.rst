GPITS — Cho (2026), the Heller decision
========================================

.. currentmodule:: mlsynth

Path A, the paper's own empirical result, plus a cell-by-cell cross-validation
against the author's R package. Both reproduce.

The target
----------

Cho (2026, Section 6) asks whether the Supreme Court's 2008 decision in
*District of Columbia v. Heller* changed legal handgun purchases. The ruling
bound every U.S. jurisdiction at once, which is what makes it a universal
treatment with no donor pool, but it struck down only D.C.'s handgun ban, so the
practical effect should land in one place and nowhere else.

The outcome is monthly FBI background checks per 100,000 population, 2002-07 to
2008-10, with the intervention at 2008-07. The window stops before the November
election so that post-election anxiety about future gun restrictions cannot be
confused with the ruling.

The reported figure is a cumulative four-month effect for D.C. of 15.1 checks per
100,000, with a 95% interval of [13.0, 17.3].

What reproduces
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Quantity
     - Paper
     - ``mlsynth``
   * - Cumulative four-month effect
     - 15.1
     - 15.1323
   * - 95% interval
     - [13.0, 17.3]
     - [12.9687, 17.2960]

Exact at the precision the paper reports. The monthly path has the shape the
paper describes: effects indistinguishable from zero in July and August 2008,
then 6.83 and 8.10 in September and October. All four placebo periods cover zero.

Cross-validation against ``gpss``
---------------------------------

The estimator ships as an R package, ``gpss`` (Rcpp/Armadillo), and the paper's
pipeline is at `soonhong-cho/gpits <https://github.com/soonhong-cho/gpits>`_
under MIT. Running ``gpss::gp_its`` live in R 4.3.3 against this implementation
on the same series:

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Quantity
     - Max relative difference
   * - Length-scale ``b``
     - 2.0e-12
   * - Noise variance ``s2``
     - 5.5e-11
   * - Counterfactual
     - 9.6e-11
   * - Per-period effect
     - 1.6e-12
   * - Cumulative effect
     - 4.8e-12
   * - Cumulative standard error
     - 1.8e-11
   * - Placebo estimates
     - 2.1e-11

Two independent implementations at floating-point agreement. What remains is the
bounded-Brent optimizer's convergence path, not the arithmetic: both the
length-scale rule and the noise-variance fit go through R's ``optimize()``, and
this implementation matches its tolerances so the comparison stays exact.

Four conventions of the reference are preserved deliberately, because reading the
paper alone would get them wrong and the published numbers would not reproduce:
the periodic and linear components run over every column of the design, the
one-hot month indicators included, not the time column alone; one-hot columns are
scaled by :math:`\sqrt{0.5}` and never centred; the seasonal period is rescaled
by the standard deviation of the first continuous column; and the marginal
likelihood uses ``sum(log(diag(L)))``, which is half the log-determinant. Each is
flagged at its site in :mod:`mlsynth.utils.gpits_helpers.kernels` and
:mod:`mlsynth.utils.gpits_helpers.pipeline`.

Two findings from the replication
---------------------------------

The intervals are conservative, not calibrated. Reproducing the paper's Section 5
simulation at 200 replications, GP coverage runs 0.986 to 1.000 across all
fifteen cells while a segmented regression runs 0.201 to 0.844. The qualitative
claim holds and is not an artifact of a weak baseline: a segmented regression's
interval reflects residual variance around its own fitted form and does not widen
off support, so it under-covers everywhere and badly at short pre-periods. The
quantity the paper's coverage panels do not show is width. The Gaussian process
sits at 1.000 with intervals a median of 2.3 times wider than the segmented
regression's, up to 4.9 times at the shortest pre-periods. That is the worst-case
bound working as Section 4 intends, and it means coverage is bought with power.

The nationwide null depends on the scale. Fitting an independent process to each
of the 50 jurisdictions, D.C. is rank 1 of 50 on the standardised scale the
paper's Figure 4A uses — 36.99 against 8.73 for the next highest, with the other
49 at a median of 0.91. On the raw per-100k scale D.C. ranks 25th of 50, and six
other jurisdictions are significant at 95%, two of them larger in magnitude than
D.C.'s 15.1. The separation comes from the standardisation: D.C.'s pre-treatment
standard deviation is 0.41 against a median of 19.8 across the others, because
the near-total ban held its series near zero. The paper states this and reports
the magnitude on the count and per-capita scales for that reason. Read Figure 4A
alongside the pre-period standard deviation, which
``GPITSResults.fit_diagnostics`` and the input series both expose.

Reproducing
-----------

.. code-block:: bash

   python benchmarks/cases/gpits.py

The panel is committed as ``basedata/dc_handgun_heller.csv``: the D.C. rows of
the FBI NICS monthly series divided by Census population and scaled per 100,000.
The pre-build spike, including the R cross-validation harness and the coverage
simulation, is at ``benchmarks/reference/gpits_heller/``.

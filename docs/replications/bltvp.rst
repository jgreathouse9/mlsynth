BLTVP: Klinenberg (2023) on Proposition 99
==========================================

Path A. :class:`mlsynth.BLTVP` is validated against the empirical result the
paper reports on the authors' own data.

The paper
---------

Klinenberg, D. (2023). "Synthetic Control with Time Varying Coefficients: A
State Space Approach with Bayesian Shrinkage." *Journal of Business & Economic
Statistics* 41(4):1065-1076.
`doi:10.1080/07350015.2022.2102025 <https://doi.org/10.1080/07350015.2022.2102025>`__

Table 2 (p. 1075) reports the average reduction in California's per-capita
cigarette sales after Proposition 99 for seven methods. The BL-TVP row is the
target here.

What was matched
----------------

The panel is Abadie, Diamond and Hainmueller (2010): 39 states over 1970-2000,
California treated in 1989, 38 donors, :math:`T_0 = 19`. It ships with mlsynth
as ``basedata/smoking_data.csv`` and is ingested through
:func:`mlsynth.utils.datautils.dataprep`, which reproduces the panel the
author's script builds by pivoting ``mixtape::smoking`` by hand.

Over four chains of 10,000 draws with 5,000 burn-in --- the settings the
author's ``3-BLTVP.R`` uses --- every quantity in the row agrees within Monte
Carlo error:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Paper
     - mlsynth
     - MCSE
     - z
   * - Average decrease
     - 17.7
     - 17.78
     - 0.11
     - +0.72
   * - 2.5th percentile
     - -16.1
     - -16.65
     - 0.51
     - -1.09
   * - 97.5th percentile
     - 51.7
     - 52.28
     - 0.48
     - +1.22

The substantive conclusion carries across too: the credible interval covers
zero, so the estimated effect is not statistically distinguishable from none.
That is the paper's own reading (sec. 6), consistent with Arkhangelsky et al.
(2021) and at odds with the original Abadie, Diamond and Hainmueller estimate.

Two specification questions the paper alone does not settle
-----------------------------------------------------------

Both were decided by running the alternatives and comparing against the
published number, not by reading the text.

The averaging window. The author's ``ca_table.R`` averages over ``[t_0:T0]``
with ``t_0 = 19``, and period 19 is the last pre-treatment year: treatment
lands in 1989, so the post window runs from period 20 to 31. The published 17.7
therefore averages one pre-treatment year together with the twelve
post-treatment years. Averaging the post window alone gives 19.25, which is
:math:`z = +13.3` against 17.7, so the window is identifiable from the number
and the paper's is the one that includes period 19. The benchmark case
reproduces that window.

The intercept. Model 1 carries an intercept as the :math:`(J+1)`-th regressor,
but ``bitto_model_revised.R`` passes ``y ~ +-1+.``, which suppresses it, and
``3-BLTVP.R`` adds no column of ones. Fitting both ways, the point estimate is
insensitive (17.78 without, 17.81 with) while the interval bounds are not, and
the intercept-free fit that the script actually ran is the closer match. The
``intercept`` config field defaults to ``False`` for that reason and can be set
``True`` to fit Model 1 as printed.

A caveat on the pre-treatment fit
----------------------------------

The pre-treatment RMSE is 0.040 against a California series averaging 116 packs
per capita. That is interpolation: 38 donors carry 76 free coefficients against
19 pre-treatment observations, so a near-zero pre-period residual is what an
overparameterised model produces and is not on its own evidence for the
dynamics. The evidence for those is Table 1, where BL-TVP is the only one of
seven methods to reach nominal coverage while holding the lowest mean squared
forecast error. That table is a separate validation and is not yet pinned.

Reproducing it
--------------

.. code-block:: bash

   python benchmarks/run_benchmarks.py --case bltvp_prop99

The case is `benchmarks/cases/bltvp_prop99.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/bltvp_prop99.py>`__.
It fits :class:`mlsynth.BLTVP` at the author's settings and pins all three
Table 2 quantities, plus the claim that the band covers zero. A readable
standalone port of the paper's sampler, used to demonstrate the replication
before the estimator existed, is kept alongside at
``benchmarks/reference/bltvp_prop99/bltvp_reference.py``.

Tolerances carry one chain's Monte Carlo error --- the measured across-chain
standard deviations are 0.22 for the average and about 1.0 for each bound ---
and are set at a little over three standard deviations, which leaves the case
able to fail on the question it exists to settle: the post-window variant
misses the average by 1.55.

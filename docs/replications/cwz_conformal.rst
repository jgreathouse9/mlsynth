.. _replication-cwz-conformal:

Conformal inference for synthetic controls (Chernozhukov, Wüthrich & Zhu 2021)
==============================================================================

:Estimator: :doc:`../vanillasc` — :class:`mlsynth.VanillaSC`, ``inference="conformal"``
:Source: Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021), *"An Exact and
   Robust Conformal Inference Method for Counterfactual and Synthetic
   Controls,"* Journal of the American Statistical Association 116(536),
   1849–1864.
:Replication type: cross-validation — the authors' own R, run live on the
   panel their paper uses as its empirical application.
:Status: verified — every deterministic quantity reproduced exactly.
:Durable case: `benchmarks/cases/cwz_conformal.py
   <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cwz_conformal.py>`__

What the method does
--------------------

Synthetic control gives a counterfactual but not, on its own, a way to say
whether the gap it opens after treatment is larger than the gaps it was already
making before. Conformal inference answers that by asking a question with a
finite-sample answer: if the treatment had no effect, would the post-treatment
residual look out of place among the pre-treatment ones?

The procedure is a test of a sharp null. To test :math:`H_0: \tau = \tau_0`,
subtract :math:`\tau_0` from the treated unit's post-treatment outcomes, refit
the synthetic control on the adjusted series across every period, and compute a
statistic on the post-treatment block of residuals — here
:math:`S = \sum_{t > T_0} |\hat u_t|`. Under the null the residual path is
exchangeable, so the observed :math:`S` should be an ordinary draw from the
statistics obtained by permuting that path. The p-value is the fraction of
permutations whose statistic is at least as large.

Two permutation schemes, and the choice is an assumption about the errors. The
i.i.d. scheme draws random permutations of the whole path, which is exact when
the errors are exchangeable. The moving-block scheme uses the :math:`T` cyclic
shifts of the path, which preserves serial dependence and is the authors'
default; because one of the :math:`T` shifts is the observed path itself, its
p-value cannot fall below :math:`1/T`.

Inverting the test gives a confidence interval. Sweep a grid of candidate
effects, keep the ones the test does not reject at level :math:`\alpha`, and
report the range. Nothing here needs a large sample, an estimated long-run
variance, or a limiting distribution, which is what "exact and robust" refers
to.

The application
---------------

Section 5 studies Rhode Island, whose courts inadvertently decriminalized indoor
prostitution in 2003, on log female gonorrhea incidence per 100,000 (Cunningham
& Shah 2018). The panel is 25 years, 1985–2009, with Rhode Island against 50
control states: :math:`T_0 = 19` and :math:`T_1 = 6`. It ships in the authors'
replication package and is vendored at ``basedata/logfemrate.txt``.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import VanillaSC

   Y = pd.read_csv("basedata/logfemrate.txt", sep="\t").to_numpy(float)
   units = ["Rhode Island"] + [f"donor{j:02d}" for j in range(1, Y.shape[1])]
   df = pd.DataFrame({
       "state": np.repeat(units, len(Y)),
       "year": np.tile(np.arange(1985, 1985 + len(Y)), Y.shape[1]),
       "logfemrate": Y.T.ravel(),
   })
   df["treated"] = ((df.state == "Rhode Island") & (df.year >= 2004)).astype(int)

   res = VanillaSC({
       "df": df, "outcome": "logfemrate", "treat": "treated",
       "unitid": "state", "time": "year", "backend": "outcome-only",
       "inference": "conformal", "conformal_type": "block", "alpha": 0.1,
       "conformal_grid": np.round(np.arange(-5.0, 2.0 + 1e-9, 0.01), 10),
       "display_graphs": False,
   }).fit()

   res.inference.details["joint_p_value"]   # 0.04
   res.inference.details["pi_lower"]        # -0.26 -0.86 -0.81 -1.16 -1.46 -1.31
   res.inference.details["pi_upper"]        #  0.70 -0.01  0.17 -0.33 -0.33 -0.19

The reference
-------------

Two implementations of this method exist and both are the authors': the
``scinference`` package, and the ``functions_conformal_final.R`` that produced
the published tables. The reference run uses the package at ``v1.0.0``
(``567c688``) — the version their later replication package pins — and
re-derives the supplement's three functions beside it, so the packaged and
published forms are checked against each other on the same panel. They agree to
the last digit on every deterministic quantity, which is recorded in the bundle
under the ``_supplement`` keys.

``scinference`` solves the synthetic control through ``limSolve::lsei``, whose
``type = 1`` reports "inequalities contradictory" on this panel and returns
weights off the simplex — 50 donors against 25 periods is the shape that breaks
it, so ``type = 2`` here is the only feasible solve and not a preference. The
supplement uses it, and so does the reference run. The carbon
tax panel behind :doc:`cwz_ttest <../vanillasc>` is the other shape, 14 donors
against 46 periods, and uses ``type = 1`` there.

What was reproduced
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - mlsynth
     - ``scinference``
     - Agreement
   * - :math:`p`-value, moving block
     - 0.040000
     - 0.040000
     - exact
   * - 90% interval, 2004
     - [−0.26, 0.70]
     - [−0.26, 0.70]
     - exact
   * - 90% interval, 2005
     - [−0.86, −0.01]
     - [−0.86, −0.01]
     - exact
   * - 90% interval, 2006
     - [−0.81, 0.17]
     - [−0.81, 0.17]
     - exact
   * - 90% interval, 2007
     - [−1.16, −0.33]
     - [−1.16, −0.33]
     - exact
   * - 90% interval, 2008
     - [−1.46, −0.33]
     - [−1.46, −0.33]
     - exact
   * - 90% interval, 2009
     - [−1.31, −0.19]
     - [−1.31, −0.19]
     - exact
   * - placebo test, :math:`T_1 = 1`
     - 0.315789
     - 0.315789
     - exact
   * - placebo test, :math:`T_1 = 2`
     - 0.315789
     - 0.315789
     - exact
   * - placebo test, :math:`T_1 = 3`
     - 0.263158
     - 0.263158
     - exact
   * - :math:`p`-value, i.i.d., 5000 draws
     - 0.023000
     - 0.022795
     - 2.1e−4 (Monte Carlo)

The intervals are swept on the application's own grid, ``seq(-5, 2, 0.01)``,
passed through ``conformal_grid``. Sharing the grid is what makes the comparison
value-for-value: an inversion returns grid points, so two implementations left
to choose their own grids would be compared at their resolutions and not at
their answers.

The placebo rows are the paper's specification tests. They hold out the last
one, two and three pre-treatment periods and test them as if they were the
post-period, on a window where no effect exists. Their p-values sit far from
the level, which is the check that the procedure is calibrated on this panel
before its answer on the real post-period is read.

The i.i.d. row is the only stochastic quantity, and the gap is the distance
between two independent 5000-draw estimates of the same number. One such
estimate has a standard error of :math:`\sqrt{p(1-p)/5000} = 0.0021`, so a
difference of 2.1e−4 is well inside sampling error.

What the reproduction found
---------------------------

The band did not match at first: all six intervals came out 15 to 40 percent too
wide while the p-value matched exactly, which localised the disagreement to the
inversion. mlsynth kept candidates with :math:`p \geq \alpha`; a level-:math:`\alpha`
test rejects at :math:`p \leq \alpha`, so the band was keeping nulls its own
p-values rejected.

Why that had gone unnoticed is a property of the panel it was checked on. A
conformal p-value from a single-period inversion has :math:`T_0 + 1` members in
its reference set, so it takes only the values :math:`k/(T_0+1)`, and the strict
and inclusive readings differ only where :math:`\alpha` is one of them. On the
carbon tax panel :math:`T_0 + 1 = 31` and :math:`\alpha = 0.1` falls between
:math:`3/31` and :math:`4/31`, so the two readings coincide exactly. On the
authors' own application :math:`T_0 + 1 = 20` and :math:`0.1` is attained. The
rule now lives in
:func:`mlsynth.utils.conformal.inversion.confidence_set_bounds`.

Reproducing it
--------------

.. code-block:: bash

   bash benchmarks/R/install_scinference.sh
   Rscript benchmarks/reference/cwz_conformal/reference.R
   python benchmarks/reference/generate.py cwz_conformal
   python benchmarks/run_benchmarks.py --case cwz_conformal

The captured bundle in ``benchmarks/reference/cwz_conformal/`` holds the
reference script, its verbatim output, the parsed values the case pins against,
and the provenance — R version, package versions, and a checksum of the panel.
The case reads its expected values from that bundle, so the constant and the
captured run cannot drift apart.

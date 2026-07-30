jackknife+ for ridge ASCM -- against augsynth on the Kansas panel
=================================================================

:doc:`../vanillasc`'s ``inference="jackknife_plus"`` is cross-validated against
the authors' own R package, `ebenmichael/augsynth
<https://github.com/ebenmichael/augsynth>`_, pinned at commit ``7a90ea48``
(augsynth 0.2.0). The reference is not vendored; install it with
``bash benchmarks/R/install_augsynth.sh`` and regenerate the comparison with
``Rscript benchmarks/reference/ascm_jackknife_plus/reference.R``. The captured
gold is committed, so the benchmark and the tests run without an R toolchain.

The setup is augsynth's own Kansas tax-cut example: quarterly log GDP per
capita, 50 states, treatment at 2012 Q2, 89 pre-treatment periods and 16 post,
ridge-augmented synthetic control with no auxiliary covariates.

What the procedure is
---------------------

Despite the name, this is not the jackknife-plus of Barber, Candès, Ramdas and
Tibshirani, which resamples over observations, and it is not the delete-one-unit
jackknife that :doc:`ppscm` uses. It resamples over pre-treatment periods.

Each pre-period is held out in turn, the augmented control is refitted on the
remaining pre-periods, and the refit is asked to predict the period it never saw.
A period the fit cannot recover when hidden is evidence that the post-treatment
prediction is also uncertain, so the interval is built from those held-out
errors. With 89 pre-periods that is 89 refits for one interval.

What matches, seam by seam
--------------------------

Because :doc:`ascm_kansas` already pins the ridge ASCM point estimate against a
live augsynth run on this same panel, the fit is not a confound here and
everything below isolates the inference.

.. list-table::
   :header-rows: 1

   * - quantity
     - agreement
   * - held-out error, per dropped pre-period
     - 89/89, worst 3.5e-8
   * - refit counterfactual path, per dropped pre-period
     - 89 x 17, worst 1.7e-7
   * - assembled bounds, default branch
     - worst 3.8e-9
   * - assembled bounds, conservative branch
     - worst 1.8e-7
   * - bound on the post-period average
     - 6.4e-10 (lower), 3.2e-9 (upper)

The per-drop rows are pinned as well as the assembled ones, deliberately. The
interval alone would report that a regression happened; the per-drop seam
reports which refit caused it.

Why the two branches earn different tolerances
----------------------------------------------

The base simplex fit underneath every refit is a constrained quadratic program,
solved here by mlsynth's own exact solver and in the reference by ``quadprog``
and osqp. Two exact solvers agree to their own tolerances rather than
bit-for-bit, so roughly 1e-7 is the floor for any quantity downstream of the QP.

The default branch takes quantiles across all 89 drops, which averages that
difference away and reaches 4e-9. The conservative branch instead takes the
minimum and maximum over drops, which deliberately selects the single most
extreme refit -- exactly where the solver difference is largest -- and lands at
1.8e-7. Looser is correct there rather than lax: an interval built from order
statistics cannot be more reproducible than its worst-case term.

Two findings worth keeping
--------------------------

The penalty is pinned across refits, and this is not obvious from the paper.
``augsynth.R`` stores the fitted ridge penalty into ``extra_args`` before any
resampling (``extra_args$lambda <- augsynth$lambda``), so every held-out refit
reuses the original fit's lambda and none re-runs cross-validation. Re-selecting
per drop would give 89 different penalties and a materially different interval,
and would multiply the cost by a CV sweep per refit. Read from the source, not
inferred from the output.

``conservative = TRUE`` is not reliably conservative. It widens by the
:math:`1-\alpha` quantile of the absolute held-out errors, where the default
branch reaches the :math:`1-\alpha/2` quantile of
:math:`\widehat{y}^{N}_{1t}(j) + |e_j|`. When the held-out errors are
right-skewed -- on this panel the largest absolute error is 0.037 against a 95th
percentile of 0.013 -- the default's deeper reach into that tail dominates the
wider min/max envelope, and the "conservative" interval comes out *tighter*. It
is narrower at 16 of the 17 reported periods here. mlsynth reproduces this rather
than imposing an ordering the reference does not have, and the benchmark pins the
branch's construction instead of a width comparison so that nobody later
"corrects" it.

Verification
------------

Pinned in `benchmarks/cases/ascm_jackknife_plus.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ascm_jackknife_plus.py>`_
(10 rows) and in ``mlsynth/tests/test_ascm_jackknife_plus.py`` (31 tests), which
asserts the per-drop seam before the assembled bounds so a failure localises.

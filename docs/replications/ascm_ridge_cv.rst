Ridge lambda cross-validation -- against augsynth on two panels
===============================================================

The ridge penalty in :doc:`../vanillasc`'s Augmented SCM (``augment="ridge"``) is
selected by leave-one-pre-period-out cross-validation and a 1-SE rule. This page
records that selection being cross-validated against the authors' own R package,
`ebenmichael/augsynth <https://github.com/ebenmichael/augsynth>`_ at commit
``7a90ea48`` (0.2.0), and two defects it uncovered.

A wrong penalty does not announce itself. It is not an error or an implausible
number -- it is a well-behaved fit at the wrong amount of shrinkage, which then
propagates into every effect and every interval built on top of it.

Two panels, one of which cannot see the problem
-----------------------------------------------

.. list-table::
   :header-rows: 1

   * - panel
     - shape
     - role
   * - augsynth's ``kansas``
     - 89 pre-periods, 49 donors
     - negative control -- the selected penalty is identical whether the code is
       right or wrong
   * - Song et al. PM2.5, "2+26 cities" 2015
     - 25 pre-periods, 37 donors
     - diagnostic -- the final pre-period is a ~9x outlier fold, so the defects
       move the answer

The Kansas panel agreed on the selected penalty to fifteen significant figures
both before and after the fixes below. Any suite testing only that panel would
report the cross-validation as correct while it was not, which is exactly what
happened. The second panel was added because its final pre-treatment period sits
on the seasonal ramp into the heating season and carries about nine times the
held-out error of the other twenty-four, so whether it counts as a fold at all
moves the whole curve.

What agreed already
-------------------

The lambda grid and the selection rule were faithful and are now pinned.
``create_lambda_list`` builds ``lambda_max * scaler^k`` for ``k`` in
``0..n_lambda``, which is :math:`n_{\lambda}+1` points rather than
:math:`n_{\lambda}` -- an easy off-by-one to inherit from R's
``seq(0:n_lambda) - 1``. ``lambda_max`` is the squared largest singular value of
the centered control matrix, and matches to 1e-13 on the outcome-only
specification. ``choose_lambda`` takes the largest lambda whose mean error is
within one standard error of the minimum, with the comparison inclusive.

Two defects, and why one hid the other
--------------------------------------

The fold count was one too many. augsynth's ``get_lambda_errors`` loops
``i in 1:(ncol(X_c) - holdout_length)`` and sizes its error matrix to match, so
with a holdout length of one the final pre-period is never held out. mlsynth
iterated one extra fold. On Kansas that is a 1-in-89 perturbation of an average
and changed nothing detectable. On the Song cell the extra fold contributes about
nine times the mean error, which moved the curve by up to 22 and the selected
penalty from augsynth's 28.5157 to 11.3523 -- a factor of 2.5 in shrinkage.

The standard error used the wrong denominator. augsynth reports
``sd(x) / sqrt(length(x))`` and R's ``sd`` is the sample standard deviation,
while numpy's ``.std`` defaults to the population one. The reported error was
therefore too small by :math:`\sqrt{(n-1)/n}`. This is not cosmetic: it feeds the
1-SE threshold directly, so it can change which penalty is chosen.

After both fixes, on the outcome-only specification:

.. list-table::
   :header-rows: 1

   * - quantity
     - kansas
     - song
   * - lambda grid
     - 1e-13
     - 1e-13
   * - CV error curve
     - 1e-10 (was 5.1e-6)
     - 1e-6 (was 22.2)
   * - CV standard error
     - 1e-10
     - 2.5e-7
   * - selected penalty
     - exact
     - exact (was 11.35 vs 28.52)

What the fix exposed
--------------------

Fixing the cross-validation made the covariate cells of :doc:`ascm_kansas`
disagree *more*, from about 0.002 to 0.005 on the ATT. That is a cancellation
being removed rather than a regression, and it is worth stating plainly because
the surface reading is the opposite.

mlsynth's ``lambda_max`` for the covariate specification is 128.4583 where
augsynth's is 128.6077 -- a pre-existing difference of about 0.1 percent in the
standardized covariate block, which the cross-validation fix does not touch. The
old, wrong fold count happened to select a point on that already-wrong grid that
landed nearer augsynth's answer. With the fold count corrected the disagreement
is attributed to its actual cause instead of being masked by a second one. The
outcome-only specification, where no covariate block exists, is exact.

That remaining covariate difference is tracked separately; the plausible cause is
the covariate matrix construction rather than the standardization, since
``ascm_kansas`` already documents a difference in how rows with a missing revenue
value are dropped before averaging.

Verification
------------

Pinned in ``mlsynth/tests/test_ascm_ridge_cv.py`` (16 tests), which asserts the
grid, the curve, the standard error and the selected penalty on both panels, and
separately pins the fold count and the fact that the final pre-period is never
held out so the off-by-one cannot silently return. Gold is generated by
`benchmarks/reference/ascm_ridge_cv/reference.R
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/reference/ascm_ridge_cv/reference.R>`_
and committed, so the tests need no R toolchain.

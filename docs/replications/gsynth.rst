.. _replication-gsynth:

Generalized synthetic control — Election Day Registration (Xu 2017)
===================================================================

:Estimator: :doc:`../gsynth` — generalized synthetic control at two
   unobserved factors, with additive state and year effects.
:Source: Xu, Y. (2017), *"Generalized Synthetic Control Method: Causal
   Inference with Interactive Fixed Effects Models,"* Political Analysis
   25(1):57-76, `10.1017/pan.2016.2 <https://doi.org/10.1017/pan.2016.2>`_,
   Table 2 columns (3) and (4).
:Replication type: Path A — the paper's published estimates on the author's
   own data — and cross-validation against a live ``fect`` 2.4.5 run of the
   same specification.
:Status: Verified against both. Both ATTs reproduce to within half the
   paper's last printed digit, and against the live reference the agreement
   is machine precision across every rank from zero to five on both
   specifications.

What the paper estimates
------------------------

Voting in the United States takes two trips: one to register, one to vote.
Election Day Registration lets an eligible voter do both at once, and the
question is whether that raises turnout. Nine states had adopted it by 2012 —
Maine, Minnesota and Wisconsin in 1976, Idaho, New Hampshire and Wyoming in
the 1990s, Iowa and Montana in 2008, Connecticut in 2012 — and thirty-eight
had not.

A two-way fixed effects regression on this panel returns 0.87 and 0.78
percentage points with standard errors of about 3, so it cannot distinguish
the effect from zero. That specification assumes the adopting and
non-adopting states share a common year effect. The paper's argument is that
they do not: states differ in how strongly they respond to national forces,
and conditioning on two unobserved factors alongside the additive effects
changes the answer to about 5 percentage points with a standard error of 2.3.

Why this estimator
------------------

Columns (3) and (4) are the generalized synthetic control method itself, so
this replication is the estimator's own reference point. :doc:`../gsynth` is a
port of it: the interactive fixed effects fit on the thirty-eight
never-adopting states, each adopter's loadings from its own pre-adoption
elections, and the difference over what followed.

Two reference bases
-------------------

This page carries two comparisons. Comparing mlsynth to the printed table
measures two things at once — whether mlsynth implements this specification,
and whether the printed table was produced by it — and one number cannot
separate them. So the case also runs the same specification through
``fect`` 2.4.5, whose ``method = "gsynth"`` path is what the ``gsynth``
package itself now dispatches to, and compares against that.

The separation is clean here. The live reference lands on 5.1305 and 4.8958,
which round to the printed 5.13 and 4.90, so the table came from this
estimator at this specification. That makes the live comparison the binding
one, and it is tight enough to be uninformative about anything except a
regression.

Results
-------

Table 2's two GSC columns, against the live reference and against what
mlsynth returns:

.. list-table::
   :header-rows: 1
   :widths: 30 12 20 20 12

   * - Quantity
     - Paper
     - Live ``fect`` 2.4.5
     - mlsynth
     - Paper SE
   * - ATT, no covariates
     - 5.13
     - 5.1304927
     - 5.1304927
     - 2.27
   * - ATT, with covariates
     - 4.90
     - 4.8957798
     - 4.8957798
     - 2.27
   * - Universal mail-in registration
     - 0.15
     - 0.15467938
     - 0.15467938
     - 0.80
   * - Motor voter registration
     - −1.05
     - −1.05149676
     - −1.05149676
     - 0.79
   * - Unobserved factors
     - 2
     - 2
     - 2
     -

The paper prints two decimals, so a published figure pins the true one only
to within 0.005. Both ATTs sit inside that, as do both covariate
coefficients. This is the tightest agreement the published precision can
express.

Against the live reference the comparison is far sharper, and it runs over
the whole rank grid instead of the two published cells. Across
:math:`r = 0, \dots, 5` on both specifications — twelve fits — the ATTs agree
to 7.7e-14, the covariate coefficients to 4.8e-9, which is the alternating
least squares' own stopping tolerance, and the full 33-point effect path by
horizon to 2.3e-13. The two implementations compute the same intermediates in
the same order, so they agree to floating point.

Standard errors
---------------

Column (3)'s error is a parametric bootstrap of 2,000 draws blocked at the
state level. Reproducing 2.27 to the digit would mean reproducing R's random
number stream, not the estimator, so the question the case can answer is
whether the inference lands in the same place. mlsynth returns 2.34 at 200
draws and 2.18 at the paper's 2,000, against the printed 2.27 — 3 and 4
percent out; the live reference returns 2.30. A broken inference routine would
miss by far more, which is what the 15 percent tolerance is set to catch.

Rank selection is the whole result
----------------------------------

The replication turned up one thing the printed table does not show, and it
matters for anyone trying to reproduce this paper today.

The ATT is not monotone in the number of factors. Over the plausible range it
runs 1.26, 3.56, 5.13, 2.36, 3.72, 4.48 at :math:`r = 0` through :math:`5`,
and the cross-validation criterion barely separates the middle of that range.
So the rank rule decides the headline number.

Xu's Algorithm 1 holds back one pre-treatment period from the treated units
and scores a rank by how well it predicts the held-out cell. It is
deterministic, it selects two factors on both specifications, and it
reproduces 5.13 and 4.90. Its criterion, which mlsynth reproduces to four
decimals against the reference, is a clean interior minimum:

.. list-table::
   :header-rows: 1
   :widths: 16 14 14 14 14 14 14

   * - :math:`r`
     - 0
     - 1
     - 2
     - 3
     - 4
     - 5
   * - MSPE, no covariates
     - 20.68
     - 11.95
     - 10.33
     - 11.41
     - 16.24
     - 16.09
   * - MSPE, with covariates
     - 22.14
     - 12.04
     - 10.31
     - 11.48
     - 16.29
     - 15.79

The ``gsynth`` package is now a shell over ``fect``, and ``fect``'s default
cross-validation changed in version 2.3.0 to a rolling scheme that masks a
random tenth of the control units over twenty folds. On this panel that
choice is seed-dependent:

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13 13

   * - seed
     - 02139
     - 1
     - 42
     - 123
     - 2024
     - 7
   * - rank selected
     - 4
     - 1
     - 2
     - 2
     - 1
     - 1
   * - ATT, no covariates
     - 3.72
     - 3.56
     - 5.13
     - 5.13
     - 3.56
     - 3.56

``cv.method = "all_units"`` picks one factor on every seed and returns 3.56.
``cv.method = "loo"`` is Algorithm 1, picks two on every seed, and returns
5.13. So the paper reproduces exactly, and it reproduces only under the
paper's own rank rule.

mlsynth implements Algorithm 1 and nothing else, and the benchmark pins the
selected rank at 2 alongside the ATTs, so a change to the rule fails the case
instead of moving the estimate.

Reading the data
----------------

``basedata/xu_edr_turnout.parquet`` is ``turnout.rda`` from
`xuyiqing/fect <https://github.com/xuyiqing/fect>`_ written to Parquet
unchanged: 47 states by 24 quadrennial presidential elections, 1,128 rows,
which is the Observations row of Table 2. The treatment is absorbing and
staggered over four adoption dates, and no state repeals, so each adopter
reaches Step 2 with one contiguous pre-adoption block.

The covariates are indicators for universal mail-in registration and motor
voter registration. Both are time-varying at the state level, which is what
GSC's covariate slot wants — a time-invariant column would be collinear with
the unit effects, and the estimator drops such a column and reports it in
``design.dropped_covariates`` instead of failing on a singular matrix.

Running the reference
---------------------

The committed gold under ``benchmarks/reference/gsynth_xu_turnout/`` is what
the case reads by default, so it runs in CI without R. To re-run the
reference live::

   bash benchmarks/R/install_fect.sh
   MLSYNTH_BENCH_LIVE_R=1 python benchmarks/run_benchmarks.py \
       --case gsynth_xu_turnout

That regenerates the gold into a temporary directory and checks it against the
committed copy. A regenerated run currently reproduces it at 0.0, and the case
raises if that moves by more than 1e-8, because a row that exists only when R
does could not be pinned and an unpinned row is one nobody checks.

Case
----

``benchmarks/cases/gsynth_xu_turnout.py``. Every row is a distance from a
reference, so a regression moves it and cannot be absorbed by re-fitting.

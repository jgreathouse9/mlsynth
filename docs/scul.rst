Synthetic Control Using Lasso (SCUL)
====================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method of Abadie and co-authors [ABADIE2010]_ builds
a counterfactual for one treated unit as a weighted average of donor units that
reproduces the treated unit's pre-treatment outcome path. The textbook version
restricts the weights to be non-negative and to sum to one, so the synthetic
control is a convex combination that stays inside the range of the donors. That
restriction is comfortable when the donor pool is small and the treated unit
sits inside the donors' convex hull, but it breaks down in two common modern
situations: the donor pool is high-dimensional -- more candidate donors than
pre-treatment periods -- and the donors come from many variable types, not just
the same outcome measured elsewhere.

SCUL [HollingsworthWing2022]_ handles both. It builds the synthetic control as a
lasso regression of the treated unit's pre-treatment outcome on the donor pool.
The lasso's :math:`\ell_1` penalty selects a sparse set of donors automatically,
so the pool can be larger than the pre-period and can mix variable types (in the
California example below, every donor state's cigarette sales and its retail
cigarette price are candidates). The weights are unrestricted -- they may be
negative, and an intercept is included -- so the synthetic control is allowed to
extrapolate beyond the donors' convex hull when that gives a better fit. The
penalty is chosen by a rolling-origin cross-validation that respects the time
ordering, guarding against overfitting the pre-period noise.

Reach for SCUL when the donor pool is large or multi-type, when a convex
synthetic control cannot fit the treated unit because it lies outside the donor
hull, or when you want model selection over donors to be automatic and
reproducible rather than hand-curated. It is the lasso sibling of the penalized
synthetic control :doc:`pda` (LASSO) variant: both select donors with an
:math:`\ell_1` penalty, but SCUL pairs that with the rolling-origin
cross-validation and the unit-free fit diagnostics of the source paper.

Notation
--------

One treated unit :math:`i = 0` is observed over :math:`T` periods, treated after
:math:`T_0`. Write :math:`y_t` for its outcome and :math:`X_t \in \mathbb{R}^P`
for the donor pool at period :math:`t` -- a (possibly high-dimensional,
multi-type) vector of candidate predictors built from the donor units. The
synthetic control is the lasso fit

.. math::

   (\hat{\beta}_0, \hat{\beta}) \;=\;
     \arg\min_{\beta_0,\,\beta}\ \frac{1}{2T_0}\sum_{t=1}^{T_0}
       \bigl(y_t - \beta_0 - X_t' \beta\bigr)^2
       \;+\; \lambda \lVert \beta \rVert_1 ,

with the donor columns standardised to unit (population) variance before the
penalty is applied. The synthetic series is
:math:`\hat{y}_t = \hat{\beta}_0 + X_t' \hat{\beta}` over all :math:`t`, and the
treatment effect in a post period is :math:`\hat{\tau}_t = y_t - \hat{y}_t`. The
reported ATT is the post-period mean
:math:`\frac{1}{T-T_0}\sum_{t>T_0}\hat{\tau}_t`.

The penalty :math:`\lambda` is selected by a rolling-origin expanding-window
cross-validation: for each expanding training window of pre-periods, a lasso is
fit over a grid of penalties and scored on the next ``training_post_length``
pre-periods (an out-of-sample window mimicking the post-treatment forecast); the
selected penalty is the median across windows of each window's
error-minimising value (the reference's ``lambda.median``).

Assumptions
-----------

Assumption 1 (single treated unit, balanced panel).
   Exactly one unit is treated, after a known period :math:`T_0`, on a strongly
   balanced panel.

   Remark. SCUL estimates one counterfactual series; staggered or multi-unit
   designs are out of scope for this estimator.

Assumption 2 (predictive pre-period relationship).
   The treated unit's untreated outcome is well approximated, in the
   pre-period, by a sparse linear combination of the donor pool. SCUL does not
   require the weights to be convex; it requires only that such a (possibly
   extrapolating) combination exists and is stable across the pre/post boundary.

   Remark. Unrestricted weights buy flexibility at the cost of a higher
   overfitting risk than convex SC. The rolling-origin cross-validation is the
   guardrail: a penalty that only fits in-sample noise scores poorly on the
   held-out windows and is not selected.

Assumption 3 (continuous, high-dimensional donors are admissible).
   The pool may contain more donors than pre-periods and donors of several
   variable types. For continuously distributed donors the lasso solution is
   unique [TIBSHIRANI2013]_, so the selected synthetic control is well defined.

   Remark. This is the assumption that makes SCUL different from convex SC: it
   embraces, rather than avoids, the high-dimensional donor pool that the
   convexity restriction cannot accommodate.

Inference and Diagnostics
-------------------------

Pre-treatment fit is summarised by a unit-free Cohen's D -- the mean absolute
pre-period gap divided by the pre-period standard deviation of the outcome --
reported in ``fit_diagnostics``. Small values indicate a synthetic control that
tracks the treated unit closely before treatment.

Inference is a placebo permutation test. Each donor unit is treated, in turn, as
a fake-treated target and fit with SCUL on the remaining pool; the test
statistic is the unit-free ratio of the post- to pre-treatment root mean squared
gap [ABADIE2010]_. The p-value is the rank of the real treated unit's ratio in
the placebo distribution. Placebo units whose pre-fit exceeds
``cohensd_threshold`` are trimmed, per the paper's quality-control
recommendation. Set ``inference=False`` to skip the (re-fitting) placebo loop.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SCUL
   from mlsynth.config_models import SCULConfig

   # California (Proposition 99): cigarette sales + retail price, 1970-2000.
   df = pd.read_csv("basedata/california_panel.csv")
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   config = SCULConfig(
       df=df, outcome="cigsale", treat="treat", unitid="state", time="year",
       donor_variables=["retprice"],   # widen the pool with each state's price
       display_graphs=True,
   )
   results = SCUL(config).fit()

   print(results.effects.att)                       # post-period ATT
   print(results.method_details.parameters_used)    # selected lambda, Cohen's D, pool size

Verification
------------

SCUL is cross-validated value-for-value against the authors' reference R package
(`hollina/scul <https://github.com/hollina/scul>`_) on the California
(Proposition 99) cigarette panel: the rolling-origin cross-validation penalty
matches ``glmnet`` to ten digits, and -- since the lasso solution is unique for
continuous donors -- the weights and synthetic series agree up to solver
tolerance. See the durable benchmark
`benchmarks/cases/scul_prop99.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/scul_prop99.py>`_
and the :doc:`replications/scul` page.

Core API
--------

.. autoclass:: SCUL
   :members: fit

.. autoclass:: mlsynth.config_models.SCULConfig
   :members:

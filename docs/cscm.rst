Flexible Count Synthetic Control (CSCM)
=======================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

``CSCM`` is a synthetic control designed for count data and other non-negative
outcomes -- deaths, homicides, disease cases, accident counts, claim counts.
It is a port of the flexible synthetic control of Bonander [CSCM]_. Reach for
it when you have a single treated unit, a pool of untreated donor units, a
panel of pre- and post-intervention periods, and an outcome that cannot go
negative and is naturally read on a multiplicative (rate-ratio) scale.

The problem it solves is a tension in the ordinary synthetic control method.
The textbook estimator constrains the donor weights to be non-negative and to
sum to one. Those two constraints keep the counterfactual inside the range of
the donors, which for a non-negative outcome has a useful side effect: a
weighted average of non-negative series is itself non-negative, so the imputed
counterfactual can never turn negative. But the same constraints often produce
a poor pre-treatment fit. The usual remedies -- an unconstrained regression, an
elastic net, adding an intercept -- restore the fit but let the counterfactual
go negative, which is meaningless for a count.

CSCM keeps the property you need for counts and relaxes only the one that hurts
the fit. It retains non-negativity of the weights (so the counterfactual stays
non-negative) but drops the adding-up constraint, penalising how far the weights
stray from the classic simplex solution. A single tuning parameter dials the
amount of extrapolation: none recovers ordinary synthetic control, more allows
the weights to leave the simplex where that demonstrably improves the fit.

Reach for CSCM when
^^^^^^^^^^^^^^^^^^^

* The outcome is a count or rate that cannot be negative, and a negative
  counterfactual would be nonsensical.
* Ordinary synthetic control fits the pre-treatment path poorly, but you do not
  want to abandon the non-negativity guarantee to fix it.
* A rate ratio (a multiplicative effect) is the natural way to report the
  result.

Notation
--------

There is one treated unit :math:`i=1` and :math:`n_0` donor units, observed over
:math:`T_0` pre-treatment and :math:`T_1` post-treatment periods. Let
:math:`Y_{it}\in\mathbb{R}^{0+}` be the outcome of unit :math:`i` at time
:math:`t`. Stack the treated unit's pre-treatment outcomes (and any auxiliary
covariates) into :math:`\mathbf{X}_1` and the donors' into :math:`\mathbf{X}_0`.
The synthetic control is a weighted donor average
:math:`\sum_{i=2}^{n} Y_{it} w_i`, and :math:`\mathbf{V}` is a diagonal matrix
of feature-importance weights.

The classic simplex weights solve

.. math::

   \mathbf{W}^{scm} = \arg\min_{\mathbf{W}}
   (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{W})' \mathbf{V}
   (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{W}),\quad
   w_i \ge 0,\ \textstyle\sum_i w_i = 1 .

CSCM keeps :math:`w_i \ge 0` but replaces the sum-to-one constraint with a
penalty toward :math:`\mathbf{W}^{scm}`:

.. math::

   \mathbf{W}^{*} = \arg\min_{\mathbf{W}\ge 0}
   (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{W})' \mathbf{V}
   (\mathbf{X}_1 - \mathbf{X}_0 \mathbf{W})
   + \lambda \lVert \mathbf{W} - \mathbf{W}^{scm} \rVert_2^2 .

As :math:`\lambda\to\infty` the solution returns to :math:`\mathbf{W}^{scm}`; at
:math:`\lambda=0` it allows the most extrapolation the non-negativity constraint
permits. Because :math:`\mathbf{W}^{scm}` sums to one, penalising the distance
from it indirectly penalises departures from the simplex, so :math:`\lambda`
controls how far the weights may leave the convex hull of the donors.

Assumptions
-----------

1. Non-negative outcome. The outcome is a count or otherwise non-negative
   quantity. CSCM rejects a panel whose outcome column has negative values.

   Remark. Non-negativity of the weights guarantees a non-negative
   counterfactual only because the donor outcomes are themselves non-negative;
   this is what makes the relaxation safe for counts where an intercept-based
   relaxation is not.

2. A synthetic control that matches the treated unit on lagged outcomes (and
   covariates) in the pre-period tracks its untreated potential outcome in the
   post-period.

   Remark. This is the usual synthetic-control identifying assumption (a linear
   factor structure with matched loadings); the relaxation changes only how the
   weights are estimated, not what they are assumed to recover.

3. No interference. One unit's treatment does not affect another's outcome, so
   the donors are untreated throughout.

   Remark. Standard SUTVA/consistency; excludes spillover onto the donor pool.

Inference
---------

The effect is reported as a rate ratio -- the multiplicative analogue of the
ATT, natural for counts -- of the summed post-period observed outcome to the
summed post-period counterfactual. Its confidence interval comes from a K-fold
cross-fitting procedure (Chernozhukov, Wuthrich and Zhu [CWZ]_) that also
removes the synthetic control's finite-sample bias: the pre-period is split into
:math:`K` blocks; for each block the weights are refit on the block's
complement, the post-period log rate ratio is corrected by the block's placebo
log rate ratio, and the corrected estimates are combined into a
:math:`t`-interval with :math:`K-1` degrees of freedom. ``K=2`` is preferable on
short pre-periods and ``K=3`` on long ones; with small :math:`K` the interval is
wide, which the degrees of freedom make explicit.

The predictor-importance matrix :math:`\mathbf{V}` is chosen by a leave-one-out
Poisson ridge of the donors' post-period outcome on the balance features
(``v_method="poisson_ridge"``), or set to equal weights
(``v_method="uniform"``), the simple default. On panels where the ridge shrinks
heavily the two coincide.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import CSCM

   # Vision Zero: Sweden's 1997 road-safety policy; road-death rate.
   df = pd.read_csv("basedata/viszero.csv")
   donors = [2, 3, 5, 7, 9, 10, 13, 14, 16]
   df = df[df["ID"].isin([25] + donors)].copy()
   df["treated"] = ((df["ID"] == 25) & (df["TIME"] >= 1996)).astype(int)

   res = CSCM({
       "df": df, "outcome": "deathrate_mln", "treat": "treated",
       "unitid": "ID", "time": "TIME", "K": 2, "v_method": "uniform",
       "display_graphs": False,
   }).fit()

   rr = res.effects.additional_effects["rate_ratio"]      # ~1.06 (no detectable effect)
   lo, hi = res.inference.ci_lower, res.inference.ci_upper  # wide, spans 1
   res.weights.donor_weights                               # relaxed weights, sum < 1

Verification
------------

CSCM is cross-validated against the authors' own R implementation on the Vision
Zero panel: the classic warm-start concentrates on Finland, the relaxed weights
extrapolate below the simplex (their sum falls below one), and the cross-fitted
rate ratio matches the reference to about one percent. See the durable
benchmark ``benchmarks/cases/cscm_viszero.py`` and the replication page
:doc:`replications/cscm`.

Core API
--------

.. autoclass:: CSCM
   :members:

.. autoclass:: mlsynth.utils.cscm_helpers.config.CSCMConfig
   :members:

.. [CSCM] Bonander, C. (2021). A (Flexible) Synthetic Control Method for Count
   Data and Other Non-Negative Outcomes. *Epidemiology*, 32(4), e18-e19.

.. [CWZ] Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). Practical and robust
   t-test based inference for synthetic control and related methods.

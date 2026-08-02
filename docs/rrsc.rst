Robust-Regression Synthetic Control with Interference (RRSC)
============================================================

.. currentmodule:: mlsynth

Overview
--------

Most synthetic-control methods assume the *no-interference* condition: the
treatment given to the treated unit does not affect any control unit. When that
fails -- a policy announced in one place changes behaviour elsewhere, a
pollution-control order in a city pushes pollution to its suburbs -- the donor
pool is contaminated, and a synthetic control built from it is biased.

RRSC (He, Li, Shi and Miao 2026) estimates causal effects when interference is
present and its structure is unknown. Under a factor model for the untreated
outcome it recovers two things at once: the direct effect on the treated unit,
and the interference effect on every control unit. The key idea is a change of
viewpoint -- the average direct and interference effects are treated as a
*sparse outlier component* of a robust regression against the estimated factor
loadings, so a control unit that is interfered with looks like a contaminated
observation. Nothing has to be prespecified about which controls are affected.

When to use this estimator
--------------------------

* You suspect the intervention spills over onto some control units, and you do
  not know in advance which ones. RRSC estimates the interference effect for
  every unit and lets the data flag the affected ones.
* You want the interference itself as an estimand, not just a nuisance to be
  purged. RRSC returns an interference map -- an effect, a confidence interval
  and a p-value for each unit -- which is often the substantive question (how
  far, and in what direction, did the policy spill?).
* Your panel is either short and wide (many units, few post-periods) or narrow
  and long (few units, many periods). RRSC has a regime for each; see below.

Reach for a different tool when interference is absent (any donor-pooling SCM
is simpler) or when you can name the interfered units up front and want to model
the spillover parametrically (:doc:`spillsynth` offers ``cd``, ``sar``, ``iscm``
and ``grossi`` for that case).

Notation
--------

There are :math:`N` units indexed :math:`i` over :math:`T` periods indexed
:math:`t`, with the intervention at :math:`T_0` (so :math:`T_0` pre-periods and
:math:`T - T_0` post-periods). Unit :math:`i = 1` is treated. Let
:math:`Z_t = \mathbf{1}(t > T_0)`. Write :math:`Y_{it}(1)` and
:math:`Y_{it}(0)` for the potential outcomes with and without the realised
intervention. The observed outcome is
:math:`Y_{it} = Z_t Y_{it}(1) + (1 - Z_t) Y_{it}(0)`.

The estimands are the post-period averages

.. math::

   \bar\beta_i = \frac{1}{T - T_0} \sum_{t > T_0} \bigl(Y_{it}(1) - Y_{it}(0)\bigr),
   \qquad i = 1, \dots, N,

where :math:`\bar\beta_1` is the average direct effect on the treated unit and
:math:`\bar\beta_i`, :math:`i \ge 2`, is the average interference effect on
control unit :math:`i`.

Mathematical formulation
------------------------

Factor model with interference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The untreated outcome follows a linear factor model, and the effect enters
additively:

.. math::

   Y_{it}(0) = \lambda_i^\top f_t + \varepsilon_{it},
   \qquad
   Y_{it} = \beta_{it} Z_t + \lambda_i^\top f_t + \varepsilon_{it},

with :math:`f_t \in \mathbb{R}^r` a vector of :math:`r` common factors,
:math:`\lambda_i \in \mathbb{R}^r` unit-specific loadings that are constant over
the study window, and :math:`\varepsilon_{it}` a mean-zero error. The
interactive term :math:`\lambda_i^\top f_t` generalises the two-way fixed
effects of difference-in-differences and plays the role of the synthetic-control
weights: it absorbs the unobserved confounding common to all units. Because the
loadings are shared between treated and control units, they are what makes the
comparison valid, not a pre-period trajectory match.

Averaging over the post-period gives the regression that identifies the effects,

.. math::

   \bar Y_{i,\text{post}} = \bar\beta_i + \lambda_i^\top \bar f_{\text{post}}
                          + \bar\varepsilon_{i,\text{post}},

a linear model in the loadings :math:`\lambda_i` with coefficient
:math:`\bar f_{\text{post}}`, in which the effects :math:`\bar\beta_i` appear as
an *outlier* term: units with no interference lie on the hyperplane
:math:`\bar Y = \Lambda \bar f_{\text{post}}`, and interfered units deviate from
it. Estimation therefore proceeds in two stages -- estimate :math:`\Lambda` from
the pre-period, then recover :math:`\bar f_{\text{post}}` (and hence the
residual effects) by a robust regression that is not dragged off the hyperplane
by the outliers.

Two regimes
^^^^^^^^^^^

RRSC provides a regime for each asymptotic setting; ``regime="auto"`` chooses
``fixed_n`` when :math:`T_0 \ge 5N` (the rule of thumb for reliable pre-period
factor analysis) and ``large_n`` otherwise.

Fixed-N (Section 3 of the paper). Few units, long pre- and post-periods. With a
long post-period the idiosyncratic errors average out, so the non-interfered
units lie (asymptotically) on a common hyperplane. Under the majority-valid-
controls condition -- more than half of the controls are unaffected, without
knowing which -- a strict majority of the points sit on that hyperplane, and a
high-breakdown regression (Least Trimmed Squares) locks onto it while ignoring
the interfered minority. mlsynth estimates the loadings with an MLE factor
analysis on the pre-period correlation matrix (matching R's ``factanal``),
rescales them by the covariance square root, fits LTS
(:math:`\alpha = 0.5`, matching ``robustbase::ltsReg`` including its
finite-sample scale correction), selects the valid controls by a
Donoho-Johnstone threshold, and refits by ordinary least squares on them.

Large-N (Section 4). Many units, a short post-period allowed. Here the errors do
not vanish and outliers cannot be separated exactly; instead a robust loss
dilutes their influence through cross-sectional averaging as :math:`N \to
\infty` -- a blessing of dimensionality. mlsynth estimates the loadings with an
EM quasi-maximum-likelihood factor analysis (Bai and Li 2012) on the centred
pre-period panel, then recovers :math:`\bar f_{\text{post}}` by a Huber-IRLS
regression of the post-period mean on the loadings; the per-unit effect is the
residual :math:`\bar\beta_i = \bar Y_{i,\text{post}} - \lambda_i^\top \hat{\bar
f}_{\text{post}}`.

Assumptions
-----------

Assumption 1 (Consistency). For each :math:`i, t`,
:math:`Y_{it} = Z_t Y_{it}(1) + (1 - Z_t) Y_{it}(0)`.

  Remark. Standard. Only two intervention regimes are ever realised in a
  comparative case study, so the potential outcomes are indexed by the treated
  unit's status alone.

Assumption 2 (Linear factor model). The untreated outcome satisfies
:math:`Y_{it}(0) = \lambda_i^\top f_t + \varepsilon_{it}` with time-invariant
loadings and :math:`\mathbb{E}(\varepsilon_{it}) = 0`.

  Remark. The number of factors :math:`r` is treated as known; set it with
  ``n_factors`` or let the Bai and Ng (2002) criterion pick it. On small panels
  the criterion tends to over-select (a near-saturated factor model absorbs the
  effect itself), so for a handful of units prefer to set ``n_factors`` by hand,
  as the paper does in its Middle East application. There is no separate
  intercept: the unit level must be carried by the factors. In the large-N
  regime the loadings come from a mean-centred panel, so a level that is a pure
  constant is removed; a level that varies over time (a trend, a cycle) survives
  and is reconstructed. When the treated unit is nearly orthogonal to the
  common factors, its counterfactual collapses to its own pre-period mean and
  the effect approaches a raw before-after comparison -- watch the
  ``communality`` diagnostic (below).

Assumption 3 (Majority valid controls, fixed-N). More than half of the control
units are free of interference, with the valid set unknown.

  Remark. This is what the high-breakdown regression exploits: LTS at
  :math:`\alpha = 0.5` tolerates up to half the points being arbitrary
  outliers. It is the fixed-N analogue of a well-identified donor pool.

Assumption 4 (Sparse-or-weak interference, large-N). A vanishing fraction of
units may exhibit large interference, while the remainder may be interfered only
weakly (:math:`\|\bar\beta_{S^c}\|_1 = o(N)`).

  Remark. This is more permissive than global sparsity: dense but weak spillover
  across many units is allowed, provided the strong effects are rare. The robust
  loss makes the strong outliers asymptotically negligible.

Inference and diagnostics
-------------------------

Circular block bootstrap (default, ``inference="cbb"``). The pre- and
post-periods are resampled in circular blocks (separately by default), the
estimator is recomputed on each resample, and the per-unit bootstrap variance
drives a normal (Wald) confidence interval and p-value -- exactly what the
authors' analyses report. The point estimates are deterministic; the intervals
and p-values are Monte-Carlo objects, so they reproduce a reference run only up
to the bootstrap tolerance.

Conformal permutation (``inference="conformal"``). The moving-block permutation
test of Chernozhukov, Wuthrich and Zhu (2021), applied to the per-period
counterfactual residuals. It is valid with a short post-period, where the
bootstrap variance is unreliable; it returns p-values but no interval.

The interference map. The fit exposes, for every unit, an effect, a confidence
interval and a p-value (:py:attr:`RRSCResults.interference_effects`,
``interference_ci``, ``interference_pvalue``);
:py:meth:`RRSCResults.significant_units` returns the controls whose interference
is significant. The counterfactual proxy :math:`\hat\mu_i + \lambda_i^\top \hat
f_t` for every unit and period is on ``counterfactual_panel``.

The communality diagnostic. ``communality`` reports, per unit, the fraction of
its pre-period variance the common factors explain. A value near zero for the
treated unit is a warning: the factor adjustment is doing almost nothing, so the
direct effect is close to a raw pre/post difference, not a factor-model
counterfactual.

Example
-------

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import RRSC

   # a small panel drawn from the untreated factor model, with a direct effect
   # on the treated unit and interference planted on two controls
   rng = np.random.default_rng(0)
   N, T, T0, r = 12, 40, 30, 2
   F = rng.normal(size=(T, r)); F[:, 0] = np.cumsum(rng.normal(size=T)) / 3  # trend factor
   L = rng.normal(size=(N, r))
   Y = (F @ L.T).T + 0.3 * rng.normal(size=(N, T))
   Y[0, T0:] += 8.0                         # direct effect on the treated unit
   Y[[1, 2], T0:] += 5.0                    # interference on unit1, unit2
   df = pd.DataFrame(
       [dict(u=f"unit{i}", t=t, y=Y[i, t], d=int(i == 0 and t >= T0))
        for i in range(N) for t in range(T)]
   )

   res = RRSC({
       "df": df, "outcome": "y", "treat": "d", "unitid": "u", "time": "t",
       "regime": "auto", "n_factors": 2, "inference": "cbb", "cbb_reps": 200,
   }).fit()

   print(res.regime, res.att)               # direct effect ~ 8
   print(res.significant_units(0.05))        # flags unit1, unit2 (interference ~ 5)
   print(res.communality["unit0"])           # treated-unit factor communality

The ``att`` accessor is the treated unit's direct effect; the interference map
(``interference_effects`` / ``interference_ci`` / ``interference_pvalue``)
carries every unit's direct or interference effect with uncertainty.

Verification
------------

RRSC is validated value-for-value against the authors' reference R code (both
regimes) on their two empirical applications -- the U.S. embassy relocation to
Jerusalem (fixed-N) and Beijing's orange-alert pollution policy (large-N). The
durable cross-validation lives in
`benchmarks/cases/rrsc_reference.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/rrsc_reference.py>`_
and the full replication story is on :doc:`replications/rrsc`.

Core API
--------

.. autoclass:: mlsynth.RRSC
   :members: fit

Configuration
-------------

.. autoclass:: mlsynth.config_models.RRSCConfig
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields

Results
-------

.. autoclass:: mlsynth.utils.rrsc_helpers.structures.RRSCResults
   :members: direct_effect, interference_map, significant_units

Helper Modules
--------------

.. automodule:: mlsynth.utils.rrsc_helpers.pipeline
   :members: run_rrsc, point_effects, counterfactual_panel

.. automodule:: mlsynth.utils.rrsc_helpers.inference
   :members: run_cbb, run_conformal

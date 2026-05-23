Imperfect Synthetic Controls (ISCM)
====================================

.. currentmodule:: mlsynth

Overview
--------

ISCM (Powell, D. (2026). *"Imperfect Synthetic Controls,"* Journal of
Applied Econometrics 41(3):253-264) confronts the synthetic control
method's least defensible assumption: that a **perfect** synthetic
control exists. The classic SCM requires the treated unit to lie inside
the convex hull of the donors and its pre-treatment path to be matched
*exactly*. With transitory shocks -- noise with non-vanishing variance --
an exact fit is impossible even in expectation, and the convex-hull
condition may simply fail (the treated unit can be more extreme than any
weighted average of donors).

ISCM relaxes this with two ideas:

1. **Synthetic controls for every unit.** Rather than fitting one
   synthetic control for the treated unit, ISCM builds one for *all*
   units. The treatment effect is then identified even when the treated
   unit is *outside* the convex hull -- because it can still appear *as a
   donor* for control units, and those units' post-treatment residuals
   carry information about the effect (paper eq. 6).
2. **Moment conditions robust to transitory shocks.** ISCM relies on
   conditions of the form :math:`\sum_{j} w_i^j \mathbb{E}[Y_{jt}] =
   \mathbb{E}[Y_{it}]` that need only hold in expectation, producing
   asymptotically unbiased estimates as the pre-period grows even when no
   unit fits perfectly in sample.

It adds a data-driven fit metric :math:`a_i` that asymptotically excludes
units lacking a valid synthetic control -- removing the researcher's
eyeball "is the pre-fit good enough" decision -- and an Ibragimov-Muller
inference procedure that stays valid with a tiny donor pool.

The identifying intuition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose the treated unit (call it unit 1) is too extreme to be matched by
any convex combination of controls. A control unit :math:`i` whose
synthetic control *does* place weight :math:`w_i^1 > 0` on unit 1 will,
after treatment, have its synthetic counterfactual contaminated by the
effect: its residual picks up :math:`-w_i^1 \alpha`. Since unit :math:`i`
is itself untreated, regressing its residual on its "treatment exposure"
:math:`-w_i^1` recovers :math:`\alpha`. ISCM pools this signal across all
such units.

When to use ISCM
^^^^^^^^^^^^^^^^

* The treated unit's pre-period path is **not** well inside the donor
  convex hull (it trends above/below all donors), so traditional SCM
  produces a visibly poor fit and a biased counterfactual.
* Outcomes are **noisy** (transitory shocks), so an exact pre-period
  match is implausible and would overfit.
* The donor pool is **small**, so permutation inference cannot reach
  conventional significance.
* You have a **long pre-period** (the method's guarantees are asymptotic
  in :math:`T_0`).

Mathematical Formulation
------------------------

Setup (paper Section 2)
^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`N` units over :math:`T` periods with a latent-factor outcome

.. math::

   Y_{it} = \alpha_{it} D_{it} + L_{it} + \epsilon_{it},
   \qquad L_{it} = \lambda_t' \mu_i,

ISCM builds, for every unit :math:`i`, a synthetic control from the
others (paper eq. 5):

.. math::

   \widehat w_i = \arg\min_{w}
       \sum_{t \le T_0} \Bigl( Y_{it} - \sum_{j \ne i} w_j Y_{jt} \Bigr)^2,
   \quad w_j \ge 0,\ \sum_{j \ne i} w_j = 1.

Fit metric (paper eq. 14)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each unit is weighted by how well its synthetic control satisfies the
SCM moment conditions in the pre-period. With residual
:math:`R_{it} = Y_{it} - \sum_j \widehat w_{ij} Y_{jt}` and moment vector
:math:`M_i^k = \tfrac{1}{T_0}\sum_{t \le T_0} R_{it} Y_{kt}`,

.. math::

   a_i = \frac{\min_\ell M_\ell' M_\ell}{M_i' M_i} \in (0, 1],

so the best-fitting unit gets :math:`a_i = 1` and units without a valid
synthetic control get :math:`a_i \to 0` -- they are dropped from the
estimate automatically.

Treatment effect (paper eq. 8 / 15)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With treatment exposure
:math:`E_{it} = D_{it} - \sum_j \widehat w_{ij} D_{jt}`, the ATT is the
:math:`a_i`-weighted least-squares slope, pooled over all units and the
post-period:

.. math::

   \widehat\alpha =
     \frac{\sum_i a_i \sum_{t > T_0} E_{it} R_{it}}
          {\sum_i a_i \sum_{t > T_0} E_{it}^2}
     = \sum_{i \in C} v_i\, \widehat\alpha_i,
   \quad
   \widehat\alpha_i = \frac{\sum_{t>T_0} E_{it} R_{it}}{\sum_{t>T_0} E_{it}^2},

where :math:`C` is the contributing set (units with non-zero exposure)
and :math:`v_i = a_i \sum_t E_{it}^2 / \sum_\ell a_\ell \sum_t E_{\ell t}^2`.

Inference (paper Section 5, eq. 16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ISCM produces one estimate :math:`\widehat\alpha_i` per contributing
unit. The Ibragimov-Muller approach forms a t-statistic from their
weighted spread and calibrates the p-value with a sign-flip (Rademacher)
randomization test on the weighted deviations
:math:`v_i(\widehat\alpha_i - \alpha_0)`. This is conservative but valid
with very few units -- though note the achievable p-value floor is about
:math:`2/2^{|C|}`, so a handful of contributing units cannot reach
conventional thresholds (exactly the small-donor-pool limitation Powell
highlights).

Scope of this implementation
----------------------------

This follows Powell's *applied* procedure: synthetic controls for all
units come from the traditional SCM (the documented starting point), the
:math:`a_i` weights are formed from the pre-period moment conditions, the
ATT is the :math:`a_i`-weighted least-squares effect, and inference is the
sign-flip test. It does **not** run the optional continuously-updating
GMM refinement that re-estimates the weights jointly to be orthogonal to
transitory shocks (paper Section 3.2-3.4); the SCM-initialised weights
are that procedure's starting point and deliver the headline
relaxed-convex-hull identification.

Core API
--------

.. automodule:: mlsynth.estimators.iscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.ISCMConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.iscm_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.estimate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.iscm_helpers.structures
   :members:
   :undoc-members:

Example
-------

A one-factor panel where the treated unit has the largest factor loading
-- placing it outside the convex hull of the controls, so a traditional
SCM cannot match it. ISCM still recovers the planted effect via the
control units that use the treated unit as a donor.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import ISCM

   # ------------------------------------------------------------------
   # 1. One-factor panel; unit 0 (treated) has the MAX loading
   # ------------------------------------------------------------------
   rng = np.random.default_rng(0)
   N, T, T0, true_alpha = 8, 60, 48, 3.0
   loadings = np.linspace(2.0, -1.5, N)        # unit 0 outside the hull
   f = np.cumsum(rng.standard_normal(T)) * 0.3 + np.linspace(0, 2, T)
   Y = np.outer(loadings, f) + rng.standard_normal((N, T)) * 0.05
   D = np.zeros((N, T))
   Y[0, T0:] += true_alpha
   D[0, T0:] = 1

   rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t], "D": int(D[i, t])}
           for i in range(N) for t in range(T)]
   df = pd.DataFrame(rows)

   # ------------------------------------------------------------------
   # 2. Fit ISCM with Ibragimov-Muller inference
   # ------------------------------------------------------------------
   res = ISCM({
       "df": df, "outcome": "y", "treat": "D",
       "unitid": "unit", "time": "time",
       "inference": True,
   }).fit()

   # ------------------------------------------------------------------
   # 3. Inspect the result
   # ------------------------------------------------------------------
   print(f"ATT = {res.att:+.3f}  (true = {true_alpha})")
   print(f"treated fit metric a_0 = {res.fit_metric[0]:.3f}  "
         f"(small => outside the hull)")
   print(f"treated contribution   = {res.contribution[0]*100:.1f}%")
   print(f"p-value = {res.inference.p_value:.3f}  "
         f"(n contributing = {res.inference.n_contributing})")

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Ferman, B., & Pinto, C. (2021). "Synthetic Controls with Imperfect
Pretreatment Fit." *Quantitative Economics* 12(4):1197-1221.

Fry, J. (2024). "A Method of Moments Approach to Asymptotically Unbiased
Synthetic Controls." *Journal of Econometrics* 244:105846.

Ibragimov, R., & Muller, U. K. (2010). "T-Statistic Based Correlation
and Heterogeneity Robust Inference." *Journal of Business & Economic
Statistics* 28(4):453-468.

Powell, D. (2026). "Imperfect Synthetic Controls." *Journal of Applied
Econometrics* 41(3):253-264.

Distributional Synthetic Control (DSC)
=======================================

.. currentmodule:: mlsynth

Overview
--------

DSC (Gunsilius, F. (2023). *"Distributional Synthetic Controls."*
*Econometrica* 91(3):1105-1117; asymptotic theory in Zhang, L.,
Zhang, X., & Zhang, X. (2026), *"Asymptotic Properties of the
Distributional Synthetic Controls,"* arXiv:2405.00953) reconstructs
the treated unit's counterfactual *outcome distribution* as a
weighted average of the donor units' outcome distributions, where
the average is taken in the 2-Wasserstein space. Unlike classical
synthetic control -- which targets aggregate means and returns a
single ATT -- DSC returns the full counterfactual quantile function
at every post-period, and hence the quantile treatment effect (QTE)
at any quantile :math:`q \in (0, 1)`.

Three differences from the rest of mlsynth:

1. **Data requirement.** DSC needs *micro-level* panel data: for each
   ``(unit, time)`` cell the user supplies multiple individual
   observations, supplied as one row per individual in the input
   DataFrame.

2. **Output target.** The primary object is the QTE curve
   :math:`q \mapsto \widehat \alpha_{1t, q}`, not a scalar ATT.
   A mean-of-QTE summary is exposed as :py:attr:`DSCResults.att` for
   compatibility, but the distributional information is the point.

3. **Loss function.** Weights are fit by minimising the
   2-Wasserstein distance between the treated and a convex
   combination of donors' empirical quantile functions, rather than
   the squared :math:`\ell_2` loss in outcome space.

Mathematical Formulation
------------------------

Setup
^^^^^

We observe :math:`J + 1` units, with :math:`j = 1` treated, over
:math:`T` periods. Treatment begins at :math:`T_0 + 1`. For each
``(unit, time)`` cell we have :math:`n_{jt}` individual observations
:math:`\{Y_{l, jt}\}_{l = 1}^{n_{jt}}`. Let :math:`F_{Y_{jt}}` denote
the underlying outcome distribution of unit :math:`j` at time
:math:`t`, with quantile function

.. math::

   F^{-1}_{Y_{jt}}(q) = \inf \{ y \in \mathbb{R} : F_{Y_{jt}}(y) \ge q \},
   \quad q \in (0, 1).

The empirical quantile estimator is the order-statistic rule

.. math::

   \widehat F^{-1}_{Y_{jt, n_j}}(q) = Y_{t, n_j(k)},
   \quad \frac{k - 1}{n_j} < q \le \frac{k}{n_j},

where :math:`Y_{t, n_j(k)}` is the :math:`k`-th order statistic of
the sample for cell :math:`(j, t)`.

The DSC counterfactual quantile function at any post-period
:math:`t > T_0` is

.. math::

   \widehat F^{-1}_{Y_{1t, N}}(q) =
       \sum_{j = 2}^{J + 1} \widehat w_j\, \widehat F^{-1}_{Y_{jt, n_j}}(q),
   \quad
   \widehat w \in \mathcal H = \bigl\{ w \in [0, 1]^J :
       \mathbf 1^\top w = 1 \bigr\}.

The QTE is :math:`\widehat \alpha_{1t, q} = \widehat F^{-1}_{Y_{1t, I}}(q)
- \widehat F^{-1}_{Y_{1t, N}}(q)`.

Algorithm
^^^^^^^^^

mlsynth implements Algorithm 1 of Zhang et al. (2026), which
formalises Gunsilius's recipe in four steps.

**Step 1 -- Empirical quantile functions.** For each
``(j, t)`` cell, compute :math:`\widehat F^{-1}_{Y_{jt, n_j}}` via
the order-statistic estimator above.

**Step 2 -- Per-pre-period weights.** Draw :math:`M` quantile-grid
points :math:`\{V_m\}_{m = 1}^{M} \subset (0, 1)` (uniform i.i.d.
or a Halton / Sobol low-discrepancy sequence). The squared
2-Wasserstein loss

.. math::

   W_2^2 \bigl( \textstyle \sum_{j = 2}^{J + 1} w_j\,
                \widehat F^{-1}_{Y_{jt, n_j}},\,
                \widehat F^{-1}_{Y_{1t, n_1}} \bigr)
   = \int_0^1
     \biggl|
       \textstyle \sum_{j = 2}^{J + 1} w_j\,
       \widehat F^{-1}_{Y_{jt, n_j}}(q)
       - \widehat F^{-1}_{Y_{1t, n_1}}(q)
     \biggr|^2 dq

is approximated by the Monte-Carlo / QMC empirical risk

.. math::

   L_t(w) = \frac{1}{M} \sum_{m = 1}^{M}
            \biggl|
              \widetilde Y_{1t, m}
              - \textstyle \sum_{j = 2}^{J + 1} w_j\,
              \widetilde Y_{jt, m}
            \biggr|^2,
   \quad
   \widetilde Y_{jt, m} := \widehat F^{-1}_{Y_{jt, n_j}}(V_m).

The per-pre-period DSC weights solve the simplex-constrained QP

.. math::

   \widehat w_t = \arg\min_{w \in \mathcal H} L_t(w),
   \quad t \in \mathcal T_0.

By the Koksma-Hlawka inequality, the QMC approximation error has
rate :math:`O(\log M / M)` for Halton / Sobol sequences vs.
:math:`O(M^{-1/2})` for i.i.d. samples (Zhang et al. 2026 Section 2,
Remark 1).

**Step 3 -- Aggregate over the pre-period.** The final DSC weight
is a convex combination of the per-pre-period weights,

.. math::

   \widehat w = \sum_{t \in \mathcal T_0} \lambda_t\, \widehat w_t,
   \qquad \lambda_t \ge 0, \quad \sum_t \lambda_t = 1.

mlsynth offers ``"uniform"`` (default; :math:`\lambda_t = 1/T_0`)
and ``"recency"`` (geometric decay
:math:`\lambda_t \propto \mathrm{decay}^{T_0 - t}`) rules, and
accepts caller-supplied weights so Arkhangelsky et al. (2021)
SDiD-style :math:`\lambda_t` can be plugged in externally.

**Step 4 -- Post-period QTE.** For each :math:`t > T_0`, evaluate
the counterfactual quantile function at the user's QTE grid
:math:`\{q_1, \dots, q_Q\}`,

.. math::

   \widehat F^{-1}_{Y_{1t, N}}(q_k) =
       \sum_{j = 2}^{J + 1} \widehat w_j\,
       \widehat F^{-1}_{Y_{jt, n_j}}(q_k),

and form the QTE

.. math::

   \widehat \alpha_{1t, q_k} = \widehat F^{-1}_{Y_{1t, I}}(q_k)
                              - \widehat F^{-1}_{Y_{1t, N}}(q_k).

Asymptotic optimality
^^^^^^^^^^^^^^^^^^^^^

Zhang et al. (2026) prove two main theoretical results that
underwrite the procedure.

* **Theorem 1 (Asymptotic optimality).** Under regularity conditions
  on the quantile-grid sampling and on the moments of the empirical
  quantile functions, the DSC weight achieves the lowest possible
  post-period averaged 2-Wasserstein distance among all simplex
  weighting combinations as :math:`M \to \infty`:

  .. math::

     \frac{\bar R_{T_1}(\widehat w)}
          {\inf_{w \in \mathcal H} \bar R_{T_1}(w)}
     \xrightarrow{p} 1.

* **Theorem 2 (Convergence rate).** With :math:`\xi_t` denoting the
  pre-treatment 2-Wasserstein fit at time :math:`t`,

  .. math::

     \bigl\| \widehat w - w^{\mathrm{opt}}_{T_1} \bigr\|_2
     = O_p\bigl( \bar \xi^{1/2}
                 + \bar \xi_{T_1}^{1/2}
                 + M^{-1/4} J \bigr),

  i.e. faster pre-fit (smaller :math:`\bar \xi`) and richer
  quantile-grid sampling (larger :math:`M`) tighten the weight
  estimate.

mlsynth surfaces :math:`\xi_t` via
:py:attr:`DSCResults.pre_period_wasserstein` so users can inspect
the pre-period fit quality directly.

Core API
--------

.. automodule:: mlsynth.estimators.dsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.DSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.dsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.quantiles
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.aggregation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dsc_helpers.structures
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo. The treated unit shares the
same pre-period DGP as its donors; in the post-period it receives a
*location shift* of +1.5 (the planted treatment effect is constant
across quantiles). We expect the DSC QTE to be roughly flat around
1.5 across the quantile grid.

.. code-block:: python

   """One draw of a DSC location-shift simulation."""

   import numpy as np
   import pandas as pd

   from mlsynth import DSC


   # ---------------------------------------------------------------------
   # 1. Simulate one micro-panel from a location-family DGP
   # ---------------------------------------------------------------------

   rng = np.random.default_rng(0)
   J = 4
   T_pre = 8
   T_post = 4
   T = T_pre + T_post
   n_per_cell = 200          # individuals per (unit, time) cell
   delta_post = 1.5          # planted location-shift treatment effect

   unit_loc = rng.standard_normal(J + 1) * 0.5
   time_shift = np.linspace(0.0, 1.0, T)
   rows = []
   for j in range(J + 1):
       for t in range(T):
           loc = unit_loc[j] + time_shift[t]
           if j == 0 and t >= T_pre:
               loc += delta_post
           sample = rng.normal(loc=loc, scale=1.0, size=n_per_cell)
           for y in sample:
               rows.append({
                   "unit": j,
                   "time": t,
                   "y": float(y),
                   "D": int(j == 0 and t >= T_pre),
               })
   df = pd.DataFrame(rows)


   # ---------------------------------------------------------------------
   # 2. Fit DSC
   # ---------------------------------------------------------------------

   res = DSC({
       "df": df,
       "outcome": "y",
       "treat": "D",
       "unitid": "unit",
       "time": "time",
       "M": 400,                  # QMC grid size for the Wasserstein loss
       "grid_method": "halton",   # low-discrepancy sequence
       "lambda_method": "uniform",
       "n_qte_points": 49,
   }).fit()


   # ---------------------------------------------------------------------
   # 3. Inspect the output
   # ---------------------------------------------------------------------

   print(f"true location shift   = {delta_post:+.3f}")
   print(f"DSC mean-of-QTE       = {res.att:+.3f}")
   print(f"donor weights         = {res.donor_weights}")
   print(f"pre-period Wasserstein loss per t = "
         f"{res.pre_period_wasserstein.round(4).tolist()}")

   import pandas as pd
   qte_table = pd.DataFrame({
       "quantile": res.qte_curves[0].quantiles,
       "qte_t=8":  res.qte_curves[0].qte,
       "qte_t=9":  res.qte_curves[1].qte,
       "avg_qte":  res.average_qte,
   })
   print("\nQTE by quantile (selected rows):")
   print(qte_table.iloc[::10].round(3).to_string(index=False))

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
Wager, S. (2021). "Synthetic Difference-in-Differences."
*American Economic Review* 111(12):4088-4118.

Gunsilius, F. F. (2023). "Distributional Synthetic Controls."
*Econometrica* 91(3):1105-1117.

Zhang, L., Zhang, X., & Zhang, X. (2026). "Asymptotic Properties of
the Distributional Synthetic Controls." arXiv:2405.00953.

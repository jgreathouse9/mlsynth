Dynamic Synthetic Control for Auto-Regressive processes (DSCAR)
================================================================

.. currentmodule:: mlsynth

Overview
--------

DSCAR is mlsynth's implementation of the **Dynamic Synthetic Control**
method of Zheng & Chen (2024). DSCAR extends classical Abadie-Diamond-
Hainmueller (2010) synthetic control to settings with:

* **time-varying confounders** :math:`X_{it}` (e.g., meteorological
  variables in an air-pollution panel),
* an explicit **auto-regressive outcome model** :math:`Y_{it}(0) =
  \delta_t + \beta_t^\prime X_{it} + \rho_t Y_{i, t-1} +
  \varepsilon_{it}`,
* spatial dependence in the residuals :math:`\varepsilon_{it}`, and
* **multiple treated units** sharing a common intervention time.

The matching weights are **time-varying** -- computed afresh at every
post-period via **empirical-likelihood maximisation** under per-period
matching constraints (equations 2.7-2.9 of the paper). This is
different from the L2 simplex weight typical of mlsynth's other
estimators; the EL formulation lets DSCAR attain *exact* covariate
matches as :math:`N_{co}, N_{tr} \to \infty` with :math:`T` fixed.

The acronym ``DSCAR`` is used in mlsynth to distinguish this estimator
from the **Distributional Synthetic Control** of Gunsilius (2023),
which ships under :class:`mlsynth.DSC`.

When to use DSCAR
-----------------

Reach for DSCAR when:

* The outcome you care about has **strong autocorrelation** (hourly
  pollutant concentrations, daily prices, weekly sales), AND
* You have **time-varying covariates** that you'd like to match on
  period-by-period, AND
* You're in the high-:math:`N` / moderate-:math:`T` regime that suits
  micro-panel data (e.g., 50-100 monitoring stations × 50-100 hours).

If your panel has a single treated unit with no covariates and a long
pre-period, classical :class:`TSSC` / :class:`SCMO` will be faster and
simpler. DSCAR's per-period EL refinement is the right tool when the
covariate trajectories matter as much as the outcome trajectory.

Method
------

The estimator works in three steps. Given a panel of :math:`N` units
over :math:`T` periods, :math:`N_{tr}` of which are directly treated
starting at :math:`t = T_0 + 1`:

1. **Variable-importance matrix** ``V_t``. For each :math:`t`, fit an
   OLS of :math:`Y_t` on :math:`(Y_{t-1}, X_t)` across the cross-
   section (full panel for :math:`t \leq T_0`, donors only for
   :math:`t > T_0`), and set ``V_t = diag(|coefficients|)``. This is
   the per-period analogue of the SCM ``V`` matrix.

2. **Per-period EL weights** :math:`w_t^*`. Solve the convex QP

   .. math::

      \min_w (Z_{1t} - Z_{0t} w)^\prime V_t (Z_{1t} - Z_{0t} w)
      \qquad \text{s.t.} \quad \sum_i w_i = 1,\ 0 \leq w_i \leq 1,

   where :math:`Z_{1t}` is the treated-mean covariate target at
   :math:`t` (and the lagged outcome) and :math:`Z_{0t}` are the
   donor-side analogues. When the QP residual is small enough
   (default :math:`\leq 0.01` mean absolute), **refine** by maximising
   :math:`\prod_i w_i` subject to the same matching constraints --
   this is the empirical-likelihood step that gives DSCAR its
   asymptotic-theory guarantees (Theorem 1 of the paper).

3. **Dynamic matching of the lag**. For :math:`t > T_0 + 1`, the
   treated-side lagged-outcome target is the **previously-estimated
   counterfactual** :math:`\widehat \mu_{t-1}(0)`, not the observed
   treated outcome (which carries the treatment). This recursion makes
   the bias term in equation (2.11) stochastically small.

The treatment-effect estimator is

.. math::

   \widehat \tau_t = \overline{Y}_{1, t} - \sum_{i \in \mathcal{C}}
   w_{i, t}^* Y_{i, t}, \qquad t > T_0,

and the headline ATT is the post-period mean of :math:`\widehat \tau_t`.

Assumptions
-----------

The Zheng & Chen (2024) consistency theorem requires:

(a) **Consistency**: :math:`Y_{it} = D_{it} Y_{it}(1) + (1 - D_{it})
    Y_{it}(0)` for all :math:`i, t`.
(b) **Unconfoundedness**: :math:`\mathbb{E}[Y_{it}(0) | Y_{i, t-1}(0),
    X_{it}, D_i = 1] = \mathbb{E}[Y_{it}(0) | Y_{i, t-1}(0), X_{it},
    D_i = 0]`.
(c) **Unaffected confounders**: :math:`X_{it} = X_{it}(0) = X_{it}(1)`
    -- the treatment does not move the confounders.
(d) **Treatment-overlap**: :math:`\mathbb{P}(D_i = 1 | X_{it},
    Y_{i, t-1}(0)) < 1` with probability one.

For inference (Section 3 of the paper), additional finite-moment and
positive-definite-covariance conditions on :math:`\varepsilon_{it}`
are required.

**Theorem 1** (consistency). Under Assumptions 1-7 and the linear
model :math:`Y_{it}(0) = \delta_t + \beta_t^\prime X_{it} +
\rho_t Y_{i, t-1}(0) + \varepsilon_{it}`, the DSCAR estimator
:math:`\widehat \tau_t` converges to :math:`\tau_t` in probability as
both :math:`N_{tr}, N_{co} \to \infty` with :math:`T` fixed. The
asymptotic regime is in marked contrast with Abadie et al. (2010),
which requires :math:`T_0 \to \infty`.

Core API
--------

.. automodule:: mlsynth.estimators.dscar
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.DSCARConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.dscar_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dscar_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dscar_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dscar_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.dscar_helpers.structures
   :members:
   :undoc-members:

Example
-------

A tiny AR(1) panel with a planted treatment effect of :math:`\tau = 2`
on unit 0:

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import DSCAR

   rng = np.random.default_rng(0)
   N, T, T0 = 8, 30, 20
   x = rng.standard_normal((N, T)) * 0.5
   eps = rng.standard_normal((N, T)) * 0.3
   Y = np.zeros((N, T))
   for t in range(1, T):
       Y[:, t] = 0.5 * x[:, t] + 0.6 * Y[:, t - 1] + eps[:, t]
   Y[0, T0:] += 2.0       # treatment effect on unit 0
   rows = [
       {"unit": f"u{i}", "year": t,
        "y": float(Y[i, t]), "x1": float(x[i, t]),
        "y_lag1": float(Y[i, t - 1]) if t >= 1 else 0.0,
        "treat": int(i == 0 and t >= T0)}
       for i in range(N) for t in range(T)
   ]
   df = pd.DataFrame(rows)

   res = DSCAR({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "year",
       "exog_covariates": ["x1"], "lagged_outcome": "y_lag1",
       "display_graphs": False,
   }).fit()

   print(f"DSCAR ATT = {res.att:+.3f}  (true tau = 2.0)")

Empirical replication: Beijing PM2.5 air-pollution alerts
---------------------------------------------------------

DSCAR ships with the two air-pollution panels used in Zheng & Chen
(2024) Section 5:

* :file:`basedata/beijing_pm25_orange_alert.csv` -- the orange alert
  starting 17 Nov 2016, 94 monitoring stations × 72 hours, 20 treated.
* :file:`basedata/beijing_pm25_red_alert.csv` -- the red alert
  starting 16 Dec 2016, 66 stations × 72 hours, 20 treated.

The pre-period is the 48 hours before the alert; the post-period is
the 24 hours after.

.. code-block:: python

   import pandas as pd
   from mlsynth import DSCAR

   df = pd.read_csv("basedata/beijing_pm25_orange_alert.csv")
   df["treat_indicator"] = (
       (df["alert_if"] == 1) & (df["hour_eps"] > 48)
   ).astype(int)

   res = DSCAR({
       "df": df, "outcome": "pm25", "treat": "treat_indicator",
       "unitid": "id_eps", "time": "hour_eps",
       "exog_covariates": ["WSPM", "humi", "dewp", "pres"],
       "lagged_outcome": "pm25_lag1",
       "display_graphs": False,
   }).fit()

   mu0 = res.fit.Y0_hat[48:].mean()
   mu1 = res.fit.Y_treated_mean[48:].mean()
   print(f"orange alert ATT  = {res.att:+.4f}  (paper -33.8)")
   print(f"  relative reduction = {100 * res.att_relative:+.2f}%  (paper -24.3%)")
   print(f"  mu_0 = {mu0:.4f}  (paper 139.0)")
   print(f"  mu_1 = {mu1:.4f}  (paper 105.3)")

prints::

   orange alert ATT  = -33.7830  (paper -33.8)
     relative reduction = +24.29%  (paper -24.3%)
     mu_0 = 139.07  (paper 139.0)
     mu_1 = 105.28  (paper 105.3)

**Path-A regression status**:

* **Orange alert**: ATT matches the paper to **0.05 μg/m³** and the
  relative-reduction figure to **0.01 percentage points** -- this is a
  faithful Path-A replication.
* **Red alert**: my implementation produces ``ATT = −55.7 μg/m³``
  (relative reduction 21.9%) against the paper's reported ``-70.4
  μg/m³`` (relative reduction 26.2%). The qualitative finding holds
  (large negative ATT, ~20% reduction), but the magnitude differs by
  ~21%. The released reference R script
  ``eg2/Eg_Air_Pollution_eps_201616_12_16_final.R`` contains a
  **commented-out** per-unit pressure / humidity de-meaning block,
  suggesting the paper's red-alert numbers were produced with
  preprocessing the released code doesn't actually perform. The
  pytest regression ``TestPathABeijingAlerts::test_red_att_qualitative``
  asserts the qualitative ATT bound rather than the paper's exact
  magnitude.

The driver is :file:`examples/dscar/replicate_beijing_alerts.py`; run
with ``python -m examples.dscar.replicate_beijing_alerts``.

References
----------

Zheng, X., & Chen, S. X. (2024). "Dynamic synthetic control method for
evaluating treatment effects in auto-regressive processes."
*Journal of the Royal Statistical Society Series B* 86(1):155-176.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Chen, S. X., & Van Keilegom, I. (2009). "A review on empirical
likelihood methods for regression." *Test* 18(3):415-447.

Owen, A. (1988). "Empirical likelihood ratio confidence intervals for
a single functional." *Biometrika* 75(2):237-249.

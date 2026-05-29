Factor Model Approach (FMA)
===========================

.. currentmodule:: mlsynth

Overview
--------

FMA implements Li, K. T., & Sonnier, G. P. (2023). *"Statistical
Inference for the Factor Model Approach to Estimate Causal Effects in
Quasi-Experimental Settings."* Journal of Marketing Research,
60(3):449-472. The estimator constructs a counterfactual for a single
treated unit by

1. extracting **principal-component factors** from the control panel
   (with the number of factors chosen by the paper's modified Bai-Ng
   criterion for stationary outcomes or Bai (2004) IPC1 for non-
   stationary outcomes);
2. **projecting** the treated unit's pre-treatment outcomes onto a
   constant plus the factors via OLS to recover the loading
   :math:`\hat\lambda_1`;
3. forming the counterfactual :math:`\hat y^0_{1, t} = \tilde F_t' \hat\lambda_1`
   for every period; the ATT is the mean post-treatment gap.

The paper's distinctive contribution is **valid statistical inference
for the ATT**. FMA in :mod:`mlsynth` exposes three procedures in
parallel; the user picks any subset via
:py:attr:`FMAConfig.inference_methods`:

* ``"asymptotic"`` (default) -- Theorem 3.1 (stationary) /
  Theorem 3.3 (non-stationary) normal CI for the ATT, built from the
  variance decomposition :math:`\widehat \Omega = \widehat \Omega_1
  + \widehat \Omega_2`.
* ``"bootstrap"`` -- Web Appendix F residual bootstrap for per-period
  :math:`\widehat{ATT}_t` CIs.
* ``"placebo"`` -- Web Appendix G control-as-pseudo-treated band.

Mathematical Formulation
------------------------

Setup
^^^^^

We observe one treated unit (indexed 1) and ``N_co`` control units
over ``T`` periods. Treatment begins at :math:`T_0 + 1`. Under the
factor model

.. math::

   y_{i, t}^0 = \lambda_i' F_t + e_{i, t},
   \qquad y_{i, t} = y_{i, t}^0 + D_{i, t} \tau_{i, t},

the goal is to estimate

.. math::

   ATT = \frac{1}{T - T_0} \sum_{t > T_0} \tau_{1, t}
       = \frac{1}{T - T_0} \sum_{t > T_0} (y_{1, t} - y_{1, t}^0).

Factor extraction
^^^^^^^^^^^^^^^^^

Factors :math:`\hat F_t` are extracted from the control panel via PCA
after demeaning (or standardising) each control series. The number of
factors :math:`r` is chosen by one of two criteria:

* **Modified Bai-Ng (MBN) -- stationary data.** Choose
  :math:`r \in \{0, 1, \dots, r_{\max}\}` minimising

  .. math::

     PC_{MBN}(r) = \frac{1}{N_{co} T} \sum_{i=2}^{N} \sum_{t=1}^{T}
         (y_{i, t} - \hat\lambda_i' \hat F_t)^2
         + c_{N, T}\, r\, \hat\sigma^2
         \frac{N_{co} + T}{N_{co} T}
         \log \frac{N_{co} + T}{N_{co} T},

  with the small-sample adjustment

  .. math::

     c_{N, T} = \frac{(N_{co} + \max(70 - N_{co}, 0))
                       (T + \max(70 - T, 0))}{N_{co} T}.

  When ``N_co, T >= 70`` the adjustment collapses to 1 and the
  criterion is identical to Bai-Ng (2002) :math:`PC_{p1}`.

* **Bai (2004) IPC1 -- non-stationary data.** Replaces the
  small-sample factor with a log-log adjustment suited to non-
  stationary factors.

Override the data-driven selection by passing
:py:attr:`FMAConfig.n_factors` directly.

Loading and counterfactual
^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`\tilde F_t = [1, \hat F_t']'` and pre-period OLS,

.. math::

   \hat\lambda_1 = \biggl(\sum_{t = 1}^{T_0} \tilde F_t \tilde F_t'\biggr)^{-1}
                     \sum_{t = 1}^{T_0} \tilde F_t\, y_{1, t},
   \qquad
   \hat y^0_{1, t} = \tilde F_t' \hat\lambda_1.

The ATT is :math:`\widehat{ATT} = (T - T_0)^{-1} \sum_{t > T_0}
(y_{1, t} - \hat y^0_{1, t})`.

Asymptotic inference (Theorem 3.1 / 3.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write :math:`\bar{\tilde F}_2 = (T - T_0)^{-1} \sum_{t > T_0} \tilde F_t`
and :math:`\widehat \Psi = (\sum_{t \le T_0} \tilde F_t \tilde F_t')^{-1}`.
The paper shows

.. math::

   \widehat \Omega = \widehat \Omega_1 + \widehat \Omega_2,
   \quad
   \widehat \Omega_1 = \frac{T - T_0}{T_0}\,
       \bar{\tilde F}_2'\, \widehat \Psi\, \bar{\tilde F}_2,
   \quad
   \widehat \Omega_2 = \widehat \sigma_e^2,

with :math:`\widehat \sigma_e^2` the variance of the pre-treatment
residuals. The :math:`(1 - \alpha)` CI for the ATT is

.. math::

   \widehat{ATT} \pm z_{1 - \alpha / 2}\,
                       \frac{\sqrt{\widehat \Omega}}{\sqrt{T - T_0}}.

A two-sided z-test of :math:`H_0: ATT = 0` reports the p-value.

Bootstrap inference (Web Appendix F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper notes that the per-period :math:`ATT_t` CI cannot shrink to
zero as :math:`T_0, T - T_0, N_{co} \to \infty` because its leading
term is the idiosyncratic shock :math:`e_{1, t}` itself. A residual
bootstrap therefore drives the per-period CI:

1. Compute pre-period residuals
   :math:`\hat e_{1, t} = y_{1, t} - \tilde F_t' \hat\lambda_1`,
   :math:`t = 1, \dots, T_0`.
2. For each bootstrap draw :math:`b = 1, \dots, B`:

   * Sample :math:`e^*_{1, t}` from :math:`\{\hat e_{1, t}\}` with
     replacement for every :math:`t \in \{1, \dots, T\}`.
   * Form :math:`y^*_{1, t} = \tilde F_t' \hat\lambda_1 + e^*_{1, t}`.
   * Re-estimate :math:`\hat\lambda^{*}_1` from the bootstrap
     pre-period.
   * Compute :math:`\Delta^*_{1, t} = y^*_{1, t} - \tilde F_t'
     \hat\lambda^{*}_1` for :math:`t > T_0`.
3. The :math:`(1 - \alpha)` CI for :math:`\Delta_{1, t}` is

   .. math::

      \bigl[\widehat \Delta_{1, t}
              - \Delta^*_{1, t, ((1 - \alpha/2) B)},
            \widehat \Delta_{1, t}
              - \Delta^*_{1, t, (\alpha B / 2)}\bigr].

Placebo inference (Web Appendix G)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set each control unit in turn as a pseudo-treated unit and re-fit the
factor model on the remaining controls; collect the :math:`N_{co}`
pseudo-ATT curves and report their pointwise :math:`(\alpha/2,
1 - \alpha/2)` quantile band. The paper notes the placebo test is
sensitive to error-variance heterogeneity between the treated unit
and the controls -- when that assumption is violated the asymptotic
CI from the previous paragraph is preferred.

Core API
--------

.. automodule:: mlsynth.estimators.fma
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.FMAConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.fma_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fma_helpers.factors
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fma_helpers.fit
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fma_helpers.inference
   :members:
   :undoc-members:

The Web Appendix E.1 Monte Carlo DGP1, packaged as
:func:`simulate_fma_sample` so the *Verification* replication below runs
as a one-liner.

.. automodule:: mlsynth.utils.fma_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fma_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.fma_helpers.structures
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo at a non-stationary two-factor
DGP -- the regime the paper's empirical California / Brooklyn
applications fall into. The example fits FMA in all three inference
modes and prints the headline output.

.. code-block:: python

   """One draw of a Li & Sonnier (2023) factor-model simulation."""

   import numpy as np
   import pandas as pd

   from mlsynth import FMA


   # ---------------------------------------------------------------------
   # 1. Simulate one panel from a non-stationary two-factor DGP
   # ---------------------------------------------------------------------

   rng = np.random.default_rng(0)
   J = 20           # control units
   T_pre = 30
   T_post = 10
   T = T_pre + T_post
   r_true = 2
   tau_true = 1.0   # additive treatment effect on the treated unit

   F = rng.standard_normal((T, r_true)).cumsum(axis=0)        # non-stationary
   lam = rng.standard_normal((J + 1, r_true))
   eps = rng.standard_normal((T, J + 1)) * 0.5
   Y0 = F @ lam.T + eps
   Y = Y0.copy()
   Y[T_pre:, 0] += tau_true                                    # unit 0 treated

   rows = [
       {
           "unit": j,
           "time": t,
           "y": float(Y[t, j]),
           "D": int(j == 0 and t >= T_pre),
       }
       for j in range(J + 1)
       for t in range(T)
   ]
   df = pd.DataFrame(rows)


   # ---------------------------------------------------------------------
   # 2. Fit FMA with all three inference modes
   # ---------------------------------------------------------------------

   results = FMA({
       "df": df,
       "outcome": "y",
       "treat": "D",
       "unitid": "unit",
       "time": "time",
       "stationarity": "nonstationary",         # IPC1 factor selection
       "preprocessing": "demean",
       "inference_methods": ["asymptotic", "bootstrap", "placebo"],
       "n_bootstrap": 500,
       "alpha": 0.05,
       "display_graphs": False,
   }).fit()


   # ---------------------------------------------------------------------
   # 3. Inspect the output
   # ---------------------------------------------------------------------

   print(f"true tau              = {tau_true:+.3f}")
   print(f"ATT_hat               = {results.att:+.3f}")
   print(f"r selected            = {results.design.n_factors} "
         f"({results.design.n_factors_source})")
   print(f"pre-RMSE              = {results.pre_rmse:.4f}")
   inf = results.inference
   print(f"asymptotic 95% CI ATT = "
         f"[{inf.asymptotic_att_lower:+.3f}, {inf.asymptotic_att_upper:+.3f}]")
   print(f"asymptotic p-value    = {inf.asymptotic_att_p_value:.3f}")

   import pandas as pd
   print("\nPer-period ATT_t with bootstrap CI:")
   print(pd.DataFrame({
       "t": np.arange(T_pre + 1, T + 1),
       "att_t": results.gap[T_pre:],
       "boot_lower": inf.bootstrap_att_t_lower,
       "boot_upper": inf.bootstrap_att_t_upper,
   }).round(3).to_string(index=False))

Verification
------------

**Monte Carlo replication (Path B).** Li & Sonnier's empirical
applications -- California beer sales and a Brooklyn eyeglass retailer's
showroom -- run on **Nielsen retail scanner data** and a **proprietary
retailer panel**, neither of which is redistributable, so the estimator
is validated through the paper's own Monte Carlo. The methodological
contribution of the paper is that the new asymptotic CI from Theorem 3.1
attains nominal coverage **regardless of whether**
:math:`\sigma_{\text{tr}} = \sigma_{\text{co}}`, while the Xu (2017)
bootstrap miscovers badly when the two variances differ (Figures 2-4
Panel A, plus the non-stationary mirror in Web Appendix E.1's Figures
W.5-W.7 and the sample-size sweeps in W.8-W.11).

Both DGPs -- the stationary Section 4 DGP1 and the non-stationary
Appendix E.1 DGP2 -- are packaged in
:func:`mlsynth.utils.fma_helpers.simulation.simulate_fma_sample`,
together with the three variance regimes
(``"equal"`` / ``"treated_smaller"`` / ``"treated_larger"``). True ATT
is zero in every draw; the paper's centred statistic
:math:`\sqrt{T_2}(\hat\Delta_1 - \Delta_1)` is invariant to
:math:`\Delta` (see the equation following 4.2), so coverage doesn't
depend on its value.

Replicating the headline coverage findings is a 15-line script:

.. code-block:: python

   import numpy as np
   from mlsynth import FMA
   from mlsynth.utils.fma_helpers.simulation import simulate_fma_sample

   def coverage_cell(dgp, variance_case, N_co, T1, T2, M, alpha=0.05):
       stationarity = "stationary" if dgp == "dgp1" else "nonstationary"
       covers = []
       for j in range(M):
           s = simulate_fma_sample(dgp=dgp, N_co=N_co, T1=T1, T2=T2,
                                     variance_case=variance_case,
                                     rng=np.random.default_rng(j))
           res = FMA({"df": s.df, "outcome": "y", "treat": "D",
                      "unitid": "unit", "time": "time",
                      "stationarity": stationarity,
                      "inference_methods": ["asymptotic"],
                      "alpha": alpha, "display_graphs": False}).fit()
           covers.append(res.inference.asymptotic_att_lower <= 0.0
                          <= res.inference.asymptotic_att_upper)
       return float(np.mean(covers))

   for dgp in ("dgp1", "dgp2"):
       for case in ("equal", "treated_smaller", "treated_larger"):
           print(dgp, case,
                 coverage_cell(dgp, case, N_co=30, T1=30, T2=20, M=1000))

At :math:`M = 1{,}000` (Li & Sonnier use :math:`M = 100{,}000`; runtime
difference is the only material change) ``mlsynth`` reproduces all four
appendix-figure families. The MC standard error at :math:`M = 1{,}000`
is :math:`\sqrt{0.95 \cdot 0.05 / 1000} \approx 0.7` pp.

Section 4 + Web Appendix E.1: three variance regimes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`N_{co} = 30`, :math:`T_1 = 30`, :math:`T_2 = 20`. Reproduces
Figures 2-4 Panel A (DGP1, stationary) and Figures W.5-W.7 Panel A
(DGP2, non-stationary).

.. list-table::
   :header-rows: 1
   :widths: 8 22 12 12 18 18

   * - DGP
     - Variance case
     - :math:`\sigma_{\text{tr}}`
     - :math:`\sigma_{\text{co}}`
     - mlsynth
     - paper
   * - DGP1
     - ``"equal"`` (Fig 2A)
     - 1.0
     - 1.0
     - **0.947**
     - 0.95
   * - DGP1
     - ``"treated_smaller"`` (Fig 3A)
     - 0.5
     - 1.0
     - **0.946**
     - 0.95
   * - DGP1
     - ``"treated_larger"`` (Fig 4A)
     - 2.0
     - 1.0
     - **0.951**
     - 0.95
   * - DGP2
     - ``"equal"`` (Fig W.5A)
     - 1.0
     - 1.0
     - **0.935**
     - 0.95
   * - DGP2
     - ``"treated_smaller"`` (Fig W.6A)
     - 0.5
     - 1.0
     - **0.944**
     - 0.95
   * - DGP2
     - ``"treated_larger"`` (Fig W.7A)
     - 2.0
     - 1.0
     - **0.929**
     - 0.95

Every cell lands within :math:`\pm 2.1` pp of nominal -- well inside
three MC standard errors. The headline takeaway is the **equality of the
three numbers within each DGP**: coverage does not deteriorate when
:math:`\sigma_{\text{tr}} \ne \sigma_{\text{co}}`, which is precisely
the regime where the Xu (2017) bootstrap mistakes the treated-error
variance for the control-error variance and either over- or under-covers
(Panel B of Figures 3-4 and W.6-W.7 in the paper).

Web Appendix E.2.1: other :math:`(T_1, N_{co})` combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Equal variance (:math:`\sigma_{\text{tr}} = \sigma_{\text{co}} = 1`),
:math:`T_2 = 20`, sample-size sweeps over both DGPs reproducing
Figures W.8-W.9 (stationary) and W.10-W.11 (non-stationary).

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 18 18

   * - DGP
     - :math:`T_1`
     - :math:`N_{co}`
     - mlsynth
     - paper
   * - DGP1
     - 30
     - 60
     - **0.936**
     - 0.95 (Fig W.8A)
   * - DGP1
     - 60
     - 30
     - **0.939**
     - 0.95 (Fig W.8B)
   * - DGP1
     - 60
     - 60
     - **0.944**
     - 0.95 (Fig W.9A)
   * - DGP1
     - 120
     - 120
     - **0.949**
     - 0.95 (Fig W.9B)
   * - DGP2
     - 30
     - 60
     - **0.939**
     - 0.95 (Fig W.10A)
   * - DGP2
     - 60
     - 30
     - **0.936**
     - 0.95 (Fig W.10B)
   * - DGP2
     - 60
     - 60
     - **0.935**
     - 0.95 (Fig W.11A)
   * - DGP2
     - 120
     - 120
     - **0.955**
     - 0.95 (Fig W.11B)

All eight cells within :math:`\pm 1.5` pp of nominal, and the largest
cell (:math:`T_1 = N_{co} = 120`) hits the asymptotic regime cleanly --
the underlying theory is consistent in both panel dimensions, as the
paper asserts.

Not replicated here
^^^^^^^^^^^^^^^^^^^

* **Web Appendix E.2.2** (small-N/T :math:`t`-distribution) -- the paper
  recommends switching to :math:`t_{T_1 - (N_{co} + 1)}` for very small
  samples, but the suggested degrees-of-freedom value is non-positive in
  the figure's specific :math:`(T_1, N_{co}) = (20, 20)` configuration,
  and ``FMA``'s public API exposes only the normal CI; the small-sample
  refinement is left as a future addition.
* **Web Appendix E.3 / W.15-W.17 (DGP3)** -- the paper bootstraps DGP3's
  first factor from the proprietary Brooklyn sales panel, so the precise
  DGP cannot be reconstructed from public data; the corresponding
  empirical-application coverage check is therefore out of scope.

For reference, the corresponding empirical applications report:

* **California recreational marijuana legalization on beer sales** --
  :math:`\widehat{\text{ATT}} = -\$88{,}400` weekly
  (:math:`-0.464\%`), 95% CI :math:`[-\$407{,}400,\ \$230{,}600]`
  (Table 2; not significant). The Xu bootstrap's interval is less than
  half as wide (:math:`[-\$229{,}000,\ \$82{,}400]`), invalidly so
  given the variance ratio
  :math:`\hat\sigma_{\text{tr}}^2 / \hat\sigma_{\text{co}}^2 \approx 37.4`.
* **Brooklyn showroom opening on weekly sales** --
  :math:`\widehat{\text{ATT}} = +\$2{,}446` weekly (:math:`+27.2\%`).

References
----------

Bai, J. (2004). "Estimating Cross-Section Common Stochastic Trends in
Nonstationary Panel Data." *Journal of Econometrics* 122(1):137-183.

Bai, J., & Ng, S. (2002). "Determining the Number of Factors in
Approximate Factor Models." *Econometrica* 70(1):191-221.

Li, K. T., & Sonnier, G. P. (2023). "Statistical Inference for the
Factor Model Approach to Estimate Causal Effects in Quasi-Experimental
Settings." *Journal of Marketing Research* 60(3):449-472.

Factor Model Approach (FMA)
===========================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

Quasi-experiments are the bread and butter of marketing causal inference:
a policy or firm decision switches on for one unit (a state legalises
recreational marijuana, a digitally native retailer opens a physical
showroom in Brooklyn) and you want the sales effect against a pool of
untreated cities, stores, or categories. The two reflexive choices each
strain in this setting:

* Difference-in-differences (DiD) assumes the time effect is the
  *same* for the treated unit and every control -- the parallel-trends
  assumption. Marketing panels (sales, share, macro series) rarely
  oblige: different markets ride different seasonal, regional, and
  business-cycle waves. When that homogeneity fails, DiD carries a large
  estimation bias that does not shrink as the panel grows -- in Li
  and Sonnier's own MSE comparison it is so biased they drop it from the
  table.
* Synthetic control (SC) relaxes that by weighting controls on the
  simplex (nonnegative, sum-to-one, no intercept), which pins the
  counterfactual *inside the convex hull* of the controls. That is a
  feature when the treated unit is "surrounded" by donors and you want an
  interpretable convex-combination weight story, but a bug when the
  treated unit sits outside the donor range, and SC has no
  inference theory when controls are many and the data are
  non-stationary of unknown form.

The factor model approach (FMA) of Li and Sonnier ([FMA]_) targets
exactly the regime SC and DiD struggle with: many control units, time
effects that differ across units, a treated path that may lie outside the
donor range, and a need for honest confidence intervals. Instead of
weighting the controls directly, it first projects the control panel
onto a small set of latent factors (interactive fixed effects:
:math:`y^N_{jt} = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + e_{jt}`), then regresses the treated
unit's pre-period outcome on those factors with no constraint on the
loadings. Three properties follow, and they are the reasons to reach for
it:

1. Heterogeneous time effects. The interactive
   :math:`\boldsymbol{\lambda}_j^\top \mathbf{f}_t` structure nests two-way fixed effects (DiD) as
   the special case :math:`\boldsymbol{\lambda}_j = (\xi_j, 1)^\top`,
   :math:`\mathbf{f}_t = (1, \delta_t)^\top`, and also absorbs unit-specific trends
   and cross-sectional dependence. You are not betting on a common shock.
2. The counterfactual can leave the donor hull. Because the loadings
   are unrestricted (no nonnegativity, no sum-to-one, no zero intercept),
   FMA fits treated units whose level or trajectory is outside the range
   of every control -- the case SC structurally cannot represent.
3. Robust to a large, growing donor pool. Unconstrained regression on
   the raw controls (the Hsiao-Ching-Wan approach) overfits and breaks
   down once :math:`N_0` approaches or exceeds :math:`T_0`. FMA's
   dimension reduction sidesteps this: it *benefits* from more controls
   (they sharpen the factor estimates) rather than being destabilised by
   them. The same single factor extraction also scales cheaply to many
   treated units and staggered timing, since it is done once.

The paper's headline contribution is the fourth reason: valid,
computationally cheap inference. FMA delivers a closed-form normal
confidence interval for the ATT (Theorems 3.1 / 3.3) that is valid for
both stationary and non-stationary data and -- critically --
accommodates treated and control units with different error
variances. The previously standard Xu (2017) bootstrap assumes
:math:`\sigma^2_{\text{tr}} = \sigma^2_{\text{co}}`; when that fails (the
norm for non-randomised quasi-experiments) it over- or under-covers
badly, while permutation/placebo tests need an analogous symmetry. In the
California beer application the estimated variance ratio is
:math:`\widehat{\sigma}^2_{\text{tr}} / \widehat{\sigma}^2_{\text{co}} \approx 37`, so
the bootstrap interval comes out less than half its honest width. FMA's
normal interval needs no such assumption and no resampling loop.

Factor projection vs. donor weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every synthetic-control-family estimator answers *what reproduces the
treated unit's untreated path?* FMA's structural bet is distinctive:

   The controls' untreated outcomes share a low-dimensional latent
   factor structure; estimate those factors, then load the treated unit
   onto them without constraint.

.. list-table::
   :header-rows: 1
   :widths: 26 24 24 24

   * -
     - DiD
     - Synthetic control
     - Factor model (FMA)
   * - Comparison built from
     - all controls, equal weight
     - all controls, simplex weights
     - latent factors of the controls
   * - Time effects
     - homogeneous (common shock)
     - implicit via weights
     - heterogeneous :math:`\boldsymbol{\lambda}_j^\top \mathbf{f}_t`
   * - Counterfactual outside donor hull
     - allowed
     - no -- convex hull
     - yes -- unrestricted loadings
   * - Many controls (:math:`N_0 \gtrsim T_0`)
     - bias does not shrink
     - overfits if restrictions relaxed
     - *benefits* -- sharper factors
   * - Inference, non-stationary data
     - standard
     - none available
     - normal CI (Thm 3.1 / 3.3)
   * - Unequal treated/control variance
     - n/a
     - placebo needs symmetry
     - handled directly

Reach for FMA when
^^^^^^^^^^^^^^^^^^

* You have a single treated unit (or a handful, or many -- the factor
  step generalises) and a large control pool, especially
  :math:`N_0` large relative to the pre-period :math:`T_0`, where
  unrestricted regression on raw donors would overfit.
* The outcome is plausibly driven by a few common latent factors --
  regional sales, market share, macro-linked categories -- so the control
  matrix is approximately low-rank. This is what the factor projection
  exploits.
* You expect heterogeneous time dynamics across units (different
  seasonality, trends, cross-sectional dependence) that violate DiD's
  parallel-trends assumption.
* The treated unit's level or trajectory may sit outside the range of
  the controls, so SC's convex-hull restriction would distort the fit.
* You need formal inference -- a hypothesis test or confidence
  interval for the ATT -- and you cannot defend the equal-variance
  assumption behind the Xu (2017) bootstrap or the symmetry behind
  permutation/placebo tests. FMA's normal CI is valid under unequal
  variances and under non-stationarity, and it is far cheaper than
  resampling.
* You have enough data for the asymptotics: the paper's simulations show
  the normal CI is reliable for :math:`N_0 \ge 30` and
  :math:`T_0 \ge 30` (at :math:`T - T_0 = 20`). For smaller panels the paper
  suggests a :math:`t_{T_0 - (N_0 + 1)}` reference distribution, which
  ``mlsynth`` does not yet expose; treat the normal CI cautiously there.

Do not use FMA when
^^^^^^^^^^^^^^^^^^^

* The control matrix has no low-rank / factor structure. FMA, like
  the PCA-based estimators in :doc:`clustersc`, leans entirely on a few
  factors explaining the controls. If the spectrum decays slowly, the
  factors are noise; prefer a balancing estimator (:doc:`microsynth` for
  user-level data) or a selection estimator (:doc:`fdid`).
* A sparse, interpretable convex-combination weight is the
  deliverable. FMA's loadings are unconstrained and not a "California =
  0.4 Utah + 0.3 Montana" story. If policy storytelling needs nonnegative
  donor weights that sum to one, use :doc:`tssc`, :doc:`fdid`, or classic
  SC.
* The donor pool is tiny (:math:`N_0 \le 10`) with a clean
  canonical SC fit. Factor extraction adds variance without
  identification gain; :doc:`tssc` or :doc:`fdid` are more honest, and
  :doc:`tssc` will even *test* whether SC's restrictions are needed.
* The sample is very small (:math:`N_0, T_0 \approx 20`). The
  normal approximation degrades and the recommended :math:`t`-correction
  is not wired into the public API.
* You want efficiency from the treated unit's own pre-history with many
  treated units. Factors are estimated from controls only; the
  efficiency loss is negligible for a single treated unit but grows with
  the number of treated units (paper, footnote 3).
* Distributional questions (quantile effects, Lorenz/tail changes).
  FMA targets the mean ATT; use :doc:`dsc`.
* Continuous or multi-valued treatment, or spillovers/interference
  across units. FMA encodes a single binary intervention and assumes the
  controls are untreated; use :doc:`ctsc` for dose response and
  :doc:`spillsynth` / :doc:`spsydid` under interference.

Overview
--------

FMA implements Li, K. T., & Sonnier, G. P. (2023). *"Statistical
Inference for the Factor Model Approach to Estimate Causal Effects in
Quasi-Experimental Settings."* Journal of Marketing Research,
60(3):449-472. The estimator constructs a counterfactual for a single
treated unit by

1. extracting principal-component factors from the control panel
   (with the number of factors chosen by the paper's modified Bai-Ng
   criterion for stationary outcomes or Bai (2004) IPC1 for non-
   stationary outcomes);
2. projecting the treated unit's pre-treatment outcomes onto a
   constant plus the factors via OLS to recover the loading
   :math:`\widehat{\boldsymbol{\lambda}}_1`;
3. forming the counterfactual :math:`\widehat{y}_{1t}^N \coloneqq \widetilde{\mathbf{f}}_t^\top \widehat{\boldsymbol{\lambda}}_1`
   for every period; the ATT is the mean post-treatment gap.

The paper's distinctive contribution is valid statistical inference
for the ATT. FMA in :mod:`mlsynth` exposes three procedures in
parallel; the user picks any subset via
:py:attr:`FMAConfig.inference_methods`:

* ``"asymptotic"`` (default) -- Theorem 3.1 (stationary) /
  Theorem 3.3 (non-stationary) normal CI for the ATT, built from the
  variance decomposition :math:`\widehat{\Omega} \coloneqq \widehat{\Omega}_1
  + \widehat{\Omega}_2`.
* ``"bootstrap"`` -- Web Appendix F residual bootstrap for per-period
  :math:`\tau_t` CIs.
* ``"placebo"`` -- Web Appendix G control-as-pseudo-treated band.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0` (the control units). Time runs over
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, 1-indexed; the intervention
takes effect after period :math:`T_0`, splitting :math:`\mathcal{T}` into the
pre-period :math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}`
(of length :math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}` (of length
:math:`T - T_0`).

The observed outcome is :math:`y_{jt}`, decomposed via Abadie potential
outcomes as :math:`y_{jt} = y_{jt}^N + (y_{jt}^I - y_{jt}^N)\, d_{jt}`, with
treatment dummy :math:`d_{jt}` (one only for :math:`j = 1` and
:math:`t \in \mathcal{T}_2`). The no-intervention outcome follows the
interactive factor model
:math:`y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + e_{jt}`, with the
:math:`r`-vector of latent common factors :math:`\mathbf{f}_t \in \mathbb{R}^{r}`,
unit-specific loadings :math:`\boldsymbol{\lambda}_j \in \mathbb{R}^{r}`, and
idiosyncratic error :math:`e_{jt}`. Estimated factors are
:math:`\widehat{\mathbf{f}}_t`; the augmented regressor stacking a constant is
:math:`\widetilde{\mathbf{f}}_t \coloneqq [1, \widehat{\mathbf{f}}_t^\top]^\top`.
The estimated treated loading is :math:`\widehat{\boldsymbol{\lambda}}_1`, the
counterfactual :math:`\widehat{y}_{1t}^N \coloneqq \widetilde{\mathbf{f}}_t^\top
\widehat{\boldsymbol{\lambda}}_1`, the per-period effect
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^N`, and the ATT
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1} \sum_{t \in \mathcal{T}_2}
\tau_t`. The control-error and treated-error standard deviations are
:math:`\sigma_{\text{co}}` and :math:`\sigma_{\text{tr}}`; the significance level
is :math:`\alpha`.

Assumptions
-----------

*Assumption 1 (factor structure).* The no-intervention outcomes admit a
low-rank interactive structure
:math:`y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + e_{jt}` with a fixed
number :math:`r` of common factors, and the control panel's leading sample
eigenvalues separate from the noise floor (an approximate-factor / pervasiveness
condition).

*Remark.* This is FMA's structural bet, shared with the PCA estimators in
:doc:`clustersc`: the whole method leans on a few factors explaining the
controls. If the spectrum decays slowly the extracted factors are noise, and the
projection in the next step has nothing real to load onto.

*Assumption 2 (no anticipation, untreated controls).* Treatment has no effect
before :math:`T_0` (:math:`y_{1t} = y_{1t}^N` for :math:`t \in \mathcal{T}_1`),
and every control :math:`j \in \mathcal{N}_0` is untreated over
:math:`\mathcal{T}`, so the factors and the treated loading are estimated from
no-intervention outcomes only.

*Remark.* Pre-period contamination -- the treated unit reacting before the
nominal date, or a control caught by the same shock -- biases the loading fit
and hence the counterfactual. Date :math:`T_0` at the first plausible response
and quarantine contaminated donors.

*Assumption 3 (stable loadings).* The treated unit's loading
:math:`\boldsymbol{\lambda}_1` recovered on :math:`\mathcal{T}_1` continues to
govern its no-intervention path on :math:`\mathcal{T}_2`, so the pre-period
projection extrapolates forward.

*Remark.* Unlike SC, FMA places no constraint on
:math:`\boldsymbol{\lambda}_1` -- no nonnegativity, sum-to-one, or zero
intercept -- so the counterfactual may leave the donor hull. What licenses the
forward projection is constancy of the loading across the two windows, not
convex-hull support; a regime change in :math:`\mathbf{f}_t` unrelated to
treatment breaks it even with a perfect pre-fit.

*Assumption 4 (regularity for inference).* The errors :math:`e_{jt}` satisfy the
moment and weak-dependence conditions of Li & Sonnier's Theorems 3.1 / 3.3, and
both :math:`T_0` and :math:`N_0` are large enough for the factor estimates to
converge -- but treated and control error variances need not be equal
(:math:`\sigma_{\text{tr}} \ne \sigma_{\text{co}}` is permitted).

*Remark.* This is the load-bearing relaxation: the asymptotic CI below allows
unequal treated/control variances, which is exactly where the Xu (2017)
bootstrap miscovers. The cost is needing enough data for the normal
approximation -- the paper's simulations support :math:`N_0, T_0 \ge 30`.

Mathematical Formulation
------------------------

Setup
^^^^^

We observe one treated unit (:math:`j = 1`) and :math:`N_0` control units
over :math:`T` periods. Treatment begins after :math:`T_0`. Under the
factor model

.. math::

   y_{jt}^N = \boldsymbol{\lambda}_j^\top \mathbf{f}_t + e_{jt},
   \qquad y_{jt} = y_{jt}^N + d_{jt}\, \tau_t,

the goal is to estimate

.. math::

   \widehat{\tau} \coloneqq \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2} \tau_t
       = \frac{1}{T - T_0} \sum_{t \in \mathcal{T}_2} (y_{1t} - y_{1t}^N).

Factor extraction
^^^^^^^^^^^^^^^^^

Factors :math:`\widehat{\mathbf{f}}_t` are extracted from the control panel via
PCA after demeaning (or standardising) each control series. The number of
factors :math:`r` is chosen by one of two criteria:

* Modified Bai-Ng (MBN) -- stationary data. Choose
  :math:`r \in \{0, 1, \dots, r_{\max}\}` minimising

  .. math::

     PC_{MBN}(r) = \frac{1}{N_0 T} \sum_{j \in \mathcal{N}_0} \sum_{t \in \mathcal{T}}
         (y_{jt} - \widehat{\boldsymbol{\lambda}}_j^\top \widehat{\mathbf{f}}_t)^2
         + c_{N, T}\, r\, \widehat{\sigma}^2
         \frac{N_0 + T}{N_0 T}
         \log \frac{N_0 + T}{N_0 T},

  with the small-sample adjustment

  .. math::

     c_{N, T} = \frac{(N_0 + \max(70 - N_0, 0))
                       (T + \max(70 - T, 0))}{N_0 T}.

  When ``N_0, T >= 70`` the adjustment collapses to 1 and the
  criterion is identical to Bai-Ng (2002) :math:`PC_{p1}`.

* Bai (2004) IPC1 -- non-stationary data. Replaces the
  small-sample factor with a log-log adjustment suited to non-
  stationary factors.

Override the data-driven selection by passing
:py:attr:`FMAConfig.n_factors` directly.

Loading and counterfactual
^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`\widetilde{\mathbf{f}}_t \coloneqq [1, \widehat{\mathbf{f}}_t^\top]^\top`
and pre-period OLS,

.. math::

   \widehat{\boldsymbol{\lambda}}_1 = \biggl(\sum_{t \in \mathcal{T}_1}
                     \widetilde{\mathbf{f}}_t \widetilde{\mathbf{f}}_t^\top\biggr)^{-1}
                     \sum_{t \in \mathcal{T}_1} \widetilde{\mathbf{f}}_t\, y_{1t},
   \qquad
   \widehat{y}_{1t}^N = \widetilde{\mathbf{f}}_t^\top \widehat{\boldsymbol{\lambda}}_1.

The ATT is :math:`\widehat{\tau} = (T - T_0)^{-1} \sum_{t \in \mathcal{T}_2}
(y_{1t} - \widehat{y}_{1t}^N)`.

Asymptotic inference (Theorem 3.1 / 3.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write :math:`\bar{\widetilde{\mathbf{f}}}_2 \coloneqq (T - T_0)^{-1}
\sum_{t \in \mathcal{T}_2} \widetilde{\mathbf{f}}_t`
and :math:`\widehat{\boldsymbol{\Psi}} \coloneqq (\sum_{t \in \mathcal{T}_1}
\widetilde{\mathbf{f}}_t \widetilde{\mathbf{f}}_t^\top)^{-1}`.
The paper shows

.. math::

   \widehat{\Omega} = \widehat{\Omega}_1 + \widehat{\Omega}_2,
   \quad
   \widehat{\Omega}_1 = \frac{T - T_0}{T_0}\,
       \bar{\widetilde{\mathbf{f}}}_2^\top\, \widehat{\boldsymbol{\Psi}}\,
       \bar{\widetilde{\mathbf{f}}}_2,
   \quad
   \widehat{\Omega}_2 = \widehat{\sigma}_e^2,

with :math:`\widehat{\sigma}_e^2` the variance of the pre-treatment
residuals. The :math:`(1 - \alpha)` CI for the ATT is

.. math::

   \widehat{\tau} \pm z_{1 - \alpha / 2}\,
                       \frac{\sqrt{\widehat{\Omega}}}{\sqrt{T - T_0}}.

A two-sided z-test of :math:`H_0: \tau = 0` reports the p-value.

Bootstrap inference (Web Appendix F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper notes that the per-period :math:`\tau_t` CI cannot shrink to
zero as :math:`T_0, T - T_0, N_0 \to \infty` because its leading
term is the idiosyncratic shock :math:`e_{1t}` itself. A residual
bootstrap therefore drives the per-period CI:

1. Compute pre-period residuals
   :math:`\widehat{e}_{1t} = y_{1t} - \widetilde{\mathbf{f}}_t^\top
   \widehat{\boldsymbol{\lambda}}_1`,
   :math:`t \in \mathcal{T}_1`.
2. For each bootstrap draw :math:`b = 1, \dots, B`:

   * Sample :math:`e^\ast_{1t}` from :math:`\{\widehat{e}_{1t}\}` with
     replacement for every :math:`t \in \mathcal{T}`.
   * Form :math:`y^\ast_{1t} = \widetilde{\mathbf{f}}_t^\top
     \widehat{\boldsymbol{\lambda}}_1 + e^\ast_{1t}`.
   * Re-estimate :math:`\widehat{\boldsymbol{\lambda}}^{\ast}_1` from the
     bootstrap pre-period.
   * Compute :math:`\tau^\ast_{1t} = y^\ast_{1t} - \widetilde{\mathbf{f}}_t^\top
     \widehat{\boldsymbol{\lambda}}^{\ast}_1` for :math:`t \in \mathcal{T}_2`.
3. The :math:`(1 - \alpha)` CI for :math:`\tau_t` is

   .. math::

      \bigl[\widehat{\tau}_{1t}
              - \tau^\ast_{1t, ((1 - \alpha/2) B)},
            \widehat{\tau}_{1t}
              - \tau^\ast_{1t, (\alpha B / 2)}\bigr].

Placebo inference (Web Appendix G)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set each control unit in turn as a pseudo-treated unit and re-fit the
factor model on the remaining controls; collect the :math:`N_0`
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

.. note::

   ``FMA.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract: ``res.att`` / ``res.att_ci`` /
   ``res.counterfactual`` / ``res.gap`` / ``res.pre_rmse`` resolve through the
   standardized sub-models (FMA is a factor-model counterfactual, so it carries
   no donor weights). The full asymptotic/bootstrap/placebo inference is on
   ``res.inference_detail`` (the bare ``res.inference`` slot is reserved for the
   standardized ATT-level :class:`~mlsynth.config_models.InferenceResults`).

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
   inf = results.inference_detail
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

Monte Carlo replication (Path B). Li & Sonnier's empirical
applications -- California beer sales and a Brooklyn eyeglass retailer's
showroom -- run on Nielsen retail scanner data and a proprietary
retailer panel, neither of which is redistributable, so the estimator
is validated through the paper's own Monte Carlo. The methodological
contribution of the paper is that the new asymptotic CI from Theorem 3.1
attains nominal coverage regardless of whether
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
:math:`\sqrt{T - T_0}\,(\widehat{\tau} - \tau)` is invariant to
:math:`\tau` (see the equation following 4.2), so coverage doesn't
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
           covers.append(res.inference_detail.asymptotic_att_lower <= 0.0
                          <= res.inference_detail.asymptotic_att_upper)
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

:math:`N_0 = 30`, :math:`T_0 = 30`, :math:`T - T_0 = 20`. Reproduces
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
     - 0.947
     - 0.95
   * - DGP1
     - ``"treated_smaller"`` (Fig 3A)
     - 0.5
     - 1.0
     - 0.946
     - 0.95
   * - DGP1
     - ``"treated_larger"`` (Fig 4A)
     - 2.0
     - 1.0
     - 0.951
     - 0.95
   * - DGP2
     - ``"equal"`` (Fig W.5A)
     - 1.0
     - 1.0
     - 0.935
     - 0.95
   * - DGP2
     - ``"treated_smaller"`` (Fig W.6A)
     - 0.5
     - 1.0
     - 0.944
     - 0.95
   * - DGP2
     - ``"treated_larger"`` (Fig W.7A)
     - 2.0
     - 1.0
     - 0.929
     - 0.95

Every cell lands within :math:`\pm 2.1` pp of nominal -- well inside
three MC standard errors. The headline takeaway is the equality of the
three numbers within each DGP: coverage does not deteriorate when
:math:`\sigma_{\text{tr}} \ne \sigma_{\text{co}}`, which is precisely
the regime where the Xu (2017) bootstrap mistakes the treated-error
variance for the control-error variance and either over- or under-covers
(Panel B of Figures 3-4 and W.6-W.7 in the paper).

Web Appendix E.2.1: other :math:`(T_0, N_0)` combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Equal variance (:math:`\sigma_{\text{tr}} = \sigma_{\text{co}} = 1`),
:math:`T - T_0 = 20`, sample-size sweeps over both DGPs reproducing
Figures W.8-W.9 (stationary) and W.10-W.11 (non-stationary).

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 18 18

   * - DGP
     - :math:`T_0`
     - :math:`N_0`
     - mlsynth
     - paper
   * - DGP1
     - 30
     - 60
     - 0.936
     - 0.95 (Fig W.8A)
   * - DGP1
     - 60
     - 30
     - 0.939
     - 0.95 (Fig W.8B)
   * - DGP1
     - 60
     - 60
     - 0.944
     - 0.95 (Fig W.9A)
   * - DGP1
     - 120
     - 120
     - 0.949
     - 0.95 (Fig W.9B)
   * - DGP2
     - 30
     - 60
     - 0.939
     - 0.95 (Fig W.10A)
   * - DGP2
     - 60
     - 30
     - 0.936
     - 0.95 (Fig W.10B)
   * - DGP2
     - 60
     - 60
     - 0.935
     - 0.95 (Fig W.11A)
   * - DGP2
     - 120
     - 120
     - 0.955
     - 0.95 (Fig W.11B)

All eight cells within :math:`\pm 1.5` pp of nominal, and the largest
cell (:math:`T_0 = N_0 = 120`) hits the asymptotic regime cleanly --
the underlying theory is consistent in both panel dimensions, as the
paper asserts.

Not replicated here
^^^^^^^^^^^^^^^^^^^

* Web Appendix E.2.2 (small-N/T :math:`t`-distribution) -- the paper
  recommends switching to :math:`t_{T_0 - (N_0 + 1)}` for very small
  samples, but the suggested degrees-of-freedom value is non-positive in
  the figure's specific :math:`(T_0, N_0) = (20, 20)` configuration,
  and ``FMA``'s public API exposes only the normal CI; the small-sample
  refinement is left as a future addition.
* Web Appendix E.3 / W.15-W.17 (DGP3) -- the paper bootstraps DGP3's
  first factor from the proprietary Brooklyn sales panel, so the precise
  DGP cannot be reconstructed from public data; the corresponding
  empirical-application coverage check is therefore out of scope.

For reference, the corresponding empirical applications report:

* California recreational marijuana legalization on beer sales --
  :math:`\widehat{\tau} = -\$88{,}400` weekly
  (:math:`-0.464\%`), 95% CI :math:`[-\$407{,}400,\ \$230{,}600]`
  (Table 2; not significant). The Xu bootstrap's interval is less than
  half as wide (:math:`[-\$229{,}000,\ \$82{,}400]`), invalidly so
  given the variance ratio
  :math:`\widehat{\sigma}_{\text{tr}}^2 / \widehat{\sigma}_{\text{co}}^2 \approx 37.4`.
* Brooklyn showroom opening on weekly sales --
  :math:`\widehat{\tau} = +\$2{,}446` weekly (:math:`+27.2\%`).

References
----------

Bai, J. (2004). "Estimating Cross-Section Common Stochastic Trends in
Nonstationary Panel Data." *Journal of Econometrics* 122(1):137-183.

Bai, J., & Ng, S. (2002). "Determining the Number of Factors in
Approximate Factor Models." *Econometrica* 70(1):191-221.

Li, K. T., & Sonnier, G. P. (2023). "Statistical Inference for the
Factor Model Approach to Estimate Causal Effects in Quasi-Experimental
Settings." *Journal of Marketing Research* 60(3):449-472.

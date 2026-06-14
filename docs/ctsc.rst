Continuous-Treatment Synthetic Control (CTSC)
==============================================

.. currentmodule:: mlsynth

Overview
--------

CTSC (Powell, D. (2022). *"Synthetic Control Estimation Beyond
Comparative Case Studies: Does the Minimum Wage Reduce Employment?,"*
Journal of Business & Economic Statistics 40(3):1302-1314) generalises
the synthetic control method to settings the original was never built
for: continuous and/or multi-valued treatments in panels where there
is no clean treated / never-treated split.

The canonical synthetic control estimates the effect of one unit
adopting a single binary policy, using never-treated units as donors.
But many policy variables are continuous and adopted (and changed) by
every unit -- every U.S. state has a minimum wage that moves over time,
so there is no "untreated" state to serve as a clean control. CTSC (the
paper's "GSC") handles exactly this case.

.. note::

   The paper names the estimator GSC (generalized synthetic control).
   mlsynth calls it CTSC to avoid collision with Xu (2017)'s
   differently constructed Generalized Synthetic Control (``gsynth``),
   which is an interactive-fixed-effects estimator.

What CTSC does
^^^^^^^^^^^^^^

* Builds a synthetic control for every unit out of the other units'
  *untreated* outcomes.
* Jointly estimates a unit-specific treatment-slope vector
  :math:`\boldsymbol{\alpha}_i` (allowing fully heterogeneous marginal
  effects) and the synthetic-control weights.
* Reports the population-weighted average marginal effect
  :math:`\alpha^{AE} \coloneqq \sum_{i \in \mathcal{N}} \pi_i\,
  \boldsymbol{\alpha}_i`.
* Permits an interactive-fixed-effects (factor) outcome structure, so it
  is consistent when the treatment is correlated with unobserved factors
  and trends -- precisely the case where two-way fixed-effects regression
  is badly biased.

When to use CTSC
^^^^^^^^^^^^^^^^

* The policy variable is continuous or multi-valued (minimum wage,
  tax rate, dosage) rather than a 0/1 indicator.
* Every unit is "treated" with a time-varying intensity, so there is
  no never-treated donor pool and comparative-case-study SC cannot be
  applied.
* You suspect the treatment is correlated with unobserved
  trends/factors, making two-way fixed effects inconsistent.
* You want unit-specific marginal effects (heterogeneity), not just
  an average.

Notation
^^^^^^^^

CTSC departs from the single-treated-unit setting, so there is no
distinguished :math:`j = 1`: every unit carries its own treatment intensity.
Index the units by :math:`i \in \mathcal{N} \coloneqq \{1, \dots, N\}` and time
by :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}` (1-indexed). The observed
scalar outcome of unit :math:`i` at time :math:`t` is :math:`y_{it}`, with its
no-treatment potential outcome (Abadie superscript) written :math:`y_{it}^N`.

The treatment of unit :math:`i` at time :math:`t` is a :math:`K`-vector
:math:`\mathbf{D}_{it}` of continuous or multi-valued intensities, and
:math:`\boldsymbol{\alpha}_i \in \mathbb{R}^{K}` is its unit-specific
marginal-effect (slope) vector -- the estimand. The factor term
:math:`\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i` carries an interactive
fixed effect, with time factors :math:`\boldsymbol{\lambda}_t` and loadings
:math:`\boldsymbol{\mu}_i`, and :math:`\epsilon_{it}` is mean-zero idiosyncratic
noise. When unit :math:`i` is synthesized from the others, its donor weights
:math:`\boldsymbol{\phi}^i` lie on the simplex
:math:`\Delta^{N-1} \coloneqq \{\boldsymbol{\phi} \in \mathbb{R}_{\ge 0}^{N-1} :
\|\boldsymbol{\phi}\|_1 = 1\}`, and :math:`\Omega_i` is its per-unit fit weight.
The population-weighted average marginal effect is
:math:`\alpha^{AE} \coloneqq \sum_{i \in \mathcal{N}} \pi_i\,
\boldsymbol{\alpha}_i`, with population shares :math:`\pi_i`.

Assumptions (and how to spot violations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CTSC is a *much* more permissive estimator than vanilla SC -- it does
not require a clean treated/control split, it does not assume
homogeneous effects, and it does not require the analyst to specify the
number of factors. But it does still rely on a small set of identifying
assumptions. Each is given here together with the symptom you would see
if it failed.

1. Interactive fixed-effects DGP for the untreated outcome. The paper assumes
   :math:`y_{it}^N = \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i +
   \epsilon_{it}`, i.e. the counterfactual (no-treatment) outcome is a low-rank
   factor model plus mean-zero idiosyncratic noise.

   *Remark.* This is plausibly violated when the outcome has strong
   unit-specific nonlinear trends, structural breaks affecting only some units,
   or unit-specific deterministic time polynomials that cannot be written as
   :math:`\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i`. To spot it, fit an
   interactive-FE model (e.g. ``gsynth``) to the pre-/non-treatment data and
   inspect the residuals -- if there is visible unit-specific curvature left,
   the factor structure is misspecified.

2. Convex-hull condition on donor loadings. For each treated-period observation
   of unit :math:`i`, its factor loading :math:`\boldsymbol{\mu}_i` must lie
   (approximately) in the convex hull of the other units' loadings, so that a
   simplex-weighted combination of donors can reproduce its untreated
   trajectory.

   *Remark.* This is plausibly violated when the treated unit is an outlier on
   level, seasonality, or factor exposure -- a coastal mega-state with no
   interior analog, a country with idiosyncratic policy that no donor shares.
   To spot it, look at the per-unit fit weights :math:`\Omega_i` returned by
   CTSC; units with very poor untreated fit get heavily down-weighted, and if
   most units fall in that bucket the hull condition is failing population-wide.

3. Large :math:`T` regime (consistency, not factor count). Powell shows the
   estimator is consistent as :math:`T \to \infty` *without* the user having to
   specify the rank of :math:`\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i`.
   The trade-off is that very short panels leave the simplex weights and slopes
   weakly identified.

   *Remark.* This is plausibly violated when :math:`T` is on the order of
   :math:`N` or smaller (so the per-unit least squares with :math:`N-1` simplex
   weights becomes ill-posed). To spot it, re-run on a longer pre-period or
   aggregate to a coarser frequency and check whether the average effect and
   weights are stable; large swings suggest :math:`T` is too short.

4. Linearity of the outcome in the treatment vector. The paper writes
   :math:`y_{it} = y_{it}^N + \mathbf{D}_{it}^\top \boldsymbol{\alpha}_i`, i.e.
   the treatment effect is linear in :math:`\mathbf{D}_{it}` (with unit-specific
   slopes :math:`\boldsymbol{\alpha}_i`). Multi-valued :math:`\mathbf{D}_{it}`
   is fine, and interactions/polynomial terms can be entered as extra columns of
   :math:`\mathbf{D}_{it}`, but CTSC does not estimate a fully nonparametric
   dose-response curve.

   *Remark.* This is plausibly violated when the dose-response is strongly
   nonlinear inside the support and you have not encoded the nonlinearity (e.g.
   sharp regime-switching, kinked schedules). To spot it, add a quadratic or
   spline term to ``treatment_vars`` and test whether the higher-order
   coefficient is jointly significant under the sign-flip distribution.

5. Slope heterogeneity :math:`\boldsymbol{\alpha}_i` is unit-specific but
   time-invariant. Each unit gets its own marginal effect, but it does not drift
   over time within a unit.

   *Remark.* This is plausibly violated when the marginal effect itself moves --
   a minimum-wage elasticity that changes after a labour-market reform, a
   dose-response that shifts when patient populations change. To spot it, split
   the post-treatment window in half, refit, and compare the recovered
   :math:`\widehat{\boldsymbol{\alpha}}_i`; large within-unit drift indicates
   the time-invariance assumption is binding.

6. No simultaneity in :math:`\mathbf{D}_{it}`. CTSC allows
   :math:`\mathbf{D}_{it}` to be correlated with the factors
   :math:`\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i` (this is the whole
   point), but it still assumes :math:`\mathbf{D}_{it}` is mean-independent of
   the idiosyncratic shock :math:`\epsilon_{it}`. Reverse causality from
   contemporaneous :math:`\epsilon_{it}` to :math:`\mathbf{D}_{it}` breaks
   identification.

   *Remark.* This is plausibly violated when the policy responds within the same
   period to the outcome -- e.g. the minimum wage being raised *because*
   employment surprised on the upside this quarter. To spot it, regress
   :math:`\Delta \mathbf{D}_{it}` on lagged residuals from a no-treatment factor
   fit; significant feedback is a red flag.

When not to use CTSC
^^^^^^^^^^^^^^^^^^^^^^^^

* Single treated unit with a clean binary policy. A canonical
  comparative-case-study set-up (one state passes one law on one date,
  others never do) is exactly what vanilla SC was built for; CTSC's
  per-unit weight system is unnecessary overhead and its sign-flip
  inference is weaker than the placebo / conformal inference available
  in the binary-treatment world. Use :doc:`tssc` or :doc:`fdid`.

* Short panels. Powell's consistency story is in :math:`T \to
  \infty`. With :math:`T \lesssim N` the per-unit least squares with
  :math:`N-1` simplex weights is ill-posed and the average effect can
  swing dramatically with the seed. Prefer :doc:`fdid` (which is
  designed for short panels by stepwise donor selection) or a factor
  estimator that explicitly regularises the rank.

* Treated trajectory outside the donor convex hull. If the treated
  unit's untreated trend cannot be expressed as a simplex combination
  of the donors' untreated trends -- coastal vs. interior states, an
  outlier sector, a country with idiosyncratic seasonality -- CTSC has
  no fix; the per-unit :math:`\Omega_i` will collapse and the average
  effect is dominated by a handful of well-fit units. Use :doc:`fma`
  or :doc:`scmo` (auxiliary outcomes) to widen the donor information
  set before forcing a hull fit.

* Treatment effect that drifts over time within a unit. CTSC fixes
  :math:`\boldsymbol{\alpha}_i` across time. If the dose-response is genuinely
  time-varying (a minimum-wage elasticity that changes after a recession,
  a drug whose effect attenuates with tolerance), CTSC will return an
  average across the post-window that masks the dynamics. Use a
  time-varying-effects estimator (:doc:`tasc` for state-space dynamics,
  :doc:`dscar` for autoregressive treated processes) instead.

* Strongly nonlinear or kinked dose-response that you cannot encode.
  CTSC is linear in :math:`\mathbf{D}_{it}`. If the policy effect has a sharp
  kink or saturation that no parsimonious basis expansion captures,
  fall back to a doubly-robust panel estimator or a changes-in-changes
  design.

* Contemporaneous reverse causality from outcome to treatment. CTSC
  permits the treatment to be correlated with unobserved factors, but
  not with the *same-period* idiosyncratic shock. If the policy is set
  in response to within-period outcome surprises, you need an
  instrument or a timing-based identification strategy; CTSC alone is
  not enough.

Mathematical Formulation
------------------------

Model (paper eq. 4)
^^^^^^^^^^^^^^^^^^^^

.. math::

   y_{it} = \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i
            + \mathbf{D}_{it}^\top \boldsymbol{\alpha}_i + \epsilon_{it},

where :math:`\mathbf{D}_{it}` is a :math:`K`-vector of (continuous/discrete)
treatment variables, :math:`\boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_i` is
an interactive fixed-effects (factor) term, and :math:`\boldsymbol{\alpha}_i`
is the unit-specific slope.

Joint estimation (paper eq. 5-6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CTSC minimises, over slopes :math:`\mathbf{b}` and per-unit simplex weights
:math:`\boldsymbol{\phi}`,

.. math::

   \frac{1}{2NT}\sum_{i \in \mathcal{N}} \Omega_i^{-1}
     \sum_{t \in \mathcal{T}}
     \Bigl[ y_{it} - \mathbf{D}_{it}^\top \mathbf{b}_i
            - \sum_{j \ne i} \phi_j^i
              (y_{jt} - \mathbf{D}_{jt}^\top \mathbf{b}_j) \Bigr]^2,
   \quad \phi_j^i \ge 0,\ \sum_{j \ne i} \phi_j^i = 1,

where :math:`y_{it} - \mathbf{D}_{it}^\top \mathbf{b}_i` is unit :math:`i`'s
untreated outcome, its synthetic control is a convex combination of the *other*
units' untreated outcomes, and :math:`\Omega_i` is a per-unit fit weight (a
two-step measure, eq. 6) that down-weights units lacking a good synthetic
control. The average effect (paper eq. 7) is
:math:`\alpha^{AE} \coloneqq \sum_{i \in \mathcal{N}} \pi_i\,
\boldsymbol{\alpha}_i`.

Inference (paper Section 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because CTSC mechanically correlates units (each control is built from
the others), inference uses unit-level moment scores at the
null-restricted estimate and calibrates a Wald statistic with a
Rademacher (sign-flip) randomization distribution, valid under arbitrary
within- and cross-unit dependence.

Implementation note
-------------------

The paper minimises the objective with Nelder-Mead over all
:math:`NK + N(N-1)` parameters. mlsynth exploits the biconvex
structure -- weighted linear least squares in the slopes for fixed
weights (a single closed-form linear solve) and :math:`N` independent
simplex-constrained least squares in the weights for fixed slopes -- and
solves it by block coordinate descent. This optimises the same
objective with far better stability and speed. The null-restricted fit
used for inference imposes :math:`\sum_i \pi_i \boldsymbol{\alpha}_i = a_0`
via a KKT-augmented linear system.

Calibration to the paper's simulation
--------------------------------------

:mod:`mlsynth.utils.ctsc_helpers.simulation` reproduces the paper's
Section 5 / Table 1 Monte Carlo (Models 1-4). The data-generating process
is

.. math::

   y_{it} = \beta_i d_{it} + 5 \sum_{k=1}^{2} \lambda_t^{(k)} \mu_i^{(k)}
            + \epsilon_{it},

with piecewise factor paths :math:`\lambda_t^{(k)}`,
:math:`\mu_i^{(k)} \sim U(0,1)`, :math:`\epsilon_{it} \sim N(0, \tfrac14)`,
and :math:`\beta_i = \sum_k \mu_i^{(k)} - \tfrac1N \sum_i \sum_k
\mu_i^{(k)}` (so the true average effect is exactly zero). The
continuous treatment :math:`d_{it}` is a function of the same factors,
so it correlates with the interactive fixed effects.

Reproduced calibration (mlsynth, fewer Monte-Carlo draws than the paper's
1000):

============  ============================  =========================
Model         CTSC mean bias (paper)        Two-way FE bias (paper)
============  ============================  =========================
1 (n=10)      ~0.00 (0.011)                 ~0.80 (0.850)
2 (n=30)      ~0.00 (0.005)                 ~0.82 (0.846)
3 (within)    ~0.00 (-0.001)                ~0.49 (0.423)
4 (T=20)      ~0.01 (-0.002)                ~0.53 (0.263)
============  ============================  =========================

CTSC is essentially unbiased across all models while two-way fixed
effects is badly biased; the inference rejects the (true) zero null at
roughly the nominal 5% rate (paper Panel B ~0.044).

.. code-block:: python

   from mlsynth.utils.ctsc_helpers.simulation import run_simulation

   for model in (1, 2, 3, 4):
       s = run_simulation(model, n_sims=100, seed=0)
       print(f"Model {s.model}: CTSC bias={s.ctsc_mean_bias:+.3f} "
             f"MAD={s.ctsc_mad:.3f} | FE bias={s.fe_mean_bias:+.3f}")

Core API
--------

.. automodule:: mlsynth.estimators.ctsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.CTSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.ctsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ctsc_helpers.estimate
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ctsc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ctsc_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ctsc_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.ctsc_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd

   from mlsynth import CTSC
   from mlsynth.utils.ctsc_helpers.simulation import generate_model

   # A continuous-treatment panel (here from the paper's Model 1 DGP).
   rng = np.random.default_rng(3)
   Y, D, true_ae = generate_model(1, rng)
   n, T = Y.shape
   rows = [{"state": f"s{i}", "qtr": t, "emp": Y[i, t], "minwage": D[i, t, 0]}
           for i in range(n) for t in range(T)]
   df = pd.DataFrame(rows)

   res = CTSC({
       "df": df,
       "outcome": "emp",
       "treat": "minwage",          # placeholder for the base config
       "treatment_vars": ["minwage"],
       "unitid": "state",
       "time": "qtr",
       "inference": True,
   }).fit()

   print(f"average marginal effect = {res.average_effect[0]:+.4f}  "
         f"(true = {true_ae})")
   print(f"sign-flip Wald p-value  = {res.inference.p_value[0]:.3f}")
   # Unit-specific slopes (heterogeneity):
   print("unit effects:", np.round(res.unit_effects[:, 0], 3))

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., &
Wager, S. (2021). "Synthetic Difference-in-Differences."
*American Economic Review* 111(12):4088-4118.

Canay, I. A., Romano, J. P., & Shaikh, A. M. (2017). "Randomization
Tests Under an Approximate Symmetry Assumption." *Econometrica*
85(3):1013-1030.

Dube, A., & Zipperer, B. (2015). "Pooling Multiple Case Studies Using
Synthetic Controls: An Application to Minimum Wage Policies." IZA DP.

Powell, D. (2022). "Synthetic Control Estimation Beyond Comparative Case
Studies: Does the Minimum Wage Reduce Employment?" *Journal of Business &
Economic Statistics* 40(3):1302-1314.

Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference
with Interactive Fixed Effects Models." *Political Analysis* 25(1):57-76.

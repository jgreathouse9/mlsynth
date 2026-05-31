Continuous-Treatment Synthetic Control (CTSC)
==============================================

.. currentmodule:: mlsynth

Overview
--------

CTSC (Powell, D. (2022). *"Synthetic Control Estimation Beyond
Comparative Case Studies: Does the Minimum Wage Reduce Employment?,"*
Journal of Business & Economic Statistics 40(3):1302-1314) generalises
the synthetic control method to settings the original was never built
for: **continuous and/or multi-valued treatments** in panels where there
is **no clean treated / never-treated split**.

The canonical synthetic control estimates the effect of one unit
adopting a single binary policy, using never-treated units as donors.
But many policy variables are continuous and adopted (and changed) by
every unit -- every U.S. state has a minimum wage that moves over time,
so there is no "untreated" state to serve as a clean control. CTSC (the
paper's "GSC") handles exactly this case.

.. note::

   The paper names the estimator **GSC** (generalized synthetic control).
   mlsynth calls it **CTSC** to avoid collision with Xu (2017)'s
   differently constructed Generalized Synthetic Control (``gsynth``),
   which is an interactive-fixed-effects estimator.

What CTSC does
^^^^^^^^^^^^^^

* Builds a synthetic control for **every** unit out of the other units'
  *untreated* outcomes.
* Jointly estimates a **unit-specific treatment-slope vector**
  :math:`\alpha_i` (allowing fully heterogeneous marginal effects) and
  the synthetic-control weights.
* Reports the **population-weighted average marginal effect**
  :math:`\alpha^{AE} = \sum_i \pi_i \alpha_i`.
* Permits an interactive-fixed-effects (factor) outcome structure, so it
  is consistent when the treatment is correlated with unobserved factors
  and trends -- precisely the case where two-way fixed-effects regression
  is badly biased.

When to use CTSC
^^^^^^^^^^^^^^^^

* The policy variable is **continuous or multi-valued** (minimum wage,
  tax rate, dosage) rather than a 0/1 indicator.
* **Every unit is "treated"** with a time-varying intensity, so there is
  no never-treated donor pool and comparative-case-study SC cannot be
  applied.
* You suspect the treatment is **correlated with unobserved
  trends/factors**, making two-way fixed effects inconsistent.
* You want **unit-specific marginal effects** (heterogeneity), not just
  an average.

Assumptions (and how to spot violations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CTSC is a *much* more permissive estimator than vanilla SC -- it does
not require a clean treated/control split, it does not assume
homogeneous effects, and it does not require the analyst to specify the
number of factors. But it does still rely on a small set of identifying
assumptions. Each is given here together with the symptom you would see
if it failed.

(a) **Interactive fixed-effects DGP for the untreated outcome.**
    The paper assumes
    :math:`Y_{it}^N = \lambda_t' \mu_i + \epsilon_{it}`, i.e. the
    counterfactual (no-treatment) outcome is a low-rank factor model
    plus mean-zero idiosyncratic noise.

    *Plausibly violated when* the outcome has strong unit-specific
    nonlinear trends, structural breaks affecting only some units, or
    unit-specific deterministic time polynomials that cannot be written
    as :math:`\lambda_t' \mu_i`. *Diagnostic*: fit an interactive-FE
    model (e.g. ``gsynth``) to the pre-/non-treatment data and inspect
    the residuals -- if there is visible unit-specific curvature left,
    the factor structure is misspecified.

(b) **Convex-hull condition on donor loadings.**
    For each treated-period observation of unit :math:`i`, its factor
    loading :math:`\mu_i` must lie (approximately) in the convex hull
    of the other units' loadings, so that a simplex-weighted combination
    of donors can reproduce its untreated trajectory.

    *Plausibly violated when* the treated unit is an outlier on level,
    seasonality, or factor exposure -- a coastal mega-state with no
    interior analog, a country with idiosyncratic policy that no donor
    shares. *Diagnostic*: look at the per-unit fit weights
    :math:`\Omega_i` returned by CTSC; units with very poor untreated
    fit get heavily down-weighted, and if most units fall in that
    bucket the hull condition is failing population-wide.

(c) **Large** :math:`T` **regime (consistency, not factor count).**
    Powell shows the estimator is consistent as :math:`T \to \infty`
    *without* the user having to specify the rank of
    :math:`\lambda_t' \mu_i`. The trade-off is that very short panels
    leave the simplex weights and slopes weakly identified.

    *Plausibly violated when* :math:`T` is on the order of :math:`n` or
    smaller (so the per-unit least squares with :math:`n-1` simplex
    weights becomes ill-posed). *Diagnostic*: re-run on a longer
    pre-period or aggregate to a coarser frequency and check whether
    the average effect and weights are stable; large swings suggest
    :math:`T` is too short.

(d) **Linearity of the outcome in the treatment vector.**
    The paper writes :math:`Y_{it} = Y_{it}^N + D_{it}' \alpha_i`, i.e.
    the treatment effect is linear in :math:`D_{it}` (with unit-specific
    slopes :math:`\alpha_i`). Multi-valued :math:`D_{it}` is fine, and
    interactions/polynomial terms can be entered as extra columns of
    :math:`D_{it}`, but CTSC does **not** estimate a fully nonparametric
    dose-response curve.

    *Plausibly violated when* the dose-response is strongly nonlinear
    inside the support and you have not encoded the nonlinearity (e.g.
    sharp regime-switching, kinked schedules). *Diagnostic*: add a
    quadratic or spline term to ``treatment_vars`` and test whether the
    higher-order coefficient is jointly significant under the sign-flip
    distribution.

(e) **Slope heterogeneity** :math:`\alpha_i` **is unit-specific but
    time-invariant.**
    Each unit gets its own marginal effect, but it does not drift over
    time within a unit.

    *Plausibly violated when* the marginal effect itself moves -- a
    minimum-wage elasticity that changes after a labour-market reform,
    a dose-response that shifts when patient populations change.
    *Diagnostic*: split the post-treatment window in half, refit, and
    compare the recovered :math:`\alpha_i`; large within-unit drift
    indicates the time-invariance assumption is binding.

(f) **No simultaneity in** :math:`D_{it}`.
    CTSC allows :math:`D_{it}` to be correlated with the factors
    :math:`\lambda_t' \mu_i` (this is the whole point), but it still
    assumes :math:`D_{it}` is mean-independent of the idiosyncratic
    shock :math:`\epsilon_{it}`. Reverse causality from contemporaneous
    :math:`\epsilon_{it}` to :math:`D_{it}` breaks identification.

    *Plausibly violated when* the policy responds within the same
    period to the outcome -- e.g. the minimum wage being raised
    *because* employment surprised on the upside this quarter.
    *Diagnostic*: regress :math:`\Delta D_{it}` on lagged residuals
    from a no-treatment factor fit; significant feedback is a red
    flag.

When **not** to use CTSC
^^^^^^^^^^^^^^^^^^^^^^^^

* **Single treated unit with a clean binary policy.** A canonical
  comparative-case-study set-up (one state passes one law on one date,
  others never do) is exactly what vanilla SC was built for; CTSC's
  per-unit weight system is unnecessary overhead and its sign-flip
  inference is weaker than the placebo / conformal inference available
  in the binary-treatment world. Use :doc:`tssc` or :doc:`fdid`.

* **Short panels.** Powell's consistency story is in :math:`T \to
  \infty`. With :math:`T \lesssim n` the per-unit least squares with
  :math:`n-1` simplex weights is ill-posed and the average effect can
  swing dramatically with the seed. Prefer :doc:`fdid` (which is
  designed for short panels by stepwise donor selection) or a factor
  estimator that explicitly regularises the rank.

* **Treated trajectory outside the donor convex hull.** If the treated
  unit's untreated trend cannot be expressed as a simplex combination
  of the donors' untreated trends -- coastal vs. interior states, an
  outlier sector, a country with idiosyncratic seasonality -- CTSC has
  no fix; the per-unit :math:`\Omega_i` will collapse and the average
  effect is dominated by a handful of well-fit units. Use :doc:`fma`
  or :doc:`scmo` (auxiliary outcomes) to widen the donor information
  set before forcing a hull fit.

* **Treatment effect that drifts over time within a unit.** CTSC fixes
  :math:`\alpha_i` across time. If the dose-response is genuinely
  time-varying (a minimum-wage elasticity that changes after a recession,
  a drug whose effect attenuates with tolerance), CTSC will return an
  average across the post-window that masks the dynamics. Use a
  time-varying-effects estimator (:doc:`tasc` for state-space dynamics,
  :doc:`dscar` for autoregressive treated processes) instead.

* **Strongly nonlinear or kinked dose-response that you cannot encode.**
  CTSC is linear in :math:`D_{it}`. If the policy effect has a sharp
  kink or saturation that no parsimonious basis expansion captures,
  fall back to a doubly-robust panel estimator or a changes-in-changes
  design.

* **Contemporaneous reverse causality from outcome to treatment.** CTSC
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

   Y_{it} = \lambda_t' \mu_i + D_{it}' \alpha_i + \epsilon_{it},

where :math:`D_{it}` is a :math:`K`-vector of (continuous/discrete)
treatment variables, :math:`\lambda_t' \mu_i` is an interactive
fixed-effects (factor) term, and :math:`\alpha_i` is the unit-specific
slope.

Joint estimation (paper eq. 5-6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CTSC minimises, over slopes :math:`b` and per-unit simplex weights
:math:`\phi`,

.. math::

   \frac{1}{2nT}\sum_i \Omega_i^{-1} \sum_t
     \Bigl[ Y_{it} - D_{it}' b_i
            - \sum_{j \ne i} \phi_j^i (Y_{jt} - D_{jt}' b_j) \Bigr]^2,
   \quad \phi_j^i \ge 0,\ \sum_{j \ne i} \phi_j^i = 1,

where :math:`Y_{it} - D_{it}' b_i` is unit :math:`i`'s untreated outcome,
its synthetic control is a convex combination of the *other* units'
untreated outcomes, and :math:`\Omega_i` is a per-unit fit weight (a
two-step measure, eq. 6) that down-weights units lacking a good
synthetic control. The average effect (paper eq. 7) is
:math:`\alpha^{AE} = \sum_i \pi_i \alpha_i`.

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
:math:`nK + n(n-1)` parameters. mlsynth exploits the **biconvex**
structure -- weighted linear least squares in the slopes for fixed
weights (a single closed-form linear solve) and :math:`n` independent
simplex-constrained least squares in the weights for fixed slopes -- and
solves it by **block coordinate descent**. This optimises the same
objective with far better stability and speed. The null-restricted fit
used for inference imposes :math:`\sum_i \pi_i \alpha_i = a_0` via a
KKT-augmented linear system.

Calibration to the paper's simulation
--------------------------------------

:mod:`mlsynth.utils.ctsc_helpers.simulation` reproduces the paper's
Section 5 / Table 1 Monte Carlo (Models 1-4). The data-generating process
is

.. math::

   Y_{it} = \beta_i d_{it} + 5 \sum_{k=1}^{2} \lambda_t^{(k)} \mu_i^{(k)}
            + \epsilon_{it},

with piecewise factor paths :math:`\lambda_t^{(k)}`,
:math:`\mu_i^{(k)} \sim U(0,1)`, :math:`\epsilon_{it} \sim N(0, \tfrac14)`,
and :math:`\beta_i = \sum_k \mu_i^{(k)} - \tfrac1n \sum_i \sum_k
\mu_i^{(k)}` (so the **true average effect is exactly zero**). The
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

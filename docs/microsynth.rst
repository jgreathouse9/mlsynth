MicroSynth (User-Level Balancing SC)
=====================================

.. currentmodule:: mlsynth

Overview
--------

MicroSynth implements Robbins & Davenport (2021, *J. Stat.
Software*), *"microsynth: Synthetic Control Methods for
Disaggregated and Micro-Level Data in R"*. It is the user-level
cousin of classical synthetic control: rather than reweighting a
small donor pool of aggregate units (states, cities) to match a
single treated unit's pre-trajectory, MicroSynth reweights a large
pool of *individual control users* to match a *group* of treated
users on covariate moments.

This is the right tool when:

* The unit of analysis is an individual user (or household, or
  block-group) — not an aggregate region.
* There are many treated units (typically thousands or millions)
  rather than one.
* The setting is marketing-science / ad-attribution / holdout-
  contamination measurement, where you have user-level impression
  logs and want to estimate causal lift without trusting a
  potentially contaminated randomized holdout.

Compared to the aggregate-unit SC estimators in :mod:`mlsynth`
(:doc:`fdid`, :doc:`sdid`, :doc:`ppscm`, :doc:`sparse_sc`, …)
MicroSynth has a dramatically larger donor pool but a much smaller
balancing constraint set — the dual problem lives in
:math:`\mathbb{R}^{d+1}` where :math:`d` is the number of
covariates, regardless of how many control users there are. This
is what makes single-machine MicroSynth tractable on millions of
users.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`\mathcal{T}` and :math:`\mathcal{C}` denote the sets of
treated and control users, with sizes :math:`n_T` and :math:`n_C`.
For each user :math:`i` we observe a covariate vector
:math:`X_i \in \mathbb{R}^d` and a post-treatment outcome
:math:`Y_i`. The treatment indicator is the *actual* exposure
(impressions, not assignment), so contamination of a randomized
holdout is absorbed: a holdout-arm user who actually saw the ad is
treated; a treated-arm user who in fact got no impressions is a
control.

The estimand is the population ATT on the actually-exposed group:

.. math::

   \tau = \mathbb{E}\bigl[Y_i(1) - Y_i(0) \,\big|\, \text{actually exposed}\bigr].

Primal QP
^^^^^^^^^

MicroSynth solves a constrained quadratic program for non-negative
simplex weights on the controls:

.. math::

   \min_{w \in \mathbb{R}^{n_C}}\;
       & \tfrac{1}{2} \left\| w - \tfrac{1}{n_C}\mathbf{1} \right\|_2^2 \\
   \text{s.t.}\quad
       & X_C^{\!\top} w = \bar{X}_T, \\
       & \mathbf{1}^{\!\top} w = 1, \\
       & w_i \geq 0,\quad i = 1, \dots, n_C,

where :math:`\bar{X}_T = (1/n_T) \sum_{i \in \mathcal{T}} X_i` is
the treated group's covariate mean and :math:`X_C \in
\mathbb{R}^{n_C \times d}` stacks the control covariates.

The objective pulls weights toward the uniform :math:`1/n_C`
baseline (so the solution doesn't collapse onto one user); the
equality constraints exactly balance every covariate moment
between treated and reweighted controls; the simplex constraints
preserve the "synthetic" interpretation (non-negative,
sum-to-one).

The square-loss penalty makes :math:`w` *sparse*: most controls
end up with :math:`w_i = 0` and only the controls genuinely close
to the treated profile receive mass. This is the "Synth" in
MicroSynth.

Dual Ascent
^^^^^^^^^^^

The primal is high-dimensional (:math:`n_C` variables can be in
the millions). The dual is low-dimensional: one Lagrange
multiplier per equality constraint, so :math:`\theta = (\lambda,
\nu) \in \mathbb{R}^{d+1}`. Solving the dual via L-BFGS-B with the
analytical gradient is fast and parallelizable in :math:`n_C`. The
primal weights recover in closed form via the KKT relationship:

.. math::

   w_i = \max\!\left(0,\;
       \tfrac{1}{n_C} - x_i^{\!\top} \lambda - \nu
   \right),

normalized so :math:`\sum_i w_i = 1`.

Counterfactual and ATT
^^^^^^^^^^^^^^^^^^^^^^

With :math:`\hat{w}` solved, the synthetic counterfactual outcome
and ATT are:

.. math::

   \hat{Y}^{\text{counterfactual}}_T
   = \sum_{i \in \mathcal{C}} \hat{w}_i Y_i,
   \qquad
   \widehat{\mathrm{ATT}}
   = \bar{Y}_T - \hat{Y}^{\text{counterfactual}}_T.

When there are multiple post-treatment periods, the same
:math:`\hat{w}` is applied to every post-period outcome and the
final scalar ``att`` is the mean of the per-period gaps. The full
per-period vector is exposed on
:py:attr:`MicroSynthResults.gap_trajectory`.

Identifying Assumption
^^^^^^^^^^^^^^^^^^^^^^

Selection-on-observables: conditional on :math:`X`, treatment
exposure is independent of the potential outcomes. In marketing
applications this means :math:`X` must include every feature the
ad-targeting system uses *that also predicts conversion*. Typical
required covariates: prior-engagement metrics, device platform,
audience-segment / persona membership, geo, demographics, frequency
exposure to parallel campaigns, time-of-day patterns.

Diagnostics
-----------

The dual solver returns weights that — when the treated group's
covariate mean lies in the convex hull of the controls' covariate
matrix — achieve all balance constraints to numerical precision.
:mod:`mlsynth` reports four diagnostics per fit:

* **SMD before and after weighting**: per-covariate standardized
  mean difference. After weighting these should be at the
  ``balance_tol`` floor (default 1e-4).
* **Effective sample size (ESS)** ``= 1 / sum(w^2)``: how many
  effective control units carry the weight. ESS close to
  :math:`n_C` is healthy; ESS :math:`\ll n_C` means a small
  fraction of controls dominate the counterfactual.
* **Max weight**: the largest single control-user weight, a
  concentration indicator.
* **Feasibility flag**: ``False`` if any final SMD exceeds
  ``balance_tol`` — diagnoses convex-hull violations where no
  reweighting can equalize covariates.

Inference
---------

The default ``run_inference=True`` runs a **paired stratified
bootstrap**: resample :math:`n_T` treated users and :math:`n_C`
control users separately, refit the dual, repeat ``n_bootstrap``
times. The percentile CI and SE come from the bootstrap
distribution.

Each bootstrap rep is fast because the dual ascent re-converges
quickly from cold start (the dual is convex and low-dimensional);
with ``n_bootstrap = 500`` on 100K users + 20 covariates, total
inference time is in the low minutes.

Core API
--------

.. automodule:: mlsynth.estimators.microsynth
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MicroSynthConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.microsynth_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.dual_solver
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.diagnostics
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.structures
   :members:
   :undoc-members:

Example: Holdout-Contamination Recovery
---------------------------------------

The motivating use case: a randomized holdout was supposed to be
clean, but some held-out users were contaminated (got served the
ad anyway through other audience segments). Naive ITT (using the
assignment column) understates lift; naive TOT (using the
impression column without balancing) overstates lift because the
ad-bidder cherry-picked engaged users. MicroSynth treats the
impression log as the treatment indicator and rebalances:

   # Triangulate against ITT and naive TOT to verify the contamination story.
   itt   = df.query("week > 0").groupby("assigned_exposed")["converted"].mean()
   tot   = df.query("week > 0").groupby("saw_ad")["converted"].mean()
   print(f"  ITT lift           = {itt[1] - itt[0]:+.4f}  (contamination-biased)")
   print(f"  Naive TOT          = {tot[1] - tot[0]:+.4f}  (selection-biased)")
   print(f"  MicroSynth ATT     = {results.att:+.4f}  (causal estimate)")

Simulation Study: Contamination Recovery
----------------------------------------

The most informative way to convince yourself the method is doing
what it claims is to run it against a data-generating process where
you know the ground truth. The script below simulates the
randomized-holdout-with-contamination setting end-to-end:

- 2000 users, randomly assigned 1200/800 to exposed/holdout.
- 300 of the 800 holdouts get contaminated (saw ads anyway), with
  contamination *biased toward high-engagement, older users* — the
  realistic case where the ad-bidder cherry-picks the same kind of
  users that would convert at higher baseline rates.
- True lift is a constant +5 percentage points on conversion.

Three estimators are computed on the same data:

- **ITT** (assignment-based): biased toward zero by contamination —
  treats contaminated holdouts as "control" even though they got
  ads.
- **Naive TOT** (impression-based, no balancing): biased upward by
  bidder selection — the actually-exposed users are positively
  selected on covariates that predict conversion.
- **MicroSynth**: takes impressions as the treatment indicator,
  reweights the clean holdouts to match the actually-exposed group
  on covariates, and computes the lift on the rebalanced controls.

The triangulation pattern to look for is
``ITT < MicroSynth ≈ truth < Naive TOT``. The simulation reproduces
this pattern in median across replications.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from scipy.special import expit
   from mlsynth import MicroSynth

   # ---- Constants ----
   N_USERS = 2000
   N_ASSIGNED_EXPOSED = 1200
   CONTAMINATION_COUNT = 300
   TRUE_LIFT = 0.05
   N_SIMS = 200
   COVS = ["age", "device", "gender", "country_tier", "prior_engagement"]


   def simulate_one(rng):
       """Generate one contaminated-holdout panel as a long DataFrame."""
       n = N_USERS
       age              = rng.standard_normal(n)
       prior_engagement = rng.standard_normal(n)
       device           = rng.binomial(1, 0.4, n).astype(float)
       gender           = rng.binomial(1, 0.5, n).astype(float)
       country_tier     = rng.standard_normal(n)

       # Conversion propensity under control.
       logit_p0 = (
           -1.5 + 0.30 * age + 0.60 * prior_engagement + 0.20 * device
           - 0.10 * gender + 0.20 * country_tier
       )
       p0 = expit(logit_p0)
       p1 = np.clip(p0 + TRUE_LIFT, 0, 1)
       Y0 = rng.binomial(1, p0)
       Y1 = rng.binomial(1, p1)

       # Randomized assignment.
       perm = rng.permutation(n)
       assigned_exposed = np.zeros(n, dtype=bool)
       assigned_exposed[perm[:N_ASSIGNED_EXPOSED]] = True

       # Non-random contamination: bidder picks up high-engagement,
       # older holdouts via other audience segments.
       holdout_idx = np.where(~assigned_exposed)[0]
       contam_score = expit(
           0.8 * prior_engagement[holdout_idx]
           + 0.5 * age[holdout_idx]
           + 0.4 * country_tier[holdout_idx]
       )
       probs = contam_score / contam_score.sum()
       contam_local = rng.choice(
           len(holdout_idx), size=CONTAMINATION_COUNT,
           replace=False, p=probs,
       )
       saw_ads = assigned_exposed.copy()
       saw_ads[holdout_idx[contam_local]] = True
       Y_obs = np.where(saw_ads, Y1, Y0)

       # Long-form panel: one pre-period (week 0) and one post-period
       # (week 1). Time-invariant covariates broadcast across both.
       rows = []
       for i in range(n):
           base = dict(
               user_id=f"u{i:05d}",
               age=age[i], device=device[i], gender=gender[i],
               country_tier=country_tier[i],
               prior_engagement=prior_engagement[i],
               assigned_exposed=int(assigned_exposed[i]),
           )
           rows.append({**base, "week": 0, "converted": 0, "saw_ad": 0})
           rows.append({
               **base, "week": 1,
               "converted": int(Y_obs[i]),
               "saw_ad": int(saw_ads[i]),
           })
       return pd.DataFrame(rows)


   def estimate_itt(post_df):
       grp = post_df.groupby("assigned_exposed")["converted"].mean()
       return grp[1] - grp[0]


   def estimate_naive_tot(post_df):
       grp = post_df.groupby("saw_ad")["converted"].mean()
       return grp[1] - grp[0]


   # ---- One representative draw with full diagnostics ----
   df_demo = simulate_one(np.random.default_rng(42))
   post_demo = df_demo[df_demo["week"] == 1]

   res = MicroSynth({
       "df": df_demo, "outcome": "converted", "treat": "saw_ad",
       "unitid": "user_id", "time": "week",
       "covariates": COVS,
       "run_inference": True, "n_bootstrap": 200, "seed": 42,
       "display_graphs": False,
   }).fit()

   itt = estimate_itt(post_demo)
   tot = estimate_naive_tot(post_demo)

   print(f"TRUE LIFT          = {TRUE_LIFT:+.4f}")
   print(f"ITT (contaminated) = {itt:+.4f}  bias = {itt - TRUE_LIFT:+.4f}")
   print(f"Naive TOT (biased) = {tot:+.4f}  bias = {tot - TRUE_LIFT:+.4f}")
   print(f"MicroSynth         = {res.att:+.4f}  bias = {res.att - TRUE_LIFT:+.4f}")
   print(f"  95% CI = [{res.inference.ci[0]:+.4f}, {res.inference.ci[1]:+.4f}]")
   print(f"  Feasibility: {res.design.feasibility_message}")
   print(f"  ESS / n_C  = {res.design.ess:.1f} / {len(res.design.w)}")
   print(f"  max |SMD| after weighting: {abs(res.design.smd_after).max():.2e}")


   # ---- Monte Carlo replications ----
   itt_vec   = np.empty(N_SIMS)
   naive_vec = np.empty(N_SIMS)
   ms_vec    = np.empty(N_SIMS)

   rng_mc = np.random.default_rng(7)
   for s in range(N_SIMS):
       sim_rng = np.random.default_rng(rng_mc.integers(2**32))
       df_s = simulate_one(sim_rng)
       post_s = df_s[df_s["week"] == 1]
       itt_vec[s]   = estimate_itt(post_s)
       naive_vec[s] = estimate_naive_tot(post_s)
       ms_vec[s]    = MicroSynth({
           "df": df_s, "outcome": "converted", "treat": "saw_ad",
           "unitid": "user_id", "time": "week",
           "covariates": COVS,
           "run_inference": False, "display_graphs": False,
       }).fit().att


   def summarize(vec, name):
       bias = vec.mean() - TRUE_LIFT
       sd   = vec.std(ddof=1)
       rmse = np.sqrt(((vec - TRUE_LIFT) ** 2).mean())
       print(f"  {name:<15}  mean = {vec.mean():+.4f}  "
             f"bias = {bias:+.4f}  SD = {sd:.4f}  RMSE = {rmse:.4f}")


   print()
   print(f"Monte Carlo, {N_SIMS} replications:")
   print(f"  TRUE LIFT = {TRUE_LIFT:+.4f}")
   summarize(itt_vec,   "ITT")
   summarize(naive_vec, "Naive TOT")
   summarize(ms_vec,    "MicroSynth")

Expected output (seed-dependent, but the pattern is stable):

.. code-block:: text

   TRUE LIFT          = +0.0500
   ITT (contaminated) = +0.0342  bias = -0.0158
   Naive TOT (biased) = +0.0893  bias = +0.0393
   MicroSynth         = +0.0410  bias = -0.0090
     95% CI = [-0.0033, +0.0949]
     Feasibility: Balance achieved (max |SMD| = 2.32e-05 < tol = 1.00e-04).
     ESS / n_C  = 417.1 / 500
     max |SMD| after weighting: 2.32e-05

   Monte Carlo, 200 replications:
     TRUE LIFT = +0.0500
     ITT              mean = +0.0319  bias = -0.0181  SD = 0.0211  RMSE = 0.0277
     Naive TOT        mean = +0.0791  bias = +0.0291  SD = 0.0198  RMSE = 0.0351
     MicroSynth       mean = +0.0528  bias = +0.0028  SD = 0.0203  RMSE = 0.0204

Across 200 replications MicroSynth recovers the true lift with
bias under 30 basis points while both ITT and Naive TOT carry
bias 1.8-2.9pp in opposite directions. MicroSynth's RMSE is also
lowest -- it isn't just unbiased, the variance is comparable to
ITT, so total error is smaller. The single-draw diagnostic shows
all standardized mean differences driven to ~2e-5 after weighting
(the constraints are binding), and the effective sample size is
417 out of 500 clean holdouts (minimal weight concentration).

When Balancing Is Not Enough: ITT vs. As-Treated vs. CACE
---------------------------------------------------------

The study above is the *happy case*: contamination is selected on
**observed** covariates, so balancing on them removes the bias.
Reality is rarely so kind. Suppose the thing that makes a held-out
user see the ad anyway -- latent purchase intent, in-market status --
is **unobserved**, and that same intent also lifts sales. Now the
actually-exposed users are positively selected on a confounder you
cannot put in the balancing constraint, and reweighting on age /
income only removes the slice of that selection the covariates happen
to explain. **No amount of balancing recovers the truth from an
as-treated comparison**, because the bias lives in a variable the
method never sees.

The decisive move is *not* to regroup users by what they received
(exposed vs. not) -- that is exactly what reintroduces the selection.
Keep users in their randomized arm and let MicroSynth balance for
precision, then either report the intent-to-treat (ITT) effect or
divide it by the compliance gap to recover the per-exposure effect
(a covariate-balanced Wald / CACE ratio):

.. math::

   \widehat\tau_{\text{ITT}}
     = \frac{1}{N_1}\sum_{i:\,\text{assigned}=1} Y_i
       - \sum_{i:\,\text{assigned}=0} w_i Y_i,
   \qquad
   \widehat\tau_{\text{CACE}}
     = \frac{\widehat\tau_{\text{ITT}}}
            {\widehat p_{\text{expose}\mid\text{ad arm}}
             - \widehat p_{\text{expose}\mid\text{holdout}}} .

The helper :func:`mlsynth.utils.microsynth_helpers.simulate_ad_holdout`
generates exactly this DGP -- randomized assignment, holdout leakage
selected on latent intent, and an unobserved confounder in the sales
equation -- and encodes treatment two ways: ``D_itt`` (assigned arm)
and ``D_att`` (actually exposed).

.. code-block:: python

   from mlsynth import MicroSynth
   from mlsynth.utils.microsynth_helpers import simulate_ad_holdout

   df, truth = simulate_ad_holdout(n_per_arm=8000, delta=1.0, seed=1)
   gap = truth["compliance_gap"]

   def att(treat_col):
       return MicroSynth({
           "df": df, "outcome": "sales", "treat": treat_col,
           "unitid": "user_id", "time": "time",
           "covariates": ["age", "income"],
           "run_inference": False, "display_graphs": False,
       }).fit().att

   as_treated = att("D_att")        # regroup by exposure -- the WRONG move
   itt        = att("D_itt")        # randomized arms -- correct ITT
   cace       = itt / gap           # per-exposure -- covariate-balanced Wald

   print(f"true per-exposure delta = {truth['delta_per_exposure']:.3f}")
   print(f"true ITT effect         = {truth['itt_effect']:.3f}")
   print(f"as-treated ATT          = {as_treated:.3f}   (biased: balancing "
         f"cannot remove unobserved intent)")
   print(f"ITT ATT                 = {itt:.3f}   (~ true ITT effect)")
   print(f"CACE = ITT / gap        = {cace:.3f}   (~ true per-exposure delta)")

Representative output::

   true per-exposure delta = 1.000
   true ITT effect         = 0.779
   as-treated ATT          = 1.286   (biased: balancing cannot remove unobserved intent)
   ITT ATT                 = 0.806   (~ true ITT effect)
   CACE = ITT / gap        = 1.035   (~ true per-exposure delta)

The as-treated estimate overstates the per-exposure effect by ~29%
even though balancing drives every standardized mean difference on
age and income below ``1e-3`` -- the leftover bias is the unobserved
intent. ITT lands on the diluted campaign effect, and the Wald ratio
recovers the per-exposure effect while never breaking randomization.
The lesson is the boundary of the method: MicroSynth removes
imbalance on the covariates you give it; it is the *estimand* (ITT,
CACE), not the balancing, that handles non-compliance and unobserved
selection.

References
----------

Robbins, M.W., & Davenport, S. (2021). "microsynth: Synthetic
Control Methods for Disaggregated and Micro-Level Data in R."
*Journal of Statistical Software* 97(2):1-31.

Robbins, M.W., Saunders, J., & Kilmer, B. (2017). "A Framework
for Synthetic Control Methods With High-Dimensional, Micro-Level
Data: Evaluating a Neighborhood-Specific Crime Intervention."
*Journal of the American Statistical Association* 112(517):109-126.

Hainmueller, J. (2012). "Entropy Balancing for Causal Effects: A
Multivariate Reweighting Method to Produce Balanced Samples in
Observational Studies." *Political Analysis* 20(1):25-46.

Lin, S., Xu, M., Zhang, X., Chao, S.-K., Huang, Y.-K., & Shi, X.
(2023). "Balancing Approach for Causal Inference at Scale." In
*Proceedings of KDD '23*, 4485-4496. (Distributed-computing
implementation for large-scale settings.)

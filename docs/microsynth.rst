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

.. code-block:: python

   import pandas as pd
   from mlsynth import MicroSynth

   # Long-format panel: one row per (user, time).
   # 'saw_ad' = 1 if the user actually saw the ad in the post-period,
   # regardless of which arm they were assigned to.
   df = pd.read_csv("user_campaign_panel.csv")

   results = MicroSynth({
       "df":             df,
       "outcome":        "converted",
       "treat":          "saw_ad",
       "unitid":         "user_id",
       "time":           "week",
       "covariates":     ["age", "device", "gender",
                          "country_tier", "prior_engagement"],
       # Optional: fold pre-treatment engagement weeks in as additional
       # balancing constraints.
       "outcome_lag_periods": [-4, -2, -1],
       "run_inference":  True,
       "n_bootstrap":    500,
       "display_graphs": True,
   }).fit()

   print(f"ATT = {results.att:+.4f}")
   print(f"95% CI = [{results.inference.ci[0]:+.4f}, "
         f"{results.inference.ci[1]:+.4f}]")
   print(f"ESS / n_C = {results.design.ess:.0f} / {len(results.design.w)}")
   print(results.design.feasibility_message)

   # Triangulate against ITT and naive TOT to verify the contamination story.
   itt   = df.query("week > 0").groupby("assigned_exposed")["converted"].mean()
   tot   = df.query("week > 0").groupby("saw_ad")["converted"].mean()
   print(f"  ITT lift           = {itt[1] - itt[0]:+.4f}  (contamination-biased)")
   print(f"  Naive TOT          = {tot[1] - tot[0]:+.4f}  (selection-biased)")
   print(f"  MicroSynth ATT     = {results.att:+.4f}  (causal estimate)")

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

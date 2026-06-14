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

Notation
^^^^^^^^

We follow the mlsynth notation canon (``agents/agents_docs.md``); MicroSynth's
unit model is a *group* of treated units against a large control pool, so a few
page-specific symbols are fixed here.

* Units. :math:`\mathcal{I}_1` is the set of treated units (users, blocks,
  or areas) with :math:`|\mathcal{I}_1| = n_T`; :math:`\mathcal{I}_0` is the
  control pool with :math:`|\mathcal{I}_0| = n_C`, typically
  :math:`n_C \gg n_T`. (*Bridge:* the canon's single treated unit
  :math:`j = 1` generalises here to the whole set :math:`\mathcal{I}_1`.)
* Covariates. Each unit :math:`j` carries
  :math:`\mathbf{x}_j \in \mathbb{R}^d`; stack the controls as
  :math:`\mathbf{X}_0 \in \mathbb{R}^{n_C \times d}` and the treated as
  :math:`\mathbf{X}_1 \in \mathbb{R}^{n_T \times d}` (one row per unit).
* Time and outcomes. :math:`t \in \mathcal{T} \coloneqq \{1,\dots,T\}`,
  split at :math:`T_0` into pre-period :math:`\mathcal{T}_1` and post-period
  :math:`\mathcal{T}_2`. The outcome of unit :math:`j` at time :math:`t` is
  :math:`y_{jt}`.
* Weights. Control weights :math:`\mathbf{w} \in \mathbb{R}^{n_C}_{\ge 0}`,
  with optimiser :math:`\mathbf{w}^\ast`. The treated units are not reweighted
  (each carries weight 1).

Every weighting program below is an instance of the canon's SC-family shape

.. math::

   \mathbf{w}^\ast \in \operatorname*{argmin}_{\mathbf{w}\in\mathcal{C}}
      \; \mathcal{L}(\mathbf{w}) + \mathcal{P}(\mathbf{w})
      \quad\text{s.t.}\quad \mathcal{B}(\mathbf{w}) = \mathbf{0},

with fit loss :math:`\mathcal{L}`, penalty :math:`\mathcal{P}`, balance map
:math:`\mathcal{B}`, and feasible set :math:`\mathcal{C}`.

Two weighting modes
^^^^^^^^^^^^^^^^^^^^

MicroSynth exposes two weight schemes through ``weight_method``, for two
distinct regimes; they share the data-ingestion and diagnostics machinery but
solve different programs and report on different scales.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - ``"simplex"`` (default)
     - ``"panel"``
   * - Regime
     - Micro/holdout study: many individual users, one cross-section of
       exposure
     - Aggregated-area panel (the R ``microsynth`` port): repeated
       cross-sections, treated *area* vs synthetic area
   * - Feasible set
     - simplex :math:`\Delta^{n_C}` (:math:`\mathbf{w}\ge 0`,
       :math:`\|\mathbf{w}\|_1 = 1`)
     - non-negative cone :math:`\mathbb{R}^{n_C}_{\ge 0}`,
       :math:`\|\mathbf{w}\|_1 = n_T`
   * - Balance
     - covariate means :math:`\bar{\mathbf{x}}_1`
     - covariate totals + lagged-outcome totals
   * - Contrast
     - per-unit weighted mean ATT
     - treated-area total minus synthetic total, per period
   * - Inference
     - paired stratified bootstrap
     - placebo permutation

Mode A --- simplex (micro / holdout studies)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here the treatment indicator is the *actual* exposure (impressions, not
assignment), so contamination of a randomized holdout is absorbed: a
holdout-arm user who actually saw the ad is treated; a treated-arm user who in
fact got no impressions is a control. The estimand is the population ATT on the
actually-exposed group,

.. math::

   \tau \coloneqq \mathbb{E}\bigl[y_j(1) - y_j(0) \,\big|\, \text{actually exposed}\bigr].

MicroSynth solves a min-variance balancing QP for non-negative simplex weights
on the controls --- the canon shape with
:math:`\mathcal{C} = \Delta^{n_C}`,
:math:`\mathcal{P}(\mathbf{w}) = \tfrac12\|\mathbf{w} - n_C^{-1}\mathbf{1}\|_2^2`,
:math:`\mathcal{L} \equiv 0`, and balance map
:math:`\mathcal{B}(\mathbf{w}) = \mathbf{X}_0^{\!\top}\mathbf{w} -
\bar{\mathbf{x}}_1`:

.. math::

   \mathbf{w}^\ast \in \operatorname*{argmin}_{\mathbf{w} \in \mathbb{R}^{n_C}}\;
       & \tfrac{1}{2} \bigl\| \mathbf{w} - n_C^{-1}\mathbf{1} \bigr\|_2^2 \\
   \text{s.t.}\quad
       & \mathbf{X}_0^{\!\top} \mathbf{w} = \bar{\mathbf{x}}_1, \\
       & \mathbf{1}^{\!\top} \mathbf{w} = 1, \quad \mathbf{w} \ge \mathbf{0},

where :math:`\bar{\mathbf{x}}_1 \coloneqq n_T^{-1} \sum_{j \in \mathcal{I}_1}
\mathbf{x}_j` is the treated group's covariate mean. The equality constraints
exactly balance every covariate moment between treated and reweighted
controls; the simplex constraints preserve the "synthetic" interpretation; and
the quadratic penalty pulls weights toward the uniform :math:`n_C^{-1}` baseline
so the solution does not collapse onto a single user.

Dual ascent. The primal is high-dimensional (:math:`n_C` can be in the
millions) but the dual is :math:`(d+1)`-dimensional --- one multiplier
:math:`\boldsymbol{\lambda} \in \mathbb{R}^d` per covariate balance constraint
plus :math:`\nu` for the sum-to-one constraint. ``solve_microsynth_dual``
minimises the dual potential with L-BFGS-B (analytical gradient, parallelisable
in :math:`n_C`); the primal weights recover in closed form from the KKT
conditions,

.. math::

   w_j^\ast = \max\!\left(0,\;
       n_C^{-1} - \mathbf{x}_j^{\!\top} \boldsymbol{\lambda} - \nu
   \right),

renormalised so :math:`\mathbf{1}^{\!\top}\mathbf{w}^\ast = 1`. The
:math:`\max(0,\cdot)` makes :math:`\mathbf{w}^\ast` *sparse*: only controls
genuinely close to the treated profile receive mass. This dimension-reduction
(work in :math:`\mathbb{R}^{d+1}` regardless of :math:`n_C`) is what makes
single-machine MicroSynth tractable on millions of users.

Counterfactual and ATT. With :math:`\mathbf{w}^\ast` solved, the per-period
synthetic counterfactual is the weighted-mean control outcome and the
per-period effect is the canon's :math:`\tau_t`:

.. math::

   \widehat{y}_{1t} = \sum_{j \in \mathcal{I}_0} w_j^\ast\, y_{jt},
   \qquad
   \tau_t = \bar{y}_{1t} - \widehat{y}_{1t},
   \qquad
   \widehat{\tau} = |\mathcal{T}_2|^{-1}\!\!\sum_{t \in \mathcal{T}_2} \tau_t,

with :math:`\bar{y}_{1t} = n_T^{-1}\sum_{j\in\mathcal{I}_1} y_{jt}` the treated
mean. The same :math:`\mathbf{w}^\ast` is applied to every post-period; the
scalar ``att`` is :math:`\widehat{\tau}` and the per-period vector
:math:`(\tau_t)` is exposed on
:py:attr:`MicroSynthResults.gap_trajectory`.

Identifying assumptions (simplex mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Selection-on-observables: conditional on :math:`\mathbf{X}`, treatment
exposure is independent of the potential outcomes. In marketing
applications this means :math:`\mathbf{X}` must include every feature the
ad-targeting system uses *that also predicts conversion*. Typical
required covariates: prior-engagement metrics, device platform,
audience-segment / persona membership, geo, demographics, frequency
exposure to parallel campaigns, time-of-day patterns.

Selection-on-observables is the headline assumption, but in a Snap-style
ad-attribution deployment several others are doing silent work. Each is
listed here together with the realistic failure mode you would see in a
marketing-science setting and a diagnostic that flags it.

*Assumption S1 (selection-on-observables on every conversion-predictive
feature).*
The covariate vector :math:`\mathbf{X}` must contain every signal the bidder /
targeting model conditions on *that also predicts conversion*. If a
targeting feature is missing, MicroSynth's reweighting closes balance
only on the features you gave it and leaves selection bias on the
one you did not.

*Remark.* *Plausibly violated when* the bidder optimises against a model that
uses features the analyst does not have access to -- on-device
signals, third-party audience segments, latent embeddings,
in-market scoring. *Diagnostic*: probe the unobserved-intent
residual by regressing post-period conversion on the residual of
a saw-ad model that conditions on :math:`\mathbf{X}`; a non-zero coefficient
is unobserved confounding that MicroSynth cannot remove. The
existing "When Balancing Is Not Enough" section below makes this
concrete: when intent is latent, the as-treated MicroSynth ATT
overstates the per-exposure effect by ~29% even with all SMDs
below 1e-3.

*Assumption S2 (SUTVA at the user level --- no network spillovers in
conversion).*
The synthetic-control framing treats each user's potential outcome
as a function of *their own* exposure only. Exposed users
influencing unexposed users (a friend talks about the ad, an
organic post amplifies the campaign) breaks the comparison: the
control pool itself has been partially treated.

*Remark.* *Plausibly violated when* the campaign is viral or social by
design -- influencer-led launches, group-chat-shareable AR lenses,
referral mechanics. *Diagnostic*: split controls by social
distance to the exposed cohort (e.g. friends-of-treated vs.
network-distant controls) and refit; a non-trivial gap between the
two ATTs is a SUTVA failure. For genuinely spillover-prone
designs, switch to a spillover-aware aggregate estimator
(:doc:`spillsynth`, :doc:`spsydid`).

*Assumption S3 (overlap --- the treated covariate mean lies in the convex
hull of the controls).*
The primal QP enforces
:math:`\mathbf{X}_0^{\!\top} \mathbf{w} = \bar{\mathbf{x}}_1` with
:math:`\mathbf{w}` on the simplex. There is a feasible solution if and only
if :math:`\bar{\mathbf{x}}_1` is in the convex hull of the rows of
:math:`\mathbf{X}_0`; if not, no reweighting can balance every constraint
and the dual still returns a vector, but the residual imbalance is
real.

*Remark.* *Plausibly violated when* the campaign targeted a covariate cell
that the control pool barely contains -- a brand-new
audience-segment launch, a country where the ad ran but very few
organic users live, an iOS-only push with mostly Android in the
control pool. *Diagnostic*: read
:py:attr:`MicroSynthResults.design.feasibility_message` and the
per-covariate ``smd_after``; if the feasibility flag is False or
any SMD exceeds ``balance_tol``, the hull condition is failing.
The fix is to widen the control pool (drop sub-population
filters), drop a covariate that is genuinely outside support, or
accept the residual imbalance and discuss its sign.

*Assumption S4 (linear functional form, or sufficient basis expansion, of
the outcome in* :math:`\mathbf{X}` *).*
Balancing only the *first moments* of :math:`\mathbf{X}` gives an unbiased
ATT when the conditional expectation
:math:`\mathbb{E}[Y(0) \mid \mathbf{X}]` is linear in :math:`\mathbf{X}`. If the
expectation is nonlinear (e.g. age enters as a smooth bump rather
than a slope), first-moment balance is not enough -- the doubly
robust property of the balancing approach (Lin et al. 2023) only
holds under linearity in *one* of the outcome or selection models.

*Remark.* *Plausibly violated when* engagement metrics enter non-linearly
(saturation effects, threshold heaps in prior-engagement). *
Diagnostic*: add quadratic terms and selected interactions to
``covariates`` and rerun -- if the ATT moves materially, the
linear specification was binding. The KDD paper (Section 4)
explicitly recommends including higher-order moments of skewed
user-engagement covariates for exactly this reason.

*Assumption S5 (pre-period parallel mean for the rebalanced control group).*
Because the constraints are *contemporaneous* moment balance, the
counterfactual at :math:`t > T_0` is trustworthy only if the
rebalanced controls would have moved in parallel with the treated
group absent treatment. The covariates should therefore include
pre-period outcome levels (the Roanoke / Snap recipe: include
pre-intervention outcome trajectories as constraint moments).

*Remark.* *Plausibly violated when* the analyst forgot to include
pre-period outcomes in the constraint set, or when there is a
secular trend in the treated pool's outcome that no covariate
captures. *Diagnostic*: plot
:py:attr:`MicroSynthResults.gap_trajectory` over the pre-period
(include enough pre-periods to see a trend) -- a non-flat pre-period
gap is a parallel-trends violation. Robbins, Saunders &
Kilmer (2017) build the constraint set explicitly out of all
pre-period outcome-by-time cells for this reason.

*Assumption S6 (stable covariates over the analysis window --- no
compositional drift).*
The primal solves a single :math:`\mathbf{w}` and applies it to every
post-period. Implicit: the donor pool's covariate vector is
sufficient to characterise it across :math:`\mathcal{T}`.

*Remark.* *Plausibly violated when* the user base churns mid-campaign (new
cohorts join, old cohorts age out), or when a covariate itself
shifts after :math:`T_0` (e.g. ``country_tier`` re-classification,
persona-segment redefinition). *Diagnostic*: rebuild :math:`\mathbf{X}` on
the post-period sample only, recompute :math:`\bar{\mathbf{x}}` for the
rebalanced controls, and check that ``smd_after`` is still tight;
drift shows up as post-period SMDs that have crept above the
pre-period tolerance.

*Assumption S7 (treatment indicator is the actually-realised exposure, used
consistently with the estimand).*
MicroSynth identifies the ATT on the actually-exposed group
when ``treat`` is the impression column. If you instead use the
assignment column, you get an ITT under balancing on :math:`\mathbf{X}`.
Mixing the two -- naming an assignment column ``treat`` but
interpreting the answer per-exposure -- is a specification error,
not an assumption failure of the method.

*Remark.* *Plausibly violated when* the team operationalises "treated" as
"assigned" because that is what the experimentation platform
logs, but reports per-exposure lift. *Diagnostic*: always sanity
check the printed treated-fraction against the impression log;
if they disagree, the wrong column was passed.

When not to use MicroSynth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clean randomised AB test with full compliance. MicroSynth's whole
  selling point is removing observational selection bias. If the
  experimentation platform delivered a non-contaminated holdout and
  exposure compliance is near-complete, a plain difference of means is
  both unbiased and lower-variance. MicroSynth then *adds* variance
  (the bootstrap, the constraint set) without buying identification.

* Confounding is dominated by unobserved features (latent intent).
  This is the boundary case spelled out below in "When Balancing Is
  Not Enough". When holdout leakage is driven by an in-market signal
  the analyst does not have, MicroSynth zeros out SMDs on every
  observed covariate and still returns a biased ATT. Stay on the
  randomised arms -- report ITT under MicroSynth balancing for
  precision, and divide by the compliance gap to get a covariate-
  balanced CACE / Wald (the section below shows the full recipe).

* Aggregate region-level data, single treated unit. A one-state,
  one-policy DMI-style design is what classical aggregate SC was
  built for. MicroSynth's dual is :math:`(d + 1)`-dimensional but the
  primal must have many controls; with a handful of aggregate donors
  the QP degenerates and the convex-hull / overlap argument is
  exactly the classical SC argument. Use *canonical SCM*, :doc:`tssc`,
  :doc:`fdid`, or :doc:`fma` instead.

* The distribution of the outcome is the object of interest.
  MicroSynth balances means (or moments you specify) and returns a
  scalar ATT. If the question is "does the campaign compress the
  lower tail of session length?" or "what is the QTE at the 90th
  percentile of basket size?", switch to :doc:`dsc` -- the
  Wasserstein-barycenter machinery is designed exactly for that.

* The treatment is continuous or multi-valued (ad dose). MicroSynth
  encodes a binary saw-ad / not-saw-ad column. Multi-valued exposure
  (one impression vs ten vs a hundred), spend dose, or auction price
  needs the continuous-treatment framework in :doc:`ctsc`.

* Spillovers / interference within the user graph. SUTVA at the
  user level is a hard assumption; viral and social-by-design
  campaigns violate it. Covariate balancing on the user pool does
  nothing about spillovers. Switch to a spillover-aware design
  (:doc:`spillsynth`, :doc:`spsydid`) and accept that you are now
  identifying an aggregate quantity, not a user-level lift.

* Convex-hull condition fails on the targeting axis. If the
  campaign was narrowly targeted -- an iOS-only push to a brand-new
  audience segment with almost no organic match in the control pool --
  ``feasibility_message`` will fire and the residual SMDs will be
  visibly above tolerance. There is no balancing fix here: either
  widen the control pool (relax the segment filter, pool across
  countries), drop the constraint that is outside support, or
  acknowledge the residual imbalance in the writeup.

* You have billions of users and a single-machine budget. The
  in-memory dual scales as :math:`O(N d K)`; at Snap-scale this is a
  cluster job, not a workstation job. Switch to the distributed
  DistEB / DistMS variants in Lin et al. (2023), which are designed
  to run as MapReduce gradient steps over PySpark.

* Tiny treated cohort (handful of users) with many covariates.
  With :math:`n_T` small, :math:`\bar{\mathbf{x}}_1` is itself noisy, the
  balance constraints are noisy targets, and the bootstrap CI widens
  to uselessness. Aggregate the treated cohort up to a meaningful
  unit (campaign-level, segment-level) and run an aggregate SC, or
  prune covariates to those with credible cross-validated predictive
  signal.




Diagnostics
-----------

The dual solver returns weights that — when the treated group's
covariate mean lies in the convex hull of the controls' covariate
matrix — achieve all balance constraints to numerical precision.
:mod:`mlsynth` reports four diagnostics per fit:

* SMD before and after weighting: per-covariate standardized
  mean difference. After weighting these should be at the
  ``balance_tol`` floor (default 1e-4).
* Effective sample size (ESS) ``= 1 / sum(w^2)``: how many
  effective control units carry the weight. ESS close to
  :math:`n_C` is healthy; ESS :math:`\ll n_C` means a small
  fraction of controls dominate the counterfactual.
* Max weight: the largest single control-user weight, a
  concentration indicator.
* Feasibility flag: ``False`` if any final SMD exceeds
  ``balance_tol`` — diagnoses convex-hull violations where no
  reweighting can equalize covariates.

Mode B --- panel method (the R ``microsynth`` port)
---------------------------------------------------

Setting ``weight_method="panel"`` switches to a faithful port of the
panel-data weighting in the R ``microsynth`` package (Robbins et al.), for
the aggregated-area / repeated-cross-section setting --- e.g. the Seattle Drug
Market Intervention, where a treated *area* (a set of census blocks) is compared
to a synthetic area built from the untreated blocks.

The weight program
^^^^^^^^^^^^^^^^^^^

Reading the R source (``microsynth/R/weights.r``), the panel weights come from
``my.qp`` (a ``LowRankQP`` solve), not from raking calibration: a
non-negative QP that exactly balances the covariate totals (hard equality)
and least-squares-fits the pre-period outcomes (soft). Write
:math:`\mathbf{G}_0 = [\mathbf{1}\ \ \mathbf{X}_0] \in \mathbb{R}^{n_C\times(d+1)}`
(controls' covariates with an intercept column) and, for the matched outcomes,
the control lagged-outcome matrix
:math:`\mathbf{L}_0 \in \mathbb{R}^{n_C \times m}` whose columns are
:math:`\{y_{j t}\}` for :math:`t \in \mathcal{T}_1` (and, with multiple matched
outcomes, stacked across them --- see below, so :math:`m = (\#\text{outcomes})
\times |\mathcal{T}_1|`). The treated-area totals are
:math:`\mathbf{h} = \bigl(n_T,\ \mathbf{1}^{\!\top}\mathbf{X}_1\bigr)^{\!\top}`
and :math:`\boldsymbol{\ell} = \mathbf{1}^{\!\top}\mathbf{L}_1`. The program is
the canon shape with :math:`\mathcal{C} = \mathbb{R}^{n_C}_{\ge 0}`, a
least-squares fit loss, a ridge penalty, and an exact-balance map:

.. math::

   \mathbf{w}^\ast \in \operatorname*{argmin}_{\mathbf{w} \ge \mathbf{0}}\;
       \underbrace{\tfrac12\bigl\|\mathbf{L}_0^{\!\top}\mathbf{w}
           - \boldsymbol{\ell}\bigr\|_2^2}_{\mathcal{L}(\mathbf{w})}
       + \underbrace{\tfrac{\rho}{2}\|\mathbf{w}\|_2^2}_{\mathcal{P}(\mathbf{w})}
       \quad\text{s.t.}\quad
       \underbrace{\mathbf{G}_0^{\!\top}\mathbf{w} = \mathbf{h}}_{\mathcal{B}(\mathbf{w})=\mathbf{0}} .

The intercept row of :math:`\mathbf{G}_0^{\!\top}\mathbf{w} = \mathbf{h}` forces
:math:`\mathbf{1}^{\!\top}\mathbf{w} = n_T`, so the weights sum to the treated
count rather than to one. ``solve_panel_qp`` solves this with cvxpy's CLARABEL
interior-point solver; an infeasible covariate target (the treated totals lie
outside the non-negative cone spanned by the controls) raises
:class:`~mlsynth.exceptions.MlsynthEstimationError` rather than returning a
degenerate fit.

Why a ridge: non-identification of the counterfactual
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fit loss :math:`\mathcal{L}` depends on :math:`\mathbf{w}` only through the
:math:`m` lagged-outcome totals :math:`\mathbf{L}_0^{\!\top}\mathbf{w}`, and the
balance map pins only :math:`d+1` covariate totals --- together
:math:`O(m + d)` linear functionals of an :math:`n_C`-vector. Over a large
control pool (:math:`n_C \gg m + d`) the optimum is therefore a
high-dimensional face, not a point: the counterfactual is not identified by
the constraints alone. On the Seattle panel, solving the LP that minimises and
maximises the post-period synthetic total over the exact-balance feasible set
gives a feasible range for the period-13 effect of roughly
:math:`[-392,\ +153]` --- the R package's ``LowRankQP`` merely returns whichever
interior-point iterate it lands on.

mlsynth removes this ambiguity with the strictly-convex ridge
:math:`\tfrac{\rho}{2}\|\mathbf{w}\|_2^2` (``panel_ridge``, default
:math:`\rho = 10^{-6}`), which selects the unique minimum-norm /
maximum-ESS point on that face --- the most diffuse synthetic control
consistent with exact covariate balance and the best lagged-outcome fit. This
makes the estimate reproducible *and*, because ``LowRankQP``'s interior-point
iterate is itself near the minimum-norm point, it coincides with the R package's
output to 3--4 significant figures (see :doc:`replications/microsynth`).

Effects on totals
^^^^^^^^^^^^^^^^^

The panel contrast is on totals, not per-unit means: the treated-area total
minus the weighted control total, per post-period,

.. math::

   \tau_t = \sum_{j\in\mathcal{I}_1} y_{jt}
            - \sum_{j\in\mathcal{I}_0} w_j^\ast\, y_{jt},
            \qquad t \in \mathcal{T}_2,

with ``att`` :math:`= |\mathcal{T}_2|^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`. The
package's reported ``Pct.Chng`` is :math:`100(\mathrm{Trt}-\mathrm{Con})/
\mathrm{Con}` over the post window, where :math:`\mathrm{Trt} =
\sum_{t\in\mathcal{T}_2}\sum_{j\in\mathcal{I}_1} y_{jt}` and :math:`\mathrm{Con}
= \sum_{t\in\mathcal{T}_2}\sum_{j\in\mathcal{I}_0} w_j^\ast y_{jt}`.

Multi-outcome joint match
^^^^^^^^^^^^^^^^^^^^^^^^^^

``match_outcomes`` reproduces microsynth's multi-outcome ``match.out``: the soft
block :math:`\mathbf{L}_0` stacks the pre-period values of *every* listed
outcome, so one shared :math:`\mathbf{w}^\ast` balances every outcome's
trajectory simultaneously. The reported effect is for the primary ``outcome``;
running once per outcome with the same ``match_outcomes`` set yields the
*identical* weight vector and the package's per-outcome results table (the JSS
Table 2 reproduction in :doc:`replications/microsynth`).

Identifying assumptions (panel mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Assumption B1 (overlap / feasibility).* The treated-area covariate totals
:math:`\mathbf{h}` lie in the non-negative cone spanned by the control rows, so
the exact-balance constraint :math:`\mathbf{G}_0^{\!\top}\mathbf{w}=\mathbf{h}`
admits a :math:`\mathbf{w}\ge\mathbf{0}` solution. *Remark.* This is the
aggregate-SC convex-hull condition transposed to totals; when it fails CLARABEL
reports infeasibility and the fit raises rather than returning a biased
near-solution.

*Assumption B2 (pre-period fit / parallel trends).* Matching every pre-period
outcome total drives the treated and synthetic areas onto the same
pre-intervention trajectory, so absent treatment they would have moved in
parallel and :math:`\sum_j w_j^\ast y_{jt}` is a credible counterfactual for
:math:`t \in \mathcal{T}_2`. *Remark.* This is why the soft block should be the
full pre-window (``outcome_lag_periods`` :math:`= \mathcal{T}_1`); a
non-flat pre-period gap is the diagnostic that B2 is failing.

*Assumption B3 (regularisation selects a credible point).* Because the
counterfactual is not identified by the constraints (above), the reported effect
is the one implied by the maximum-ESS tie-break. *Remark.* The ridge is not a
nuisance knob to tune: with :math:`\rho` small the lagged-outcome fit and exact
covariate balance dominate, and the tie-break only chooses *among* equally
balanced, equally well-fitting weightings --- the most diffuse one, which
maximises effective sample size. Report the achieved imbalance (it is ~0) and
the ESS alongside the effect.

Propensity-score mode (``propensity_mode=True``)
------------------------------------------------

Setting ``propensity_mode=True`` reproduces microsynth's ``match.out=FALSE``
cross-sectional usage. The soft block is dropped
(:math:`\mathbf{L}_0` empty, :math:`\mathcal{L}\equiv 0`), so the program reduces
to the minimum-norm non-negative weighting that exactly balances the covariate
totals,

.. math::

   \mathbf{w}^\ast \in \operatorname*{argmin}_{\mathbf{w}\ge\mathbf{0}}\;
       \tfrac{\rho}{2}\|\mathbf{w}\|_2^2
       \quad\text{s.t.}\quad \mathbf{G}_0^{\!\top}\mathbf{w} = \mathbf{h} ,

and the data may be a single-period cross-section (no pre/post window
needed). The deliverable is the balancing weights themselves
(``res.donor_weights`` / ``res.design.w``): non-negative covariate-balancing
weights on the controls, summing to the treated count, that exactly match the
treated group's covariate totals --- usable as inverse-propensity-style weights
in a downstream analysis. The placebo-permutation test applies here too.

Inference
---------

``run_inference=True`` (the default) attaches a
:class:`~mlsynth.utils.microsynth_helpers.structures.MicroSynthInference`. The
method depends on the weight scheme: a paired stratified bootstrap for the
simplex mode, a placebo-permutation test for the panel/propensity mode. Both
populate ``res.inference`` with ``method``, ``att``, ``se``, ``ci``; the
permutation path additionally fills ``p_value``, ``p_values_by_period`` and
``test``.

Simplex mode --- paired stratified bootstrap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~mlsynth.utils.microsynth_helpers.inference.paired_bootstrap_ci` resamples
the treated and control blocks separately with replacement, preserving the
original :math:`(n_T, n_C)` allocation (a stratified, or "paired", bootstrap ---
pairing the two strata rather than resampling the pooled sample, which would
perturb the treated fraction). For replication
:math:`b = 1, \dots, B` (``n_bootstrap``):

#. draw :math:`n_T` treated rows and :math:`n_C` control rows i.i.d. uniformly
   with replacement;
#. refit the dual on the resampled controls and recompute the ATT
   :math:`\widehat{\tau}^{(b)}`.

Replications whose dual fails to converge are dropped; the surviving
:math:`B' \le B` estimates form the bootstrap distribution. The standard error
is its sample SD (``ddof=1``) and the CI is the percentile interval at
``ci_level`` (default 95%):

.. math::

   \widehat{\mathrm{se}} = \operatorname{sd}\bigl(\widehat{\tau}^{(b)}\bigr),
   \qquad
   \mathrm{CI}_{1-\alpha} = \Bigl[\,
       q_{\alpha/2}\bigl(\widehat{\tau}^{(b)}\bigr),\;
       q_{1-\alpha/2}\bigl(\widehat{\tau}^{(b)}\bigr)
   \Bigr].

Each rep is cheap --- the dual re-converges quickly from a cold start because it
is convex and :math:`(d+1)`-dimensional --- so ``n_bootstrap = 500`` on 100K
users and 20 covariates runs in the low minutes. Single-user weight
bootstrapping is deliberately *not* used: it would require re-standardisation
that complicates the comparison; block resampling is the standard ATT bootstrap
(Wang--Zubizarreta 2019) and matches the Robbins--Davenport reference.

Panel / propensity mode --- placebo permutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~mlsynth.utils.microsynth_helpers.panel_inference.panel_permutation_test`
ports microsynth's ``perm`` / ``test`` inference: the treated area is compared to
placebo areas drawn from the control pool. For
:math:`r = 1, \dots, R` (``n_permutations``):

#. sample :math:`n_T` controls uniformly without replacement as a placebo
   "treated area"; the remaining :math:`n_C - n_T` controls are the placebo donor
   pool;
#. refit the panel QP from the donor pool (same :math:`\rho`, same hard/soft
   blocks) and record the placebo per-period effects
   :math:`\tau_t^{(r)}` and placebo ATT
   :math:`\widehat{\tau}^{(r)} = |\mathcal{T}_2|^{-1}\sum_t \tau_t^{(r)}`.

Placebo groups whose QP is infeasible are skipped. The collection
:math:`\{\widehat{\tau}^{(r)}\}` is the null distribution against which the
observed ATT :math:`\widehat{\tau}` is ranked. The ``permutation_test`` tail sets
the p-value, with the add-one convention (so it is never exactly zero and is
valid as a finite-sample randomisation test):

.. math::

   p_{\text{lower}}    &= \frac{1 + \#\{r : \widehat{\tau}^{(r)} \le \widehat{\tau}\}}{1 + R}, \\
   p_{\text{upper}}    &= \frac{1 + \#\{r : \widehat{\tau}^{(r)} \ge \widehat{\tau}\}}{1 + R}, \\
   p_{\text{twosided}} &= \frac{1 + \#\{r : |\widehat{\tau}^{(r)}| \ge |\widehat{\tau}|\}}{1 + R}.

Per-period p-values (``p_values_by_period``) apply the same rule to each
:math:`\tau_t` against :math:`\{\tau_t^{(r)}\}`. The permutation SE is the SD of
the placebo ATTs, and the CI inverts the placebo distribution (which is centred
near zero under the sharp null) around the observed effect:

.. math::

   \widehat{\mathrm{se}} = \operatorname{sd}\bigl(\widehat{\tau}^{(r)}\bigr),
   \qquad
   \mathrm{CI}_{1-\alpha} = \Bigl[\,
       \widehat{\tau} - q_{1-\alpha/2}\bigl(\widehat{\tau}^{(r)}\bigr),\;
       \widehat{\tau} - q_{\alpha/2}\bigl(\widehat{\tau}^{(r)}\bigr)
   \Bigr].

.. note::

   Convention vs. the R package. microsynth's ``get.pval`` reports the bare
   fraction :math:`\#\{\cdot\}/R` (no add-one), so its floor is 0; mlsynth uses
   the add-one randomisation-test form, floor :math:`1/(1+R)`. The two agree on
   the conclusion --- on the Seattle DMI joint match both flag felonies,
   misdemeanors and total crime as significant reductions and drug crimes as
   not (:doc:`replications/microsynth`).

Cost scales as :math:`R` times one QP solve, so on a large control pool keep
``n_permutations`` modest; the placebo draws are independent and parallelise
trivially. Set ``n_permutations=0`` (or ``run_inference=False``) to skip
inference and return ``method="none"``.

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

Result Containers
-----------------

``MicroSynth.fit()`` returns a
:class:`~mlsynth.utils.microsynth_helpers.structures.MicroSynthResults` ---
MicroSynth is an *observational* estimator, so this is an ``EffectResult``-style
container (it reports a realised effect, not an experimental design). The surface
is grouped so each quantity has one home.

* ``res.att`` --- the scalar ATT :math:`\widehat{\tau}` (mean of the per-period
  effects); ``res.gap_trajectory`` --- the per-period vector :math:`(\tau_t)`
  over :math:`\mathcal{T}_2`; ``res.counterfactual`` --- the synthetic
  trajectory :math:`(\widehat{y}_{1t})`; ``res.gap`` --- the per-period
  contrast in result shape.
* ``res.design``
  (:class:`~mlsynth.utils.microsynth_helpers.structures.MicroSynthDesign`) ---
  the weighting: ``w`` (:math:`\mathbf{w}^\ast`), the dual variables
  ``dual_lambda`` / ``dual_nu``, the balance diagnostics ``smd_before`` /
  ``smd_after``, ``ess``, ``max_weight``, the ``feasible`` flag with
  ``feasibility_message``, and the solver ``converged`` / ``n_iterations``.
* ``res.inference``
  (:class:`~mlsynth.utils.microsynth_helpers.structures.MicroSynthInference`) ---
  ``method`` (``"paired_bootstrap"`` / ``"permutation"`` / ``"none"``), ``se``,
  ``ci``, the distribution ``bootstrap_atts`` (bootstrap or placebo), and ---
  for the permutation path --- ``p_value``, ``p_values_by_period`` and ``test``.
* ``res.donor_weights`` --- ``{control_id: w_j}`` for every control with
  positive weight; ``res.inputs`` --- the pre-processed matrices
  (:class:`~mlsynth.utils.microsynth_helpers.structures.MicroSynthInputs`).

.. autoclass:: mlsynth.utils.microsynth_helpers.structures.MicroSynthResults
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.microsynth_helpers.structures.MicroSynthDesign
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.microsynth_helpers.structures.MicroSynthInference
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

.. automodule:: mlsynth.utils.microsynth_helpers.panel_qp
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.panel_inference
   :members:
   :undoc-members:

Verification
------------

The panel method is cross-validated against the R ``microsynth`` package on the
Seattle Drug Market Intervention example — see :doc:`replications/microsynth`
(durable case ``benchmarks/cases/microsynth_seattle.py``).

.. automodule:: mlsynth.utils.microsynth_helpers.diagnostics
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.microsynth_helpers.plotter
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

- ITT (assignment-based): biased toward zero by contamination —
  treats contaminated holdouts as "control" even though they got
  ads.
- Naive TOT (impression-based, no balancing): biased upward by
  bidder selection — the actually-exposed users are positively
  selected on covariates that predict conversion.
- MicroSynth: takes impressions as the treatment indicator,
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
observed covariates, so balancing on them removes the bias.
Reality is rarely so kind. Suppose the thing that makes a held-out
user see the ad anyway -- latent purchase intent, in-market status --
is unobserved, and that same intent also lifts sales. Now the
actually-exposed users are positively selected on a confounder you
cannot put in the balancing constraint, and reweighting on age /
income only removes the slice of that selection the covariates happen
to explain. No amount of balancing recovers the truth from an
as-treated comparison, because the bias lives in a variable the
method never sees.

The decisive move is *not* to regroup users by what they received
(exposed vs. not) -- that is exactly what reintroduces the selection.
Keep users in their randomized arm and let MicroSynth balance for
precision, then either report the intent-to-treat (ITT) effect or
divide it by the compliance gap to recover the per-exposure effect
(a covariate-balanced Wald / CACE ratio):

.. math::

   \widehat\tau_{\text{ITT}}
     \coloneqq \frac{1}{N_1}\sum_{i:\,\text{assigned}=1} Y_i
       - \sum_{i:\,\text{assigned}=0} w_i Y_i,
   \qquad
   \widehat\tau_{\text{CACE}}
     \coloneqq \frac{\widehat\tau_{\text{ITT}}}
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

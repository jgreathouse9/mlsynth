Synthetic Difference-in-Differences (SDID)
==========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Difference-in-differences (DiD) and synthetic control (SC) are usually
pitched as tools for *different* problems. DiD is used when many units are
treated and you are willing to assume **parallel trends** -- that treated
and control outcomes would have moved in lockstep absent treatment, after
removing additive unit and time fixed effects. SC is used when *one* (or a
few) units are treated and parallel trends plainly fails, so you instead
**re-weight the donors** to match the treated unit's pre-treatment path.

Synthetic Difference-in-Differences (SDID), due to Arkhangelsky, Athey,
Hirshberg, Imbens and Wager (2021, *AER*) [aersdid]_, argues these two
strategies rest on closely related assumptions and combines the best of
both. It fits a two-way fixed-effects regression that is **doubly
weighted** -- by SC-style **unit weights** :math:`\omega_i` *and* DiD-style
**time weights** :math:`\lambda_t`:

.. math::

   (\hat\tau, \hat\mu, \hat\alpha, \hat\beta) =
   \arg\min_{\tau, \mu, \alpha, \beta}
   \sum_{i=1}^{N}\sum_{t=1}^{T}
   \bigl(Y_{it} - \mu - \alpha_i - \beta_t - W_{it}\tau\bigr)^2\,
   \hat\omega_i\, \hat\lambda_t .

The weights make the regression **local**: it leans on control units whose
*past* resembles the treated unit's, and on pre-periods that resemble the
post-period. Reach for SDID when:

* **DiD is tempting but pre-trends are not parallel.** SDID re-weights
  controls so their trend becomes *parallel* (not identical -- the unit
  fixed effects absorb level gaps) to the treated unit, then runs DiD on
  the re-weighted panel. It "automates" the usual practice of hunting for
  comparable units/periods to make parallel trends plausible, *with*
  statistical guarantees -- addressing the pre-testing concerns of Roth.
* **SC is tempting but the pre-fit is imperfect or you want valid
  inference.** Adding unit fixed effects (and an intercept in the weight
  problem) means the donors only need to be *parallel* to the treated
  unit, not match it exactly, and the design admits large-panel inference.
* **You want robustness without choosing.** Where DiD has been used, SDID
  is competitive with or better than DiD; where SC has been used, it is
  competitive with or better than SC. The weighting also often *improves
  precision* by removing predictable structure -- in the Prop 99 study,
  SDID's standard error (8.4) is smaller than DiD's (17.7) despite being
  the more flexible estimator.

.. note::

   The localization is not a free lunch: if outcomes have little
   systematic heterogeneity across units or periods, unequal weighting can
   *worsen* precision relative to plain DiD. SDID helps most when there is
   real structure (trends, levels) for the weights to exploit.

Do not use SDID when
^^^^^^^^^^^^^^^^^^^^^

* **Spillovers / interference contaminate the donor pool.** SDID assumes
  the controls are untreated and unaffected by the treatment (SUTVA). If
  treatment leaks to neighbours -- cross-border shopping, migration,
  geographic advertising -- the weighted controls are biased. Use
  :doc:`spsydid`, which separates the direct ATT from the spillover term.
* **Staggered adoption where you want partial pooling or an interactive
  fixed-effects guarantee.** SDID runs per cohort and averages, which is
  fine for an overall ATT, but it does not *pool* information across
  cohorts the way :doc:`ppscm` does, nor does it give the oracle-OLS
  efficiency of :doc:`seq_sdid`. Prefer those when cohorts are many and
  individual cohort fits are noisy.
* **The treated unit sits far outside the donor convex hull / the donor
  pool is huge and noisy.** SDID's unit weights are non-negative and
  (softly) sum-constrained; a treated path no linear convex combination
  can parallel will fit poorly. A factor-model estimator (:doc:`fma`) or a
  low-rank/denoising approach (:doc:`clustersc`, :doc:`mcnnm`) is better
  suited there.
* **A single treated unit, short panel, and you want the interpretable
  sparse convex-weight story** as the deliverable. Classic SC and its
  refinements (:doc:`tssc`, :doc:`fdid`, :doc:`scmo`) are more transparent;
  SDID's double weighting buys little when there is only one treated unit
  and no time structure for the time weights to exploit.
* **Distributional questions** (quantile effects, Lorenz, tails). SDID
  targets the mean ATT; use :doc:`dsc`.

What SDID Does in Practice
--------------------------

Beyond the econometrics: SDID answers "what would the treated unit have
done?" by building a synthetic comparison that is **parallel** to it, not a
clone, and by trusting the *recent, relevant* past more than the distant
past.

* **Policy / geo evaluation.** A state raises cigarette taxes (Prop 99); a
  city introduces congestion pricing; a country reunifies. You have a long
  panel of comparison regions whose levels differ wildly and whose
  pre-trends are not parallel. SDID re-weights the comparison regions to
  parallel the treated one and downweights ancient history that no longer
  looks like the policy window.
* **Marketing / pricing roll-outs.** A pricing change launches in some
  markets. Plain DiD over all markets is biased if the treated markets were
  on a different trajectory; pure SC ignores that fixed level differences
  are harmless. SDID handles both, and -- via time weights -- discounts
  pre-launch months that don't resemble the post-launch regime (seasonal
  shifts, a pre-launch promo).
* **Staggered roll-outs.** When units adopt at different dates, SDID runs
  per cohort and aggregates (Clarke et al., 2023), yielding both an overall
  ATT and a dynamic **event-study** path (Ciccia, 2024).

Notation
--------

Let :math:`Y_{it}` be the outcome of unit :math:`i` in period :math:`t`,
with :math:`i \in \{1, \dots, N\}` and :math:`t \in \{1, \dots, T\}`, and
let :math:`W_{it} \in \{0, 1\}` be the treatment indicator. The first
:math:`N_{co}` units are never-treated **controls** (donors); the remaining
:math:`N_{tr} = N - N_{co}` are **treated**, exposed after their adoption
period. :math:`T_{pre}` and :math:`T_{post}` count pre- and post-treatment
periods. The unit weights :math:`\omega_i` are supported on the controls
and the time weights :math:`\lambda_t` on the pre-period; :math:`\zeta` is
the unit-weight regularization parameter. The estimand is the average
treatment effect on the treated, :math:`\tau` (denoted :math:`\widehat{ATT}`
in aggregate).

.. admonition:: Notation bridge

   The mlsynth implementation generalizes the single-treated block design
   to **cohorts**: cohort :math:`a` is the set :math:`I^a` of units first
   treated in period :math:`a`, with size :math:`N_{tr}^a` and
   :math:`T_{tr}^a = T - a + 1` post-periods. The classical
   single-treated case (California) is the one-cohort special case, where
   the cohort ATT and the overall ATT coincide.

Assumptions
-----------

SDID's formal guarantees are developed under an **interactive
fixed-effects (latent factor) model** for the control potential outcome,

.. math::

   Y_{it} = \boldsymbol{\gamma}_i^\top \boldsymbol{v}_t + \tau W_{it} + \varepsilon_{it},

where :math:`\boldsymbol{\gamma}_i` are latent unit factors and
:math:`\boldsymbol{v}_t` latent time factors (a generalization of additive
:math:`\alpha_i + \beta_t` two-way fixed effects).

**Assumption 1 (latent factor outcome model).** The systematic part of the
outcome is :math:`\boldsymbol{\gamma}_i^\top \boldsymbol{v}_t`; deviations
:math:`\varepsilon_{it}` are mean-zero given the systematic component and
the treatment assignment.

*Remark.* This is strictly more general than DiD's additive
:math:`\alpha_i + \beta_t`. When the factor structure *is* additive, plain
DiD is already consistent; SDID is designed to also handle the interactive
case, where DiD is biased.

**Assumption 2 (selection on the systematic part only).** Treatment
assignment :math:`W` may depend on the latent factors
:math:`\boldsymbol{\gamma}_i, \boldsymbol{v}_t` (units are *not* randomized)
but **not** on the idiosyncratic error :math:`\varepsilon`.

*Remark.* This is what lets policies be adopted non-randomly -- California
was not a coin flip -- yet still be identified: the confounding must run
through the persistent latent structure that the weights and fixed effects
soak up, not through transitory shocks.

**Assumption 3 (weak cross-unit dependence).** The error vectors
:math:`\varepsilon_i` are independent *across units*, though correlation
*within a unit over time* is allowed.

*Remark.* Serial correlation within a unit is the norm in panel data and
is permitted; this is why the time-weight problem is left **unregularized**
(it must accommodate within-unit temporal correlation) while the
unit-weight problem is regularized. Cross-unit independence is what powers
the placebo variance estimator.

**Assumption 4 (weighted parallel trends, achieved by construction).**
There exist unit weights making the treated trajectory parallel to the
weighted control trajectory over the pre-period, and time weights making
each control's post-period mean a constant offset from its weighted
pre-period mean.

*Remark.* Unlike DiD -- which *assumes* parallel trends on the raw data --
SDID *constructs* weights to make parallel trends hold on the re-weighted
panel, then proceeds. The graphical "parallel trends" check is thus
performed on adjusted data, automatically and with guarantees.

Why Unit Weights and Why Time Weights
-------------------------------------

**Unit weights** are chosen so the treated unit's pre-treatment path is
*parallel* to the weighted-control path. Two differences from classical SC
(Abadie et al., 2010) make this work inside a fixed-effects regression:

1. an **intercept** :math:`\omega_0` is allowed, so the weights need only
   make trends *parallel* rather than coincident -- the unit fixed effects
   :math:`\alpha_i` absorb any constant level gap; and
2. a **ridge penalty** :math:`\zeta^2 \|\omega\|_2^2` is added (with
   :math:`\zeta = (N_{tr} T_{post})^{1/4}\hat\sigma`, :math:`\hat\sigma`
   the SD of first-differenced control outcomes) to disperse and uniquely
   pin down the weights.

**Time weights** are chosen so that, for the control units, the weighted
average of pre-treatment outcomes predicts the post-treatment average up to
a constant. The argument for them mirrors the argument for unit weights:
down-weighting pre-periods that look nothing like the post-period **removes
bias** and **improves precision**. This is the data-driven counterpart to
event-study practice, which implicitly puts all comparison weight on the
last pre-period -- SDID instead lets the data choose which pre-periods are
informative. The time-weight problem is left unregularized (Assumption 3).

Together, unit *and* time weights plus unit fixed effects make the DiD
contrast both more robust (it leans on comparable units and periods) and,
typically, more precise (predictable structure is removed), which is why
SDID's standard errors can be *smaller* than DiD's despite its added
flexibility.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`Y_{i, t}` denote the outcome of unit :math:`i` in period
:math:`t`, with :math:`i \in \{1, \dots, N\}` and
:math:`t \in \{1, \dots, T\}`. There are :math:`N_{co}` never-treated
control units, and the treated units are partitioned into cohorts by
their adoption period: cohort :math:`a` is the set
:math:`I^a \subseteq \{N_{co} + 1, \dots, N\}` of units that first
receive treatment in period :math:`a`. Let
:math:`A = \{a_1, \dots, a_K\}` denote the set of distinct adoption
periods, :math:`N_{tr}^a = |I^a|` the cohort size, and
:math:`T_{tr}^a = T - a + 1` the number of post-treatment periods in
cohort :math:`a`. Aggregate post-treatment exposure (Clarke et al.,
2023) is :math:`T_{post} = \sum_{a \in A} N_{tr}^a T_{tr}^a`.

The classical Arkhangelsky et al. (2021) SDID estimator targets a
single cohort. The mlsynth implementation runs that estimator
*per cohort*, accumulates the cohort-specific effects, and then
aggregates them in two complementary ways (Ciccia, 2024).

Cohort-Specific SDID (Equation 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a single cohort :math:`a`, SDID fits weights :math:`\omega_i` over
:math:`N_{co}` donor units and :math:`\lambda_t` over the cohort's
pre-treatment window :math:`t < a` by solving two convex programs:

.. math::

   \omega
   \;=\;
   \arg\min_{\sum \omega_i = 1,\ \omega_i \geq 0}
     \sum_{t = 1}^{a - 1}
       \left(
         \bar Y_{I^a, t}
         -
         \omega_0 - \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
       \right)^{\!2}
     + T_0 \zeta^2 \|\omega\|_2^2,

.. math::

   \lambda
   \;=\;
   \arg\min_{\sum \lambda_t = 1,\ \lambda_t \geq 0}
     \sum_{i = 1}^{N_{co}}
       \left(
         \bar Y_{i, [a, T]}
         -
         \lambda_0 - \sum_{t = 1}^{a - 1} \lambda_t Y_{i, t}
       \right)^{\!2},

where :math:`\bar Y_{I^a, t}` is the treated-unit mean at time
:math:`t`, :math:`\bar Y_{i, [a, T]}` is donor :math:`i`'s mean over
the post-treatment window, and :math:`\zeta` is a regularization
parameter scaled by the standard deviation of first-differenced donor
outcomes. The cohort-specific SDID estimator is then

.. math::

   \hat\tau_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{t = a}^{T}
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right)
   -
   \sum_{t = 1}^{a - 1} \lambda_t
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right).

This is Equation 2 of Ciccia (2024). Each cohort is fit independently
inside
:func:`mlsynth.utils.sdid_helpers.cohort.estimate_cohort_sdid_effects`.

Cohort-Specific Event Study (Equation 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cohort ATT is the average of a sequence of *dynamic* effects, one
per post-treatment offset :math:`\ell \in \{1, \dots, T_{tr}^a\}`:

.. math::

   \hat\tau_{a, \ell}^{\,sdid}
   \;=\;
   \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, a - 1 + \ell}
   \;-\;
   \sum_{i = 1}^{N_{co}} \omega_i Y_{i, a - 1 + \ell}
   \;-\;
   \sum_{t = 1}^{a - 1} \lambda_t
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} Y_{i, t}
       -
       \sum_{i = 1}^{N_{co}} \omega_i Y_{i, t}
     \right).

The first two terms are the *post-treatment gap* between the treated
cohort and its synthetic control at offset :math:`\ell`; the third
term is the time-weighted *pre-treatment baseline*. By construction,

.. math::

   \hat\tau_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{\ell = 1}^{T_{tr}^a} \hat\tau_{a, \ell}^{\,sdid},

i.e. the cohort ATT is the sample mean of its dynamic effects
(Equation 4 of Ciccia 2024). These effects are exposed on the result
object as :py:attr:`SDIDCohort.event_effects`.

Pooled Event Study (Equation 6)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`A_\ell = \{a \in A : a - 1 + \ell \le T\}` be the set of
cohorts for which the :math:`\ell`-th dynamic effect is computable,
and :math:`N_{tr}^\ell = \sum_{a \in A_\ell} N_{tr}^a` the
corresponding treated-unit count. The pooled event-study estimator is

.. math::

   \hat\tau_\ell^{\,sdid}
   \;=\;
   \sum_{a \in A_\ell}
     \frac{N_{tr}^a}{N_{tr}^\ell}
     \hat\tau_{a, \ell}^{\,sdid},

a treated-unit-weighted average of the cohort-specific dynamic effects.
This is the central quantity Ciccia (2024) recommends researchers
report. In the :mod:`mlsynth` API it is :py:attr:`SDIDEventStudy.tau`,
indexed by the corresponding event time on
:py:attr:`SDIDEventStudy.event_times`.

Overall ATT (Equation 7)
^^^^^^^^^^^^^^^^^^^^^^^^

Define :math:`T_{tr} = \max_{a \in A} T_{tr}^a`, the post-treatment
length of the earliest cohort. The overall ATT of Clarke et al. (2023)
admits the equivalent disaggregated form

.. math::

   \widehat{ATT}
   \;=\;
   \frac{1}{T_{post}} \sum_{\ell = 1}^{T_{tr}} N_{tr}^\ell \,
     \hat\tau_\ell^{\,sdid},

i.e. the average of the pooled event-study effects weighted by the
number of treated units contributing to each offset. This is
:py:attr:`SDIDInference.att`, with a placebo-based standard error and
confidence interval at :py:attr:`SDIDInference.se` /
:py:attr:`SDIDInference.ci`.

Placebo Inference
^^^^^^^^^^^^^^^^^

Variance estimation follows the placebo procedure of Arkhangelsky
et al. (2021), generalized to cohort and event-time effects by Clarke
et al. (2023). For each of :math:`B` iterations
(:py:attr:`SDIDConfig.B`), the donor pool is sampled to replace the
true treated units with pseudo-treated controls, the full SDID
pipeline is rerun, and the sample variance of the resulting effects
is taken as the variance of the actual estimator. The implementation
lives in
:func:`mlsynth.utils.sdid_helpers.inference.estimate_placebo_variance`.

The two-sided placebo p-value reported on
:py:attr:`SDIDInference.p_value` uses the canonical
:math:`((k + 1) / (B + 1))` correction, where :math:`k` is the count
of placebo iterations whose :math:`|\hat\tau^{\,*}_{att}|` is at least
as large as the observed :math:`|\widehat{ATT}|`.

Two-DataFrame and Single-Cohort Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the panel has a single treated unit (e.g., California in the
Proposition 99 study), :func:`mlsynth.utils.datautils.dataprep` returns
a single-treated payload rather than a cohorts dict. The
:func:`mlsynth.utils.sdid_helpers.setup.prepare_sdid_inputs` helper
unifies both shapes into a single ``cohorts_dict`` keyed by adoption
period *index* (1-based), which is what the cohort estimator's
``\ell = t - (a - 1)`` math requires. In the single-cohort case, the
cohort ATT and the overall ATT are numerically identical by
construction.

Core API
--------

.. automodule:: mlsynth.estimators.sdid
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SDIDConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.sdid_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.cohort
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.event_study
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sdid_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SDID

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df["Proposition 99"] = df["Proposition 99"].astype(int)

   results = SDID({
       "df":       df,
       "outcome":  "cigsale",
       "treat":    "Proposition 99",
       "unitid":   "state",
       "time":     "year",
       "B":        500,        # placebo iterations
       "display_graphs": True,
   }).fit()

   # Overall ATT (Ciccia 2024 Eq. 7) and placebo inference.
   print(results.inference.att)        # -15.605 (matches Arkhangelsky et al. 2021)
   print(results.inference.se)
   print(results.inference.ci)
   print(results.inference.p_value)

   # Pooled event-study trajectory (Ciccia 2024 Eq. 6).
   es = results.event_study
   for ell, tau, se in zip(es.event_times, es.tau, es.se):
       print(f"ell={int(ell):>3}  tau={tau:+.3f}  se={se:.3f}")

   # Per-cohort decomposition (Ciccia 2024 Eqs. 2 and 3).
   for adoption_period, cohort in results.cohorts.items():
       print(adoption_period, cohort.n_treated, cohort.att)
       print(cohort.event_effects[1])  # the first-period dynamic effect

Replication: Proposition 99
---------------------------

.. note::

   **Empirical replication (Path A).** Run on the California smoking panel
   (39 states, 1970-2000; California treated by Proposition 99 from 1989),
   ``mlsynth``'s SDID reproduces the headline estimate of [aersdid]_ **to
   three significant figures**:

   .. list-table::
      :header-rows: 1
      :widths: 30 24 24

      * - Quantity
        - mlsynth
        - Reference
      * - Overall ATT
        - **-15.605**
        - -15.6 (Arkhangelsky et al. 2021, Table 1; ``synthdid`` R: -15.604)
      * - Placebo SE (B = 500)
        - 7.58
        - 8.4 (placebo SE, Table 1)
      * - 95% CI
        - (-30.5, -0.7)
        -
      * - Placebo p-value
        - 0.032
        -

   The point estimate matches the authors' ``synthdid`` package
   (-15.604) essentially exactly. The placebo standard error is in the
   same range (7.6 vs. 8.4); it is a resampling estimate and varies with
   the placebo draw and ``B``. As Arkhangelsky et al. emphasize, SDID's
   -15.6 sits well below the DiD estimate (-27.3) and below SC (-19.6),
   and its SE is *smaller* than DiD's (17.7) -- the localization payoff.

   Per the project's replication contract
   (``agents/agents_estimators.md``), SDID is considered **done**: the
   published empirical ATT is reproduced on the same data to machine
   precision in the point estimate.

   **Cross-validation.** The same estimate is matched cell-for-cell to
   ``causaltensor.SDID`` (:math:`|\Delta| = 3.1\times 10^{-3}`) and pinned in
   ``benchmarks/cases/sdid_prop99.py``; see the dedicated page
   :doc:`replications/sdid`.

References
----------

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager,
S. (2021). "Synthetic Difference-in-Differences." *American Economic
Review* 111(12):4088-4118.

Ciccia, D. (2024). "A Short Note on Event-Study Synthetic
Difference-in-Differences Estimators." `arXiv:2407.09565
<https://arxiv.org/abs/2407.09565>`_.

Clarke, D., Pailanir, D., Athey, S., & Imbens, G. (2023). "Synthetic
difference in differences estimation." arXiv preprint.

Synthetic Difference-in-Differences (SDID)
==========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Difference-in-differences (DiD) and synthetic control (SC) are usually
pitched as tools for *different* problems. DiD is used when many units are
treated and you are willing to assume parallel trends -- that treated
and control outcomes would have moved in lockstep absent treatment, after
removing additive unit and time fixed effects. SC is used when *one* (or a
few) units are treated and parallel trends plainly fails, so you instead
re-weight the donors to match the treated unit's pre-treatment path.

Synthetic Difference-in-Differences (SDID), due to Arkhangelsky, Athey,
Hirshberg, Imbens and Wager (2021, *AER*) [aersdid]_, argues these two
strategies rest on closely related assumptions and combines the best of
both. It fits a two-way fixed-effects regression that is doubly
weighted -- by SC-style unit weights :math:`w_i` *and* DiD-style
time weights :math:`\lambda_t`:

.. math::

   (\widehat{\tau}, \widehat{\mu}, \widehat{\alpha}, \widehat{\beta}) =
   \operatorname*{arg\,min}_{\tau, \mu, \alpha, \beta}
   \sum_{i \in \mathcal{N}}\sum_{t \in \mathcal{T}}
   \bigl(y_{it} - \mu - \alpha_i - \beta_t - d_{it}\tau\bigr)^2\,
   \widehat{w}_i\, \widehat{\lambda}_t .

The weights make the regression local: it leans on control units whose
*past* resembles the treated unit's, and on pre-periods that resemble the
post-period. Reach for SDID when:

* DiD is tempting but pre-trends are not parallel. SDID re-weights
  controls so their trend becomes *parallel* (not identical -- the unit
  fixed effects :math:`\alpha_i` absorb level gaps) to the treated unit, then runs DiD on
  the re-weighted panel. It "automates" the usual practice of hunting for
  comparable units/periods to make parallel trends plausible, *with*
  statistical guarantees -- addressing the pre-testing concerns of Roth.
* SC is tempting but the pre-fit is imperfect or you want valid
  inference. Adding unit fixed effects (and an intercept in the weight
  problem) means the donors only need to be *parallel* to the treated
  unit, not match it exactly, and the design admits large-panel inference.
* You want robustness without choosing. Where DiD has been used, SDID
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

* Spillovers / interference contaminate the donor pool. SDID assumes
  the controls are untreated and unaffected by the treatment (SUTVA). If
  treatment leaks to neighbours -- cross-border shopping, migration,
  geographic advertising -- the weighted controls are biased. Use
  :doc:`spsydid`, which separates the direct ATT from the spillover term.
* Staggered adoption where you want partial pooling or an interactive
  fixed-effects guarantee. SDID runs per cohort and averages, which is
  fine for an overall ATT, but it does not *pool* information across
  cohorts the way :doc:`ppscm` does, nor does it give the oracle-OLS
  efficiency of :doc:`seq_sdid`. Prefer those when cohorts are many and
  individual cohort fits are noisy.
* The treated unit sits far outside the donor convex hull / the donor
  pool is huge and noisy. SDID's unit weights are non-negative and
  (softly) sum-constrained; a treated path no linear convex combination
  can parallel will fit poorly. A factor-model estimator (:doc:`fma`) or a
  low-rank/denoising approach (:doc:`clustersc`, :doc:`mcnnm`) is better
  suited there.
* A single treated unit, short panel, and you want the interpretable
  sparse convex-weight story as the deliverable. Classic SC and its
  refinements (:doc:`tssc`, :doc:`fdid`, :doc:`scmo`) are more transparent;
  SDID's double weighting buys little when there is only one treated unit
  and no time structure for the time weights to exploit.
* Distributional questions (quantile effects, Lorenz, tails). SDID
  targets the mean ATT; use :doc:`dsc`.

What SDID Does in Practice
--------------------------

Beyond the econometrics: SDID answers "what would the treated unit have
done?" by building a synthetic comparison that is parallel to it, not a
clone, and by trusting the *recent, relevant* past more than the distant
past.

* Policy / geo evaluation. A state raises cigarette taxes (Prop 99); a
  city introduces congestion pricing; a country reunifies. You have a long
  panel of comparison regions whose levels differ wildly and whose
  pre-trends are not parallel. SDID re-weights the comparison regions to
  parallel the treated one and downweights ancient history that no longer
  looks like the policy window.
* Marketing / pricing roll-outs. A pricing change launches in some
  markets. Plain DiD over all markets is biased if the treated markets were
  on a different trajectory; pure SC ignores that fixed level differences
  are harmless. SDID handles both, and -- via time weights -- discounts
  pre-launch months that don't resemble the post-launch regime (seasonal
  shifts, a pre-launch promo).
* Staggered roll-outs. When units adopt at different dates, SDID runs
  per cohort and aggregates (Clarke et al., 2023), yielding both an overall
  ATT and a dynamic event-study path (Ciccia, 2024).

Notation
--------

Let :math:`y_{it}` be the outcome of unit :math:`i` in period :math:`t`,
with units :math:`i \in \mathcal{N} \coloneqq \{1, \dots, N\}` and periods
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, 1-indexed, and let
:math:`d_{it} \in \{0, 1\}` be the treatment indicator. Unlike the
single-treated SC family, SDID admits several treated units, so there is no
distinguished :math:`i = 1`. The first :math:`N_{co}` units are
never-treated controls (donors); the remaining :math:`N_{tr} = N - N_{co}`
are treated, exposed after their adoption period. :math:`T_{pre}` and
:math:`T_{post}` count pre- and post-treatment periods. The unit weights
:math:`\mathbf{w} = (w_1, \dots, w_{N_{co}})^\top` are supported on the
controls and lie on the simplex
:math:`\Delta^{N_{co}} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_{co}} :
\|\mathbf{w}\|_1 = 1\}`; the time weights :math:`\boldsymbol{\lambda} =
(\lambda_1, \dots)^\top` are supported on the pre-period (Arkhangelsky et
al.'s :math:`\lambda`, kept distinct from the regularization symbols below).
:math:`\zeta` is the unit-weight regularization parameter,
:math:`\zeta = (N_{tr} T_{post})^{1/4}\,\widehat\sigma` with
:math:`\widehat\sigma` the standard deviation of the first-differenced control
outcomes (Arkhangelsky et al. 2021; the ``synthdid`` ``zeta.omega``). The
treated count :math:`N_{tr}` enters per cohort, so a block with several treated
units is regularized more strongly than a single-treated design on the same
panel; for one treated unit it reduces to :math:`(T_{post})^{1/4}\widehat\sigma`.
The optimisers are written :math:`\mathbf{w}^\ast` and
:math:`\boldsymbol{\lambda}^\ast`.
The estimand is the average treatment effect on the treated, :math:`\tau`
(denoted :math:`\widehat{ATT}` in aggregate).

.. admonition:: Notation bridge

   The mlsynth implementation generalizes the single-treated block design
   to cohorts: cohort :math:`a` is the set :math:`I^a \subseteq \{N_{co} +
   1, \dots, N\}` of units first treated in period :math:`a`, with size
   :math:`N_{tr}^a = |I^a|` and :math:`T_{tr}^a = T - a + 1` post-periods;
   :math:`A = \{a_1, \dots, a_K\}` collects the distinct adoption periods,
   and :math:`T_{post} = \sum_{a \in A} N_{tr}^a T_{tr}^a` is the aggregate
   post-treatment exposure (Clarke et al., 2023). The classical
   single-treated case (California) is the one-cohort special case, where
   the cohort ATT and the overall ATT coincide (and this aggregate exposure
   reduces to the post-period count :math:`T_{post}` above).

Assumptions
-----------

SDID's formal guarantees are developed under an interactive
fixed-effects (latent factor) model for the control potential outcome,

.. math::

   y_{it} = \boldsymbol{\gamma}_i^\top \mathbf{v}_t + \tau d_{it} + \varepsilon_{it},

where :math:`\boldsymbol{\gamma}_i` are latent unit factors and
:math:`\mathbf{v}_t` latent time factors (a generalization of additive
:math:`\alpha_i + \beta_t` two-way fixed effects).

Assumption 1 (latent factor outcome model). The systematic part of the
outcome is :math:`\boldsymbol{\gamma}_i^\top \mathbf{v}_t`; deviations
:math:`\varepsilon_{it}` are mean-zero given the systematic component and
the treatment assignment.

*Remark.* This is strictly more general than DiD's additive
:math:`\alpha_i + \beta_t`. When the factor structure *is* additive, plain
DiD is already consistent; SDID is designed to also handle the interactive
case, where DiD is biased.

Assumption 2 (selection on the systematic part only). Treatment
assignment :math:`d_{it}` may depend on the latent factors
:math:`\boldsymbol{\gamma}_i, \mathbf{v}_t` (units are *not* randomized)
but not on the idiosyncratic error :math:`\varepsilon`.

*Remark.* This is what lets policies be adopted non-randomly -- California
was not a coin flip -- yet still be identified: the confounding must run
through the persistent latent structure that the weights and fixed effects
soak up, not through transitory shocks.

Assumption 3 (weak cross-unit dependence). The error vectors
:math:`\varepsilon_i` are independent *across units*, though correlation
*within a unit over time* is allowed.

*Remark.* Serial correlation within a unit is the norm in panel data and
is permitted; this is why the time-weight problem is left unregularized
(it must accommodate within-unit temporal correlation) while the
unit-weight problem is regularized. Cross-unit independence is what powers
the placebo variance estimator.

Assumption 4 (weighted parallel trends, achieved by construction).
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

Unit weights are chosen so the treated unit's pre-treatment path is
*parallel* to the weighted-control path. Two differences from classical SC
(Abadie et al., 2010) make this work inside a fixed-effects regression:

1. an intercept :math:`w_0` is allowed, so the weights need only
   make trends *parallel*, not coincident -- the unit fixed effects
   :math:`\alpha_i` absorb any constant level gap; and
2. a ridge penalty :math:`\zeta^2 \|\mathbf{w}\|_2^2` is added (with
   :math:`\zeta = (N_{tr} T_{post})^{1/4}\widehat{\sigma}`, :math:`\widehat{\sigma}`
   the SD of first-differenced control outcomes) to disperse and uniquely
   pin down the weights.

Time weights are chosen so that, for the control units, the weighted
average of pre-treatment outcomes predicts the post-treatment average up to
a constant. The argument for them mirrors the argument for unit weights:
down-weighting pre-periods that look nothing like the post-period removes
bias and improves precision. This is the data-driven counterpart to
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

Using the cohort notation introduced above (:math:`I^a`,
:math:`N_{tr}^a`, :math:`T_{tr}^a`, the adoption-period set :math:`A`, and
the aggregate exposure :math:`T_{post}`), recall that the classical
Arkhangelsky et al. (2021) SDID estimator targets a single cohort. The
mlsynth implementation runs that estimator *per cohort*, accumulates the
cohort-specific effects, and then aggregates them in two complementary
ways (Ciccia, 2024).

Cohort-Specific SDID (Equation 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a single cohort :math:`a`, SDID fits unit weights :math:`\mathbf{w}` over
:math:`N_{co}` donor units and time weights :math:`\boldsymbol{\lambda}` over the cohort's
pre-treatment window :math:`t < a` by solving two convex programs:

.. math::

   \mathbf{w}^\ast
   \;=\;
   \operatorname*{arg\,min}_{\sum w_i = 1,\ w_i \geq 0}
     \sum_{t = 1}^{a - 1}
       \left(
         \bar y_{I^a, t}
         -
         w_0 - \sum_{i = 1}^{N_{co}} w_i\, y_{it}
       \right)^{\!2}
     + T_0\, \zeta^2 \|\mathbf{w}\|_2^2,

.. math::

   \boldsymbol{\lambda}^\ast
   \;=\;
   \operatorname*{arg\,min}_{\sum \lambda_t = 1,\ \lambda_t \geq 0}
     \sum_{i = 1}^{N_{co}}
       \left(
         \bar y_{i, [a, T]}
         -
         \lambda_0 - \sum_{t = 1}^{a - 1} \lambda_t\, y_{it}
       \right)^{\!2},

where :math:`\bar y_{I^a, t}` is the treated-unit mean at time
:math:`t`, :math:`\bar y_{i, [a, T]}` is donor :math:`i`'s mean over
the post-treatment window, and :math:`\zeta` is a regularization
parameter scaled by the standard deviation of first-differenced donor
outcomes. The cohort-specific SDID estimator is then

.. math::

   \widehat{\tau}_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{t = a}^{T}
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} y_{it}
       -
       \sum_{i = 1}^{N_{co}} w_i\, y_{it}
     \right)
   -
   \sum_{t = 1}^{a - 1} \lambda_t
     \left(
       \frac{1}{N_{tr}^a} \sum_{i \in I^a} y_{it}
       -
       \sum_{i = 1}^{N_{co}} w_i\, y_{it}
     \right).

This is Equation 2 of Ciccia (2024). Each cohort is fit independently
inside
:func:`mlsynth.utils.sdid_helpers.cohort.estimate_cohort_sdid_effects`.

Choosing the unit-weight penalty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The term :math:`T_0 \zeta^2 \|\mathbf{w}\|_2^2` in the unit-weight program is
the one free quantity in the two displays above, and to be explicit
about what sets it. Arkhangelsky et al. (2021) calibrate it to the noise the
weights are being asked not to chase,

.. math::

   \zeta \;=\; \bigl(N_{tr}\, T_{post}\bigr)^{1/4}\, \widehat\sigma,
   \qquad
   \widehat\sigma^{\,2}
   \;=\;
   \frac{1}{N_{co}(T_{pre} - 1) - 1}
   \sum_{i=1}^{N_{co}} \sum_{t=1}^{T_{pre}-1}
     \bigl(\Delta_{it} - \overline{\Delta}\bigr)^{2},
   \qquad
   \Delta_{it} = y_{i,t+1} - y_{it},

with :math:`\overline{\Delta}` the mean first difference over the same donor
block. Two things follow from the shape of :math:`\zeta`. It grows with the
volatility of the donors, so a noisy panel is pulled further toward equal
weights; and it grows in :math:`N_{tr} T_{post}`, so a design with many treated
units and a long post-period is regularized more strongly than a
single-treated, short-horizon one on identical outcomes. That is the default,
and it is what :math:`\zeta` means everywhere else on this page.

It is not, however, universal. A published SDID analysis may set the penalty
itself, and the value it sets is part of the specification, not a
detail: at :math:`\zeta = 0` the program is the unpenalised simplex least
squares, which fits the pre-period more closely and tends to put weight on
fewer donors, while as :math:`\zeta \to \infty` the objective is dominated by
:math:`\|\mathbf{w}\|_2^2` and :math:`\mathbf{w}^\ast` approaches the uniform
weights :math:`1/N_{co}`, i.e. plain difference-in-differences against the
donor average. The estimate therefore moves continuously between a synthetic
control and a DiD as the penalty rises, and reproducing someone else's number
means using their point on that path.

:py:attr:`SDIDConfig.zeta` supplies it. Left as ``None`` (the default) the
formula above is used, recomputed per cohort from that cohort's own donors and
horizon; given a number, that number is used for every cohort instead. Time
weights are untouched either way -- their penalty is a separate quantity that
SDID fixes at machine precision, for the reason given under Assumption 3.

.. code-block:: python

   # de Brabander, Juodis & Miyazato Szini (2025) run their Brexit study with
   # the unit-weight penalty switched off.
   res = SDID({
       "df": df, "outcome": "lgdp", "treat": "brexit",
       "unitid": "country", "time": "quarter",
       "zeta": 0.0,
   }).fit()

Cohort-Specific Event Study (Equation 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cohort ATT is the average of a sequence of *dynamic* effects, one
per post-treatment offset :math:`\ell \in \{1, \dots, T_{tr}^a\}`:

.. math::

   \widehat\tau_{a, \ell}^{\,sdid}
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

   \widehat\tau_a^{\,sdid}
   \;=\;
   \frac{1}{T_{tr}^a} \sum_{\ell = 1}^{T_{tr}^a} \widehat\tau_{a, \ell}^{\,sdid},

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

   \widehat\tau_\ell^{\,sdid}
   \;=\;
   \sum_{a \in A_\ell}
     \frac{N_{tr}^a}{N_{tr}^\ell}
     \widehat\tau_{a, \ell}^{\,sdid},

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
     \widehat\tau_\ell^{\,sdid},

i.e. the average of the pooled event-study effects weighted by the
number of treated units contributing to each offset. This is
:py:attr:`SDIDInference.att`, with a placebo-based standard error and
confidence interval at :py:attr:`SDIDInference.se` /
:py:attr:`SDIDInference.ci`.

Inference
^^^^^^^^^

Arkhangelsky et al. (2021) give three procedures for the variance of the ATT,
generalized to cohort and event-time effects by Clarke et al. (2023). The
:py:attr:`SDIDConfig.vce` option selects among them; the label used is recorded
on :py:attr:`SDIDInference.method`.

``placebo`` (the default, Algorithm 4)
  For each of :math:`B` iterations (:py:attr:`SDIDConfig.B`), a control unit is
  reassigned as a pseudo-treated unit and *removed from the donor pool*, the
  full SDID pipeline is rerun on the remaining controls, and the variance of the
  resulting placebo effects estimates the variance of the actual estimator.
  This is the only procedure defined for a single treated unit, and it is what
  the canonical Proposition 99 example uses. The implementation lives in
  :func:`mlsynth.utils.sdid_helpers.inference.estimate_placebo_variance`. The
  two-sided placebo p-value on :py:attr:`SDIDInference.p_value` uses the
  canonical :math:`((k + 1) / (B + 1))` correction, where :math:`k` counts the
  placebo iterations whose :math:`|\widehat\tau^{\,*}_{att}|` is at least as
  large as the observed :math:`|\widehat{ATT}|`.

  The :math:`B` draws are solved together. Each draw poses the same two weight
  programs on a different column subset of the same donor matrix, and nothing in
  that needs its own factorisation: the intercept is profiled out by centring,
  which is done column by column and so survives subsetting; the ridge
  :math:`T_{pre}\zeta^2` is folded in as extra rows carrying no target, so with
  the weights summing to one it enters as :math:`+\,T_{pre}\zeta^2 \mathbf{I}`
  even though :math:`\zeta` is recomputed for every draw. Section
  :ref:`sdid-batched-weights` sets out the reduction the batch rests on. All
  :math:`B` draws are then one call to an active set that certifies them
  together, which on Proposition 99 at the default :math:`B = 500` takes the fit
  from 1.77s to 0.75s. The draws are made in the order the one-at-a-time version
  made them, so the same controls are cast as pseudo-treated; the ATT is
  unchanged bit for bit, and the placebo standard error moves in its eleventh
  significant figure.

``jackknife`` (Algorithm 3)
  The fitted unit weights :math:`\widehat\omega` and time weights
  :math:`\widehat\lambda` are held fixed and each unit is left out in turn; the
  variance is the standard fixed-weights jackknife
  :math:`\tfrac{N-1}{N}\sum_i (\widehat{ATT}_{(-i)} - \overline{ATT})^2`. It is
  deterministic and fast (no re-solve of the weight problems), but is undefined
  when a cohort has a single treated unit -- leaving out the sole treated unit
  is undefined -- and returns ``NaN`` there, matching the ``synthdid`` R
  package.

``bootstrap`` (Algorithm 2)
  Units are resampled with replacement, degenerate all-treated or all-control
  resamples are discarded, and the full SDID estimate (weights re-fit) is
  recomputed on each resample; the variance is that of the resampled estimates.
  Like the jackknife it needs more than one treated unit and returns ``NaN``
  otherwise.

``noinference``
  Skips variance estimation; :py:attr:`SDIDInference.se`, the interval, and the
  p-value are ``NaN``.

The jackknife and bootstrap are implemented for the block (single adoption
period) design, matching ``synthdid``'s ``vcov.R``; a staggered-adoption panel
raises, directing you to the placebo procedure. For the jackknife and bootstrap
the p-value on :py:attr:`SDIDInference.p_value` is the asymptotic-normal
:math:`2\,(1 - \Phi(|\widehat{ATT}| / \widehat{se}))`, matching the confidence
intervals those methods construct.

The three methods are cross-validated against ``synthdid``: on a three-treated
block panel the deterministic jackknife reproduces the authors' R
value-for-value (:math:`10.557`), and the placebo and bootstrap match in
magnitude (they are stochastic, with independent RNG streams across the two
languages).

.. seealso::

   To check that the SDID effect is robust to the pretreatment horizon, run the
   Truncated History diagnostic (:doc:`truncated_history`), which re-estimates
   SDID on truncated pre-treatment windows. It reproduces the California
   Proposition 99 left-TH profile of Spoelstra et al. (2025) to the decimal.

.. _sdid-batched-weights:

How the Placebo Draws Are Solved Together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both SDID weight programs minimise a squared error over the simplex, and the
placebo procedure solves them :math:`B` times over. What makes the repetition
avoidable is that the simplex constraint removes the design matrix from the
problem. Writing the centred design as :math:`\mathbf{B}` and the centred target
as :math:`\mathbf{a}`, and using :math:`\mathbf{1}^\top\mathbf{w} = 1`,

.. math::

   \mathbf{B}\mathbf{w} - \mathbf{a}
   \;=\; \mathbf{B}\mathbf{w} - \mathbf{a}\,(\mathbf{1}^\top\mathbf{w})
   \;=\; (\mathbf{B} - \mathbf{a}\mathbf{1}^\top)\,\mathbf{w},

so with :math:`\mathbf{R} = \mathbf{B} - \mathbf{a}\mathbf{1}^\top` and
:math:`\mathbf{G} = \mathbf{R}^\top\mathbf{R}` the objective is the quadratic
form :math:`\mathbf{w}^\top \mathbf{G}\, \mathbf{w}`. Geometrically the weights
are the point of least norm in the convex hull of the columns of
:math:`\mathbf{R}` -- each column being one donor's discrepancy from the target
-- which is Wolfe's (1976) problem [wolfe1976]_, and an active set over the
donors solves it exactly and in finitely many steps.

A whole family of these is then carried by its :math:`\mathbf{G}` matrices
alone, which for the placebo draws are a submatrix of one Gram formed once plus
terms in the target and the ridge. The active set runs the family in lockstep:
each iteration is a single batched linear solve over the current supports, so
:math:`B` draws cost what the hardest single draw costs and not :math:`B` times
what the average one costs.

The reduction is not always available, and
:func:`mlsynth.utils.bilevel.minnorm.gram_reduction_is_safe` decides. Forming
:math:`\mathbf{G}` squares the design's condition number, which is free only
where the design has full column rank. SDID's designs are overdetermined --
pre-periods by donors for the unit weights, donors by pre-periods for the time
weights -- so they qualify, and the batched and one-at-a-time solvers return the
same weights and not merely the same fit. Each draw is checked, and one that
fails falls back to the one-at-a-time solve.

What that test is standing in for is whether the minimiser is a point or a face.
Where it is a face, every point of it is optimal and two exact solvers may
return different weights -- the same fit, a different donor table. Full column
rank rules that out, which is why the test is written on the shape; but it is
sufficient and not necessary, and the gap matters for panels with more donors
than pre-treatment periods. There the minimiser is usually still unique, because
the objective is flat along a direction only where that direction is feasible at
the solution, and synthetic-control solutions are too sparse for one to be.
:func:`mlsynth.utils.bilevel.minnorm.simplex_optimum_is_unique` settles it
exactly, on the support, once a solution is in hand.

Two-DataFrame and Single-Cohort Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the panel has a single treated unit (e.g., California in the
Proposition 99 study), :func:`mlsynth.utils.datautils.dataprep` returns
a single-treated payload, not a cohorts dict. The
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

.. automodule:: mlsynth.utils.sdid_helpers.covariates
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

.. note::

   ``SDID.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract: ``res.att`` / ``res.att_ci`` /
   ``res.counterfactual`` / ``res.gap`` / ``res.pre_rmse`` resolve through the
   standardized sub-models (the flat ``counterfactual`` / ``gap`` are the
   treated-unit-weighted aggregate across cohorts). The placebo inference, the
   pooled event study, and the per-cohort decomposition stay on
   ``res.inference_detail`` / ``res.event_study`` / ``res.cohorts`` (the bare
   ``res.inference`` slot is reserved for the standardized ATT-level
   :class:`~mlsynth.config_models.InferenceResults`).

.. automodule:: mlsynth.utils.sdid_helpers.structures
   :members:
   :undoc-members:

Synthetic triple difference (SC-DDD)
------------------------------------

Sometimes a plain difference across states is not enough. The treatment may
also be defined by a subgroup: in Virginia's 2008 HPV vaccine mandate, only the
adolescents who later aged into the 20-24 band were exposed, while older age
bands in the same state were not. A triple difference (DDD) compares the treated
state to control states and, within each state, the exposed subgroup to the
unexposed one. When parallel trends is doubtful across all three dimensions
(state, time, and subgroup), one wants the synthetic-control machinery applied
to that triple difference. This is the synthetic triple difference of Zhuang
([Zhuang2024]_).

The idea is a change of variable that reduces the triple difference to a
difference-in-differences. For each unit in the exposed (target) subgroup, the
outcome is demeaned by the non-target subgroup within the same
treatment-group-by-time cell,

.. math::

   W_{it} = Y_{it} - \bar Y_{\text{non-target},\, g(i),\, t},

where :math:`g(i)` is the unit's treatment-group indicator (1 if the unit is
ever treated, 0 otherwise) and :math:`\bar Y_{\text{non-target},\,g,\,t}` is the
mean outcome over the non-target subgroup rows in that group-by-time cell
(Zhuang 2024, following Olden and Møen 2022, who show a triple difference needs
only one parallel-trends assumption). A difference-in-differences on :math:`W`
recovers the triple-difference effect, so running SDID on :math:`W` over the
target subgroup gives the synthetic triple difference -- the counterfactual is a
weighted combination of control states, not a parallel-trends
extrapolation.

To switch this on, pass ``subgroup`` (the column naming the subgroup dimension)
and ``target_subgroup`` (the exposed value). The panel is then
``unit x subgroup x time``; SDID computes :math:`W` and collapses to the usual
``unit x time`` panel over the target subgroup. Everything else -- unit and time
weights, placebo inference, the event study -- is unchanged, and
``method_details.method_name`` reports ``"SDID-DDD"``.

.. code-block:: python

   import pandas as pd
   from mlsynth import SDID

   # Virginia HPV mandate: age 20-24 is the exposed subgroup, older bands are
   # the within-state controls; Virginia treated from 2016 (Feldman & Semprini).
   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/hpv_cervical_ddd.csv"
   )
   df["treated"] = ((df["state"] == "Virginia") & (df["year"] >= 2016)
                    & (df["age"] == "20-24")).astype(int)

   res = SDID({
       "df": df, "outcome": "cervix_adj", "treat": "treated",
       "unitid": "state", "time": "year",
       "subgroup": "age", "target_subgroup": "20-24",
       "display_graphs": False,
   }).fit()
   res.effects.att                       # +1.559 (SC-DDD)

Reach for the SC-DDD mode when the exposure has a within-unit subgroup structure
and you distrust parallel trends across the extra dimension; keep the ordinary
SDID (no ``subgroup``) when the treated/control split across units is all you
need. The transform needs at least one non-target subgroup value in every
treatment-group-by-time cell to demean by.

Time-varying covariates
-----------------------

SDID has no slot for control variables. Its whole design is a two-way
comparison -- unit effects and time effects -- and a covariate that moves within
a unit over time, such as a state's unemployment rate or a firm's headcount, has
nowhere to go. If that covariate also drives the outcome, it sits in the
residual the synthetic control is trying to match, and the estimate absorbs it.

Three answers exist in the literature and mlsynth implements all three. They are
different estimators, so the method is named explicitly, not inferred:
``covariates`` takes a dictionary keyed by ``"adjust"``, ``"optimized"`` or
``"match"``.

Write :math:`\mathbf{x}_{it} \in \mathbb{R}^{K}` for the covariate vector of
unit :math:`i` at time :math:`t`, and :math:`\mathcal{U} \coloneqq \{(i, t) :
d_{it} = 0\}` for the rows with no treatment in force -- every control unit in
every period, plus the treated units before adoption.

Adjusting the outcome (Kranz 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first answer removes the covariates from the outcome before the estimator
ever runs. Fit the two-way fixed-effects regression on :math:`\mathcal{U}`,

.. math::

   y_{it} = \alpha_i + \gamma_t + \mathbf{x}_{it}^{\top}\boldsymbol{\beta}
            + \varepsilon_{it},
   \qquad (i, t) \in \mathcal{U},

and subtract only the covariate part, evaluated on the whole panel:

.. math::

   \tilde y_{it} \;=\; y_{it} - \mathbf{x}_{it}^{\top}
   \widehat{\boldsymbol{\beta}},
   \qquad (i, t) \in \mathcal{N} \times \mathcal{T}.

Ordinary SDID then runs on :math:`\tilde y`. The weight programs never see a
covariate.

Three details decide the answer, and each is the opposite of a natural-looking
alternative.

The regression is fit on :math:`\mathcal{U}` but applied to
:math:`\mathcal{N} \times \mathcal{T}`. Fitting on untreated rows keeps the
treatment effect out of :math:`\widehat{\boldsymbol{\beta}}`; applying it
everywhere is what makes the adjustment useful, since the treated rows are the
ones the estimate depends on.

Only :math:`\widehat{\boldsymbol{\beta}}` is removed, not
:math:`\widehat\alpha_i` or :math:`\widehat\gamma_t`. The fixed effects stay in
the outcome because SDID constructs its own unit and time weights and handles
them itself; subtracting them here would give a different estimator, not a
cleaner one.

The covariates must vary within a unit over time. A covariate constant within
units is absorbed by :math:`\alpha_i`, and one constant across units at each
date is absorbed by :math:`\gamma_t`; either way it contributes nothing and
leaves the design rank deficient.

Optimising jointly with the weights (Arkhangelsky et al. 2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second answer does not estimate :math:`\boldsymbol{\beta}` first. It
estimates it at the same time as the weights, from one objective in
:math:`(\boldsymbol{\beta}, \boldsymbol{\omega}, \boldsymbol{\lambda})`. This is
footnote 4 of Arkhangelsky et al., and for reading applied work it
is the default in Stata's ``sdid``: a paper whose code says ``covariates(x)``
and nothing else used this, not the Kranz projection above.

Write :math:`\mathbf{Y}(\boldsymbol{\beta})` for the outcome with the covariates
removed, :math:`y_{it} - \mathbf{x}_{it}^{\top}\boldsymbol{\beta}`, and let
:math:`\bar y_{t}^{\,tr}(\boldsymbol{\beta})` be its mean over the treated units
and :math:`\bar y_{i}^{\,post}(\boldsymbol{\beta})` its mean over the
post-treatment periods. Both weight programs carry a free intercept,
:math:`a_{\omega}` and :math:`a_{\lambda}`. The joint objective is the sum of
the two weight criteria,

.. math::

   \ell(\boldsymbol{\beta}, \boldsymbol{\omega}, \boldsymbol{\lambda})
   \;=\;
   \frac{1}{T_{pre}} \sum_{t \le T_{pre}}
     \Bigl( \bar y_{t}^{\,tr}(\boldsymbol{\beta}) - a_{\omega}
       - \sum_{i \in \mathcal{C}} \omega_i \, y_{it}(\boldsymbol{\beta})
     \Bigr)^{2}
   \;+\;
   \frac{1}{N_{co}} \sum_{i \in \mathcal{C}}
     \Bigl( \bar y_{i}^{\,post}(\boldsymbol{\beta}) - a_{\lambda}
       - \sum_{t \le T_{pre}} \lambda_t \, y_{it}(\boldsymbol{\beta})
     \Bigr)^{2}
   \;+\; \zeta^{2} \lVert \boldsymbol{\omega} \rVert^{2},

minimised over :math:`\boldsymbol{\beta} \in \mathbb{R}^{K}` and the two weight
vectors on the simplex, with :math:`\zeta` the same ridge the covariate-free
estimator uses.

The intercepts are not decoration. Without them :math:`\boldsymbol{\beta}` can
lower the objective by shifting the level of a covariate, not by
explaining the outcome, and the minimiser runs off to wherever that shift is
largest.

Being the default does not make it the better choice. The authors of the Stata
package caveat their own default: it "has been observed to be problematic at
times (refer to Kranz (2022))", and it is sensitive to covariates with high
dispersion. Reach for ``optimized`` when the goal is to reproduce a published
specification that used it, and for ``adjust`` when you are choosing freshly.

Two properties of this estimator are easy to miss, and both are
easy to get wrong and one is genuinely surprising.

In a staggered design the coefficient is fitted separately for each adoption
cohort. A cohort has its own donor pool, its own pre-treatment window and its
own :math:`\zeta`, so :math:`\ell` is a different function for each; a single
panel-wide :math:`\boldsymbol{\beta}` would not be the minimiser of any of them.
Each cohort's fitted coefficient is available on its payload as
``optimized_beta``.

The reference implementation does not reach the minimum, and mlsynth reproduces
that instead of correcting it. ``synthdid`` alternates a Frank-Wolfe step on
each weight vector with a :math:`1/t` gradient step on
:math:`\boldsymbol{\beta}`. Since :math:`\sum_{t \le n} 1/t` grows like
:math:`\log n`, the total distance :math:`\boldsymbol{\beta}` can travel is
logarithmic in the iteration cap, and on real panels the cap binds while
:math:`\boldsymbol{\beta}` is still moving:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - iteration cap
     - 10
     - 100
     - 1000
     - 10000
     - 100000
   * - fitted :math:`\beta`
     - 0.074
     - 0.145
     - 0.213
     - 0.278
     - 0.339

One consequence stands on its own, because it is easy to assume the
opposite. The exact minimiser of :math:`\ell` is scale-equivariant: multiply a
covariate by :math:`c` and its coefficient divides by :math:`c`, leaving
:math:`\mathbf{x}^{\top}\boldsymbol{\beta}` and the estimate untouched. A
descent that stops early on a fixed schedule is not. The step is not scaled by
the curvature, so a covariate whose dispersion is far from the outcome's makes
the first step overshoot, and the iteration diverges instead of converging
slowly. mlsynth therefore scales each covariate to unit dispersion before
descending and undoes the scaling on the fitted coefficient. That is part of
reproducing the reference, not a numerical nicety layered on it: without
it, a panel whose income covariate has seventy times the outcome's dispersion
returns an estimate eleven orders of magnitude too large.

Because :math:`\ell` is very nearly flat in :math:`\boldsymbol{\beta}` -- on the
panel above it moves from 20.973 at :math:`\beta = 0` to 20.922 at its
minimum near :math:`\beta = 1.05`, a quarter of one percent -- this early stop
decides the answer. It acts as
shrinkage toward zero on a direction the data barely identify. Minimising
:math:`\ell` properly is a different estimator: it returns :math:`\beta = 8.4`
on one cohort of that panel and moves the ATT to 8.011, against the 8.051 every
published ``optimized`` result was computed with. So mlsynth fits
:math:`\boldsymbol{\beta}` with the reference's own iteration at its own default
cap, and then solves the weight programs exactly, as it does everywhere else.
This is the one place in the SDID implementation where a deliberately inexact
solver is ported, not replaced, and
:func:`mlsynth.utils.sdid_helpers.covariates.sdid_covariate_objective` is public
so the claim can be checked.

Matching on the covariates (de Brabander et al. 2025)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second answer leaves the outcome alone and puts the covariates inside the
unit-weight problem, in the manner of Abadie, Diamond and Hainmueller. SDID's
unit weights are already the demeaned synthetic control of Ferman and Pinto, so
collect the pre-treatment control outcomes in
:math:`\mathbf{Y} \in \mathbb{R}^{T_{pre} \times N_{co}}` with treated column
:math:`\mathbf{y} \in \mathbb{R}^{T_{pre}}`, and centre each series on its own
pre-period mean,

.. math::

   \ddot{\mathbf{Y}} \;=\; \bigl(\mathbf{I}_{T_{pre}}
     - \tfrac{1}{T_{pre}} \mathbf{1}_{T_{pre}} \mathbf{1}_{T_{pre}}^{\top}
     \bigr)\mathbf{Y},
   \qquad
   \ddot{\mathbf{y}} \;=\; \bigl(\mathbf{I}_{T_{pre}}
     - \tfrac{1}{T_{pre}} \mathbf{1}_{T_{pre}} \mathbf{1}_{T_{pre}}^{\top}
     \bigr)\mathbf{y}.

Summarise the covariates by their pre-period means, one value per unit per
covariate, giving :math:`\mathbf{Z} \in \mathbb{R}^{K \times N_{co}}` for the
controls and :math:`\mathbf{z} \in \mathbb{R}^{K}` for the treated unit. Let
:math:`\mathcal{M} \subseteq \{1, \dots, T_{pre}\}` be the periods the matching
uses, :math:`m = |\mathcal{M}|`, and stack

.. math::

   \mathbf{G} \;=\;
   \begin{bmatrix} \ddot{\mathbf{Y}}_{\mathcal{M}} \\[2pt] \mathbf{Z}
   \end{bmatrix} \in \mathbb{R}^{(m + K) \times N_{co}},
   \qquad
   \mathbf{g} \;=\;
   \begin{bmatrix} \ddot{\mathbf{y}}_{\mathcal{M}} \\[2pt] \mathbf{z}
   \end{bmatrix} \in \mathbb{R}^{m + K}.

Rows arrive on unrelated scales -- a log GDP deviation and an employment share
-- so let :math:`\mathbf{S} = \operatorname{diag}(s_1, \dots, s_{m+K})` hold
each row's standard deviation across the controls. The estimator is then a
nested pair of programs. Given a diagonal, non-negative
:math:`\mathbf{V} = \operatorname{diag}(v_1, \dots, v_{m+K})` on the simplex,
the inner problem is a weighted simplex least squares,

.. math::

   \mathbf{w}(\mathbf{V}) \;=\;
   \operatorname*{arg\,min}_{\mathbf{w} \in \Delta^{N_{co}}}\;
   \bigl(\mathbf{g} - \mathbf{G}\mathbf{w}\bigr)^{\top}
   \mathbf{S}^{-1} \mathbf{V} \mathbf{S}^{-1}
   \bigl(\mathbf{g} - \mathbf{G}\mathbf{w}\bigr),

and the outer problem chooses :math:`\mathbf{V}` by pre-treatment fit on the
outcome alone:

.. math::

   \mathbf{V}^{\ast} \;=\;
   \operatorname*{arg\,min}_{\mathbf{V}}\;
   \bigl\|\ddot{\mathbf{y}} - \ddot{\mathbf{Y}}\,\mathbf{w}(\mathbf{V})
   \bigr\|_{2}^{2},
   \qquad
   \mathbf{w}^{\ast} = \mathbf{w}(\mathbf{V}^{\ast}).

The asymmetry decides everything below: the
inner problem matches on :math:`\mathcal{M}`, but the outer problem always
scores :math:`\mathbf{V}` against the full pre-period :math:`\ddot{\mathbf{y}}`.

This program carries no ridge. SDID's :math:`\zeta` is calibrated on the
volatility of first-differenced outcomes, while the rows of
:math:`\mathbf{S}^{-1}\mathbf{G}` have unit variance by construction, so
:math:`T_{pre}\zeta^{2}` is not a penalty on this design at any rescaling. The
reference implementation reaches the same place from the other direction: it
takes :math:`\mathbf{w}^{\ast}` from an unpenalised program and passes
:math:`\zeta = 0`, reporting the penalised variant separately.

How many pre-periods to match on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice of :math:`\mathcal{M}` is not a tuning knob. Kaul, Klössner, Pfeifer
and Schieler (2022) show that when every pre-treatment outcome is a predictor --
:math:`\mathcal{M} = \{1, \dots, T_{pre}\}` -- the outer problem drives the
covariate rows of :math:`\mathbf{V}^{\ast}` to zero and the covariates are
irrelevant. The reason is visible in the two displays above: with all periods in
:math:`\mathbf{G}`, the inner problem already minimises the quantity the outer
problem scores, so any weight moved onto :math:`\mathbf{Z}` can only make the
outer fit worse.

On the Brexit panel the reference shows this is a gradient, not a switch.
Writing :math:`\pi = \sum_{r > m} v_r^{\ast}` for the share of
:math:`\mathbf{V}^{\ast}` on the covariate rows:

.. list-table::
   :header-rows: 1
   :widths: 26 16 20 24

   * - ``match_pre_periods``
     - :math:`m`
     - :math:`\pi`
     - pre-treatment loss
   * - ``"all"``
     - 86
     - 0.024
     - 3.11e-05
   * - ``"half"``
     - 43
     - 0.087
     - 3.79e-05
   * - ``"last"``
     - 1
     - 0.798
     - 8.53e-05

So the covariates buy influence with pre-treatment fit, and the trade is
explicit. ``match_pre_periods`` defaults to ``"last"``, the only setting under
which the covariates demonstrably bind; ``"half"`` takes the later half,
``"all"`` every period, and an integer :math:`m` the last :math:`m`.

Using them
~~~~~~~~~~

.. code-block:: python

   # residualise the outcome (Kranz); Stata's covariates(x, projected)
   res = SDID({..., "covariates": {"adjust": ["unemployment", "log_income"]}}).fit()

   # fit the coefficients with the weights; Stata's default, covariates(x)
   res = SDID({..., "covariates": {"optimized": ["log_gdp"]}}).fit()

   # match donors on covariates (de Brabander et al.)
   res = SDID({..., "covariates": {"match": ["gdp_pc"]},
                    "match_pre_periods": "last"}).fit()

   # compose: residualise for seasonality, match donors on income
   res = SDID({..., "covariates": {"adjust": ["seasonal"], "match": ["gdp_pc"]}}).fit()

To reproduce a Stata result, read its ``covariates()`` call: bare or with
``optimized`` maps to ``"optimized"``, and ``projected`` to ``"adjust"``.

The methods compose because they act on different objects -- ``adjust`` and
``optimized`` on :math:`y`, ``match`` on :math:`\mathbf{w}` -- though a column
may not appear under two keys at once, since residualising the outcome for a
variable and then matching donors on it counts the same variation twice.

Verification
~~~~~~~~~~~~

All three columns of Table 1 in Clarke, Pailanir, Athey and Imbens (2024) are
reproduced on the authors' quota panel, each to within the solver tolerance the
rest of this page documents:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - specification
     - Stata ``sdid``
     - mlsynth
     - difference
   * - no covariates
     - 8.034
     - 8.038
     - 0.004
   * - ``covariates(lngdp)``
     - 8.051
     - 8.048
     - 0.003
   * - ``covariates(lngdp, projected)``
     - 8.059
     - 8.054
     - 0.005

The residuals are the Frank-Wolfe versus exact-simplex difference described
under the weight solvers, not a difference in specification. Pinned in
``mlsynth/tests/test_sdid_optimized_covariates.py``.

A bare list is rejected. Before these options existed ``covariates`` meant the
Kranz adjustment, and silently reinterpreting it would change which estimator
runs; pass ``{"adjust": [...]}`` for that behaviour. Omitting ``covariates``
leaves every existing estimate unchanged. Neither method can be combined with
``subgroup``: SC-DDD collapses the panel over the subgroup dimension, leaving no
unit-by-time panel for either to be defined on.

Both methods are projections onto observed covariates, so both inherit the usual
caveat about controlling for variables that are themselves affected by the
treatment. Fitting on :math:`\mathcal{U}` guards against the treated units' own
response contaminating :math:`\widehat{\boldsymbol{\beta}}` or
:math:`\mathbf{z}`, but it cannot rescue a covariate that is a channel of the
effect, not a nuisance.

One structural requirement fails silently. Matching
needs the treated unit inside the convex hull of the controls on the matched
rows. Where it does not hold, the solution of the inner program sits at a vertex
of :math:`\Delta^{N_{co}}` and :math:`\mathbf{V}` cannot move it, so the
covariates are inert no matter how :math:`\mathcal{M}` is set -- not a numerical
failure, but the geometry of the simplex.

Verification
~~~~~~~~~~~~

The ``adjust`` path is checked against Kranz's ``xsynthdid`` at the fitted
coefficient and the adjusted outcome element-wise before the estimate, in
`test_sdid_covariates.py
<https://github.com/jgreathouse9/mlsynth/blob/main/mlsynth/tests/test_sdid_covariates.py>`_
against `benchmarks/reference/sdid_kranz/
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/reference/sdid_kranz/reference.R>`_.

The ``match`` path is checked against the authors' own ``Synth`` code on the
Brexit panel at :math:`\mathbf{w}^{\ast}` and :math:`\mathbf{V}^{\ast}`, not
the ATT -- their construction takes :math:`\mathbf{w}^{\ast}` from one fit
and the time weights from another, so an ATT comparison could not say which
moved. Correlation of the weight vectors is 0.998 under ``"last"``. See
`test_sdid_match_seam.py
<https://github.com/jgreathouse9/mlsynth/blob/main/mlsynth/tests/test_sdid_match_seam.py>`_
and `benchmarks/reference/brabander_sdid_match/
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/reference/brabander_sdid_match/reference.R>`_.

The estimator as a whole is checked against that paper's published results in
:doc:`replications/brabander_brexit`: all fourteen cells of its Table 1 and all
twenty-one of its Table 7 in-sample placebo, pinned by
`brabander_brexit_table1.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/brabander_brexit_table1.py>`_
and `brabander_brexit_insample.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/brabander_brexit_insample.py>`_.
Both need ``zeta=0`` and ``intercept_adjust=True``, which is what that paper's
specification is.

On staggered real data, SDID reproduces the ``synthdid`` numbers published by
Ronczewski (2026) for the cannabis-alcohol panel: each of the three adoption
cohorts to 2.3e-04 and the cell-count-weighted aggregate to 6.1e-05. The
paper assembles that aggregate by hand from three balanced blocks; handing
mlsynth the staggered panel in one call returns the same number. See `benchmarks/cases/ronczewski_cannabis.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/ronczewski_cannabis.py>`_.

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
       "B":        500,        # placebo / bootstrap resamples
       "vce":      "placebo",  # or "jackknife" / "bootstrap" / "noinference"
       "display_graphs": True,
   }).fit()

   # Overall ATT (Ciccia 2024 Eq. 7) and placebo inference.
   print(results.inference_detail.att)        # -15.605 (matches Arkhangelsky et al. 2021)
   print(results.inference_detail.se)
   print(results.inference_detail.ci)
   print(results.inference_detail.p_value)

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

   Empirical replication (Path A). Run on the California smoking panel
   (39 states, 1970-2000; California treated by Proposition 99 from 1989),
   ``mlsynth``'s SDID reproduces the headline estimate of [aersdid]_ to
   three significant figures:

   .. list-table::
      :header-rows: 1
      :widths: 30 24 24

      * - Quantity
        - mlsynth
        - Reference
      * - Overall ATT
        - -15.605
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
   (``agents/agents_estimators.md``), SDID is considered done: the
   published empirical ATT is reproduced on the same data to machine
   precision in the point estimate.

   Cross-validation. The same estimate is matched to the authors' own
   ``synthdid`` R package (:math:`|\Delta| = 1.6\times 10^{-3}`) and pinned in
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

Kranz, S. (2022). "Synthetic Difference-in-Differences with Time-Varying
Covariates." Working paper; implemented in the `xsynthdid
<https://github.com/skranz/xsynthdid>`_ R package. The two-step adjustment
behind SDID's ``covariates`` option.

.. [wolfe1976] Wolfe, P. (1976). "Finding the Nearest Point in a Polytope."
   *Mathematical Programming* 11:128-149. The minimum-norm-point active set the
   batched weight solve rests on.

.. [Zhuang2024] Zhuang, C. C. (2024). "A Way to Synthetic Triple
   Difference." `arXiv:2409.12353 <https://arxiv.org/abs/2409.12353>`_.
   The synthetic triple-difference construction behind SDID's ``subgroup``
   / SC-DDD mode; applied to Virginia's HPV vaccine mandate by Feldman &
   Semprini (2026), *Journal of Cancer Policy* 49:100777, whose SC-DDD
   estimate (+1.559) mlsynth reproduces
   (``benchmarks/cases/sdid_ddd_hpv.py``; see :doc:`replications/sdid`).

A practitioner's decision tree
==============================

.. currentmodule:: mlsynth

mlsynth ships dozens of estimators because each one is a named answer to a
specific complication that breaks the method before it. The trick to not
drowning is to start with the simplest credible method and escalate only when a
concrete complication forces you to. That is exactly how this page is organised:
a few identification gates first, then -- within each branch -- a ladder that
runs from the easy, canonical case to the specialised, harder ones.

Answer each question *yes* or *no*. A yes sends you to a method (or a short
list); a no moves you to the next question.

Reason forward: data, then estimand, then assumptions
-----------------------------------------------------

Before you open this tree, adopt the discipline that Baker, Callaway,
Cunningham, Goodman-Bacon and Sant'Anna call *forward engineering*
(*Difference-in-Differences Designs: A Practitioner's Guide*, 2025,
`arXiv:2503.13323 <https://arxiv.org/abs/2503.13323>`_). The temptation many
analysts face is to reverse-engineer: reach for a method because it sounds
powerful or modern, run it, and only then back out later after they discover
their assumptions were incorrect. Forward engineering goes the other way -- and
it is the right way:

#. What state are your data in? Take honest stock first: panel or a single
   series, how many treated units and whether they adopt at the same time,
   whether assignment was randomized, the length of the pre-period versus the
   number of donors (:math:`N` vs :math:`T_0`), missing cells, stationarity,
   plausible spillovers, the presence of covariates or an instrument.
#. Given a data structure, what estimand can those data actually support? Fix
   the *target parameter* before the estimator -- a mean ATT, a population ATE,
   a per-arm contrast, a quantile / distributional effect. Do not aim at a
   parameter your design cannot identify just because a method will return a
   number for it.
#. Which identifying assumptions are most defensible for that estimand, given
   that data state? Parallel trends? The convex-hull / no-extrapolation
   condition? No interference (SUTVA)? Exogeneity conditional on the latent
   factors? Proxy or instrument validity? Then, and only then, choose the method
   whose assumptions you can actually defend.

The questions below operationalize exactly that order as best as possible --
data state, then estimand, then assumptions. But one caveat matters more than
any single gate: this tree routes you to a method whose assumptions *match* your
answers; it does not *verify* those assumptions for you.

Each estimator encodes technical conditions a one-line summary cannot capture.
Follow the link at every leaf and read the original paper for the precise
assumptions, the inference theory, and the documented failure modes. Treat no
method as infallible: every estimate here is conditional on assumptions that are
*your* job to defend, not the software's.

.. note::

   The gates are a guide, not a strict partition: several methods answer more
   than one question, and a real problem can trip two at once.

At a glance
-----------

::

   GATE 0 — Identification pre-screen (answer these first)
   ────────────────────────────────────────────────────────
   Are you DESIGNING the experiment (treatment not yet assigned)?  ── yes ─► PART 3
   Is assignment RANDOMIZED?                  ── yes, many small units ─► difference-in-means
                                              └─ yes, few large units  ─► MUSC
   Do CONTROL UNITS exist at all?             ── no (everyone treated) ─► SHC
   Is treatment ENDOGENOUS (SC can't absorb)? ── have an instrument    ─► SIV
                                              └─ have proxies / NCs     ─► PROXIMAL
   Do parallel trends hold AND fixed-T/large-N? ── yes ─► plain Difference-in-Differences
                                                              (you may not need SC)
   …otherwise:  HOW MANY TREATED UNITS?  ── one ─► PART 1   └─ more than one ─► PART 2

   PART 1 — ONE treated unit   (easy ───────────────────────► hard)
   ─────────────────────────────────────────────────────────────────
   Start:  FDID   (or VanillaSC for the textbook simplex SC; TSSC for a pre-trends test)
     ↓ then escalate ONLY if one of these is true:
   Spillovers onto donors (SUTVA)?      ─► SPILLSYNTH · SpSyDiD (spatial) · SPOTSYNTH (unknown which) · ISCM (outside hull)
   Nonstationary / spurious trend?      ─► SBC · HSC
   Time-varying dynamics / heavy noise? ─► TASC · DSCAR · FMA
   Nonlinear outcome surface?           ─► NSC
   Donor pool N ≳ T0 (overfitting)?     ─► CLUSTERSC · SparseSC · PDA · RESCM · FSCM · BVSS
   Missing cells in the panel?          ─► SNN · MCNNM · RMSI (side information)
   Interpolation across dissimilar donors? ─► MASC
   Different ESTIMAND / treatment type? ─► DSC (dist.) · CTSC (dose) · SCMO (multi-outcome) · SI (arms)

   PART 2 — MANY treated units   (easy ─────────────────────► hard)
   ─────────────────────────────────────────────────────────────────
   Same adoption time?  ─► SDID            (micro units ─► MicroSynth; two-level ─► MLSC)
     + many treated at once, disaggregated/high-dim donors ─► MSQRT
   Staggered (different times)?  ─► SDID · ROLLDID (rolling-transformation DiD)
     + simplex SC per unit, never-treated pool, CFPT intervals ─► VanillaSC (staggered)
     + want pooling / oracle efficiency  ─► PPSCM · SequentialSDID
     + long pre-period, few never-treated, event study ─► SSC
     + spillovers                        ─► SpSyDiD
     + missing cells / gaps              ─► MCNNM

   PART 3 — DESIGNING an experiment   (by what you care about)
   ─────────────────────────────────────────────────────────────────
   Care only about the ATT (effect on the treated)? ─► SYNDES · SPCD · weakly-targeted MAREX
   Care about the ATE (population effect)?           ─► MAREX · LEXSCM
   Geo roll-out — every unit treated or control, no pure donors? ─► PANGEO (supergeo)
   Geo lift test — pick which markets to treat under a budget?    ─► GEOLIFT · MULTICELLGEOLIFT (multi-cell)


Gate 0 — Identification pre-screen
----------------------------------

These come first because they decide whether synthetic control is even the
right family. Get one wrong and no amount of donor weighting saves you.

Q0.1 · Are you designing the experiment? Has the treatment *not yet* been
assigned, and you are choosing whom to treat?

* Yes -- jump to Part 3 (experimental design).
* No -- the treatment already happened; continue.

Q0.2 · Is assignment randomized (or as-good-as-random)?

* Yes, and you have many small exchangeable units -- you do not need synthetic
  control; a difference-in-means (or a regression with controls) is unbiased.
* Yes, but only one or a few large aggregate units (markets, states) -- a single
  random draw can still leave baselines far apart. :doc:`musc` makes the effect
  *finite-sample unbiased under random assignment* and is the only estimator
  here with an unbiased finite-sample variance and exact randomization
  intervals.
* No -- continue.

Q0.3 · Do control units exist at all?

* No -- every unit is treated (a nationwide policy, a global shock like
  COVID-19), so there is no donor pool -- :doc:`shc` rebuilds the comparison
  from overlapping historical blocks of the treated unit's own series.
* Yes -- continue.

Q0.4 · Is the treatment endogenous in a way SC cannot absorb? This is the home
of the *proximal* methods: they are fundamentally tools for unmeasured
confounding / endogeneity, not for any particular outcome shape. The danger is
selection on *time-varying* unobservables -- the pre-fit can look perfect and
the ATT still be biased.

* You have a (partially valid) instrument -- a shift-share, a tariff schedule, a
  supply shock -- :doc:`siv` SC-debiases the (outcome, treatment, instrument)
  triple, then runs 2SLS; the instrument need only be valid *conditional on the
  factors*.
* You have valid proxies / negative controls -- extra controls associated with
  the latent confounder but with no direct path to the outcome -- :doc:`proximal`
  instruments the confounder via GMM (and also covers the single-proxy,
  doubly-robust, and surrogate variants).
* Neither -- selection is on the latent factors only (SC's standard premise) --
  continue.

Q0.5 · Do parallel trends hold, and are you in a fixed-T / large-N regime?

* Yes -- you may not need synthetic control at all: plain
  difference-in-differences is simpler and more powerful. The nearest
  in-library options are :doc:`sdid` or :doc:`seq_sdid`.
* No -- you are in SCM territory. Ask the master question:

Q0.6 · How many treated units?

* One -- go to Part 1.
* More than one -- go to Part 2.

Part 1 — A single treated unit
------------------------------

Begin with the simplest method that could work and escalate only when a named
complication applies.

Start here
^^^^^^^^^^

With one treated unit, a sharp intervention, and a scalar ATT, start with
:doc:`fdid` -- Forward DiD greedily selects the donors that share the treated
trend, needs no convex-hull assumption, and gives valid inference even under
nonstationarity, all with one estimated parameter. If you want the textbook
simplex synthetic control of Abadie and co-authors, that is :doc:`vanillasc`;
if the Forward Parallel Trends Assumption does not hold and you want a formal
pre-trends test, use :doc:`tssc`. If none of the escalations below applies, you
are done.

*FDID versus SCM.* Forward DiD is arguably simpler than synthetic control and, as
Li (2024) frames it, the natural first stop. Standard difference-in-differences
puts *equal* weights on *all* donors and so needs every donor to parallel-trend
with the treated unit -- usually too much to ask; synthetic control instead
solves for *convex* weights, which is more flexible but estimates one weight per
donor (and can overfit), requires the treated unit inside the donors' convex hull,
carries no intercept, and -- Li stresses -- relies on inference theory that does
not hold under nonstationarity of unknown structure. Forward DiD splits the
difference: it forward-*selects* the subset of donors that best matches the
treated unit's pre-period and then runs plain DiD on that subset. Each candidate
model has a *single* unknown parameter (the DiD intercept :math:`\alpha`), so
there is no overfitting however many donors there are; the intercept absorbs a
level gap (as in DiD, unlike SCM); the pre-period :math:`R^2` is a transparent fit
diagnostic; and the inference is valid for stationary *and* nonstationary data. So
when a selected subset of donors genuinely parallel-trends with the treated unit
-- a condition you can read off that :math:`R^2` -- Forward DiD is the simpler,
lower-variance choice and you need go no further. Escalate to :doc:`vanillasc`
when even the best equal-weighted subset cannot track the treated unit and you
need flexible convex weights to balance heterogeneous factor loadings: that is the
off-ramp back to SCM, taken only when FDID's parallel-trends-on-a-subset
assumption fails.

Now walk the escalations, easy to hard:

Q1.1 · Are your donors contaminated by the treatment (SUTVA / spillovers)?

* No -- next question.
* Yes, spatial with a known weight matrix W -- :doc:`spsydid`.
* Yes, an enumerable per-unit spillover set -- :doc:`spillsynth`.
* You don't know which donors are contaminated (a large pool, no a-priori
  validity knowledge, or a suspiciously-good-match donor) -- :doc:`spotsynth`,
  which screens each donor by a pre-intervention forecast test and excludes the
  ones that fail it.
* Yes, and the treated unit sits outside the donor hull (but is itself a useful
  donor for others) -- :doc:`iscm`.

*Which spillover method? (the beast).* mlsynth offers a whole family here because
spillover problems differ along three axes: whether you *know which* donors are
contaminated, whether the contamination is *spatial / network-structured*, and
whether the spillover is a *nuisance to purge* or an *effect to estimate*. Take
them in turn. When you do **not** know which donors are affected -- a large pool
with no a-priori validity knowledge, or a suspiciously good-fitting donor --
:doc:`spotsynth` (O'Riordan and Gilligan-Lee, 2025) *detects* it: a theorem from
proximal causal inference says a clean donor's post-treatment values are
forecastable from pre-treatment data alone, so a donor that fails that forecast
test has either changed regime or been hit by spillover; it screens out the
failures and bounds the residual bias by sensitivity analysis. When you **do**
know the affected set, the rest of the family (the :doc:`spillsynth` dispatcher,
plus :doc:`spsydid`) divides as follows. If the spillover is *spatial* with a
known weight matrix :math:`W`: :doc:`spsydid` (Serenini and Masek) extends
*synthetic* DiD, so it keeps SyDiD's intercept and time weights (a robust direct
effect under relaxed parallel trends) and splits the effect into direct and
indirect parts -- use it when you want the spillover *and* DiD-style robustness;
``method="sar"`` (Sakaguchi and Tagawa, 2026) instead models the outcomes with a
spatial-autoregressive process and does *Bayesian* inference (horseshoe priors),
the better choice in small samples (few units, short pre-period). If the spillover
is not spatial but its *structure* is specifiable (linear in unknown parameters),
``method="cd"`` (Cao and Dowd) estimates direct and spillover effects from *all*
units -- the one option that still works when *every* unit is contaminated, and it
ships a test for the assumed structure. If the spillover is itself a quantity of
*interest* and you have clean, far-away donors that still fit well,
``method="grossi"`` (Grossi et al., 2025) restricts the pool to the unaffected
units and estimates direct *and* spillover effects under partial interference.
Finally, if excluding the affected donors would *wreck* the pre-treatment fit --
the textbook case being a heavily weighted affected donor, like Austria's 42
percent in synthetic West Germany -- ``method="iscm"`` (Di Stefano and Mellace,
2024; also :doc:`iscm`) *keeps* those donors and nets out their intervention
effects through a small system of equations; Melnychuk's (2024) simulation study
finds iSCM the most accurate of the family, with his ``method="iterative"``
waterfall a close, simpler-to-implement second. In short: unknown affected set ->
``spotsynth``; spatial -> :doc:`spsydid` (robust direct effect) or ``sar``
(Bayesian, small samples); specifiable structure with everyone possibly hit ->
``cd``; spillover of interest with clean donors -> ``grossi``; affected donors
too good to drop -> ``iscm`` (or ``iterative`` for simplicity).

Q1.2 · Is the treated unit outside the donors' convex hull even without
spillovers (a true outlier)?

* No -- next question.
* Yes -- relax the simplex: :doc:`nsc` (affine weights), :doc:`rescm`
  (penalised / :math:`L_\infty`), or the unconstrained :doc:`pda` (the
  Hsiao--Ching--Wan panel-data regression and its modern penalised cousins).

*PDA versus SCM.* This convex-hull gate is exactly where the panel data approach
and synthetic control part ways, and the two camps argue it on their own terms.
Wan, Xie and Hsiao (2018) cast it as constrained versus unconstrained
regression: SCM forces convex weights (non-negative, summing to one) and no
intercept, whereas PDA leaves the weights free and adds an intercept that absorbs
a level (fixed-effect) gap between the treated unit and its donors. When SCM's
constraints hold -- the treated unit lies in the donors' convex hull and shares
their level -- they are *valid* restrictions and SCM is the more efficient
estimator; Gardeazabal and Vega-Bayo (2017) find it then gives a smaller,
tighter-spread post-treatment error (more so with covariates and a longer
pre-period) and is more robust to changes in the donor pool. When the constraints
are *invalid* -- no convex combination of donors reproduces the treated unit, or
a persistent level difference remains -- SCM is biased, while PDA's free weights
and intercept stay unbiased, and PDA's accuracy improves as the pre-period
lengthens. So prefer SCM when you have a genuine convex match; prefer :doc:`pda`
when the treated unit sits outside the hull or at a different level. With a large
donor pool the unconstrained regression must be regularised -- which PDA variant
to use is the next remark.

*Within PDA: which regulariser?* :doc:`pda` bundles four ways to fit the
unconstrained regression, and the choice is governed by the size of the donor
pool relative to the pre-period and by whether a few donors or many carry the
signal. The original Hsiao--Ching--Wan best subset (``method="hcw"``) picks the
donor subset by AICc; it is exact and certifiable but enumerates :math:`2^N`
candidate models and, being least squares, needs fewer donors than pre-periods
(:math:`N < T_0`) -- so it suits a *small* pool. When the pool is large,
best-subset becomes infeasible and HCW's fixed-:math:`N`, large-:math:`T`
asymptotics degrade. Li and Bell (2017) propose **Lasso** (``method="lasso"``)
for exactly this case: it allows more donors than pre-periods (:math:`N > T_0`),
is far cheaper than AICc/BIC, and lowers the ATE's predictive error -- use it when
a *sparse* handful of donors is plausibly relevant. Shi and Huang (2023) propose
**forward selection** (``method="fs"``): a sequence of OLS fits that approximates
best-subset at scale (valid even as :math:`N/T \to \infty`) and -- its headline
contribution -- supplies *valid post-selection inference* on the ATE (a
conditional :math:`t`-test), whether the underlying coefficients are sparse *or*
dense; use it when you want HCW-style selection with a defensible standard error
in a large pool. Shi and Wang (2024) **L2-relaxation** (``method="l2"``) targets
the opposite end of the sparse--dense axis: when the donors share a latent-factor
structure so that *all* of them are weakly relevant and no sparse few stand out,
its dense weighting diversifies prediction risk and attains oracle accuracy. In
short: small pool -> ``hcw``; large and sparse -> ``lasso`` (or ``fs`` when you
also want inference); large and dense -> ``l2``.

Q1.3 · Is the outcome nonstationary, so a tight pre-fit might be a *spurious*
match?

* No -- next question.
* Yes -- decompose first: :doc:`sbc` (Hamilton trend/cycle split, match the
  cycle) or :doc:`hsc` (soft levels-vs-differences allocation).

*Spurious fit on nonstationary data -- SBC versus HSC versus plain SC.* Both
papers behind this gate warn that the standard synthetic control of Part 1 is
unsafe on nonstationary macro series: a convex combination of donors can track
the treated unit's pre-period through *coincidental* co-movement of unit-specific
stochastic trends, giving an excellent in-sample fit with no out-of-sample
validity -- the *spurious synthetic control* problem (Shi, Xi and Xie, 2025; Liu
and Xu, 2026), an instance of spurious regression that, crucially, does *not* go
away as the pre-period lengthens. So plain :doc:`vanillasc` is appropriate only
when the outcome is stationary (or its stochastic trend is genuinely shared
across units). The two fixes differ in how much they assume. :doc:`sbc` (Shi, Xi
and Xie, 2025) commits to a trend/cycle split: it treats the nonstationary
*trend* as unit-specific and forecasts it from the treated unit's own history,
and uses the donors *only* for the common business *cycle* -- the right division
when comovement genuinely lives in the cycle (a country's idiosyncratic growth
path plus a synchronised global cycle). :doc:`hsc` (Liu and Xu, 2026) refuses to
commit, because whether the stochastic trend is *shared* (which SC should keep for
matching) or *idiosyncratic* (which it must remove) is usually unknown ex ante,
and a binary choice to difference or not fails in whichever regime is wrong. HSC
instead *softly allocates* between donor matching and a treated-unit-specific
self-forecast, with a tuning parameter chosen by rolling-origin cross-validation
that interpolates continuously between SC on differenced outcomes and SC on raw
outcomes with a trend; it adapts across regimes where an estimator fixed to one
can fail. So prefer :doc:`sbc` when you are confident the matchable comovement is
the business cycle and the trend is the treated unit's own; prefer :doc:`hsc`
when you are unsure whether the nonstationarity is shared or idiosyncratic and
want the data to decide.

Q1.4 · Are there persistent latent factors / time-varying dynamics / heavy
observation noise?

* No -- next question.
* Yes -- :doc:`tasc` (time-aware state-space model) or :doc:`fma` (PC factors
  with a residual-bootstrap test). (For micro panels with observed time-varying
  *confounders* and autoregressive outcomes, :doc:`dscar` is a different
  paradigm -- see the remark below.)

*DSCAR -- a different beast.* :doc:`dscar` (Zheng and Chen, 2024) is not a variant
of the synthetic control above; it is best understood by contrast with the vanilla
method. Standard SC builds *fixed* weights that match the treated unit's whole
pre-treatment outcome *trajectory*, identifies the counterfactual through a latent
factor model and the convex hull, and is built for a single aggregate treated unit
with no time-varying confounders. DSCAR changes nearly all of that. It is designed
for *micro-level* panels -- many units, often many treated ones (monitoring sites,
wearables, individuals) -- in which the outcome is strongly *autoregressive*, the
confounders are *time-varying* and observed, and the units are *spatially
dependent*. Instead of one fixed weight vector it constructs *dynamic*
(time-varying) weights by maximising an empirical likelihood subject to matching
the *current* state of the time-varying confounders and the lagged outcome at each
period; because the match is to the current confounder state rather than to a long
pre-treatment path, an exact match is attainable with probability approaching one.
And it identifies the effect through *unconfoundedness* conditional on the
covariates and the lagged outcome -- a selection-on-observables assumption testable
on the pre-period -- rather than SC's factor structure. So prefer :doc:`dscar` over
:doc:`vanillasc` when the data are micro-level with observed time-varying
confounders, autocorrelated outcomes, spatial dependence, or multiple treated
units; stay with the vanilla synthetic control when you have a single aggregate
treated unit, time-invariant structure, and the factor-model / convex-hull premise
is what you are willing to assume.

Q1.5 · Is the untreated outcome a nonlinear function of the predictors?

* No -- next question.
* Yes -- :doc:`nsc`.

Q1.6 · Is the donor pool large relative to the pre-period (N >> T0)? This is the
most common reason to leave the standard workhorses: unrestricted fits overfit
the pre-period and predict the post-period worse.

* No -- next question.
* Yes -- escalate, roughly easiest first: :doc:`fscm` (tune the donor *count*),
  :doc:`sparse_sc` (L1 predictor/covariate selection), :doc:`pda` (the Panel Data
  Approach: L2-relaxation, Lasso, Shi--Huang forward selection, or the original
  Hsiao--Ching--Wan best-subset regression -- ``method="hcw"``, the classic
  unconstrained PDA), :doc:`rescm` (one program from simplex SC to
  :math:`L_\infty` to DiD), :doc:`clustersc` (denoise + cluster donors), or
  :doc:`bvss` (Bayesian spike-and-slab with a soft simplex).

*Dense versus sparse weights -- when to relax SCM.* Standard synthetic control
constrains the weights to the simplex, which (as Doudchenko and Imbens (2016)
observe) tends to produce *sparse* solutions loading on a handful of donors. Two
recent papers argue this sparsity is a mechanical byproduct of the optimisation
rather than a virtue, and motivate the :doc:`rescm` family -- a relaxed program
spanning simplex SC, the :math:`L_\infty` (dense) norm, and DiD. Liao, Shi and
Zheng (2025) note that once you have invested in a large donor pool there is
often no reason to believe only a few controls are relevant: a *dense* scheme
that uses them all -- spreading weight evenly within latent donor groups --
diversifies prediction risk and reaches oracle accuracy even when the donor count
exceeds the pre-period (:math:`J \gg T_0`). Wang, Xing and Ye (2025) make the same
case via the :math:`L_\infty` norm: concentrating weight on a few units amplifies
sensitivity to their idiosyncrasies (higher variance, and bias if one key donor
deviates), which bites hardest in volatile environments or when control units
differ in dynamics; their dense scheme reduces that over-reliance, raises the
chance of satisfying parallel trends, and -- like DiD -- admits a level intercept
and valid long-panel asymptotics. So prefer :doc:`rescm` over standard SCM when
the donor pool is large and broadly relevant, when robustness to any single donor
matters, or when the outcome is volatile; keep the sparse, transparent standard
SC when a few genuinely similar donors match well in a stable setting (the
canonical Proposition 99 case).

Q1.7 · Are there missing cells in the panel?

* No -- next question.
* Yes, informative (MNAR) with a fully observed anchor block -- :doc:`snn`.
* Yes, arbitrary sparse / low-rank -- :doc:`mcnnm`.
* Block-missing (treated cells), and you have unit/time covariates -- :doc:`rmsi`
  exploits side information on both margins (a four-component sieve +
  nuclear-norm completion) to impute the treated counterfactual; it reduces to a
  low-rank completion when the covariates are uninformative.

Q1.8 · Is your estimand or treatment effect non-standard (not a scalar mean ATT
for one binary treatment)?

* No -- you are done; use the *Start here* method.
* The whole distribution (quantiles, Lorenz, tails) -- :doc:`dsc`.
* A continuous or multi-valued dose with no clean control -- :doc:`ctsc`.
* Several related outcomes (helps most with a short pre-period) -- :doc:`scmo`.
* Several distinct intervention arms to compare -- :doc:`si`.

Q1.9 · Are you worried about interpolation bias -- the synthetic control having
to *interpolate* across donors that are individually far from the treated unit,
so the fit blends dissimilar units?

* No -- you are done; use the *Start here* method.
* Yes -- :doc:`masc` blends extrapolation-free nearest-neighbour matching with
  the SC simplex and chooses the mix that minimises estimated bias, directly
  targeting interpolation bias.

Part 2 — Many treated units
---------------------------

The base case for multiple treated units is :doc:`sdid` -- Synthetic
Difference-in-Differences, doubly weighted by unit *and* time weights -- which
works whether adoption is simultaneous or staggered and degrades gracefully when
parallel trends or exact matching fail. Escalate from there.

Q2.1 · Do all treated units adopt at the same time?

* Yes -- :doc:`sdid` is the base case. If the units are individual / micro-level
  (thousands to millions), use :doc:`microsynth`; if treatment is assigned at a
  coarse level but outcomes are observed at a finer level (state policy, county
  outcomes), use :doc:`mlsc`.
* Yes, and many units are treated at once with a high-dimensional donor pool
  (disaggregated stores / ZIPs / geos, donor count :math:`n \gtrsim T_0`) --
  :doc:`msqrt` pools the treated units into one matrix regression and selects
  donors with a pivotal square-root-lasso penalty, borrowing strength across
  treated units instead of fitting each noisily on its own.
* No -- adoption is staggered -- continue.

Q2.2 · Staggered: do you just want the overall / event-study ATT?

* Yes, just staggered -- :doc:`sdid` (per-cohort + aggregate) is the simplest. A
  lightweight alternative is :doc:`rolldid`, the Lee--Wooldridge
  rolling-transformation DiD, which rolls each cohort's pre/post contrast
  forward and pools them. If you want partial pooling across cohorts or oracle
  efficiency under interactive fixed effects, escalate to :doc:`ppscm` or
  :doc:`seq_sdid`.
* Staggered, long pre-period, few/no never-treated units, and you want an event
  study without parallel trends -- :doc:`ssc` (Staggered Synthetic Control). It
  builds each unit's synthetic control from all other units (not-yet-treated
  included), so it needs no never-treated pool, and gives event-time ATTs with
  Andrews end-of-sample inference. Best with a long pre-period (large :math:`T`,
  moderate :math:`N`).
* Staggered, with a clean never-treated donor pool, and you want the *simplex*
  synthetic-control answer -- one synthetic control per treated unit, on its own
  pre-period -- with the full Cattaneo--Feng--Palomba--Titiunik causal
  predictands (per-unit ATTs, the event-time average effect, the overall ATT)
  and their prediction intervals -- :doc:`vanillasc` handles staggered adoption
  natively. It detects the multiple treated units from the treatment column,
  fits each on the never-treated donors and aggregates, with or without
  covariate (multi-feature) matching, reproducing the ``scpi`` package. The
  choice between it and the :doc:`sdid` base case follows the two methods' own
  arguments. Arkhangelsky et al. (2021) motivate SDID by its unit fixed effects,
  which match cohorts on pre-treatment *trends* rather than levels -- absorbing a
  constant level gap, and so deliberately loosening synthetic control's
  requirement that the treated unit lie inside the donors' convex hull -- plus
  time weights that downweight uninformative pre-periods. Cattaneo, Feng, Palomba
  and Titiunik (2025) build instead on the canonical *convex* (simplex) SC, which
  does not extrapolate, and contribute non-asymptotic prediction intervals whose
  guarantees hold in the small samples SC applications typically have. So prefer
  :doc:`sdid` when no convex combination of donors can match a cohort's *level*
  (you need the intercept to absorb the gap) or some pre-periods are
  unrepresentative; prefer staggered :doc:`vanillasc` when the simplex fit
  already tracks the cohorts on levels -- keeping interpretable, non-extrapolating
  weights and the finite-sample CFPT intervals, with no DiD intercept or
  reweighted periods.
* Staggered *and* spillovers onto donors -- :doc:`spsydid`.
* Staggered *and* missing cells / gaps -- :doc:`mcnnm` (matrix completion handles
  staggered missingness natively).

*Which staggered synthetic control?* Three SC-family methods target this same
setting on different arguments, and the choice turns on your donor pool, sample
length, and estimand. Staggered :doc:`vanillasc` (Cattaneo, Feng, Palomba and
Titiunik, 2025) fits the canonical *convex* SC for each treated unit on a
*never-treated* donor pool and is built around *non-asymptotic* (finite-sample)
prediction intervals -- the authors stress these precisely because SC
applications usually have small samples; reach for it when you have clean
never-treated donors, want non-extrapolating convex weights, and need valid
intervals even with a short pre-period. :doc:`ppscm` (Ben-Michael, Feller and
Rothstein, 2022) instead targets the *average* effect across treated units: it
shows that fitting weights separately per unit (good unit fits, poor average) or
pooling them (good average, poor unit fits) each biases one of the two
imbalances, and proposes *partially pooled* SCM that trades them off, with a
de-meaning (intercept) step that turns it into a weighted
difference-in-differences; reach for it when the average ATT is the headline
estimand and a level shift between treated units and donors must be absorbed.
:doc:`ssc` (Cao, Lu and Wu, 2026) drops the never-treated requirement
altogether -- it builds each unit's control from *all other units,
not-yet-treated included*, explicitly because methods that lean on never-treated
units deteriorate when those are scarce -- and bases inference on Andrews'
end-of-sample test under large-:math:`T` asymptotics; reach for it when most
units are eventually treated, the pre-period is long, and you want event-time ATT
inference without parallel trends. The distinct case of *many* treated units in a
high-dimensional, disaggregated panel adopting at the *same* time (Q2.1) is
:doc:`msqrt` (Shen, Song and Abadie, 2025): it pools the treated units into one
matrix regression with a tuning-free square-root-lasso, chosen for computational
efficiency and to preserve *individual* counterfactuals where fitting each unit
separately is slow and aggregating them would blur unit-level effects or add
interpolation bias.

Part 3 — Designing an experiment
--------------------------------

You are choosing *whom* to treat, not estimating an effect; these return
assignments and power / MDE curves, not ATTs. Order by what estimand you care
about, easiest target first.

Q3.1 · Do you only care about the ATT (the effect on the treated units)?

* Yes -- :doc:`syndes` (a MIP that minimises the *ATT estimator's* MSE, exactly
  :math:`K` treated) or :doc:`spcd` (a fast spectral phase-synchronisation
  design). A weakly-targeted :doc:`marex` design can also be pointed at the
  treated set if you want a convex design that leans ATT-ward.

Q3.2 · Do you care about the population ATE (a population-level contrast, not
just the treated)?

* Yes -- base :doc:`marex` matches synthetic-treated and synthetic-control units
  to *population* predictor means; :doc:`lexscm` adds an explicit power /
  minimum-detectable-effect report (validity *then* power) and budget / cost
  constraints when only a subset is eligible.

Q3.3 · Must every unit end up either treated or control -- no pure-donor pool
left over -- as in a geo roll-out?

* Yes -- :doc:`pangeo` groups geos into balanced *supergeos*, trims no unit, and
  matches on the full pre-period *trajectory* for a downstream
  difference-in-differences read.

Q3.4 · Are you planning a marketing geo-lift test -- pick which markets to treat
so the untreated markets form a clean control, often under a budget?

* Yes, one treatment cell -- :doc:`geolift` selects the treated markets and
  reports the design's power / minimum detectable lift.
* Yes, several treatment cells (e.g. testing more than one creative or spend
  level at once) -- :doc:`multicellgeolift` extends the selection to multiple
  cells simultaneously.

Q3.5 · Designing across groups (regions / arms / strata)? Three mechanisms look
similar but are distinct -- pick by what you want, not by the word "group":

* A *separate experiment per group* -- its own estimand and donor pool:
  :doc:`marex`'s ``cluster`` (baked into the per-cluster objective) or
  :doc:`syndes`'s ``arm`` (separate solves). Use when each region/arm is its own
  study.
* *One* design representative of *every* group (coverage): the stratum quota
  ``stratum_col`` + ``min_per_stratum`` / ``max_per_stratum`` -- one shared donor
  pool, one estimand. In MAREX this quota is just the per-cluster cardinality.
* *One* design with geographic / forcing limits: the restriction suite (force
  in/out, border conflict, size band, donor rules) on SYNDES, MAREX, or GEOLIFT.

A constraint cannot turn one design into K designs, so ``cluster`` / ``arm`` are
not special cases of the quota; the quota is the lighter choice when you only
need coverage in a single design.

Failure-mode index
------------------

A reverse lookup: the symptom, and the method named for it.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Complication
     - Reach for
   * - No control group (everyone treated)
     - :doc:`shc`
   * - Randomized, few large units
     - :doc:`musc`
   * - Endogenous treatment, have an instrument
     - :doc:`siv`
   * - Endogenous treatment, have proxies / negative controls
     - :doc:`proximal`
   * - Honest CI / t-test for the ATT despite partially identified weights
     - :doc:`orthsc`
   * - Parallel trends holds (fixed-T, large-N)
     - difference-in-differences (off-ramp); :doc:`sdid`, :doc:`fdid`
   * - Single treated unit, no complications
     - :doc:`fdid`, :doc:`vanillasc`, :doc:`tssc`
   * - Spillovers onto donors (SUTVA), spatial
     - :doc:`spsydid`
   * - Spillovers onto donors, enumerable per-unit
     - :doc:`spillsynth`
   * - Contaminated donors unknown / large pool to screen
     - :doc:`spotsynth`
   * - Treated unit outside the donor convex hull
     - :doc:`iscm`, :doc:`nsc`, :doc:`rescm`, :doc:`pda`
   * - Nonstationary / spurious-trend matching
     - :doc:`sbc`, :doc:`hsc`
   * - Time-varying dynamics / persistent factors / noise
     - :doc:`tasc`, :doc:`fma`, :doc:`dscar`
   * - Nonlinear outcome surface
     - :doc:`nsc`
   * - Donor pool large vs pre-period (N ≳ T0)
     - :doc:`fscm`, :doc:`sparse_sc`, :doc:`pda`, :doc:`rescm`, :doc:`clustersc`, :doc:`bvss`
   * - Missing cells, MNAR
     - :doc:`snn`, :doc:`mcnnm`
   * - Block-missing with unit/time covariates (side information)
     - :doc:`rmsi`
   * - Distributional estimand (QTE, Lorenz, tails)
     - :doc:`dsc`
   * - Continuous / multi-valued treatment
     - :doc:`ctsc`
   * - Several related outcomes / short pre-period
     - :doc:`scmo`
   * - Several distinct intervention arms
     - :doc:`si`
   * - Interpolation bias (interpolating across dissimilar donors)
     - :doc:`masc`
   * - Many treated, same time
     - :doc:`sdid`, :doc:`microsynth`, :doc:`mlsc`
   * - Many treated at once, high-dimensional donor pool (block design)
     - :doc:`msqrt`
   * - Many treated, staggered adoption
     - :doc:`sdid`, :doc:`vanillasc` (simplex SC per unit, CFPT intervals),
       :doc:`rolldid`, :doc:`ppscm`, :doc:`seq_sdid`, :doc:`mcnnm`
   * - Staggered, long pre-period, few never-treated (event study)
     - :doc:`ssc`
   * - Designing for the ATT
     - :doc:`syndes`, :doc:`spcd`, :doc:`marex`
   * - Designing for the ATE
     - :doc:`marex`, :doc:`lexscm`
   * - Designing a geo roll-out (no pure donors)
     - :doc:`pangeo`
   * - Designing a geo-lift test (pick markets, one or many cells)
     - :doc:`geolift`, :doc:`multicellgeolift`

When in doubt, fit two or three of the candidate methods and compare the
counterfactuals and ATTs. Disagreement is itself diagnostic: it usually means
one of the gates above is binding harder than you thought.

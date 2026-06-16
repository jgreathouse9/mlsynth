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

Q1.2 · Is the treated unit outside the donors' convex hull even without
spillovers (a true outlier)?

* No -- next question.
* Yes -- relax the simplex: :doc:`nsc` (affine weights), :doc:`rescm`
  (penalised / :math:`L_\infty`), or the unconstrained :doc:`pda`.

Q1.3 · Is the outcome nonstationary, so a tight pre-fit might be a *spurious*
match?

* No -- next question.
* Yes -- decompose first: :doc:`sbc` (Hamilton trend/cycle split, match the
  cycle) or :doc:`hsc` (soft levels-vs-differences allocation).

Q1.4 · Are there persistent latent factors / time-varying dynamics / heavy
observation noise?

* No -- next question.
* Yes -- :doc:`tasc` (time-aware state-space model), :doc:`fma` (PC factors with
  a residual-bootstrap test), or :doc:`dscar` (time-varying weights for strongly
  autocorrelated panels with time-varying confounders).

Q1.5 · Is the untreated outcome a nonlinear function of the predictors?

* No -- next question.
* Yes -- :doc:`nsc`.

Q1.6 · Is the donor pool large relative to the pre-period (N >> T0)? This is the
most common reason to leave the standard workhorses: unrestricted fits overfit
the pre-period and predict the post-period worse.

* No -- next question.
* Yes -- escalate, roughly easiest first: :doc:`fscm` (tune the donor *count*),
  :doc:`sparse_sc` (L1 predictor/covariate selection), :doc:`pda`
  (L2-relaxation / Lasso / forward), :doc:`rescm` (one program from simplex SC
  to :math:`L_\infty` to DiD), :doc:`clustersc` (denoise + cluster donors), or
  :doc:`bvss` (Bayesian spike-and-slab with a soft simplex).

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
* Staggered *and* spillovers onto donors -- :doc:`spsydid`.
* Staggered *and* missing cells / gaps -- :doc:`mcnnm` (matrix completion handles
  staggered missingness natively).

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
     - :doc:`sdid`, :doc:`rolldid`, :doc:`ppscm`, :doc:`seq_sdid`, :doc:`mcnnm`
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

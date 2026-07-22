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
   Spillovers onto donors (SUTVA)?      ─► SPILLSYNTH · SpSyDiD (spatial) · BPSCS (spatial, unknown which) · SPOTSYNTH (unknown which) · ISCM (outside hull)
   Nonstationary / spurious trend?      ─► SBC · HSC
   Time-varying dynamics / heavy noise? ─► TASC · DSCAR · FMA · BFSC (Bayesian, credible band)
   Nonlinear outcome surface?           ─► NSC
   Donor pool N ≳ T0 (overfitting)?     ─► CLUSTERSC · SparseSC · PDA · RESCM · FSCM · BVSS
   Missing cells in the panel?          ─► SNN · MCNNM · RMSI (side information)
   Interpolation across dissimilar donors? ─► MASC
   Grouped microdata / repeated cross-sections? ─► SCD (differenced group means, √n bands) · DSC (distribution)
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

*Within the proximal family: instrument, two proxies, one proxy, or surrogates.*
These methods share a single premise -- some donors are not valid members of the
synthetic control but are still informative about the latent confounder, so they
can be repurposed as proxies (or negative controls) rather than discarded -- yet
the authors motivate four distinct entry points. Shi, Li, Miao and Tchetgen
Tchetgen (2026) give the foundational :doc:`proximal` framework: classical SC was
built for settings with a near-perfect pre-treatment fit, and when that fit is
poor even with a long pre-period, control units that do not help the fit can
serve as proxies of the unmeasured confounders, identifying the ATT through a
confounding bridge function (and extending naturally to nonlinear, binary, and
count outcomes the standard linear SC literature leaves understudied). Their
construction needs two kinds of proxy. Qiu, Shi, Miao, Dobriban and Tchetgen
Tchetgen (2024) relax the modelling burden: their doubly robust variant pairs an
outcome model with a weighting model and stays consistent if *either* is correct,
so you are not forced to specify the confounding-bridge outcome model exactly.
Park and Tchetgen Tchetgen (2025) cut the proxy requirement instead of the model
requirement: their single-proxy approach views the donor outcomes themselves as
the only proxies needed -- no separate group of treatment proxies -- and pairs it
with conformal inference, which buys valid intervals without a long
post-treatment series. Liu, Tchetgen Tchetgen and Varjão (2024) point the
framework forward in time: when the pre-period is short or the post-period long,
post-intervention *surrogates* (time-varying correlates of the effect) sharpen
estimation, and they show conditions under which post-treatment data alone can
identify the effect. So reach for :doc:`siv` when you hold a genuine instrument
and the worry is endogenous exposure; reach for :doc:`proximal` when you hold
proxies/negative controls instead -- the doubly robust route when you distrust
your outcome model, the single-proxy route when you have only one kind of proxy
and a short post-period, and the surrogate route when the leverage is in
post-treatment correlates rather than a long clean pre-period.

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
* Yes, spatial but you do *not* know which donors are contaminated -- you have
  per-unit coordinates and covariates and are willing to assume spillover risk
  grows with proximity -- :doc:`bpscs`, a Bayesian SC that *down-weights* (rather
  than excludes) likely-contaminated neighbours via a distance-and-covariate
  shrinkage prior, and returns a full posterior band (needs the ``[bayes]`` extra).
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
waterfall a close, simpler-to-implement second. A different tack when you have
per-unit coordinates and covariates but *cannot* name the contaminated donors:
:doc:`bpscs` (Fernández-Morales et al., 2026) assumes only that spillover risk
grows with proximity and *softly down-weights* likely-affected neighbours through
a Bayesian shrinkage prior whose scale blends spatial distance and covariate
similarity -- it keeps every donor (no hard exclusion) and returns a full
posterior band, at the cost of the ``[bayes]`` (NumPyro) dependency. In short:
unknown affected set -> ``spotsynth``; spatial with known W -> :doc:`spsydid`
(robust direct effect) or ``sar`` (Bayesian, small samples); spatial but the
affected set is unknown and you have coordinates -> :doc:`bpscs` (distance-based
shrinkage); specifiable structure with everyone possibly hit -> ``cd``; spillover
of interest with clean donors -> ``grossi``; affected donors too good to drop ->
``iscm`` (or ``iterative`` for simplicity).

Q1.2 · Is the treated unit outside the donors' convex hull even without
spillovers (a true outlier)?

* No -- next question.
* Yes -- relax the simplex: :doc:`nsc` (affine weights), :doc:`rescm`
  (penalised / :math:`L_\infty`), :doc:`src` (per-donor matching plus a
  Mallows-:math:`C_p` box synthesis that regularises the extrapolation and
  stays deterministic), :doc:`bscm` (Bayesian shrinkage -- horseshoe or
  spike-and-slab -- on unconstrained weights, with a credible interval on the
  effect), or the unconstrained :doc:`pda` (the
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

Q1.3b · Do you want the mechanism -- how much of the effect runs through one
observed channel (a mediator)?

* No -- next question.
* Yes -- :doc:`medsc` (mediation-analysis synthetic control).

*Decomposing the effect into channels -- MEDSC.* Everything else in Part 1
targets the total effect. :doc:`medsc` (Mellace and Pasquini, 2022) goes one
step further when you also observe a mediator -- a channel variable the
intervention moves and that in turn moves the outcome (for Proposition 99, the
retail price of cigarettes). It splits the synthetic-control effect into a
direct effect and an indirect effect that runs through the mediator, by building
a second, cross-world synthetic control that matches not only the treated unit's
pre-treatment outcome path but also its post-treatment mediator path; the gap
between the two controls is the channel. Reach for it only when the mechanism is
the question and the "treatment moves mediator moves outcome" ordering is
credible; for the total effect alone, a plain :doc:`vanillasc` is simpler. It
needs a donor pool wide enough to bracket the treated unit's post-treatment
mediator values -- so it exposes two donor pools, the wider one for the direct
fit.

Q1.4 · Are there persistent latent factors / time-varying dynamics / heavy
observation noise?

* No -- next question.
* Yes -- :doc:`tasc` (time-aware state-space model), :doc:`fma` (PC factors
  with a residual-bootstrap test), :doc:`bfsc` (a Bayesian latent-factor
  model that returns a full posterior credible band on the counterfactual and
  prunes surplus factors with a horseshoe+ prior; needs the ``[bayes]`` extra),
  or :doc:`mtgp` (a Bayesian factor model whose factors are *smooth in time* --
  a Gaussian-process prior -- so the counterfactual band widens the further the
  post-period extrapolates; also needs the ``[bayes]`` extra).
  (For micro panels with observed time-varying *confounders* and autoregressive
  outcomes, :doc:`dscar` is a different paradigm -- see the remark below.) If you
  also observe many time-varying *covariates* and want them to identify the
  counterfactual, :doc:`cscipca` instruments the factor loadings with the
  covariates -- see the remark below.

*FMA versus BFSC -- frequentist or Bayesian factor SC.* Both fit the untreated
outcome with a latent-factor model rather than a donor weighting, so both handle
a treated unit outside the donors' convex hull. :doc:`fma` (Li and Sonnier,
2023) estimates the factors by principal components, regresses the treated unit
on the estimated loadings, and contributes a formal inference theory (a
residual bootstrap giving valid intervals without the equal-variance
assumption). :doc:`bfsc` (Pinkney, 2021) instead estimates the factors and
loadings jointly in one Bayesian model, masks the treated post-period as missing
data, and reads the counterfactual off the posterior -- so the credible band
propagates the uncertainty in the factors themselves, and a horseshoe+ prior on
the loadings makes the factor count a soft upper bound rather than a choice you
must commit to. Prefer :doc:`fma` when you want a fast, dependency-free point
estimate with bootstrap intervals; prefer :doc:`bfsc` when you want a full
posterior band and would rather not fix the number of factors, and you can take
on the ``[bayes]`` (NumPyro) dependency.

*Factor SC with covariate-instrumented loadings.* :doc:`cscipca` (Wang, 2024)
is the factor estimator to reach for when you observe many time-varying
covariates that plausibly drive the outcome. Instead of learning a free loading
for each unit from its outcome history, it projects the loadings onto the
covariates (instrumented principal component analysis), so the covariates -- not
a convex-hull condition -- carry the counterfactual, and the treated unit may
sit well outside the donor range. Its edge over an outcome-only factor model
(:doc:`fma`, :doc:`cfm`) appears precisely when the covariates are informative
and only partially observed, and vanishes when every relevant covariate is
already in hand. It needs at least as many covariates as factors and a
pre-period longer than covariates-times-factors, and reports a per-period
moving-block conformal band. With no covariates, use :doc:`fma` or :doc:`cfm`.

*Which Bayesian synthetic control?* Six estimators are Bayesian, and they
split on what carries the prior. :doc:`mvbbsc`, :doc:`bscm`, and :doc:`bvss`
put a prior on the *donor weights* and all report donor weights -- reach for
them, at Q1.2 or Q1.6, when you want interpretable weights with a credible
interval. They differ in the constraint: :doc:`mvbbsc` (Martinez and
Vives-i-Bastida) keeps the *hard simplex* with a uniform prior, standardizes
internally so the fit is unit-free, and carries a Bernstein-von Mises guarantee
that makes its credible interval a valid confidence interval -- the choice when
the treated unit is inside the donors' hull and you want principled inference on
the classical convex-combination model; :doc:`bscm` drops the simplex for
shrinkage (horseshoe or spike-and-slab) on *unconstrained* weights; :doc:`bvss`
keeps a *soft* simplex and adds spike-and-slab donor selection with inclusion
probabilities. :doc:`bfsc` and :doc:`mtgp` put the prior on a *latent-factor
model* of the outcome and report a counterfactual band with no donor weights --
reach for them when a shared factor structure rather than a weighted average of
donors is the right model. The two factor models differ in one thing:
:doc:`bfsc` leaves the factors unconstrained over time, while :doc:`mtgp` puts a
Gaussian-process (squared-exponential) prior on them, so its factor paths are
smooth and its post-period band grows with extrapolation distance. Prefer
:doc:`mtgp` when the untreated series are smooth trends and you want that
widening band; prefer :doc:`bfsc` when the shared structure is best left
unconstrained. The sixth, :doc:`bpscs`, puts the prior on *donor coefficients*
but scales it by an external covariate-and-distance utility -- reach for it, at
Q1.1, when the concern is spatial spillover contaminating the donor pool and you
want close-by donors down-weighted rather than trusted or dropped.

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

*Low-rank (matrix) methods -- when the donor matrix has factor structure.* This
gate is also the entry point to a family that takes the latent-factor view
literally: if a few common factors drive every unit, the untreated outcome matrix
is approximately *low-rank*, so one can *denoise* or *complete* it before fitting
weights. They divide by what corruption they guard against. When the data are
*disaggregate* (individual-level: health records, income, store sales) so the
donor count dwarfs the pre-period (:math:`n \gg T_0`) and the panel is noisy,
:doc:`clustersc` (Rho et al., 2025) first *clusters* donors by their
latent-factor signature -- keeping only the group that behaves like the target --
then denoises: ``method="pcr"`` keeps the top singular values by hard thresholding
(Amjad et al., 2018; Agarwal et al., 2021), good for *Gaussian* noise, while
``method="rpca"`` uses robust PCA / principal component pursuit (Candes et al.),
separating a low-rank part from a *sparse* one and so tolerating *outliers and
missing entries*. :doc:`fma` (Li and Sonnier, 2023) takes the factor model
head-on -- it projects the donors onto a low-dimensional factor space and
regresses the treated unit on the estimated loadings with *no* simplex or
convex-hull constraint, so it handles a treated unit *outside* the donors' range
and many treated units; its contribution is a *formal* inference theory (valid
confidence intervals without the equal-variance assumption the usual factor-model
bootstrap needs), at the cost of overfitting if the donor pool is large.
:doc:`tasc` (Rho et al., 2025) observes that all of the above are *time-agnostic*
-- permuting the time index leaves the matrix spectrum unchanged -- and embeds the
low-rank panel in a *state-space* model (Kalman filter / RTS smoother) to exploit
the temporal structure too, which pays off under *strong trends and high
observation noise*. And :doc:`rmsi` (Agarwal et al., 2026) extends low-rank
*completion* to use row- and column-side *covariates*, decomposing the matrix into
covariate-driven and residual low-rank parts -- preferable when you have
informative side information and *missing* cells (including the block-missing
pattern of a causal panel). Rule of thumb: disaggregate and noisy ->
:doc:`clustersc` (``pcr`` for noise, ``rpca`` for outliers/missing); treated
outside the hull or you need valid factor-model inference -> :doc:`fma`; strong
trends plus noise -> :doc:`tasc`; informative covariates with missing cells ->
:doc:`rmsi`.

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
  :doc:`bvss` (Bayesian spike-and-slab with a soft simplex), or -- when the
  high dimension is in the *covariates* rather than the donors, and only a few
  matter -- :doc:`beast` (covariate-balancing weights under sparsity, with a
  doubly-robust, analytically-inferred ATT).

*A privacy constraint on the donors.* If the donor pool is sensitive -- patient
records in a clinical external-control arm, proprietary firm-level series in a
data cooperative -- and the counterfactual must be released externally,
:doc:`dpsc` (Rho, Cummings and Misra, 2023) is the only estimator here with a
formal privacy guarantee: a ridge synthetic control fitted with differentially
private empirical risk minimisation, so publishing the effect leaks a provably
bounded amount about any single donor. It buys a privacy certificate, not a
better point estimate, and it is worth its accuracy cost only when the donor
pool is large and the pre-period long; on the usual small donor pool, and for
public aggregates, prefer :doc:`vanillasc`.

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

*Which matrix-completion estimator -- and why the missingness mechanism decides.*
The three estimators differ less in the imputation machinery than in what they
assume about *why* cells are missing. :doc:`mcnnm` (Athey, Bayati, Doudchenko,
Imbens and Khosravi (2021)) imputes the untreated potential outcomes by
nuclear-norm-regularised low-rank completion of the whole panel, with two-way
fixed effects. Their headline argument is regime-robustness: synthetic control
and the unconfoundedness/horizontal regression each work well only in one shape
of panel -- unconfoundedness fails when :math:`T \gg N`, synthetic control fails
when :math:`N \gg T` -- whereas the completion objective nests both as special
cases (differing only in how hard a restriction they place on the factorisation)
and stays accurate across all regimes and arbitrary, staggered missing patterns.
That argument presumes the *pattern* of missingness is essentially ignorable
(missing at random given the low-rank structure). :doc:`snn` (Agarwal, Dahleh,
Shah and Shen (2021)) is built for exactly the case MCNNM sets aside: missingness
that is *not* at random -- the probability a cell is observed depends on the
cell's own latent value (a policymaker adopts where outcomes are favourable; a
user only rates films they chose to watch), entries can be deterministically
missing (positivity violated), and the missingness of one cell can depend on
others. SNN imputes each target cell from a fully observed *anchor* block of rows
and columns and delivers entry-wise (max-norm) guarantees -- accurate inference
for each individual :math:`(i,j)` cell rather than for a row average. So prefer
:doc:`mcnnm` when the gaps are plausibly incidental and you want one estimator
that travels across short, long, and square panels; prefer :doc:`snn` when the
gaps are informative -- selected on the outcome itself -- and you need a credible
counterfactual for specific cells; reach for :doc:`rmsi` when the missing block is
the treated region and you have margin covariates that carry signal about it.

Q1.8 · Is your estimand or treatment effect non-standard (not a scalar mean ATT
for one binary treatment)?

* No -- you are done; use the *Start here* method.
* The whole distribution (quantiles, Lorenz, tails) -- :doc:`dsc`.
* A continuous or multi-valued dose with no clean control -- :doc:`ctsc`.
* Several related outcomes (helps most with a short pre-period) -- :doc:`scmo`.
* A high-frequency outcome where you must decide whether to aggregate the
  pre-period (months vs years) -- :doc:`scta`.
* Several distinct intervention arms to compare -- :doc:`si`.

*When to reach for Synthetic Interventions.* :doc:`si` (Agarwal, Shah and Shen
(2024)) is the multi-arm member of this same low-rank family, and the comparison
is cleanest stated through the question it answers. Standard synthetic control
recovers one slice of the potential-outcomes array -- outcomes under a single
treatment, usually control -- because its matrix factor model carries latent
factors only for units and time. SI lifts that to a *tensor* factor model with an
added latent factorisation over treatments, so the same panel can be completed
under interventions a unit never actually received: what would California's
cigarette sales have been under a tax increase rather than the program it
adopted? Prefer :doc:`si` when you have several intervention arms and want each
unit's counterfactual under arms it did not take -- the multi-treatment
generalisation Abadie (2021) posed as an open question -- rather than a single
average contrast against one control condition.

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

*Micro-level versus disaggregated: two ways to use granular data.* :doc:`microsynth`
and :doc:`mlsc` both reach below the level at which treatment is assigned, but
they answer different questions and the authors motivate them differently.
Robbins, Saunders and Kilmer (2017) design :doc:`microsynth` for settings where
the treated region itself is a bundle of many micro-units (census blocks in a
neighbourhood) measured on many covariates and several outcomes at once. Their
contribution is *calibration*: weights are chosen so the synthetic control
matches the treated region exactly across all of those covariates and outcomes
simultaneously -- a survey-weighting construction rather than a single-outcome
fit -- and inference comes from a permutation procedure over placebo areas plus
an omnibus statistic that tests jointly across outcomes and post-periods, so the
many-outcome problem is handled without ad hoc multiple-comparison patching. Use
it when the granularity is in the *treated unit* and you need one set of weights
to balance a wide panel of characteristics and outcomes. Bottmer (2025) frames
:doc:`mlsc` around a different decision: when outcomes are observed below the
assignment level (county outcomes under a state policy), should you fit
aggregated, disaggregated, or some blend? Disaggregating the *controls* expands
the donor pool and can sharply improve aggregate-level precision, but enlarging
it past the pre-period count risks overfitting and non-uniqueness. mlSC makes the
aggregation choice data-driven, regularising toward the classical aggregated SC
and letting the data decide how much disaggregated control variation to exploit.
Prefer :doc:`mlsc` when the question is how aggressively to disaggregate a donor
pool, and :doc:`microsynth` when the treated side is a granular many-outcome
bundle you need to balance exactly.

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
* Exposure defined by a within-unit *subgroup* (a triple difference), and you
  distrust parallel trends across that third dimension -- :doc:`sdid` in its
  synthetic triple-difference mode (``subgroup`` / ``target_subgroup``; Zhuang
  2024). It demeans the outcome by the non-target subgroup within each
  treatment-group-by-time cell, reducing the DDD to a DID, then runs SDID on the
  exposed subgroup -- so the counterfactual is a weighted combination of control
  units rather than a parallel-trends extrapolation across states *and* subgroups.
  Reach for it when, e.g., only one age band in a state is policy-exposed.

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

*Why design at all, and which design method.* The shared premise of this family,
argued most directly by Doudchenko et al. (2021) and Abadie and Zhao (2026), is
that when treatment can only be applied to a few large, expensive units (media
markets, regions, whole products) randomization is unbiased *ex ante* but, over
the single assignment you actually run, routinely hands you treated and control
groups with very different baselines -- a draw you cannot average away. If you
hold pre-treatment panel data, you can do better by *choosing* the split. The
methods then split on two axes: the estimand they target, and how they solve the
(NP-hard) assignment problem. Doudchenko et al. cast the joint choice of treated
set and donor weights as a mixed-integer program that directly minimises the
*ATT estimator's* mean squared error (:doc:`syndes`) -- provably optimal but
combinatorial. Lu, Li, Ying and Blanchet (2022) attack the same covariate-
balancing design but reformulate it as a phase-synchronisation problem solved by
a spectrally-initialised power method (:doc:`spcd`), trading the MIP's
exactness for a *global* optimality guarantee under the linear factor model and
a runtime in seconds rather than minutes -- prefer it when the unit count makes
the MIP slow. Abadie and Zhao instead target the *population* ATE: they choose
synthetic-treated and synthetic-control groups whose pre-experiment predictors
match the population means (:doc:`marex`), a convex design that lowers bias
relative to randomization and (their Theorem 1) shifts structure from unobserved
loadings into observed covariates. Vives-i-Bastida (2022) extends that framework
to multiple outcomes and adds the minimum-detectable-effect machinery and
practical exclusion / fairness constraints that :doc:`lexscm` uses to optimise
validity first and power second. The two geo-rollout members -- :doc:`pangeo`
(no unit left untreated, trajectory-matched supergeos) and :doc:`geolift` /
:doc:`multicellgeolift` (market selection under a budget) -- are the
applied-marketing specialisations; the questions below route to each.

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
  pool, one estimand. Honoured by all four design methods; in MAREX it can also
  be expressed as the per-cluster cardinality when each region is its own design.
* *One* design with geographic / forcing limits: the restriction suite (force
  in/out, border conflict, cluster, coverage quota, size band) is honoured by all
  four design methods -- SYNDES, GEOLIFT, LEXSCM, and MAREX -- so a constraint
  binds whichever method (or comparison) you run.

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
   * - Decompose the effect through a mediator (mechanism)
     - :doc:`medsc`
   * - Time-varying dynamics / persistent factors / noise
     - :doc:`tasc`, :doc:`fma`, :doc:`bfsc`, :doc:`mtgp`, :doc:`dscar`
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
   * - High-frequency outcome / aggregate the pre-period
     - :doc:`scta`
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

Whichever method you pick, the :doc:`truncated_history` robustness check
re-estimates it on truncated pre-treatment windows and profiles the effect
against the pretreatment horizon -- a stable profile supports the causal
reading, an unstable one says report an interval. It is the pretreatment-horizon
companion to the in-space placebo and leave-one-out checks.

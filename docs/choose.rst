Choose an estimator
===================

.. currentmodule:: mlsynth

mlsynth ships dozens of synthetic-control estimators because there is
no single "right" one. Each makes different trade-offs in bias,
variance, computational cost, and the assumptions it imposes on the
data. This page is a guide to which estimator fits your problem.

Two top-level questions determine the family you should be looking
at:

1. Are you doing **observational causal inference** -- you already
   have a panel and you want an ATT estimate?
2. Or are you doing **experimental design** -- you are deciding which
   units to assign treatment to, in order to maximise statistical
   power or pre-treatment balance for a *prospective* experiment?

These are fundamentally different problems with different APIs in
mlsynth. The observational side splits into nine sub-families, each
designed for a specific complication that classical SC cannot handle
on its own.

Observational methods
---------------------

Canonical workhorses
^^^^^^^^^^^^^^^^^^^^

The "single treated unit, sharp intervention, scalar ATT, raw outcome
series" case. Start here unless you know your problem has a
complication that pushes you elsewhere.

* :doc:`tssc` -- Two-Step SC with a *formal pre-trends test*. Fits the
  full SC family (SC, MSCa, MSCb, MSCc) and uses a subsampling test
  to pick the variant the data supports. Subsampling confidence
  intervals.
* :doc:`fdid` -- Forward Difference-in-Differences. Greedy
  forward-step donor selection with a DiD safety net if SC's
  parallel-trends assumption fails.

Decomposition-first SC
^^^^^^^^^^^^^^^^^^^^^^

These estimators do not fit on the raw outcome series. They first
*decompose* the panel into a trend, a cycle, or a spectral component
and apply SC to only one piece. This makes them robust to
nonstationarity and spurious trends that fool classical SC.

* :doc:`sbc` -- Synthetic Business Cycle. Splits the outcome via a
  Hamilton filter into trend + cycle, then matches the cycle. Built
  for macro panels with strong but non-causal trends.
* :doc:`hsc` -- Harmonic SC. Soft, data-driven allocation between
  matching on *levels* and matching on *differences* in a spectral
  sense; robust to mixed I(0) / I(1) donors.

Generalising the estimand, treatment, or unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical SC assumes one binary treatment, one outcome, a scalar ATT,
and a unit at the aggregate level. Each of the estimators below relaxes
*one* of those four constraints.

* :doc:`scmo` -- *Multiple outcomes.* Common SC weights balanced
  across a domain of related outcome series simultaneously.
* :doc:`ctsc` -- *Continuous treatment.* Generalised SC for
  continuous- or multi-valued treatment intensity, with interactive
  fixed effects and unit-specific slopes.
* :doc:`dsc` -- *Distributional ATT.* Targets the full counterfactual
  outcome *distribution* (2-Wasserstein barycenter), not just its
  mean.
* :doc:`si` -- *Multiple treatment arms.* Counterfactuals for several
  distinct interventions at once via a tensor-factor structure.
* :doc:`microsynth` -- *Fine-grained / individual-level units.*
  User-level balancing SC that reweights millions of control
  individuals to match treated-group covariate moments.

Relaxing the convex hull
^^^^^^^^^^^^^^^^^^^^^^^^

Classical SC weights live on the simplex, which means the synthetic
control is a *convex combination* of donors. If the treated unit's
outcomes lie outside the donors' convex hull -- as they often do for
outliers like Hong Kong's GDP in the late 1990s -- the simplex is too
restrictive. These estimators relax it in different ways.

* :doc:`iscm` -- Imperfect SC. Builds a synthetic control for *every*
  unit and uses moment conditions that are valid even under
  transitory shocks with nonzero asymptotic variance, then weighs
  units by the appropriateness of their own synthetic controls.
* :doc:`nsc` -- Nonlinear SC. Affine (signed) weights with an
  elastic-net penalty that adapts to nonlinear outcome surfaces.

High-dimensional donor pools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the number of donors :math:`N` is comparable to or larger than
the number of pre-periods :math:`T_0`, the classical quadratic
program loses its unique solution and Lasso-style alternatives tend
to over-select. The estimators below handle the high-dimensional
regime in different ways.

* :doc:`bvss` -- Bayesian spike-and-slab variable selection with a
  *soft* simplex constraint. The posterior of :math:`\tau` tells you
  whether the simplex should bind.
* :doc:`clustersc` -- Cluster-then-pool SC. Groups similar donors
  with k-means or robust PCA before weighting; tolerant to noise and
  outliers.
* :doc:`mlsc` -- Multi-level SC. Hierarchical penalty shrinks
  disaggregate weights toward the classical SC solution.
* :doc:`fscm` -- Forward-selected SC. Bilevel optimisation picking
  both the donor count and their weights.
* :doc:`sparse_sc` -- Sparse SC. Explicit L1 sparsity penalty on
  predictor weights.
* :doc:`pda` -- Panel Data Approach. Forward selection or
  L2-relaxation / Lasso on the donor regression.
* :doc:`rescm` -- Relaxed SC. Unified convex program nesting
  classical SC, L-infinity SC, and equal-weights.

Time-aware and latent-factor models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical SC is permutation-invariant in time -- shuffling
pre-treatment dates does not change the fit. When the data carry
strong, persistent latent factors that evolve in time, that
information is wasted. The estimators here model the temporal
dynamics explicitly without first decomposing the series.

* :doc:`tasc` -- Time-Aware SC. Linear-Gaussian state-space model
  fitted with EM (Kalman filter + RTS smoother). Robust under high
  observation noise; provides posterior credible bands.
* :doc:`fma` -- Factor Model Approach. Principal-component extraction
  from the donor panel with a formal residual-bootstrap test.

Staggered adoption and multiple cohorts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple treated units adopt at *different* times, the single-
treated-unit framework breaks down. These estimators aggregate
cohort-level effects and produce event-study (per-event-time) ATTs
with appropriate inference.

* :doc:`sdid` -- Synthetic Difference-in-Differences. Doubly weighted
  by unit *and* time weights; robust when parallel trends or exact
  matching fail.
* :doc:`seq_sdid` -- Sequential SDiD for staggered adoption.
  Cohort-level aggregates with sequential imputation and bootstrap
  CIs.
* :doc:`spsydid` -- Spatial Synthetic-DiD. Extends SDiD with
  spillover exposure for geographic interference.
* :doc:`ppscm` -- Partially-Pooled SC. Interpolates between separate
  per-cohort fits and a single fully-pooled SC.

Missing data and matrix completion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the outcome panel has missing cells -- staggered adoption with
gaps, unbalanced panels, MNAR data -- treat the imputation of the
treated counterfactual as a low-rank matrix-completion problem.

* :doc:`mcnnm` -- Matrix Completion with Nuclear-Norm Minimisation.
  Imputes missing outcomes via nuclear-norm-regularised low-rank
  recovery; handles both single and staggered treatment.
* :doc:`snn` -- Synthetic Nearest Neighbours. Causal MC under MNAR
  via principal component regression on anchor blocks.

Identification under endogeneity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical SC assumes the treatment is exogenous conditional on the
factor structure. When that fails -- the treatment is endogenous,
the instrument is only conditionally valid, or unmeasured
confounders break the SC identification -- you need an estimator
that uses an *external* identification source.

* :doc:`siv` -- Synthetic IV. SC-debias the (outcome, treatment,
  instrument) triple before running 2SLS; only the instrument needs
  to be *partially* valid (orthogonal conditional on the factors).
* :doc:`proximal` -- Proximal SC. Splits controls into donors and
  *proxies* (negative controls) to instrument latent confounders.

Experimental design
-------------------

These estimators target a fundamentally different problem: the
treatment has *not yet been assigned*. You are deciding which units
to treat, which to use as controls, or how to compose synthetic
comparison groups, in order to maximise the statistical precision
of the planned experiment. They share a design-stage API (often
``MAREXConfig``-based) and return assignments, not ATTs.

* :doc:`lexscm` -- Lexicographic SC. Chooses which units to treat to
  jointly maximise validity (balance) and statistical power.
* :doc:`marex` -- Matrix Exclusion. Picks treatment / control
  assignments to minimise pre-experiment imbalance.
* :doc:`syndes` -- Synthetic Design. Mixed-integer program jointly
  optimising treatment assignment *and* synthetic weights for
  minimum estimator MSE.
* :doc:`pangeo` -- Parallel-Trends Supergeo Design. Chooses geos and
  forms composite supergeos to maximise pre-period trajectory
  parallelism (geo-experiment setting).
* :doc:`spcd` -- Synthetic Principal Component Design. Spectral
  phase-synchronisation for treatment assignment with optimal
  precision.

By inference type
-----------------

A cross-cutting view: which estimators give you which kind of
uncertainty quantification?

**Frequentist confidence intervals.** :doc:`tssc` (subsampling),
:doc:`fdid` (Wald-style), :doc:`sdid` and :doc:`seq_sdid`
(bootstrap), :doc:`fma` (residual bootstrap), :doc:`siv` (asymptotic
IV sandwich or split-conformal).

**Bayesian credible intervals.** :doc:`bvss` (spike-and-slab
posterior), :doc:`tasc` (Kalman posterior bands).

**Conformal / model-free.** :doc:`siv` with ``inference_method =
"conformal"``; :doc:`proximal` for proximal causal inference under
unmeasured confounding.

**Per-event-time ATTs (event study).** :doc:`seq_sdid`,
:doc:`spsydid`, :doc:`ppscm`.

Still not sure?
---------------

* If your panel looks like the canonical Abadie-Diamond-Hainmueller
  Proposition 99 study -- one treated state, dozens of donor states,
  decades of pre-treatment data -- start with :doc:`tssc`. Its
  *Verification* section reproduces the published Brooklyn-showroom
  and Figure-2 simulation numbers exactly.
* If your :math:`N / T_0` ratio is large or you suspect the simplex
  constraint is too restrictive, start with :doc:`bvss`. Its
  *Verification* section reproduces Xu & Zhou's China anti-corruption
  ATT to three decimals.
* If your outcome series carries a strong trend that is not the
  causal signal you are after, start with :doc:`sbc` or :doc:`hsc`
  -- they decompose first and match on the cyclical component only.
* If your application has an obvious instrument (a shift-share, a
  tariff schedule, a supply shock), start with :doc:`siv`.
* If you are designing a prospective experiment (deciding which
  geos to treat), start with :doc:`syndes` or :doc:`pangeo`.

When in doubt, fit two or three estimators and compare. mlsynth's
single long-DataFrame API makes that cheap to do.

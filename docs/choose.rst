Choose an estimator
===================

.. currentmodule:: mlsynth

mlsynth ships dozens of synthetic-control estimators because there is no
single "right" one -- they trade off bias, variance, computational cost,
and the assumptions they impose on the data. This page is a short guide
to which estimator to reach for first.

If you only read one thing, read the table immediately below: it covers
the eight settings that account for the bulk of applied work. The
sections that follow drill in by treatment design, data shape, and the
kind of inference you need.

The two-minute version
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Setting
     - Estimator
     - Why
   * - Single treated unit, classical SC, want a built-in pre-trends test
     - :doc:`tssc`
     - Tests SC's adding-up and zero-intercept restrictions, falls back
       to MSCa / MSCb / MSCc when violated. Subsampling CIs.
   * - Single treated unit, want forward-step donor selection plus a DiD
       safety net
     - :doc:`fdid`
     - Forward Difference-in-Differences -- greedy donor selection with
       a DiD fallback if SC fails.
   * - Donors :math:`\gg` pre-periods (high-dimensional)
     - :doc:`bvss`
     - Bayesian spike-and-slab with a *soft* simplex constraint. The
       posterior tells you whether the constraint should bind.
   * - High-dim donors with heterogeneous structure
     - :doc:`clustersc`
     - Cluster-then-pool synthetic control: groups similar donors before
       weighting.
   * - Staggered adoption, multiple treated cohorts
     - :doc:`seq_sdid` or :doc:`spsydid`
     - Sequential SDiD / staggered SyDiD with per-event-time ATTs and
       bootstrap inference.
   * - Treatment is endogenous, an instrument is available
     - :doc:`siv`
     - Synthetic IV: SC-debias the (outcome, treatment, instrument)
       triple before running 2SLS.
   * - Strong temporal trends; want a generative state-space model
     - :doc:`tasc`
     - Kalman-filter + RTS-smoother + EM. Tolerates high observation
       noise and provides posterior bands.
   * - Outcome matrix is sparse / partially missing
     - :doc:`mcnnm`
     - Nuclear-norm matrix completion of the (units :math:`\times` time)
       panel.

Each estimator's page contains the math, an empirical example, and (where
applicable) a *Verification* section reproducing the original paper's
reported numbers.

By treatment design
-------------------

**Single treated unit, sharp intervention.** This is the classical SC
setting (a state, a country, a market). The workhorses are :doc:`tssc`
(if the donor pool is small relative to :math:`T_0`), :doc:`fdid`
(forward-selection DiD), :doc:`fma` (factor model approach),
:doc:`scmo` (SC with covariates), and :doc:`pda` (panel-data approach
via forward selection). When the donor pool is comparable to or
exceeds :math:`T_0`, switch to :doc:`bvss`, :doc:`clustersc`, or
:doc:`mlsc`.

**Staggered adoption (many cohorts adopting at different times).** Use
:doc:`seq_sdid` for Arkhangelsky-Samkov sequential synthetic
difference-in-differences with per-event-time ATTs and bootstrap CIs.
:doc:`spsydid` is the staggered Synthetic-DiD variant.
:doc:`iscm` and :doc:`ctsc` handle clustered staggered designs.

**Multiple treated units, large panels.** :doc:`microsynth` is the
canonical micro-level SC method for many small treated cells.

**Spillover concerns.** :doc:`spill` provides spillover-aware
inference; pair with :doc:`iscm` when the spillovers themselves have a
cluster structure.

By data shape
-------------

**Donors :math:`\ll` pre-periods (low-dim, classical regime).** Most
estimators work here. Default to :doc:`tssc` for the formal pre-trends
test plus efficiency, or :doc:`fdid` for the DiD safety net.

**Donors :math:`\gg` pre-periods (high-dim).** The classical QP loses
its unique solution. Use a variable-selection or penalised estimator:

* :doc:`bvss` -- Bayesian spike-and-slab, soft simplex.
* :doc:`clustersc` -- cluster-and-pool, robust to noise and outliers.
* :doc:`mlsc` -- multi-output machine-learning SC.
* :doc:`marex` -- matrix exclusion of contaminated donors.
* :doc:`lexscm` / :doc:`rescm` -- lexicographic and L2-relaxation SC.
* :doc:`sparse_sc` -- explicit sparsity constraint.
* :doc:`ppscm` -- partially-pooled SC for staggered adoption.

**Sparse / partially missing outcomes.** Switch to
:doc:`mcnnm` (nuclear-norm matrix completion) or :doc:`snn` (synthetic
nearest neighbours).

**Strong, persistent trends in the pre-period.** Permutation-invariant
SC variants discard temporal information; use :doc:`tasc` instead --
its state-space backbone fits the trend explicitly. :doc:`si`
(Synthetic Interventions) and :doc:`fma` (Factor Model Approach) also
explicitly model latent factors.

By inference needs
------------------

**Frequentist confidence intervals.** :doc:`tssc` (subsampling),
:doc:`fdid` (Wald-style), :doc:`sdid` and :doc:`seq_sdid` (bootstrap),
:doc:`fma` (residual bootstrap), :doc:`siv` (asymptotic IV sandwich or
split-conformal).

**Bayesian credible intervals.** :doc:`bvss` (spike-and-slab
posterior), :doc:`tasc` (Kalman posterior bands), :doc:`ctsc`
(conformal time-series SC).

**Conformal / model-free.** :doc:`siv` with ``inference_method =
"conformal"``; :doc:`proximal` for proximal causal inference under
unmeasured confounding.

**Per-event-time ATTs (event study).** :doc:`seq_sdid`,
:doc:`spsydid`, :doc:`iscm` and :doc:`ppscm` all produce
:math:`\widehat{\mathrm{ATT}}_k` indexed by event time :math:`k`.

By special features
-------------------

**Instrumental variable available.** :doc:`siv` is the only IV-aware
synthetic-control estimator in the toolbox.

**Time-invariant covariates.** :doc:`scmo` supports auxiliary
predictors alongside the outcome series; :doc:`syndes` runs synthetic
design optimisation jointly over covariates.

**Unmeasured confounding with proxies.** :doc:`proximal` implements
proximal-causal SC.

**Distributed / privacy-preserving fits.** :doc:`dsc` (distributed SC)
partitions the donor pool across nodes.

Still not sure?
---------------

* If your panel looks like the canonical Abadie-Diamond-Hainmueller
  Proposition 99 study (one treated state, dozens of donor states,
  decades of pre-treatment data), start with :doc:`tssc`. The
  *Verification* section on that page reproduces the published
  Brooklyn-showroom and Figure-2 numbers exactly.
* If your :math:`N / T_0` ratio is large or you suspect the simplex
  constraint is too restrictive, start with :doc:`bvss`. Its
  *Verification* section reproduces Xu & Zhou's China anti-corruption
  ATT to three decimals.
* If your application has an obvious instrument (a shift-share, a
  tariff schedule, a supply shock), start with :doc:`siv`.

When in doubt, fit two or three estimators and compare. mlsynth's
single long-DataFrame API makes that cheap to do.

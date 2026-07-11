Bayesian Factor Synthetic Control (BFSC)
========================================

.. currentmodule:: mlsynth

When to use BFSC -- and when not to
-----------------------------------

Bayesian Factor Synthetic Control (Pinkney 2021) fits the no-intervention
outcome of every unit with a Bayesian latent factor model and reads the treated
unit's counterfactual off the posterior. Reach for it when you have a panel with
a clear common structure -- seasonality, shared macro or market shocks -- and you
want honest uncertainty on the effect rather than a single line, and you would
rather not commit in advance to a number of factors. It is a good choice when:

* You want a full posterior credible band on the counterfactual and the ATT,
  propagating the uncertainty in the factors themselves, not just a point path.
* The donor pool shares latent factors with the treated unit (a factor / interactive
  fixed-effects structure), so a weighted average of donors is the wrong model but
  a shared low-rank term is the right one.
* You do not want to pick the number of factors by hand: a horseshoe+ prior on the
  loadings prunes the factors the data do not support, so the factor count is a soft
  upper bound.

Do not use BFSC when:

* You want an interpretable set of donor weights -- BFSC is a factor model and
  reports no donor weights. Use a weighting method (:doc:`vanillasc`, :doc:`src`)
  or the Bayesian weighting :doc:`bscm`.
* You need a dependency-free estimate. BFSC draws its posterior with NUTS
  (NumPyro), so it needs the ``[bayes]`` optional dependency
  (``pip install 'mlsynth[bayes]'``). For a dependency-free Bayesian SC use
  :doc:`bscm` (a pure-numpy Gibbs sampler over donor weights).
* You want a frequentist factor-model SC. Use :doc:`fma` (Li & Sonnier) or the
  generalized-SC family.

The Bayesian SC family
----------------------

mlsynth carries three Bayesian synthetic-control estimators; they differ in
what they place a prior on, and BFSC is the one that models the outcome rather
than the weights.

* :doc:`bscm` (Kim, Lee and Gupta) -- shrinkage (horseshoe or spike-and-slab)
  on unconstrained donor weights; a pure-numpy Gibbs sampler, and it reports
  donor weights.
* :doc:`bvss` (Xu and Zhou) -- spike-and-slab donor selection on a soft
  simplex whose tightness is learned; a pure-numpy Metropolis-within-Gibbs
  sampler, and it reports donor weights and inclusion probabilities.
* :doc:`bfsc` (Pinkney) -- a Bayesian latent-factor model, not a donor
  weighting; NUTS through the ``[bayes]`` optional dependency, and it reports a
  counterfactual credible band and no donor weights.

Reach for a weighting prior (:doc:`bscm`, :doc:`bvss`) when you want
interpretable donor weights; reach for BFSC when a shared factor structure --
not a weighted average of donors -- is the right model for the untreated
outcome.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit and
:math:`\mathcal{N}_0` the donor pool, over units
:math:`\mathcal{N} \coloneqq \{1, \ldots, N\}`; time is
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}` with the intervention after
:math:`T_0`, pre-period :math:`\mathcal{T}_1` and post-period
:math:`\mathcal{T}_2`. BFSC models the no-intervention outcome with the canon's
linear factor model plus additive effects,

.. math::

   y_{jt}^N = \mathbf{f}_t^{\top}\boldsymbol{\lambda}_j + \delta_t + \kappa_j
              + u_{jt}, \qquad u_{jt} \sim \mathcal{N}(0, \sigma^2),

with latent factors :math:`\mathbf{f}_t \in \mathbb{R}^{L}` (the rows of a
:math:`T\times L` factor matrix :math:`\mathbf{F}`), unit loadings
:math:`\boldsymbol{\lambda}_j` (the paper's :math:`\boldsymbol{\beta}_j`), a time
effect :math:`\delta_t`, and a unit effect :math:`\kappa_j`. Unlike a
frequentist factor SC, :math:`\mathbf{F}` and the loadings are estimated jointly
in one Bayesian model; :math:`\mathbf{F}` is fixed to lower-trapezoidal
(:math:`f_{tl} = 0` for :math:`l > t`) for identification.

Standardization. Each series is centered on its last pre-period value and scaled
by its pre-period standard deviation before fitting; the paper credits this
pre-processing for the bulk of its sampling-efficiency gain. No post-period
information enters the scaling, so it cannot bias the treated estimate.

Shrinkage. The loadings carry a horseshoe+ prior (Bhadra et al. 2017):
:math:`\lambda_{lj}` is a product of a per-factor, a global, and a per-series
half-Cauchy scale, so factors the data do not need are pruned. This is why the
factor count :math:`L` is an upper bound rather than a choice.

Counterfactual. The treated unit's post-period outcomes are masked -- treated as
missing data and imputed by the model, which contains no treatment-effect
parameter -- so their posterior is the no-intervention counterfactual
:math:`\widehat{y}_{1t}^N`. The donors contribute all periods, pinning the
factors post-treatment; the treated loadings, fit on the pre-period, project
that structure forward. The per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^N` and the ATT is
:math:`\widehat{\tau} \coloneqq T_2^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`, each with
a full posterior credible band.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after :math:`T_0`;
   every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.bfsc_helpers.setup`) enforces
   both through :func:`~mlsynth.utils.datautils.dataprep`, translating a
   violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. A shared factor structure generates the no-intervention outcomes. Donors and
   the treated unit load on the same latent factors :math:`\mathbf{f}_t`.

   Remark. This is the interactive-fixed-effects assumption, weaker than the
   convex-hull requirement of simplex SC: the treated unit need not lie inside
   the donors' hull, only inside their factor span. If the treated loadings fall
   far outside the donors', the post-period counterfactual is an extrapolation
   and the credible band widens to say so.

#. Homoskedastic idiosyncratic noise. :math:`u_{jt} \sim \mathcal{N}(0,\sigma^2)`
   with a single :math:`\sigma` across units and time.

   Remark. The paper notes a series- or time-varying :math:`\sigma` is possible
   but hurts sampling; the scalar :math:`\sigma` is the practical default.

Inference and diagnostics
-------------------------

BFSC is inferential by construction: ``res.inference.ci_lower`` /
``res.inference.ci_upper`` give the ATT credible interval, and the
counterfactual band is on ``res.inference_detail``
(``counterfactual_lower`` / ``counterfactual_upper``). Because the factors are
sampled rather than plugged in, the band includes factor uncertainty. NUTS
diagnostics are surfaced on ``res.weights.summary_stats`` --
``nuts_accept_prob``, ``nuts_divergences``, ``max_rhat``. Read convergence on the
counterfactual and :math:`\sigma` (the identified quantities), not on the raw
loadings: a factor model is rotation/sign non-identified, so individual loadings
need not mix while the reported counterfactual does. As a placebo check, refit
with each donor as the pseudo-treated unit and compare the treated band to the
placebo distribution.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BFSC   # requires: pip install 'mlsynth[bayes]'

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
       "basedata/german_reunification.csv")
   df["treat"] = df["Reunification"].astype(int)

   res = BFSC({
       "df": df, "outcome": "gdp", "treat": "treat",
       "unitid": "country", "time": "year",
       "n_factors": 8, "seed": 0, "display_graphs": False,
   }).fit()

   print(f"mean post ATT   : {res.att:+.0f}")
   print(f"95% credible    : [{res.inference.ci_lower:+.0f}, {res.inference.ci_upper:+.0f}]")
   print(f"NUTS max r-hat  : {res.weights.summary_stats['max_rhat']:.3f}")

On West Germany this recovers the classical reunification result -- a mean
post-1990 effect near :math:`-1600` PPP-USD per capita, statistically
indistinguishable from traditional synthetic control through the 1990s and a
somewhat larger effect after 2000 -- now with a credible band that widens through
the post-period.

Verification
------------

The BFSC model is cross-validated against the author's own reference: the
appendix Stan program of Pinkney (2021), run on the German reunification panel.
The NumPyro posterior counterfactual matches the Stan posterior cell-for-cell --
correlation :math:`0.999999`, maximum discrepancy :math:`0.48\%` of the outcome
level, and :math:`\widehat{\sigma}` to four decimals -- the residual being pure
Monte-Carlo error between two independent NUTS runs. See the replication page
:doc:`replications/bfsc` and the durable case
``benchmarks/cases/bfsc_germany.py``. The estimator, config validation, the
missing-dependency guard, effect recovery, and the result contract are
unit-tested (``mlsynth/tests/test_bfsc.py``).

Core API
--------

.. automodule:: mlsynth.estimators.bfsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.BFSCConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``BFSC.fit()`` returns a
:class:`~mlsynth.utils.bfsc_helpers.structures.BFSCResults` -- an
``EffectResult`` whose standardized sub-models carry the ATT, the counterfactual
and gap paths, and the credible interval, with the posterior draws and NUTS
diagnostics on ``res.posterior``.

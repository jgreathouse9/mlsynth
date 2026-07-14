Multitask Gaussian Process Synthetic Control (MTGP)
===================================================

.. currentmodule:: mlsynth

When to use MTGP -- and when not to
-----------------------------------

Multitask Gaussian Process synthetic control (Ben-Michael et al. 2023) models
the no-intervention outcome of every unit as a Gaussian process whose kernel is
separable over time and units, then reads the treated unit's counterfactual off
the posterior. Reach for it when the untreated series are smooth functions of
time -- trends and slow drifts rather than sharp jumps -- and you want a
counterfactual whose credible band grows the further you extrapolate past the
intervention. It is a good choice when:

* You want a full posterior credible band on the counterfactual and the ATT, and
  you want that band to widen honestly the further the post-period runs from the
  last observed treated data -- the signature Gaussian-process behavior.
* The untreated outcomes are smooth in time. A GP with a squared-exponential
  kernel encodes ``nearby years look alike``; it borrows strength across time,
  not only across donors.
* Units differ in size and you want the noise to reflect it. Given a population
  (or exposure) column, MTGP scales the observation noise as
  :math:`\sqrt{1/\text{pop}}`, so a small state's noisy rate is downweighted
  relative to a large one, exactly as in the paper's rate data.

Do not use MTGP when:

* You want interpretable donor weights -- MTGP is a factor / kernel model and
  reports none. Use a weighting method (:doc:`vanillasc`, :doc:`src`) or the
  Bayesian weighting :doc:`bscm`.
* You need a dependency-free estimate. MTGP draws its posterior with NUTS
  (NumPyro, in double precision), so it needs the ``[bayes]`` optional
  dependency (``pip install 'mlsynth[bayes]'``). For a dependency-free Bayesian
  SC use :doc:`bscm`.
* The untreated series are jagged rather than smooth. A squared-exponential GP
  will over-smooth genuine high-frequency structure; a factor model that does
  not impose temporal smoothness (:doc:`bfsc`, :doc:`fma`) is the better fit.

Relation to BFSC
----------------

MTGP and :doc:`bfsc` are close cousins: both mask the treated post-period cells,
impute them from a Bayesian latent-factor model fit on the donors and the
treated pre-period, and read the counterfactual off the posterior. They differ
in what the factors are allowed to look like over time.

* :doc:`bfsc` (Pinkney) places no smoothness prior on the factors -- each period
  is a free parameter. This is MTGP with an identity time kernel.
* MTGP (Ben-Michael et al.) puts a Gaussian-process prior on the factors, so the
  factor paths are smooth functions of time. It adds a shared global time-GP
  trend on top of the low-rank factor term, and scales the noise by population.

The practical consequence is the post-period band. Because BFSC's factors are
unconstrained, its post-period imputation is pinned only by the donors; because
MTGP's factors are smooth, its band grows with distance from the last treated
observation -- the further you extrapolate, the less the GP is willing to commit.
Reach for BFSC when the shared structure is best left unconstrained; reach for
MTGP when the untreated outcomes are genuinely smooth in time and you want the
band to reflect extrapolation distance.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit and
:math:`\mathcal{N}_0` the donor pool, over units
:math:`\mathcal{N} \coloneqq \{1, \ldots, N\}`; time is
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}` with the intervention after
:math:`T_0`, pre-period :math:`\mathcal{T}_1` and post-period
:math:`\mathcal{T}_2`. MTGP models the no-intervention outcome as

.. math::

   y_{jt}^N = \alpha + \kappa_j + g_t + \sum_{r=1}^{R} \ell_{tr}\,k_{rj}
              + u_{jt}, \qquad
   u_{jt} \sim \mathcal{N}\!\left(0,\ \sigma^2 / \text{pop}_{jt}\right),

with a global intercept :math:`\alpha`, unit intercepts :math:`\kappa_j`, a
shared global time trend :math:`g_t`, and a rank-:math:`R` factor term (the
paper's :math:`R` is ``n_factors``) built from latent time-GP factors
:math:`\ell_{tr}` and unit coregionalization loadings :math:`k_{rj}`.

Gaussian-process priors. The global trend :math:`\mathbf{g} = (g_1,\ldots,g_T)`
and each latent factor :math:`\boldsymbol{\ell}_r = (\ell_{1r},\ldots,\ell_{Tr})`
are drawn from mean-zero Gaussian processes over time with squared-exponential
kernels,

.. math::

   \operatorname{Cov}(g_s, g_t)
     = \sigma_g^2 \exp\!\left(-\frac{(s-t)^2}{2\rho_g^2}\right),
   \qquad
   \operatorname{Cov}(\ell_{sr}, \ell_{tr})
     = \sigma_f^2 \exp\!\left(-\frac{(s-t)^2}{2\rho_f^2}\right),

so nearby periods are correlated and the factor paths are smooth. The
length-scales :math:`\rho_f, \rho_g` carry ``InverseGamma`` priors; the marginal
scales :math:`\sigma_f, \sigma_g` and the noise scale :math:`\sigma` carry
half-normal priors. The rank-:math:`R` term is an intrinsic-coregionalization
model: a single set of :math:`R` shared time factors, mixed per unit by the
loadings :math:`k_{rj}`.

Heteroskedastic noise. When a population column is supplied, the per-cell noise
variance is :math:`\sigma^2 / \text{pop}_{jt}` (internally the estimator carries
``inv_pop`` :math:`= \overline{\text{pop}} / \text{pop}` so :math:`\sigma`
remains on the outcome scale), matching the sampling variance of a rate computed
from a finite population. With no population column the noise is homoskedastic
(:math:`\text{pop}_{jt} \equiv 1`).

Counterfactual. The treated unit's post-period cells are masked -- treated as
missing and imputed by the model, which contains no treatment-effect parameter
-- so their posterior is the no-intervention counterfactual
:math:`\widehat{y}_{1t}^N`. The donors contribute all periods, pinning the shared
trend and factors; the treated loadings, fit on the pre-period, project that
structure forward under the GP smoothness prior. The per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^N` and the ATT is
:math:`\widehat{\tau} \coloneqq T_2^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`, each with
a full posterior credible band.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after :math:`T_0`;
   every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.mtgp_helpers.setup`) enforces
   both through :func:`~mlsynth.utils.datautils.dataprep`, translating a
   violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. The no-intervention outcomes are smooth in time. The global trend and the
   latent factors are draws from squared-exponential Gaussian processes.

   Remark. This is what distinguishes MTGP from :doc:`bfsc`: smoothness is a
   prior belief that borrows strength across adjacent periods. It is well suited
   to trending rate data and ill suited to series with genuine jumps or
   high-frequency structure, which the kernel would smooth away.

#. A low-rank coregionalization structure links the units. Donors and the
   treated unit share :math:`R` latent time factors, mixed by unit-specific
   loadings.

   Remark. As in a factor SC this is weaker than the convex-hull requirement of
   simplex SC: the treated unit need not lie inside the donors' hull, only inside
   their factor span. If the treated loadings fall outside the donors', the
   post-period is an extrapolation and -- because the factors are smooth GPs --
   the credible band widens with distance to say so.

#. Noise variance is known up to scale, inversely proportional to population.
   :math:`u_{jt} \sim \mathcal{N}(0, \sigma^2/\text{pop}_{jt})` with a single
   :math:`\sigma`.

   Remark. This matches the sampling variance of a rate from a finite population
   and lets large units carry more weight. Omit the population column for
   homoskedastic noise; the estimator then sets :math:`\text{pop}_{jt}\equiv 1`.

Inference and diagnostics
-------------------------

MTGP is inferential by construction: ``res.inference.ci_lower`` /
``res.inference.ci_upper`` give the ATT credible interval, and the counterfactual
band is on ``res.inference_detail`` (``counterfactual_lower`` /
``counterfactual_upper``). Because the trend and factors are sampled rather than
plugged in, the band includes GP uncertainty and widens post-treatment. NUTS
diagnostics are surfaced on ``res.weights.summary_stats`` -- ``nuts_accept_prob``,
``nuts_divergences``, ``max_rhat`` -- alongside the posterior-mean length-scales
(``lengthscale_f``, ``lengthscale_global``) and noise scale (``sigma_post_mean``).
Read convergence on the counterfactual and :math:`\sigma` (the identified
quantities), not on the raw loadings: a coregionalization model is
rotation/sign non-identified, so individual loadings need not mix while the
reported counterfactual does. As a placebo check, refit with each donor as the
pseudo-treated unit and compare the treated band to the placebo distribution.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MTGP   # requires: pip install 'mlsynth[bayes]'

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
       "basedata/german_reunification.csv")
   df["treat"] = df["Reunification"].astype(int)

   res = MTGP({
       "df": df, "outcome": "gdp", "treat": "treat",
       "unitid": "country", "time": "year",
       "n_factors": 5, "seed": 0, "display_graphs": False,
   }).fit()

   print(f"mean post ATT   : {res.att:+.0f}")
   print(f"95% credible    : [{res.inference.ci_lower:+.0f}, {res.inference.ci_upper:+.0f}]")
   print(f"NUTS max r-hat  : {res.weights.summary_stats['max_rhat']:.3f}")

The counterfactual tracks West Germany through the pre-period and its band widens
through the post-period, the further the extrapolation runs from 1990. Supply a
``population`` column to scale the noise by unit size.

Verification
------------

MTGP is cross-validated against the authors' own reference: the Stan program from
the paper's replication package (Ben-Michael et al. 2023), run on the study's
California panel (the Armed and Prohibited Persons System, homicide rates per
100,000 across the 50 states, 1997--2018, California treated in 2007). Ported to
NumPyro, the posterior counterfactual of California matches the Stan posterior
cell-for-cell -- correlation :math:`0.99993`, maximum discrepancy :math:`0.2\%` of
the rate level -- and the mean post-2007 ATT agrees to :math:`0.002` per 100,000
(:math:`-1.029` vs :math:`-1.031`), the residual being pure Monte-Carlo error
between two independent NUTS runs. See the replication page
:doc:`replications/mtgp` and the durable case
``benchmarks/cases/mtgp_california.py``. The estimator, config validation, the
missing-dependency guard, effect recovery, the post-period band-widening, the
heteroskedastic-population branch, and the result contract are unit-tested
(``mlsynth/tests/test_mtgp.py``).

Core API
--------

.. automodule:: mlsynth.estimators.mtgp
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MTGPConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MTGP.fit()`` returns a
:class:`~mlsynth.utils.mtgp_helpers.structures.MTGPResults` -- an ``EffectResult``
whose standardized sub-models carry the ATT, the counterfactual and gap paths,
and the credible interval, with the posterior draws, GP length-scales, and NUTS
diagnostics on ``res.posterior``.

Bayesian Synthetic Control of Martinez & Vives-i-Bastida (MVBBSC)
=================================================================

.. currentmodule:: mlsynth

When to use MVBBSC -- and when not to
-------------------------------------

MVBBSC is the Bayesian synthetic control of Martinez and Vives-i-Bastida (2024).
It keeps the standard synthetic-control model -- the untreated outcome of the
treated unit is a simplex-weighted average of the donor pool -- but replaces the
single optimized weight vector with a posterior over weights, so every quantity
it reports (the counterfactual, the per-period effect, the ATT) arrives with a
credible band rather than a point. Reach for it when you want the interpretable,
non-extrapolating fit of a classical synthetic control and honest uncertainty on
the effect, drawn from one coherent Bayesian model. It is a good choice when:

* You want donor weights that are non-negative and sum to one -- the same
  convex-combination the classical method reports -- but with a full posterior,
  not a single solve.
* You want a principled interval. Under the paper's conditions the Bayesian
  credible interval is also a valid frequentist confidence interval (a
  Bernstein-von Mises result), so the band is usable inference, not just a
  visual aid.
* The treated unit lies inside (or near) the donors' convex hull, so a weighted
  average is the right model and a good pre-period fit is attainable.

Do not use MVBBSC when:

* The treated unit is outside the donor hull. A hard simplex cannot reach it and
  the pre-period fit will be poor; a factor model (:doc:`bfsc`, :doc:`fma`) or an
  unconstrained-weight method (:doc:`bscm`) is the better model there.
* You need a dependency-free estimate. MVBBSC draws its posterior with NUTS
  (NumPyro), so it needs the ``[bayes]`` optional dependency
  (``pip install 'mlsynth[bayes]'``). For a dependency-free Bayesian SC use
  :doc:`bscm` (a pure-numpy Gibbs sampler).

The Bayesian SC family
----------------------

mlsynth carries four Bayesian synthetic-control estimators; they differ in what
they place a prior on and whether the weights are constrained to the simplex.

* :doc:`bscm` (Kim, Lee and Gupta) -- shrinkage (horseshoe or spike-and-slab)
  on unconstrained donor weights; a pure-numpy Gibbs sampler.
* :doc:`bvss` (Xu and Zhou) -- spike-and-slab donor selection on a soft simplex
  whose tightness is learned; reports inclusion probabilities.
* :doc:`bfsc` (Pinkney) -- a Bayesian latent-factor model, not a donor
  weighting; reports a counterfactual band and no donor weights.
* :doc:`mvbbsc` (Martinez and Vives-i-Bastida) -- a uniform prior over the
  hard simplex, standardized internally, with the frequentist-inference
  (Bernstein-von Mises) guarantee.

Reach for MVBBSC when you want the classical convex-combination model with a
principled interval; reach for :doc:`bvss` when you additionally want the model
to select which donors enter; reach for :doc:`bscm` when you are willing to drop
the simplex for shrinkage on unconstrained weights; reach for :doc:`bfsc` when a
shared factor structure, not a weighted average, is the right model.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit and
:math:`\mathcal{N}_0 \coloneqq \{2, \ldots, J+1\}` the donor pool; time is
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}` with the intervention after
:math:`T_0`, pre-period :math:`\mathcal{T}_1` and post-period
:math:`\mathcal{T}_2`. Write :math:`y_{1t}` for the treated outcome and
:math:`\mathbf{y}_{0t} = (y_{2t}, \ldots, y_{J+1,t})^{\top}` for the donor
outcomes at :math:`t`. MVBBSC models the no-intervention outcome as a
simplex-weighted average of the donors,

.. math::

   y_{1t}^N = \mathbf{y}_{0t}^{\top}\mathbf{w} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2),
   \qquad \mathbf{w} \in \Delta^{J},

where :math:`\Delta^{J}` is the unit simplex
(:math:`w_j \ge 0,\ \sum_j w_j = 1`).

Prior. The paper's characterization of when the risk minimizer is a synthetic
control motivates placing the weights in the simplex; MVBBSC does so with a
uniform Dirichlet prior and a weakly-informative scale,

.. math::

   \mathbf{w} \sim \mathrm{Dirichlet}(\mathbf{1}_J),
   \qquad \sigma \sim \mathrm{HalfNormal}(1).

Standardization. Before fitting, the treated series and every donor series are
centered on their pre-period mean and scaled by their pre-period standard
deviation; the counterfactual is transformed back to the outcome scale
afterwards. No post-period information enters the scaling, so it cannot bias the
treated estimate, and because the fit happens on standardized data the estimate
is invariant to the units of the outcome.

Counterfactual. The posterior is drawn with NUTS. The counterfactual is the
posterior-predictive draw of the untreated outcome,
:math:`\widehat{y}_{1t}^{N,(m)} = \mathbf{y}_{0t}^{\top}\mathbf{w}^{(m)} +
\eta_t^{(m)}` with :math:`\eta_t^{(m)} \sim \mathcal{N}(0, \sigma^{(m)2})` for
each posterior draw :math:`m`, so its pointwise quantiles are a credible band.
The per-period effect is :math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^N` and
the ATT is
:math:`\widehat{\tau} \coloneqq T_2^{-1}\sum_{t\in\mathcal{T}_2}\tau_t`, each with
a full posterior credible band.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after :math:`T_0`;
   every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.mvbbsc_helpers.setup`)
   enforces both through :func:`~mlsynth.utils.datautils.dataprep`, translating a
   violation (including a second treated cohort) to
   :class:`~mlsynth.exceptions.MlsynthDataError`.

#. The treated unit is a convex combination of the donors. There exist simplex
   weights under which the donors reproduce the treated unit's pre-period path.

   Remark. This is the classical synthetic-control design assumption. When it
   fails -- the treated unit outside the donors' hull -- the hard simplex cannot
   track the pre-period, the fit is poor, and (as the paper stresses) the
   estimate is likely biased. A poor pre-period fit is the diagnostic; read it
   before trusting the effect.

#. Gaussian idiosyncratic noise. :math:`\varepsilon_t \sim \mathcal{N}(0,
   \sigma^2)` with a single :math:`\sigma`.

   Remark. The Gaussian likelihood is what the paper's Bernstein-von Mises
   argument uses to link the Bayesian posterior predictive to the frequentist
   sampling distribution as :math:`T_0, J \to \infty`.

Inference and diagnostics
-------------------------

MVBBSC is inferential by construction. ``res.inference.ci_lower`` /
``res.inference.ci_upper`` give the ATT credible interval, and the counterfactual
band is on ``res.inference_detail`` (``counterfactual_lower`` /
``counterfactual_upper``). Under the paper's conditions this credible interval is
also a valid confidence interval, so it can be read as frequentist inference.
NUTS diagnostics are surfaced on ``res.weights.summary_stats`` --
``nuts_accept_prob``, ``nuts_divergences``, ``max_rhat`` -- alongside
``sigma_post_mean``. The posterior-mean simplex weights are on
``res.weights.donor_weights``. Read the pre-period fit
(``res.fit_diagnostics.rmse_pre``) first: a large value means the treated unit is
not in the donors' hull and the effect should not be trusted.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MVBBSC   # requires: pip install 'mlsynth[bayes]'

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/main/"
       "basedata/german_reunification.csv")
   df["treat"] = df["Reunification"].astype(int)

   res = MVBBSC({
       "df": df, "outcome": "gdp", "treat": "treat",
       "unitid": "country", "time": "year",
       "seed": 0, "display_graphs": False,
   }).fit()

   print(f"pre-period RMSE : {res.fit_diagnostics.rmse_pre:.1f}")
   print(f"mean post ATT   : {res.att:+.0f}")
   print(f"95% credible    : [{res.inference.ci_lower:+.0f}, {res.inference.ci_upper:+.0f}]")
   print(f"NUTS max r-hat  : {res.weights.summary_stats['max_rhat']:.3f}")

On West Germany this fits the pre-1990 path to an RMSE near :math:`62` (in
PPP-USD per capita) and estimates a mean post-1990 effect near :math:`-2080` --
about a :math:`7.5\%` decline in per-capita GDP -- with a 95% credible band of
roughly :math:`[-2600, -1600]` that widens through the post-period as the
counterfactual extrapolates. The posterior weight is spread across Norway, Italy,
Austria, and the USA.

Verification
------------

The MVBBSC model is cross-validated against the reference ``bsynth`` R package
(Martinez and Vives-i-Bastida's own implementation). On the German reunification
panel a fresh NumPyro fit matches the ``bsynth`` posterior to Monte-Carlo error:
pre-period RMSE :math:`62.2` vs :math:`62.2`, mean post-ATT :math:`-2078` vs
:math:`-2075`, and the 95% credible band agrees year-by-year to within a few
percent (ratios :math:`0.95`--:math:`1.07`), with in-sample coverage of the
observed series at :math:`96.7\%` against the nominal :math:`95\%`. See the
replication page :doc:`replications/mvbbsc` and the durable case
``benchmarks/cases/mvbbsc_germany.py``. The estimator, config validation, the
missing-dependency guard, effect recovery, scale invariance, and the result
contract are unit-tested (``mlsynth/tests/test_mvbbsc.py``).

Core API
--------

.. automodule:: mlsynth.estimators.mvbbsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MVBBSCConfig
   :members:
   :undoc-members:

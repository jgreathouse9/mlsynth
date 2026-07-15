Bayesian Penalized Synthetic Control under Spillovers (BPSCS)
=============================================================

.. currentmodule:: mlsynth

When to use BPSCS -- and when not to
------------------------------------

Bayesian Penalized Synthetic Control under Spillovers (Fernandez-Morales,
Oganisian and Lee 2026) is a Bayesian synthetic control for the situation where
some donors are contaminated by the treatment. When a policy spills over onto nearby areas --
cross-border shopping after a city soda tax, say -- those neighbours are no longer
clean controls: their post-period outcomes move because of the treatment, which
biases an ordinary synthetic control. BPSCS keeps such donors in the pool but
shrinks their contribution, using a utility that blends covariate similarity and
spatial distance to the treated unit. Reach for it when:

* Your donors are geographic units and the intervention plausibly leaks into the
  ones nearest the treated unit, so excluding them wholesale throws away
  information but trusting them fully biases the estimate.
* You want a full posterior credible band on the counterfactual and the effect,
  not a single line.
* You have per-unit spatial coordinates and baseline covariates, and you want the
  donor down-weighting to be data-driven from those rather than a hand-picked
  exclusion list.

Do not use BPSCS when:

* There is no spillover and no spatial structure to exploit -- a plain
  weighting or factor SC (:doc:`vanillasc`, :doc:`bfsc`) is simpler and the
  utility term buys you nothing.
* You need a dependency-free estimate. BPSCS draws its posterior with NUTS
  (NumPyro), so it needs the ``[bayes]`` optional dependency
  (``pip install 'mlsynth[bayes]'``).
* You can enumerate exactly which donors are contaminated and by how much -- then
  a structural spillover model (:doc:`spillsynth`, :doc:`spsydid`) that estimates
  the spillover directly may serve better. BPSCS assumes only that spillover risk
  is a monotone function of distance, and down-weights accordingly.

Relation to the other spillover and Bayesian estimators
-------------------------------------------------------

BPSCS sits at the intersection of two families. Like :doc:`spillsynth` and
:doc:`spsydid` it targets spillover onto the donor pool; unlike them it does not
model the spillover structurally -- it treats spillover risk as increasing with
proximity and mitigates it through the prior. Like :doc:`bscm`, :doc:`bfsc` and
:doc:`mtgp` it is a Bayesian SC with a full posterior; unlike :doc:`bscm` (which
shrinks donor weights from the data fit alone) its shrinkage is informed by an
external covariate-and-distance utility. Reach for BPSCS specifically when the
threat is distance-driven contamination of the controls.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit and
:math:`\mathcal{N}_0` the donor pool over units
:math:`\mathcal{N} \coloneqq \{1, \ldots, N\}`; time is
:math:`t \in \mathcal{T} \coloneqq \{1, \ldots, T\}` with the intervention after
:math:`T_0`. Each unit carries baseline covariates :math:`\mathbf{X}_i` and
spatial coordinates :math:`\mathbf{P}_i`. BPSCS models the treated unit's
no-intervention outcome as an autoregressive linear synthetic control,

.. math::

   y_{1t}^N = \psi\, y_{1,t-1}^N + \mathbf{X}_1^{\top}\boldsymbol{\phi}
              + \sum_{i\in\mathcal{N}_0} \beta_i\, y_{it}
              + u_{t}, \qquad u_t \sim \mathcal{N}(0, \sigma^2),

with an autoregressive coefficient :math:`\psi`, a treated-covariate effect
:math:`\boldsymbol{\phi}`, and donor coefficients :math:`\boldsymbol{\beta}` that
play the role of synthetic-control weights but are unconstrained (they may be
negative and need not sum to one).

Utility. Each donor's coefficient is shrunk according to a utility that blends
covariate similarity and spatial distance to the treated unit,

.. math::

   u^C_{i} = \kappa_d\, \frac{1}{1 + \lVert \mathbf{X}_i - \mathbf{X}_1 \rVert}
             + (1 - \kappa_d)\, \frac{\lVert \mathbf{P}_i - \mathbf{P}_1 \rVert}{K},

where :math:`\kappa_d \in [0,1]` trades off the two terms and :math:`K` is the
maximum distance (a normalizer). With :math:`\kappa_d = 0` only spatial distance
matters (the paper's emphasized spillover regime); with :math:`\kappa_d = 1` only
covariate similarity does.

Two priors. The utility scales a shrinkage prior on :math:`\beta_i` in one of two
ways. The distance-horseshoe (``dhs``) gives a continuous shrinkage,
:math:`\lambda_i \sim \mathrm{Half\text{-}Cauchy}(0, u^C_i)`, so a low-utility
(spatially close) donor gets a small scale and is pulled toward zero. The
distance-spike-and-slab (``ds2``) applies a hard cutoff: donors with
:math:`u^C_i` below an inclusion radius :math:`\rho` (a quantile of the utility
scores) fall into a spike at zero, the rest into a diffuse slab.

Standardization. Outcomes are standardized on the pre-period and the estimated
effect is returned on the treated unit's original scale; covariates are z-scored
across units before the utility is computed.

Counterfactual. The treated post-period outcomes are imputed by a free-running
forward simulation of the fitted model, so their posterior is the no-intervention
counterfactual :math:`\widehat{y}_{1t}^N`; the per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}^N`. Point summaries use the
posterior median, which is robust to the heavy upper tail the recursive
simulation can produce (see the remark under Inference).

Assumptions
-----------

#. Single treated unit, balanced panel with per-unit covariates and coordinates.
   One unit is treated after :math:`T_0`; every unit is observed at every period;
   each carries the covariate and coordinate columns named in the config.

   Remark. The setup boundary (:mod:`mlsynth.utils.bpscs_helpers.setup`) enforces
   the panel through :func:`~mlsynth.utils.datautils.dataprep` and validates the
   covariate / coordinate columns, translating violations to
   :class:`~mlsynth.exceptions.MlsynthDataError` /
   :class:`~mlsynth.exceptions.MlsynthConfigError`.

#. Spillover risk increases with proximity. A donor's chance of being
   contaminated is a monotone function of its distance to the treated unit.

   Remark. This is the paper's core, minimal assumption -- weaker than naming the
   contaminated donors, but it does require that distance is a sensible proxy for
   exposure. Where spillover is not distance-driven, the utility mis-targets the
   shrinkage and residual bias can remain.

#. A shared linear structure with autoregressive dynamics generates the
   no-intervention outcome. The treated series is a linear combination of donor
   outcomes plus its own lag and baseline covariates.

   Remark. The unconstrained :math:`\boldsymbol{\beta}` (unlike simplex SC) lets
   the treated unit sit outside the donors' convex hull, at the usual cost that
   the counterfactual can extrapolate; the shrinkage prior guards against
   over-fitting when :math:`T_0` is short.

Inference and diagnostics
-------------------------

BPSCS is inferential by construction: ``res.inference.ci_lower`` /
``res.inference.ci_upper`` give the ATT credible interval, and the counterfactual
band is on ``res.inference_detail`` (``counterfactual_lower`` /
``counterfactual_upper``). The signed donor coefficients are surfaced as
``res.weights.donor_weights`` (posterior medians), and the prior, the importance
weight :math:`\kappa_d`, the inclusion radius, and NUTS diagnostics
(``nuts_accept_prob``, ``nuts_divergences``, ``max_rhat``) are on
``res.weights.summary_stats``.

Remark on the point estimate. The counterfactual is a recursive forward
simulation, so a posterior draw with a large autoregressive coefficient can make
the imputed path explode; the posterior *mean* of the effect is therefore not a
usable summary (a handful of draws dominate it). BPSCS reports the posterior
*median* for the counterfactual and the ATT, which is stable to that tail; the
credible band's far edge is genuinely wide in the late post-period and is
reported as such rather than papered over.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BPSCS   # requires: pip install 'mlsynth[bayes]'

   # df has columns: unit, period, sales, treat (0/1), plus per-unit
   # baseline covariates (income, density) and coordinates (lat, lon)
   res = BPSCS({
       "df": df, "outcome": "sales", "treat": "treat",
       "unitid": "unit", "time": "period",
       "covariates": ["income", "density"], "coords": ["lat", "lon"],
       "prior": "dhs", "kappa_d": 0.0,       # spatial-only shrinkage
       "seed": 0, "display_graphs": False,
   }).fit()

   print(f"median post ATT : {res.att:+.3f}")
   print(f"95% credible    : [{res.inference.ci_lower:+.3f}, {res.inference.ci_upper:+.3f}]")
   print(f"donors shrunk    : {sum(abs(v) < 1e-3 for v in res.weights.donor_weights.values())}")

Set ``prior="ds2"`` for the hard-cutoff spike-and-slab, and raise ``kappa_d``
toward 1 to weight covariate similarity over distance.

Verification
------------

BPSCS is cross-validated against the authors' own reference: the Stan programs
(``sc_dhs.stan`` / ``sc_ds2.stan``) from their replication repository, run live
via ``rstan`` on the shipped Philadelphia beverage-tax example panel. Ported to
NumPyro, BPSCS matches the reference posterior credible band cell-for-cell -- the
counterfactual band tracks with correlation :math:`0.997`--:math:`0.9995` for the
distance-horseshoe and :math:`0.969`--:math:`0.998` for the spike-and-slab, the
residual being Monte-Carlo error between two independent NUTS runs (amplified only
in the extreme lower tail of the free-running counterfactual, where the model is
numerically fragile). See the replication page :doc:`replications/bpscs`. Because
the reference is GPL-licensed it is fetched and run at cross-check time rather
than redistributed; the durable self-contained case
``benchmarks/cases/bpscs_synthetic.py`` checks effect recovery and distance-based
shrinkage on a simulated spatial panel. The estimator, config validation, the
missing-dependency guard, effect recovery, both priors, and the result contract
are unit-tested (``mlsynth/tests/test_bpscs.py``).

Core API
--------

.. automodule:: mlsynth.estimators.bpscs
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.BPSCSConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``BPSCS.fit()`` returns a
:class:`~mlsynth.utils.bpscs_helpers.structures.BPSCSResults` -- an ``EffectResult``
whose standardized sub-models carry the median ATT, the counterfactual and gap
paths, the credible interval, and the signed donor coefficients, with the
posterior draws and NUTS diagnostics on ``res.posterior``.

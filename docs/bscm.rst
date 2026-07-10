Bayesian Synthetic Control Methods (BSCM)
=========================================

.. currentmodule:: mlsynth

When to use BSCM -- and when not to
-----------------------------------

Bayesian Synthetic Control Methods (Kim, Lee and Gupta 2020) are for the case
where you want a synthetic control that both fits a hard-to-match treated unit
and reports honest uncertainty. The canonical synthetic control restricts the
donor weights to the unit simplex (non-negative, summing to one), which keeps
the counterfactual inside the convex hull of the donors but cannot track a
treated unit near or outside that hull, and it offers no built-in measure of
how precisely the effect is estimated. BSCM relaxes the simplex and places a
Bayesian shrinkage prior on the donor weights instead, so the counterfactual
can extrapolate in a regularised way and every quantity comes with a posterior.
It is a good choice when:

* You have a single treated unit and a donor pool where a simplex synthetic
  control leaves a visible pre-treatment gap, and you are willing to let the
  counterfactual extrapolate a little beyond the donor hull.
* You want a credible interval on the effect, not just a point estimate or a
  placebo distribution. Because the sampler returns the full posterior, the ATT
  and the counterfactual path both carry credible bands directly.
* You have many donors relative to pre-periods (the "large p, small n" regime),
  where an unpenalised regression would overfit. The shrinkage prior handles
  this without a separate tuning step.

BSCM offers two priors:

* ``horseshoe`` -- a global-local continuous shrinkage prior. The default:
  fast, and it shrinks noise donors hard toward zero while letting genuine
  signal donors escape via heavy tails.
* ``spike_slab`` -- a discrete variable-selection prior. Additionally reports a
  per-donor inclusion probability :math:`P(\gamma_j = 1 \mid y)`, so you can
  read off which donors the model considers real signal.

Do not use BSCM when:

* A simplex synthetic control already fits well and you want an interpretable
  convex weighting. Use canonical SCM (:doc:`vanillasc`) or :doc:`tssc`; BSCM's
  weights can be negative and do not sum to one.
* You want a soft simplex kept, with Bayesian inference on top. That is
  :doc:`bvss` (Xu-Zhou), which layers a spike-and-slab on a soft simplex prior.
* You need a deterministic, seed-free point estimate with a risk-criterion
  weighting. That is :doc:`smc` (Zhu), the frequentist per-donor matching
  cousin.
* There are multiple treated units, spillovers, or a continuous dose -- BSCM
  encodes a single binary intervention on one unit.

Notation
--------

We use the synthetic-control canon. Let :math:`j = 1` denote the treated unit
and :math:`\mathcal{N}_0 \coloneqq \{2, \ldots, J + 1\}` the donor pool. The
intervention takes effect after period :math:`T_0`. Write
:math:`Y_{1t}` for the treated unit's outcome and :math:`Y_{jt}` for donor
:math:`j`'s. BSCM models the treated pre-treatment outcome as a linear
combination of the donors with an intercept,

.. math::

   Y_{1t} = \beta_0 + \sum_{j=2}^{J+1} \beta_j\, Y_{jt} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2),

for :math:`t \le T_0`. There is no simplex constraint: the weights
:math:`\beta_j` may be negative and need not sum to one. The counterfactual for
the treated unit under no treatment is the fitted line extended past
:math:`T_0`,

.. math::

   \widehat{Y}_{1t}(0) = \beta_0 + \sum_{j=2}^{J+1} \beta_j\, Y_{jt},

and the treatment effect at :math:`t > T_0` is
:math:`Y_{1t} - \widehat{Y}_{1t}(0)`.

Priors. The weights carry a Bayesian shrinkage prior that identifies them in
place of the simplex constraint. The horseshoe (Carvalho, Polson and Scott
2010) is

.. math::

   \beta_j \mid \lambda_j \sim \mathcal{N}(0, \lambda_j^2), \qquad
   \lambda_j \mid \tau \sim \mathcal{C}^+(0, \tau), \qquad
   \tau \mid \sigma \sim \mathcal{C}^+(0, \sigma),

a global scale :math:`\tau` that shrinks every weight, and local scales
:math:`\lambda_j` with heavy (half-Cauchy) tails that let true signals escape.
The spike-and-slab (George and McCulloch 1993) instead mixes a tight spike at
zero with a diffuse slab,

.. math::

   \beta_j \sim \gamma_j\, \mathcal{N}(0, \tau_j^2)
             + (1 - \gamma_j)\, \mathcal{N}(0, c^2),

with a small fixed spike variance :math:`c^2` and an inclusion indicator
:math:`\gamma_j`; its posterior mean is the inclusion probability.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after
   :math:`T_0`; every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.bscm_helpers.setup`) enforces
   both, translating a violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. At least two pre-treatment periods. The regression needs pre-period
   variation to fit the donor weights.

   Remark. In the "large p, small n" regime (donors not fewer than
   pre-periods) the pre-fit will be near-perfect -- the model interpolates --
   and the shrinkage prior, not the data, controls the counterfactual. Read the
   credible band, not the point pre-fit, as the measure of confidence.

#. Regularised extrapolation is acceptable. The weights can leave the simplex,
   so the counterfactual can extrapolate; the shrinkage prior bounds how much.

   Remark. If extrapolation is unacceptable (you need a convex, interpretable
   weighting), BSCM is the wrong tool -- use a simplex SCM.

Inference and diagnostics
-------------------------

BSCM returns a full posterior. The ATT is the posterior mean of the mean
post-treatment gap, and ``res.att_ci`` is its credible interval at level
:math:`1 - \text{ci\_alpha}`. The counterfactual path carries pointwise credible
bands (``res.inference_detail.counterfactual_lower`` /
``counterfactual_upper``). The posterior-mean donor weights ``res.donor_weights``
may be negative; for the ``spike_slab`` prior, ``res.inclusion_probs`` gives the
per-donor posterior probability of being a signal. Multiple chains
(``chains``) are sampled and pooled, enabling convergence checks. The sampler is
seeded (``seed``), so a given call is reproducible.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BSCM

   df = pd.read_csv("basedata/basque_data.csv")
   df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                  & (df["year"] >= 1970)).astype(int)

   res = BSCM({
       "df": df, "outcome": "gdpcap", "treat": "treat",
       "unitid": "regionname", "time": "year",
       "prior": "horseshoe", "seed": 2019,
       "display_graphs": False,
   }).fit()

   lo, hi = res.att_ci
   print(f"pre-period RMSE : {res.pre_rmse:.3f}")
   print(f"ATT            : {res.att:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

On the Basque Country study of ETA terrorism, the horseshoe BSCM tracks the
pre-treatment path near-exactly (the donor pool is large relative to the 15
pre-periods, so the model interpolates) and then estimates a post-1970 ATT of
about :math:`-0.7` (thousand-1986-USD per capita) with a credible interval that
excludes or straddles zero depending on the horizon -- the familiar
Abadie-Gardeazabal shape, recovered here by a Bayesian regression rather than a
simplex, with a posterior attached.

Switch ``"prior": "spike_slab"`` to additionally read off which donors are
selected:

.. code-block:: python

   res = BSCM({..., "prior": "spike_slab"}).fit()
   for u, p in sorted(res.inclusion_probs.items(), key=lambda kv: -kv[1])[:3]:
       print(f"  {u:<28s} P(include) {p:.2f}")

Verification
------------

BSCM's weight computation is cross-validated against the authors' reference
Stan implementation (``clarencejlee/bscm``: ``Horseshoe_Publish.stan`` and
``Spike_Slab_Publish.stan``, sampled with rstan). On the identical Basque
matching problem (treatment 1970), the pure-numpy Gibbs sampler reproduces the
reference posterior: the counterfactual paths agree to a correlation above
0.999 and the ATTs match within Monte-Carlo noise for both priors. The paper's
own empirical application uses proprietary Nielsen soda-tax data, so the durable
case validates on the public Basque panel instead. See the replication page
:doc:`replications/bscm` and the durable case ``benchmarks/cases/bscm_basque.py``.
The sampler, setup, inference, plotter and result contract are unit-tested
(``mlsynth/tests/test_bscm.py``, full coverage).

Core API
--------

.. automodule:: mlsynth.estimators.bscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.BSCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``BSCM.fit()`` returns a
:class:`~mlsynth.utils.bscm_helpers.structures.BSCMResults` -- an
``EffectResult`` whose standardized sub-models carry the ATT, counterfactual,
gap and pre-RMSE, with the posterior draws, per-draw ATT samples, counterfactual
credible bands and (for ``spike_slab``) inclusion probabilities on the typed
fields. The prepared NumPy panel is exposed as a
:class:`~mlsynth.utils.bscm_helpers.structures.BSCMInputs`.

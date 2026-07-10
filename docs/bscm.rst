Bayesian Synthetic Control Methods (BSCM)
=========================================

.. currentmodule:: mlsynth

When to use BSCM -- and when not to
-----------------------------------

Bayesian Synthetic Control Methods (Kim, Lee and Gupta 2020) are for the case
where you want a synthetic control that fits a hard-to-match treated unit, works
when the donor pool is large relative to the number of pre-treatment periods,
and reports honest finite-sample uncertainty. The canonical synthetic control
restricts the donor weights to the unit simplex (non-negative, summing to one),
which keeps the counterfactual inside the convex hull of the donors but cannot
track a treated unit near or outside that hull, and offers no mechanism for the
"large p, small n" problem. BSCM relaxes the simplex and places a Bayesian
shrinkage prior on the donor weights, so the counterfactual can extrapolate in a
regularised way and every quantity comes with a posterior -- credible intervals
that fall out of the fit rather than a separate inference step (the classical
synthetic control now has its own valid inference options too; see
:doc:`vanillasc`). It is a good choice when:

* You have a single treated unit and a donor pool where a simplex synthetic
  control leaves a visible pre-treatment gap, and you are willing to let the
  counterfactual extrapolate a little beyond the donor hull.
* You have many donors relative to pre-periods (the "large p, small n" regime),
  where an unpenalised regression would be under-determined. The shrinkage
  prior identifies the weights without a separate tuning step.
* You want a credible interval on the effect. Because the sampler returns the
  full posterior, the ATT and the counterfactual path both carry credible bands
  directly -- valid in finite samples, without asymptotics or a placebo test.

BSCM offers two priors:

* ``horseshoe`` -- a global-local continuous shrinkage prior. The default:
  fast, and it shrinks noise donors hard toward zero while letting genuine
  signal donors escape via heavy tails.
* ``spike_slab`` -- a discrete variable-selection prior. Additionally reports a
  per-donor inclusion probability :math:`P(\gamma_j = 1 \mid y)`, so you can
  read off which donors the model considers signal.

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

Notation and the causal estimand
---------------------------------

We use the synthetic-control canon. Suppose we observe :math:`J + 1` units over
:math:`T` periods, with unit :math:`0` treated after period :math:`T_0` and
units :math:`\{1, \ldots, J\}` untreated (the donor pool). Let
:math:`Y_{jt}^{N}` and :math:`Y_{jt}^{I}` be the potential outcomes for unit
:math:`j` at time :math:`t` without and with the intervention, and let
:math:`D_{jt} = 1` if unit :math:`j` is under treatment at :math:`t` (only
:math:`D_{0t} = 1` for :math:`t > T_0`). The observed outcome is

.. math::

   Y_{jt} = Y_{jt}^{N} + \alpha_{jt}\, D_{jt},
   \qquad \alpha_{jt} = Y_{jt}^{I} - Y_{jt}^{N}.

The object of interest is the treatment effect on the treated unit,

.. math::

   \alpha_{0t} = \underbrace{Y_{0t}^{I}}_{\text{observed}}
                 - \underbrace{Y_{0t}^{N}}_{\text{unobserved}},
   \qquad t > T_0.

Since :math:`Y_{0t}^{N}` is never observed after treatment, causal inference
reduces to predicting the treated unit's untreated path from the donor pool --
the out-of-sample accuracy of that prediction is what a synthetic control lives
or dies by.

The standard synthetic control (Abadie, Diamond and Hainmueller 2010) solves a
constrained least-squares problem over the pre-treatment window,

.. math::

   \widehat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta \in \Lambda}
     \sum_{t=1}^{T_0} \Bigl( Y_{0t} - \beta_0
        - \textstyle\sum_{j=1}^{J} \beta_j Y_{jt} \Bigr)^2,
   \qquad
   \Lambda = \Bigl\{ \beta_0 = 0,\ \beta_j \ge 0,\ \textstyle\sum_j \beta_j = 1 \Bigr\}.

Kim, Lee and Gupta, writing in 2020, identify three limitations of this
program. First, the constraints :math:`\Lambda` are restrictive: the
no-intercept and sum-to-one/non-negativity rules force the synthetic control
into the convex hull of the donors and implicitly assume the treated unit is
positively correlated with them, biasing the estimator when the treated unit is
extremal. Second, the classical estimator's inference rested on a placebo
(permutation) test whose validity relies on a symmetry assumption that is
generally violated. Third, there is no explicit mechanism for the "large p,
small n" or sparsity problem, where the number of donors approaches or exceeds
the number of pre-periods and the unconstrained problem has no unique solution.

The second point should be read in its 2020 context, not as a current verdict.
A substantial literature has since given the classical synthetic control a
formal inference toolkit -- conformal test-inversion intervals (Chernozhukov,
Wüthrich and Zhu), prediction intervals (Cattaneo, Feng and Titiunik),
subsampling and :math:`t`-test approaches, and refined placebo procedures --
several of which mlsynth exposes directly on the ordinary synthetic control
through the ``inference`` option of :doc:`vanillasc`. So classical SCM is no
longer without valid inference. What BSCM offers is a different and internally
coherent route: finite-sample credible intervals that fall out of the posterior
as a byproduct of estimation, for the weights, the counterfactual and the ATT at
once, without a separate inference procedure.

The Bayesian synthetic-control framework
----------------------------------------

BSCM drops the constraint set and writes the pre-treatment relationship as an
ordinary linear model with an intercept,

.. math::

   Y_{0t} = \beta_0 + \sum_{j=1}^{J} \beta_j\, Y_{jt} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2), \quad t \le T_0,

and regularises the weights :math:`\boldsymbol\beta` with a prior rather than a
hard constraint. The counterfactual is the fitted line carried past
:math:`T_0`,

.. math::

   \widehat{Y}_{0t}^{N} = \beta_0 + \sum_{j=1}^{J} \beta_j\, Y_{jt},

so the effective coefficient on donor :math:`j` is :math:`\beta_j`, which may be
negative and need not sum to one.

The framework is unifying: the choice of prior on :math:`\boldsymbol\beta`
recovers a whole family of estimators (Kim, Lee and Gupta 2020, Table 1). An
uninformative prior with the simplex constraints reproduces the standard SCM; a
normal prior gives ridge regression (the augmented SCM of Ben-Michael, Feller
and Rothstein); a Laplace prior gives the lasso (the ArCo of Carvalho, Masini
and Medeiros); a scale-mixture-of-normals prior gives the elastic net (the
modified SCM of Doudchenko and Imbens). BSCM contributes two priors whose
frequentist analogues do not exist:

The horseshoe prior (Carvalho, Polson and Scott 2010) is the global-local
shrinkage prior

.. math::

   \beta_j \mid \lambda_j \sim \mathcal{N}(0, \lambda_j^2), \qquad
   \lambda_j \mid \tau \sim \mathcal{C}^+(0, \tau), \qquad
   \tau \mid \sigma \sim \mathcal{C}^+(0, \sigma), \qquad
   \sigma \sim \mathcal{C}^+(0, 10),

where :math:`\mathcal{C}^+` is the half-Cauchy. The global scale :math:`\tau`
shrinks every weight toward zero; the local scales :math:`\lambda_j` have heavy
(half-Cauchy) tails that let a genuine signal escape that shrinkage. The
resulting marginal prior on :math:`\beta_j` has a tall spike at zero and flat
tails -- it separates noise donors (shrunk hard) from signal donors (left
alone), which is exactly the behaviour a sparse synthetic control needs.

The spike-and-slab prior (George and McCulloch 1993; Mitchell and Beauchamp
1988) is the discrete mixture

.. math::

   \beta_j \sim \gamma_j\, \mathcal{N}(0, \tau_j^2)
             + (1 - \gamma_j)\, \mathcal{N}(0, c^2), \qquad
   \gamma_j \sim \text{Uniform}(0, 1), \qquad
   \tau_j^2 \sim \text{Inv-Gamma}\!\left(\tfrac12, \tfrac12\right),

with a small fixed spike variance :math:`c^2` (the paper uses
:math:`c^2 = 0.001`). The spike concentrates a donor at zero; the slab is a
diffuse prior for donors that carry signal; the inclusion indicator
:math:`\gamma_j` has posterior mean equal to the probability that donor
:math:`j` is a signal. The spike-and-slab is the reference sparse-estimation
prior; the horseshoe is a continuous relaxation of it that is typically faster
to sample and, when no small set of donors clearly dominates, shrinks more
gracefully.

Three advantages follow from being Bayesian. First, finite-sample inference
comes for free: because MCMC returns samples from the posterior, credible
intervals for the weights, the counterfactual and the ATT are all available
directly from one fit, with no asymptotics and no separate inference procedure,
and this holds even when :math:`n` is small. This is not a claim that the
frequentist synthetic control cannot do inference -- as noted above, it now can,
several ways -- but a coherent alternative in which the interval is a byproduct
of estimation rather than a bolt-on, and which sidesteps the instability of
bootstrapping a penalised regression whose coefficients sit near zero. Second,
the shrinkage prior is a built-in mechanism for the "large p, small n" and
sparsity problems -- the third limitation -- so BSCM remains well posed when
:math:`J \gtrsim T_0`. Third, the penalty (the shrinkage scales) and the
coefficients are estimated jointly in one posterior, avoiding the
"double-shrinkage" bias of the frequentist elastic net. In the paper's simulations, across data-generating processes that are
sparse, dense, low-:math:`n`, or that violate the SCM constraints, the horseshoe
and spike-and-slab almost always match or beat the standard SCM and the other
Bayesian regularised regressions on leave-one-out predictive accuracy.

Assumptions
-----------

#. Single treated unit, balanced panel. One unit is treated after
   :math:`T_0`; every unit is observed at every period.

   Remark. The setup boundary (:mod:`mlsynth.utils.bscm_helpers.setup`) enforces
   both, translating a violation to :class:`~mlsynth.exceptions.MlsynthDataError`.

#. Enough pre-treatment periods to identify the weights. Formally BSCM needs at
   least two pre-periods, but the practical requirement is stronger: the
   shrinkage prior can only pin down a stable sparse fit if :math:`T_0` is large
   enough. What matters is the absolute number of pre-periods, not the
   donor-to-period ratio.

   Remark. When :math:`T_0` is small relative to :math:`J`, the model can
   interpolate the pre-period -- the in-sample fit looks near-perfect but does
   not generalise. Judge the fit by held-out or leave-one-out prediction and by
   the width of the credible band, never by the in-sample pre-RMSE. See
   "Inference and diagnostics" below.

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

Overfitting is real, and the in-sample pre-fit does not diagnose it. Relaxing
the simplex removes a strong regularizer, so with a large donor pool and few
pre-periods the unconstrained regression can drive the in-sample pre-RMSE to
almost zero while generalising poorly -- a beautiful pre-fit that means nothing.
The honest diagnostics are the ones the paper uses: leave-one-out
cross-validation, or a held-out pre-period test (hold out the last few
pre-treatment periods, where the true effect is zero, and check the prediction
there). The credible band is a second, automatic guard -- when many weightings
fit the pre-period equally well the posterior widens, so a wide band on the ATT
is the model telling you the pre-fit is less informative than it looks. Two
worked cases make the point. On the China anti-corruption watch panel (87
donors, 35 pre-periods, genuinely :math:`p > n`) the horseshoe pre-RMSE equals
the regularised simplex fit and the held-out error is close to the in-sample
error: the shrinkage regularises rather than interpolates. On the Basque panel
(16 donors, 15 pre-periods, :math:`p \approx n`) the same estimator interpolates
to a near-zero pre-RMSE whose held-out error is dozens of times larger: a
cautionary case, not a validation. The difference is the number of pre-periods,
not the donor-to-period ratio.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import BSCM

   # China anti-corruption watch-demand panel: 1 treated series ("watches"),
   # 87 donor series, treatment at month index 35 (2013-01).
   df = pd.read_csv("basedata/china_watches_long.csv")

   res = BSCM({
       "df": df, "outcome": "y", "treat": "treat",
       "unitid": "unit", "time": "time",
       "prior": "horseshoe", "seed": 2019,
       "display_graphs": False,
   }).fit()

   lo, hi = res.att_ci
   print(f"pre-period RMSE : {res.pre_rmse:.3f}")
   print(f"ATT            : {res.att:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
   top = max(res.donor_weights, key=lambda k: abs(res.donor_weights[k]))
   print(f"dominant donor : {top} = {res.donor_weights[top]:+.3f}")

With 87 donors and 35 pre-periods this is the "large p, small n" regime BSCM is
built for. The horseshoe shrinks all but a handful of donors to near-zero,
concentrating on a dominant donor (``C60``), fits the pre-period to about the
same RMSE as a simplex SCM (no interpolation), and returns an ATT whose 95%
credible interval crosses zero -- no statistically credible effect of the 2013
anti-corruption campaign on this residualised watch-demand series. Switch
``"prior": "spike_slab"`` to additionally read off ``res.inclusion_probs``.

Verification
------------

BSCM's horseshoe is cross-validated three ways on the China anti-corruption
watch panel (``china_watches_long.csv``), the "large p, small n" benchmark for
the forward-selection panel-data approach. Against the authors' reference Stan
horseshoe (``clarencejlee/bscm``, sampled with rstan) the pure-numpy Gibbs
matches essentially exactly -- the ATT agrees to four decimals
(:math:`-0.0221`), the pre-RMSE and dominant donor ``C60`` agree, and the
credible bands coincide. Against the forward-selected panel-data approach
(:class:`mlsynth.PDA` with ``method="fs"``, the dataset's original benchmark) it
selects the same dominant donor and returns the same near-null ATT. Against a
simplex SCM the horseshoe pre-fit equals the regularised simplex fit, confirming
that the shrinkage regularises rather than interpolates in this regime. See the
replication page :doc:`replications/bscm` and the durable case
``benchmarks/cases/bscm_china_watches.py``. The sampler, setup, inference,
plotter and result contract are unit-tested (``mlsynth/tests/test_bscm.py``,
full coverage).

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

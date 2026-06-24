CMBSTS — Causal Multivariate Bayesian Structural Time Series
============================================================

:class:`mlsynth.CMBSTS` fits the multivariate Bayesian structural time series
causal model of Menchetti and Bojinov (2022) [MenchettiBojinov2022]_, the
multivariate extension of the Bayesian structural time series counterfactual of
Brodersen et al. (2015) [Brodersen2015]_ (the ``CausalImpact`` model). It models
a small group of outcome series jointly, forecasts the counterfactual from the
posterior predictive distribution, and reports the effect as observed minus
predicted, per series, with credible bands.

When to use CMBSTS instead of something else
--------------------------------------------

Most synthetic-control estimators assume that treating one unit leaves every
other unit's outcome untouched. That assumption fails whenever the treated unit
and some of its comparisons are economic substitutes. Cut the price of a store
brand and its direct competitor's sales move too; take a few markets dark and
neighbouring markets pick up some of the lost demand. The competitor is not a
clean control, and pretending it is biases the estimate.

CMBSTS is the tool for that regime. It treats a predefined group — the treated
unit together with the handful of units it can plausibly interfere with — as a
multivariate outcome vector and models them jointly, so the effect on the
treated unit and the spillover onto its group-mates are estimated together. Two
features make it a natural fit for applied measurement:

- It is a forecasting model, not a donor-weighting one. The counterfactual is a
  structural time series — trend, weekly seasonality, an optional cycle, and a
  spike-and-slab regression on control series and covariates — so it handles the
  trends and calendar effects that dominate retail and marketing data directly,
  rather than hoping a convex combination of donors reproduces them.
- It is Bayesian. Every quantity comes with a full posterior, so the effect on
  each series arrives with a credible interval and the regressor inclusion
  probabilities tell you which controls the model actually used.

Reach for it when a price change, promotion, or go-dark test plausibly
cannibalises or boosts a known set of rival series, and you want the effect on
the treated unit and the spillover in one coherent model. Reach for a
single-series tool (:doc:`vanillasc`, :doc:`tasc`) when no-interference is
credible, and for the spillover-specific :doc:`spillsynth` when you want a
weighting estimator with an explicit spillover coefficient rather than a
Bayesian state-space forecast.

Notation
--------

Let :math:`j = 1` denote the treated unit. CMBSTS models a group of :math:`d`
series, :math:`\mathcal{G} \coloneqq \{1, \dots, d\}`, with the treated unit
first; the remaining group members are the units it may interfere with. Time is
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, 1-indexed, with the
intervention taking effect after period :math:`T_0`; the pre-period is
:math:`\mathcal{T}_1 \coloneqq \{t : t \le T_0\}` and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t : t > T_0\}`, of length
:math:`T_2 \coloneqq |\mathcal{T}_2|`.

Stack the group outcomes at time :math:`t` into the vector
:math:`\mathbf{y}_t \coloneqq (y_{1t}, \dots, y_{dt})^\top \in \mathbb{R}^d`. A
matrix :math:`\mathbf{X}_t \in \mathbb{R}^{1 \times p}` collects the
:math:`p` regressors — control-unit outcome paths and exogenous covariates. The
latent state at time :math:`t` is :math:`\boldsymbol{\alpha}_t`. As elsewhere in
the library, the per-period effect on series :math:`k` is the scalar
:math:`\tau_{kt} \coloneqq y_{kt} - \widehat{y}_{kt}`, the effect path is the
vector :math:`\boldsymbol{\tau}_k \coloneqq (\tau_{kt})_{t \in \mathcal{T}_2}`,
and the per-series ATT is
:math:`\widehat{\tau}_k \coloneqq T_2^{-1} \sum_{t \in \mathcal{T}_2} \tau_{kt}`.

The model
---------

In the no-intervention regime the group outcome follows a multivariate
structural time series. An observation equation links the data to the state and
the regressors,

.. math::

   \mathbf{y}_t = \mathbf{Z}\,\boldsymbol{\alpha}_t
   + \mathbf{X}_t\,\boldsymbol{\beta} + \boldsymbol{\varepsilon}_t,
   \qquad \boldsymbol{\varepsilon}_t \sim \mathcal{N}_d(\mathbf{0},
   \boldsymbol{\Sigma}_{\varepsilon}),

and a state equation governs the latent components,

.. math::

   \boldsymbol{\alpha}_{t+1} = \mathbf{T}\,\boldsymbol{\alpha}_t
   + \mathbf{R}\,\boldsymbol{\eta}_t,
   \qquad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}).

The state assembles the requested components — a local level (``"trend"``),
optionally a slope (``"slope"``), a dummy seasonal of period :math:`S`
(``"seasonal"``), and a stochastic cycle (``"cycle"``) — with the :math:`d`
series sharing each component's :math:`d \times d` disturbance covariance, so
cross-series co-movement is modelled explicitly rather than assumed away. The
regression coefficient matrix :math:`\boldsymbol{\beta}` carries a spike-and-slab
prior, so the model selects which control paths and covariates to keep. The
disturbance covariances take conjugate Inverse-Wishart priors with scale
:math:`s\,\widehat{\boldsymbol{\Sigma}}_{\text{pre}}` (the pre-period outcome
covariance times a small factor :math:`s`, ``prior_scale``) and an optional
cross-series correlation :math:`\rho` (``prior_rho``) that encodes a prior on
substitution.

Assumptions (and how to spot violations)
----------------------------------------

Assumption 1 (single sustained intervention).
   The treated unit is untreated through :math:`T_0` and treated at every period
   after, and the intervention is a single persistent change.

   Remark. CMBSTS estimates one counterfactual per series from the
   pre-intervention fit; an intervention that switches on and off, or a second
   distinct intervention, breaks the single-counterfactual logic. Use a
   staggered-adoption tool (:doc:`sdid`, :doc:`seq_sdid`) for on/off designs.

Assumption 2 (partial interference within the group).
   The intervention on the treated unit may change the outcomes of units inside
   the group :math:`\mathcal{G}`, but leaves units outside it unaffected.

   Remark. This is the assumption that earns its keep. It is what lets the
   control series and covariates serve as an untreated basis for the
   counterfactual while still permitting the competitor's sales to respond. Put
   a unit in ``group_units`` exactly when its outcome can react to the treatment;
   put it in ``control_units`` only when it cannot. A control that is secretly
   affected (a same-category product that also gets discounted) violates this and
   contaminates the counterfactual.

Assumption 3 (covariate–treatment independence).
   Every regressor in :math:`\mathbf{X}_t` — control paths and covariates — is
   unaffected by the intervention.

   Remark. A covariate touched by the treatment is a second outcome, not a
   predictor, and including it absorbs the effect you are trying to measure. The
   supermarket study handles the post-intervention store price (which the
   treatment moves) by freezing it at its last pre-period value, so the column it
   feeds the model is genuinely exogenous. Build such columns the same way.

Assumption 4 (nonanticipating, individualistic assignment).
   Assignment depends only on past outcomes and covariates, not on future ones,
   and the treated unit's assignment does not depend on other units.

   Remark. This is the panel analogue of unconfoundedness. It fails if treatment
   is timed in anticipation of a shock the controls do not see; a posterior
   predictive check (does the fitted model reproduce the pre-period data?) is the
   first diagnostic.

Inference and diagnostics
-------------------------

A Gibbs sampler draws the latent states (by a Durbin–Koopman simulation
smoother), the disturbance covariances, and the spike-and-slab regression at each
iteration. The counterfactual is the posterior predictive forecast of the
no-intervention outcome over :math:`\mathcal{T}_2`; the per-period effect
:math:`\tau_{kt}` and the per-series ATT :math:`\widehat{\tau}_k` are summarised
by their posterior mean and an equal-tailed credible interval at level
:math:`1 - \alpha` (``ci_alpha``). The intervals are Bayesian, and their width
reflects posterior predictive uncertainty that grows with the forecast horizon —
so a long post-window yields wide bands by construction. ``inclusion_probs``
reports, for each regressor, the posterior probability that the model included
it.

Return object
-------------

``CMBSTS.fit()`` returns a :class:`~mlsynth.utils.cmbsts_helpers.structures.CMBSTSResults`,
an :class:`~mlsynth.config_models.EffectResult`. The flat accessors resolve over
the treated series (series ``0``):

- ``res.att`` — the treated unit's posterior-mean ATT
  :math:`\widehat{\tau}_1`; ``res.att_ci`` its credible interval.
- ``res.counterfactual`` / ``res.gap`` — the treated unit's counterfactual path
  (pre-period fit then forecast) and effect path :math:`\tau_{1t}`.
- ``res.inference.method`` is ``"bayesian_posterior"``; ``res.weights`` is empty
  with a method note (CMBSTS is a state-space estimator, with no donor weights).

The full multivariate detail is typed on the result:

- ``res.inference_detail`` — per-series ``att_mean`` / ``att_lower`` /
  ``att_upper``, cumulative effects, the per-period effect paths and pointwise
  bands, and the counterfactual over the whole horizon.
- ``res.posterior`` — the run summary and ``inclusion_probs``.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import CMBSTS

   # A small retail panel: a store brand (treated), its competitor (group), and
   # two control products that do not react to the discount. A shared weekly
   # level drives all of them; a price cut lifts the store and dents the rival.
   rng = np.random.default_rng(0)
   T, T0 = 120, 90
   level = 50 + np.cumsum(rng.normal(0, 0.3, T))
   rows = []
   for unit, off in [("store", 0.0), ("rival", -8.0), ("ctrlA", 5.0), ("ctrlB", -3.0)]:
       series = level + off + rng.normal(0, 1.0, T)
       for t in range(T):
           bump = 6.0 if (unit == "store" and t >= T0) else (-2.0 if (unit == "rival" and t >= T0) else 0.0)
           rows.append({"item": unit, "week": t, "sales": series[t] + bump,
                        "treated": int(unit == "store" and t >= T0),
                        "weekend": float(t % 7 in (5, 6))})
   panel = pd.DataFrame(rows)

   res = CMBSTS({
       "df": panel, "outcome": "sales", "unitid": "item", "time": "week",
       "treat": "treated",
       "group_units": ["rival"],            # jointly modelled with the store
       "control_units": ["ctrlA", "ctrlB"], # regressor series (unaffected)
       "covariates": ["weekend"],           # an exogenous calendar dummy
       "components": ["trend"], "niter": 1000, "burn": 200, "seed": 0,
       "display_graphs": False,
   }).fit()

   print("store ATT:", round(res.att, 2), "credible interval:",
         tuple(round(x, 2) for x in res.att_ci))
   det = res.inference_detail
   for name, a in zip(det.series_names, det.att_mean):
       print(f"  {name}: {a:.2f}")
   print("regressors used:", {k: round(v, 2) for k, v in res.posterior.inclusion_probs.items()})

The treated-store ATT recovers the injected lift, the rival series carries a
negative spillover, and the spike-and-slab keeps the two genuine control
products.

Verification
------------

CMBSTS is cross-validated cell-by-cell against the authors' ``CausalMBSTS`` R
package. The durable benchmark
`benchmarks/cases/cmbsts_vignette.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cmbsts_vignette.py>`_
runs the package's own vignette example (a bivariate weekly series with a
trend-plus-cycle model and a ``+2`` intervention) and confirms that the mlsynth
port reproduces the R posterior-mean effects to about ``0.01`` and the credible
bounds to a few hundredths — within Monte-Carlo error of two independent Gibbs
samplers. A second case,
`benchmarks/cases/cmbsts_supermarket.py <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cmbsts_supermarket.py>`_,
reproduces the paper's empirical Table 3 on the Florence supermarket study at the
one-month horizon — the store-brand effects cross-validate against the R package
and the substantive finding (large positive store effects, no significant
competitor effect) holds. See :doc:`replications/cmbsts` for the full validation.

Core API
--------

.. autoclass:: mlsynth.config_models.CMBSTSConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.CMBSTS
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.cmbsts_helpers.structures.CMBSTSResults
   :members:
   :undoc-members:
   :show-inheritance:

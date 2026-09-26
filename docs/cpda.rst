Covariate-Adjusted Panel Data Approach (CPDA)
=============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

CPDA is the semiparametric middle of the three constructions in Hsiao and Zhou
[HZ2019]_. It answers the same question as every estimator in this library:
what would have happened to one treated unit if the treatment had not arrived.
It differs in how much of the answer it asks the control units to carry.

Suppose the outcome responds to some variables you observe, and also to shocks
you do not. A state's cigarette consumption moves with income and with a
national shift in attitudes that nobody measured. Two routes are already
available. The parametric route, :class:`GSYNTH`, writes the unobserved part as
a small number of common factors, estimates them, and needs a long panel and
many units to do so. The panel data approach, :class:`PDA`, models nothing:
it regresses the treated unit on the control units and lets them stand in for
everything at once.

CPDA sits between the two. It removes the part of the outcome the observed
variables explain, and only then runs the control regression on what is left.
The controls carry the unobserved part alone, which is a smaller job, and the
number of unobserved factors never has to be chosen. Hsiao and Zhou's Remark 2
gives that as the method's reason for existing, and their simulations bear it
out: across the 28 cells of their Tables 1 to 8, CPDA has the lowest mean
absolute bias in 13, more than any other construction they compare.

Reach for it when

- one unit is treated and the rest are not;
- you observe covariates that plausibly move the outcome, and they vary over
  time;
- the treated unit's covariates move in ways the control units' do not, which
  is where the adjustment buys the most;
- the pre-period is too short to estimate a factor structure with confidence.

Reach for :class:`PDA` instead when there are no covariates to adjust for.
CPDA with an empty covariate list is PDA with extra steps, so the configuration
refuses that case instead of silently becoming another estimator.

Notation
--------

Let :math:`i = 1, \dots, N` index units and :math:`t = 1, \dots, T` index
periods. Unit 1 is treated from :math:`T_0 + 1` onward and units
:math:`2, \dots, N` never are. Write :math:`y_{it}` for the outcome,
:math:`\mathbf{x}_{it}` for a :math:`k`-vector of observed covariates, and
:math:`y^0_{1t}` for the treated unit's untreated outcome, which is observed
before :math:`T_0` and is the object to be predicted after it.

The untreated outcome is

.. math::

   y^0_{it} = \mathbf{x}_{it}' \boldsymbol{\beta} + v_{it},
   \qquad v_{it} = \boldsymbol{\gamma}_i' \mathbf{f}_t + u_{it},

so :math:`v_{it}` splits into a part driven by :math:`r` unobserved common
factors :math:`\mathbf{f}_t`, whose effect :math:`\boldsymbol{\gamma}_i`
differs by unit, and an idiosyncratic part :math:`u_{it}`. Putting the unit and
time effects in product form nests the familiar additive two-way model, and
lets a shock at time :math:`t` land differently on different units.

Collect the control units' residuals as
:math:`\tilde{\mathbf{v}}_t = (v_{2t}, \dots, v_{Nt})'`. The treatment effect at
:math:`t > T_0` is :math:`\Delta_{1t} = y_{1t} - y^0_{1t}`.

How It Works
------------

The construction is Hsiao and Zhou's Equations 11 to 15, in four steps.

Step 1, the slope. Estimate :math:`\boldsymbol{\beta}` on the pre-period from
the control units. Two estimators are offered, and the paper names both without
saying which produced which of its published columns.

Pesaran's common correlated effects estimator [Pesaran2006]_ is the default. It
uses the cross-sectional averages of :math:`(y_{it}, \mathbf{x}_{it})` as
stand-ins for the unobserved factors and projects them out:

.. math::

   \hat{\boldsymbol{\beta}}_{\mathrm{CCE}} =
   \Big( \sum_{i} \mathbf{X}_i' M_{\bar{Z}} \mathbf{X}_i \Big)^{-1}
   \sum_{i} \mathbf{X}_i' M_{\bar{Z}} \mathbf{y}_i,
   \qquad
   M_{\bar{Z}} = I_{T_0} - \bar{Z}(\bar{Z}'\bar{Z})^{-1}\bar{Z}'.

It is consistent as :math:`N` grows with :math:`T` fixed, which is the regime
most applied panels are in. Bai's interactive fixed effects estimator
[Bai2009]_ is the alternative, available as ``beta_method="bai"``; it needs both
:math:`N` and :math:`T` large, which Remark 1 calls a luxury.

Step 2, residualise. Subtract the covariate part from the treated unit and from
every control:

.. math::

   \tilde{v}_{it} = y_{it} - \mathbf{x}_{it}' \hat{\boldsymbol{\beta}}.

Step 3, fit the controls to what is left. Choose an intercept :math:`\mu` and
weights :math:`\mathbf{w}` to minimise the squared pre-period error,

.. math::

   \min_{\mu, \mathbf{w}} \sum_{t=1}^{T_0}
   \big( \tilde{v}_{1t} - \mu - \mathbf{w}' \tilde{\mathbf{v}}^*_t \big)^2,

where :math:`\tilde{\mathbf{v}}^*_t` is a subset of the control residuals.
The weights are ordinary regression coefficients: CPDA places no sign
restriction on them and does not require them to sum to one, which is what
separates it from the synthetic control family.

Step 4, predict and difference. For :math:`t > T_0`,

.. math::

   \hat{y}^0_{1t} = \mathbf{x}_{1t}' \hat{\boldsymbol{\beta}}
   + \hat{\mathbf{w}}' \tilde{\mathbf{v}}^*_t + \hat{\mu},
   \qquad
   \hat{\Delta}_{1t} = y_{1t} - \hat{y}^0_{1t}.

The reported ATT is the mean of :math:`\hat{\Delta}_{1t}` over the post window.

Which Controls Enter
--------------------

Step 3 leaves open which controls make up :math:`\tilde{\mathbf{v}}^*_t`.
Hsiao and Zhou say the subset "can be chosen using a model selection criterion
as in Hsiao, Ching, and Wan (2012), or the least absolute shrinkage and
selection operator (LASSO) method (Tibshirani, 1996), as suggested by Li and
Bell (2017)", and stop there.

That sentence does not pin the answer, and the estimate moves with it.
Measured on the paper's own Table 9 panel, 19 pre-treatment years against 38
control states, the four selectors this estimator offers span a mean absolute
effect of 3.79 to 14.04, around a published 9.56.

.. list-table::
   :header-rows: 1
   :widths: 22 12 18 14

   * - ``selector``
     - controls kept
     - nested LOO error
     - mean absolute effect
   * - ``lasso_cv`` (default)
     - 9
     - 2.167
     - 4.93
   * - ``aicc``
     - 5
     - 2.559
     - 5.04
   * - ``lasso_bic``
     - 15
     - 3.849
     - 14.04
   * - ``all``
     - 38
     - --
     - 8.62

The default is ``lasso_cv`` because it has the lowest leave-one-pre-period-out
error when the selection is repeated inside every fold, and not because it
agrees with any published figure. The two selectors landing nearest the
published 9.56 score worse on that criterion, so nothing measurable from the
pre-period picks the published number out; choosing by closeness to it would be
fitting to the post-period, which is the one thing a counterfactual may not do.

Set ``sensitivity=True`` to fit under every selector and read the spread off
``results.fit.sensitivity``. On a short pre-period with many controls, reporting
a point estimate alone claims an identification the method does not have.

Assumptions
-----------

1. The untreated outcome is linear in the observed covariates with a slope
   common across units, and the unobserved part is a factor structure.

   Remark. The factor form nests the additive two-way model, so assuming it
   costs less than it appears to. What it does assume is that the same
   :math:`\boldsymbol{\beta}` applies to the treated unit as to the controls,
   since :math:`\boldsymbol{\beta}` is estimated from the controls and then
   applied to the treated unit's covariates in Step 4.

2. The covariates and the unobserved terms are strictly exogenous with respect
   to the idiosyncratic error: :math:`E(u_{it} \mid \mathbf{x}_{is},
   \mathbf{f}_s, \boldsymbol{\gamma}_i) = 0`.

   Remark. This rules out feedback from past shocks into later covariates. A
   covariate that responds to the outcome breaks it, which is why Hsiao and
   Zhou replace Abadie's price, beer and per-capita GDP predictors in their
   smoking application: those are themselves treated.

3. The control units are unaffected by the treatment, in outcome and in
   covariates.

   Remark. The standard no-interference condition. A control that shares a
   border with the treated unit, or a market with it, is a candidate for
   violating it; :class:`SPILLSYNTH` is the estimator for that case.

4. The control residuals span the treated unit's residual well enough for the
   pre-period fit to extrapolate, which requires the control loadings to have
   full rank :math:`r < N - 1`.

   Remark. This is Assumption 4 of the paper, and it is what replaces knowing
   :math:`r`. The method never estimates the number of factors; it requires
   only that the controls carry them.

5. The pre-period is long enough to estimate :math:`\mu` and
   :math:`\mathbf{w}` on the kept controls.

   Remark. This one is a matter of degree and it is where the selector bites.
   With 19 pre-periods and 38 controls, a selector that keeps 15 is fitting 16
   parameters on 19 observations, and the table above shows what that does to
   the answer.

Inference and Diagnostics
-------------------------

The ATT's standard error is the heteroskedasticity- and autocorrelation-
consistent long-run variance of the post-period gap, divided by the number of
post-periods, with a Bartlett kernel and the usual
:math:`\lfloor 4 (T_2/100)^{2/9} \rfloor` truncation lag. Set ``lrvar_lag`` to
fix the lag. The interval and p-value are two-sided normal at ``alpha``.

``results.fit_diagnostics.rmse_pre`` is the pre-period root mean squared error.
Read it alongside the number of kept controls in
``results.weights.summary_stats``: a pre-period error near zero with a subset
almost as wide as the pre-period is interpolation, not fit.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import CPDA

   rng = np.random.default_rng(0)
   T, T0, N = 40, 30, 13
   f = rng.standard_normal((T, 2))
   gam = rng.standard_normal((N, 2))
   x0 = 10.0 + rng.standard_normal((T, N))
   x0[:, 0] = 10.0 + 8.0 * rng.standard_normal(T)   # treated moves on its own
   x1 = rng.standard_normal((T, N))
   Y = 2.0 * x0 - 1.0 * x1 + f @ gam.T + 0.5 * rng.standard_normal((T, N))
   Y[T0:, 0] -= 5.0                                  # the planted effect

   units = np.repeat(np.arange(N), T)
   times = np.tile(np.arange(T), N)
   panel = pd.DataFrame({
       "unit": units, "time": times, "y": Y.T.ravel(),
       "x0": x0.T.ravel(), "x1": x1.T.ravel(),
       "D": ((units == 0) & (times >= T0)).astype(int),
   })

   res = CPDA({
       "df": panel, "outcome": "y", "treat": "D",
       "unitid": "unit", "time": "time",
       "covariates": ["x0", "x1"],
       "sensitivity": True,
       "display_graphs": False,
   }).fit()

   print(res.effects.att)                 # near -5
   print(res.fit.selected_donors)         # which controls were kept
   print(res.fit.sensitivity)             # the ATT under every selector

Verification
------------

The slope step is cross-validated against an independent implementation and the
construction is measured against the paper's own empirical tables. The
:doc:`replications/cpda` page has the comparison, and the durable case is
`benchmarks/cases/cpda.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/cpda.py>`_.
The study behind those numbers, including the selector measurements quoted
above, is `benchmarks/studies/hsiao_zhou_counterfactuals
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/hsiao_zhou_counterfactuals>`_.

Core API
--------

.. autoclass:: CPDA
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.CPDAConfig
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.cpda_helpers.structures.CPDAResults
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.cpda_helpers.structures.CPDAFit
   :members:
   :undoc-members:

References
----------

.. [HZ2019] Hsiao, C., & Zhou, Q. (2019). Panel parametric, semiparametric, and
   nonparametric construction of counterfactuals. *Journal of Applied
   Econometrics*, 34(4), 463-481.

.. [Pesaran2006] Pesaran, M. H. (2006). Estimation and inference in large
   heterogeneous panels with a multifactor error structure. *Econometrica*,
   74(4), 967-1012.

.. [Bai2009] Bai, J. (2009). Panel data models with interactive fixed effects.
   *Econometrica*, 77(4), 1229-1279.

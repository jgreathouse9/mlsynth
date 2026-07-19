Counterfactual Synthetic Control with Instrumented PCA (CSC-IPCA)
=================================================================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

You have a single treated unit and a pool of untreated controls observed over
many periods, an intervention switches on at a known date, and — this is the
distinguishing feature — you also observe a rich set of time-varying covariates
for every unit. The units move together through shared, unobserved forces but
not in parallel, and the treated unit may sit outside the range spanned by the
controls, so a convex donor-weight fit would have to extrapolate. This is the
regime where difference-in-differences (which asserts parallel trends) and
plain synthetic control (which pins the counterfactual to a convex combination
of controls, and so refuses to extrapolate) are hard to justify, and where you
would like the many covariates to do real work rather than being collapsed to a
single pre-period average.

The CSC-IPCA estimator of Wang ([Wang2024]_) targets that regime by writing
each unit's untreated outcome as a small number of latent common factors
:math:`\mathbf{f}_t` weighted by factor loadings — but instead of free
unit-specific loadings :math:`\boldsymbol{\lambda}_i`, it makes the loadings a
linear projection of the observed covariates,

.. math::

   \boldsymbol{\lambda}_{it} = \mathbf{x}_{it}^\top \boldsymbol{\Gamma},

where :math:`\mathbf{x}_{it}` is the unit's covariate vector at time :math:`t`
and :math:`\boldsymbol{\Gamma}` is an :math:`L \times K` mapping matrix shared
across units. This is instrumented principal component analysis (Kelly, Pruitt
and Su [KPS2019]_), imported from asset pricing. The loadings inherit the time
variation of the covariates, and the covariates — not a convex hull condition
— carry the information that pins the counterfactual.

Why instrument the loadings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A plain interactive-fixed-effects estimator (:doc:`cfm`, :doc:`fma`) learns a
free loading for each unit from its outcome history alone. CSC-IPCA instead
projects the loadings onto the covariates, which buys three things the paper
emphasises. It needs no convex-hull or common-support condition, so the treated
unit can lie outside the donor range. It extracts signal from many covariates
at once through the dimension-reducing map :math:`\boldsymbol{\Gamma}`, rather
than from the outcome path alone. And it does not require the interactive-fixed
-effects model to be correctly specified in the covariates, because the loadings
absorb their predictive content. The cost is that the covariates must genuinely
carry the loading information: the method's edge over a plain factor model
appears precisely when the covariates are informative and only partially
observed, and shrinks to nothing when every relevant covariate is already in
hand (then a correctly specified factor model does as well).

Reach for CSC-IPCA when
^^^^^^^^^^^^^^^^^^^^^^^^^

* you observe many time-varying covariates that plausibly drive the outcome,
  and want them to identify the counterfactual;
* the treated unit sits outside the donor convex hull, so a simplex synthetic
  control must extrapolate;
* you distrust a hand-specified interactive-fixed-effects model and would
  rather let covariates instrument the loadings;
* the pre-period is long relative to the number of covariates times factors.

Do not use CSC-IPCA when
^^^^^^^^^^^^^^^^^^^^^^^^^

* you have no covariates, or only a handful of weakly predictive ones — the
  loadings are then unidentified or uninformative, and an outcome-only factor
  estimator (:doc:`cfm`, :doc:`fma`) is the right tool;
* the pre-period is shorter than :math:`L \times K`, so the treated mapping is
  not identified (see Assumption 4);
* you have fewer covariates than factors, :math:`L < K`, so the projected
  loadings cannot span the factor space.

Notation
--------

Let :math:`i = 1` denote the treated unit and :math:`i = 2, \dots, n` the
controls, observed for :math:`t = 1, \dots, T`. The intervention occurs at
:math:`T_0 + 1`, so periods :math:`t \le T_0` are pre-treatment and
:math:`t > T_0` post-treatment. Each unit carries an :math:`L`-vector of
observed covariates :math:`\mathbf{x}_{it}`. The untreated potential outcome
follows the instrumented factor model

.. math::

   Y_{it}(0) = (\mathbf{x}_{it}^\top \boldsymbol{\Gamma})\, \mathbf{f}_t
               + \epsilon_{it},

with :math:`\mathbf{f}_t` a :math:`K`-vector of latent common factors,
:math:`\boldsymbol{\Gamma}` the :math:`L \times K` mapping matrix, and
:math:`\epsilon_{it}` idiosyncratic noise. The treated unit's observed outcome
is :math:`Y_{it} = Y_{it}(0) + \delta_{it}\,D_{it}`, and the estimand is the
per-period effect :math:`\delta_{1t}` for :math:`t > T_0` together with its
post-period average, the ATT
:math:`\overline{\delta} = (T - T_0)^{-1} \sum_{t > T_0} \delta_{1t}`. The
imputed counterfactual is
:math:`\widehat Y_{1t}(0) = (\mathbf{x}_{1t}^\top \widehat{\boldsymbol{\Gamma}})
\widehat{\mathbf{f}}_t` and the reported effect is
:math:`Y_{1t} - \widehat Y_{1t}(0)`.

Assumptions
-----------

1. Instrumented factor structure. The untreated outcome admits the factor model
   above with loadings that are a linear projection of the covariates,
   :math:`\boldsymbol{\lambda}_{it} = \mathbf{x}_{it}^\top \boldsymbol{\Gamma}`,
   for a fixed number of factors :math:`K` common across treated and controls.

   Remark. The controls identify :math:`\boldsymbol{\Gamma}` and
   :math:`\mathbf{f}_t`; the treated unit's covariates then generate its
   loadings without ever fitting a free treated loading. This is what lets the
   treated unit lie outside the donor hull — its counterfactual is built from
   its own covariates through the shared map, not from a donor average.

2. Covariate orthogonality (exclusion). The covariates are orthogonal to the
   idiosyncratic errors, :math:`\mathbb{E}[\mathbf{x}_{it}\,\epsilon_{it}] = 0`.

   Remark. This is the exclusion restriction of the instrumenting step, exactly
   as in two-stage regression: the covariates may drive the loadings but must
   not correlate with the leftover noise, or the mapping is contaminated.

3. Enough covariates. There are at least as many covariates as factors,
   :math:`L \ge K`.

   Remark. The projected loadings live in an :math:`L`-dimensional space mapped
   to :math:`K` factors; if :math:`L < K` the map cannot span the factor space
   and the factor solve is singular. The config rejects :math:`L < K` early.

4. Long enough pre-period. The treated mapping is solved from the treated
   unit's :math:`T_0` pre-treatment periods, which identify the :math:`LK`
   entries of :math:`\boldsymbol{\Gamma}` only if :math:`T_0 \ge L K`.

   Remark. With a single treated unit each pre-period contributes one
   observation to the :math:`LK` normal equations, so a short pre-period or
   too many covariate-factor products leaves the mapping unidentified. The
   setup raises a clear error when :math:`T_0 < LK`.

5. Weak dependence and stationarity. The covariates and errors are weakly
   dependent with finite second moments, and the mapping lives in a compact
   parameter space.

   Remark. These are the regularity conditions of Wang (2024, Assumption 2)
   and Kelly, Pruitt and Su (2019): they deliver consistency of the
   alternating-least-squares estimates and the central limit theorem behind the
   conformal calibration.

Estimation
----------

The mapping and factors are estimated by alternating least squares, because the
loadings are :math:`\mathbf{x}_{it}^\top \boldsymbol{\Gamma}` rather than a free
:math:`\boldsymbol{\lambda}_i`, so eigendecomposition does not apply. With the
factors held fixed the :math:`\boldsymbol{\Gamma}` step is a single linear solve
of the :math:`LK` normal equations; with :math:`\boldsymbol{\Gamma}` fixed each
:math:`\mathbf{f}_t` is a :math:`K`-vector least-squares solve. The four steps
are: estimate :math:`(\widehat{\boldsymbol{\Gamma}}_{\text{ctrl}},
\widehat{\mathbf{f}}_t)` on the control panel over the whole period;
re-estimate the treated mapping :math:`\widehat{\boldsymbol{\Gamma}}_{\text{tr}}`
on the treated pre-period holding :math:`\widehat{\mathbf{f}}_t` fixed;
normalise :math:`(\widehat{\boldsymbol{\Gamma}}_{\text{tr}},
\widehat{\mathbf{f}})` to the identifiable rotation
(:math:`\boldsymbol{\Gamma}^\top \boldsymbol{\Gamma} = \mathbf{I}_K`,
:math:`\mathbf{f}\mathbf{f}^\top / T` diagonal); and impute
:math:`\widehat Y_{1t}(0)` over every period. The normalisation fixes the
rotational indeterminacy of any factor model and leaves the counterfactual
unchanged.

Inference and diagnostics
-------------------------

Inference is the moving-block conformal procedure of Chernozhukov, Wuthrich and
Zhu ([CWZ2021]_), which needs no asymptotic variance formula and is robust to
misspecification. For a candidate effect :math:`\theta` the treated series is
adjusted under the sharp null :math:`Y_{1t} - \theta`, the model is
re-estimated, and the post-treatment residual is compared against the
pre-treatment residuals by permutation; the candidate is kept in the
:math:`(1 - \alpha)` band when its block-permutation p-value is at least
:math:`\alpha`. Because the factors are learned from the controls only, they do
not depend on :math:`\theta` and are reused across the null grid. The estimator
returns a per-period band — the primary effect-over-time output — from a
block of length one, and an ATT band from a moving block of length
:math:`T - T_0` under a constant-effect null.

The number of factors :math:`K` is set directly by ``n_factors`` (default 2,
the paper's choice for its empirical study); there is no data-driven selection,
so a one-versus-two-factor robustness pass is advisable.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import CSCIPCA

   # Wang (2024): Brexit -> UK foreign direct investment. The UK is treated
   # from 2017; nine macro covariates instrument the loadings.
   df = pd.read_csv("basedata/fdi_oecd_brexit.csv")
   covs = ["log_gdp", "log_gdp_percap", "import_to_gdp", "export_to_gdp",
           "inflation_gdp_deflator", "gross_capital_forma_gdp", "unemployment",
           "employment_15", "log_population"]

   res = CSCIPCA({
       "df": df, "outcome": "fdi", "treat": "treated",
       "unitid": "country", "time": "year",
       "covariates": covs,
       "n_factors": 2,
       "alpha": 0.10,                    # 90% conformal band
       "display_graphs": False,
   }).fit()

   res.att                               # mean post-2017 FDI effect
   res.design.tau                        # per-year effect path (2017: ~-7.8, ...)
   res.inference_detail.ci_lower_t       # per-year conformal band
   res.inference_detail.att_p_value      # block-permutation p-value of H0: ATT = 0

The reported ``gap`` is :math:`Y_{1t} - \widehat Y_{1t}(0)`; ``counterfactual``
is the imputed untreated path
:math:`(\mathbf{x}_{1t}^\top \widehat{\boldsymbol{\Gamma}})
\widehat{\mathbf{f}}_t`. Because CSC-IPCA is a factor-model counterfactual, it
carries no donor weights.

Verification
------------

CSC-IPCA reproduces Wang's Brexit / UK foreign-direct-investment application on
the author's data (``basedata/fdi_oecd_brexit.csv``): the reported per-year ATT
path — :math:`-7.8` (2017), :math:`-12.9` (2018), :math:`-18.3` (2019) — is
matched to the second decimal (durable case
``benchmarks/cases/cscipca_brexit.py``). It is further validated by a
cross-validation against the authors' reference implementation (counterfactual
and ATT matched to machine precision through ``dataprep``) and by the paper's
linear factor Monte Carlo, where the bias shrinks as the share of observed
covariates rises and the estimator beats an extrapolating simplex synthetic
control (``benchmarks/cases/cscipca_mc.py``). See :doc:`replications/cscipca`
for the full comparison.

Core API
--------

.. autoclass:: CSCIPCA
   :members: fit

.. autoclass:: mlsynth.config_models.CSCIPCAConfig
   :members:

References
----------

.. [Wang2024] Wang, C. (2024). "Counterfactual and Synthetic Control Method:
   Causal Inference with Instrumented Principal Component Analysis." Job Market
   Paper.

.. [KPS2019] Kelly, B. T., Pruitt, S., & Su, Y. (2019). "Characteristics Are
   Covariances: A Unified Model of Risk and Return." *Journal of Financial
   Economics* 134(3):501-524.

.. [CWZ2021] Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). "An Exact and
   Robust Conformal Inference Method for Counterfactual and Synthetic Control."
   *Journal of the American Statistical Association* 116(536):1849-1864.

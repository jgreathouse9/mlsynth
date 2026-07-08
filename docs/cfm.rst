Causal Factor Model (CFM)
=========================

.. currentmodule:: mlsynth

When to Use This Method
-----------------------

You have a single treated unit and a pool of untreated control units
observed over many periods, an intervention switches on at a known date,
and the units move together through shared, unobserved forces — an
aggregate business cycle, an industry-wide demand shift, a regional shock —
but not in parallel. Each unit responds to those common forces with its own
sensitivity. This is the regime where difference-in-differences (which
assumes the treated and control units share a common time effect, the
parallel-trends assumption) and plain synthetic control (which pins the
counterfactual to a convex combination of controls) are hard to justify.

The causal factor model of Bai and Wang ([BaiWang2026]_) targets that
regime with a different idea about what the intervention *does*. It writes
each unit's outcome as a small number of latent common factors
:math:`\mathbf{f}_t` (the shared forces) weighted by unit-specific factor
loadings :math:`\boldsymbol{\lambda}_i` (how strongly that unit feels
them), and it lets the intervention change the treated unit's loadings. The
treatment effect is then a structural break in how the treated unit responds
to the common shocks, and the target is the systematic causal effect

.. math::

   \tau^*_{it} = [\boldsymbol{\lambda}_i(1) - \boldsymbol{\lambda}_i(0)]^\top
                 \mathbf{f}_t + [a_i(1) - a_i(0)],

the difference between the treated unit's post- and pre-intervention
systematic components. Here :math:`\boldsymbol{\lambda}_i(0)` and
:math:`\boldsymbol{\lambda}_i(1)` are the loadings before and after the
intervention, and :math:`a_i(0), a_i(1)` are unit-specific intercepts.

Why the systematic target matters
----------------------------------

The methods you already know model only the untreated potential outcome
:math:`Y_{it}(0)`, form a fitted counterfactual :math:`\widehat Y_{it}(0)`,
and report the difference :math:`Y_{it}(1) - \widehat Y_{it}(0)`. Bai and
Wang show this carries a one-sided idiosyncratic error: the observed treated
outcome contains its own noise term :math:`\varepsilon_{it}(1)` that the
imputed counterfactual cannot cancel, and that term does not vanish for a
fixed unit and date. When the idiosyncratic variation is large relative to
the predictable part — common in panel applications with low
explanatory power — the naive gap is an unstable object for inference.

CFM instead models *both* potential outcomes inside the same factor
structure and re-estimates the treated unit's loading after the
intervention, so the reported effect is the change in the systematic
component directly. The individual effect
:math:`\tau_{it} = \tau^*_{it} + [\varepsilon_{it}(1) - \varepsilon_{it}(0)]`
is not point-identified for a fixed unit and date, because the idiosyncratic
difference is unobserved; the systematic part :math:`\tau^*_{it}` is. This is
the essential contrast with :doc:`fma`, which is a single-equation imputation
estimator on the same factor scaffolding. On data with no loading break the
two coincide; they diverge exactly when the intervention changes the treated
unit's exposure to the common shocks.

Reach for CFM when
^^^^^^^^^^^^^^^^^^^

* the treated and control units co-move through latent common factors but
  not in parallel (heterogeneous trends);
* the pre-period is long enough to estimate the treated unit's factor
  loading;
* you want a formal confidence band on the effect path without asserting
  parallel trends or a convex donor-weight fit;
* you specifically suspect the intervention changed the treated unit's
  *response* to shared shocks — a loading break — rather than adding a
  fixed increment.

Do not use CFM when
^^^^^^^^^^^^^^^^^^^

* you need the individual (not systematic) effect and the idiosyncratic
  variance is large — that object is not identified here;
* the pre-period is too short to estimate loadings on the chosen number of
  factors;
* the intervention plausibly changed the common factor process itself and
  the treated group is large — that is the potential-factors regime
  (Bai and Wang Section 4.2, Proposition 2), which this estimator does not
  yet cover.

Notation
--------

Let :math:`i = 1` denote the treated unit and :math:`i = 2, \dots, n` the
controls, observed for :math:`t = 1, \dots, T`. The intervention occurs at
:math:`T_0 + 1`, so periods :math:`t \le T_0` are pre-treatment and
:math:`t > T_0` are post-treatment. Each outcome follows a factor model

.. math::

   y_{it} = a_i + \boldsymbol{\lambda}_i^\top \mathbf{f}_t + \varepsilon_{it},

with :math:`\mathbf{f}_t` an :math:`r \times 1` vector of latent common
factors, :math:`\boldsymbol{\lambda}_i` the unit's loadings, :math:`a_i` a
unit intercept, and :math:`\varepsilon_{it}` idiosyncratic noise. For the
treated unit the loadings and intercept take the value
:math:`(a_1(0), \boldsymbol{\lambda}_1(0))` before the intervention and
:math:`(a_1(1), \boldsymbol{\lambda}_1(1))` after. The estimand is the
systematic causal effect :math:`\tau^*_{1t}` for :math:`t > T_0`, and its
post-period average, the systematic ATT
:math:`\bar\tau^* = (T - T_0)^{-1} \sum_{t > T_0} \tau^*_{1t}`. The
intercept shift :math:`\kappa = a_1(1) - a_1(0)` collects the constant part
of the break, including any constant shift in the factor process.

Assumptions
-----------

1. Factor structure. Both potential outcomes admit the same factor model
   with a fixed number of common factors :math:`r`; the factors are common
   across treated and control units.

   Remark. The controls identify the factors, so their outcomes must be
   driven by the same latent forces as the treated unit. This is what
   replaces parallel trends: the units need not move together, only respond
   to the same shocks.

2. The intervention does not change the factor process. The policy alters
   the treated unit's loadings (and possibly slope coefficients on
   covariates) but leaves the common factors :math:`\mathbf{f}_t`
   themselves unchanged.

   Remark. This is the Proposition 1 benchmark: appropriate for a targeted
   or partial-equilibrium intervention whose feedback to the aggregate
   trends is negligible. A constant shift in the factor process is
   permitted, because with unit intercepts it is absorbed into the
   post-treatment intercept (the constant-shift specification of Section
   4.2), and is what :math:`\kappa` measures.

3. Long panel on both sides. The number of controls, the pre-period length,
   and the post-period length all grow; the treated unit may be unique
   (:math:`n_0 = 1`).

   Remark. The factors are learned from the controls, so a wide control
   pool sharpens them; the treated loadings are identified from the
   time-series variation within each regime, so both the pre- and
   post-periods must be long enough to estimate an :math:`r`-vector.

4. Weak dependence of the idiosyncratic errors. The
   :math:`\varepsilon_{it}` are weakly cross-sectionally and serially
   dependent with finite variance, so cross-sectional and time averages of
   the noise vanish asymptotically.

   Remark. This is why the systematic effect is identified while the
   individual effect is not: averaging over units or time kills the
   idiosyncratic term, but for a fixed :math:`(i, t)` it survives.

Inference and diagnostics
-------------------------

The standard error of :math:`\widehat\tau^*_{1t}` has two
asymptotically-uncorrelated pieces (Bai and Wang appendix A.2), and the
total variance is their sum:

* a treated-regression component :math:`V^{\text{reg}}`, from estimating the
  pre- and post-treatment loadings and intercept. It is a block-additive
  heteroskedasticity-robust (HC1) sandwich: the pre- and post-block scores
  are asymptotically independent, so
  :math:`\mathrm{Var}(\widehat\tau^*_{1t}) = \mathbf{c}_t^\top
  \widehat{\mathrm{Var}}(\widehat{\boldsymbol\theta}_1)\, \mathbf{c}_t +
  \mathbf{c}_t^\top \widehat{\mathrm{Var}}(\widehat{\boldsymbol\theta}_0)\,
  \mathbf{c}_t` with :math:`\mathbf{c}_t = (1, \mathbf{f}_t^\top)^\top`;
* a factor-estimation component :math:`V^f`, from learning the common
  factors off the controls (appendix A.20). It enters through
  :math:`[\boldsymbol\lambda_1(1) - \boldsymbol\lambda_1(0)]^\top
  \mathrm{Var}(\widehat{\mathbf f}_t)
  [\boldsymbol\lambda_1(1) - \boldsymbol\lambda_1(0)]` and can be turned off
  with ``factor_variance=False`` to report the regression component alone.

The per-period ``(1 - alpha)`` interval is
:math:`\widehat\tau^*_{1t} \pm z_{\alpha/2}\,\widehat{\mathrm{SE}}_t`, and the
ATT interval aggregates the same components over the post-period.

Two diagnostics accompany the fit. The intercept-shift test reports
:math:`\widehat\kappa` and its t-statistic (a post-treatment level break, or
a constant shift in the factor process); a Chow F-statistic tests parameter
stability of the treated-unit factor regression at the intervention date. A
large Chow statistic supports allowing the treated loadings to break.

Number of factors
------------------

The factor count is chosen from the controls by the Ahn and Horenstein
(2013) eigenvalue-ratio (``factor_selection="er"``) or growth-ratio
(``"gr"``) criteria — maximisers of successive eigenvalue ratios of the
control covariance — or by the Bai and Ng (2002) information criterion
(``"bai_ng"``). Pass ``n_factors`` to fix it directly; a one-versus-two
factor robustness pass is the paper's habit.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import CFM

   # California Prop 99: cigarette sales, treated from 1989.
   df = pd.read_csv("basedata/smoking_data.csv")
   df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)

   res = CFM({
       "df": df, "outcome": "cigsale", "treat": "treat",
       "unitid": "state", "time": "year",
       "factor_selection": "er",       # ER and GR both pick 1 factor here
       "display_graphs": False,
   }).fit()

   res.att                              # mean systematic reduction in pack sales
   res.att_ci                           # asymptotic 95% CI for the ATT
   res.design.tau                       # per-period systematic effect path
   res.inference_detail.ci_lower_t      # per-period CI band
   res.metadata["kappa_t"]              # intercept-shift test t-statistic
   res.metadata["chow_fstat"]           # structural-break diagnostic

The reported ``gap`` is the systematic causal effect
:math:`\tau^*` — not ``observed - counterfactual`` — because that is the
estimand; ``counterfactual`` is the systematic untreated path
:math:`a_1(0) + \boldsymbol\lambda_1(0)^\top \mathbf{f}_t`.

Verification
------------

CFM reproduces Bai and Wang's two empirical applications — California Prop 99
and German reunification — on the authors' data. See
:doc:`replications/cfm` for the cell-by-cell comparison (both applications
select a single factor by ER and GR; the Chow break statistic and the
intercept-shift t-statistics match the paper), and the durable case
``benchmarks/cases/cfm.py``.

Core API
--------

.. autoclass:: CFM
   :members: fit

.. autoclass:: mlsynth.config_models.CFMConfig
   :members:

References
----------

.. [BaiWang2026] Bai, J., & Wang, P. (2026). "Causal Inference Using Factor
   Models." Working paper.

Ahn, S. C., & Horenstein, A. R. (2013). "Eigenvalue Ratio Test for the
Number of Factors." *Econometrica* 81(3):1203-1227.

Bai, J. (2009). "Panel Data Models with Interactive Fixed Effects."
*Econometrica* 77(4):1229-1279.

Bai, J., & Ng, S. (2002). "Determining the Number of Factors in Approximate
Factor Models." *Econometrica* 70(1):191-221.

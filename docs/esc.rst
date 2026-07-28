Ensemble Synthetic Control (ESC)
================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The synthetic control (SC) method builds a counterfactual for one treated unit
as a weighted combination of donor units that reproduces the treated unit's
pre-treatment outcome path, then extrapolates that combination into the
post-treatment period. The difficulty is rarely the point estimate; it is
inference. Estimating the donor weights uses only the pre-treatment window, of
length :math:`T_0`, and when :math:`T_0` is short relative to the number of
donors the counterfactual must be extrapolated from very little history. In that
regime the usual devices strain: placebo p-values have coarse resolution and
assume comparable pre-fit across units, and prediction intervals that reuse the
pre-treatment residuals as a model of post-treatment error rest on an
approximation that is fragile when the series trends.

Ensemble Synthetic Control [LianPu2026]_ is built for the short-panel case. It
treats SC inference as a small-sample prediction problem and quantifies
uncertainty directly, without asking pre-treatment errors to stand in for
post-treatment errors. ESC fits a collection of deliberately perturbed,
low-complexity base learners and aggregates them. Each learner is a lasso
regression of the treated unit's pre-treatment outcome on a random subset of the
donors, fit on a block-subsampled slice of the pre-period; the block structure
respects serial dependence, and the donor-subspace draw stabilises estimation
when donors outnumber periods. The point counterfactual is the ensemble average,
and the pointwise prediction interval is read straight off the empirical
quantiles of the ensemble's counterfactuals.

Reach for ESC when the pre-treatment window is short, when the donor pool is
large relative to :math:`T_0`, or when you want interval evidence that does not
depend on equating pre- and post-treatment error distributions. It is the
ensemble sibling of the single-lasso :doc:`scul`: both build the synthetic
control from an :math:`\ell_1`-penalised regression on a high-dimensional donor
pool, but ESC bags many perturbed learners so that the spread across them becomes
the prediction interval.

Notation
--------

One treated unit :math:`i = 1` and donor units :math:`i = 2, \dots, J+1` are
observed over :math:`t = 1, \dots, T`, with treatment beginning at
:math:`T_0 + 1`. Write :math:`y_t` for the treated outcome and
:math:`Y_{0t} \in \mathbb{R}^{J}` for the donor outcomes at period :math:`t`.
ESC works from the local pre-treatment approximation

.. math::

   y_t \;=\; Y_{0t}' w^\star + u_t, \qquad t \le T_0 ,

read as a stable predictive relationship, not a structural model. Each base
learner is indexed by a perturbation :math:`\xi = (S, I)`, where
:math:`S \subseteq \{1, \dots, J\}` is a random donor subset of size
:math:`\lfloor \rho J \rfloor` and :math:`I \subseteq \{1, \dots, T_0\}` is a
block subsample of the pre-period. Restricted to :math:`S` and fit on :math:`I`,
the learner solves the lasso

.. math::

   \hat{w}_S(I) \;=\; \arg\min_{w \in \mathbb{R}^{|S|}}
     \frac{1}{|I|} \bigl\lVert y_{1,I} - Y_{0,I}^{(S)} w \bigr\rVert_2^2
     \;+\; \lambda \lVert w \rVert_1 ,

with :math:`\lambda` chosen by cross-validation on the subsample (the
one-standard-error rule by default). Its counterfactual is
:math:`\hat{y}_t(\xi) = Y_{0t}' \hat{w}(\xi)`. Over :math:`B` perturbations the
ensemble point counterfactual is the average
:math:`\hat{y}_t^{\,\mathrm{ESC}} = B^{-1} \sum_b \hat{y}_t(\xi_b)`, and the
estimated effect is :math:`\hat{\tau}_t = y_t - \hat{y}_t^{\,\mathrm{ESC}}`. For
level :math:`1 - \alpha` the pointwise prediction interval for the counterfactual
is the pair of empirical quantiles
:math:`\bigl[\hat{Q}_{\alpha/2,t},\, \hat{Q}_{1-\alpha/2,t}\bigr]` of
:math:`\{\hat{y}_t(\xi_b)\}_b`, and the interval for the effect is
:math:`\bigl[y_t - \hat{Q}_{1-\alpha/2,t},\, y_t - \hat{Q}_{\alpha/2,t}\bigr]`.

Assumptions
-----------

1. Sparse predictive approximation. There is a sparse weight vector
   :math:`w^\star` with :math:`\lVert w^\star \rVert_0 = s \ll T_0` such that the
   pre-treatment approximation above holds with error :math:`u_t`.

   Remark. This is a local predictive premise, not a claim of structural
   identification or of consistency as :math:`T_0 \to \infty`. Sparsity is what
   lets a lasso learner control its non-asymptotic prediction error at rate
   :math:`O_p(s \log J / T_0)`, so extrapolation stays stable even with a short
   pre-window.

2. Weak temporal dependence. The sequence :math:`\{(Y_{0t}, u_t)\}` is weakly
   dependent (for example mixing), allowing serial correlation and persistent
   components.

   Remark. Time is the effective sample dimension, so the perturbations must
   respect it. ESC subsamples contiguous blocks rather than individual periods;
   independent resampling would implicitly set the autocovariances to zero and
   inflate noise-driven instability when :math:`T_0` is small.

3. Restricted eigenvalue. The donor design satisfies a time-series restricted
   eigenvalue (compatibility) condition on sparse subsets.

   Remark. This is the standard regularity condition under which the lasso
   estimator is well behaved; it supports stable prediction and interval
   construction rather than structural recovery of :math:`w^\star`.

Inference and Diagnostics
-------------------------

ESC's uncertainty is predictive. The pointwise band on the counterfactual (and
its mirror on the effect) comes from the empirical quantiles of the ensemble, so
it carries no reliance on an asymptotic variance or on the assumption that the
pre-treatment residuals describe post-treatment error. The band is stored on
``time_series`` as ``counterfactual_lower`` / ``counterfactual_upper`` (with
``prediction_interval_level`` and ``prediction_interval_kind``); its presence is
reported by ``time_series.has_prediction_interval``. A scalar interval on the ATT
is formed from the quantiles of the per-learner average post-treatment gaps and
is exposed on ``inference`` (``ci_lower`` / ``ci_upper``, with the ensemble
standard error). Pre-treatment fit is the ensemble RMSE on ``fit_diagnostics``.
Because each learner regularises rather than interpolating the pre-period, ESC's
pre-fit is deliberately looser than an unregularised synthetic control's: the
base learners' bias is not a defect but the source of the interval.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import ESC

   # California Proposition 99 tobacco panel (treated 1989).
   df = pd.read_csv("basedata/california_panel.csv")
   df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   res = ESC({
       "df": df, "outcome": "cigsale", "treat": "treat",
       "unitid": "state", "time": "year",
       "n_learners": 300, "block_length": 3, "subspace_ratio": 0.5,
       "alpha": 0.10, "display_graphs": False,
   }).fit()

   print(res.att)                              # average post-treatment effect
   print(res.att_ci)                           # ensemble ATT interval
   ts = res.time_series
   print(ts.counterfactual_lower[-1], ts.counterfactual_upper[-1])  # 90% band, last year

Verification
------------

ESC is validated on two empirical case studies. On California's Proposition 99,
the ensemble counterfactual reproduces the classic synthetic-control path
(correlation about 0.997 with the convex synthetic California) and the ATT
matches the literature, while the prediction interval places observed California
below its lower bound throughout the post-period. On the São Paulo homicide panel
(:math:`T_0 = 9`), ESC returns a moderate central effect with a wide, honest
interval on lives saved that spans the range the individual synthetic-control
variants give as point estimates. See the benchmark cases `esc_prop99.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/esc_prop99.py>`_
and `esc_saopaulo.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/esc_saopaulo.py>`_,
and the :doc:`replications/esc` page.

A caveat on coverage. The band ESC ships is the equation-(7) spread of the
ensemble counterfactuals -- an estimation-uncertainty band. In our Monte Carlo
checks a faithful port of this band under-covers the realized counterfactual in
factor-model designs, because the ensemble spread reflects weight-estimation
variability rather than the treated unit's own disturbance :math:`u_t`. The
paper's nominal-coverage claim therefore did not reproduce from the equations as
written; a calibrated coverage benchmark is deferred pending the authors'
reference implementation.

Core API
--------

.. autoclass:: ESC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.config_models.ESCConfig
   :members:
   :undoc-members:
   :show-inheritance:

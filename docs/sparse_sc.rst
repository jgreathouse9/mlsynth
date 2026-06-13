Sparse Synthetic Control (SparseSC)
===================================

.. currentmodule:: mlsynth

Overview
--------

SparseSC implements the L1-penalized predictor-weighting variant of
canonical synthetic control proposed by Vives-i-Bastida (2023,
*Predictor Selection for Synthetic Controls*). It targets the same
Abadie, Diamond, and Hainmueller (2010) framework as classical SCM,
but adds a *lasso* penalty on the predictor-importance vector
:math:`v` to deliver interpretable *predictor selection*: as the L1
penalty grows, uninformative predictors get :math:`v`-weights of
exactly zero and are dropped from the fit.

Compared with the canonical SCM data-driven :math:`V` choice (a
cross-validated grid search over diagonal :math:`V` minimizing
pre-period MSE), SparseSC

* selects predictors explicitly via L1 sparsity rather than
  implicitly via small but nonzero :math:`v`-weights;
* picks the L1 penalty :math:`\lambda` on a held-out validation
  block of the pre-period (a 75/25 train/validation split by
  default, which matches the 14/5-year split Vives used in the
  empirical Prop 99 application); and
* anchors the first predictor's :math:`v`-weight at 1, which fixes
  the overall scale and removes the trivial :math:`v = 0` minimum
  that the L1 penalty would otherwise admit.

The donor weights :math:`w` solve the usual SCM simplex QP given
:math:`v`.

Inference defaults to a moving-block conformal CI for the ATT
in the spirit of Chernozhukov, Wuethrich and Zhu (2021), calibrated
on the validation-block residuals. Vives's Abadie-style placebo
permutation is still available via ``inference_method="placebo"``.

Mathematical Formulation
------------------------

Setup
^^^^^

Let :math:`Y_{1, t}` denote the treated outcome and :math:`Y_{0, t}
\in \mathbb{R}^N` the donor outcomes at time :math:`t`. The pre-
treatment window :math:`t = 1, \dots, T_0` is partitioned into a
*training* block :math:`t = 1, \dots, T_0^{\text{tr}}` and a
*validation* block :math:`t = T_0^{\text{tr}} + 1, \dots, T_0`.
Predictors enter through a treated vector :math:`X_1 \in
\mathbb{R}^P` and a donor matrix :math:`X_0 \in \mathbb{R}^{P
\times N}`. After standardization each row of :math:`[X_0, X_1]`
has unit sample standard deviation across units.

Inner W-weight QP
^^^^^^^^^^^^^^^^^

Given :math:`v \in \mathbb{R}^P_{\ge 0}` the donor weights solve

.. math::

   w^*(v) = \arg\min_{w \in \Delta_N}
   \; w^\top X_0^\top \mathrm{diag}(v)\, X_0\, w
   - 2\, X_1^\top \mathrm{diag}(v)\, X_0\, w,

where :math:`\Delta_N = \{w \in \mathbb{R}^N_{\ge 0} :
\mathbf{1}^\top w = 1\}` is the donor simplex. This is exactly the
QP MATLAB's ``quadprog`` solves inside
``sparse_synth/loss_function.m``.

:mod:`mlsynth` calls Clarabel directly (bypassing CVXPY's
canonicalization layer), which is the single biggest performance
fix versus the prior CVXPY-based implementation: CVXPY parsing
overhead was ~10-50 ms per call for a 39-donor problem, while the
underlying Clarabel solve itself takes microseconds. The constraint
skeleton ``(A, b, cones, settings)`` is cached per donor count
:math:`N` so only the data terms :math:`H = X_0^\top \mathrm{diag}(v)
X_0` and :math:`q = -2 X_0^\top \mathrm{diag}(v) X_1` are rebuilt per
call.

For numerical robustness — the augmented k > N spec is rank-
deficient and Clarabel can return ``InsufficientProgress`` at tight
tolerance — the inner solve retries with a trace-scaled ridge and
looser tolerances before falling back to a uniform-w feasible point.
This prevents the outer L-BFGS-B sweep from aborting mid-run on a
single bad exploration step.

Outer V-weight problem
^^^^^^^^^^^^^^^^^^^^^^

The :math:`v`-weights minimize a penalized outcome MSE plus the L1
penalty on :math:`v`:

.. math::

   v^*(\lambda)
   = \arg\min_{v \in \mathbb{R}^P_{\ge 0},\; v_1 = 1}
   \; \frac{1}{|\mathcal{T}|}
   \sum_{t \in \mathcal{T}}
   \bigl(Y_{1, t} - Y_{0, t}^\top w^*(v)\bigr)^2
   + \lambda \, \|v\|_1.

The :math:`v_1 = 1` anchor is what prevents the trivial all-zero
solution at any :math:`\lambda > 0`: without it the outer objective
is positive-scale-invariant in :math:`v` and the L1 penalty would
push every component to zero (Vives 2023, Appendix 6.1).

The window :math:`\mathcal{T}` is set by ``outer_loss_window``:

* ``"training"`` (default) — :math:`\mathcal{T} = \{1, \dots,
  T_0^{\text{tr}}\}`. Matches the unpublished MATLAB driver
  ``sparse_synth.m`` and reproduces the Prop 99 estimates Vives
  reports in the empirical section.
* ``"validation"`` — :math:`\mathcal{T} = \{T_0^{\text{tr}} + 1,
  \dots, T_0\}`. Matches the page-5 :math:`L_V` definition in
  Vives's Algorithm 1 literally; useful for ablations but produces
  notably worse in-sample fit than the training variant.

Each evaluation of the outer objective invokes the inner QP, so the
outer problem is a smooth bound-constrained NLP solved with
L-BFGS-B (``scipy.optimize``).

Gradient computation
~~~~~~~~~~~~~~~~~~~~

L-BFGS-B needs gradients of the outer objective in :math:`v`. Two
modes are available, controlled by ``use_analytical_grad``:

* ``False`` (default) — central-difference numerical gradient. Each
  outer step pays :math:`2(P-1)` inner-QP solves.
* ``True`` — closed-form gradient via the envelope theorem applied
  at the inner optimum :math:`w^*(v)`. With active set
  :math:`\mathcal{A} = \{i : w_i^* > 0\}`, one
  :math:`(|\mathcal{A}| + 1) \times (|\mathcal{A}| + 1)` Cholesky on
  the reduced KKT matrix yields all :math:`P - 1` gradient components
  in :math:`O(P |\mathcal{A}|)` work:

  .. math::

     \frac{\partial L}{\partial v_k}
     = -\frac{4}{|\mathcal{T}|}\, r_k \cdot
       \bigl(X_0[k, \mathcal{A}]\, z\bigr) + \lambda,

  where :math:`r_k = X_{1k} - X_0[k, \mathcal{A}] w_{\mathcal{A}}^*`
  is the predictor-:math:`k` pre-fit residual and :math:`z` solves

  .. math::

     \begin{pmatrix}
       2 H_{\mathcal{A}\mathcal{A}} & \mathbf{1} \\
       \mathbf{1}^\top & 0
     \end{pmatrix}
     \begin{pmatrix} z \\ \mu_z \end{pmatrix}
     =
     \begin{pmatrix} Z_0[:, \mathcal{A}]^\top r_{\text{outer}} \\ 0 \end{pmatrix}.

  The analytical gradient is exact (verified against central FD to
  ~1e-7 at random interior points). It yields a ~5–10× speedup on
  the outer sweep, but the cleaner gradient lets L-BFGS-B settle at
  the first critical point near the cold init on the non-convex L1-
  penalized V-objective. The FD path's implicit gradient noise tends
  to find better local optima at non-zero lambda, so the default
  is FD for correctness. Opt in to the analytical path when
  running large placebo sweeps where throughput matters more than
  exact local-optimum reproducibility. When ``use_analytical_grad =
  True``, the L-BFGS-B ``ftol`` auto-tightens to ``1e-12`` because
  the clean gradient converges in many fewer iterations and the
  default ``1e-8`` terminates the loop before convergence.

Lambda selection
^^^^^^^^^^^^^^^^

The penalty :math:`\lambda` is selected by the *unpenalized*
validation-block outcome MSE:

.. math::

   \hat\lambda
   = \arg\min_{\lambda \in \Lambda}
   \; \frac{1}{T_0 - T_0^{\text{tr}}}
   \sum_{t = T_0^{\text{tr}} + 1}^{T_0}
   \bigl(Y_{1, t} - Y_{0, t}^\top w^*(v^*(\lambda))\bigr)^2.

The default grid is :math:`\Lambda = \{0\} \cup
\text{logspace}(10^{-4}, 1, 50)`. Setting :math:`\lambda = 0`
recovers the unpenalized data-driven SCM with a unit-anchored
first predictor.

Predictor selection
^^^^^^^^^^^^^^^^^^^

As :math:`\lambda` grows, the L1 penalty drives uninformative
:math:`v_p` to exactly zero; the corresponding predictor effectively
drops out of the fit. The selected predictor set is

.. math::

   \mathcal{S}(\hat\lambda)
   = \{p : v_p^*(\hat\lambda) > 0\}.

This is what makes the method *Sparse* SC: the explanation of the
treated unit's pre-trajectory is interpretable in terms of a small
subset of predictors.

ATT and Counterfactual
^^^^^^^^^^^^^^^^^^^^^^

With :math:`\hat v = v^*(\hat\lambda)` and :math:`\hat w =
w^*(\hat v)` recovered on the full pre-period, the counterfactual
and ATT are

.. math::

   \widehat{Y}_{1, t} = Y_{0, t}^\top \hat w,
   \qquad
   \widehat{\mathrm{ATT}}
   = \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^T \bigl(Y_{1, t} - \widehat{Y}_{1, t}\bigr).

Conformal ATT inference (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inference defaults to a moving-block conformal CI for the ATT,
following the philosophy of Chernozhukov, Wuethrich and Zhu (2021):
treat the in-sample residuals as a *calibration sample* of what
"noise" should look like under the no-treatment null, and invert a
permutation test in :math:`\theta` to bracket the ATT.

Define the residual series :math:`e_t = Y_{1, t} - Y_{0, t}^\top
\hat w`. The calibration set is

.. math::

   e^{\text{calib}}
   = \begin{cases}
     \{e_t : t \in (T_0^{\text{tr}}, T_0]\} &
       \text{if ``conformal\_window = "validation"`` (default)} \\
     \{e_t : t \in [1, T_0]\} &
       \text{if ``conformal\_window = "pre"``}
   \end{cases}.

The validation block is genuinely out-of-sample under the chosen
:math:`v`; the full pre-block gives a larger calibration sample but
its training-block residuals are in-sample under :math:`v`.

The conformity score for a block :math:`B` of size
:math:`b = \max(3, \lfloor\sqrt{T - T_0}\rfloor)` is

.. math::

   s(B) = \frac{1}{b}\sum_{t \in B} |e_t|,

and the calibration distribution is built by sliding the block
across :math:`e^{\text{calib}}` (with wrap-around blocks for
boundary coverage). The post-treatment test statistic at the
candidate ATT :math:`\theta` is

.. math::

   s_{\text{post}}(\theta)
   = \frac{1}{T - T_0}\sum_{t = T_0 + 1}^T
     \bigl|e_t - \theta\bigr|.

The :math:`(1 - \alpha)` conformal CI is

.. math::

   \mathrm{CI}_{1 - \alpha}
   = \{\theta : \Pr_{B}\bigl(s(B) \ge s_{\text{post}}(\theta)\bigr)
              > \alpha\},

which we compute by grid search over a generous neighbourhood of
:math:`\widehat{\mathrm{ATT}}`. The two-sided p-value for
:math:`H_0 : \mathrm{ATT} = 0` is

.. math::

   p_{\text{conf}}
   = \Pr_{B}\bigl(s(B) \ge s_{\text{post}}(0)\bigr)
   = \frac{1}{|\mathcal{B}|}\sum_{B \in \mathcal{B}}
     \mathbb{1}\{s(B) \ge s_{\text{post}}(0)\}.

Pointwise per-period bands use the :math:`(1 - \alpha)`-quantile of
the calibration scores directly:

.. math::

   [e_t - q_{1 - \alpha},\; e_t + q_{1 - \alpha}],
   \quad q_{1 - \alpha} = \mathrm{Quantile}_{1 - \alpha}\{s(B)\}.

This inferential procedure trades the cross-donor exchangeability
assumption of Vives's placebo (every donor is equally likely to be
the treated unit) for a within-unit exchangeability assumption on
the residuals (validation-period residuals look like the
no-treatment counterfactual's noise). On Prop 99 the conformal
95% CI is typically :math:`[-20, -18]` versus the placebo's much
wider bounds, because conformal leverages the actual model's
residual structure rather than donor-level heterogeneity.

Abadie-style placebo (opt-in)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``inference_method="placebo"`` to recover Vives's procedure.
For each donor :math:`j`, swap that donor into the treated slot,
remove it from the donor pool, refit SparseSC *at the already-
selected* :math:`\hat\lambda` (or, optionally, re-run the full
:math:`\lambda` sweep) and record the placebo ATT. The two-sided
permutation p-value is

.. math::

   p = \frac{\#\{j : |\mathrm{ATT}_j^{\text{placebo}}|
   \ge |\widehat{\mathrm{ATT}}|\} + 1}{B + 1},

where :math:`B` is the number of completed placebos. Re-using
:math:`\hat\lambda` makes the placebo loop tractable; set
``placebo_resweep=True`` to re-select :math:`\lambda` for every
placebo (much slower).

Predictor Convention
--------------------

Like every other :mod:`mlsynth` estimator, SparseSC is fed a single
long-format ``df`` with one row per ``(unit, time)``. Predictors are
constructed under the hood from the same frame, in two flavors:

* ``covariates`` — column names in ``df`` whose per-unit pre-
  treatment mean is taken as the predictor value. Time-invariant
  unit characteristics collapse trivially; time-varying covariates
  are summarized by the pre-period mean.
* ``outcome_lag_periods`` — specific pre-treatment time labels (as
  found in the ``time`` column) whose outcome values become
  additional predictor rows. These are the canonical Abadie,
  Diamond & Hainmueller (2010) lagged-outcome predictors (e.g., the
  ``smk_75``, ``smk_80``, ``smk_88`` rows in the Prop 99 example).

The two lists are concatenated to form the predictor matrix; the
first predictor (first entry of ``covariates`` if any, otherwise
the first outcome lag) is the *anchor* whose :math:`v`-weight is
fixed at 1. The anchor choice matters in finite samples — Vives
recommends picking a predictor known to be informative, or
treating the anchor as a hyperparameter and sweeping it (Vives
2023, Appendix 6.1).

Performance notes
-----------------

The single biggest cost in a SparseSC fit is the inner W-weight QP,
which is invoked

.. math::

   |\Lambda| \times (\text{outer iters}) \times g

times where :math:`g = 2(P-1)` under finite-difference gradients
and :math:`g = 1` under the analytical gradient. For Vives's
augmented k=40 spec that's ~50,000 inner-QP calls just for the
fit, plus another :math:`B \approx 38` placebos under
``inference_method="placebo"``. Two optimizations in this build
matter:

* Direct Clarabel removes CVXPY canonicalization (~30-60× per
  call). Speedup applies universally; no correctness tradeoff.
* Analytical gradient (opt-in via ``use_analytical_grad=True``)
  removes the :math:`2(P-1)` finite-difference factor (~5-10× on
  the outer loop). Tradeoff: the cleaner gradient can settle in
  worse local optima of the non-convex L1-penalized outer
  objective; FD's implicit gradient noise tends to escape them.
  Default off for correctness.

Empirically, the combination puts the canonical ADH-7 California
Prop 99 fit at ~5 s with analytical gradient and ~23 s with FD
(versus a CVXPY+SCS baseline that would hang for minutes on the
augmented k=40 spec).

Core API
--------

.. automodule:: mlsynth.estimators.sparse_sc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SparseSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.sparse_sc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.inner
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.objective
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.sparse_sc_helpers.structures
   :members:
   :undoc-members:

Replication
-----------

SparseSC is verified on the canonical Proposition 99 panel: handed an
over-rich augmented predictor set, the L1 penalty prunes to 6 of 33
predictors and the effect lands at :math:`-17.9` packs (95% conformal CI
:math:`[-21.3, -15.4]`) on the Abadie-Diamond-Hainmueller donor pool. See the
dedicated :doc:`replication page <replications/sparse_sc>` for the full
specification, code, and number-match.

Example
-------

The canonical empirical example is Vives's augmented California
Proposition 99 study. Load the reshaped long-form panel and run
SparseSC with the original ADH-7 predictor set:

.. code-block:: python

   """Run SparseSC on the long-form augmented California dataset."""

   from __future__ import annotations

   import pandas as pd

   from mlsynth import SparseSC


   # ---------------------------------------------------------------------
   # Load long-form panel
   # ---------------------------------------------------------------------

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/augmented_cali_long.csv"
   )

   LAG_PERIODS = [1975, 1980, 1988]

   COVARIATES = [
       "p_cig",
       "loginc",
       "pct15-24",
       "pc_beer",
   ]


   # ---------------------------------------------------------------------
   # SparseSC fit
   # ---------------------------------------------------------------------

   results = SparseSC(
       {
           "df": df,
           "outcome": "cigsale",
           "treat": "Proposition 99",
           "unitid": "state",
           "time": "year",
           "covariates": COVARIATES,
           "outcome_lag_periods": LAG_PERIODS,
           "display_graphs": True,
           "run_inference": False,
       }
   ).fit()

Enable inference (validation-block conformal is the default) and
inspect the ATT CI:

.. code-block:: python

   results = SparseSC({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",
       "unitid": "state",
       "time": "year",
       "covariates": COVARIATES,
       "outcome_lag_periods": LAG_PERIODS,
       "alpha": 0.05,                       # CI level
       "run_inference": True,
       "display_graphs": False,
   }).fit()

   print(results.att)                       # post-period ATT
   lo, hi = results.att_ci                  # 95% conformal CI (standardized)
   print(lo, hi)
   print(results.inference.p_value)         # H_0: ATT = 0 (standardized slot)
   print(results.inference_detail.method)   # raw placebo/conformal object
   print(results.design.opt_lambda)         # selected L1 penalty
   print(results.predictor_weights)         # {predictor: v_p}
   print(results.donor_weights)             # {donor: w_j} on the simplex

   # Lambda sweep diagnostics.
   import matplotlib.pyplot as plt
   plt.plot(results.design.lambda_grid, results.design.val_mse_curve)
   plt.xscale("log"); plt.xlabel("lambda"); plt.ylabel("validation MSE")

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies: Estimating the Effect of
California's Tobacco Control Program." *Journal of the American
Statistical Association* 105(490):493-505.

Chernozhukov, V., Wuethrich, K., & Zhu, Y. (2021). "An Exact and
Robust Conformal Inference Method for Counterfactual and Synthetic
Controls." *Journal of the American Statistical Association*
116(536):1849-1864.

Vives-i-Bastida, J. (2023). "Predictor Selection for Synthetic
Controls." arXiv:2203.11576v2.

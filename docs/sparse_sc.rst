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
* picks the L1 penalty :math:`\lambda` on a *held-out* validation
  block of the pre-period (a 75/25 train/validation split by
  default), not on the in-sample training block; and
* anchors the first predictor's :math:`v`-weight at 1, which fixes
  the overall scale and removes the trivial :math:`v = 0` minimum.

The donor weights :math:`w` solve the usual SCM simplex QP given
:math:`v`.

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
QP that MATLAB's ``quadprog`` solves inside
``sparse_synth/loss_function.m``; :mod:`mlsynth` uses cvxpy with
OSQP (and SCS as fallback).

Outer V-weight problem
^^^^^^^^^^^^^^^^^^^^^^

The :math:`v`-weights minimize a penalized *validation*-block
outcome MSE (the upper-level loss :math:`L_V` of the paper):

.. math::

   v^*(\lambda)
   = \arg\min_{v \in \mathbb{R}^P_{\ge 0},\; v_1 = 1}
   \; \frac{1}{T_0 - T_0^{\text{tr}}}
   \sum_{t = T_0^{\text{tr}} + 1}^{T_0}
   \bigl(Y_{1, t} - Y_{0, t}^\top w^*(v)\bigr)^2
   + \lambda \, \|v\|_1.

The :math:`v_1 = 1` anchor is what prevents the trivial all-zero
solution at any :math:`\lambda > 0`. Each evaluation of the outer
objective invokes the inner QP, so the outer problem is a smooth
bound-constrained NLP solved with L-BFGS-B (``scipy.optimize``).

The unpublished MATLAB driver ``sparse_synth.m`` evaluates the
outer outcome MSE on the *training* block instead, which gives
genuinely different :math:`v^*(\lambda)`. That behavior is
available via ``outer_loss_window="training"``. The default
follows the paper.

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

This is what makes the method "Sparse" SC: the explanation of the
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

Abadie-style placebo inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
long-format ``df`` with one row per ``(unit, time)``. Predictors
are constructed under the hood from the same frame, in two flavors:

* ``covariates`` — column names in ``df`` whose **per-unit pre-
  treatment mean** is taken as the predictor value. Time-invariant
  unit characteristics collapse trivially; time-varying covariates
  are summarized by the pre-period mean.
* ``outcome_lag_periods`` — specific pre-treatment time labels (as
  found in the ``time`` column) whose **outcome values** become
  additional predictor rows. These are the canonical Abadie,
  Diamond & Hainmueller (2010) lagged-outcome predictors (e.g., the
  ``smk_75``, ``smk_80``, ``smk_88`` rows in the Prop 99 example).

The two lists are concatenated to form the predictor matrix; the
first predictor (first entry of ``covariates`` if any, otherwise
the first outcome lag) is the *anchor* whose :math:`v`-weight is
fixed at 1. Choose an anchor that is at least somewhat informative
— its scale fixes the overall penalty scale.

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

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import SparseSC

   # Long-format panel: one row per (state, year). Covariate columns
   # (p_cig, loginc, pct15-24, pc_beer) sit alongside cigsale.
   df = pd.read_csv("smoking_long.csv")

   results = SparseSC({
       "df":                  df,
       "outcome":             "cigsale",
       "treat":               "Proposition 99",
       "unitid":              "state",
       "time":                "year",
       "covariates":          ["p_cig", "loginc", "pct15-24", "pc_beer"],
       "outcome_lag_periods": [1975, 1980, 1988],  # ADH lagged-outcome predictors
       "outer_loss_window":   "validation",        # default (paper)
       "standardize":         True,                # default
       "run_inference":       True,
       "n_placebo":           None,                # use all donors
       "placebo_resweep":     False,
       "display_graphs":      True,
   }).fit()

   print(results.att)                     # post-period ATT
   print(results.design.opt_lambda)       # selected L1 penalty
   print(results.design.v)                # v-weights (first = 1)
   print(results.predictor_weights)       # {predictor: v_p}
   print(results.donor_weights)           # {donor: w_j} on the simplex
   print(results.inference.p_value)       # Abadie placebo p-value

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

Vives-i-Bastida, J. (2023). "Predictor Selection for Synthetic
Controls." arXiv:2203.11576v2.

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
:math:`\mathbf{v}` to deliver interpretable *predictor selection*: as the L1
penalty grows, uninformative predictors get :math:`v_p`-weights of
exactly zero and are dropped from the fit.

Compared with the canonical SCM data-driven :math:`\mathbf{V}` choice (a
cross-validated grid search over diagonal :math:`\mathbf{V}` minimizing
pre-period MSE), SparseSC

* selects predictors explicitly via L1 sparsity rather than
  implicitly via small but nonzero :math:`v_p`-weights;
* picks the L1 penalty :math:`\lambda` on a held-out validation
  block of the pre-period (a 75/25 train/validation split by
  default, which matches the 14/5-year split Vives used in the
  empirical Prop 99 application); and
* anchors the first predictor's :math:`v_p`-weight at 1, which fixes
  the overall scale and removes the trivial :math:`\mathbf{v} = \mathbf{0}` minimum
  that the L1 penalty would otherwise admit.

The donor weights :math:`\mathbf{w}` solve the usual SCM simplex QP given
:math:`\mathbf{v}`.

Inference defaults to a moving-block conformal CI for the ATT
in the spirit of Chernozhukov, Wuethrich and Zhu (2021), calibrated
on the validation-block residuals. Vives's Abadie-style placebo
permutation is still available via ``inference_method="placebo"``.

When to use this estimator
--------------------------

Reach for SparseSC when you have one treated unit, a rich predictor
set, and you want the fit to tell you *which* predictors matter rather
than carry all of them with small but nonzero weights. The lasso
penalty on the predictor-importance vector drives the uninformative
predictors to exactly zero, so the synthetic control's explanation of
the treated pre-trajectory is interpretable in terms of a small,
named subset.

A concrete example: a state passes a tobacco-control law and you have
dozens of candidate predictors of cigarette sales -- prices, income,
beer consumption, the youth share, and several lagged outcomes. With
the canonical data-driven :math:`\mathbf{V}` every predictor keeps some
weight and the story is muddy. SparseSC prunes the over-rich set down
to the handful that actually drive the pre-law fit, selects the L1
penalty on a held-out validation block, and reads the policy effect as
the post-law gap -- with a conformal interval calibrated on the
validation residuals rather than a coarse donor-permutation grid.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of cardinality
:math:`N_0`. Time runs over :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.

The treated outcome is :math:`y_{1t}`; each donor :math:`j \in \mathcal{N}_0`
contributes a series :math:`\mathbf{y}_j`, stacked into the donor matrix
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0} \in
\mathbb{R}^{T \times N_0}` (one column per donor); write
:math:`\mathbf{y}_{0t} \in \mathbb{R}^{N_0}` for the donor outcomes at time
:math:`t` (the :math:`t`-th row). Donor weights are
:math:`\mathbf{w} \in \mathbb{R}^{N_0}`, constrained to the unit simplex
:math:`\Delta^{N_0} \coloneqq \{\mathbf{w} \in \mathbb{R}_{\ge 0}^{N_0} :
\|\mathbf{w}\|_1 = 1\}`; the optimiser is :math:`\mathbf{w}^\ast`. The
synthetic counterfactual is :math:`\widehat{y}_{1t} \coloneqq
\mathbf{y}_{0t}^\top \mathbf{w}^\ast`, the per-period effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1} \sum_{t \in \mathcal{T}_2}
\tau_t`. The significance level is :math:`\alpha`.

Predictors enter through a treated vector :math:`\mathbf{x}_1 \in
\mathbb{R}^P` and a donor matrix :math:`\mathbf{X}_0 \in \mathbb{R}^{P
\times N_0}` over :math:`P` predictors; after standardization each row of
:math:`[\mathbf{X}_0, \mathbf{x}_1]` has unit sample standard deviation
across units. The predictor-importance vector is
:math:`\mathbf{v} \in \mathbb{R}^P_{\ge 0}` with components :math:`v_p`;
the L1 penalty strength is :math:`\lambda \ge 0`, and :math:`\mathbf{V} =
\mathrm{diag}(\mathbf{v})` is the diagonal predictor-weight matrix.

The pre-period is further split for tuning. The training block is
:math:`\mathcal{T}_1^{\mathrm{tr}} \coloneqq \{1, \dots, T_0^{\mathrm{tr}}\}`
and the validation block is :math:`\mathcal{T}_1^{\mathrm{val}} \coloneqq
\{T_0^{\mathrm{tr}} + 1, \dots, T_0\}`, with
:math:`\mathcal{T}_1 = \mathcal{T}_1^{\mathrm{tr}} \cup
\mathcal{T}_1^{\mathrm{val}}` (default 75/25). This pre-period split is
internal to predictor and penalty selection and is distinct from the
canonical pre/post split at :math:`T_0`.

Identifying assumptions
-----------------------

1. Pre-treatment fit / convex-hull support. There exist weights
   :math:`\mathbf{w} \in \Delta^{N_0}` and predictor weights :math:`\mathbf{v}`
   under which the treated pre-period predictors are matched by the donors,
   :math:`\mathbf{x}_1 \approx \mathbf{X}_0 \mathbf{w}` -- equivalently, the
   treated unit lies inside (or near) the convex hull of the donors over the
   selected predictors (Abadie, Diamond & Hainmueller 2010).

   *Remark.* This is the workhorse identifying condition for any synthetic
   control: a good pre-period match is the empirical certificate one inspects.
   SparseSC adds that the match should be achievable with *few* predictors -- if
   it is, the lasso recovers a sparse, interpretable explanation; if the treated
   unit can only be matched by leaning on many predictors at once, the selected
   set will not be sparse.

2. Informative anchor. The first predictor (the anchor, with
   :math:`v_1 = 1`) is genuinely informative about the treated unit's
   pre-trajectory (Vives 2023, Appendix 6.1).

   *Remark.* The anchor fixes the overall scale of :math:`\mathbf{v}` and removes
   the trivial :math:`\mathbf{v} = \mathbf{0}` minimum the L1 penalty would
   otherwise admit. Because the anchor's weight is pinned at 1, a poorly chosen
   anchor biases the fit; Vives recommends picking a predictor known to be
   informative, or treating the anchor as a hyperparameter and sweeping it.

3. No anticipation. Treatment has no effect before :math:`T_0`:
   :math:`y_{1t} = y_{1t}^N` for all :math:`t \in \mathcal{T}_1`, so the
   pre-period outcomes and predictors reflect the no-intervention path.

   *Remark.* If the treated unit reacts in advance of the formal intervention
   date, the pre-period fit -- and the validation-block residuals the conformal
   interval is calibrated on -- are contaminated by the effect itself. Date
   :math:`T_0` at the first plausible response, not the nominal policy date.

4. Outcome-model stability. The no-intervention outcomes follow a stable
   data-generating process across :math:`\mathcal{T}`, so weights selected on
   :math:`\mathcal{T}_1` continue to reproduce the treated unit's
   no-intervention path on :math:`\mathcal{T}_2` (Abadie, Diamond &
   Hainmueller 2010).

   *Remark.* This is what licenses extrapolating the pre-period fit forward, and
   it is also what the validation block stress-tests: holding out
   :math:`\mathcal{T}_1^{\mathrm{val}}` checks that the selected predictors and
   penalty generalise within the pre-period before they are asked to generalise
   past :math:`T_0`.

Mathematical Formulation
------------------------

Inner W-weight QP
^^^^^^^^^^^^^^^^^

Given :math:`\mathbf{v} \in \mathbb{R}^P_{\ge 0}` the donor weights solve

.. math::

   \mathbf{w}^\ast(\mathbf{v}) = \operatorname*{argmin}_{\mathbf{w} \in \Delta^{N_0}}
   \; \mathbf{w}^\top \mathbf{X}_0^\top \mathrm{diag}(\mathbf{v})\, \mathbf{X}_0\, \mathbf{w}
   - 2\, \mathbf{x}_1^\top \mathrm{diag}(\mathbf{v})\, \mathbf{X}_0\, \mathbf{w},

where :math:`\Delta^{N_0} = \{\mathbf{w} \in \mathbb{R}^{N_0}_{\ge 0} :
\mathbf{1}^\top \mathbf{w} = 1\}` is the donor simplex. This is exactly the
QP MATLAB's ``quadprog`` solves inside
``sparse_synth/loss_function.m``.

:mod:`mlsynth` calls Clarabel directly (bypassing CVXPY's
canonicalization layer), which is the single biggest performance
fix versus the prior CVXPY-based implementation: CVXPY parsing
overhead was ~10-50 ms per call for a 39-donor problem, while the
underlying Clarabel solve itself takes microseconds. The constraint
skeleton ``(A, b, cones, settings)`` is cached per donor count
:math:`N_0` so only the data terms :math:`\mathbf{H} = \mathbf{X}_0^\top
\mathrm{diag}(\mathbf{v}) \mathbf{X}_0` and :math:`\mathbf{q} = -2
\mathbf{X}_0^\top \mathrm{diag}(\mathbf{v}) \mathbf{x}_1` are rebuilt per
call.

For numerical robustness — the augmented k > N spec is rank-
deficient and Clarabel can return ``InsufficientProgress`` at tight
tolerance — the inner solve retries with a trace-scaled ridge and
looser tolerances before falling back to a uniform-w feasible point.
This prevents the outer L-BFGS-B sweep from aborting mid-run on a
single bad exploration step.

Outer V-weight problem
^^^^^^^^^^^^^^^^^^^^^^

The :math:`\mathbf{v}`-weights minimize a penalized outcome MSE plus the L1
penalty on :math:`\mathbf{v}`, over a window :math:`\mathcal{W}`:

.. math::

   \mathbf{v}^\ast(\lambda)
   = \operatorname*{argmin}_{\mathbf{v} \in \mathbb{R}^P_{\ge 0},\; v_1 = 1}
   \; \frac{1}{|\mathcal{W}|}
   \sum_{t \in \mathcal{W}}
   \bigl(y_{1t} - \mathbf{y}_{0t}^\top \mathbf{w}^\ast(\mathbf{v})\bigr)^2
   + \lambda \, \|\mathbf{v}\|_1 .

The :math:`v_1 = 1` anchor is what prevents the trivial all-zero
solution at any :math:`\lambda > 0`: without it the outer objective
is positive-scale-invariant in :math:`\mathbf{v}` and the L1 penalty would
push every component to zero (Vives 2023, Appendix 6.1).

The window :math:`\mathcal{W}` is set by ``outer_loss_window``:

* ``"training"`` (default) — :math:`\mathcal{W} =
  \mathcal{T}_1^{\mathrm{tr}}`. Matches the unpublished MATLAB driver
  ``sparse_synth.m`` and reproduces the Prop 99 estimates Vives
  reports in the empirical section.
* ``"validation"`` — :math:`\mathcal{W} = \mathcal{T}_1^{\mathrm{val}}`.
  Matches the page-5 :math:`L_V` definition in
  Vives's Algorithm 1 literally; useful for ablations but produces
  notably worse in-sample fit than the training variant.

Each evaluation of the outer objective invokes the inner QP, so the
outer problem is a smooth bound-constrained NLP solved with
L-BFGS-B (``scipy.optimize``).

Gradient computation
~~~~~~~~~~~~~~~~~~~~

L-BFGS-B needs gradients of the outer objective in :math:`\mathbf{v}`. Two
modes are available, controlled by ``use_analytical_grad``:

* ``False`` (default) — central-difference numerical gradient. Each
  outer step pays :math:`2(P-1)` inner-QP solves.
* ``True`` — closed-form gradient via the envelope theorem applied
  at the inner optimum :math:`\mathbf{w}^\ast(\mathbf{v})`. With active set
  :math:`\mathcal{A} = \{i : w_i^\ast > 0\}`, one
  :math:`(|\mathcal{A}| + 1) \times (|\mathcal{A}| + 1)` Cholesky on
  the reduced KKT matrix yields all :math:`P - 1` gradient components
  in :math:`O(P |\mathcal{A}|)` work:

  .. math::

     \frac{\partial \mathcal{L}}{\partial v_p}
     = -\frac{4}{|\mathcal{W}|}\, r_p \cdot
       \bigl(\mathbf{X}_0[p, \mathcal{A}]\, \mathbf{z}\bigr) + \lambda,

  where :math:`r_p = x_{1p} - \mathbf{X}_0[p, \mathcal{A}]\, \mathbf{w}_{\mathcal{A}}^\ast`
  is the predictor-:math:`p` pre-fit residual and :math:`\mathbf{z}` solves

  .. math::

     \begin{pmatrix}
       2 \mathbf{H}_{\mathcal{A}\mathcal{A}} & \mathbf{1} \\
       \mathbf{1}^\top & 0
     \end{pmatrix}
     \begin{pmatrix} \mathbf{z} \\ \mu_z \end{pmatrix}
     =
     \begin{pmatrix} \mathbf{Z}_0[:, \mathcal{A}]^\top \mathbf{r}_{\text{outer}} \\ 0 \end{pmatrix}.

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

   \widehat{\lambda}
   = \operatorname*{argmin}_{\lambda \in \Lambda}
   \; \frac{1}{T_0 - T_0^{\mathrm{tr}}}
   \sum_{t \in \mathcal{T}_1^{\mathrm{val}}}
   \bigl(y_{1t} - \mathbf{y}_{0t}^\top \mathbf{w}^\ast(\mathbf{v}^\ast(\lambda))\bigr)^2 .

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

   \mathcal{S}(\widehat{\lambda})
   = \{p : v_p^\ast(\widehat{\lambda}) > 0\}.

This is what makes the method *Sparse* SC: the explanation of the
treated unit's pre-trajectory is interpretable in terms of a small
subset of predictors.

ATT and Counterfactual
^^^^^^^^^^^^^^^^^^^^^^

With :math:`\widehat{\mathbf{v}} = \mathbf{v}^\ast(\widehat{\lambda})` and
:math:`\widehat{\mathbf{w}} = \mathbf{w}^\ast(\widehat{\mathbf{v}})` recovered on
the full pre-period, the counterfactual and ATT are

.. math::

   \widehat{y}_{1t} = \mathbf{y}_{0t}^\top \widehat{\mathbf{w}},
   \qquad
   \widehat{\tau}
   = \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^T \bigl(y_{1t} - \widehat{y}_{1t}\bigr).

Conformal ATT inference (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inference defaults to a moving-block conformal CI for the ATT,
following the philosophy of Chernozhukov, Wuethrich and Zhu (2021):
treat the in-sample residuals as a *calibration sample* of what
"noise" should look like under the no-treatment null, and invert a
permutation test in :math:`\theta` to bracket the ATT.

Define the residual series :math:`e_t = y_{1t} - \mathbf{y}_{0t}^\top
\widehat{\mathbf{w}}`. The calibration set is

.. math::

   e^{\text{calib}}
   = \begin{cases}
     \{e_t : t \in (T_0^{\text{tr}}, T_0]\} &
       \text{if ``conformal\_window = "validation"`` (default)} \\
     \{e_t : t \in [1, T_0]\} &
       \text{if ``conformal\_window = "pre"``}
   \end{cases}.

The validation block is genuinely out-of-sample under the chosen
:math:`\mathbf{v}`; the full pre-block gives a larger calibration sample but
its training-block residuals are in-sample under :math:`\mathbf{v}`.

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
selected* :math:`\widehat\lambda` (or, optionally, re-run the full
:math:`\lambda` sweep) and record the placebo ATT. The two-sided
permutation p-value is

.. math::

   p = \frac{\#\{j : |\mathrm{ATT}_j^{\text{placebo}}|
   \ge |\widehat{\mathrm{ATT}}|\} + 1}{B + 1},

where :math:`B` is the number of completed placebos. Re-using
:math:`\widehat\lambda` makes the placebo loop tractable; set
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

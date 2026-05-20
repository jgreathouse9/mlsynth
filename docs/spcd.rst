Synthetic Principal Component Design (SPCD)
===========================================

.. currentmodule:: mlsynth

Overview
--------

Synthetic Principal Component Design (SPCD)
`arXiv:2211.15241 <https://arxiv.org/abs/2211.15241>`_ is an
experimental-design estimator for synthetic-control settings. Like
:doc:`scdi`, it selects a treated group and constructs synthetic
contrasts directly from the pre-treatment outcomes — but instead of
solving a mixed-integer program or a simulated-annealing relaxation,
SPCD reformulates the design problem as a *phase-synchronization*
problem and solves it with a spectrally-initialized generalized power
method that converges globally under standard linear-factor models.

SPCD exposes two orthogonal choices:

- ``variant``: which power-method update box to iterate.

  - ``"spcd"`` — generalized power iteration (Eq. (4)/(7) of the paper).
  - ``"norm_spcd"`` — *normalized* generalized power iteration
    (Eq. (5)/(8)). This is the variant used in all of the paper's
    Section 4 experiments and is the package default.

- ``weights``: how to compute the final unit weights once the design's
  sign vector has converged.

  - ``"empirical"`` — closed-form weights from Eq. (9). Fast,
    differentiable, and is the choice the authors use in every
    experiment they report.
  - ``"exact"`` — solves the convex QP Eq. (6) via cvxpy. Slower but
    matches Algorithm 1 of the paper to the letter.

Mathematical Formulation
------------------------

Let :math:`Y \in \mathbb{R}^{N \times T}` be the pre-treatment outcome
matrix with units in rows and periods in columns. Each unit :math:`i`
is assigned a sign :math:`D_i \in \{-1, +1\}`: :math:`D_i = +1` if the
unit is selected into the treated group, :math:`D_i = -1` if into the
control group. SPCD jointly chooses :math:`\{D_i\}_{i=1}^N` and
non-negative weights :math:`\{w_i\}_{i=1}^N` so that the weighted
pre-treatment trajectories of the two groups *track each other as
closely as possible*.

Following Doudchenko et al. (2021) and Abadie & Zhao (2021), the
expected MSE of the difference-in-weighted-means treatment-effect
estimator decomposes as

.. math::

   \mathbb{E}\!\left[(\hat\tau - \tau)^2 \,\middle|\, \{D_i, w_i\}\right]
   \;=\;
   \underbrace{\left(
       \sum_{i: D_i=1} w_i \mu_{i, T+1}
       -
       \sum_{i: D_i=-1} w_i \mu_{i, T+1}
   \right)^2}_{\text{weighted covariate balancing}}
   \;+\;
   \sigma^2 \sum_{i=1}^N w_i^2.

Minimizing the first term over the *pre-treatment window* gives the
mixed-integer program SPCD solves:

.. math::

   \begin{aligned}
   \min_{\{D_i, w_i\}_{i=1}^N} \quad
   & \frac{1}{T} \sum_{t=1}^T
     \left(
       \sum_{i: D_i=1} w_i Y_{it}
       -
       \sum_{i: D_i=-1} w_i Y_{it}
     \right)^2
     + \sigma \sum_{i=1}^N w_i^2 \\
   \text{s.t.} \quad
   & w_i \geq 0,\quad D_i \in \{-1, +1\}, \\
   & \sum_{i: D_i=1} w_i \;=\; \sum_{i: D_i=-1} w_i \;=\; 1.
   \end{aligned}

This is Eq. (1) of the paper.

Reformulation as :math:`\ell_1`-PCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key observation of the paper is that the change of variable
:math:`W_i = w_i D_i` collapses the assignment and weight variables
into a single signed vector :math:`W \in \mathbb{R}^N`. Under
:math:`W`, :math:`D_i = \operatorname{sgn}(W_i)` and
:math:`w_i = |W_i|`, and the design problem becomes

.. math::

   \min_{W \in \mathbb{R}^N,\ \mathbf{1}^\top W = 0,\ \|W\|_1 = 1}
       W^\top \left( Y Y^\top + \sigma I \right) W.

The hard equality :math:`\mathbf{1}^\top W = 0` is absorbed into the
objective via a quadratic penalty:

.. math::

   \min_{W \in \mathbb{R}^N,\ \|W\|_1 = 1}
       W^\top \left( Y Y^\top + \sigma I + \lambda \mathbf{1}\mathbf{1}^\top \right) W.

Theorem 1 of the paper shows that for :math:`\lambda` large enough,
the *sign vector* :math:`\operatorname{sgn}(W^*)` of any global
minimum coincides with the sign vector of an associated phase
synchronization problem. Once the signs are recovered, the magnitudes
follow from a small convex QP (or, in practice, the closed form in
Eq. (9)).

The Iteration Matrix
^^^^^^^^^^^^^^^^^^^^

All four code-paths of SPCD operate on the same :math:`N \times N`
iteration matrix from Eq. (2) of the paper:

.. math::

   M \;=\; Y Y^\top \;+\; \alpha I \;+\; \lambda \mathbf{1} \mathbf{1}^\top,

where :math:`\alpha` plays the role of :math:`\sigma` (noise variance)
and :math:`\lambda` is the constraint-absorbing penalty. When the user
does not supply these hyperparameters, ``formulation.py`` auto-defaults
them from the spectrum of :math:`Y Y^\top`.

Spectral Initialization
^^^^^^^^^^^^^^^^^^^^^^^

Both Algorithm 1 and Algorithm 2 of the paper start from the same warm
start: the sign of the smallest eigenvector of :math:`M`,

.. math::

   y^{0} \;=\; \operatorname{sgn}(v),
   \qquad
   v \;=\; \arg\min_{\|v\|_2 = 1} v^\top M v.

Appendix 3.2.1 of the paper (Lemma 4) shows that under the linear
latent-factor model :math:`Y_{it} = \delta_t + \theta_t^\top \mu_i +
e_{it}` (Assumption 1) together with the realizability assumption
(Assumption 2), the sign vector :math:`\operatorname{sgn}(v)` already
agrees with the global optimum :math:`\operatorname{sgn}(W^*)` up to a
bounded fraction of entries.

The (Normalized) Generalized Power Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To refine the spectral initialization, SPCD performs sign-iterations
on :math:`M^{-1}`. The two ``variant`` choices correspond to the two
update boxes of Algorithm 1 / Algorithm 2:

- ``variant="spcd"`` (Eq. (4)/(7)) — *Generalized Power Iteration*:

  .. math::

     y^{t+1} \;=\;
     \operatorname{sgn}\!\left[\,
         \left( M^{-1} + \beta I \right) \, y^{t}
     \,\right].

- ``variant="norm_spcd"`` (Eq. (5)/(8)) — *Normalized Generalized
  Power Iteration*:

  .. math::

     y^{t+1} \;=\;
     \operatorname{sgn}\!\left[\,
         \left( M^{-1} + \beta I \right) \,
         \big( y^{t} \,/\, d \big)
     \,\right],
     \quad
     d \;=\; \sqrt{\operatorname{diag}(M^{-1})}.

Here :math:`\beta > 0` is the step parameter (auto-defaulted to
:math:`1/\lambda_{\max}(M)`), and :math:`/` denotes element-wise
division. The loop terminates as soon as the sign vector stops
changing between successive iterations.

Under Assumptions 1 and 2 with small enough idiosyncratic noise,
Theorem 3 of the paper guarantees:

- ``variant="spcd"`` converges globally if :math:`\epsilon \,>\,
  (\sqrt{3}/2) - 1`;
- ``variant="norm_spcd"`` converges *linearly* and globally for any
  :math:`\epsilon > 0`.

This is why the normalized variant is the package default.

Final Weight Step
^^^^^^^^^^^^^^^^^

Once the iteration converges to a sign vector :math:`y^* \in \{-1,
+1\}^N`, the unit weights are produced by one of two procedures:

- ``weights="empirical"`` (Eq. (9), Algorithm 2 — the paper's
  experimental default):

  .. math::

     w \;=\;
     \frac{2 \, M^{-1} y^*}{\left\| M^{-1} y^* \right\|_1}.

  The optimality condition of the original QP (Eq. (6)) implies that
  :math:`\operatorname{sgn}(w) = y^*` whenever the iteration has
  converged to a fixed point of the closed-form map, so the signed
  vector :math:`w` simultaneously encodes group membership *and*
  weights.

- ``weights="exact"`` (Eq. (6), Algorithm 1) — solves the convex QP

  .. math::

     \begin{aligned}
     \min_{w \geq 0} \quad
     & \frac{1}{T} \sum_{t=1}^T
       \left(
         \sum_{i:\, y^*_i = +1} w_i Y_{it}
         -
         \sum_{i:\, y^*_i = -1} w_i Y_{it}
       \right)^2
       + \alpha \sum_{i=1}^N w_i^2 \\
     \text{s.t.} \quad
     & \sum_{i:\, y^*_i = +1} w_i \;=\; \sum_{i:\, y^*_i = -1} w_i \;=\; 1.
     \end{aligned}

  via ``cvxpy``. Use this when you need the exact Algorithm-1 weights
  rather than the closed-form approximation.

Minority Convention
^^^^^^^^^^^^^^^^^^^

Per the bottom of Algorithm 1 (page 7 of the paper), SPCD then applies
the *minority-group rule*

.. math::

   \text{Treat unit } i \iff y^*_i \;=\; -\operatorname{sgn}\!\left(\sum_{j} y^*_j\right),

which flips the sign vector (if necessary) so that the smaller of the
two groups is the treated group. The treated unit labels reported by
:py:meth:`SPCDResults.selected_unit_labels` follow this convention.

Treatment Effect and Pre-Period Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For post-treatment periods :math:`t = T+1, \dots, T+S`, the SPCD
treatment-effect estimator at the bottom of Algorithm 1 is

.. math::

   \hat\tau
   \;=\;
   \frac{1}{S} \sum_{t = T+1}^{T+S}
   \left(
     \sum_{i: y^*_i = +1} w_i \, Y_{i,t}
     -
     \sum_{i: y^*_i = -1} w_i \, Y_{i,t}
   \right),

which is precisely the mean of the post-period synthetic gap reported
as ``results.att``. When no post period is supplied (design-only mode),
the estimator reports ``att = 0.0``.

The pre-period RMSE between the synthetic treated and synthetic
control trajectories,

.. math::

   \mathrm{RMSE}_{\text{pre}}
   \;=\;
   \sqrt{
     \frac{1}{T} \sum_{t = 1}^T
     \left(
       \sum_{i: y^*_i = +1} w_i \, Y_{i,t}
       -
       \sum_{i: y^*_i = -1} w_i \, Y_{i,t}
     \right)^2
   },

measures the residual of Eq. (1) on the chosen sign vector and is
reported as ``results.rmse_pre``.

Algorithm 3 and Algorithm 4 of the Paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Appendix 3.2 also defines two further numbered algorithms
— *Algorithm 3* (Generalized Power Method on an abstract Hermitian
matrix) and *Algorithm 4* (its normalized counterpart). These are not
implemented as separate code paths in :mod:`mlsynth`: they are the
abstract meta-versions of Algorithms 1 and 2 used to prove the global
convergence theorem (Theorem 3) and operate on a generic matrix
:math:`C = z z^\top + \Delta` rather than on the SPCD-specific
iteration matrix :math:`M`. The two ``variant`` options exposed in the
API already cover both procedures applied to :math:`M`.

Core API
--------

.. automodule:: mlsynth.estimators.spcd
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.SPCDConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.spcd_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.formulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.spectral_init
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.iteration_spcd
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.iteration_norm_spcd
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.weights_empirical
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.weights_exact
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.treatment_effect
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.results_assembly
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   from mlsynth import SPCD

   # SPCD accepts either an SPCDConfig instance or a plain dict.
   config = {
       "df": df,
       "outcome": "sales",
       "unitid": "unitid",
       "time": "time",
       "post_col": "post",        # or T0=...
       "variant": "norm_spcd",    # iteration box; "spcd" also available
       "weights": "empirical",    # paper's experimental default; "exact" uses cvxpy
       "display_graph": True,
   }

   results = SPCD(config).fit()

   # Selected units and design diagnostics
   print(results.selected_unit_labels)
   print(results.design.n_iterations, results.design.converged)
   print(results.design.alpha_ridge, results.design.lam_balance, results.design.beta)

   # Standardized result bundle — same shape as the rest of mlsynth
   print(results.att)                  # mean post-period synthetic gap (= 0 if no post)
   print(results.rmse_pre)             # pre-period RMSE between synthetic groups
   print(results.rmse_post)            # post-period RMSE (None if no post)
   print(results.donor_weights)        # {unit_label: signed contrast weight}

   # Or via the full BaseEstimatorResults object
   summary = results.summary
   summary.effects.att
   summary.fit_diagnostics.rmse_pre
   summary.time_series.observed_outcome      # synthetic treated trajectory
   summary.time_series.counterfactual_outcome  # synthetic control trajectory
   summary.time_series.estimated_gap           # synthetic gap (signed)
   summary.weights.donor_weights
   summary.method_details.method_name          # "SPCD (norm_spcd, weights=empirical)"
   summary.method_details.parameters_used      # variant, alpha/lam/beta, iterations

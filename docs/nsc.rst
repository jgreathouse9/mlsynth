Nonlinear Synthetic Control (NSC)
=================================

.. currentmodule:: mlsynth

Overview
--------

NSC implements Tian (2023), *"The Synthetic Control Method with
Nonlinear Outcomes: Estimating the Impact of the 2019
Anti-Extradition Law Amendments Bill Protests on Hong Kong's
Economy"* (arXiv:2306.01967). NSC generalises the canonical
Abadie-Diamond-Hainmueller (2010) synthetic-control method to
panel-data settings where the untreated potential outcome is a
nonlinear function of the underlying predictors.

Three structural changes versus canonical SC:

1. **Drops the non-negativity restriction** on donor weights — only
   the adding-up constraint :math:`\sum_j w_j = 1` remains. The
   resulting *affine-weight* SC widens the set of treated units the
   method can handle, since it no longer requires the treated unit
   to sit inside the convex hull of the donors.

2. **Adds an elastic-net penalty** on the weights, with the L1 term
   weighted by the **pairwise pretreatment matching discrepancies**
   between the treated unit and each donor. This biases the
   estimator towards near-neighbour matching when the outcome is
   highly nonlinear and towards spread-out weights when it is more
   linear.

3. **Eigenvalue-scales the tuning parameters** so the dimensionless
   :math:`(a^*, b^*) \in [0, 1]` admit a coarse cross-validation
   grid. The paper recommends grid size 0.1 with coordinate-descent
   convergence.

Inference defaults to the **Doudchenko-Imbens (2017)** variance
estimator: for every period the variance of the gap is approximated
by the MSE of predicting each donor's outcome from the other donors
under the same :math:`(a^*, b^*)` regime. Per-period and ATT
normal-based confidence intervals are returned.

Mathematical Formulation
------------------------

Setup
^^^^^

For each unit :math:`i` we observe an outcome :math:`Y_{it}` and a
``(K \times 1)`` vector of pretreatment matching variables
:math:`Z_i = [X_i; Y_{i,1}, \dots, Y_{i,T_0}]'`. Unit 1 receives
treatment at :math:`T_0 + 1` and remains treated thereafter. The
untreated potential outcome follows the interactive fixed-effects
model

.. math::

   Y_{it}^0 = F(X_i' \beta_t + \mu_i' \lambda_t + \varepsilon_{it}),

with :math:`F(\cdot)` a strictly monotonic link function and a
smooth conditional-expectation function :math:`G(\cdot) = E_\varepsilon[F(\cdot)]`.

NSC weight problem
^^^^^^^^^^^^^^^^^^

The weights solve (Tian 2023, eq. 7)

.. math::

   \min_{\{w_j\}} \quad
       \biggl\| Z_1 - \sum_j w_j Z_j \biggr\|^2
       + a \sum_j |w_j| \, \| Z_1 - Z_j \|
       + b \sum_j w_j^{\,2}
   \quad \text{s.t.} \quad \sum_j w_j = 1.

* **Pretreatment fit term** (first norm) -- minimised by ordinary
  affine SC weights.
* **L1 penalty** :math:`a \sum_j |w_j| \, \|Z_1 - Z_j\|` -- pushes
  weight onto donors that are *close* to the treated unit in the
  matching variables. As :math:`a \to \infty` it collapses to the
  one-nearest-neighbour estimator.
* **L2 penalty** :math:`b \sum_j w_j^2` -- spreads the weights. As
  :math:`b \to \infty` the weights become uniform and the
  estimator collapses to a difference-in-differences.

Eigenvalue scaling
^^^^^^^^^^^^^^^^^^

The paper scales the raw multipliers so the dimensionless tuning
parameters live on :math:`[0, 1]`. With :math:`n = \min(J, K)` and
:math:`\lambda_1 \le \dots \le \lambda_n` the sorted non-zero
eigenvalues of :math:`Z_0 Z_0'`,

.. math::

   b = b^* \, \lambda_{\lceil n b^* \rceil},
   \qquad
   a = a^* \, \tilde \lambda_{\lceil n a^* \rceil},

where :math:`\tilde \lambda_*` come from the eigenvalues of
:math:`Z_0 Z_0' + \mathrm{diag}(b)`. At :math:`a^* = 1` only the
nearest neighbour receives weight; at :math:`b^* = 1` the weights
are roughly uniform.

Cross-validation
^^^^^^^^^^^^^^^^

Default: **predict each donor's pretreatment outcome from the
others** under candidate :math:`(a^*, b^*)`, sum the MSPE across
donors, repeat on a grid (default step 0.1) on :math:`[0, 1]`.
Coordinate descent:

1. Initialise :math:`b^* = 0`.
2. Sweep :math:`a^*` over the grid; pick the minimiser.
3. Hold :math:`a^*`; sweep :math:`b^*`; pick the minimiser.
4. Iterate until :math:`(a^*, b^*)` stops moving.

The alternative ``cv_target = "treated"`` predicts the treated
unit's pretreatment outcomes instead — cheaper but uses the same
data for fitting and scoring on the treated side.

Inference
^^^^^^^^^

Doudchenko-Imbens (2017): for each period :math:`t` the variance of
the gap :math:`\hat\tau_t = Y_{1, t} - Y_{1, t}^{\text{SC}}` is
estimated by averaging the squared leave-one-control prediction
residuals at that period. Normal-based CIs follow:

.. math::

   \hat\tau_t \pm z_{1 - \alpha/2} \cdot \hat \sigma_t,
   \quad
   \widehat{\text{ATT}} \pm z_{1 - \alpha/2} \cdot
       \frac{\sqrt{\overline{\hat\sigma_t^{\,2}}}}{\sqrt{n_{\text{post}}}}.

A two-sided z-test of :math:`H_0: \text{ATT} = 0` reports the
p-value.

Core API
--------

.. automodule:: mlsynth.estimators.nsc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.NSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.nsc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.nsc_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.nsc_helpers.crossval
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.nsc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.nsc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.nsc_helpers.structures
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo at the paper's nonlinear DGP
(Tian 2023, Section 4): latent linear factor model, rescale to
:math:`[0, 1]`, square (so :math:`r = 2`), apply a small treatment
effect to the treated unit's post-period.

.. code-block:: python

   """One draw of the Tian (2023) nonlinear-outcome simulation."""

   import numpy as np
   import pandas as pd

   from mlsynth import NSC


   # ---------------------------------------------------------------------
   # 1. Simulate one nonlinear panel
   # ---------------------------------------------------------------------

   rng = np.random.default_rng(0)

   J = 12                    # donors
   T_pre = 12                # pre-treatment periods
   T_post = 6                # post-treatment periods
   T = T_pre + T_post
   tau_true = 0.10           # treatment effect (on the unit interval after rescaling)

   # Latent linear factor model: Y* = X' beta_t + mu' lambda_t + epsilon
   X = rng.uniform(0.0, np.sqrt(12.0), size=(J + 1, 2))
   mu = rng.uniform(0.0, np.sqrt(12.0), size=(J + 1, 4))
   beta_t = rng.normal(10.0, 1.0, size=(T, 2))
   lam_t = rng.normal(10.0, 1.0, size=(T, 4))
   eps = rng.normal(0.0, 1.0, size=(T, J + 1))
   Y_star = (X @ beta_t.T).T + (mu @ lam_t.T).T + eps

   # Rescale to [0, 1] then apply nonlinear transformation (r = 2 in the paper).
   Yn = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
   Y_control = Yn ** 2

   # Apply the additive treatment effect to unit 0 in the post-period.
   Y = Y_control.copy()
   Y[T_pre:, 0] += tau_true

   rows = [
       {
           "unit": j,
           "time": t,
           "y": float(Y[t, j]),
           "D": int(j == 0 and t >= T_pre),
       }
       for j in range(J + 1)
       for t in range(T)
   ]
   df = pd.DataFrame(rows)


   # ---------------------------------------------------------------------
   # 2. Fit NSC with CV-selected (a*, b*)
   # ---------------------------------------------------------------------

   results = NSC({
       "df": df,
       "outcome": "y",
       "treat": "D",
       "unitid": "unit",
       "time": "time",
       "cv_target": "controls",     # paper default
       "cv_grid_size": 0.1,
       "cv_max_iterations": 3,
       "alpha": 0.05,
       "run_inference": True,
       "display_graphs": False,
   }).fit()


   # ---------------------------------------------------------------------
   # 3. Inspect the output
   # ---------------------------------------------------------------------

   print(f"truth   tau = {tau_true:+.3f}")
   print(f"ATT_hat     = {results.att:+.3f}")
   print(f"a* = {results.design.a_star:.2f}, b* = {results.design.b_star:.2f}")
   print(f"pre-RMSE    = {results.pre_rmse:.4f}")
   print(f"95% CI ATT  = [{results.inference.att_lower:+.3f}, "
         f"{results.inference.att_upper:+.3f}]")
   print(f"p-value     = {results.inference.p_value:.3f}")

   # Per-period diagnostics:
   import pandas as pd
   print(pd.DataFrame({
       "t": np.arange(T),
       "gap": results.gap,
       "ci_low": results.inference.gap_lower,
       "ci_high": results.inference.gap_upper,
   }).round(3).to_string(index=False))

References
----------

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies: Estimating the Effect of
California's Tobacco Control Program." *Journal of the American
Statistical Association* 105(490):493-505.

Doudchenko, N., & Imbens, G. W. (2017). "Balancing, Regression,
Difference-In-Differences and Synthetic Control Methods: A
Synthesis." NBER Working Paper 22791.

Tian, W. (2023). "The Synthetic Control Method with Nonlinear
Outcomes: Estimating the Impact of the 2019 Anti-Extradition Law
Amendments Bill Protests on Hong Kong's Economy." arXiv:2306.01967.

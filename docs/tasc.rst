Time-Aware Synthetic Control (TASC)
===================================

.. currentmodule:: mlsynth

Overview
--------

Time-Aware Synthetic Control (TASC)
`arXiv:2601.03099 <https://arxiv.org/abs/2601.03099>`_ is a
state-space synthetic-control estimator. Unlike classical SC
(:doc:`fdid`, :doc:`tssc`) or robust-SC variants
(:doc:`clustersc`, :doc:`proximal`), which treat the ordering of
pre-intervention time indices as interchangeable, TASC explicitly
models the temporal evolution of the latent factors driving the panel.
It embeds the standard SC outcome matrix inside a linear-Gaussian
state-space model with a constant trend matrix :math:`A`, fits the
model parameters via the Expectation-Maximization (EM) algorithm with
a Kalman-filter + Rauch-Tung-Striebel (RTS) smoother E-step and a
closed-form M-step, and produces both a point counterfactual and a
posterior-based confidence band in one pass.

Two structural properties distinguish TASC from the rest of the
``mlsynth`` toolkit:

- **Time-awareness.** Because :math:`A` is shared across periods,
  permuting the pre-intervention time indices changes the fit.
  Permutation-invariant methods (classical SC, robust SC, nuclear-norm
  matrix completion) produce identical counterfactuals under the same
  permutation; TASC does not. Section 5.1 of the paper formalizes this
  via a data-processing-inequality argument (Proposition A.1).
- **Approximately low-rank signal under omnidirectional noise.** The
  observation matrix decomposes as :math:`Y = H X + E` where
  :math:`H X` is exactly rank-:math:`d` and :math:`E` is full-rank
  observation noise. TASC therefore tolerates substantial measurement
  noise — even when PCA-style denoising (used by Robust SC) breaks
  down — because it does not assume the principal directions are
  noise-free.

Mathematical Formulation
------------------------

Let :math:`Y \in \mathbb{R}^{N \times T}` be the outcome matrix with
units in rows and periods in columns. The first row corresponds to the
treated target unit; the remaining :math:`n = N - 1` rows are donors.
Pre-intervention periods are :math:`t = 1, \dots, T_0`; the post-
intervention window is :math:`t = T_0 + 1, \dots, T`, during which the
target row is unobserved (the very quantity TASC reconstructs).

State-Space Model
^^^^^^^^^^^^^^^^^

The TASC generative model (Eqs. (2)-(3) of the paper) is a classical
linear-Gaussian state-space model:

.. math::

   \begin{aligned}
   x_t &= A \, x_{t-1} + q_{t-1},
       & q_{t-1} &\sim \mathcal{N}(0, Q), \\
   y_t &= H \, x_t + r_t,
       & r_t     &\sim \mathcal{N}(0, R),
   \end{aligned}

with initial state :math:`x_0 \sim \mathcal{N}(m_0, P_0)`. The hidden
state :math:`x_t \in \mathbb{R}^d` has dimension :math:`d \ll
\min(n, T)`, which is precisely what preserves the low-rank structure
of the signal :math:`H X`. The complete parameter set is

.. math::

   \theta \;=\; \{A, H, Q, R, m_0, P_0\},
   \quad
   A \in \mathbb{R}^{d \times d},
   \quad
   H \in \mathbb{R}^{N \times d},
   \quad
   Q \in \mathbb{R}^{d \times d},
   \quad
   R \in \mathbb{R}^{N \times N}.

All three covariance matrices :math:`Q, R, P_0` are positive definite.
The :class:`TASCConfig` flags ``diagonal_Q`` and ``diagonal_R`` control
whether the M-step constrains :math:`Q` and :math:`R` to be diagonal
(the paper's default — see Algorithm 7) or updates the full symmetric
covariance.

Relationship to the Linear Factor Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The classical SC linear factor model from Abadie & Gardeazabal (2003),

.. math::

   Y_{i,t} \;=\; \delta_t + \theta_t^\top Z_i + \lambda_t^\top \mu_i + \epsilon_{i,t},

can be cast as a state-space model with latent state
:math:`x_t = (\delta_t, \theta_t, \lambda_t)` and observation rows
:math:`h_i = (1, Z_i, \mu_i)`. The crucial distinction is that linear
factor models impose *no* dynamics on :math:`x_t` (or equivalently
:math:`A = 0`, :math:`x_t = q_t`), whereas TASC enforces a stable
trend through :math:`A`. This is what gives TASC its long-horizon
forecast accuracy under correct specification, at the cost of greater
sensitivity to misspecification when temporal dynamics are complex.

The Counterfactual via Infinite-Variance Kalman Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the post-intervention window the target's observed value is
unavailable. TASC handles this by formally setting the target's
observation-noise variance to :math:`+\infty` (Section 4.2 of the
paper). Partition

.. math::

   y_t = \begin{pmatrix} y_{t,1} \\ y_{t,2} \end{pmatrix},
   \quad
   H   = \begin{pmatrix} h_1^\top \\ H_2 \end{pmatrix},
   \quad
   R'  = \begin{pmatrix} \infty & 0 \\ 0 & R_2 \end{pmatrix},

where :math:`y_{t,2}, r_{t,2} \in \mathbb{R}^n`, :math:`H_2 \in
\mathbb{R}^{n \times d}`, and :math:`R_2 \in \mathbb{R}^{n \times n}`.
Under :math:`R'`, the Schur-complement inverse of the innovation
covariance

.. math::

   (S_k)^{-1}
   \;=\;
   \begin{pmatrix} 0 & 0 \\ 0 & (H_2 P_{k|k-1} H_2^\top + R_2)^{-1} \end{pmatrix}

has a zero in its (1, 1) block. The Kalman gain therefore picks up no
contribution from the target row, and the post-intervention filter
update depends only on the donor block. This is implemented in
:func:`mlsynth.utils.tasc_helpers.filtering.kalman_filter_inf_variance_step`
(Algorithm 5), and the full forward pass is composed by
:func:`mlsynth.utils.tasc_helpers.filtering.kalman_filter_full`
following Algorithm 3.

Once the forward pass produces :math:`(m_k, P_k)_{k=0}^T`, the
backward Rauch-Tung-Striebel smoother
(:func:`mlsynth.utils.tasc_helpers.smoothing.rts_smoother`, Algorithm
6) returns the smoothed posterior

.. math::

   m_k^s, \; P_k^s, \; G_k
   \quad \text{for } k = T, T-1, \dots, 0,

with

.. math::

   \begin{aligned}
   m_{k+1|k} &= A \, m_k, \\
   P_{k+1|k} &= A \, P_k \, A^\top + Q, \\
   G_k       &= P_k \, A^\top \, P_{k+1|k}^{-1}, \\
   m_k^s     &= m_k + G_k \left( m_{k+1}^s - m_{k+1|k} \right), \\
   P_k^s     &= P_k + G_k \left( P_{k+1}^s - P_{k+1|k} \right) G_k^\top.
   \end{aligned}

The counterfactual for the target unit is then read off the smoothed
latent state via :math:`h_1`:

.. math::

   \hat y_{0, t} \;=\; h_1^\top \, m_t^s,
   \qquad t = 1, \dots, T,

and the posterior variance of the *observation* (not just the latent
target) is

.. math::

   \operatorname{Var}(y_{0, t} \mid y_{1:T_0}, y_{2:N, \, T_0+1:T})
   \;=\;
   h_1^\top \, P_t^s \, h_1 \;+\; R_{1, 1}.

The corresponding :math:`(1 - \alpha)`-confidence band is

.. math::

   \hat y_{0, t} \;\pm\; z_{1 - \alpha / 2} \,
   \sqrt{\,h_1^\top P_t^s h_1 + R_{1, 1}\,}.

These are populated unconditionally on :attr:`TASCResults.inference`,
with :math:`\alpha` controlled by the :class:`TASCConfig.alpha` field.

Learning :math:`\theta` from Pre-Intervention Data (EM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameter set :math:`\theta` is learned by Expectation-Maximization
on the pre-intervention slice :math:`Y_{\text{pre}} \in
\mathbb{R}^{N \times T_0}`. Each outer iteration of
:func:`mlsynth.utils.tasc_helpers.em.em_pre` (Algorithm 2) runs:

1. **E-step (filtering pass):** apply the standard Kalman filter
   (Algorithm 4) for :math:`k = 1, \dots, T_0` to obtain
   :math:`(m_k, P_k)`.
2. **E-step (smoothing pass):** apply the RTS smoother backward to
   obtain :math:`(m_k^s, P_k^s, G_k)` for :math:`k = T_0, \dots, 0`.
3. **M-step (closed-form MLE update):** Algorithm 7. Define the
   sufficient statistics

   .. math::

      \begin{aligned}
      \Sigma &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( P_k^s + m_k^s {m_k^s}^\top \right), &
      \Phi   &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( P_{k-1}^s + m_{k-1}^s {m_{k-1}^s}^\top \right), \\
      B      &= \frac{1}{T_0} \sum_{k=1}^{T_0} y_k \, {m_k^s}^\top, &
      C      &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( P_k^s G_{k-1}^\top + m_k^s {m_{k-1}^s}^\top \right), \\
      D      &= \frac{1}{T_0} \sum_{k=1}^{T_0} y_k \, y_k^\top.
      \end{aligned}

   The update is then

   .. math::

      \begin{aligned}
      A'    &\leftarrow C \, \Phi^{-1}, &
      H'    &\leftarrow B \, \Sigma^{-1}, \\
      Q'    &\leftarrow \operatorname{Diag}\!\left(
                          \Sigma - 2 C A'^\top + A' \Phi A'^\top
                       \right), &
      R'    &\leftarrow \operatorname{Diag}\!\left(
                          D - 2 B H'^\top + H' \Sigma H'^\top
                       \right), \\
      m_0'  &\leftarrow m_0^s, &
      P_0'  &\leftarrow P_0^s + (m_0^s - m_0)(m_0^s - m_0)^\top,
      \end{aligned}

   where :math:`\operatorname{Diag}(\cdot)` zeroes the off-diagonal
   entries when ``diagonal_Q=True`` / ``diagonal_R=True`` (the paper's
   default) or returns the symmetric matrix unchanged otherwise.

The loop terminates after :class:`TASCConfig.n_em_iter` outer
iterations or, if :class:`TASCConfig.em_tol` is set, as soon as the
maximum absolute change in :math:`(A, H)` between successive iterations
falls below the threshold.

Spectral Initialization
^^^^^^^^^^^^^^^^^^^^^^^

EM is sensitive to initialization (as noted in the paper's Section 7).
TASC therefore warm-starts :math:`\theta^{(0)}` from a truncated SVD
of the pre-intervention matrix
(:func:`mlsynth.utils.tasc_helpers.setup.initialize_parameters`):

.. math::

   Y_{\text{pre}}
   \;=\;
   U \, \operatorname{diag}(s) \, V^\top,
   \qquad
   H^{(0)} = U_{:, 1:d} \, \operatorname{diag}(s_{1:d}),
   \qquad
   X^{(0)} = V_{:, 1:d}.

The transition matrix :math:`A^{(0)}` is obtained from a ridge-
regularized AR(1) least-squares fit on the latent trajectory
:math:`X^{(0)}`; :math:`Q^{(0)}` and :math:`R^{(0)}` are seeded from
the corresponding residual variances; and :math:`m_0^{(0)}, P_0^{(0)}`
are taken from the first row of :math:`X^{(0)}` and :math:`Q^{(0)}`
respectively.

Treatment Effect and Pre-Period Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For post-treatment periods :math:`t = T_0 + 1, \dots, T`, the average
treatment effect on the treated is the mean of the post-period gap
between the observed target and its TASC reconstruction:

.. math::

   \widehat{\mathrm{ATT}}
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^{T}
     \left( y_{0, t} - \hat y_{0, t} \right),

reported as :attr:`TASCResults.att`. The pre-period RMSE between the
observed target and the smoother's pre-treatment fit,

.. math::

   \mathrm{RMSE}_{\text{pre}}
   \;=\;
   \sqrt{
     \frac{1}{T_0}
     \sum_{t = 1}^{T_0}
       \left( y_{0, t} - \hat y_{0, t} \right)^2
   },

is reported as :attr:`TASCResults.pre_rmse` and serves as the primary
fit diagnostic.

Complexity
^^^^^^^^^^

The dominant cost of TASC is :math:`O(N_1 \, T_0 \, N^3)`, where
:math:`N_1` is the number of EM iterations and the :math:`N^3` term
arises from inverting the innovation covariance during the Kalman
filter. The post-EM full-window pass adds :math:`O(T \, N^3)`, which
is negligible when :math:`T \ll N_1 \, T_0`. Constraining :math:`R`
to be diagonal in the M-step (the default) does not change the
filter's inner-loop complexity but does reduce parameter-count
variance and improves numerical stability in moderate-:math:`N`
regimes.

Algorithm 1 and the Theoretical Appendix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Algorithm 1 is the abstract "SC Family of Methods" frame
(target-side regression on donors), which TASC instantiates implicitly
through the state-space machinery rather than as a discrete code path.
Appendix A's Proposition A.1 (Kalman sufficiency, information loss by
permutation invariance, dominance) is the theoretical justification
for TASC's edge over permutation-invariant SC variants; it does not
correspond to a separate routine in :mod:`mlsynth`.

Core API
--------

.. automodule:: mlsynth.estimators.tasc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.TASCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.tasc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.filtering
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.smoothing
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.mstep
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.em
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.plotter
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.tasc_helpers.structures
   :members:
   :undoc-members:

Example
-------

.. code-block:: python

   from mlsynth import TASC

   # TASC accepts either a TASCConfig instance or a plain dict.
   config = {
       "df": df,
       "outcome": "sales",
       "unitid": "state",
       "time": "year",
       "treat": "treated",        # binary 0/1 treatment indicator
       "d": 2,                    # hidden state dimension (small)
       "n_em_iter": 50,           # N_1 in Algorithm 2
       "em_tol": 1e-4,            # optional early-stopping on max |delta(A, H)|
       "diagonal_Q": True,        # paper default; set False for full covariance
       "diagonal_R": True,
       "alpha": 0.05,             # significance level for posterior CIs
       "display_graphs": True,
   }

   results = TASC(config).fit()

   # Point estimate and fit diagnostic
   print(results.att)              # mean post-period gap, y_{0,t} - h_1' m_t^s
   print(results.pre_rmse)         # pre-period RMSE of the smoother's target fit

   # Counterfactual path and posterior confidence band
   cf = results.inference.counterfactual     # length-T vector
   lo = results.inference.ci_lower           # length-T vector
   hi = results.inference.ci_upper           # length-T vector

   # Learned model and EM diagnostics
   theta = results.design.parameters         # A, H, Q, R, m0, P0
   print(theta.A.shape, theta.H.shape)
   print(results.design.n_em_iter_used)      # number of EM iterations executed
   print(results.design.em_param_deltas)     # per-iteration max |delta(A, H)|

   # Smoothed latent trajectory (useful for downstream diagnostics)
   m_s = results.design.smoothed.m_s         # shape (T + 1, d)
   P_s = results.design.smoothed.P_s         # shape (T + 1, d, d)

   # Inputs preserved on the result object for plotting / re-analysis
   results.inputs.Y_full.shape               # (N, T)
   results.inputs.Y_pre.shape                # (N, T0)
   results.inputs.Y_post_donors.shape        # (N - 1, T - T0)  (None if no post)
   results.inputs.treated_unit_name
   results.inputs.donor_names

References
----------

Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V.
(2026). *Time-Aware Synthetic Control.*
`arXiv:2601.03099 <https://arxiv.org/abs/2601.03099>`_.

Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
Space Methods.* Oxford Statistical Science Series **38**, 2nd edition.
Oxford University Press.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Verification
------------

**Empirical replication against the authors' published numbers (Path A)
plus a Section 5 state-space Monte Carlo (Path B).** Path A reruns the
classical Proposition 99 California-tobacco illustration from Section
6.1 of [TASC]_ using the long-form panel
:file:`basedata/prop99_packsales.csv` shipped with ``mlsynth``, and
reproduces the post-1988 divergence between observed California
cigarette sales and the TASC counterfactual that the paper's Figure 10
displays. Path B replicates the four-cell :math:`(Q, R)` ablation grid
(Figures 3 and 4) by drawing panels directly from TASC's own
generative state-space model and comparing ``mlsynth.TASC`` against a
simplex-constrained Synthetic Control baseline -- the same baseline
the paper benchmarks.

Path A: Proposition 99 California (Section 6.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper runs TASC on per-capita cigarette sales alone (no auxiliary
predictors) with hidden-state dimension :math:`d = 2`. ``mlsynth.TASC``
on the same long-form panel reproduces the qualitative pattern of the
paper's Figure 10 directly:

.. code-block:: python

   import pandas as pd
   from mlsynth import TASC

   df = pd.read_csv("basedata/prop99_packsales.csv")
   df["treat"] = ((df["state"] == "California")
                   & (df["year"] >= 1989)).astype(int)

   res = TASC({"df": df, "outcome": "cigsale", "unitid": "state",
                "time": "year", "treat": "treat", "d": 2,
                "n_em_iter": 50, "em_tol": 1e-4, "alpha": 0.05,
                "seed": 0, "display_graphs": False}).fit()
   yhat = res.inference.counterfactual
   print(f"pre-RMSE = {res.pre_rmse:.3f}  ATT = {res.att:.3f}")

prints::

   pre-RMSE = 0.767  ATT = -16.793

with the year-by-year trajectory

.. list-table::
   :header-rows: 1
   :widths: 10 22 22 18

   * - Year
     - California (observed)
     - TASC counterfactual
     - Gap
   * - 1985
     - 102.80
     - 102.66
     - +0.14
   * - 1988
     - 90.10
     - 91.88
     - -1.78
   * - 1989
     - 82.40
     - 88.30
     - -5.90
   * - 1990
     - 77.80
     - 84.37
     - -6.57
   * - 1995
     - 56.40
     - 76.50
     - -20.10
   * - 2000
     - 41.60
     - 65.14
     - -23.54

The 1985-1988 fit is essentially tight on California's observed
series (pre-RMSE :math:`= 0.77` packs against an outcome scale of
roughly 100 packs), the divergence opens at the 1989 intervention,
and the gap widens monotonically -- reaching a roughly :math:`-24`
pack difference by 2000 against the paper's Figure-10 gap of about
:math:`-25` to :math:`-30` packs at the same horizon. The average
post-1989 treatment effect is :math:`\widehat{\mathrm{ATT}} = -16.8`
packs per year, in the same neighbourhood as Abadie, Diamond and
Hainmueller's classical estimate.

Path B: Section 5 state-space ablation grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Section 5.2 ablation sweeps a :math:`2 \times 2` grid of
state-perturbation and observation-noise covariance scales
:math:`(Q, R)` (Figures 3-4): a "small" covariance has diagonal
variance :math:`0.01` (average :math:`|r_t| \approx 0.084`) and a
"big" covariance has diagonal variance :math:`1.0` (average
:math:`|r_t| \approx 0.836`). Panels are drawn from TASC's own
generative model (Equations 2-3), so this is a *correctly-specified*
Monte Carlo. The DGP is packaged as
:func:`mlsynth.utils.tasc_helpers.simulation.simulate_tasc_sample`;
the panel below compares the post-period RMSE of ``mlsynth.TASC``
(:math:`d_{\mathrm{fit}} = d_{\mathrm{true}} = 5`) against a
simplex-constrained Synthetic Control baseline.

.. code-block:: python

   import numpy as np
   import scipy.optimize as opt
   from mlsynth import TASC
   from mlsynth.utils.tasc_helpers.simulation import simulate_tasc_sample

   def sc_simplex(Y, T0):
       y, X = Y[0], Y[1:].T
       n = X.shape[1]
       cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
       bnds = [(0.0, 1.0)] * n
       r = opt.minimize(lambda w: ((X[:T0] @ w - y[:T0]) ** 2).sum(),
                         np.full(n, 1.0 / n), method="SLSQP",
                         bounds=bnds, constraints=cons)
       return float(np.sqrt(np.mean((y[T0:] - X[T0:] @ r.x) ** 2)))

   def tasc_rmse(sample, d_fit=5):
       r = TASC({"df": sample.df, "outcome": "y", "treat": "treat",
                   "unitid": "unit", "time": "time", "d": d_fit,
                   "n_em_iter": 30, "em_tol": 1e-4, "alpha": 0.05,
                   "seed": 0, "display_graphs": False}).fit()
       y0, T0 = sample.Y[0], sample.T0
       return float(np.sqrt(np.mean(
           (y0[T0:] - r.inference.counterfactual[T0:]) ** 2)))

   M = 30
   for q, r in [(0.01, 0.01), (0.01, 1.0), (0.10, 0.01), (0.10, 1.0)]:
       sc_, tasc_ = [], []
       for seed in range(M):
           s = simulate_tasc_sample(q_scale=q, r_scale=r,
                                      rng=np.random.default_rng(seed))
           sc_.append(sc_simplex(s.Y, s.T0))
           tasc_.append(tasc_rmse(s))
       print(f"q={q:.2f} r={r:.2f}  TASC={np.median(tasc_):.3f}  "
              f"SC={np.median(sc_):.3f}")

prints (at :math:`M = 30`, :math:`N = 38`, :math:`T = 100`,
:math:`T_0 = 50`, :math:`d_{\mathrm{true}} = 5`):

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18

   * - Regime
     - TASC median RMSE
     - SC median RMSE
     - Margin (SC / TASC)
   * - small Q, small R
     - 0.116
     - 0.196
     - 1.7x
   * - small Q, big R
     - 1.103
     - 1.198
     - 1.1x
   * - big Q, small R
     - 0.117
     - 0.524
     - 4.5x
   * - big Q, big R
     - 1.130
     - 1.301
     - 1.2x

TASC carries the lowest median RMSE in all four regimes, and the
margin over SC is largest precisely in the high-:math:`Q` /
low-:math:`R` cell -- the same regime where the paper's Figure 4
identifies TASC's strongest dominance (a fitted state-space model
extracts the persistent low-rank signal that the simplex projection
cannot exploit). Under high observation noise (:math:`R = 1.0`), the
SC simplex projection still trails TASC but by a narrower margin,
reflecting the noise floor common to both estimators. The paper's
Figures 3-4 also include the Robust Synthetic Control of Amjad,
Shah and Shen (2018) and the Causal Impact Model of Brodersen et al.
(2015) as additional comparators that are not in ``mlsynth``; the
ordering above against the canonical simplex-SC baseline is the
slice of those comparisons that ``mlsynth`` can reproduce directly.

The takeaway carried into the published TASC procedure is the
paper's headline finding: when the data-generating process carries a
persistent low-rank temporal signal -- as it does in many policy
panels with strong trends -- explicitly fitting that temporal
structure through a state-space model lowers post-period prediction
error relative to permutation-invariant alternatives, and the
advantage widens as the latent signal strengthens (large :math:`Q`).

References
----------

Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V.
(2026). *Time-Aware Synthetic Control.*
`arXiv:2601.03099 <https://arxiv.org/abs/2601.03099>`_.

Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
Space Methods.* Oxford Statistical Science Series **38**, 2nd edition.
Oxford University Press.

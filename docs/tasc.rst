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
state-space model with a constant trend matrix :math:`\mathbf{A}`, fits
the model parameters via the Expectation-Maximization (EM) algorithm
with a Kalman-filter + Rauch-Tung-Striebel (RTS) smoother E-step and a
closed-form M-step, and produces both a point counterfactual and a
posterior-based confidence band in one pass.

Two structural properties distinguish TASC from the rest of the
``mlsynth`` toolkit:

- Time-awareness. Because :math:`\mathbf{A}` is shared across periods,
  permuting the pre-intervention time indices changes the fit.
  Permutation-invariant methods (classical SC, robust SC, nuclear-norm
  matrix completion) produce identical counterfactuals under the same
  permutation; TASC does not. Section 5.1 of the paper formalizes this
  via a data-processing-inequality argument (Proposition A.1).
- Approximately low-rank signal under omnidirectional noise. The
  observation matrix decomposes as
  :math:`\mathbf{Y} = \mathbf{H}\mathbf{X} + \mathbf{E}` where
  :math:`\mathbf{H}\mathbf{X}` is exactly rank-:math:`d` and
  :math:`\mathbf{E}` is full-rank observation noise. TASC therefore
  tolerates substantial measurement noise — even when PCA-style
  denoising (used by Robust SC) breaks down — because it does not
  assume the principal directions are noise-free.



When to use TASC instead of something else
------------------------------------------

The Rho-Illick-Narasipura-Abadie-Hsu-Misra (2026) paper runs a 4-cell
ablation comparing TASC against vanilla SC, Robust SC, and the
Causal Impact Model under independent variation of
the observation-noise covariance :math:`\mathbf{R}` and the
state-perturbation covariance :math:`\mathbf{Q}` (Section 5.2,
Figures 3-4 of the paper). The clean recommendation:

* Use TASC when observation noise is high. Across the two
  large-:math:`\mathbf{R}` cells (small-:math:`\mathbf{Q}` and
  large-:math:`\mathbf{Q}`) TASC delivers the smallest median RMSE in
  the paper's simulation. PCA-style denoising (Robust SC) and simplex
  shrinkage (vanilla SC) break down because they assume the principal
  directions of the observation matrix are noise-free; TASC's
  full-rank :math:`\mathbf{r}_t \sim \mathcal{N}(0, \mathbf{R})`
  assumption is a much better fit when the noise is omnidirectional.
* Use TASC when the donor panel has a persistent, smoothly-varying
  trend. "Persistent" means the trend extends past the intervention
  point. This is the strong-trend regime (small :math:`\mathbf{Q}`,
  non-trivial :math:`\mathbf{A}`). The Kalman + RTS smoother
  extrapolates the trend forward; PCA / nuclear-norm methods don't.
* Use TASC when you need a posterior credible band for free. TASC
  is a generative model. The RTS smoother returns the full posterior
  covariance at every period, so a ``+/- 1.96 sigma`` band on the
  counterfactual is part of the fit's output. The other mlsynth
  estimators that ship credible bands are :doc:`bvss` (Bayesian
  spike-and-slab) and :doc:`tasc` itself; the rest require an
  external bootstrap or subsampling pass.

When not to reach for TASC:

* The pre-intervention trend is weak or absent (the paper writes
  ":math:`\mathbf{A} \approx 0`" — large :math:`\mathbf{Q}` regime).
  The smaller the trend, the smaller TASC's edge over classical SC; in
  the small-:math:`\mathbf{R}`, large-:math:`\mathbf{Q}` cell of the
  paper's ablation, vanilla SC matches or beats TASC.
* Observation noise is small AND structured low-dimensional.
  Under small :math:`\mathbf{R}`, hard singular-value thresholding
  (Robust SC) cleans the signal exactly, and TASC's
  omnidirectional-:math:`\mathbf{R}` prior is paying a price for
  flexibility it doesn't need.
* Long-horizon forecasting in noisy regimes. The paper's
  Figures 5-6 show that under large :math:`\mathbf{R}` and large
  :math:`\mathbf{Q}`, TASC's RMSE rises noticeably from horizon 51-60
  to horizon 91-100 (small-:math:`\mathbf{Q}` is stable). If you need
  a 5-year-out
  counterfactual on a noisy panel, look at :doc:`fma` or :doc:`mcnnm`
  first.
* Time indices are not really ordered (you're modelling a
  cross-section that happens to be indexed by time, or the periods
  are interchangeable up to relabelling). Permuting time indices
  costs TASC 48.5% on mean RMSE and 25.7% on the RMSE standard
  deviation in the paper's controlled test (Section 5.1, Figure 2).
  If the time ordering is meaningless, use a permutation-invariant
  estimator like :doc:`tssc` or :doc:`clustersc`.

Notation
--------

Let :math:`j = 1` denote the treated unit, with all units
:math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and donor pool
:math:`\mathcal{N}_0 \coloneqq \mathcal{N} \setminus \{1\}` of
cardinality :math:`N_0`. Time runs over
:math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`, 1-indexed; the
intervention takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-period
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of
length :math:`T_0`) and the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.

The treated series is
:math:`\mathbf{y}_1 = (y_{11}, \dots, y_{1T})^\top \in \mathbb{R}^{T}`
with scalar outcomes :math:`y_{1t}`; each donor
:math:`j \in \mathcal{N}_0` contributes a series :math:`\mathbf{y}_j`,
stacked into the donor matrix
:math:`\mathbf{Y}_0 \coloneqq [\mathbf{y}_j]_{j \in \mathcal{N}_0}
\in \mathbb{R}^{T \times N_0}` (one column per donor). The full panel
stacks the treated series above the donors, so the observation vector
at time :math:`t` is :math:`\mathbf{y}_t \in \mathbb{R}^{N}` with
:math:`N = N_0 + 1`.

TASC carries the panel through a latent state
:math:`\mathbf{x}_t \in \mathbb{R}^d` of dimension
:math:`d \ll \min(N_0, T)`, with transition matrix
:math:`\mathbf{A} \in \mathbb{R}^{d \times d}`, observation (loading)
matrix :math:`\mathbf{H} \in \mathbb{R}^{N \times d}` whose row
:math:`\mathbf{h}_1^\top` loads the treated unit, state-perturbation
covariance :math:`\mathbf{Q}`, and observation-noise covariance
:math:`\mathbf{R}`. The synthetic counterfactual for the treated unit
is :math:`\widehat{\mathbf{y}}_1` with entries :math:`\widehat{y}_{1t}`,
the per-period treatment effect is
:math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and the ATT is
:math:`\widehat{\tau} \coloneqq |\mathcal{T}_2|^{-1}
\sum_{t \in \mathcal{T}_2} \tau_t`. The significance level is
:math:`\alpha`.

Assumptions (and how to spot violations)
----------------------------------------

TASC inherits the assumptions of a linear-Gaussian state-space model.
Section 3 of the paper lays them out; the practitioner-facing
restatement is:

(a) Linear-Gaussian dynamics. The hidden state evolves as
    :math:`\mathbf{x}_t = \mathbf{A}\mathbf{x}_{t-1} + \mathbf{q}_t`
    with :math:`\mathbf{q}_t` zero-mean Gaussian. Equivalently: the
    trend in the latent factors is well-approximated by a stable
    linear AR(1) at the level of the state vector, and the
    perturbations around the trend are homoscedastic and uncorrelated
    across time.

    *Remark.* Plausibly violated when the latent factor evolution is
    strongly nonlinear (regime switches, breakpoints, structural
    breaks), has fat-tailed shocks, or has volatility clustering.
    Diagnostic: examine the smoothed state residuals from the
    pre-period fit; non-Gaussian QQ-plot tails or
    Ljung-Box-significant autocorrelation suggest misspecification.

(b) Constant trend matrix :math:`\mathbf{A}`. The dynamics that hold
    over the pre-period are assumed to continue unchanged through the
    post-period. This is the "trend persists past the intervention
    point" assumption that gives TASC its long-horizon advantage.

    *Remark.* Plausibly violated when the intervention itself triggers
    a regime change in the donor units (e.g.\\ a tax change that
    affects neighbouring states' growth dynamics, not just their
    levels). TASC is by construction unable to detect a post-period
    change in :math:`\mathbf{A}` — the post-period target outcomes are
    treated as missing, so they cannot inform the update.
    Diagnostic: split the pre-period into two halves and refit on
    each. If the estimated :math:`\mathbf{A}` differs materially, the
    constant-:math:`\mathbf{A}` assumption is shaky and the
    post-period forecast is suspect.

(c) Observation model
    :math:`\mathbf{y}_t = \mathbf{H}\mathbf{x}_t + \mathbf{r}_t`,
    :math:`\mathbf{R}` full rank, :math:`d \ll \min(N_0, T)`. The
    signal is low-rank with rank :math:`d` (the latent-state
    dimension); the noise :math:`\mathbf{r}_t` is Gaussian with a
    positive-definite covariance. Importantly, :math:`\mathbf{R}` does
    NOT have to be diagonal: TASC handles correlated cross-donor noise
    via the full :math:`\mathbf{R}` (set ``diagonal_R = False`` in the
    config).

    *Remark.* Plausibly violated when the residual cross-section is
    rank-deficient (some donors are exact linear combinations of
    others, e.g.\\ aggregated subseries paired with their
    components), or when the true signal is full-rank (no shared
    factors — every donor moves independently). In both cases the
    EM estimate of :math:`d` ends up wrong and either underfits
    (low :math:`d`) or overfits (high :math:`d`).

(d) No unobserved confounders that affect donors AND treated
    unit between :math:`T_0` and :math:`T_0 + 1`. This is the
    standard SC unconfoundedness assumption, not specific to TASC,
    but worth restating: TASC's counterfactual is informative about
    the treatment effect only if any post-period shock to the donor
    pool is also reflected in what the target would have done absent
    treatment.

    *Remark.* Plausibly violated when a covariate that drives the
    treated unit's outcome (but is uncorrelated with the donors)
    shifts at the intervention time. TASC has no covariate hook, so
    this kind of confounding can only be diagnosed externally.

(e) Hidden-state dimension :math:`d` correctly specified. TASC
    takes :math:`d` as a user hyperparameter
    (``hidden_state_dim``). The paper's Section 5.3 shows that
    underestimating :math:`d` is worse than overestimating — if
    in doubt, err on the high side.

    *Remark.* Plausibly violated when the data has more latent factors
    than you've allowed for. Diagnostic: increase :math:`d` and refit;
    if the RMSE on a held-out pre-period segment drops materially, you
    were underfitting.



Mathematical Formulation
------------------------

The state-space machinery operates on the panel stacked with units in
rows: let :math:`\mathbf{Y} \in \mathbb{R}^{N \times T}` be the
outcome matrix whose first row is the treated unit
:math:`j = 1` and whose remaining :math:`N_0` rows are the donor pool
:math:`\mathcal{N}_0` (the same series collected column-wise in the
canonical donor matrix :math:`\mathbf{Y}_0 \in \mathbb{R}^{T \times N_0}`
of the Notation block). The column :math:`\mathbf{y}_t \in \mathbb{R}^{N}`
is the time-:math:`t` observation vector. Pre-intervention periods are
:math:`t \in \mathcal{T}_1`; over the post-intervention window
:math:`t \in \mathcal{T}_2` the treated row is unobserved (the very
quantity TASC reconstructs).

State-Space Model
^^^^^^^^^^^^^^^^^

The TASC generative model (Eqs. (2)-(3) of the paper) is a classical
linear-Gaussian state-space model:

.. math::

   \begin{aligned}
   \mathbf{x}_t &= \mathbf{A} \, \mathbf{x}_{t-1} + \mathbf{q}_{t-1},
       & \mathbf{q}_{t-1} &\sim \mathcal{N}(\mathbf{0}, \mathbf{Q}), \\
   \mathbf{y}_t &= \mathbf{H} \, \mathbf{x}_t + \mathbf{r}_t,
       & \mathbf{r}_t     &\sim \mathcal{N}(\mathbf{0}, \mathbf{R}),
   \end{aligned}

with initial state
:math:`\mathbf{x}_0 \sim \mathcal{N}(\mathbf{m}_0, \mathbf{P}_0)`. The
hidden state :math:`\mathbf{x}_t \in \mathbb{R}^d` has dimension
:math:`d \ll \min(N_0, T)`, which is precisely what preserves the
low-rank structure of the signal :math:`\mathbf{H}\mathbf{X}`. The
complete parameter set is

.. math::

   \theta \;\coloneqq\; \{\mathbf{A}, \mathbf{H}, \mathbf{Q}, \mathbf{R},
   \mathbf{m}_0, \mathbf{P}_0\},
   \quad
   \mathbf{A} \in \mathbb{R}^{d \times d},
   \quad
   \mathbf{H} \in \mathbb{R}^{N \times d},
   \quad
   \mathbf{Q} \in \mathbb{R}^{d \times d},
   \quad
   \mathbf{R} \in \mathbb{R}^{N \times N}.

All three covariance matrices :math:`\mathbf{Q}, \mathbf{R}, \mathbf{P}_0`
are positive definite. The :class:`TASCConfig` flags ``diagonal_Q`` and
``diagonal_R`` control whether the M-step constrains :math:`\mathbf{Q}`
and :math:`\mathbf{R}` to be diagonal (the paper's default — see
Algorithm 7) or updates the full symmetric covariance.

Relationship to the Linear Factor Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The classical SC linear factor model from Abadie & Gardeazabal (2003),

.. math::

   y_{jt} \;=\; \delta_t + \boldsymbol{\theta}_t^\top \mathbf{Z}_j
   + \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_j + \epsilon_{jt},

can be cast as a state-space model with latent state
:math:`\mathbf{x}_t = (\delta_t, \boldsymbol{\theta}_t, \boldsymbol{\lambda}_t)`
and observation rows
:math:`\mathbf{h}_j = (1, \mathbf{Z}_j, \boldsymbol{\mu}_j)`. The crucial
distinction is that linear factor models impose *no* dynamics on
:math:`\mathbf{x}_t` (or equivalently :math:`\mathbf{A} = \mathbf{0}`,
:math:`\mathbf{x}_t = \mathbf{q}_t`), whereas TASC enforces a stable
trend through :math:`\mathbf{A}`. This is what gives TASC its long-horizon
forecast accuracy under correct specification, at the cost of greater
sensitivity to misspecification when temporal dynamics are complex.

The Counterfactual via Infinite-Variance Kalman Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the post-intervention window the target's observed value is
unavailable. TASC handles this by formally setting the target's
observation-noise variance to :math:`+\infty` (Section 4.2 of the
paper). Partition

.. math::

   \mathbf{y}_t = \begin{pmatrix} y_{1t} \\ \mathbf{y}_{0t} \end{pmatrix},
   \quad
   \mathbf{H}   = \begin{pmatrix} \mathbf{h}_1^\top \\ \mathbf{H}_0 \end{pmatrix},
   \quad
   \mathbf{R}'  = \begin{pmatrix} \infty & \mathbf{0} \\ \mathbf{0} & \mathbf{R}_0 \end{pmatrix},

where :math:`\mathbf{y}_{0t}, \mathbf{r}_{0t} \in \mathbb{R}^{N_0}`,
:math:`\mathbf{H}_0 \in \mathbb{R}^{N_0 \times d}`, and
:math:`\mathbf{R}_0 \in \mathbb{R}^{N_0 \times N_0}`. Under
:math:`\mathbf{R}'`, the Schur-complement inverse of the innovation
covariance

.. math::

   (\mathbf{S}_k)^{-1}
   \;=\;
   \begin{pmatrix} 0 & \mathbf{0} \\ \mathbf{0} &
   (\mathbf{H}_0 \mathbf{P}_{k|k-1} \mathbf{H}_0^\top + \mathbf{R}_0)^{-1} \end{pmatrix}

has a zero in its (1, 1) block. The Kalman gain therefore picks up no
contribution from the target row, and the post-intervention filter
update depends only on the donor block. This is implemented in
:func:`mlsynth.utils.tasc_helpers.filtering.kalman_filter_inf_variance_step`
(Algorithm 5), and the full forward pass is composed by
:func:`mlsynth.utils.tasc_helpers.filtering.kalman_filter_full`
following Algorithm 3.

Once the forward pass produces
:math:`(\mathbf{m}_k, \mathbf{P}_k)_{k=0}^T`, the
backward Rauch-Tung-Striebel smoother
(:func:`mlsynth.utils.tasc_helpers.smoothing.rts_smoother`, Algorithm
6) returns the smoothed posterior

.. math::

   \mathbf{m}_k^s, \; \mathbf{P}_k^s, \; \mathbf{G}_k
   \quad \text{for } k = T, T-1, \dots, 0,

with

.. math::

   \begin{aligned}
   \mathbf{m}_{k+1|k} &= \mathbf{A} \, \mathbf{m}_k, \\
   \mathbf{P}_{k+1|k} &= \mathbf{A} \, \mathbf{P}_k \, \mathbf{A}^\top + \mathbf{Q}, \\
   \mathbf{G}_k       &= \mathbf{P}_k \, \mathbf{A}^\top \, \mathbf{P}_{k+1|k}^{-1}, \\
   \mathbf{m}_k^s     &= \mathbf{m}_k + \mathbf{G}_k \left( \mathbf{m}_{k+1}^s - \mathbf{m}_{k+1|k} \right), \\
   \mathbf{P}_k^s     &= \mathbf{P}_k + \mathbf{G}_k \left( \mathbf{P}_{k+1}^s - \mathbf{P}_{k+1|k} \right) \mathbf{G}_k^\top.
   \end{aligned}

The counterfactual for the target unit is then read off the smoothed
latent state via :math:`\mathbf{h}_1`:

.. math::

   \widehat{y}_{1t} \;=\; \mathbf{h}_1^\top \, \mathbf{m}_t^s,
   \qquad t = 1, \dots, T,

and the posterior variance of the *observation* (not just the latent
target) is

.. math::

   \operatorname{Var}\!\left(y_{1t} \mid \mathbf{y}_{1, \mathcal{T}_1},
   \mathbf{Y}_{0, \mathcal{T}_2}\right)
   \;=\;
   \mathbf{h}_1^\top \, \mathbf{P}_t^s \, \mathbf{h}_1 \;+\; \mathbf{R}_{1, 1}.

The corresponding :math:`(1 - \alpha)`-confidence band is

.. math::

   \widehat{y}_{1t} \;\pm\; z_{1 - \alpha / 2} \,
   \sqrt{\,\mathbf{h}_1^\top \mathbf{P}_t^s \mathbf{h}_1 + \mathbf{R}_{1, 1}\,}.

These are populated unconditionally on :attr:`TASCResults.inference`,
with :math:`\alpha` controlled by the :class:`TASCConfig.alpha` field.

Learning :math:`\theta` from Pre-Intervention Data (EM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameter set :math:`\theta` is learned by Expectation-Maximization
on the pre-intervention slice :math:`\mathbf{Y}_{\text{pre}} \in
\mathbb{R}^{N \times T_0}`. Each outer iteration of
:func:`mlsynth.utils.tasc_helpers.em.em_pre` (Algorithm 2) runs:

1. E-step (filtering pass): apply the standard Kalman filter
   (Algorithm 4) for :math:`k = 1, \dots, T_0` to obtain
   :math:`(\mathbf{m}_k, \mathbf{P}_k)`.
2. E-step (smoothing pass): apply the RTS smoother backward to
   obtain :math:`(\mathbf{m}_k^s, \mathbf{P}_k^s, \mathbf{G}_k)` for
   :math:`k = T_0, \dots, 0`.
3. M-step (closed-form MLE update): Algorithm 7. Define the
   sufficient statistics

   .. math::

      \begin{aligned}
      \boldsymbol{\Sigma} &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( \mathbf{P}_k^s + \mathbf{m}_k^s {\mathbf{m}_k^s}^\top \right), &
      \boldsymbol{\Phi}   &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( \mathbf{P}_{k-1}^s + \mathbf{m}_{k-1}^s {\mathbf{m}_{k-1}^s}^\top \right), \\
      \mathbf{B}      &= \frac{1}{T_0} \sum_{k=1}^{T_0} \mathbf{y}_k \, {\mathbf{m}_k^s}^\top, &
      \mathbf{C}      &= \frac{1}{T_0} \sum_{k=1}^{T_0}
                \left( \mathbf{P}_k^s \mathbf{G}_{k-1}^\top + \mathbf{m}_k^s {\mathbf{m}_{k-1}^s}^\top \right), \\
      \mathbf{D}      &= \frac{1}{T_0} \sum_{k=1}^{T_0} \mathbf{y}_k \, \mathbf{y}_k^\top.
      \end{aligned}

   The update is then

   .. math::

      \begin{aligned}
      \mathbf{A}'    &\leftarrow \mathbf{C} \, \boldsymbol{\Phi}^{-1}, &
      \mathbf{H}'    &\leftarrow \mathbf{B} \, \boldsymbol{\Sigma}^{-1}, \\
      \mathbf{Q}'    &\leftarrow \operatorname{Diag}\!\left(
                          \boldsymbol{\Sigma} - 2 \mathbf{C} \mathbf{A}'^\top + \mathbf{A}' \boldsymbol{\Phi} \mathbf{A}'^\top
                       \right), &
      \mathbf{R}'    &\leftarrow \operatorname{Diag}\!\left(
                          \mathbf{D} - 2 \mathbf{B} \mathbf{H}'^\top + \mathbf{H}' \boldsymbol{\Sigma} \mathbf{H}'^\top
                       \right), \\
      \mathbf{m}_0'  &\leftarrow \mathbf{m}_0^s, &
      \mathbf{P}_0'  &\leftarrow \mathbf{P}_0^s + (\mathbf{m}_0^s - \mathbf{m}_0)(\mathbf{m}_0^s - \mathbf{m}_0)^\top,
      \end{aligned}

   where :math:`\operatorname{Diag}(\cdot)` zeroes the off-diagonal
   entries when ``diagonal_Q=True`` / ``diagonal_R=True`` (the paper's
   default) or returns the symmetric matrix unchanged otherwise.

The loop terminates after :class:`TASCConfig.n_em_iter` outer
iterations or, if :class:`TASCConfig.em_tol` is set, as soon as the
maximum absolute change in :math:`(\mathbf{A}, \mathbf{H})` between
successive iterations falls below the threshold.

Spectral Initialization
^^^^^^^^^^^^^^^^^^^^^^^

EM is sensitive to initialization (as noted in the paper's Section 7).
TASC therefore warm-starts :math:`\theta^{(0)}` from a truncated SVD
of the pre-intervention matrix
(:func:`mlsynth.utils.tasc_helpers.setup.initialize_parameters`):

.. math::

   \mathbf{Y}_{\text{pre}}
   \;=\;
   \mathbf{U} \, \operatorname{diag}(\mathbf{s}) \, \mathbf{V}^\top,
   \qquad
   \mathbf{H}^{(0)} = \mathbf{U}_{:, 1:d} \, \operatorname{diag}(\mathbf{s}_{1:d}),
   \qquad
   \mathbf{X}^{(0)} = \mathbf{V}_{:, 1:d}.

The transition matrix :math:`\mathbf{A}^{(0)}` is obtained from a ridge-
regularized AR(1) least-squares fit on the latent trajectory
:math:`\mathbf{X}^{(0)}`; :math:`\mathbf{Q}^{(0)}` and
:math:`\mathbf{R}^{(0)}` are seeded from the corresponding residual
variances; and :math:`\mathbf{m}_0^{(0)}, \mathbf{P}_0^{(0)}` are taken
from the first row of :math:`\mathbf{X}^{(0)}` and
:math:`\mathbf{Q}^{(0)}` respectively.

Treatment Effect and Pre-Period Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For post-treatment periods :math:`t \in \mathcal{T}_2`, the per-period
treatment effect is the gap between the observed target and its TASC
reconstruction, :math:`\tau_t \coloneqq y_{1t} - \widehat{y}_{1t}`, and
the average treatment effect on the treated is its post-period mean:

.. math::

   \widehat{\tau}
   \;\coloneqq\;
   |\mathcal{T}_2|^{-1}
   \sum_{t \in \mathcal{T}_2}
     \tau_t
   \;=\;
   \frac{1}{T - T_0}
   \sum_{t = T_0 + 1}^{T}
     \left( y_{1t} - \widehat{y}_{1t} \right),

reported as :attr:`TASCResults.att`. The pre-period RMSE between the
observed target and the smoother's pre-treatment fit,

.. math::

   \mathrm{RMSE}_{\text{pre}}
   \;\coloneqq\;
   \sqrt{
     \frac{1}{T_0}
     \sum_{t = 1}^{T_0}
       \left( y_{1t} - \widehat{y}_{1t} \right)^2
   },

is reported as :attr:`TASCResults.pre_rmse` and serves as the primary
fit diagnostic.

Complexity
^^^^^^^^^^

The dominant cost of TASC is :math:`O(N_1 \, T_0 \, N^3)`, where
:math:`N_1` is the number of EM iterations and the :math:`N^3` term
arises from inverting the innovation covariance during the Kalman
filter. The post-EM full-window pass adds :math:`O(T \, N^3)`, which
is negligible when :math:`T \ll N_1 \, T_0`. Constraining :math:`\mathbf{R}`
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

   # Counterfactual path and posterior confidence band (raw TASC inference).
   # ``results.counterfactual`` is the same series via the standardized accessor.
   cf = results.inference_detail.counterfactual     # length-T vector
   lo = results.inference_detail.ci_lower           # length-T vector
   hi = results.inference_detail.ci_upper           # length-T vector

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

Empirical replication against the authors' published numbers (Path A)
plus a Section 5 state-space Monte Carlo (Path B). Path A reruns the
classical Proposition 99 California-tobacco illustration from Section
6.1 of [TASC]_ using the long-form panel
:file:`basedata/prop99_packsales.csv` shipped with ``mlsynth``, and
reproduces the post-1988 divergence between observed California
cigarette sales and the TASC counterfactual that the paper's Figure 10
displays. Path B replicates the four-cell :math:`(\mathbf{Q}, \mathbf{R})` ablation grid
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
   yhat = res.inference_detail.counterfactual
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
post-1989 treatment effect is :math:`\widehat{\tau} = -16.8`
packs per year, in the same neighbourhood as Abadie, Diamond and
Hainmueller's classical estimate.

Path B: Section 5 state-space ablation grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Section 5.2 ablation sweeps a :math:`2 \times 2` grid of
state-perturbation and observation-noise covariance scales
:math:`(\mathbf{Q}, \mathbf{R})` (Figures 3-4): a "small" covariance has diagonal
variance :math:`0.01` (average :math:`|\mathbf{r}_t| \approx 0.084`) and a
"big" covariance has diagonal variance :math:`1.0` (average
:math:`|\mathbf{r}_t| \approx 0.836`). Panels are drawn from TASC's own
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
margin over SC is largest precisely in the high-:math:`\mathbf{Q}` /
low-:math:`\mathbf{R}` cell -- the same regime where the paper's Figure 4
identifies TASC's strongest dominance (a fitted state-space model
extracts the persistent low-rank signal that the simplex projection
cannot exploit). Under high observation noise (:math:`\mathbf{R} = 1.0`), the
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
advantage widens as the latent signal strengthens (large :math:`\mathbf{Q}`).

References
----------

Rho, S., Illick, C., Narasipura, S., Abadie, A., Hsu, D., & Misra, V.
(2026). *Time-Aware Synthetic Control.*
`arXiv:2601.03099 <https://arxiv.org/abs/2601.03099>`_.

Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
Space Methods.* Oxford Statistical Science Series 38, 2nd edition.
Oxford University Press.

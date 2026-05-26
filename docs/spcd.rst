Synthetic Principal Component Design (SPCD)
===========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

Most synthetic-control work takes the treated unit as *given* and asks only
how to weight the donors. SPCD, due to Lu, Li, Ying and Blanchet (2022)
[SPCD]_ `arXiv:2211.15241 <https://arxiv.org/abs/2211.15241>`_, answers the
**prior, design-time question**: you are about to run an experiment, you
have a panel of pre-treatment outcomes, and you must decide *which units to
treat* (and how to weight the rest into a synthetic comparison) so that the
treatment-effect estimate is as precise as possible. It is a sibling of
:doc:`syndes` — both choose the treated set rather than assume it — but where
SYNDES solves a mixed-integer program, SPCD reformulates the design as a
**phase-synchronization** problem and solves it with a spectrally-initialized
power method that converges *globally* under standard linear-factor models,
in seconds rather than minutes.

Reach for SPCD when treatment can only be applied to **coarse, expensive
units**, randomizing at random leaves accuracy on the table, and you want a
fast, provably-good assignment:

- **Marketing / geo experiments.** You can switch a campaign on in some
  **media markets (DMAs)**, **regions**, or **store clusters** but not at the
  individual-customer level. SPCD reads each market's pre-period **sales** or
  **traffic** history and picks the treated markets — and the synthetic
  comparison — so the two groups tracked each other *before* launch. Any
  post-launch divergence is then a low-variance read on the campaign.
- **Retail / product rollouts.** Choosing which **stores** or **SKUs** get a
  new layout, price, or feature first. Treating a handful of stores is costly,
  so you want the few you pick to be maximally informative about the whole
  chain.
- **Platform / policy pilots.** Selecting **cities** or **submarkets** for a
  pricing pilot, a new service tier, or a regulatory change, where
  interference rules out customer-level randomization and the number of
  treatable units is small.

The flip side: if assignment is *already fixed* (you know who was treated and
only need a counterfactual), this is the wrong tool — use a standard
retrospective estimator such as :doc:`fscm`, :doc:`clustersc`, or
:doc:`tssc`. SPCD's value is entirely in the **design** stage, before the
experiment runs.

Notation
--------

We observe an outcome :math:`Y_{it}` for units :math:`i \in \{1, \dots, N\}`
over pre-treatment periods :math:`t \in \{1, \dots, T\}`, stacked into the
pre-treatment matrix :math:`Y \in \mathbb{R}^{N \times T}` (units in rows,
periods in columns). At :math:`t = T` the experimenter assigns each unit a
**sign** :math:`D_i \in \{-1, +1\}`: :math:`D_i = +1` puts unit :math:`i` in
the treated group, :math:`D_i = -1` in the control group. Each unit also
carries a non-negative **weight** :math:`w_i \ge 0`, normalized to sum to one
on each side. After assignment, outcomes are observed for :math:`S` further
periods :math:`t = T+1, \dots, T+S`, with potential outcomes
:math:`Y_{it}(-1) = \mu_{it} + e_{it}` and :math:`Y_{it}(+1) = Y_{it}(-1) +
\tau`, where :math:`\mu_{it}` is the base (signal) outcome, :math:`e_{it}` is
mean-zero idiosyncratic noise with variance :math:`\sigma`, and :math:`\tau`
is the treatment effect to be estimated. The estimand is the weighted average
treatment effect on the treated (wATET).

Throughout, :math:`\mathbf{1}` is the all-ones vector, :math:`I` the
identity, :math:`\operatorname{sgn}(\cdot)` the elementwise sign,
:math:`\|\cdot\|_1` / :math:`\|\cdot\|_2` the L1 / L2 norms, and
:math:`(\cdot)^{-1}` the matrix inverse.

.. note::

   **Notation bridge.** The single-treated-unit synthetic-control canon
   (treated :math:`j=0`, donors :math:`1, \dots, N`, treatment dummy
   :math:`d_{jt} \in \{0,1\}`) does not fit a *design* problem in which the
   assignment is itself the decision variable. Following the paper, the
   assignment is the signed vector :math:`D \in \{-1,+1\}^N` and the two
   groups are symmetric (neither is privileged as "the donors"). The
   implementation stores the pre-treatment matrix transposed, as
   :math:`Y_{\text{pre}} \in \mathbb{R}^{T \times N}` (time in rows, units in
   columns), so the iteration matrix below is built in code as
   ``Y_pre.T @ Y_pre + alpha I + lambda 1 1.T``.

How SPCD Works (Plain-Language Walkthrough)
-------------------------------------------

Before the equations, here is the whole idea in five steps.

1. **The goal is a fair pre-period match.** Split the units into a treated
   group and a control group, and weight each unit, so that *before* the
   experiment the weighted treated history and the weighted control history
   are nearly identical curves. If the two groups move together in the
   pre-period, then after treatment the gap between them is the treatment
   effect — and a well-balanced design makes that gap a low-variance estimate.

2. **Trying every split is hopeless.** With :math:`N` units there are
   exponentially many ways to form two groups; choosing the best one is
   NP-hard. SPCD sidesteps the brute-force search with a change of variable.

3. **One signed number per unit.** Pack each unit's *group label* and *weight*
   into a single signed number :math:`W_i = w_i D_i`: its **sign** says which
   group the unit is in, its **magnitude** is the weight. "Make the two
   weighted groups match" then becomes "find a signed vector :math:`W` that is
   as small as possible against the data," i.e. minimize :math:`W^\top (Y
   Y^\top) W` subject to the groups balancing out. This is the
   :math:`\ell_1`-PCA / **phase-synchronization** problem — a well-studied
   shape with fast, globally-convergent solvers.

4. **Solve it with a power method, warm-started by an eigenvector.** Start
   from a smart first guess — the sign pattern of the smallest eigenvector of
   the data matrix (the *spectral initialization*). Then repeatedly refine the
   group labels: multiply the current sign vector by :math:`M^{-1}` and take
   signs again (the *generalized power method*). Keep going until the labels
   stop flipping. The **normalized** variant (the package default) rescales by
   the diagonal of :math:`M^{-1}` first and is guaranteed to converge to the
   global optimum at a linear rate.

5. **Read off the design, then estimate.** Once the labels are fixed, the
   weights follow in closed form (Eq. (9)) or from a tiny convex program
   (Eq. (6)). Treat the *smaller* of the two groups (the minority rule). Run
   the experiment. When the post-period data arrives, the treatment effect is
   simply the gap between the two weighted groups, averaged over the
   post-period.

A useful analogy: it is like drafting two evenly-matched teams from a pool of
players based on their past stats — except SPCD also decides how much each
player counts, and it finds the balanced split with a fast eigenvector
computation instead of trying every possible roster.

Mathematical Formulation
------------------------

SPCD jointly chooses the signs :math:`\{D_i\}_{i=1}^N` and non-negative
weights :math:`\{w_i\}_{i=1}^N` so that the weighted pre-treatment
trajectories of the two groups *track each other as closely as possible*.

Following Doudchenko et al. (2021) [SYNDES]_ and Abadie & Zhao (2021)
[ABADIE2024]_, the expected MSE of the difference-in-weighted-means
treatment-effect estimator decomposes as

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

The first term is a **bias** from imperfect pre-treatment balance; the second
a **variance** that grows with weight concentration. Minimizing the first term
over the *pre-treatment window* gives the mixed-integer program SPCD solves
(Eq. (1) of the paper):

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

Reformulation as :math:`\ell_1`-PCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key observation of the paper is that the change of variable
:math:`W_i = w_i D_i` collapses the assignment and weight variables into a
single signed vector :math:`W \in \mathbb{R}^N`. Under :math:`W`,
:math:`D_i = \operatorname{sgn}(W_i)` and :math:`w_i = |W_i|`, the adding-up
constraints become :math:`\mathbf{1}^\top W = 0`, and the design problem
becomes

.. math::

   \min_{W \in \mathbb{R}^N,\ \mathbf{1}^\top W = 0,\ \|W\|_1 = 1}
       W^\top \left( Y Y^\top + \sigma I \right) W.

The hard equality :math:`\mathbf{1}^\top W = 0` is absorbed into the
objective via a quadratic penalty:

.. math::

   \min_{W \in \mathbb{R}^N,\ \|W\|_1 = 1}
       W^\top \left( Y Y^\top + \sigma I + \lambda \mathbf{1}\mathbf{1}^\top \right) W.

Theorem 1 of the paper shows that for :math:`\lambda` large enough, the *sign
vector* :math:`\operatorname{sgn}(W^*)` of any global minimum coincides with
the sign vector of an associated phase-synchronization problem. Once the signs
are recovered, the magnitudes follow from a small convex QP (or, in practice,
the closed form in Eq. (9)).

The Iteration Matrix
^^^^^^^^^^^^^^^^^^^^

All four code-paths of SPCD operate on the same :math:`N \times N` iteration
matrix from Eq. (2) of the paper:

.. math::

   M \;=\; Y Y^\top \;+\; \alpha I \;+\; \lambda \mathbf{1} \mathbf{1}^\top,

where :math:`\alpha` plays the role of :math:`\sigma` (noise variance) and
:math:`\lambda` is the constraint-absorbing penalty. The paper treats both as
*pre-defined* hyperparameters and gives no formula for them.

.. note::

   **Choosing** :math:`\alpha`. The paper's appendix sets the perturbation
   ridge on the **noise** scale (:math:`\alpha = \lVert\Delta\rVert` with
   :math:`\alpha \le \lVert\Delta\rVert`), so :math:`\alpha` should track the
   idiosyncratic noise variance, *not* the dominant (signal/level) eigenvalue
   of :math:`Y Y^\top`. When :math:`N > T_{\text{pre}}` the matrix
   :math:`Y Y^\top` is rank-deficient and the post-period RMSE is a
   non-monotone, *jumpy* function of :math:`\alpha` (small changes flip the
   discrete sign vector), so no single closed-form estimate is robust. When
   the user does not supply :math:`\alpha`, ``formulation.py`` first fixes its
   *scale* with the Gavish-Donoho median-singular-value noise estimate
   (:func:`~mlsynth.utils.spcd_helpers.formulation.estimate_noise_variance`),
   and ``orchestration.select_alpha_by_holdout`` then picks the value on a
   noise-scale grid that **balances a held-out tail of the pre-period best**
   out of sample. :math:`\lambda` and :math:`\beta` still default from the
   spectrum of :math:`Y Y^\top`. If you know the noise level (e.g. the
   simulations below fix :math:`\sigma = 1`), pass ``alpha_ridge`` explicitly
   to skip selection.

Spectral Initialization
^^^^^^^^^^^^^^^^^^^^^^^

Both Algorithm 1 and Algorithm 2 of the paper start from the same warm start:
the sign of the smallest eigenvector of :math:`M`,

.. math::

   y^{0} \;=\; \operatorname{sgn}(v),
   \qquad
   v \;=\; \arg\min_{\|v\|_2 = 1} v^\top M v.

Appendix 3.2.1 of the paper (Lemma 4) shows that under the linear
latent-factor model :math:`Y_{it} = \delta_t + \theta_t^\top \mu_i + e_{it}`
(Assumption 1) together with the realizability assumption (Assumption 2), the
sign vector :math:`\operatorname{sgn}(v)` already agrees with the global
optimum :math:`\operatorname{sgn}(W^*)` up to a bounded fraction of entries.

The (Normalized) Generalized Power Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To refine the spectral initialization, SPCD performs sign-iterations on
:math:`M^{-1}`. The two ``variant`` choices correspond to the two update boxes
of Algorithm 1 / Algorithm 2:

- ``variant="spcd"`` (Eq. (4)/(7)) — *Generalized Power Iteration*:

  .. math::

     y^{t+1} \;=\;
     \operatorname{sgn}\!\left[\,
         \left( M^{-1} + \beta I \right) \, y^{t}
     \,\right].

- ``variant="norm_spcd"`` (Eq. (5)/(8)) — *Normalized Generalized Power
  Iteration*:

  .. math::

     y^{t+1} \;=\;
     \operatorname{sgn}\!\left[\,
         \left( M^{-1} + \beta I \right) \,
         \big( y^{t} \,/\, d \big)
     \,\right],
     \quad
     d \;=\; \sqrt{\operatorname{diag}(M^{-1})}.

Here :math:`\beta > 0` is the step parameter (auto-defaulted to
:math:`1/\lambda_{\max}(M)`), and :math:`/` denotes element-wise division. The
loop terminates as soon as the sign vector stops changing between successive
iterations.

Under Assumptions 1 and 2 with small enough idiosyncratic noise, Theorem 3 of
the paper guarantees:

- ``variant="spcd"`` converges globally if :math:`\epsilon \,>\, (\sqrt{3}/2) -
  1`;
- ``variant="norm_spcd"`` converges *linearly* and globally for any
  :math:`\epsilon > 0`.

This is why the normalized variant is the package default.

Final Weight Step
^^^^^^^^^^^^^^^^^

Once the iteration converges to a sign vector :math:`y^* \in \{-1, +1\}^N`,
the unit weights are produced by one of two procedures:

- ``weights="empirical"`` (Eq. (9), Algorithm 2 — the paper's experimental
  default):

  .. math::

     w \;=\;
     \frac{2 \, M^{-1} y^*}{\left\| M^{-1} y^* \right\|_1}.

  The optimality condition of the original QP (Eq. (6)) implies that
  :math:`\operatorname{sgn}(w) = y^*` whenever the iteration has converged to a
  fixed point of the closed-form map, so the signed vector :math:`w`
  simultaneously encodes group membership *and* weights.

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

  via ``cvxpy``. Use this when you need the exact Algorithm-1 weights rather
  than the closed-form approximation.

Minority Convention
^^^^^^^^^^^^^^^^^^^

Per the bottom of Algorithm 1 (page 7 of the paper), SPCD then applies the
*minority-group rule*

.. math::

   \text{Treat unit } i \iff y^*_i \;=\; -\operatorname{sgn}\!\left(\sum_{j} y^*_j\right),

which flips the sign vector (if necessary) so that the smaller of the two
groups is the treated group. The treated unit labels reported by
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

which is precisely the mean of the post-period synthetic gap reported as
``results.att``. When no post period is supplied (design-only mode), the
estimator reports ``att = 0.0``.

The pre-period RMSE between the synthetic treated and synthetic control
trajectories,

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

measures the residual of Eq. (1) on the chosen sign vector and is reported as
``results.rmse_pre``.

Algorithm 3 and Algorithm 4 of the Paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The paper's Appendix 3.2 also defines two further numbered algorithms —
*Algorithm 3* (Generalized Power Method on an abstract Hermitian matrix) and
*Algorithm 4* (its normalized counterpart). These are not implemented as
separate code paths in :mod:`mlsynth`: they are the abstract meta-versions of
Algorithms 1 and 2 used to prove the global convergence theorem (Theorem 3)
and operate on a generic matrix :math:`C = z z^\top + \Delta` rather than on
the SPCD-specific iteration matrix :math:`M`. The two ``variant`` options
exposed in the API already cover both procedures applied to :math:`M`.

Assumptions and Theory
----------------------

SPCD's global-optimality guarantee rests on two assumptions — one structural,
one a realizability condition — both stated formally below, each paired with a
plain-language Remark.

**Assumption 1 (linear latent-factor model).** Outcomes are generated by

.. math::

   Y_{jt} \;=\; \delta_t \;+\; \frac{D_{jt} + 1}{2}\,\tau \;+\; \theta_t^\top \mu_j \;+\; e_{jt},
   \qquad
   \mathbb{E}[e_{jt} \mid \delta_t, \mu_j, D_{jt}] = 0,
   \quad
   \operatorname{Var}[e_{jt} \mid \cdot] = \sigma,

where :math:`\delta_t` is a time fixed effect, :math:`\mu_j` are unobserved
common factors, :math:`\theta_t` is a vector of unknown factor loadings,
:math:`e_{jt}` is i.i.d. idiosyncratic noise, and :math:`\tau` is the
treatment effect. In the pre-treatment period :math:`D_{jt} = -1` for all
units.

*Remark.* This is the standard interactive-fixed-effects model that underlies
the consistency theory of synthetic control (Abadie et al., 2010 [ABADIE2010]_;
Xu, 2017 [Xu2017]_). It is the assumption that makes "units that shared latent
factors in the past will keep tracking each other" a defensible basis for the
design — exactly the structure SPCD exploits when it balances the two groups
on pre-period outcomes.

**Assumption 2 (realizable design).** There exists a *unique* parameter
:math:`(w_i, D_i)_{i=1}^N` satisfying: (a) :math:`D_i \in \{-1, +1\}`; (b)
:math:`w_i \ge 0` with :math:`\sum_i D_i w_i = 0`; (c) :math:`\|w\|_2^2 = N`
and :math:`\epsilon \le |w_i| \le 1/\epsilon` for all :math:`i`; and (d) the
weights balance the factors, :math:`\sum_i w_i D_i \mu_i = 0`.

*Remark.* This says a *perfect, balanced* design exists in population — a split
whose weighted factor loadings cancel exactly — and that it is unique (so the
optimizer need not arbitrate between competing perfect designs). It is the
design-time analogue of the "treated unit lies in the convex hull of the
donors" condition in retrospective SCM, and is what turns an NP-hard search
into a problem with a recoverable global optimum.

**Theorem 1 (sign recovery).** For :math:`\lambda` large enough, the global
solution :math:`W^*` of the penalized program satisfies
:math:`\operatorname{sgn}(W^*) = \operatorname{sgn}(\arg\min_{\|W\|_1=1} W^\top
(YY^\top + \sigma I + \lambda \mathbf{1}\mathbf{1}^\top) W)`. *In words:* the
quadratic penalty does not corrupt the *signs* of the optimal design, so
recovering group membership and recovering the weights can be separated.

**Theorem 2 (equivalence to phase synchronization).** The design problem is
symbolically identical to a phase-synchronization problem on the matrix
:math:`(A^\top A)^{-1}`. *In words:* SPCD inherits the entire fast-solver
toolkit developed for phase synchronization — most importantly the
spectrally-initialized generalized power method.

**Theorem 3 (global convergence).** Under Assumptions 1–2, if :math:`\sigma`
is small enough and :math:`T \ge \mathrm{poly}(N, 1/\epsilon)`, then
``variant="spcd"`` converges to the global optimum whenever :math:`\epsilon >
\sqrt{3}/2 - 1`, and ``variant="norm_spcd"`` converges to the global optimum at
a **linear rate** for any :math:`\epsilon > 0`. *In words:* the normalization
step buys global convergence under a strictly weaker condition, which is why
``norm_spcd`` is the default.

Inference and Power Analysis
----------------------------

Beyond the design itself, :class:`SPCD` produces a moving-block conformal
confidence interval for the post-period ATT and a Monte Carlo estimate of the
minimum detectable effect (MDE). The procedure mirrors LEXSCM's train-on-E /
calibrate-on-B discipline.

Estimation / Holdout Split
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the pretreatment matrix :math:`Y_{\text{pre}} \in
\mathbb{R}^{T_{\text{pre}} \times N}`, SPCD splits the pretreatment window into

.. math::

   Y_{\text{pre}}
   \;=\;
   \begin{bmatrix}
       Y_{E} \\[3pt]
       Y_{B}
   \end{bmatrix},

where :math:`Y_E` contains the first ``holdout_frac_E`` of the pretreatment
periods (default 70 %) and :math:`Y_B` contains the remainder. The SPCD design
— sign vector, treated/control weights, and the iteration matrix :math:`M` —
is fit on :math:`Y_E` *only*.

Backwards-Compatibility Guarantee
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the design is fit on :math:`Y_E` alone, two callers who share the same
pretreatment data — one in *planning mode* (no :math:`Y_{\text{post}}` yet) and
one in *retrospective mode* (with :math:`Y_{\text{post}}`) — receive
**identical** designs. The only difference between the two callers is whether a
post-period ATT and its conformal CI are reported.

Holdout Residuals
^^^^^^^^^^^^^^^^^

Applying the design weights to :math:`Y_B` gives an out-of-sample synthetic-gap
series

.. math::

   r_B \;=\; Y_B \cdot w,

where :math:`w` is the signed ``contrast_weights`` vector from the
:math:`Y_E`-fit design. Under the linear-factor model with no treatment,
:math:`r_B` is a zero-mean noise series whose empirical distribution
characterizes the noise structure of any synthetic-gap linear functional based
on the same :math:`w`. This makes :math:`r_B` the natural calibration set for
inference and the natural noise pool for power simulations.

Moving-Block Conformal CI
^^^^^^^^^^^^^^^^^^^^^^^^^

When :math:`Y_{\text{post}}` is supplied, the post-period synthetic gap is
:math:`g = Y_{\text{post}} \cdot w \in \mathbb{R}^S`, with mean

.. math::

   \hat\tau \;=\; \frac{1}{S} \sum_{t=1}^S g_t.

The test statistic is :math:`T(g) = \mathrm{mean}(|g|)`. Conformity scores are
computed by taking mean-absolute values over all sliding blocks of size
:math:`b = \max(3, \lfloor\sqrt{S}\rfloor)` (both standard and circular) of
:math:`r_B`. The conformal p-value vs. :math:`H_0\colon \tau = 0` is

.. math::

   p \;=\; \frac{
       \#\{\text{blocks with score} \geq T(g)\}
   }{\text{number of blocks}}.

The :math:`(1 - \alpha)` confidence interval for :math:`\hat\tau` is obtained
by inversion: scan a grid of candidate values :math:`\theta` around
:math:`\hat\tau`, include :math:`\theta` if the adjusted-residual score
:math:`T(g - \theta)` is in the in-distribution region of the block-score
empirical distribution. Pointwise bands at the :math:`(1 - \alpha)` quantile
:math:`q` of the block scores are reported as :math:`g_t \pm q`.

This is exchangeability-based inference: coverage holds in finite samples under
the assumption that overlapping blocks of :math:`r_B` are exchangeable with
overlapping blocks of :math:`g` under the null. This is a stronger assumption
than IID noise and weaker than perfect :math:`H_0`. It breaks if the treatment
introduces variance changes.

Monte Carlo MDE
^^^^^^^^^^^^^^^

The MDE answers the pre-experiment planning question: *given my holdout
residuals and a planned post-period horizon* :math:`S`, *what is the smallest
constant treatment effect I could detect with power* :math:`\geq \pi`?

The procedure follows :mod:`mlsynth.utils.fast_scm_helpers.power_helpers`:

1. Build the null distribution of :math:`T` at horizon :math:`S` by resampling
   :math:`r_B` (padded with Gaussian draws so that resampling of size :math:`S`
   is always feasible). Compute :math:`c_\alpha =
   \text{quantile}(\text{null}, 1 - \alpha)`.
2. For each candidate :math:`\tau` on a grid of effect sizes, draw
   :math:`n_{\text{trials}}` post-period vectors of the form :math:`\tau +
   \mathcal{N}(0, \hat\sigma_B^2)` and count the fraction exceeding
   :math:`c_\alpha`.
3. The smallest :math:`\tau` whose empirical power reaches the target
   :math:`\pi` is reported as the MDE, both on the absolute scale (``mde``) and
   as a percentage of the holdout-baseline outcome level (``mde_pct``).

A detectability curve — the MDE as a function of :math:`S` — can be requested
by passing ``mde_horizon_grid`` to the config. This answers the related
question: *how long do I need to run the experiment to detect a target effect?*

Power Analysis Always Runs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The MDE computation depends only on :math:`r_B`, not on
:math:`Y_{\text{post}}`. So power analysis runs whenever
``enable_inference=True`` and the holdout window is large enough (at least
``min_blank_size`` periods, default 5). The conformal CI only runs when
:math:`Y_{\text{post}}` is supplied; otherwise the ATT and CI are reported as
``None`` (honest absence rather than a silent zero).

Opting Out
^^^^^^^^^^

Setting ``enable_inference=False`` skips the E/B split entirely. The design is
then fit on the full pretreatment matrix (legacy behavior), no holdout
residuals are produced, and inference / power analysis are not run. Use this
when you need byte-identical reproducibility against a pre-inference SPCD
release, or when your pretreatment window is too short to spare a holdout.

Example: Reproducing the Paper's RMSE Advantage
-----------------------------------------------

The paper's headline finding (Section 4.1, Table 1, Figures 2–4) is not merely
that SPCD is unbiased — it is that **SPCD's design yields a far lower
root-mean-square error of the treatment-effect estimate than a random design**
on the linear latent-factor model. The block below reproduces that finding and
is self-contained: it draws panels from the paper's data-generating process
(:math:`Y_{it} = \text{level}_i + v_t^\top \gamma_i + e_{it}`, with a true
effect :math:`\tau = 1` and noise :math:`\sigma = 1`), and for each draw
estimates :math:`\tau` two ways — with SPCD, and with the paper's random
baseline (assign each unit to treated/control with probability one half, then
take a simple difference in group means). Because the SPCD design is fit on
pre-treatment data only, injecting a post-period effect never changes *which*
units SPCD selects, so the two SPCD calls per draw share an identical design.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import SPCD

   def factor_panel(rng, N=10, T_pre=20, T_post=10, L=8, sigma=1.0):
       """One draw from the paper's linear factor model (Section 4.1):
       Y_it = level_i + v_t^T gamma_i + e_it, with no treatment applied yet."""
       gamma = rng.standard_normal((N, L))                 # unit loadings
       v = rng.standard_normal((T_pre + T_post, L))        # time factors
       level = rng.uniform(40.0, 60.0, size=N)             # positive baselines
       noise = rng.normal(scale=sigma, size=(T_pre + T_post, N))
       return level[None, :] + v @ gamma.T + noise         # shape (T, N)

   def make_df(Y, T_pre):
       T, N = Y.shape
       return pd.DataFrame(
           [{"unit": i, "time": t, "y": float(Y[t, i]), "post": int(t >= T_pre)}
            for i in range(N) for t in range(T)]
       )

   def spcd_att(Y, T_pre, tau):
       """Fit the SPCD design on the pre-period, inject tau on the chosen
       treated units, then read the estimated ATT off the post-period."""
       cfg = dict(outcome="y", unitid="unit", time="time", post_col="post",
                  enable_inference=False, alpha_ridge=1.0)  # sigma^2 known = 1
       treated = SPCD({"df": make_df(Y, T_pre), **cfg}).fit().selected_unit_labels
       Y = Y.copy(); Y[T_pre:, np.asarray(treated)] += tau
       return SPCD({"df": make_df(Y, T_pre), **cfg}).fit().att

   def random_att(rng, Y, T_pre, tau):
       """Paper's baseline: assign each unit to treated/control with prob 1/2
       and estimate tau by a difference in (equally weighted) group means."""
       N = Y.shape[1]
       sign = rng.choice([-1, 1], size=N)
       if sign.min() == sign.max():                        # keep both groups non-empty
           sign[rng.integers(N)] *= -1
       treated, control = np.where(sign == 1)[0], np.where(sign == -1)[0]
       Y = Y.copy(); Y[T_pre:, treated] += tau
       gap = Y[T_pre:, treated].mean(axis=1) - Y[T_pre:, control].mean(axis=1)
       return gap.mean()

   rng = np.random.default_rng(0)
   T_pre, T_post, tau, n_draws = 20, 10, 1.0, 200

   # --- one draw: SPCD recovers the injected effect ---
   Y = factor_panel(rng, T_pre=T_pre, T_post=T_post)
   print(f"single-draw SPCD ATT: {spcd_att(Y, T_pre, tau):.3f}  (true {tau})")

   # --- many draws: SPCD's design slashes ATT RMSE vs a random design ---
   err_spcd, err_rand = [], []
   for _ in range(n_draws):
       Y = factor_panel(rng, T_pre=T_pre, T_post=T_post)
       err_spcd.append(spcd_att(Y, T_pre, tau) - tau)
       err_rand.append(random_att(rng, Y, T_pre, tau) - tau)
   err_spcd, err_rand = np.array(err_spcd), np.array(err_rand)

   rmse = lambda e: np.sqrt(np.mean(e ** 2))
   print(f"SPCD   mean ATT {tau + err_spcd.mean():.3f}   RMSE {rmse(err_spcd):.3f}")
   print(f"random mean ATT {tau + err_rand.mean():.3f}   RMSE {rmse(err_rand):.3f}")
   print(f"RMSE ratio (random / SPCD): {rmse(err_rand) / rmse(err_spcd):.1f}x")

Both designs are roughly **unbiased** (mean ATT :math:`\approx 1`), but SPCD's
RMSE is about **0.4** against the random design's **3.9** — an **order-of-
magnitude reduction** (≈ 9–10× on this seed), reproducing the paper's central
claim that SPCD "surpasses the random design by a large margin." The mechanism
is the balance term of Eq. (1): a single draw of the random design can split
the units so that the two group means differ substantially even before
treatment, and that pre-period imbalance carries straight through to the ATT;
SPCD instead chooses the split (and weights) that makes the groups track each
other, so almost nothing but the injected effect survives into the post-period
gap.

Switch ``variant`` (``"spcd"`` vs. ``"norm_spcd"``) and ``weights``
(``"empirical"`` vs. ``"exact"``) to compare the four code-paths; all share the
same iteration matrix :math:`M` and differ only in the refinement and weight
steps described above. Enabling inference (the default) additionally returns a
conformal p-value, CI, and the design-time MDE on each fit.

.. note::

   ``selected_unit_labels`` returns the *minority* group (the treated side) by
   the minority-group convention. Treating the smaller group keeps the control
   pool large, which is what the variance term :math:`\sigma^2 \sum_i w_i^2`
   rewards.

API Variants and Multi-Arm Designs
----------------------------------

``SPCD`` accepts either an :class:`SPCDConfig` instance or a plain dict. The
two orthogonal modelling choices are:

- ``variant``: which power-method update box to iterate. ``"spcd"`` is the
  generalized power iteration (Eq. (4)/(7)); ``"norm_spcd"`` is the
  *normalized* iteration (Eq. (5)/(8)) used in all of the paper's Section 4
  experiments and is the package default.
- ``weights``: how to compute the final unit weights. ``"empirical"`` uses the
  closed-form Eq. (9) (fast; the authors' experimental default); ``"exact"``
  solves the convex QP Eq. (6) via ``cvxpy`` (slower; matches Algorithm 1 to
  the letter).

Passing an ``arm`` column solves the SPCD design **independently within each
arm's units** and returns an :class:`SPCDMultiArmResults` (a dict of per-arm
:class:`SPCDResults`); when ``arm`` is ``None`` (default) a single
:class:`SPCDResults` is returned.

Multi-Arm Power: the Pooled Average Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each arm carries its own per-arm MDE (``arm_designs[label].mde``), but those
are *m* independent analyses with no family-wise error control. When the
estimand of interest is the **average effect across arms** — the usual
program-level question — :class:`SPCDMultiArmResults` also reports a single
**pooled average-effect MDE** (``results.pooled_mde`` /
``results.pooled_mde_pct``), computed by default whenever inference is enabled
and at least two arms have a usable holdout window.

It is formed by pooling the arms' **time-aligned** holdout residuals into one
contrast, :math:`g_t = \sum_a w_a\, r^{(a)}_{B,t}`, and running the ordinary
single-series MDE on :math:`g`. Because the arms share calendar time, summing
the *aligned* series makes the cross-arm correlation enter through
:math:`\operatorname{Var}(g) = w^\top \Sigma\, w` automatically — which is why
the residual **series** must be pooled, never resampled per arm independently
(that would drop positive correlation and report an over-optimistic MDE). The
weights :math:`w_a` are set by ``pooled_weights``: ``"size"`` (default) weights
each arm by its unit count, giving the *population-average* effect;
``"equal"`` weights arms equally.

Pooling answers an easier question than the per-arm analyses — averaging
cancels idiosyncratic noise, so the detectable *average* effect is typically
several times smaller than any single arm's MDE.

.. warning::

   The pooled MDE targets the **weighted-average** effect. If arm effects are
   heterogeneous and **opposite-signed** they can cancel in the average and go
   undetected even when individual arms move a lot. Use the per-arm
   ``arm_designs[label].mde`` (or a family-wise procedure) when detecting
   *individual* arms is what matters.

.. code-block:: python

   res = SPCD({
       "df": df, "outcome": "sales", "unitid": "unit", "time": "time",
       "arm": "region", "post_col": "post",
       "pooled_weights": "size",    # population-average effect ("equal" also available)
   }).fit()

   print(res.pooled_mde)            # smallest detectable AVERAGE effect across arms
   print(res.pooled_mde_pct)        # ... as % of the pooled baseline
   print({k: r.mde for k, r in res.arm_designs.items()})   # per-arm MDEs

How Long to Run the Study (Detectability Curve)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passing ``mde_horizon_grid`` evaluates the MDE at each candidate
post-treatment horizon, so you can read off **how many post periods are
needed** to detect a target effect. The curve is produced both per arm
(``arm_designs[label].power.detectability``) and for the **whole study**
(``pooled_power.detectability``), each a ``{horizon -> MDE percent}`` map.
The MDE shrinks as the horizon grows (more post periods → more power), and
the pooled/whole-study curve sits below every per-arm curve.

.. code-block:: python

   res = SPCD({
       "df": df, "outcome": "sales", "unitid": "unit", "time": "time",
       "arm": "region", "post_col": "post",
       "mde_horizon_grid": list(range(2, 13)),   # evaluate horizons 2..12
   }).fit()

   whole_study = res.pooled_power.detectability    # {horizon: MDE % of baseline}
   per_arm = {k: r.power.detectability for k, r in res.arm_designs.items()}

   # Recommend the shortest horizon that detects a target average effect:
   target_pct = 1.0
   feasible = [h for h, m in sorted(whole_study.items()) if m <= target_pct]
   recommended_periods = min(feasible) if feasible else None

.. code-block:: python

   from mlsynth import SPCD

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
   print(results.donor_weights)        # {control unit label: weight}

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

Result Containers
-----------------

``SPCD.fit()`` returns an
:class:`~mlsynth.utils.spcd_helpers.structures.SPCDResults`, bundling the
optimized :class:`~mlsynth.utils.spcd_helpers.structures.SPCDDesign`
(sign vector, treated/control/contrast weights, selected unit labels,
iteration diagnostics, and the auto-chosen :math:`\alpha`, :math:`\lambda`,
:math:`\beta`), the prepared
:class:`~mlsynth.utils.spcd_helpers.structures.SPCDInputs`, a standardized
``summary`` (:class:`~mlsynth.config_models.BaseEstimatorResults`), and the
optional conformal CI and power-analysis objects. When an ``arm`` column is
configured, an :class:`~mlsynth.utils.spcd_helpers.structures.SPCDMultiArmResults`
collects one independent :class:`SPCDResults` per arm.

.. automodule:: mlsynth.utils.spcd_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

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

.. automodule:: mlsynth.utils.spcd_helpers.holdout
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.spcd_helpers.power
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

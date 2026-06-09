Relaxed / Penalized Synthetic Control (RESCM)
=============================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The classic synthetic control method (SCM) of Abadie, Diamond and Hainmueller
[ABADIE2010]_ builds a counterfactual as a **convex** combination of donor
units -- weights on the simplex, no intercept. That convex hull restriction is
what makes SCM interpretable and robust, but it is also brittle when the donor
pool is large relative to the pre-period: with :math:`N` donors and only
:math:`T_1` pre-periods, the least-squares fit underlying SCM is
under-determined once :math:`N \gtrsim T_1`, the weights become unstable, and
many "equally good" solutions exist.

``RESCM`` is a **single convex program** that nests a whole family of SCM
estimators as corner cases, so you can dial the donor-pool regularization
continuously from classic SCM all the way to difference-in-differences (equal
weights), and pick the corner that suits your data. Two branches are exposed,
each with the estimation/inference theory of its own paper:

* **Penalized branch** -- :math:`\min \tfrac{1}{2}\|y_0 - \mu - Y\omega\|_2^2 +
  P(\omega)` with :math:`P` an :math:`\ell_1` / :math:`\ell_2` /
  :math:`\ell_\infty` (or mixed) penalty. The :math:`\ell_\infty` member is the
  **L-infinity-norm SCM** of Wang, Xing and Ye [LinfSC]_, which *spreads*
  weight across donors (capping the largest weight) rather than concentrating
  it; classic Abadie SCM is the no-penalty (:math:`\lambda = 0`) simplex corner,
  and equal-weights/DiD is the heavy-:math:`\ell_\infty` limit
  [DoudchenkoImbens2017]_.
* **Relaxation branch** -- the **SCM-relaxation** of Liao, Shi and Zheng
  [RelaxSC]_: keep the simplex :math:`\omega \in \Delta_J`, but *relax* the
  exact balance first-order condition to an :math:`\ell_\infty` tolerance
  :math:`\eta`, then among all weights satisfying the relaxed condition pick the
  one minimizing an information-theoretic **divergence** :math:`D(\omega)`
  (squared :math:`\ell_2`, entropy, or empirical likelihood). The divergence
  picks a unique, stable weight (e.g. closest-to-uniform), and under a latent
  group structure recovers equal-within-group weights.

Use ``RESCM`` when you have a **single treated unit and a large donor pool**
and want either (i) a dense, stable counterfactual that does not hinge on a
handful of donors (``LINF`` / ``RELAX_*``), or (ii) a one-stop interface to
compare classic SCM against its penalized and relaxed cousins on the same
panel. Pick estimators by name through ``methods``.

.. admonition:: Reference implementations (authors' code)

   The source papers' own code — useful for cross-checking and otherwise
   hard to locate:

   * **L-infinity-norm SCM** (Wang, Xing & Ye [LinfSC]_, backing ``LINF`` /
     ``L1LINF``): https://github.com/BioAlgs/LinfinitySC
   * **SCM-relaxation** (Liao, Shi & Zheng [RelaxSC]_, backing ``RELAX_L2`` /
     ``RELAX_ENTROPY`` / ``RELAX_EL``): the ``scmrelax`` Python package at
     https://github.com/metricshilab/scmrelax (installable from
     https://github.com/PanJi-0/scmrelax), with the Brexit / UK real-GDP
     application at https://github.com/YapengZheng/Relaxed_SC

Notation
--------

We use the synthetic-control canon. Unit :math:`j=0` is treated and
:math:`\mathcal{N} = \{1, \ldots, N\}` indexes the donors; :math:`\mathbf{y}_0`
is the treated outcome and :math:`\mathbf{Y} = (y_{jt})` the
:math:`T \times N` donor matrix. The intervention occurs after the pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}`; the post-period is
:math:`\mathcal{T}_2 = \{T_1+1, \ldots, T\}` with :math:`T_2 = |\mathcal{T}_2|`.
Donor weights :math:`\boldsymbol{\omega} = (\omega_1, \ldots, \omega_N)'` live
on the simplex :math:`\Delta_N = \{\boldsymbol{\omega} : \omega_j \ge 0,\,
\sum_j \omega_j = 1\}` (relaxation branch) or are penalized (penalized branch);
:math:`\mu` is an optional intercept. The counterfactual is
:math:`\hat{y}_{0t}^0 = \mu + \mathbf{Y}_{t\cdot}\,\boldsymbol{\omega}`, the
effect :math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0`, and the ATE
:math:`\bar{\Delta} = T_2^{-1}\sum_{t\in\mathcal{T}_2}\hat{\Delta}_t`.
:math:`\|\mathbf{A}\|_\infty = \max_{ij}|a_{ij}|`. With pre-period donor Gram
matrix :math:`\hat{\boldsymbol{\Sigma}} = T_1^{-1}\sum_{t\in\mathcal{T}_1}
\mathbf{Y}_{t\cdot}'\mathbf{Y}_{t\cdot}` and cross-moment
:math:`\hat{\boldsymbol{\Upsilon}} = T_1^{-1}\sum_{t\in\mathcal{T}_1}
\mathbf{Y}_{t\cdot}' y_{0t}`, the SCM least-squares first-order condition is
:math:`\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega} = \hat{\boldsymbol{\Upsilon}}`.

The unified convex program
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``RESCM`` corner case is a special case of one program. The
**penalized** form fits the in-sample loss with a regularizer,

.. math::

   \min_{\mu,\,\boldsymbol{\omega}\in\mathcal{C}} \;
     \tfrac{1}{2}\sum_{t\in\mathcal{T}_1}
       \bigl(y_{0t} - \mu - \mathbf{Y}_{t\cdot}\boldsymbol{\omega}\bigr)^2
     + \lambda\Bigl[\alpha\|\boldsymbol{\omega}\|_1
       + (1-\alpha)\,Q(\boldsymbol{\omega})\Bigr],
   \qquad Q \in \{\tfrac{1}{2}\|\cdot\|_2^2,\; \|\cdot\|_\infty\},

with constraint set :math:`\mathcal{C}` (simplex by default), mixing
:math:`\alpha\in[0,1]`, and strength :math:`\lambda` chosen by K-fold
cross-validation. The **relaxation** form instead minimizes a divergence
subject to a relaxed balance condition,

.. math::

   \min_{\boldsymbol{\omega}\in\Delta_N} \; D(\boldsymbol{\omega})
   \quad\text{s.t.}\quad
   \bigl\|\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega}
     - \hat{\boldsymbol{\Upsilon}}\bigr\|_\infty \le \eta,
   \qquad
   D \in \Bigl\{\tfrac{1}{2}\|\boldsymbol{\omega}\|_2^2,\;
     \textstyle\sum_j \omega_j\log\omega_j,\;
     -\textstyle\sum_j \log\omega_j\Bigr\},

where the three divergences are squared :math:`\ell_2`, **entropy**, and
**empirical likelihood** respectively, and :math:`\eta` is selected by
validation. Classic SCM is recovered at :math:`\lambda=0` (penalized, simplex)
or :math:`\eta = 0` (relaxation); the equal-weights/DiD estimator is the
:math:`\lambda\to\infty` :math:`\ell_\infty` limit or large-:math:`\eta` limit.

Penalized branch: L-infinity SCM (Wang, Xing & Ye)
--------------------------------------------------

**Idea.** Classic SCM tends to load heavily on one or two donors. When the
true data-generating process spreads signal across many donors, that
concentration is fragile -- a single idiosyncratic donor shock contaminates the
counterfactual. The :math:`\ell_\infty` penalty
:math:`\|\boldsymbol{\omega}\|_\infty = \max_j |\omega_j|` directly **caps the
largest weight**, so minimizing it (under the simplex) pushes the solution
toward *spreading* weight across donors -- in the limit, equal weights. The
mixed :math:`\ell_1 + \ell_\infty` penalty (``L1LINF``) trades sparsity against
spreading. Both shrink toward a stable, dense counterfactual that tolerates
:math:`N > T_1`.

**Assumptions** (Wang, Xing & Ye [LinfSC]_).

*Assumption 1 (factor model).* The outcomes follow a linear factor model
:math:`y_{jt} = \boldsymbol{\lambda}_j'\mathbf{f}_t + u_{jt}` with the treated
unit's loading in (or near) the convex hull of the donor loadings, so an
:math:`\ell_\infty`-regularized convex combination approximates the treated
factor structure.

*Assumption 2 (weak dependence).* The idiosyncratic errors :math:`u_{jt}` are
mean-zero and weakly dependent over time (strong-mixing with summable
autocovariances), permitting a HAC long-run variance and a sequential CLT for
the post-period mean.

*Assumption 3 (regularization rate).* The penalty strength :math:`\lambda`
(and mixing :math:`\alpha`) are chosen so the weight estimate is consistent for
its population target; in practice both are selected by K-fold cross-validation
on the pre-period.

*Remark.* The :math:`\ell_\infty` cap is what buys stability in high
dimensions: by refusing to let any single weight dominate, the estimator's
prediction variance stays controlled even when :math:`N \gg T_1`, where the
unconstrained least-squares (and concentrated classic SCM) solutions degrade.
This is the synthetic-control analogue of the L2-relaxation argument for dense
coefficients.

**Inference.** Wang, Xing and Ye extend the synthetic-control ATE inference of
Li [LiSCM2020]_ to the dense, weakly dependent setting. With fixed weights the
post-period gap :math:`\hat{\Delta}_t` has mean :math:`\bar{\Delta}`, and

.. math::

   \hat{Z} = \frac{\bar{\Delta}}
       {\sqrt{\hat{\rho}^2_{(1)}/T_1 + \hat{\rho}^2_{(2)}/T_2}}
   \xrightarrow{d} N(0,1),

with :math:`\hat{\rho}^2_{(1)}` the HAC long-run variance of the pre-period
prediction residuals (first-stage weight-estimation uncertainty) and
:math:`\hat{\rho}^2_{(2)}` that of the de-meaned post-period effects. This is
the two-term variance ``RESCM`` uses for **all** corner cases (see *Inference*
below).

**When to use.** Dense, factor-driven donor structure; high dimension
(:math:`N>T_1` permitted); when you want a counterfactual robust to any single
donor's idiosyncrasies rather than a sparse, concentrated fit.

Relaxation branch: SCM-relaxation (Liao, Shi & Zheng)
-----------------------------------------------------

**Idea.** Classic SCM solves a least-squares problem whose first-order
condition is :math:`\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega} =
\hat{\boldsymbol{\Upsilon}}` -- exact balance of the donor moments. When
:math:`N` is large this condition is over-strict: it is satisfied (or nearly
so) by a continuum of weights, and the chosen one is arbitrary and unstable.
SCM-relaxation keeps the simplex but **relaxes** the balance condition to an
:math:`\ell_\infty` tolerance :math:`\eta`, then selects, among all admissible
weights, the one minimizing a divergence :math:`D(\boldsymbol{\omega})`:

.. math::

   \hat{\boldsymbol{\omega}} = \operatorname*{argmin}_{\boldsymbol{\omega}\in\Delta_N}
     D(\boldsymbol{\omega})
   \quad\text{s.t.}\quad
   \bigl\|\hat{\boldsymbol{\Sigma}}\boldsymbol{\omega}
     - \hat{\boldsymbol{\Upsilon}}\bigr\|_\infty \le \eta.

The divergence breaks the tie deterministically: :math:`\ell_2`
(``RELAX_L2``) picks the minimum-norm weight, **entropy** (``RELAX_ENTROPY``)
the maximum-entropy / closest-to-uniform weight, and **empirical likelihood**
(``RELAX_EL``) the EL-optimal weight. :math:`\eta=0` reduces to (a divergence-
selected) classic SCM; large :math:`\eta` admits the whole simplex and the
divergence alone determines the weight (entropy/EL :math:`\to` equal weights).

**Assumptions** (Liao, Shi & Zheng [RelaxSC]_).

*Assumption 1 (approximate factor model with group structure).* Units load on
common factors and fall into latent groups; the treated unit's loading is
spanned by the donor loadings up to an approximation error that the relaxation
tolerance :math:`\eta` absorbs.

*Assumption 2 (identifiable divergence minimizer).* The divergence
:math:`D` is strictly convex on the simplex, so the relaxed feasible set has a
unique minimizer -- this is what restores well-posedness when the exact balance
condition does not.

*Assumption 3 (weak dependence and moments).* The errors are weakly dependent
with bounded moments, so sample moments :math:`\hat{\boldsymbol{\Sigma}},
\hat{\boldsymbol{\Upsilon}}` converge and a post-period CLT applies.

*Remark.* The key result (oracle prediction) is that the relaxed weight
predicts the treated counterfactual as well as the infeasible oracle that knows
the factor structure, and under the latent group structure the divergence-
selected weight is **equal within groups** -- a transparent, interpretable
solution that the arbitrary classic-SCM tie-break does not deliver.

**Inference.** Liao, Shi and Zheng's main theory concerns *prediction*
consistency (the oracle property). For an ATE confidence interval ``mlsynth``
applies the same weak-dependence two-term HAC test as the penalized branch
(Li [LiSCM2020]_), treating the relaxed weights as the fixed first stage. See
the caveat under *Verification*.

The named corner cases
----------------------

``methods`` selects estimators by name; each resolves to one exact call of the
convex engine.

.. list-table::
   :header-rows: 1
   :widths: 16 12 50

   * - Name
     - Branch
     - Estimator
   * - ``SC``
     - penalized
     - Classic Abadie simplex SCM (:math:`\lambda=0`).
   * - ``LASSO``
     - penalized
     - :math:`\ell_1` penalty; sparse donor weights.
   * - ``RIDGE``
     - penalized
     - :math:`\ell_2` penalty; dense shrunken weights.
   * - ``ENET``
     - penalized
     - Elastic net (:math:`\ell_1 + \ell_2`); :math:`\alpha` by CV.
   * - ``LINF``
     - penalized
     - L-infinity-norm SCM [LinfSC]_; spreads weight, nests DiD.
   * - ``L1LINF``
     - penalized
     - Mixed :math:`\ell_1 + \ell_\infty` penalty.
   * - ``RELAX_L2``
     - relaxation
     - SCM-relaxation, :math:`\ell_2` divergence [RelaxSC]_.
   * - ``RELAX_ENTROPY``
     - relaxation
     - SCM-relaxation, entropy divergence.
   * - ``RELAX_EL``
     - relaxation
     - SCM-relaxation, empirical-likelihood divergence.



Shared assumptions across the RESCM class
-----------------------------------------

The penalized and relaxation branches differ in how they regularize
the weights, but they share the same identifying stack. The
shared structural conditions, consolidated from Wang-Xing-Ye
(:math:`L_\infty` SCM) and Liao-Shi-Zheng (SCM-relaxation):

**A1 (Linear factor model for untreated outcomes).** Each unit's
untreated outcome obeys

.. math::

   y_{jt}^N \;=\; \boldsymbol\lambda_j' \mathbf f_t
              \;+\; u_{jt},
   \qquad j \in \{0\} \cup [J], \;\; t \in \mathcal T,

with :math:`\mathbf f_t` an :math:`r`-vector of latent common
factors, :math:`\boldsymbol\lambda_j` a unit-specific loading,
and :math:`u_{jt}` a mean-zero idiosyncratic shock orthogonal to
the factors. Both branches lean on this factor structure: it is
what makes the treated unit's untreated outcome a (dense) linear
combination of the donors' outcomes plus an orthogonal error.

**A2 (Single treated unit, sharp absorbing treatment).** Unit
:math:`j = 0` is the only treated unit; treatment turns on at
:math:`T_0 + 1` and stays on. Donors are untreated throughout
(no interference). Both papers' main theorems are stated for the
single-treated case; multi-treated extensions exist
(*FECT* / :doc:`sdid` for staggered designs are the
mlsynth alternatives).

**A3 (Weak temporal dependence).** The errors :math:`u_{jt}`
are mean-zero, weakly dependent (:math:`\alpha`-mixing or
:math:`\rho`-mixing with summable autocovariances), and have
bounded moments. This is what licenses the HAC long-run variance
estimator and the sequential CLT for the post-period mean. The
Wang-Xing-Ye theory explicitly accommodates stationary,
trend-stationary, and unit-root non-stationary cases under
exponential decay in correlation rates; the Liao-Shi-Zheng
theory uses an :math:`\alpha`-mixing CLT.

**A4 (High-dimensional sample-size regime).** :math:`T_0 \to
\infty` and :math:`N` may grow (potentially :math:`N \gg T_0`),
which is the entire point of regularizing toward a dense
solution. The classical-SCM least-squares first-order condition
:math:`\hat{\boldsymbol\Sigma} \boldsymbol\omega =
\hat{\boldsymbol\Upsilon}` is under-determined once
:math:`N \gtrsim T_0`; both branches restore well-posedness, but
do so under the same growth regime.

**A5 (Treated unit (approximately) in the convex hull of donor
loadings).** The treated loading :math:`\boldsymbol\lambda_0` is
spanned by the donor loadings up to an approximation error that
the regularization tolerance (:math:`\lambda` for penalized,
:math:`\eta` for relaxation) absorbs. *Remark.* If
:math:`\boldsymbol\lambda_0` is structurally outside the donor
hull -- even after relaxation -- both branches fail. Use
:doc:`iscm`.

**A6** (**Branch-specific regularization rate**).

* For the **penalized branch** (Wang-Xing-Ye): the penalty
  strength :math:`\lambda` and (when applicable) the mixing
  :math:`\alpha` are chosen so the weight estimator is
  consistent for its population target. In mlsynth both are
  selected by K-fold cross-validation on the pre-period.
* For the **relaxation branch** (Liao-Shi-Zheng): the
  tolerance :math:`\eta` (selected by validation) shrinks at a
  rate that makes the relaxed first-order condition
  asymptotically equivalent to the oracle moment condition;
  Theorem 1 of the paper gives the precise rate.

**A7 (Latent group structure -- relaxation branch only).** The
donor loadings :math:`\boldsymbol\lambda_j` fall into :math:`K`
latent groups; both :math:`K` and the factor count :math:`r` may
diverge, regardless of their relative order. Under this
structure the relaxation branch's divergence-minimizing weight
asymptotically recovers **equal weights within groups** --
Liao-Shi-Zheng's interpretable "transparent solution" that
classical SCM's arbitrary tie-break does not deliver. *Remark.*
Without latent groups the relaxation still produces a valid
counterfactual; the within-group-equal-weight interpretation is
the bonus that group structure buys.

**A8 (Strictly convex divergence -- relaxation branch only).**
The divergence :math:`D` is strictly convex on the simplex
(squared :math:`\ell_2`, entropy, empirical likelihood, or any
GEL-class function with non-negative support and restricted
strong convexity). Strict convexity is what makes the relaxed
feasible set have a **unique** minimizer -- this is the
well-posedness fix that distinguishes RESCM-relaxation from
classical SCM in the under-determined regime.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) **Latent factor model (A1).** Both branches lean on the
    factor structure. If the panel is not well-described by a
    small number of common factors, the linear-combination
    representation has a non-vanishing residual the
    regularization cannot remove.

    *Plausibly violated when* donors are largely idiosyncratic
    (unrelated processes), or when a donor has a one-time
    structural break that no factor model absorbs.
    *Diagnostic*: SVD of the pre-period donor matrix should
    show a clear factor cutoff (top few singular values
    carrying most of the energy). A flat-tailed spectrum flags
    A1 failure; in that regime, switch to balancing-aware
    estimators (:doc:`microsynth` for unit-level data) or stay
    with *canonical SCM* (which does not lean as hard on
    A1).

(b) **Weak temporal dependence (A3).** The HAC long-run variance
    estimator needs mixing with at-least-summable
    autocovariances. Strong serial correlation, long-memory
    series, or non-stationarity break this.

    *Plausibly violated when* outcomes are price levels or
    cumulative quantities. *Diagnostic*: ADF/KPSS on the
    pre-period residuals; non-stationarity flags A3 failure.
    For unit-root outcomes the Wang-Xing-Ye theory still goes
    through under exponential correlation decay, but the
    finite-sample HAC variance is often optimistic.
    First-difference before fitting, or use :doc:`sbc`
    (stationary-cycle estimator).

(c) **Sample-size regime (A4).** Both branches' asymptotics
    require :math:`T_0` large. The Verification section's
    Monte Carlo documents what happens at :math:`T_0 < N`: the
    point estimate is unbiased and powerful, but the analytic
    two-term variance over-rejects (the normal approximation
    has not kicked in).

    *Plausibly violated when* :math:`T_0 \le N` or
    :math:`T_0 < 50`-ish. *Diagnostic*: read ``res.pre_rmse``
    against the donor noise floor (the mean leave-one-out
    pre-RMSE across donors); if treated pre-RMSE is much
    smaller, the estimator is over-fitting and
    :math:`\hat\rho^2_{(1)}` understates uncertainty. Report
    a placebo / conformal CI (e.g. via *canonical SCM* with
    permutation inference) alongside the analytic CI.

(d) **Treated unit in donor convex hull (A5).** Both branches
    keep the simplex (or shrink toward the simplex). If the
    treated unit's loading is structurally outside the donor
    hull, the regularization tolerance :math:`\eta` /
    :math:`\lambda` cannot bridge it.

    *Plausibly violated when* the treated unit is qualitatively
    different from every donor (a coastal mega-state against
    only interior states; a tech-led economy against
    commodity-led donors). *Diagnostic*: read ``res.pre_rmse``
    on the un-regularized ``SC`` corner and on the most-
    regularized variant in your run. If both stay high, the
    hull condition is failing. Switch to :doc:`iscm` (which
    identifies the effect even when the treated unit is
    outside the hull) or :doc:`nsc` (which drops the simplex
    to allow negative-weight extrapolation).

(e) **Regularization rate (A6) -- choosing :math:`\lambda` /
    :math:`\eta`.** Cross-validation can be noisy on a short
    pre-period; the selected hyperparameter is then noise and
    the implied counterfactual flips with the seed.

    *Plausibly violated when* :math:`T_0 \le 20`. *Diagnostic*:
    re-run with different ``cv_folds`` or different
    train/validation splits; if the selected :math:`\lambda`
    /:math:`\eta` and the implied ATT move substantially, the
    CV is not informative. Either fix :math:`\lambda` at a
    domain-informed value via the explicit knobs, or fall back
    to *canonical SCM* / :doc:`tssc` which do not need
    CV at all.

(f) **Latent group structure (A7 -- relaxation only).** The
    within-group-equal-weights interpretation requires that
    donors actually fall into groups on their loadings.

    *Plausibly violated when* donor loadings are continuously
    distributed without natural clusters. *Diagnostic*: inspect
    the empirical CDF of ``res.fits["RELAX_ENTROPY"].donor_weights``;
    a multi-modal CDF supports the group structure, a smooth
    CDF means the group story is rhetorical rather than real.
    If groups are not present, prefer ``RELAX_L2`` (which
    targets minimum-norm weights, no group assumption needed).

(g) **Choice of divergence (A8 -- relaxation only).** The three
    divergences encode different priors: :math:`\ell_2`
    minimum-norm (defensive default), entropy /
    maximum-entropy (closest-to-uniform), empirical likelihood
    (GEL-optimal). Each is valid; the choice is a modeling
    decision.

    *Practical rule of thumb*: ``RELAX_L2`` for a defensible
    default; ``RELAX_ENTROPY`` when the policy story is "the
    counterfactual should be close to a uniform average of
    donors"; ``RELAX_EL`` when the inferential framework is
    explicitly GEL-based and you want the maximum-likelihood
    interpretation. The three rarely disagree by more than
    Monte-Carlo noise on well-behaved panels.

When to use RESCM -- and when not to
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Reach for RESCM when:**

* You have a **single treated unit and a large donor pool**
  (:math:`N` comparable to or exceeding :math:`T_0`), and a
  factor-driven panel where many donors plausibly carry
  signal. Classical SCM concentrates weight on a few donors;
  the dense-shrinkage variants in RESCM diversify prediction
  risk across the pool.
* You want a **stable, dense counterfactual** that does not
  hinge on one or two donors -- exactly the "denser weighting
  philosophy" both papers advocate. The :math:`\ell_\infty`
  penalty caps the largest weight; the relaxation branch
  spreads weight by minimizing a strictly-convex divergence
  on the simplex.
* You want a **one-stop interface** to compare classical SCM,
  Lasso-SC, Ridge-SC, elastic-net SCM, :math:`L_\infty`-SCM,
  and three relaxation variants on the same panel. The
  ``methods`` argument selects estimators by name; each maps
  to one call of the same convex engine.
* You want **HAC-based, classical-statistics inference**
  (Li 2020 two-term standard errors) on the ATE rather than a
  permutation or conformal procedure. Note the
  finite-sample-inference caveat below.

**Do not use RESCM when:**

* **You need sparse, hand-interpretable weights** as the
  headline deliverable. Both branches actively shrink toward
  dense solutions; that is the design choice. If the policy
  story has to be "California ≈ Utah + Montana + Nevada", run
  *canonical SCM* / :doc:`tssc`, or :doc:`fscm`
  (forward-selected SC with the simplex retained) for a
  sparse-by-construction donor set.
* **The treated unit is structurally outside the donor hull.**
  Both branches keep (or shrink toward) the simplex. Pre-RMSE
  stays high at every regularization level. Use :doc:`iscm`
  (identifies the effect through donors that use the treated
  unit as a positive-weight donor in their own synthetic
  controls) or :doc:`nsc` (drops the simplex restriction).
* **Very short pre-period** :math:`(T_0 < N)`. The Path-B
  Monte Carlo in the Verification section above documents
  the over-rejection: the analytic two-term variance has not
  kicked in. Either lengthen the pre-period (aggregate to a
  finer time grid), prune donors, or report a placebo /
  conformal CI alongside.
* **Non-stationary or unit-root outcomes.** A3's mixing
  assumption strains here; the Wang-Xing-Ye theory accommodates
  unit-root cases under exponential correlation decay, but the
  HAC variance is often optimistic in finite samples.
  First-difference, or use :doc:`sbc`.
* **You have multiple treated units / staggered adoption.**
  RESCM's theory is built for a single treated unit. Use
  *FECT* or :doc:`sdid` for staggered designs.
* **Spillovers across donors.** A2's no-interference clause
  fails; the factor model's orthogonality breaks. Use
  :doc:`spillsynth` or :doc:`spsydid`.
* **Continuous or multi-valued treatment.** RESCM encodes a
  single binary intervention; continuous dose belongs in
  :doc:`ctsc`.
* **Distributional questions** (Lorenz curves, QTEs, tail
  effects). RESCM targets the mean ATE through a Gaussian-
  likelihood linear projection. Use :doc:`dsc` for
  distributional effects.
* **You need Bayesian posterior credible bands.** RESCM
  returns frequentist HAC-based CIs and a single point
  estimate per corner. For posterior inclusion probabilities
  and credible bands on donor weights, use :doc:`bvss`
  (spike-and-slab with a soft simplex -- the natural Bayesian
  analogue of the dense-vs-sparse trade-off the RESCM family
  addresses frequentistically).
* **You want predictor-level (covariate + lagged-outcome)
  matching** rather than outcome-only matching. RESCM's
  workhorse projection is on donor outcomes; for
  predictor-matching with L1 sparsity on the
  predictor-weight matrix, use :doc:`sparse_sc`.
* **You want the factor model itself** (the loadings and
  factors) **estimated and reported**. RESCM is agnostic to
  factor estimation -- the factor model only motivates the
  linear projection and never enters the estimator. For
  factor-aware estimators that surface :math:`\hat F` and
  :math:`\hat\Lambda`, use :doc:`fma` or :doc:`clustersc`
  (which exposes the HSVT rank in its results).
* **Donor selection is the bottleneck, not weight
  shrinkage.** If you have a small number of donors and want
  to *select* the best subset rather than spread weight
  across a wide pool, use :doc:`fscm` (forward selection on
  donor units) or :doc:`pda` with ``methods=["fs"]``
  (forward-selected PDA with sample-splitting inference).




Inference
---------

Once the donor weights are fixed by any corner case, the post-period gap
:math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0` is a scalar series whose mean
is the ATE. ``RESCM`` reports the Li [LiSCM2020]_ two-term long-run variance
used by the L-infinity paper,

.. math::

   \widehat{\mathrm{se}}(\bar{\Delta}) =
     \sqrt{\hat{\rho}^2_{(1)}/T_1 + \hat{\rho}^2_{(2)}/T_2},

where :math:`\hat{\rho}^2_{(1)}` is the HAC long-run variance of the pre-period
prediction residuals -- carrying the **first-stage weight-estimation
uncertainty** -- and :math:`\hat{\rho}^2_{(2)}` is the HAC long-run variance of
the de-meaned post-period effects (Bartlett kernel, :math:`\lfloor
T_2^{1/4}\rfloor` lag). The pre-period term is essential for dense
penalized/relaxed weights: unlike forward-selection PDA -- whose sample
splitting makes pre/post asymptotically independent and lets a post-only
variance suffice -- the dense estimators reuse the entire pre-window to fit the
weights, so dropping :math:`\hat{\rho}^2_{(1)}` understates the standard error.

Empirical Illustration: California's Proposition 99
---------------------------------------------------

The canonical synthetic-control application [ABADIE2010]_ studies the effect of
California's 1988 Proposition 99 tobacco-control program on per-capita cigarette
sales, with 38 control states over 1970-2000. Running ``RESCM`` with the
classic ``SC`` corner alongside the two papers' headline estimators reuses the
same panel and returns each method's counterfactual, ATE, and a HAC confidence
interval.

.. code-block:: python

   import pandas as pd
   from mlsynth import RESCM

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/smoking_data.csv"
   df = pd.read_csv(url)
   df["Proposition 99"] = df["Proposition 99"].astype(int)

   res = RESCM({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                "unitid": "state", "time": "year",
                "methods": ["SC", "LINF", "RELAX_L2"], "alpha": 0.05,
                "display_graphs": True}).fit()

   for name, fit in res.fits.items():
       print(f"{name:9s} ATE {fit.att:8.3f}  SE {fit.att_se:6.3f}  "
             f"95% CI ({fit.ci[0]:7.2f},{fit.ci[1]:7.2f})  "
             f"p={fit.p_value:.3f}  donors={len(fit.donor_weights)}")

This prints::

   SC        ATE  -17.371  SE  2.304  95% CI ( -21.89, -12.86)  p=0.000  donors=6
   LINF      ATE  -17.359  SE  2.303  95% CI ( -21.87, -12.85)  p=0.000  donors=6
   RELAX_L2  ATE  -22.190  SE  4.115  95% CI ( -30.26, -14.12)  p=0.000  donors=7

The penalized corners (``SC``, ``LINF``) agree on an outcome-only effect of
about :math:`-17` cigarette packs per capita: when the donor pool fits the
pre-period well (pre-treatment :math:`R^2 \approx 0.98`, pre-RMSE
:math:`1.45`), cross-validation picks near-zero :math:`\ell_\infty` shrinkage
and ``LINF`` collapses onto classic SCM. The relaxation corner (``RELAX_L2``)
trades a little pre-fit (:math:`R^2 \approx 0.97`) for a more constrained
weight and lands at a larger effect, :math:`-22.2`, whose wider confidence
interval reflects the larger pre-period residual variance the two-term standard
error correctly propagates.

Verification
------------

.. note::

   **Empirical (Path A, Proposition 99).** ``SC``/``LINF``/``RELAX_L2`` run on
   the smoking panel (above); the penalized corners reproduce the classic
   outcome-only SCM effect (:math:`\approx -17`) and the relaxation corner the
   denser, larger estimate, both significant and consistent with the
   literature.

   **Simulation (Path B, high dimension).** A size/power Monte Carlo in the
   regime these methods target -- :math:`N=90` donors, :math:`T_1=36`
   pre-periods (:math:`N \gg T_1`), :math:`T_2=36`, three-factor AR(1) DGP,
   treated loading a convex mix of donors, idiosyncratic :math:`N(0,1)` --
   rejecting ``H0: ATE = 0`` at 5% (50 replications; :math:`\delta=0` is size,
   :math:`\delta=1` is power):

   .. list-table::
      :header-rows: 1
      :widths: 16 18 18 18

      * - :math:`\delta`
        - ``SC``
        - ``LINF``
        - ``RELAX_L2``
      * - 0 (size)
        - 0.20
        - 0.22
        - 0.28
      * - 1 (power)
        - 0.98
        - 0.98
        - 0.94

   **Estimation is unbiased and powerful** (mean ATT bias :math:`\approx 0`;
   power :math:`0.94`-:math:`0.98`), but the analytic ATE test **over-rejects**
   in this short, high-dimensional panel. A diagnostic isolates two mechanisms:
   ``SC`` genuinely over-fits the 36-period pre-window (pre-RMSE :math:`0.77` vs
   the true noise sd :math:`1.0`), so :math:`\hat{\rho}^2_{(1)}` understates
   estimation uncertainty -- it self-corrects as :math:`T_1` grows (pre-RMSE
   :math:`\to 0.94` at :math:`T_1=250`); ``RELAX_L2`` does *not* over-fit
   (pre-RMSE :math:`\approx 1.0`), so its over-rejection reflects the analytic
   influence function not capturing how strongly-relaxed dense weights feed into
   the ATE. Both papers' asymptotics require :math:`T_1` large relative to
   :math:`N`; the normal approximation is unreliable at :math:`T_1 < N`, and
   the two-term variance (which already cuts the post-only over-rejection by
   :math:`\sim 35\%`) does not fully close the gap here. For honest
   finite-sample inference in this regime, prefer a placebo / conformal
   procedure [CWZ2021]_. (Only 50 replications -- noisy; the relaxation
   :math:`\eta` is validated by CV, not fixed.)

   **Durable benchmarks.** The relaxation branch is pinned against the authors'
   own paper and code: ``rescm_brexit`` (Path A -- the Brexit / UK real-GDP
   application, ``standardize=False``) and ``rescm_relax_ref`` (cross-validation
   -- mlsynth's L2 relaxation vs the ``scmrelax`` package, cell by cell at a
   matched :math:`\eta`). See the dedicated page :doc:`replications/rescm`.

Core API
--------

.. automodule:: mlsynth.estimators.laxscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.RESCMConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``RESCM.fit()`` returns a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMResults`, whose ``fits``
maps each named corner case to a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMMethodFit` (donor weights,
intercept, counterfactual, gap, ATE, two-term HAC standard error, CI, p-value,
nonzero donor weights, fit diagnostics, and the realized hyperparameters).
Convenience aliases (``att``, ``att_se``, ``counterfactual``,
``donor_weights``) forward to the first requested method. The prepared,
NumPy-only panel is exposed as a
:class:`~mlsynth.utils.laxscm_helpers.structures.RESCMInputs`, with units and
time addressed through an :class:`IndexSet`.

.. note::

   ``RESCM.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on
   the standardized two-family contract. It is a dispatcher over the corner-case
   estimators in ``res.fits``; the selected estimator drives the flat accessors
   (``res.att`` / ``res.att_ci`` / ``res.counterfactual`` / ``res.gap`` /
   ``res.donor_weights`` / ``res.pre_rmse``), which resolve through the
   standardized sub-models. ``res.att_by_method()`` / ``res.se_by_method()``
   report every fit.

.. automodule:: mlsynth.utils.laxscm_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

The named-estimator registry: each entry maps a name to the exact convex-engine
call (branch and keyword arguments).

.. automodule:: mlsynth.utils.laxscm_helpers.specs
   :members:
   :undoc-members:

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds the
unit/time ``IndexSet``es, and splits pre/post.

.. automodule:: mlsynth.utils.laxscm_helpers.setup
   :members:
   :undoc-members:

The weak-dependence two-term HAC ATE inference (Li 2020).

.. automodule:: mlsynth.utils.laxscm_helpers.inference
   :members:
   :undoc-members:

Run loop dispatching each corner case to the convex engine and assembling the
typed per-method fits.

.. automodule:: mlsynth.utils.laxscm_helpers.estimation
   :members:
   :undoc-members:

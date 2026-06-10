Panel Data Approach (PDA)
=========================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The panel data approach (PDA) of Hsiao, Ching and Wan [HCW]_ estimates a
single treated unit's untreated counterfactual by a **linear regression on
the control units**, fit over the pre-treatment window and extrapolated
out-of-sample. Unlike the synthetic control method, PDA imposes **no
constraints** on the coefficients (no simplex, no non-negativity), so it is a
plain projection justified by a latent-factor model: if every unit loads on a
small set of common factors, the treated unit's untreated path is a linear
combination of the controls' paths plus an orthogonal error.

The challenge is **which controls** and **how many**. Classical PDA was built
for low dimensions (few controls relative to pre-periods) and chooses controls
by AIC/BIC, which break down once the number of controls ``N`` approaches or
exceeds the pre-period length ``T1``. ``mlsynth`` packages three
high-dimensional PDA variants that resolve this differently, **each with the
estimation and inference theory of its own paper**:

* **L2-relaxation** (``l2``; Shi & Wang [l2relax]_) -- a *dense* estimator (a
  "cousin of ridge") for when the factor model makes the projection
  coefficients dense; tolerates ``N > T1``; prediction is robust to
  heteroskedasticity.
* **LASSO** (``lasso``; Li & Bell [LASSOPDA]_) -- an L1 estimator that
  *selects* a sparse set of relevant controls; computationally far cheaper
  than AIC/BIC and works for ``N > T1``.
* **Forward selection** (``fs``; Shi & Huang [fsPDA]_) -- a greedy procedure
  that grows the control set one unit at a time, with valid post-selection
  inference and no sparsity requirement (works for dense *or* sparse models).

All three target the **single-treated-unit, many-candidate-controls** regime
and produce a time-varying treatment effect and an average treatment effect
(ATE) with a HAC-based confidence interval. The practical choice among them
(detailed below) follows the authors' own arguments: ``l2`` when the
coefficients are dense, ``lasso`` when only a few controls matter and you want
an interpretable selection, ``fs`` when you want a cheap, predictive ensemble
with honest post-selection inference regardless of sparsity.

Notation
--------

We adopt the notation of Shi & Wang [l2relax]_. For a positive integer ``n``,
:math:`[n] = \{1, \ldots, n\}`; :math:`\mathbf{I}_n` is the identity,
:math:`\mathbf{1}_n`, :math:`\mathbf{0}_n` the all-ones/zeros vectors. For a
matrix :math:`\mathbf{A} = (a_{ij})` and index sets
:math:`\mathcal{S}, \mathcal{Q}`, the submatrix is
:math:`\mathbf{A}_{\mathcal{S}\mathcal{Q}} = (a_{ij})_{i\in\mathcal{S}, j\in\mathcal{Q}}`
and the subvector :math:`\mathbf{x}_{\mathcal{Q}} = (x_i)_{i\in\mathcal{Q}}`.
:math:`\phi_{\min}(\cdot)`, :math:`\phi_{\max}(\cdot)` denote the smallest and
largest eigenvalues; :math:`\|\mathbf{A}\|_\infty = \max_{ij}|a_{ij}|`,
:math:`\|\mathbf{A}\|_2 = \sqrt{\phi_{\max}(\mathbf{A}'\mathbf{A})}`,
:math:`\|\mathbf{x}\|_1 = \sum_i |x_i|`.

The two **time-series operators** are central. Over an index set
:math:`\mathcal{S}` of periods,

.. math::

   \mathcal{E}_{\mathcal{S}}(\mathbf{x}_t) = \frac{1}{|\mathcal{S}|}
       \sum_{t\in\mathcal{S}} \mathbf{x}_t,
   \qquad
   \Gamma_{\mathcal{S}}(\mathbf{x}_t, \mathbf{y}_t') = \mathcal{E}_{\mathcal{S}}
       \Bigl( [\mathbf{x}_t - \mathcal{E}_{\mathcal{S}}(\mathbf{x}_t)]
              [\mathbf{y}_t - \mathcal{E}_{\mathcal{S}}(\mathbf{y}_t)]' \Bigr),

the sample mean and sample covariance of a series over :math:`\mathcal{S}`.
The intervention occurs after the pre-period
:math:`\mathcal{T}_1 = \{1, \ldots, T_1\}`; the post-period is
:math:`\mathcal{T}_2 = \{T_1+1, \ldots, T\}` with :math:`T_2 = |\mathcal{T}_2|`.
Unit :math:`j=0` is treated, :math:`\mathcal{N} = \{1,\ldots,N\}` the controls.

The shared model
~~~~~~~~~~~~~~~~

All three methods rest on a common latent-factor data-generating process: for
:math:`t \in \mathcal{T}`,

.. math::

   y_{jt}^0 = \mu_j + \boldsymbol{\lambda}_j' \mathbf{f}_t + u_{jt},
   \quad j \in \{0\}\cup\mathcal{N},

with :math:`\mathbf{f}_t` a :math:`q`-vector of latent common factors,
:math:`\boldsymbol{\lambda}_j` factor loadings, and :math:`u_{jt}` a
weakly-dependent idiosyncratic error orthogonal to the factors. Because the
common factors drive both the treated unit and the controls, the untreated
treated outcome admits a **linear projection** on the controls,

.. math::

   y_{0t} = \alpha^0 + \mathbf{x}_t' \boldsymbol{\beta}^0 + \epsilon_t,
   \qquad \mathbf{x}_t = (y_{1t}, \ldots, y_{Nt})',

with :math:`\mathbb{E}[\mathbf{x}_t \epsilon_t] = \mathbf{0}`. PDA fits
:math:`(\alpha^0, \boldsymbol{\beta}^0)` on :math:`\mathcal{T}_1` and predicts
:math:`\hat{y}_{0t}^0 = \mathcal{E}_{\mathcal{T}_1}(y_s) + [\mathbf{x}_t -
\mathcal{E}_{\mathcal{T}_1}(\mathbf{x}_s)]'\hat{\boldsymbol{\beta}}` for
:math:`t \in \mathcal{T}_2`. The treatment effect is
:math:`\hat{\Delta}_t = y_{0t} - \hat{y}_{0t}^0` and the ATE
:math:`\bar{\Delta} = \mathcal{E}_{\mathcal{T}_2}(\hat{\Delta}_t)`. The methods
differ in how they estimate :math:`\boldsymbol{\beta}` and, crucially, in the
inference theory each paper proves for :math:`\bar{\Delta}`.

L2-relaxation (``l2``, Shi & Wang)
----------------------------------

**Idea.** Under the factor model the projection coefficient
:math:`\boldsymbol{\beta}^0 = \boldsymbol{\Omega}^{-1}\boldsymbol{\Lambda}
(\boldsymbol{\Lambda}'\boldsymbol{\Omega}^{-1}\boldsymbol{\Lambda} +
\mathbf{I}_q)^{-1}\boldsymbol{\lambda}_0` is **dense** in general -- almost no
entries are exactly zero. Sparse methods (LASSO) are then mis-matched. With
:math:`\hat{\boldsymbol{\Sigma}} = \Gamma_{\mathcal{T}_1}(\mathbf{x}_t,
\mathbf{x}_t')` and :math:`\hat{\boldsymbol{\eta}} =
\Gamma_{\mathcal{T}_1}(\mathbf{x}_t, y_t)`, OLS solves the KKT condition
:math:`\hat{\boldsymbol{\Sigma}}\boldsymbol{\beta} = \hat{\boldsymbol{\eta}}`,
which is unstable or non-unique once :math:`N` is close to or exceeds
:math:`T_1`. L2-relaxation **relaxes** the sup-norm of this moment condition by
a tuning parameter :math:`\tau` and minimizes the coefficient norm:

.. math::

   \min_{\boldsymbol{\beta}} \tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2
   \quad \text{s.t.} \quad
   \|\hat{\boldsymbol{\eta}} - \hat{\boldsymbol{\Sigma}}\boldsymbol{\beta}\|_\infty
   \le \tau.

This is the "bias-variance trade-off" made explicit: tolerating a small
violation :math:`\tau` of the OLS moment condition shrinks the variance. At
:math:`\tau = 0` it reduces to (ridgeless) OLS; at :math:`\tau \ge
\|\hat{\boldsymbol{\eta}}\|_\infty` it gives :math:`\boldsymbol{\beta} =
\mathbf{0}`. ``mlsynth`` picks :math:`\tau` by **sequential out-of-sample
validation** on the tail of the training window (the validated :math:`\tau`
tracks the infeasible-optimal one, and both shrink toward zero as the sample
grows) over a log-spaced grid down to :math:`10^{-4}\max|\hat{\boldsymbol{\eta}}|`
(the optimum is often a tiny fraction of the cap). This is **time-respecting** --
the fit never sees periods later than the validation tail -- unlike the
released ``L2relax.CV``, whose 5-block K-fold trains on both past *and* future
of each block.

.. note::

   **Standardisation.** Following the authors' released ``L2relax``, the treated
   and control series are **standardised** (demeaned and scaled to unit
   variance) before forming :math:`\hat{\boldsymbol{\Sigma}}` /
   :math:`\hat{\boldsymbol{\eta}}`, and the solution is mapped back to the
   original scale. This is the default (``l2_standardize=True``) -- the
   :math:`\ell_2` penalty is scale-sensitive, so standardisation is both
   recommended and what reproduces the paper's empirical results; on the Hong
   Kong panel it moves the L2 estimate from :math:`2.48\%` to :math:`2.61\%`
   (closer to Shi & Wang's :math:`2.65\%`). Set ``l2_standardize=False`` for the
   raw-scale variant.

**Assumptions** (Shi & Wang).

*Assumption 1 (loadings).* :math:`\|\boldsymbol{\lambda}_0\|_\infty +
\|\boldsymbol{\Lambda}\|_\infty \le C`, and there is a :math:`q`-unit subset
making :math:`\boldsymbol{\Lambda}_{\mathcal{Q}\cdot}` full column rank; the
**average factor strength** :math:`\xi_N = \phi_{\min}(\boldsymbol{\Lambda}'
\boldsymbol{\Lambda}/N)` may vanish (weak factors allowed).

*Assumption 2 (errors).* The idiosyncratic covariance
:math:`\boldsymbol{\Omega}` has eigenvalues bounded between
:math:`\underline\sigma^2` and :math:`\overline\sigma^2`; errors may be
heteroskedastic and cross-sectionally dependent.

*Assumption 3-4 (sampling).* In- and out-of-sample sampling errors of the
sample moments are :math:`O_p(T_1^{-1/2})` for low-dimensional pieces and
:math:`O_p(\sqrt{\log N / (N\wedge T_1)})` for high-dimensional ones (holding
under time-series weak dependence, not just i.i.d.).

*Assumption 5 (ATE inference).* The oracle prediction error
:math:`\epsilon_t^*` and the effect-plus-error :math:`d_t^* = \Delta_t -
\mathbb{E}[\Delta_t] + \epsilon_t^*` have finite, positive long-run variances
:math:`\rho^2_{\epsilon^*}`, :math:`\rho^2_{d^*}` consistently estimable by HAC,
and a sequential CLT applies.

*Remark.* The coefficient estimator is consistent for the oracle target
(Theorem 1) and -- importantly -- the **prediction** error is asymptotically
*irrelevant to heteroskedasticity* (Theorem 2): unlike the coefficient MSE,
the out-of-sample MSE does not depend on the noise heterogeneity
:math:`\psi_{\max}`.

**Inference** (Shi & Wang, Theorem 3; single treated unit). With pre-period
prediction residuals :math:`e_t = y_t - \hat{y}_t` (:math:`t\in\mathcal{T}_1`)
and post-period effects :math:`\hat{\Delta}_t` (:math:`t\in\mathcal{T}_2`),

.. math::

   \hat{Z} = \frac{\bar{\Delta} - \Delta_{\mathcal{T}_2}}
       {\sqrt{\hat{\rho}^2_{(1)}/T_1 + \hat{\rho}^2_{(2)}/T_2}}
   \xrightarrow{d} N(0,1),

where :math:`\hat{\rho}^2_{(1)}` is the HAC long-run variance of the
pre-period residuals (first-stage estimation uncertainty) and
:math:`\hat{\rho}^2_{(2)}` is the HAC long-run variance of the de-meaned
post-period effects (post-period averaging). **Both** sources of uncertainty
enter, which matters when :math:`T_1` and :math:`T_2` are comparable.

**When to use.** Dense, factor-driven coefficients; high dimension
(:math:`N>T_1` permitted); when prediction accuracy and heteroskedasticity-
robustness matter more than identifying a handful of controls.

LASSO (``lasso``, Li & Bell)
----------------------------

**Idea.** When only a *few* controls are truly relevant, an L1 penalty selects
them and shrinks the rest. Li & Bell estimate

.. math::

   \hat{\boldsymbol{\beta}}^{\text{las}}
   = \operatorname*{argmin}_{\boldsymbol{\beta}} \;
     \sum_{t\in\mathcal{T}_1} (y_{0t} - \mathbf{x}_t'\boldsymbol{\beta})^2
     + \lambda \sum_{j} |\beta_j|,

with :math:`\lambda` chosen by (leave-one-out) cross-validation, then predict
the counterfactual as in the shared model. LASSO works for :math:`N > T_1`
(where AIC/AICC/BIC cannot even be computed) and is far cheaper.

**Assumptions** (Li & Bell). They *relax* HCW's linear-conditional-mean
assumption and drop one of HCW's identification conditions. The key conditions
are: a factor model with :math:`\mathrm{Rank}(\tilde{B}) = K`
(enough independent factor variation among the controls); a weakly dependent,
weakly stationary panel so laws of large numbers and CLTs apply to partial
sums; consistency of the pre-period least-squares pieces
(:math:`\hat{\delta}_1 - \delta_1, \hat{\delta}-\delta = O_p(T_1^{-1/2})`); and
:math:`\rho`-mixing with geometric decay plus a finite limit
:math:`\eta = \lim T_2/T_1`. Sparsity (only :math:`m` of
:math:`\boldsymbol{\beta}` non-zero, :math:`m` fixed or :math:`o(T_1)`) is
assumed for the high-dimensional selection.

*Remark.* Li & Bell prove consistency :math:`\hat{\Delta}_1 - \Delta_1 =
O_p(T_1^{-1/2} + T_2^{-1/2})` (estimation error has two parts: first-stage
:math:`O_p(T_1^{-1/2})` and post-averaging :math:`O_p(T_2^{-1/2})`), holding
even when :math:`y_t` is trend-stationary.

**Inference** (Li & Bell, Theorem 3.2). With :math:`\hat{\Sigma} =
\hat{\Sigma}_1 + \hat{\Sigma}_2`,

.. math::

   \text{T.S.} = \frac{\sqrt{T_2}\,\hat{\Delta}_1}{\sqrt{\hat{\Sigma}}}
   \xrightarrow{d} N(0,1),

where :math:`\hat{\Sigma}_2` is the Newey-West HAC long-run variance of the
post-period effects, and :math:`\hat{\Sigma}_1` is the first-stage
(pre-period estimation) variance -- the OLS prediction variance of the mean
post-period counterfactual on the selected support. Li & Bell note
:math:`\hat{\Sigma}_1` is **negligible when** :math:`T_1 \gg T_2`, so the
post-period term dominates in long-pre-period panels.

**When to use.** A genuinely sparse set of relevant controls; very large
:math:`N` (even :math:`N/T_1 \to \infty`); when an interpretable, computa-
tionally cheap selection is preferred. (For selection *consistency*, Li & Bell
note the adaptive LASSO; for prediction, plain LASSO already beats AIC/BIC and
leave-many-out CV in their simulations.)

Forward selection (``fs``, Shi & Huang)
---------------------------------------

**Idea.** Rather than penalize, *grow* the control set greedily. Start empty;
at each step add the control whose inclusion maximizes the pre-treatment OLS
:math:`R^2` (equivalently minimizes the residual sum of squares). The number of
selected controls :math:`R` is a tuning parameter chosen by a **modified BIC**
(Wang, Li & Tsai),

.. math::

   \hat{R} = \operatorname*{argmin}_{r}
     \log\bigl(\hat{\sigma}^2(\hat{U}_r)\bigr)
     + \log(\log N)\,\frac{r\,\log T_1}{T_1},

with :math:`\hat{\sigma}^2(\hat{U}_r)` the pre-period residual variance of OLS
on the :math:`r`-unit set. The counterfactual is the OLS extrapolation on the
chosen set :math:`\hat{U}_{\hat{R}}`. Forward selection evaluates
:math:`\sum_r (N-r+1)` regressions -- *linear* in :math:`N` -- versus the
:math:`2^N` of exhaustive subset search.

**Assumptions** (Shi & Huang). Asymptotics are *multi-index*: :math:`N\to\infty`
with :math:`T_1 = T_1(N)` deterministic, :math:`\log N / T_1 \to 0`, and
:math:`T_2 = T_2(N) \to \infty` with :math:`\log N / T_2 \to 0` (:math:`N`
may exceed :math:`T_1`).

*Assumption 1 (sparse Riesz / restricted eigenvalue).* The minimal eigenvalue
of the population Gram matrix over any :math:`u`-unit subset
(:math:`u \le (1+\delta_1)R`) is bounded below -- a condition the authors show
is a **natural implication of the latent factor model**, not an ad hoc
restriction.

*Assumption 2 (second moments).* Sample second moments converge at the
high-dimensional rate :math:`O_p(\sqrt{\log N / T_1})` with bounded fourth
moments.

*Assumption 3 (post-period).* Analogous convergence and long-run-variance
bounds on the post-treatment data.

*Assumption 4 (weak dependence).* The series are strong (:math:`\alpha`-)
mixing with geometric decay, so a Berry-Esseen bound for heterogeneous time
series applies.

*Remark.* The validity is **uniform** over a class of DGPs (Theorem 1) -- it
holds whether the true coefficients are **dense or sparse**, which separates
fsPDA from the post-selection-inference literature that needs sparsity or the
oracle property. Theorem 2 shows the greedy algorithm attains a regression
variance asymptotically as small as the best :math:`u`-unit subset, so the
cheap forward search is statistically efficient.

**Inference** (Shi & Huang, Eq. 4). Because forward selection uses only the
pre-period and, under weak dependence, the pre- and post-periods become
**asymptotically independent** (sample splitting), the naive conditional
t-statistic is valid:

.. math::

   \hat{\mathcal{Z}}_{\hat{U}} = \hat{\rho}_{\tau\hat{U}}^{-1}\sqrt{T_2}\,
       \bar{\Delta}_{\hat{U}} \xrightarrow{d} N(0,1),

where :math:`\hat{\rho}^2_{\tau\hat{U}}` is the HAC long-run variance of the
de-meaned post-period effects. **No first-stage variance term is needed** --
the asymptotic independence absorbs it -- which makes fsPDA's inference the
simplest of the three.

.. note::

   **Long-run-variance estimator.** ``mlsynth`` defaults to the **prewhitened
   Newey-West** estimator (Andrews-Monahan VAR(1) prewhitening + Bartlett kernel
   with the data-driven NW(1994) bandwidth + finite-sample adjustment) -- R's
   ``sandwich::lrvar(..., prewhite = TRUE, adjust = TRUE)``, which Shi & Huang
   use in their application scripts. Prewhitening is essential when the
   treatment-effect series is strongly serially dependent: monthly growth rates
   mean-revert (lag-1 autocorrelation around :math:`-0.45` in the luxury-watch
   panel), and a plain Bartlett kernel cannot absorb that, leaving
   :math:`\hat\rho` nearly double its true value and the test far too
   conservative. Setting ``lrvar_lag`` instead switches to the released
   ``est.fsPDA`` package's fixed-lag Bartlett estimator
   (default lag :math:`\lfloor T_2^{1/4}\rfloor`, capped at
   :math:`\lfloor\sqrt{T_2}\rfloor`); on the watch panel that no-prewhitening
   form gives an insignificant :math:`t \approx -1.15`, versus the prewhitened
   default's :math:`-2.51` (the paper reports :math:`-2.457`).

**When to use.** A large candidate-control pool where the goal is to
*synthesize an ensemble* that mimics the outcome (not to interpret which
controls are "causal"); when computational efficiency and honest
post-selection inference matter; and regardless of whether the underlying model
is sparse. (Shi & Huang recommend the adaptive LASSO instead when the *identity*
of a few causal controls is the object of interest.)

Choosing among the three
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 28 30

   * - Method
     - Coefficient structure
     - Inference variance
     - Use when
   * - ``l2``
     - dense (factor-implied)
     - pre + post HAC (both terms)
     - dense coefficients; ``N>T1``; prediction & heteroskedasticity-robustness
   * - ``lasso``
     - sparse
     - post HAC (+ first stage, small if ``T1>>T2``)
     - few relevant controls; interpretable selection; very large ``N``
   * - ``fs``
     - dense or sparse
     - post HAC only (sample splitting)
     - large pool; predictive ensemble; cheap; honest post-selection inference



Shared assumptions across the PDA class
---------------------------------------

The three estimators (``l2``, ``lasso``, ``fs``) differ in how they
fit :math:`\boldsymbol\beta`, but they share the same identifying
stack. Stated formally:

**A1 (Latent factor model for untreated outcomes).** All
:math:`N + 1` units share at most :math:`q` common latent factors,

.. math::

   y_{jt}^0 \;=\; \mu_j \;+\; \boldsymbol\lambda_j' \mathbf f_t
              \;+\; u_{jt},
   \qquad j \in \{0\} \cup \mathcal N, \;\; t \in \mathcal T,

with :math:`\mathbb E[\mathbf f_t u_{jt}] = 0`. This is the
*shared model* underlying HCW (Hsiao-Ching-Wan 2012), Li-Bell
(2017), Shi-Huang (2023), and Shi-Wang (l2). The factor structure
is what licenses the linear projection of the treated unit's
untreated outcome on the controls' outcomes,

.. math::

   y_{0t} \;=\; \alpha^0 \;+\; \mathbf x_t' \boldsymbol\beta^0
              \;+\; \epsilon_t,
   \qquad \mathbb E[\mathbf x_t \epsilon_t] = \mathbf 0,

with :math:`\boldsymbol\beta^0 = \boldsymbol\Omega^{-1}
\boldsymbol\Lambda (\boldsymbol\Lambda' \boldsymbol\Omega^{-1}
\boldsymbol\Lambda + \mathbf I_q)^{-1} \boldsymbol\lambda_0`.

**A2 (Single treated unit, sharp absorbing aggregate-level
treatment).** Unit :math:`j = 0` is the only treated unit;
treatment turns on at :math:`T_1 + 1` and stays on. Donors are
untreated throughout (no interference). The original HCW /
Li-Bell / Shi-Huang theorems are stated for this single-treated
case. (The l2-relaxation paper Section 4.4 sketches a
multiple-treated-units extension with a short post-window; the
mlsynth implementation tracks the single-treated form.)

**A3 (Weak temporal dependence).** The series
:math:`(\mathbf x_t, y_{0t})` are :math:`\rho`-mixing or strong-
mixing with at-least-geometric decay (the exact rate varies by
variant):

* Li-Bell A6: :math:`w_t = (\tilde y_t', \epsilon_{1t})` is a
  weakly stationary :math:`\rho`-mixing process with
  :math:`\rho(\tau) = O(\lambda^\tau)`.
* Shi-Huang A4: strong (:math:`\alpha`-) mixing with geometric
  decay, so a Berry-Esseen bound for heterogeneous time series
  applies.
* Shi-Wang A3-A4: time-series weak dependence at the
  :math:`O_p(T_1^{-1/2})` and :math:`O_p(\sqrt{\log N / (N \wedge
  T_1)})` rates for the sample moments.

This is what makes pre-period sample moments converge at the
high-dimensional rate and -- crucially -- what makes the
pre-period and post-period **asymptotically independent**, which
is the engine behind fs-PDA's sample-splitting inference and the
two-term HAC variance in l2 / lasso.

**A4 (Sample-size regime).** :math:`N \to \infty`,
:math:`T_1 = T_1(N) \to \infty` deterministically with
:math:`\log N / T_1 \to 0`, :math:`T_2 \to \infty` with
:math:`\log N / T_2 \to 0`. :math:`N` may exceed :math:`T_1`,
which is the entire point of the high-dimensional PDA literature.
Li-Bell's A7 additionally posits
:math:`\eta = \lim T_2 / T_1 \in [0, \infty)`, which determines
whether the first-stage variance term :math:`\hat\Sigma_1`
matters.

**A5 (Donor pool regularity).** The controls' Gram matrix has
enough variation:

* For ``lasso`` (Li-Bell A2): :math:`\mathrm{Rank}(\tilde B) = K`
  -- removing the first row of the loading matrix leaves
  full-rank factor variation; :math:`E[x_t x_t']` is invertible
  on the active set.
* For ``fs`` (Shi-Huang A1): a **sparse Riesz / restricted
  eigenvalue** condition -- the minimum eigenvalue of the
  population Gram matrix over any :math:`u`-unit subset
  (:math:`u \le (1 + \delta_1) R`) is bounded below. The paper
  shows this is a *natural implication* of the latent factor
  model, not an ad-hoc lasso-style assumption.
* For ``l2`` (Shi-Wang A1-A2): factor strength
  :math:`\xi_N = \phi_{\min}(\boldsymbol\Lambda' \boldsymbol\Lambda
  / N)` may vanish (weak factors allowed); the idiosyncratic
  covariance has eigenvalues bounded in
  :math:`[\underline\sigma^2, \overline\sigma^2]`.

**A6 (Variant-specific structure of** :math:`\boldsymbol\beta^0`
**).**

* ``lasso``: **sparse** :math:`\boldsymbol\beta^0` -- only
  :math:`m = o(T_1)` of its components are non-zero.
* ``l2``: **dense** :math:`\boldsymbol\beta^0` -- almost no
  exact zeros (the factor projection gives every donor a
  small-but-nonzero coefficient).
* ``fs``: agnostic -- the inference is valid uniformly over a
  class of DGPs that includes **both** dense and sparse
  :math:`\boldsymbol\beta^0` (Theorem 1 in Shi-Huang).

**A7 (Inferential regularity).** For all three, the post-period
average effect :math:`\bar\Delta` has a CLT with HAC long-run
variance consistently estimable by Newey-West. For ``l2`` and
``lasso``, both the pre-period (first-stage) and post-period
HAC variances enter; for ``fs``, sample-splitting absorbs the
first-stage term and **only** the post-period HAC variance
enters.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) **Factor-driven DGP (A1).** PDA's whole identification story
    rides on the latent factor structure. If the panel is not
    well-described by a small number of common factors, the
    linear-projection equation has a non-vanishing
    :math:`\epsilon_t` term that the high-dimensional estimators
    cannot remove.

    *Plausibly violated when* donors are largely idiosyncratic
    (each follows its own unrelated process), or when one or two
    donors have a structural break that the factor model can't
    absorb. *Diagnostic*: run an SVD on the donor pre-period
    matrix; the top few singular values should carry the bulk of
    the spectral energy. If the spectrum is flat-tailed (no
    clear factor cutoff), the factor model fails and PDA's
    linear-projection consistency is fragile. In that regime,
    use a balancing-aware estimator (:doc:`microsynth` if you
    have unit-level data) or stay with canonical SC.

(b) **Single treated unit, absorbing aggregate-level treatment
    (A2).** Multiple treated units, treatment that turns off, or
    interference among donors break the framework.

    *Plausibly violated when* the policy is rescinded mid-
    sample, or when treated and donor states are spatially or
    economically linked enough that the donors' untreated
    trajectories shift. *Diagnostic*: the config validator
    enforces single-cohort; the silent failure is treated-donor
    spillover. Split donors by geographic / economic distance
    to the treated unit and refit; large ATE shifts flag
    interference. Use :doc:`spillsynth` or :doc:`spsydid` for
    genuine spillovers, *FECT* / :doc:`sdid` for
    staggered designs.

(c) **Weak temporal dependence (A3).** All three variants
    assume mixing or :math:`\rho`-mixing pre-period series with
    geometric decay. Unit-root outcomes, long-memory series, or
    series with structural breaks fail this.

    *Plausibly violated when* the outcome is a price level, a
    cumulative quantity, or an undifferenced macroeconomic
    series. *Diagnostic*: ADF / KPSS on the pre-period
    residuals; non-stationarity flags A3 failure. The pre/post
    asymptotic-independence story (which licenses fs-PDA's
    sample-splitting inference) is then in question. First-
    difference the outcome, or use :doc:`sbc` (a stationary-
    cycle estimator) before PDA.

(d) **Sample-size regime (A4).** PDA needs both :math:`T_1` and
    :math:`T_2` growing with :math:`\log N` small relative to
    each. A short post-period (:math:`T_2 \le 5`) breaks the
    CLT on :math:`\bar\Delta`; a short pre-period
    (:math:`T_1 \le 20`) breaks the moment-convergence rates.

    *Plausibly violated when* the pre-period is short with many
    donors. *Diagnostic*: compute
    :math:`(\log N) / T_1` and :math:`(\log N) / T_2`; both
    should be visibly below 1. If they are not, the asymptotic
    approximation has not kicked in. Either lengthen the panel
    (aggregate to a finer time grid), prune donors, or move to
    *canonical SCM* / :doc:`tssc` / :doc:`fdid` which work with
    shorter panels.

(e) **Donor regularity (A5).** Each variant has its own
    Gram-matrix / factor-strength condition. The practical
    common failure is **near-collinear donors**: two donor
    series that move together up to noise produce a near-
    singular pre-period Gram matrix.

    *Plausibly violated when* the donor pool contains
    near-duplicates (two adjacent states with essentially
    identical industry mix). *Diagnostic*: read the condition
    number of :math:`\Gamma_{\mathcal T_1}(\mathbf x_t,
    \mathbf x_t')`. A condition number above ~1e6 is a red
    flag. For ``lasso`` and ``fs`` this manifests as selection
    flipping between near-clones across seeds; for ``l2`` the
    tau-validation curve gets noisy. Prune one of each clone
    pair before refitting.

(f) **Coefficient structure (A6) -- choosing the right variant.**
    The biggest practitioner-side decision is whether to assume
    sparse or dense :math:`\boldsymbol\beta^0`. Choosing wrong
    pays a real cost: ``lasso`` on a dense truth over-selects
    and inflates size (see the Path-B table above:
    LASSO's size is 0.16-0.36 under the dense factor DGP, vs
    ``fs``'s 0.05); ``l2`` on a sparse truth pays variance for
    the dense fit it doesn't need.

    *Diagnostic*: fit ``fs`` first -- it's valid in both
    regimes per Shi-Huang Theorem 1, and the selected
    :math:`\hat R` and per-step :math:`R^2` curve tell you
    whether you're in a sparse (few donors carry the fit) or
    dense (many donors add information) regime. Then run the
    matched variant (``lasso`` if ``fs`` keeps a handful;
    ``l2`` if ``fs`` keeps many).

(g) **Inferential regularity (A7).** The HAC long-run variance
    must be consistently estimable. With strong serial
    correlation and a short post-period, the Bartlett /
    Newey-West kernel needs more lags than the post-period
    supports.

    *Plausibly violated when* :math:`T_2 \le 8` *and* the
    treatment-effect series is autocorrelated. *Diagnostic*:
    plot the autocorrelation function of ``res.gap[-T2:]``; if
    it stays high beyond :math:`\sqrt{T_2}` lags, the
    Newey-West bandwidth choice is binding and the CI is
    optimistic. Lengthen the post-window if you can, or report
    bootstrap CIs alongside the HAC ones.

When to use PDA -- and when not to
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Reach for PDA when:**

* You have a **single treated unit, a moderate-to-large donor
  pool** (:math:`N` comparable to or exceeding :math:`T_1`), and
  a plausibly factor-driven panel. PDA was designed for exactly
  this regime, and unlike canonical SCM, the projection has no
  simplex / non-negativity constraint -- it can extrapolate
  through negative coefficients on far donors when the factor
  structure demands it.
* You want **HAC-based, classical-statistics inference** on the
  ATE with a closed-form normal-distribution test, not a
  permutation or conformal procedure. PDA's CLT-based inference
  is what makes it the closest synthetic-control cousin to
  difference-in-differences from an inferential standpoint.
* The treated unit's pre-trajectory **lies in the linear span**
  (not necessarily convex hull) of the controls, so a
  no-constraint linear projection makes sense. PDA cannot
  recover an effect when the projection itself is impossible.
* You're between **dense** (``l2``) and **sparse** (``lasso``)
  regimes and want a **uniformly valid** test that doesn't
  require choosing the right sparsity story -- run ``fs``.

**Do not use PDA when:**

* **You need convex (non-negative, sum-to-one) weights** as a
  policy-interpretation deliverable. PDA's no-constraint
  projection produces negative coefficients on far donors,
  which is awkward to explain in many policy contexts. Use
  *canonical SCM* / :doc:`tssc` (canonical SC) or :doc:`fscm`
  (forward-selected SC with the simplex retained).
* **The treated unit is structurally outside the donor span
  (not just the convex hull).** PDA's linear projection cannot
  reach a treated outcome that no linear combination of donor
  outcomes can express. The pre-period RMSE stays large at
  every PDA variant. Use :doc:`iscm`, whose A2(b) mechanism
  identifies the effect through donors that use the treated
  unit as a positive-weight donor in *their* synthetic
  controls.
* **Outcomes are non-stationary** (unit-root or trending
  series). A3 fails and the pre/post asymptotic-independence
  story breaks. First-difference the outcome (the
  l2-relaxation paper's empirical PPI application does
  exactly this), or use :doc:`sbc` (stationary-cycle
  estimator).
* **You have multiple treated units** with overlapping cohorts.
  PDA's theorems (with the exception of the l2 relaxerm which does support multiple treated units but is not written yet) are written for the single-treated case. Use
 :doc:`sdid` for staggered adoption.
* **Spillovers across donors.** A2's no-interference clause
  fails when donor states are economically linked to the
  treated state. Use :doc:`spillsynth` or :doc:`spsydid`.
* **Continuous or multi-valued treatment.** PDA encodes a
  single binary intervention; continuous dose belongs in
  :doc:`ctsc`.
* **Distributional questions** (Lorenz curves, QTEs).
  PDA targets the mean ATE through a Gaussian-likelihood
  linear projection. Use :doc:`dsc` for distributional
  effects.
* **You need Bayesian posterior credible bands.** PDA returns
  frequentist HAC-based CIs. For Bayesian inference and
  posterior inclusion probabilities on donors, use
  :doc:`bvss` (spike-and-slab with a soft simplex).
* **Very short pre-period** :math:`(T_1 \le 15)` **with many
  donors.** The high-dimensional approximation has not kicked
  in; the selected :math:`\hat\beta` is noise. Use *canonical SCM*
  / :doc:`tssc` / :doc:`fdid`, which work without
  high-dimensional asymptotics.
* **Very short post-period** :math:`(T_2 \le 5)`. The CLT on
  :math:`\bar\Delta` is shaky; the HAC bandwidth choice
  dominates the inference. Either accept a wider permutation
  CI from *canonical SCM* / :doc:`tssc`, or move to the
  l2-relaxation multiple-treated-units extension (Shi-Wang
  Section 4.4) which is built for this regime.
* **You want predictor-level matching (covariates +
  pre-period outcomes), not outcome-only projection.** PDA's
  workhorse projection is on **donor outcomes**, not on
  predictor moments. Use *canonical SCM* / :doc:`tssc` /
  :doc:`sparse_sc` (predictor selection with L1 penalty on the
  V-weight matrix) for the predictor-matching setup.
* **The factor model itself is the object of interest** (you
  want to identify and interpret the factors). PDA is
  agnostic to factor estimation -- the factor model only
  motivates the linear projection, never enters the
  estimator. Use :doc:`fma` (factor-model-aware estimator) if
  recovering factors is the goal.

Empirical Illustration: Hong Kong economic integration
-------------------------------------------------------

The original HCW [HCW]_ application -- and Shi & Huang's Example 1 -- evaluates
the effect of economic integration with mainland China on Hong Kong's quarterly
real-GDP growth, using 24 comparison economies. Here all three PDA variants run
on the same data, and the package returns the time-varying effect, the ATE, and
a HAC confidence interval for each.

.. code-block:: python

   import pandas as pd
   from mlsynth import PDA

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/HongKong.csv"
   df = pd.read_csv(url)

   res = PDA({"df": df, "outcome": "GDP", "treat": "Integration",
              "unitid": "Country", "time": "Time",
              "methods": ["l2", "LASSO", "fs"], "alpha": 0.05,
              "display_graphs": True}).fit()

   for name, fit in res.fits.items():
       sel = "-" if fit.selected_donors is None else len(fit.selected_donors)
       print(f"{name:6s} ATE {fit.att:.4f}  SE {fit.att_se:.4f}  "
             f"95% CI ({fit.ci[0]:.4f}, {fit.ci[1]:.4f})  p={fit.p_value:.3f}  donors={sel}")

This prints::

   l2     ATE 0.0248  SE 0.0032  95% CI (0.0185, 0.0311)  p=0.000  donors=24
   lasso  ATE 0.0330  SE 0.0054  95% CI (0.0224, 0.0436)  p=0.000  donors=11
   fs     ATE 0.0285  SE 0.0059  95% CI (0.0169, 0.0401)  p=0.000  donors=7

All three find a **significant positive integration effect** on Hong Kong's
GDP growth -- roughly +2.5 to +3.3 percentage points -- differing only by their
selection philosophy: ``l2`` keeps all 24 controls with dense, signed
coefficients (pre-RMSE 0.013); ``lasso`` keeps 11; ``fs`` keeps a parsimonious
7 (Malaysia, New Zealand, Norway, Austria, Canada, Thailand, Australia). The
estimates bracket the Forward-DiD result on the same data (0.025), a useful
cross-method check.

Verification
------------

.. note::

   **Empirical (Path A, Hong Kong).** All three variants run on the HCW Hong
   Kong panel (above) and agree on a significant positive integration effect,
   consistent with the literature and the Forward-DiD cross-check (0.025).

Simulation study (Path B): forward selection vs LASSO
-----------------------------------------------------

Shi & Huang's (2023) Table 1 compares forward selection against LASSO on a
four-factor DGP, re-implemented in
:func:`mlsynth.utils.pda_helpers.simulation.simulate_pda_panel`: four factors
(``f1`` i.i.d.; ``f2`` AR(1) 0.9; ``f3`` MA(2) (0.8,0.4); ``f4`` ARMA(1,1)
(0.5,0.5) under the *dynamic* structure, all i.i.d. ``N(0,1)`` under the
*i.i.d.* structure); loadings ``U(1,2)`` on the treated + 4 relevant controls
and ``U(-0.1,0.1)`` on the remaining 96; idiosyncratic ``N(0,0.5)``; one treated
unit, :math:`N=100` controls, :math:`T_1=T_2`. Shocks ``D1``-``D7`` set the
post-period ATE (``D1``-``D3`` null → *size*; ``D4``-``D7`` non-zero → *power*).
Driving the **packaged** ``PDA`` (``methods=["fs","LASSO"]``,
``fs_intercept=False``) at :math:`T_1=100` (200 reps for size, 60 for power):

.. list-table:: forward selection vs LASSO, :math:`T_1=100`
   :header-rows: 1
   :widths: 12 8 10 10 10

   * - factors
     - method
     - # donors
     - size (D1)
     - power (D5)
   * - i.i.d.
     - fs
     - 3.9
     - 0.090
     - 1.00
   * - i.i.d.
     - LASSO
     - 9.5
     - 0.065
     - 1.00
   * - dynamic
     - fs
     - 4.5
     - 0.075
     - 1.00
   * - dynamic
     - LASSO
     - 15.0
     - 0.140
     - 1.00

The paper's geometry reproduces. **Forward selection is parsimonious** -- it
keeps to ~the 4 relevant donors in both structures -- while **LASSO
over-selects** (9-15 donors). **Forward selection's test is correctly sized**
(≈ 0.05-0.09) under *both* factor structures, the robustness Shi & Huang
emphasise. **LASSO is correctly sized under i.i.d. factors** (0.065, matching
the paper's 0.058) but its **size inflates under dynamic factors** (0.140;
paper's modified-BIC LASSO 0.184) -- the size inflation the paper reports is a
*dynamic-factor* phenomenon, not an i.i.d. one. Both tests are fully powered at
``D5`` (mean-1 shift). Durable case: ``pda_table1``.

.. note::

   **mlsynth's LASSO is cross-validated; the paper's is not.** Shi & Huang
   select the Lasso penalty with a **modified BIC** (Remark 4 cont., p.521:
   "we tune the constants in the modified BIC to allow Lasso to take in more
   variables"); ``mlsynth``'s L1-PDA selects it with ``LassoCV`` (5-fold
   cross-validation). The two are different penalty rules, so the LASSO cells
   above are ``mlsynth``'s CV variant, not a cell-by-cell match of the paper's
   Lasso. What both share -- and what the benchmark pins -- is the geometry:
   LASSO over-selects relative to fs and its size inflates under dynamic
   factors, while forward selection (the paper's *method*, validated cell-by-
   cell on Hong Kong in ``pda_hongkong``) stays parsimonious and correctly
   sized.

.. admonition:: The ``fs_intercept`` knob -- valid size on factor data

   Achieving the correct fs size above required a fix. The released ``fsPDA``
   R package (``est.fsPDA.R``) fits the donor regression *with* an intercept;
   on the paper's mean-zero factor DGP that intercept absorbs a spurious
   pre-period constant which extrapolates into the post window, biasing the
   gap and **inflating the null rejection rate to ~0.20**. The paper's Table 1
   was produced by the *simulation* code (``FS.R``), which fits **without** an
   intercept and yields valid size. ``mlsynth`` exposes both via
   ``PDAConfig.fs_intercept`` (default ``False`` = the no-intercept, valid-size
   form; set ``True`` for panels with genuine unit level shifts). With the
   default, the fs D1 rejection rate drops from ~0.20 (intercept) to the
   ~0.05-0.09 reported above (no intercept).

   One honest caveat: under *dynamic* factors fs shows mild residual size
   inflation at small :math:`T` (the "imprecise long-run-variance" effect the
   paper itself notes), shrinking toward 5% as :math:`T_1 \to \infty`.

**L2-relaxation (Shi & Wang).** The ``l2`` method's out-of-sample MPSE falls
with :math:`T` and its test approaches the nominal 5% size as :math:`T_1 \to
\infty`, matching Shi & Wang's Table 2 (size :math:`0.142` at :math:`T_1=50`
→ :math:`0.072` at :math:`200`). Its per-fit cross-validation over the
:math:`\tau` grid makes large Monte Carlos expensive (~5 s/fit), so the full
table is summarized rather than swept here.

Core API
--------

.. automodule:: mlsynth.estimators.pda
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PDAConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``PDA.fit()`` returns a
:class:`~mlsynth.utils.pda_helpers.structures.PDAResults`, whose ``fits`` maps
each variant to a
:class:`~mlsynth.utils.pda_helpers.structures.PDAMethodFit` (coefficients,
intercept, counterfactual, gap, ATE, HAC standard error, CI, p-value, donor
weights, and the selected-donor list for ``lasso``/``fs``). The prepared,
NumPy-only panel is exposed as a
:class:`~mlsynth.utils.pda_helpers.structures.PDAInputs`, with units and time
addressed through an :class:`IndexSet`.

.. note::

   ``PDA.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract. It is a dispatcher over the variants in
   ``res.fits`` (l2 / lasso / fs); the selected variant drives the flat
   accessors (``res.att`` / ``res.att_ci`` / ``res.counterfactual`` /
   ``res.gap`` / ``res.donor_weights`` / ``res.pre_rmse``), which resolve
   through the standardized sub-models. ``res.donor_weights`` are the regression
   coefficients (PDA is a regression counterfactual, not a simplex average);
   ``res.att_by_method()`` / ``res.se_by_method()`` report every variant.

.. automodule:: mlsynth.utils.pda_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

Helper Modules
--------------

Data preparation -- the only DataFrame touchpoint: pivots to NumPy, builds the
unit/time ``IndexSet``es, and splits pre/post.

.. automodule:: mlsynth.utils.pda_helpers.setup
   :members:
   :undoc-members:

Shared HAC long-run-variance machinery (Bartlett/Newey-West) and the N(0,1)
test used by all three variants.

.. automodule:: mlsynth.utils.pda_helpers.inference
   :members:
   :undoc-members:

L2-relaxation (Shi & Wang): the relaxation primal, tau validation, and the
two-term HAC ATE inference.

.. automodule:: mlsynth.utils.pda_helpers.l2.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pda_helpers.l2.inference
   :members:
   :undoc-members:

LASSO (Li & Bell): cross-validated L1 estimation and the HAC t-test with a
first-stage variance term.

.. automodule:: mlsynth.utils.pda_helpers.lasso.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pda_helpers.lasso.inference
   :members:
   :undoc-members:

Forward selection (Shi & Huang): greedy R^2 selection with modified-BIC
stopping and the post-selection HAC t-test.

.. automodule:: mlsynth.utils.pda_helpers.fs.estimation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pda_helpers.fs.inference
   :members:
   :undoc-members:

Run loop assembling the per-variant fits.

.. automodule:: mlsynth.utils.pda_helpers.orchestration
   :members:
   :undoc-members:

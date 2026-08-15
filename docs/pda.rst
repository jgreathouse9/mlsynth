Panel Data Approach (PDA)
=========================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The panel data approach (PDA) of Hsiao, Ching and Wan [HCW]_ estimates a
single treated unit's untreated counterfactual by a linear regression on
the control units, fit over the pre-treatment window and extrapolated
out-of-sample. Unlike the synthetic control method, PDA imposes no
constraints on the coefficients (no simplex, no non-negativity), so it is a
plain projection justified by a latent-factor model: if every unit loads on a
small set of common factors, the treated unit's untreated path is a linear
combination of the controls' paths plus an orthogonal error.

The challenge is which controls and how many. Classical PDA was built
for low dimensions (few controls relative to pre-periods) and chooses controls
by AIC/BIC, which break down once the number of controls ``N`` approaches or
exceeds the pre-period length ``T0``. ``mlsynth`` packages the original
Hsiao-Ching-Wan method together with three high-dimensional variants that
resolve the ``N``-near-``T0`` problem differently, each with the estimation and
inference theory of its own paper:

* Original best-subset (``hcw``; Hsiao, Ching & Wan [HCW2012]_) -- the method
  that started the literature. The counterfactual is an *unrestricted* OLS
  regression (with intercept) of the treated pre-period on the **best subset**
  of controls, the subset and its size chosen by AICc (also AIC / BIC). Built
  for the low-dimensional regime (a moderate, pre-screened candidate pool); the
  best-subset search is combinatorial, so cap ``hcw_nvmax`` or pre-restrict the
  pool for large ``N``. The three estimators below are its scalable descendants.
* L2-relaxation (``l2``; Shi & Wang [l2relax]_) -- a *dense* estimator (a
  "cousin of ridge") for when the factor model makes the projection
  coefficients dense; tolerates ``N > T0``; prediction is robust to
  heteroskedasticity.
* LASSO (``lasso``; Li & Bell [LASSOPDA]_) -- an L1 estimator that
  *selects* a sparse set of relevant controls; computationally far cheaper
  than AIC/BIC and works for ``N > T0``.
* Forward selection (``fs``; Shi & Huang [fsPDA]_) -- a greedy procedure
  that grows the control set one unit at a time, with valid post-selection
  inference and no sparsity requirement (works for dense *or* sparse models).
  The scalable replacement for ``hcw``'s combinatorial best-subset search.

All variants target the single-treated-unit, many-candidate-controls regime
and produce a time-varying treatment effect and an average treatment effect
(ATE) with a HAC-based confidence interval. The practical choice among them
(detailed below) follows the authors' own arguments: ``l2`` when the
coefficients are dense, ``lasso`` when only a few controls matter and you want
an interpretable selection, ``fs`` when you want a cheap, predictive ensemble
with honest post-selection inference regardless of sparsity.

Notation
--------

We use mlsynth's standard notation, with the two sample operators of Shi & Wang
[l2relax]_ defined below. For a positive integer ``n``,
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

The two time-series operators are central. Over an index set
:math:`\mathcal{S}` of periods,

.. math::

   \mathcal{E}_{\mathcal{S}}(\mathbf{x}_t) = \frac{1}{|\mathcal{S}|}
       \sum_{t\in\mathcal{S}} \mathbf{x}_t,
   \qquad
   \Gamma_{\mathcal{S}}(\mathbf{x}_t, \mathbf{y}_t') = \mathcal{E}_{\mathcal{S}}
       \Bigl( [\mathbf{x}_t - \mathcal{E}_{\mathcal{S}}(\mathbf{x}_t)]
              [\mathbf{y}_t - \mathcal{E}_{\mathcal{S}}(\mathbf{y}_t)]' \Bigr),

the sample mean and sample covariance of a series over :math:`\mathcal{S}`.
The intervention takes effect after the split point :math:`T_0`. The pre-period
is :math:`\mathcal{T}_1 \coloneqq \{1, \ldots, T_0\}` with
:math:`|\mathcal{T}_1| = T_0`; the post-period is
:math:`\mathcal{T}_2 \coloneqq \{T_0+1, \ldots, T\}` with
:math:`T_2 \coloneqq |\mathcal{T}_2| = T - T_0`. Let :math:`j=1` denote the
treated unit, with treated series
:math:`\mathbf{y}_1 = (y_{11}, \ldots, y_{1T})^\top`; the donor pool is
:math:`\mathcal{N}_0 \coloneqq \mathcal{N}\setminus\{1\}`, with cardinality
:math:`N_0`, where :math:`\mathcal{N} = \{1,\ldots,N\}`.

The shared model
^^^^^^^^^^^^^^^^

All four methods rest on a common latent-factor data-generating process: for
:math:`t \in \mathcal{T}`,

.. math::

   y_{jt}^N = \mu_j + \boldsymbol{\lambda}_j' \mathbf{f}_t + u_{jt},
   \quad j \in \{1\}\cup\mathcal{N}_0,

with :math:`\mathbf{f}_t` an :math:`r`-vector of latent common factors,
:math:`\boldsymbol{\lambda}_j` factor loadings, and :math:`u_{jt}` a
weakly-dependent idiosyncratic error orthogonal to the factors. Because the
common factors drive both the treated unit and the donors, the untreated
treated outcome admits a linear projection on the donors,

.. math::

   y_{1t} = \alpha^0 + \mathbf{x}_t' \boldsymbol{\beta}^0 + \epsilon_t,
   \qquad \mathbf{x}_t = (y_{jt})_{j\in\mathcal{N}_0}',

with :math:`\mathbb{E}[\mathbf{x}_t \epsilon_t] = \mathbf{0}`. PDA fits
:math:`(\alpha^0, \boldsymbol{\beta}^0)` on :math:`\mathcal{T}_1` and predicts
:math:`\widehat{y}_{1t} = \mathcal{E}_{\mathcal{T}_1}(y_{1s}) + [\mathbf{x}_t -
\mathcal{E}_{\mathcal{T}_1}(\mathbf{x}_s)]'\widehat{\boldsymbol{\beta}}` for
:math:`t \in \mathcal{T}_2`. The per-period treatment effect is
:math:`\tau_t = y_{1t} - \widehat{y}_{1t}` and the ATE
:math:`\widehat{\tau} = \mathcal{E}_{\mathcal{T}_2}(\tau_t)`. The methods
differ in how they estimate :math:`\boldsymbol{\beta}` and in the
inference theory each paper proves for :math:`\widehat{\tau}`.

Original best subset (``hcw``, Hsiao-Ching-Wan)
-----------------------------------------------

Idea. This is the method that started the literature [HCW2012]_. The recipe is
the plainest one in the family -- ordinary least squares of the treated unit's
pre-period outcome on a handful of control series, extrapolated past the
intervention -- with one twist: *which* controls, and *how many*, are chosen for
you, not fixed in advance. HCW's Section 5 turns that choice into a
model-selection problem. It helps to walk through it as four steps.

Step 1 -- measure how well a candidate set of controls fits. Pick any subset
:math:`\mathcal{S}\subseteq\mathcal{N}_0` of the donors. Regress the treated
pre-period outcome on those donors and an intercept by OLS, and record the
residual sum of squares -- the total squared pre-period miss,

.. math::

   \mathrm{RSS}(\mathcal{S}) = \min_{\alpha,\,\boldsymbol{\beta}_{\mathcal{S}}}
       \sum_{t\in\mathcal{T}_1}
       \bigl(y_{1t} - \alpha - \mathbf{x}_{t,\mathcal{S}}'
             \boldsymbol{\beta}_{\mathcal{S}}\bigr)^2 ,
   \qquad \mathbf{x}_{t,\mathcal{S}} = (y_{jt})_{j\in\mathcal{S}}.

A smaller :math:`\mathrm{RSS}` means a tighter pre-period fit. By itself it is
not a usable score, though: adding *any* control can only lower
:math:`\mathrm{RSS}`, so "smallest RSS" always points at the largest possible
model, which memorises the short pre-period and extrapolates badly past the cut.

Step 2 -- charge for complexity with an information criterion. To stop the
search from simply taking every donor, each subset is scored by an *information
criterion*: the (log) fit plus a penalty that grows with the number of
parameters. The default is the small-sample-corrected AICc (AIC and BIC are the
same shape with a lighter or heavier penalty),

.. math::

   \mathrm{AICc}(\mathcal{S}) = \underbrace{T_0 \log\!\frac{\mathrm{RSS}
       (\mathcal{S})}{T_0}}_{\text{rewards fit}}
       + \underbrace{2K + \frac{2K(K+1)}{T_0 - K - 1}}_{\text{charges for size}},
   \qquad K = r + 2,\;\; r = |\mathcal{S}|.

Lower is better. The first term falls as the fit improves; the second rises with
the model size :math:`r`, so the criterion bottoms out where one more donor stops
paying for the variance it adds -- the bias-variance trade-off turned into a
single number. The penalised parameter count :math:`K = r + 2` counts the
:math:`r` donors, the intercept, and the error variance (the convention of the
``pampe`` R package, ``leaps::regsubsets`` + AICc + ``lm``); the small-sample
correction :math:`2K(K+1)/(T_0-K-1)` matters because PDA's pre-period is short.

Step 3 -- keep the subset with the best score. The selected controls are the
global minimiser

.. math::

   \widehat{\mathcal{S}} = \operatorname*{argmin}_{\mathcal{S}\subseteq
       \mathcal{N}_0,\; |\mathcal{S}|\le r_{\max}} \mathrm{AICc}(\mathcal{S}).

In practice this is computed the way ``leaps`` does it (HCW Section 5): for each
model size :math:`r` find the *single* lowest-:math:`\mathrm{RSS}` subset of that
size, then choose the size whose best subset minimises the criterion -- a "best
of each size, then choose the size" search. The cap :math:`r_{\max}`
(``hcw_nvmax``, pampe's ``nvmax``) limits the largest size considered and cannot
exceed the pre-period degrees of freedom (the criterion is undefined once the
donors plus intercept use up the :math:`T_0` observations).

Step 4 -- refit and extrapolate. Re-fit OLS on the chosen controls
:math:`\widehat{\mathcal{S}}` over the pre-period, then carry the coefficients
forward: the counterfactual is that fitted line evaluated on the post-period
control values, and the per-period effect is treated minus counterfactual,
exactly as in the shared model above. On HCW's Hong Kong sovereignty study this
selects :math:`\{\text{Japan},\text{Korea},\text{Taiwan},\text{USA}\}` with
:math:`\mathrm{AICc} = -171.771`, reproducing Table XVI value-for-value
(cross-validated against ``pampe`` -- see the benchmark below).

Best subset selection is HCW's classical, low-dimensional choice: AICc / AIC /
BIC are only defined while :math:`r < T_0`, and the search is combinatorial in
:math:`N_0`. HCW therefore pre-screened the donor pool (limiting Hong Kong to
ten candidate economies). [HTT2020]_ place best subset alongside its two cheaper
relatives -- forward stepwise and the lasso -- and show forward stepwise tracks
best subset closely, which is exactly why ``fs`` is ``hcw``'s scalable
descendant; ``lasso`` and ``l2`` are the others. Use ``hcw`` when the candidate
pool is moderate and pre-screened, when you want the original method or to
reproduce HCW / ``pampe``, and when an exact, certified optimum is the point.

Computing the best subset: the optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are :math:`2^{N_0}` candidate subsets, each an OLS fit, so naive
enumeration is the regime where even ``leaps`` strains. ``mlsynth`` solves the
problem exactly with a branch-and-bound built on three ideas, plus an optional
fourth that hands the problem to a mixed-integer solver.

*Furnival-Wilson sweep engine.* The default is the canonical
``leaps::regsubsets`` algorithm [FurnivalWilson]_: a depth-first search over
subsets driven by the symmetric sweep operator on the augmented cross-product
matrix :math:`[\mathbf{1},\mathbf{X},\mathbf{y}]'
[\mathbf{1},\mathbf{X},\mathbf{y}]`. Sweeping a donor in leaves the running
:math:`\mathrm{RSS}` in the response diagonal in :math:`O(p^2)`, and the reverse
sweep removes it on backtracking, so no OLS is ever re-solved from scratch. Each
subtree is bounded below by the smallest :math:`\mathrm{RSS}` it can reach
(include *every* remaining donor -- :math:`\mathrm{RSS}` is monotone in the
variable set) paired with the smallest information-criterion penalty any
descendant can carry; when that lower bound cannot beat the incumbent, the
whole subtree is pruned. The bound is a true lower bound, so the returned subset
is the exact global optimum -- identical to exhaustive enumeration, at a small
fraction of the cost.

*Discrete first-order warm start.* The incumbent is seeded, at every model
size, with the Bertsimas-King-Mazumder projected-gradient hard-thresholding
method [BKM2016]_ (their Algorithm 1): iterate
:math:`\boldsymbol{\beta}\leftarrow H_r\!\bigl(\boldsymbol{\beta} - \tfrac{1}{L}
\nabla g(\boldsymbol{\beta})\bigr)`, where :math:`H_r` keeps the :math:`r`
largest-magnitude coordinates and :math:`L` upper-bounds the largest eigenvalue
of the centred donor Gram, with a least-squares refit on the active set and
several restarts. A near-optimal incumbent tightens the bound from the outset
and prunes more subtrees; because the seed can only lower the incumbent, the
certified optimum is unchanged.

*Node budget and a certified optimality gap.* For a pool too large to certify
quickly, the search stops at a node budget and returns the best incumbent found
together with a valid lower bound -- the smallest subtree bound it never reached.
The reported :math:`\text{optimality gap} = \text{incumbent} - \text{lower
bound}` is then a genuine suboptimality certificate (zero means provably
optimal), the Bertsimas-King-Mazumder gap obtained from the branch-and-bound's
own bounds with no solver. This replaces an outright refusal of large pools:
``fits["hcw"].metadata`` reports ``certified_optimal`` and ``optimality_gap``.

*Optional exact mixed-integer backend.* Setting ``hcw_backend="scip"`` certifies
the optimum at pool sizes beyond the branch-and-bound's exact reach via the SCIP
solver. It casts the size-:math:`r` problem as the Bertsimas-King-Mazumder
cardinality-constrained least squares -- binary indicators :math:`z_i`, an SOS-1
coupling forcing :math:`z_i = 0 \Rightarrow \beta_i = 0`, and
:math:`\sum_i z_i \le r` -- solved once per size, then chooses the size by the
criterion. The dependency is optional: ``pyscipopt`` is imported only on demand
(``pip install mlsynth[scip]``). The default ``fw`` backend needs no solver and
is all the low-dimensional regime ``hcw`` targets requires.

Assumptions and inference. ``hcw`` shares the identifying stack documented
under :ref:`Shared assumptions across the PDA class <pda-shared-assumptions>`
(the latent factor model A1, single absorbing treatment A2, weak temporal
dependence A3), specialised to the low-dimensional regime :math:`N_0 < T_0` that
makes the information criteria well-defined. Inference mirrors the post-selection
HAC machinery of the scalable variants: the average effect carries a
Newey-West / Bartlett long-run-variance confidence interval (prewhitened by
default; ``lrvar_lag`` switches to a fixed-lag Bartlett estimator), and
``prediction_intervals=True`` attaches the Jiang et al. per-period intervals
described below.

L2-relaxation (``l2``, Shi & Wang)
----------------------------------

Idea. Under the factor model the projection coefficient
:math:`\boldsymbol{\beta}^0 = \boldsymbol{\Omega}^{-1}\boldsymbol{\Lambda}
(\boldsymbol{\Lambda}'\boldsymbol{\Omega}^{-1}\boldsymbol{\Lambda} +
\mathbf{I}_q)^{-1}\boldsymbol{\lambda}_1` is dense in general -- almost no
entries are exactly zero. Sparse methods (LASSO) are then mis-matched. With
:math:`\widehat{\boldsymbol{\Sigma}} = \Gamma_{\mathcal{T}_1}(\mathbf{x}_t,
\mathbf{x}_t')` and :math:`\widehat{\boldsymbol{\eta}} =
\Gamma_{\mathcal{T}_1}(\mathbf{x}_t, y_t)`, OLS solves the KKT condition
:math:`\widehat{\boldsymbol{\Sigma}}\boldsymbol{\beta} = \widehat{\boldsymbol{\eta}}`,
which is unstable or non-unique once :math:`N` is close to or exceeds
:math:`T_0`. L2-relaxation relaxes the sup-norm of this moment condition by
a tuning parameter :math:`\varepsilon` and minimizes the coefficient norm:

.. math::

   \min_{\boldsymbol{\beta}} \tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2
   \quad \text{s.t.} \quad
   \|\widehat{\boldsymbol{\eta}} - \widehat{\boldsymbol{\Sigma}}\boldsymbol{\beta}\|_\infty
   \le \varepsilon.

This is the "bias-variance trade-off" made explicit: tolerating a small
violation :math:`\varepsilon` of the OLS moment condition shrinks the variance. At
:math:`\varepsilon = 0` it reduces to (ridgeless) OLS; at :math:`\varepsilon \ge
\|\widehat{\boldsymbol{\eta}}\|_\infty` it gives :math:`\boldsymbol{\beta} =
\mathbf{0}`. ``mlsynth`` picks :math:`\varepsilon` by sequential out-of-sample
validation on the tail of the training window (the validated :math:`\varepsilon`
tracks the infeasible-optimal one, and both shrink toward zero as the sample
grows) over a log-spaced grid down to :math:`10^{-4}\max|\widehat{\boldsymbol{\eta}}|`
(the optimum is often a tiny fraction of the cap). This is time-respecting --
the fit never sees periods later than the validation tail -- unlike the
released ``L2relax.CV``, whose 5-block K-fold trains on both past *and* future
of each block.

Solving the grid, and reading the solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The grid and the fit ask the solver for different things. The fit is the
estimate that gets reported, so it is solved to :math:`10^{-9}`; the grid only
has to *order* its candidates by validation error, and its winner is refit
afterwards. That distinction is most of the runtime of an ``l2`` fit. As
:math:`\varepsilon` shrinks the feasible set thins toward the moment condition
:math:`\widehat{\boldsymbol{\Sigma}}\boldsymbol{\beta} =
\widehat{\boldsymbol{\eta}}`, and the small-:math:`\varepsilon` end of the grid
drives the ADMM solver to its iteration cap: at :math:`N = 400` donors on
:math:`T_0 = 150` pre-periods, an eighty-point grid takes 271 s at
:math:`10^{-9}` and 25 s at :math:`10^{-6}`, and picks the same
:math:`\varepsilon`.

The solver also answers with a status, and that status has to be read. On a
pre-period where :math:`\widehat{\boldsymbol{\Sigma}}` is near-singular -- 24
Hong Kong donors on 24 periods, so its rank is 23 -- OSQP returns a primal
infeasibility certificate at the small-:math:`\varepsilon` end. The certificate
is false there, since the smallest reachable :math:`\|\widehat{\boldsymbol{\eta}}
- \widehat{\boldsymbol{\Sigma}}\boldsymbol{\beta}\|_\infty` is about
:math:`10^{-12}`, well below the :math:`\varepsilon` it fires at. Either way a
certificate is not a solution to the problem that was posed, and its vector is
finite -- entries of order :math:`10^{9}` on that panel -- so a finiteness check
alone admits it into the validation ranking. ``mlsynth`` keeps the statuses that
carry a primal iterate, imprecise ones included, and refuses the rest; an
:math:`\varepsilon` with no fit drops out of the ranking, and a grid with no fit
anywhere raises.

.. note::

   Standardisation. Following the authors' released ``L2relax``, the treated
   and control series are standardised (demeaned and scaled to unit
   variance) before forming :math:`\widehat{\boldsymbol{\Sigma}}` /
   :math:`\widehat{\boldsymbol{\eta}}`, and the solution is mapped back to the
   original scale. This is the default (``l2_standardize=True``) -- the
   :math:`\ell_2` penalty is scale-sensitive, so standardisation is both
   recommended and what reproduces the paper's empirical results; on the Hong
   Kong panel it moves the L2 estimate from :math:`2.48\%` to :math:`2.61\%`
   (closer to Shi & Wang's :math:`2.65\%`). Set ``l2_standardize=False`` for the
   raw-scale variant.

Assumptions (Shi & Wang).

*Assumption 1 (loadings).* :math:`\|\boldsymbol{\lambda}_1\|_\infty +
\|\boldsymbol{\Lambda}\|_\infty \le C`, and there is an :math:`r`-unit subset
making :math:`\boldsymbol{\Lambda}_{\mathcal{Q}\cdot}` full column rank; the
average factor strength :math:`\xi_N = \phi_{\min}(\boldsymbol{\Lambda}'
\boldsymbol{\Lambda}/N)` may vanish (weak factors allowed).

*Remark.* Bounded loadings keep the projection coefficient well-defined, while
the full-rank subset is what lets the donors span the treated unit's factor
exposure; allowing :math:`\xi_N \to 0` means the method tolerates weak factors
that contribute little cross-sectional variation.

*Assumption 2 (errors).* The idiosyncratic covariance
:math:`\boldsymbol{\Omega}` has eigenvalues bounded between
:math:`\underline\sigma^2` and :math:`\overline\sigma^2`; errors may be
heteroskedastic and cross-sectionally dependent.

*Remark.* The two-sided eigenvalue bound rules out a degenerate noise
direction, but heteroskedastic, cross-sectionally dependent errors are
permitted -- the realistic case for economic panels.

*Assumption 3-4 (sampling).* In- and out-of-sample sampling errors of the
sample moments are :math:`O_p(T_0^{-1/2})` for low-dimensional pieces and
:math:`O_p(\sqrt{\log N / (N\wedge T_0)})` for high-dimensional ones (holding
under time-series weak dependence, not just i.i.d.).

*Remark.* These are the convergence rates that make the relaxed moment
condition usable in high dimension: the high-dimensional pieces pay only a
:math:`\sqrt{\log N}` price, so :math:`N` may grow with (or beyond)
:math:`T_0`.

*Assumption 5 (ATE inference).* The oracle prediction error
:math:`\epsilon_t^*` and the effect-plus-error :math:`d_t^* = \Delta_t -
\mathbb{E}[\Delta_t] + \epsilon_t^*` have finite, positive long-run variances
:math:`\rho^2_{\epsilon^*}`, :math:`\rho^2_{d^*}` consistently estimable by HAC,
and a sequential CLT applies.

*Remark.* The coefficient estimator is consistent for the oracle target
(Theorem 1) and the prediction error is asymptotically
*irrelevant to heteroskedasticity* (Theorem 2): unlike the coefficient MSE,
the out-of-sample MSE does not depend on the noise heterogeneity
:math:`\psi_{\max}`. This is what licenses the HAC-based CLT for the ATE.

Inference (Shi & Wang, Theorem 3; single treated unit). With pre-period
prediction residuals :math:`e_t = y_{1t} - \widehat{y}_{1t}` (:math:`t\in\mathcal{T}_1`)
and post-period effects :math:`\tau_t` (:math:`t\in\mathcal{T}_2`),

.. math::

   \widehat{Z} = \frac{\widehat{\tau} - \Delta_{\mathcal{T}_2}}
       {\sqrt{\widehat{\rho}^2_{(1)}/T_0 + \widehat{\rho}^2_{(2)}/T_2}}
   \xrightarrow{d} N(0,1),

where :math:`\widehat{\rho}^2_{(1)}` is the HAC long-run variance of the
pre-period residuals (first-stage estimation uncertainty) and
:math:`\widehat{\rho}^2_{(2)}` is the HAC long-run variance of the de-meaned
post-period effects (post-period averaging). Both sources of uncertainty
enter, which matters when :math:`T_0` and :math:`T_2` are comparable.

When to use. Dense, factor-driven coefficients; high dimension
(:math:`N>T_0` permitted); when prediction accuracy and heteroskedasticity-
robustness matter more than identifying a handful of controls.

LASSO (``lasso``, Li & Bell)
----------------------------

Idea. When only a *few* controls are truly relevant, an L1 penalty selects
them and shrinks the rest. Li & Bell estimate

.. math::

   \widehat{\boldsymbol{\beta}}^{\text{las}}
   = \operatorname*{argmin}_{\boldsymbol{\beta}} \;
     \sum_{t\in\mathcal{T}_1} (y_{1t} - \mathbf{x}_t'\boldsymbol{\beta})^2
     + \lambda \sum_{j} |\beta_j|,

with :math:`\lambda` chosen by (leave-one-out) cross-validation, then predict
the counterfactual as in the shared model. LASSO works for :math:`N > T_0`
(where AIC/AICC/BIC cannot even be computed) and is far cheaper.

Assumptions (Li & Bell). They *relax* HCW's linear-conditional-mean
assumption and drop one of HCW's identification conditions. The key conditions
are: a factor model with :math:`\mathrm{Rank}(\widetilde{B}) = K`
(enough independent factor variation among the controls); a weakly dependent,
weakly stationary panel so laws of large numbers and CLTs apply to partial
sums; consistency of the pre-period least-squares pieces
(:math:`\widehat{\delta}_1 - \delta_1, \widehat{\delta}-\delta = O_p(T_0^{-1/2})`); and
:math:`\rho`-mixing with geometric decay plus a finite limit
:math:`\eta = \lim T_2/T_0`. Sparsity (only :math:`m` of
:math:`\boldsymbol{\beta}` non-zero, :math:`m` fixed or :math:`o(T_0)`) is
assumed for the high-dimensional selection.

*Remark.* Li & Bell prove consistency :math:`\widehat{\tau} - \Delta_{\mathcal{T}_2} =
O_p(T_0^{-1/2} + T_2^{-1/2})` (estimation error has two parts: first-stage
:math:`O_p(T_0^{-1/2})` and post-averaging :math:`O_p(T_2^{-1/2})`), holding
even when :math:`y_{1t}` is trend-stationary.

Inference (Li & Bell, Theorem 3.2). With :math:`\widehat{\Sigma} =
\widehat{\Sigma}_1 + \widehat{\Sigma}_2`,

.. math::

   \text{T.S.} = \frac{\sqrt{T_2}\,\widehat{\tau}}{\sqrt{\widehat{\Sigma}}}
   \xrightarrow{d} N(0,1),

where :math:`\widehat{\Sigma}_2` is the Newey-West HAC long-run variance of the
post-period effects, and :math:`\widehat{\Sigma}_1` is the first-stage
(pre-period estimation) variance -- the OLS prediction variance of the mean
post-period counterfactual on the selected support. Li & Bell note
:math:`\widehat{\Sigma}_1` is negligible when :math:`T_0 \gg T_2`, so the
post-period term dominates in long-pre-period panels. When :math:`T_0` is
comparable to :math:`T_2` it is not negligible at all: on Shi & Huang's
Monte Carlo design, where :math:`T_0 = T_2`, it accounts for 44-46% of
:math:`\widehat{\Sigma}` and inflates the standard error by about a third.

Supplying ``lrvar_lag`` selects a different test -- Shi & Huang's, the one their
``lasso.BIC`` computes:

.. math::

   Z = \frac{\widehat{\tau}}{\sqrt{\widehat{\omega}(h) / T_2}},

with :math:`\widehat{\omega}(h)` the Bartlett long-run variance of the
post-period effects at the truncation lag :math:`h` and no first-stage term.
Pair it with ``lasso_criterion="mbic"`` to reproduce ``lasso.BIC`` end to end;
mlsynth's :math:`Z` then matches theirs to :math:`7\times 10^{-6}` on the panels
carried in the ``fspda_dense_mc`` benchmark bundle.

When to use. A genuinely sparse set of relevant controls; very large
:math:`N` (even :math:`N/T_0 \to \infty`); when an interpretable, computa-
tionally cheap selection is preferred. (For selection *consistency*, Li & Bell
note the adaptive LASSO; for prediction, plain LASSO already beats AIC/BIC and
leave-many-out CV in their simulations.)

.. _pda-lasso-criterion:

Choosing :math:`\lambda`: ``lasso_criterion``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cross-validation is one rule for the penalty and it is the default here, but it
is not the rule every paper in this literature uses. Shi & Huang's ``fsPDA``
package selects the penalty by an information criterion instead -- the same
modified BIC their forward selection uses, applied along the LASSO path:

.. math::

   \mathrm{IC}(\lambda) = \log\bigl(\widehat{\sigma}^2(\lambda)\bigr)
     + H\,\log(\log N)\,\frac{\log T_0}{T_0}\,k(\lambda),

where :math:`\widehat{\sigma}^2(\lambda)` is the pre-period mean squared
residual, :math:`k(\lambda)` counts the selected donors, and :math:`H` is a
constant the authors tune "to allow Lasso to take in more variables"
(Remark 4 cont., p. 521). The minimiser over their grid
:math:`\lambda \in \{0.01, 0.02, \dots, 1\}` is the penalty.

Set ``lasso_criterion="mbic"`` to use it, and ``lasso_mbic_const`` to set
:math:`H` (their default is 2; a smaller value selects more donors, a larger one
fewer). Three things move together:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``lasso_criterion="cv"`` (default)
     - ``lasso_criterion="mbic"``
   * - penalty
     - 5-fold cross-validation
     - argmin of :math:`\mathrm{IC}(\lambda)` over their grid
   * - intercept
     - fitted
     - none (``glmnet(..., intercept = FALSE)``)
   * - donor scaling
     - raw
     - divided by each column's standard deviation, coefficients returned
       on the original scale (``standardize = TRUE``)

The intercept and the scaling are part of the rule. The criterion scores a
particular fit, so the estimate it selects a penalty for has to be that fit;
changing only the criterion gives a penalty chosen for a model the estimator
never fits. The port agrees with ``glmnet`` 4.1.8 to 5e-09 on the scaled path.

Which to use. Cross-validation targets prediction and is the safer default when
the donor pool is a convenience sample and you have no reason to prefer one
sparsity level. Use ``mbic`` when you are comparing against Shi & Huang or
another ``fsPDA``-based result, or when you want the selection to be
scale-explicit and deterministic -- it has no fold randomness, so the same panel
always returns the same donors.

Forward selection (``fs``, Shi & Huang)
---------------------------------------

Idea. Instead of penalize, *grow* the control set greedily. Start empty;
at each step add the control whose inclusion maximizes the pre-treatment OLS
:math:`R^2` (equivalently minimizes the residual sum of squares). The number of
selected controls :math:`R` is a tuning parameter chosen by a modified BIC
(Wang, Li & Tsai),

.. math::

   \widehat{R} = \operatorname*{argmin}_{r}
     \log\bigl(\widehat{\sigma}^2(\widehat{U}_r)\bigr)
     + \log(\log N)\,\frac{r\,\log T_0}{T_0},

with :math:`\widehat{\sigma}^2(\widehat{U}_r)` the pre-period residual variance of OLS
on the :math:`r`-unit set. The counterfactual is the OLS extrapolation on the
chosen set :math:`\widehat{U}_{\widehat{R}}`. Forward selection evaluates
:math:`\sum_r (N-r+1)` regressions -- *linear* in :math:`N` -- versus the
:math:`2^N` of exhaustive subset search.

Computing the greedy step
^^^^^^^^^^^^^^^^^^^^^^^^^

The step as the paper states it. At step :math:`r`, with
:math:`\widehat{U}_r` the donors already chosen, forward selection scores every
remaining donor by the pre-period fit its admission would produce, and keeps the
best:

.. math::

   j^\star = \operatorname*{argmin}_{j \notin \widehat{U}_r}
             \widehat{\sigma}^2\bigl(\widehat{U}_r \cup \{j\}\bigr),
   \qquad
   \widehat{\sigma}^2(S) = \frac{1}{T_0}\,
       \min_{\boldsymbol{\beta}}
       \bigl\lVert \mathbf{y}_1 - \mathbf{X}_S\,\boldsymbol{\beta}
       \bigr\rVert_2^2 .

Read literally that is :math:`N - r` separate least-squares problems per step,
each on a :math:`T_0 \times (r+1)` design, and the recursion repeats them from
scratch at every step.

The step as ``mlsynth`` computes it. Let :math:`P_r` be the orthogonal
projection onto the span of the chosen donors (the constant adjoined when one is
fitted), and write

.. math::

   \mathbf{e}_r = (\mathbf{I} - P_r)\,\mathbf{y}_1,
   \qquad
   \mathbf{z}_j = (\mathbf{I} - P_r)\,\mathbf{x}_j

for the current pre-period residual and for a candidate donor orthogonalised
against that span. The fit after admitting :math:`j` is then available in
closed form,

.. math::

   T_0\,\widehat{\sigma}^2\bigl(\widehat{U}_r \cup \{j\}\bigr)
     = \bigl\lVert\mathbf{e}_r\bigr\rVert_2^2 - \Delta_j,
   \qquad
   \Delta_j = \frac{\bigl(\mathbf{x}_j^\top\mathbf{e}_r\bigr)^2}
                   {\bigl\lVert\mathbf{z}_j\bigr\rVert_2^2},

with :math:`\mathbf{x}_j^\top\mathbf{e}_r =
\mathbf{z}_j^\top\mathbf{e}_r` because :math:`\mathbf{e}_r` is orthogonal to the
span by construction. The subtracted term is the only part that depends on
:math:`j`, so the two selection rules coincide:

.. math::

   \operatorname*{argmin}_{j \notin \widehat{U}_r}
     \widehat{\sigma}^2\bigl(\widehat{U}_r \cup \{j\}\bigr)
   \;=\;
   \operatorname*{argmax}_{j \notin \widehat{U}_r} \Delta_j .

All :math:`N - r` numerators are one matrix-vector product
:math:`\mathbf{Z}^\top\mathbf{e}_r`, and the denominators reach the next step
through a rank-one downdate of :math:`\mathbf{Z}`. A donor lying inside the span
has :math:`\lVert\mathbf{z}_j\rVert_2 = 0` and contributes
:math:`\Delta_j = 0`, which is the closed form saying it cannot lower the
residual.

What differs, and what does not. The displayed equality is an identity in exact
arithmetic, not an approximation, so the two rules select the same donor. The
criterion :math:`\mathrm{IC}(r)` still reads a :math:`\widehat{\sigma}^2`
returned by an ordinary least-squares solve on the design the step selects, so
the stopping rule is evaluated on the same quantity in both forms. What changes
is the arithmetic spent getting there:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Form of the step
     - Arithmetic per step
     - Least-squares solves per step
   * - Scoring by refitting
     - :math:`O(N T_0 r^2)`
     - :math:`N - r`
   * - Scoring by projection
     - :math:`O(T_0 N)`
     - :math:`1`

Where the difference is felt. A single fit on :math:`N = 1000` donors over
:math:`T_0 = 300` pre-periods goes from 189 ms to 7.7 ms. The effect compounds
wherever the selection is re-run: setting ``prediction_intervals=True`` refits
the estimator on every bootstrap sample (999 by default -- see
:ref:`the prediction-interval section <pda-prediction-intervals>`), so a
forward-selected fit with intervals on 120 donors goes from 22.3 s to 1.3 s.

The estimate is unchanged, and the benchmarks check it --
``fspda_dense_mc`` reproduces the released R package's selected set on 8 of 8
Monte Carlo cells, with coefficients, ATE and z-statistic agreeing to
:math:`10^{-13}`; ``fspda_sparse_mc`` agrees with the authors' ``nonsparse/FS.R``
rule on 30 of 30 panels; ``pda_table1`` and ``pda_luxurywatch`` return the values
they returned before.

Where the two forms can name different donors. Donors that fit the pre-period
identically -- an exact duplicate of a selected donor, or a pool spanning fewer
directions than it has members -- carry equal :math:`\Delta_j` up to rounding.
The search settles such a tie by the least-squares comparison itself, across a
bounded number of the tied candidates, and on the pools an applied panel
presents it selects what the definition selects. Two regimes sit outside that
bound: a tied group larger than the shortlist, and a pre-period that has
interpolated. In the second, once :math:`\mathbf{y}_1` lies exactly in the donor
span the residual falls to the level of rounding noise, and each further step
compares one :math:`\widehat{\sigma}^2` of order :math:`10^{-31}` against
another; both forms admit a few extra donors, and which ones follows the
floating-point library. In either regime the donors named can differ within a
set that fits the pre-period equally well, the extras carry coefficients at the
:math:`10^{-16}` level, and the counterfactual agrees to machine precision.

Assumptions (Shi & Huang). Asymptotics are *multi-index*: :math:`N\to\infty`
with :math:`T_0 = T_0(N)` deterministic, :math:`\log N / T_0 \to 0`, and
:math:`T_2 = T_2(N) \to \infty` with :math:`\log N / T_2 \to 0` (:math:`N`
may exceed :math:`T_0`).

*Assumption 1 (sparse Riesz / restricted eigenvalue).* The minimal eigenvalue
of the population Gram matrix over any :math:`u`-unit subset
(:math:`u \le (1+\delta_1)R`) is bounded below -- a condition the authors show
is a natural implication of the latent factor model, not an ad hoc
restriction.

*Remark.* The bounded minimal eigenvalue is what keeps any candidate subset of
donors well-conditioned, so the greedy step can identify the next informative
control, not chase a near-singular direction; that it follows from the
factor model means it is not an extra demand on the data.

*Assumption 2 (second moments).* Sample second moments converge at the
high-dimensional rate :math:`O_p(\sqrt{\log N / T_0})` with bounded fourth
moments.

*Remark.* The high-dimensional rate is the pre-period price of admitting many
donors: with bounded fourth moments the sample Gram matrix tracks its
population counterpart even when :math:`N` is large relative to :math:`T_0`.

*Assumption 3 (post-period).* Analogous convergence and long-run-variance
bounds on the post-treatment data.

*Remark.* The matching post-period bounds are what let the average effect over
:math:`\mathcal{T}_2` obey a central limit theorem with a consistently
estimable long-run variance, the basis for the post-selection test below.

*Assumption 4 (weak dependence).* The series are strong (:math:`\alpha`-)
mixing with geometric decay, so a Berry-Esseen bound for heterogeneous time
series applies.

*Remark.* The validity is uniform over a class of DGPs (Theorem 1) -- it
holds whether the true coefficients are dense or sparse, which separates
fsPDA from the post-selection-inference literature that needs sparsity or the
oracle property. Theorem 2 shows the greedy algorithm attains a regression
variance asymptotically as small as the best :math:`u`-unit subset, so the
cheap forward search is statistically efficient.

Inference (Shi & Huang, Eq. 4). Because forward selection uses only the
pre-period and, under weak dependence, the pre- and post-periods become
asymptotically independent (sample splitting), the naive conditional
t-statistic is valid:

.. math::

   \widehat{\mathcal{Z}}_{\widehat{U}} = \widehat{\rho}_{\tau\widehat{U}}^{-1}\sqrt{T_2}\,
       \widehat{\tau}_{\widehat{U}} \xrightarrow{d} N(0,1),

where :math:`\widehat{\rho}^2_{\tau\widehat{U}}` is the HAC long-run variance of the
de-meaned post-period effects. No first-stage variance term is needed --
the asymptotic independence absorbs it -- which makes fsPDA's inference the
simplest of the three.

.. note::

   Long-run-variance estimator. ``mlsynth`` defaults to the prewhitened
   Newey-West estimator (Andrews-Monahan VAR(1) prewhitening + Bartlett kernel
   with the data-driven NW(1994) bandwidth + finite-sample adjustment) -- R's
   ``sandwich::lrvar(..., prewhite = TRUE, adjust = TRUE)``, which Shi & Huang
   use in their application scripts. Prewhitening is essential when the
   treatment-effect series is strongly serially dependent: monthly growth rates
   mean-revert (lag-1 autocorrelation around :math:`-0.45` in the luxury-watch
   panel), and a plain Bartlett kernel cannot absorb that, leaving
   :math:`\widehat\rho` nearly double its true value and the test far too
   conservative. Setting ``lrvar_lag`` instead switches to the released
   ``est.fsPDA`` package's fixed-lag Bartlett estimator
   (default lag :math:`\lfloor T_2^{1/4}\rfloor`, capped at
   :math:`\lfloor\sqrt{T_2}\rfloor`); on the watch panel that no-prewhitening
   form gives an insignificant :math:`t \approx -1.15`, versus the prewhitened
   default's :math:`-2.51` (the paper reports :math:`-2.457`). The field is read
   by ``fs``, ``hcw`` and ``lasso``, and it means the same thing in all three:
   studentize by the fixed-lag Bartlett long-run variance of the post-period
   effects alone. For ``lasso`` that also drops Li & Bell's first-stage term,
   since the test being selected is the one that does not have it.

When to use. A large candidate-control pool where the goal is to
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
     - dense coefficients; ``N>T0``; prediction & heteroskedasticity-robustness
   * - ``lasso``
     - sparse
     - post HAC (+ first stage, small if ``T0>>T2``)
     - few relevant controls; interpretable selection; very large ``N``
   * - ``fs``
     - dense or sparse
     - post HAC only (sample splitting)
     - large pool; predictive ensemble; cheap; honest post-selection inference



.. _pda-shared-assumptions:

Shared assumptions across the PDA class
---------------------------------------

The three estimators (``l2``, ``lasso``, ``fs``) differ in how they
fit :math:`\boldsymbol\beta`, but they share the same identifying
stack. Stated formally:

A1 (Latent factor model for untreated outcomes). All
:math:`N + 1` units share at most :math:`r` common latent factors,

.. math::

   y_{jt}^N \;=\; \mu_j \;+\; \boldsymbol\lambda_j' \mathbf f_t
              \;+\; u_{jt},
   \qquad j \in \{1\} \cup \mathcal N_0, \;\; t \in \mathcal T,

with :math:`\mathbb E[\mathbf f_t u_{jt}] = 0`. This is the
*shared model* underlying HCW (Hsiao-Ching-Wan 2012), Li-Bell
(2017), Shi-Huang (2023), and Shi-Wang (l2). The factor structure
is what licenses the linear projection of the treated unit's
untreated outcome on the controls' outcomes,

.. math::

   y_{1t} \;=\; \alpha^0 \;+\; \mathbf x_t' \boldsymbol\beta^0
              \;+\; \epsilon_t,
   \qquad \mathbb E[\mathbf x_t \epsilon_t] = \mathbf 0,

with :math:`\boldsymbol\beta^0 = \boldsymbol\Omega^{-1}
\boldsymbol\Lambda (\boldsymbol\Lambda' \boldsymbol\Omega^{-1}
\boldsymbol\Lambda + \mathbf I_q)^{-1} \boldsymbol\lambda_1`.

A2 (Single treated unit, sharp absorbing aggregate-level
treatment). Unit :math:`j = 1` is the only treated unit;
treatment turns on at :math:`T_0 + 1` and stays on. Donors are
untreated throughout (no interference). The original HCW /
Li-Bell / Shi-Huang theorems are stated for this single-treated
case. (The l2-relaxation paper Section 4.4 sketches a
multiple-treated-units extension with a short post-window; the
mlsynth implementation tracks the single-treated form.)

A3 (Weak temporal dependence). The series
:math:`(\mathbf x_t, y_{1t})` are :math:`\rho`-mixing or strong-
mixing with at-least-geometric decay (the exact rate varies by
variant):

* Li-Bell A6: :math:`w_t = (\widetilde y_t', \epsilon_{1t})` is a
  weakly stationary :math:`\rho`-mixing process with
  :math:`\rho(s) = O(\lambda^s)`.
* Shi-Huang A4: strong (:math:`\alpha`-) mixing with geometric
  decay, so a Berry-Esseen bound for heterogeneous time series
  applies.
* Shi-Wang A3-A4: time-series weak dependence at the
  :math:`O_p(T_0^{-1/2})` and :math:`O_p(\sqrt{\log N / (N \wedge
  T_0)})` rates for the sample moments.

This is what makes pre-period sample moments converge at the
high-dimensional rate and what makes the
pre-period and post-period asymptotically independent, which
is the engine behind fs-PDA's sample-splitting inference and the
two-term HAC variance in l2 / lasso.

A4 (Sample-size regime). :math:`N \to \infty`,
:math:`T_0 = T_0(N) \to \infty` deterministically with
:math:`\log N / T_0 \to 0`, :math:`T_2 \to \infty` with
:math:`\log N / T_2 \to 0`. :math:`N` may exceed :math:`T_0`,
which is the entire point of the high-dimensional PDA literature.
Li-Bell's A7 additionally posits
:math:`\eta = \lim T_2 / T_0 \in [0, \infty)`, which determines
whether the first-stage variance term :math:`\widehat\Sigma_1`
matters.

A5 (Donor pool regularity). The controls' Gram matrix has
enough variation:

* For ``lasso`` (Li-Bell A2): :math:`\mathrm{Rank}(\widetilde B) = K`
  -- removing the first row of the loading matrix leaves
  full-rank factor variation; :math:`E[x_t x_t']` is invertible
  on the active set.
* For ``fs`` (Shi-Huang A1): a sparse Riesz / restricted
  eigenvalue condition -- the minimum eigenvalue of the
  population Gram matrix over any :math:`u`-unit subset
  (:math:`u \le (1 + \delta_1) R`) is bounded below. The paper
  shows this is a *natural implication* of the latent factor
  model, not an ad-hoc lasso-style assumption.
* For ``l2`` (Shi-Wang A1-A2): factor strength
  :math:`\xi_N = \phi_{\min}(\boldsymbol\Lambda' \boldsymbol\Lambda
  / N)` may vanish (weak factors allowed); the idiosyncratic
  covariance has eigenvalues bounded in
  :math:`[\underline\sigma^2, \overline\sigma^2]`.

A6 (Variant-specific structure of :math:`\boldsymbol\beta^0`
).

* ``lasso``: sparse :math:`\boldsymbol\beta^0` -- only
  :math:`m = o(T_0)` of its components are non-zero.
* ``l2``: dense :math:`\boldsymbol\beta^0` -- almost no
  exact zeros (the factor projection gives every donor a
  small-but-nonzero coefficient).
* ``fs``: agnostic -- the inference is valid uniformly over a
  class of DGPs that includes both dense and sparse
  :math:`\boldsymbol\beta^0` (Theorem 1 in Shi-Huang).

A7 (Inferential regularity). For all three, the post-period
average effect :math:`\widehat\tau` has a CLT with HAC long-run
variance consistently estimable by Newey-West. For ``l2`` and
``lasso``, both the pre-period (first-stage) and post-period
HAC variances enter; for ``fs``, sample-splitting absorbs the
first-stage term and only the post-period HAC variance
enters. Shi & Huang apply the same sample-splitting argument to
their modified-BIC LASSO, which is why ``lrvar_lag`` drops the
first-stage term there too.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) Factor-driven DGP (A1). PDA's whole identification story
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

(b) Single treated unit, absorbing aggregate-level treatment
    (A2). Multiple treated units, treatment that turns off, or
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

(c) Weak temporal dependence (A3). All three variants
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

(d) Sample-size regime (A4). PDA needs both :math:`T_0` and
    :math:`T_2` growing with :math:`\log N` small relative to
    each. A short post-period (:math:`T_2 \le 5`) breaks the
    CLT on :math:`\widehat\tau`; a short pre-period
    (:math:`T_0 \le 20`) breaks the moment-convergence rates.

    *Plausibly violated when* the pre-period is short with many
    donors. *Diagnostic*: compute
    :math:`(\log N) / T_0` and :math:`(\log N) / T_2`; both
    should be visibly below 1. If they are not, the asymptotic
    approximation has not kicked in. Either lengthen the panel
    (aggregate to a finer time grid), prune donors, or move to
    *canonical SCM* / :doc:`tssc` / :doc:`fdid` which work with
    shorter panels.

(e) Donor regularity (A5). Each variant has its own
    Gram-matrix / factor-strength condition. The practical
    common failure is near-collinear donors: two donor
    series that move together up to noise produce a near-
    singular pre-period Gram matrix.

    *Plausibly violated when* the donor pool contains
    near-duplicates (two adjacent states with essentially
    identical industry mix). *Diagnostic*: read the condition
    number of :math:`\Gamma_{\mathcal T_1}(\mathbf x_t,
    \mathbf x_t')`. A condition number above ~1e6 is a red
    flag. For ``lasso`` and ``fs`` this manifests as selection
    flipping between near-clones across seeds; for ``l2`` the
    :math:`\varepsilon`-validation curve gets noisy. Prune one of each clone
    pair before refitting.

(f) Coefficient structure (A6) -- choosing the right variant.
    The biggest practitioner-side decision is whether to assume
    sparse or dense :math:`\boldsymbol\beta^0`. Choosing wrong
    pays a real cost: ``lasso`` on a dense truth over-selects
    and inflates size (see the Path-B table above:
    LASSO's size is 0.16-0.36 under the dense factor DGP, vs
    ``fs``'s 0.05); ``l2`` on a sparse truth pays variance for
    the dense fit it doesn't need.

    *Diagnostic*: fit ``fs`` first -- it's valid in both
    regimes per Shi-Huang Theorem 1, and the selected
    :math:`\widehat R` and per-step :math:`R^2` curve tell you
    whether you're in a sparse (few donors carry the fit) or
    dense (many donors add information) regime. Then run the
    matched variant (``lasso`` if ``fs`` keeps a handful;
    ``l2`` if ``fs`` keeps many).

(g) Inferential regularity (A7). The HAC long-run variance
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

Reach for PDA when:

* You have a single treated unit, a moderate-to-large donor
  pool (:math:`N` comparable to or exceeding :math:`T_0`), and
  a plausibly factor-driven panel. PDA was designed for exactly
  this regime, and unlike canonical SCM, the projection has no
  simplex / non-negativity constraint -- it can extrapolate
  through negative coefficients on far donors when the factor
  structure demands it.
* You want HAC-based, classical-statistics inference on the
  ATE with a closed-form normal-distribution test, not a
  permutation or conformal procedure. PDA's CLT-based inference
  is what makes it the closest synthetic-control cousin to
  difference-in-differences from an inferential standpoint.
* The treated unit's pre-trajectory lies in the linear span
  (not necessarily convex hull) of the controls, so a
  no-constraint linear projection makes sense. PDA cannot
  recover an effect when the projection itself is impossible.
* You're between dense (``l2``) and sparse (``lasso``)
  regimes and want a uniformly valid test that doesn't
  require choosing the right sparsity story -- run ``fs``.

Do not use PDA when:

* You need convex (non-negative, sum-to-one) weights as a
  policy-interpretation deliverable. PDA's no-constraint
  projection produces negative coefficients on far donors,
  which is awkward to explain in many policy contexts. Use
  *canonical SCM* / :doc:`tssc` (canonical SC) or :doc:`fscm`
  (forward-selected SC with the simplex retained).
* The treated unit is structurally outside the donor span
  (not just the convex hull). PDA's linear projection cannot
  reach a treated outcome that no linear combination of donor
  outcomes can express. The pre-period RMSE stays large at
  every PDA variant. Use :doc:`iscm`, whose A2(b) mechanism
  identifies the effect through donors that use the treated
  unit as a positive-weight donor in *their* synthetic
  controls.
* Outcomes are non-stationary (unit-root or trending
  series). A3 fails and the pre/post asymptotic-independence
  story breaks. First-difference the outcome (the
  l2-relaxation paper's empirical PPI application does
  exactly this), or use :doc:`sbc` (stationary-cycle
  estimator).
* You have multiple treated units with overlapping cohorts.
  PDA's theorems (with the exception of the l2 relaxerm which does support multiple treated units but is not written yet) are written for the single-treated case. Use
  :doc:`sdid` for staggered adoption.
* Spillovers across donors. A2's no-interference clause
  fails when donor states are economically linked to the
  treated state. Use :doc:`spillsynth` or :doc:`spsydid`.
* Continuous or multi-valued treatment. PDA encodes a
  single binary intervention; continuous dose belongs in
  :doc:`ctsc`.
* Distributional questions (Lorenz curves, QTEs).
  PDA targets the mean ATE through a Gaussian-likelihood
  linear projection. Use :doc:`dsc` for distributional
  effects.
* You need Bayesian posterior credible bands. PDA returns
  frequentist HAC-based CIs. For Bayesian inference and
  posterior inclusion probabilities on donors, use
  :doc:`bvss` (spike-and-slab with a soft simplex).
* Very short pre-period :math:`(T_0 \le 15)` with many
  donors. The high-dimensional approximation has not kicked
  in; the selected :math:`\widehat\beta` is noise. Use *canonical SCM*
  / :doc:`tssc` / :doc:`fdid`, which work without
  high-dimensional asymptotics.
* Very short post-period :math:`(T_2 \le 5)`. The CLT on
  :math:`\widehat\tau` is shaky; the HAC bandwidth choice
  dominates the inference. Either accept a wider permutation
  CI from *canonical SCM* / :doc:`tssc`, or move to the
  l2-relaxation multiple-treated-units extension (Shi-Wang
  Section 4.4) which is built for this regime.
* You want predictor-level matching (covariates +
  pre-period outcomes), not outcome-only projection. PDA's
  workhorse projection is on donor outcomes, not on
  predictor moments. Use *canonical SCM* / :doc:`tssc` /
  :doc:`sparse_sc` (predictor selection with L1 penalty on the
  V-weight matrix) for the predictor-matching setup.
* The factor model itself is the object of interest (you
  want to identify and interpret the factors). PDA is
  agnostic to factor estimation -- the factor model only
  motivates the linear projection, never enters the
  estimator. Use :doc:`fma` (factor-model-aware estimator) if
  recovering factors is the goal.

Empirical Illustration: Hong Kong economic integration
-------------------------------------------------------

The original HCW [HCW]_ application -- and Shi & Huang's Example 1 -- evaluates
the effect of economic integration with mainland China on Hong Kong's quarterly
real-GDP growth, using 24 comparison economies. Here all four PDA methods run
on the same data, and the package returns the time-varying effect, the ATE, and
a HAC confidence interval for each.

.. code-block:: python

   import pandas as pd
   from mlsynth import PDA

   url = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/HongKong.csv"
   df = pd.read_csv(url)

   res = PDA({"df": df, "outcome": "GDP", "treat": "Integration",
              "unitid": "Country", "time": "Time",
              "methods": ["hcw", "l2", "LASSO", "fs"], "alpha": 0.05,
              "display_graphs": True}).fit()

   for name, fit in res.fits.items():
       print(f"{name:6s} ATE {fit.att:.4f}  SE {fit.att_se:.4f}  "
             f"95% CI ({fit.ci[0]:.4f}, {fit.ci[1]:.4f})  p={fit.p_value:.3f}  "
             f"donors={len(fit.donor_weights)}")

This prints::

   hcw    ATE 0.0403  SE 0.0057  95% CI (0.0292, 0.0514)  p=0.000  donors=6
   l2     ATE 0.0261  SE 0.0033  95% CI (0.0195, 0.0326)  p=0.000  donors=24
   lasso  ATE 0.0330  SE 0.0054  95% CI (0.0224, 0.0436)  p=0.000  donors=11
   fs     ATE 0.0395  SE 0.0059  95% CI (0.0280, 0.0510)  p=0.000  donors=9

All four find a significant positive integration effect on Hong Kong's GDP
growth -- between +2.6 and +4.0 percentage points per quarter -- differing only
in how they choose controls. ``hcw`` runs the exact best-subset search and
certifies a parsimonious six-economy model (Austria, Italy, Korea, Mexico,
Norway, Singapore): on this pool the long 44-quarter pre-period gives the
branch-and-bound room to certify the global optimum (``certified_optimal`` is
``True``, gap 0) even over 24 donors. ``l2`` keeps all 24 controls with dense,
signed coefficients (pre-RMSE 0.012); ``lasso`` keeps 11; ``fs`` grows to 9. The
four estimates bracket the Forward-DiD result on the same data (0.025), a useful
cross-method check.

.. _pda-prediction-intervals:

Prediction intervals
--------------------

The HAC confidence interval above quantifies uncertainty in the *average*
treatment effect. For uncertainty in each *period's* effect, set
``prediction_intervals=True`` to attach the bootstrap prediction intervals of
Jiang, Li, Shen and Zhou [pdapi]_ to every fitted variant. These bound the
post-period untreated counterfactual :math:`Y_t` (and hence the effect
:math:`\Delta_t = y_t - \widehat Y_t`), combining the in-sample estimation error
with the out-of-sample prediction error that a confidence interval for the mean
omits. The construction (their Algorithm 2.1) resamples the pre-period
prediction error with a dependent wild bootstrap and the post-period error with a
residual bootstrap, refits the chosen estimator on each bootstrap sample at the
fixed tuning parameter, and reads quantiles of the self-normalized statistic
:math:`\widehat e_t / \sqrt{\widehat V_t + \widehat\sigma^2}`.

The implementation lives at the shared :func:`mlsynth.utils.inferutils.pda_prediction_intervals`
so any panel-data estimator can reuse it. Both the equal-tailed (``eq``) and
symmetric (``sy``) intervals are returned, and each variant reports which
studentization it used: ``sandwich`` for the post-selection OLS HAC variance
:math:`\widehat V_t` (``hcw``, ``lasso`` and ``fs``, which all select then run
OLS -- ``hcw`` is exactly the low-dimensional post-selection-OLS case of Jiang
et al.'s Remark 2.1, with best subset as the selector re-run on each bootstrap
draw), or the ``sigma2`` fallback when that sandwich is undefined -- as for the
dense L2-relaxation when the controls outnumber the pre-periods.

.. code-block:: python

   res = PDA({"df": df, "outcome": "GDP", "treat": "Integration",
              "unitid": "Country", "time": "Time",
              "methods": ["l2", "LASSO", "fs"],
              "prediction_intervals": True, "pi_n_boot": 199, "pi_seed": 0,
              "display_graphs": False}).fit()

   for name, fit in res.fits.items():
       eff = fit.prediction_intervals["effect"]
       print(name, fit.prediction_intervals["studentization"])
       for k in range(3):                       # first three post-quarters
           lo, hi = eff["eq_lower"][k], eff["eq_upper"][k]
           print(f"  t+{k+1}: effect {eff['point'][k]:+.4f}  95% PI ({lo:+.4f}, {hi:+.4f})")

The pointwise 95% effect intervals for the first three post-quarters (decimal
GDP growth)::

   method  studentization   t+1                          t+2                          t+3
   l2      sandwich         +0.0203 (-0.0128, +0.0411)   +0.0427 (+0.0076, +0.0658)   +0.0023 (-0.0347, +0.0272)
   lasso   sandwich         +0.0312 (-0.0066, +0.0499)   +0.0519 (+0.0165, +0.0706)   +0.0152 (-0.0198, +0.0397)
   fs      sandwich         +0.0249 (-0.0079, +0.0521)   +0.0467 (+0.0158, +0.0719)   +0.0103 (-0.0237, +0.0351)

Each band is wider than the ATE confidence interval, because it carries the
out-of-sample prediction error on top of the estimation error, and it tightens
or widens quarter by quarter as the fitted controls track each period better or
worse.

The cumulative effect
---------------------

The intervals above bound one period at a time. The number an event study is
usually quoted by is the running total -- how much the treated unit gained over
the first :math:`L` periods -- and ``cumulative_band=True`` attaches a band for
it:

.. code-block:: python

   res = PDA({..., "prediction_intervals": True, "cumulative_band": True}).fit()
   band = res.fits["lasso"].cumulative_band
   for L in (1, 4, 8):
       i = L - 1
       print(f"L={L}: {band.point[i]:+.3f}  ({band.lower[i]:+.3f}, {band.upper[i]:+.3f})")

An interval for a running total is not the running total of the period
intervals, and the two obvious shortcuts are both wrong. Adding the period
endpoints treats every period's error as moving in lockstep, so the width grows
in proportion to :math:`L` whatever the data does. Rescaling a single period's
interval by :math:`\sqrt{L}` assumes the opposite, that the errors are
independent, and for a donor fit the estimation error persists across horizons,
so that one is wrong as well. Neither measures anything.

What mlsynth does instead is accumulate each bootstrap replicate's error path
first and take the standard error afterwards. Whatever correlation the period
errors have is then carried into the band rather than assumed into it:
independent errors widen it like :math:`\sqrt{L}`, perfectly correlated ones
like :math:`L`, and the replicates decide which. On the Oregon opioid panel of
Wheeler's ``LassoSynth`` -- 13 post-periods, one treated state -- the standard
error grows by a factor of 4.43, against :math:`\sqrt{13} = 3.61` for
independence and :math:`13` for lockstep. It sits between them, which is the
whole point of measuring it. The resulting half-width is 1.37 where summing the
period endpoints would have given 3.78, against a cumulative point estimate of
7.59.

The band is simultaneous over horizons rather than pointwise. A cumulative path
is read as a path -- "positive by period six and never back" is a claim about
every horizon at once -- and a pointwise band read that way covers at well below
its nominal level. One shared critical value
(:func:`mlsynth.utils.supt.supt_critical_value`, following Montiel Olea and
Plagborg-Moller) restores the level for the path as a whole; on the Oregon panel
it is 2.54 where the pointwise normal quantile would be 1.96.

The band reuses the replicate paths the prediction-interval bootstrap already
produced, so it costs no extra refits, and it therefore requires
``prediction_intervals=True``. Asking for it without the bootstrap raises rather
than returning an empty field, since a caller reading a missing band as an
absent effect is the one failure worth ruling out. PPSCM's ``cumulative_band``
is the same object built the same way, sharing
:mod:`mlsynth.utils.supt`, so the two estimators cannot drift apart in what the
phrase means.

``hcw`` produces these same intervals, with one practical caveat: the bootstrap
refits the entire selection on every draw, and HCW's refit re-runs the
best-subset search. On the 24-economy pool that is roughly ten seconds a draw, so
its prediction intervals there cost minutes -- while the certified point estimate
and the HAC ATE interval need no resampling and return at once. On HCW's
intended, pre-screened pool (the ten candidate economies of the sovereignty
study) the search is tiny and the intervals come straight back:

.. code-block:: python

   cands = ["China", "Indonesia", "Japan", "Korea", "Malaysia", "Philippines",
            "Singapore", "Taiwan", "Thailand", "United States"]
   sub = df[df["Country"].isin(["Hong Kong"] + cands)]

   hk = PDA({"df": sub, "outcome": "GDP", "treat": "Integration",
             "unitid": "Country", "time": "Time", "method": "hcw",
             "prediction_intervals": True, "pi_n_boot": 199,
             "display_graphs": False}).fit()
   eff = hk.fits["hcw"].prediction_intervals["effect"]
   lo, hi = eff["eq_lower"][0], eff["eq_upper"][0]
   print(f"t+1: effect {eff['point'][0]:+.4f}  95% PI ({lo:+.4f}, {hi:+.4f})")
   # t+1: effect +0.0304  95% PI (-0.0227, +0.0703)

Verification
------------

.. note::

   Empirical (Path A, Hong Kong). All four PDA methods run on the HCW Hong Kong
   panel (above) and agree on a significant positive integration effect,
   consistent with the literature and the Forward-DiD cross-check (0.025).

   Original HCW (Path A, best subset). ``benchmarks/cases/pda_hcw_hongkong.py``
   reproduces Hsiao, Ching & Wan (2012) Tables XVI-XVII value-for-value with
   ``method="hcw"``: on the sovereignty study (ten candidate economies,
   T0 = 18), AICc selects {Japan, Korea, Taiwan, USA} with the published OLS
   weights, pre-period :math:`R^2 = 0.9314`, and an insignificant average
   effect of -3.96% -- cross-validating against the ``pampe`` R package.

   Prediction intervals (Path B, coverage). The bootstrap prediction
   intervals are validated by ``benchmarks/cases/pda_pi_coverage.py``, which
   reproduces the coverage geometry of Jiang et al. (2025) Tables 2-5 on their
   Setup 1: the equal-tailed and symmetric intervals cover near the nominal
   95%, while the normal-quantile intervals under-cover (to about 77% under
   exponential errors).

   Table 1 in full (Path B). All 108 cells of Shi & Huang's Monte Carlo are
   reproduced by ``benchmarks/cases/fspda_table1.py``, against the paper and
   against their own ``FS.R`` and ``lasso.BIC.R``. Details:
   :doc:`replications/fspda_table1`.

Simulation study (Path B): forward selection vs LASSO
-----------------------------------------------------

Shi & Huang's (2023) Table 1 compares forward selection against LASSO on a
four-factor DGP, ported from their released ``FS.simulation.dense.R`` into
:func:`mlsynth.utils.pda_helpers.simulation.simulate_pda_panel`: four factors
(``f1`` i.i.d.; ``f2`` AR(1) 0.9; ``f3`` MA(2) (0.8,0.4); ``f4`` ARMA(1,1)
(0.5,0.5) under the *dynamic* structure; under the *i.i.d.* structure factor
:math:`\ell` is ``N(0, \ell^2)``, so the fourth carries sixteen times the
variance of the first); loadings ``U(1,2)`` on the treated + 4 relevant controls
and ``U(-0.5,0.5)`` on the remaining 96; idiosyncratic ``N(0, 0.5^2)``; one
treated unit, :math:`N=100` controls, :math:`T_0=T_2`. Shocks ``D1``-``D7`` set
the post-period ATE (``D1``-``D3`` null -> *size*; ``D4``-``D7`` non-zero ->
*power*).

The irrelevant loadings are the detail that decides the design. Section 4.1 of
the paper gives them as ``U(-0.1,0.1)``; the ``loading.RData`` their driver
loads has them five times wider, and that is what Table 1 was run on. At the
wider range the 96 irrelevant donors carry real signal, the regression is dense,
and no method can select its way to a sparse truth -- which is the paper's
subject.

The full table, all 108 cells, is the ``fspda_table1`` case. Driving the
packaged ``PDA`` at its *defaults* (``LassoCV`` for the penalty, Li & Bell's
two-component variance for the test) at :math:`T_0=100`, with 200 replications
for size and 60 for power:

.. list-table:: mlsynth's defaults on the Table-1 design, :math:`T_0=100`
   :header-rows: 1
   :widths: 12 8 10 10 10

   * - factors
     - method
     - # donors
     - size (D1)
     - power (D5)
   * - i.i.d.
     - fs
     - 5.5
     - 0.095
     - 1.00
   * - i.i.d.
     - LASSO
     - 13.5
     - 0.070
     - 1.00
   * - dynamic
     - fs
     -
     - 0.095
     - 1.00
   * - dynamic
     - LASSO
     -
     - 0.045
     - 1.00

Forward selection is the parsimonious rule in both structures and its test is
sized at 0.095 under either, against the paper's 0.059 and 0.088. The
cross-validated LASSO takes in 13.5 donors where the paper's modified BIC takes
11.

The paper's size inflation does not appear here, and the reason is the penalty
rule. Shi & Huang report the LASSO's null rejection rate going from 0.058 under
i.i.d. factors to 0.184 under dynamic ones. Holding the variance estimator fixed
at Li & Bell's and changing only the penalty reproduces the split: under the
modified BIC the rate goes 0.013 to 0.087 at this length, and under
cross-validation it goes 0.070 to 0.045. The inflation belongs to their
selection rule, not to L1 selection. Both rules are fully powered at ``D5``
(mean-1 shift). Durable cases: ``fspda_table1`` for the paper's table,
``pda_table1`` for the defaults.

.. note::

   The LASSO cells above are cross-validated; the paper's are not. Shi & Huang
   select the Lasso penalty with a modified BIC (Remark 4 cont., p.521:
   "we tune the constants in the modified BIC to allow Lasso to take in more
   variables"); ``mlsynth``'s L1-PDA defaults to ``LassoCV`` (5-fold
   cross-validation). Their rule is ``lasso_criterion="mbic"`` (see
   :ref:`Choosing lambda <pda-lasso-criterion>` above), and pairing it with
   ``lrvar_lag`` gives their test as well. Under both, ``fspda_table1``
   reproduces Table 1 cell by cell; the table above is what the defaults do on
   the same design.

   The panel-level comparison against their code is a third case:
   ``fspda_dense_mc`` runs ``PDA`` on panels generated by their own
   ``FS.simulation.dense.R`` and compares to what their ``FS()`` and
   ``lasso.BIC()`` return on those same panels.

.. admonition:: Cross-validated against fsPDA on their own panels

   ``fspda_dense_mc`` is the cell-level check. On eight panels from their dense
   Monte Carlo (``T1 = T2 = 50``, 100 donors, four factors), forward selection
   picks the same donors on every one, with coefficients, ATE, pre-period
   R-squared and t-statistic agreeing to about :math:`10^{-13}`.

   Their third column, the simplex-constrained synthetic control, is covered by
   the same case through ``VanillaSC``: the two attain the same optimum of
   ``scm.R``'s program, agreeing on the objective to :math:`10^{-11}` and on the
   selected donors on every panel.

   The sparse half of their ``simulation/`` directory is ``fspda_sparse_mc``,
   which runs their ``fs()``, ``lasso_ic()`` and ``oracle()`` on the three sparse
   DGPs. Those are deliberately different rules from the ones mlsynth
   implements -- their sparse forward selection searches all donors at each step
   and scores ``var(e)``, and their LASSO criterion uses a different residual
   variance and a different penalty grid -- so that case pins agreement rates and
   error ratios instead of digits. Forward selection lands on the identical donor
   set on 28 of 30 panels regardless, within 5.4 percent on out-of-sample RMSE.

   The LASSO under ``lasso_criterion="mbic"`` agrees on the selected donors and
   the selected penalty on all eight panels when their ``lasso.BIC`` is run with
   ``glmnet`` converged, with coefficients to :math:`2\times10^{-5}`. Against
   their function as published it agrees on five of eight. The difference is
   ``glmnet``'s default ``thresh = 1e-7``, which stops short in this
   :math:`p = 100`, :math:`n = 50` design: ``mlsynth``'s coefficients attain a
   lower value of the LASSO objective than ``glmnet``'s do, and because the
   criterion scores those coefficients, an under-converged fit can hand a
   different grid point the minimum. The benchmark records both runs so the two
   claims stay separable.

.. admonition:: The ``fs_intercept`` knob -- valid size on factor data

   Achieving the correct fs size above required a fix. The released ``fsPDA``
   R package (``est.fsPDA.R``) fits the donor regression *with* an intercept;
   on the paper's mean-zero factor DGP that intercept absorbs a spurious
   pre-period constant which extrapolates into the post window, biasing the
   gap and inflating the null rejection rate to ~0.20. The paper's Table 1
   was produced by the *simulation* code (``FS.R``), which fits without an
   intercept and yields valid size. ``mlsynth`` exposes both via
   ``PDAConfig.fs_intercept`` (default ``False`` = the no-intercept, valid-size
   form; set ``True`` for panels with genuine unit level shifts). With the
   default, the fs D1 rejection rate drops from ~0.20 (intercept) to the
   ~0.05-0.09 reported above (no intercept).

   One honest caveat: under *dynamic* factors fs shows mild residual size
   inflation at small :math:`T` (the "imprecise long-run-variance" effect the
   paper itself notes), shrinking toward 5% as :math:`T_0 \to \infty`.

L2-relaxation (Shi & Wang). The ``l2`` method's out-of-sample MPSE falls
with :math:`T` and its test approaches the nominal 5% size as :math:`T_0 \to
\infty`, matching Shi & Wang's Table 2 (size :math:`0.142` at :math:`T_0=50`
→ :math:`0.072` at :math:`200`). Each fit cross-validates :math:`\varepsilon`
over its own grid, which is what makes a large Monte Carlo costly here: at the
100 donors this study uses, one fit on the default grid takes about a second at
:math:`T_0 = 50`, against 5 s before the ranking sweep was separated from the
reported fit (see the note on solving the grid above). The case runs a coarse
twelve-point grid and summarizes the table, so the full sweep stays out of the
test suite.

Multiple treated units
----------------------

When *several* units are treated by the same intervention, Shi & Wang's
L2-relaxation PDA fits a counterfactual per treated unit against the shared
control pool and aggregates the effects into a per-period cross-sectional
ATE,

.. math::

   \widehat{\mathrm{ATE}}_t = \frac{1}{J} \sum_{j=1}^{J}
       \bigl( y_{jt} - \widehat{y}_{jt} \bigr),
   \qquad
   \mathrm{s.e.} = \frac{\sqrt{\mathbf{1}'\widehat{\Sigma}_e\,\mathbf{1}}}{J},

with :math:`\widehat{\Sigma}_e` the cross-sectional covariance of the pre-period
prediction residuals (so the standard error is constant across post-periods).
:func:`mlsynth.utils.pda_helpers.multitreat.run_pda_multitreat` implements this.
Because every per-unit fit shares the same
:math:`\widehat{\boldsymbol{\Sigma}} = X'X/T_0`, all :math:`J` fits run through a
single OSQP factorisation -- the batched solver
:func:`mlsynth.utils.pda_helpers.l2.batch.l2_relax_batch` updates only the
constraint bounds per ``(unit, tau)`` (hundreds of conic solves become hundreds
of warm-started ADMM updates). On the Brexit study (52 UK firms vs 300 controls)
this reproduces the paper's first-day return ATE of :math:`-4.3\%`
(:math:`t \approx -7.4`); see ``benchmarks/cases/pda_brexit.py``.

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
unit/time ``IndexSet``\es, and splits pre/post.

.. automodule:: mlsynth.utils.pda_helpers.setup
   :members:
   :undoc-members:

Shared HAC long-run-variance machinery (Bartlett/Newey-West) and the N(0,1)
test used by every PDA variant.

.. automodule:: mlsynth.utils.pda_helpers.inference
   :members:
   :undoc-members:

L2-relaxation (Shi & Wang): the relaxation primal, :math:`\varepsilon` validation, and the
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

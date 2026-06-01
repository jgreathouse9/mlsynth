Cluster Synthetic Controls (CLUSTERSC)
======================================

.. currentmodule:: mlsynth

Overview
--------

CLUSTERSC packages two paper-aligned families of robust synthetic
control behind a single estimator.

* **PCR-SC.** Rho, Tang, Bergam, Cummings & Misra (2025),
  *ClusterSC: Advancing Synthetic Control with Donor Selection*
  (arXiv:2503.21629). Hard Singular Value Thresholding (HSVT) of the
  donor matrix, optional :math:`k`-means donor clustering on the
  right-singular-vector embedding, then ordinary least squares
  against the denoised donor matrix. Builds on the Robust Synthetic
  Control proposal of Amjad, Shah & Shen (2018) and the PCR
  consistency results of Agarwal, Shah, Shen & Song (2021). mlsynth
  exposes two alternative weight solvers on top of Algorithm 2: the
  Bayesian Robust Synthetic Control posterior over the weights (Amjad,
  Shah & Shen 2018) and a simplex-constrained variant that retains
  Abadie-Diamond-Hainmueller weights.

* **RPCA-SC.** Bayani (2021), *Robust PCA Synthetic Control*
  (arXiv:2108.12542; Chapter 1 of Bayani 2022). Functional PCA on
  pre-period trajectories, silhouette-driven :math:`k`-means on the
  resulting FPC scores, robust :math:`L + S` decomposition of the
  treated unit's cluster via Principal Component Pursuit
  (Candes, Li, Ma & Wright 2011) or half-quadratic non-convex
  regularisation (Wang, Li, So & Liu 2023), then non-negative least
  squares against the low-rank donor matrix.

Either family can be selected via :py:attr:`CLUSTERSCConfig.method`
(``"pcr"``, ``"rpca"``, or ``"both"``); when both run, the
:py:attr:`CLUSTERSCConfig.primary` field selects which fit drives the
convenience aliases (``att``, ``counterfactual``, ``gap``,
``donor_weights``) on the result object.

Mathematical Formulation
------------------------

Setup
^^^^^

We observe a single treated unit (indexed 1) and :math:`J` controls
over :math:`T` periods, with treatment starting at :math:`T_0 + 1`.
Stack the donor outcomes as :math:`Y_0 \in \mathbb{R}^{T \times J}`
(columns = donors) and write :math:`y_1 \in \mathbb{R}^{T}` for the
treated series. Pre-period slices are
:math:`Y_0^- \in \mathbb{R}^{T_0 \times J}` and :math:`y_1^- \in
\mathbb{R}^{T_0}`; post-period slices carry the ``+`` superscript.
Treatment-effect targets are the per-period gaps
:math:`\tau_t = y_{1, t} - \hat y^0_{1, t}` and the average
treatment effect on the treated,

.. math::

   ATT = \frac{1}{T - T_0} \sum_{t > T_0} \tau_t.

Both families assume the untreated potential outcome has an
approximately low-rank structure :math:`Y^0 = M + E`, where
:math:`M_{i,t} = g(\theta_i, \rho_t)` is a deterministic latent
factor signal (Athey et al. 2021) and :math:`E` collects zero-mean
idiosyncratic noise.

PCR-SC family (Rho et al. 2025)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PCR family combines three building blocks: rank-:math:`r` HSVT
denoising of the donor matrix, an optional donor-clustering
pre-step, and a weight solver. Algorithm 2 of the paper sets out
the basic pipeline; Algorithm 4 wraps it with the clustering step.

Why PCR: theory and assumptions
"""""""""""""""""""""""""""""""

**What PCR is, and why it is the right tool.** Principal component
regression (project the donors onto their top-:math:`r` principal
components, then regress) is, *without any change*, robust to the
noise and missingness that pervade panel data [Agarwal2021]_. The key
identity is that PCR is **equivalent to ordinary least squares after
hard singular-value thresholding (HSVT)** of the covariate (donor)
matrix (Agarwal et al. 2021, Prop. 2.1): retaining the top-:math:`r`
PCs and regressing yields the *same* fitted response as
:math:`\mathrm{HSVT}_r` followed by OLS. That HSVT step **implicitly
de-noises** the donors by projecting them onto the rank-:math:`r`
signal subspace, which is exactly why the synthetic control inherits
robustness to the idiosyncratic shocks every donor series carries. In
the synthetic-control language this estimator is **Robust Synthetic
Control** (Amjad-Shah-Shen [Amjad2018]_); Agarwal et al. prove PCR and
RSC are identical, so the HSVT+OLS path here carries RSC's guarantees.

**Model (error-in-variables).** In the synthetic-control setting
[ClusterSC]_ formalizes the donor panel as a noisy, partially observed
view of a low-rank signal, :math:`Y_0 = M + E`, where :math:`E` collects
mean-zero idiosyncratic shocks and (optionally) missing cells, and the
treated series obeys an *approximate* linear synthetic control on the
signal,

.. math::

   y_{1,t} = M_{\cdot,t}^\top f^\star + \varepsilon_t + \phi_t ,
   \qquad \|f^\star\| \le \mu ,

with response noise :math:`\varepsilon_t` (mean zero, variance
:math:`\le \sigma^2`) and a deterministic model-mismatch term
:math:`\phi_t`.

**Assumptions.**

*Assumption 1 (low-rank latent-factor signal).* The signal is generated
by a latent variable model :math:`M_{i,t} = g(\theta_i, \rho_t)` with
:math:`g` Lipschitz in finite-dimensional unit factors
:math:`\theta_i` and time factors :math:`\rho_t`, and bounded entries
(:math:`\|M\|_\infty \le 1`). This forces :math:`M` to be (approximately)
low rank, :math:`\mathrm{rank}(M) = r = O(\log T) < T`. *Remark.* This is
the assumption that lets PCR/HSVT work at all -- and it **earns** the
synthetic control rather than assuming it: under this model an
(approximate) linear SC :math:`f^\star` provably *exists* (Agarwal et al.
2021, Prop. 4.1), so the existence of a synthetic combination need not be
imposed as an axiom as in classical SC.

*Assumption 2 (error-in-variables donors).* We observe
:math:`Z_{jt} = M_{jt} + \eta_{jt}` with each cell present
independently with probability :math:`\rho` (missing otherwise); the
covariate noise :math:`\eta` has independent, mean-zero,
:math:`\psi_\alpha` (sub-exponential-type) rows with bounded variance.
*Remark.* SC is *inherently* an error-in-variables problem -- donor
series are noisy proxies for their signal -- which is precisely the
regime where regressing on the *raw* donors (or Lasso/Ridge) loses
consistency; the HSVT pre-step is what restores it.

*Assumption 3 (approximate linear SC).* The treated signal lies
(approximately) in the span of the donor signals,
:math:`y_1 = M^\top f^\star + \varepsilon + \phi`, with bounded
:math:`\|f^\star\|` and deterministic mismatch :math:`\phi`. *Remark.*
This relaxes the classical convex-hull (non-negative, sum-to-one)
restriction: the natural solver is unconstrained OLS on the denoised
donors (no simplex), which is the default below; the SIMPLEX option
re-imposes the convex hull when interpretability is preferred.

*Assumption 4 (independent noise sources).* The response noise
:math:`\varepsilon`, covariate noise :math:`\eta`, and the missingness
pattern are mutually independent (Agarwal et al. 2021, Rmk. 3.2).
*Remark.* PCR is **noise-model-agnostic**: unlike the error-in-variables
literature it needs no knowledge of the noise covariance, which is what
makes it practical for panels with unknown shock structure.

**Finite-sample guarantees.** Under Assumptions 1-4 the pre-period
(training) error of the HSVT+OLS estimator decomposes into three
interpretable pieces (Agarwal et al. 2021, Thm. 3.1):

.. math::

   \mathrm{MSE}_{\mathrm{pre}}(\widehat y_1)
     \;\lesssim\;
     \underbrace{\frac{\sigma^2 r}{T_0}}_{\text{regression}}
     + \underbrace{\frac{\|f^\star\|^2}{T_0}\,
        \mathbb{E}\bigl\|\mathrm{HSVT}_r(Z) - M\bigr\|_{2,\infty}^2}_{\text{donor corruption}}
     + \underbrace{\frac{\|\phi\|^2}{T_0}}_{\text{mismatch}} ,

where the donor-corruption term is controlled by a novel
:math:`\ell_{2,\infty}`-norm bound on HSVT (Lemma 3.1, stronger than the
usual Frobenius bound). With observation fraction :math:`\rho`, the
overall pre-period rate is :math:`\rho^{-4} r / \min(T_0, J) +
\|\phi\|^2/T_0` (Cor. 3.1) -- i.e. PCR matches, up to log factors, the
minimax OLS rate one would get with *perfectly observed* donors, despite
seeing only noisy, partially observed ones. The post-period (test) error
is bounded by the training error plus a generalization penalty scaling as
:math:`r^{5/2}/\sqrt{T_0}` (Thm. 3.2), which is exactly what justifies the
**data-driven rank selection** (cumulative-variance / USVT rules) used
below: choose :math:`r` to trade pre-period fit against this complexity
penalty.

**HSVT denoising.** mlsynth follows the Amjad-Shah-Shen (2018)
convention and applies HSVT to the *pre-period* donor matrix only.
The full-matrix variant proposed in Rho et al. (2025) Algorithm 2
(SVD on the entire :math:`(T, J)` panel, then slice the pre-period
rows) leaks post-period donor information into the rank-:math:`r`
reconstruction, which can wash out the very post-period deviations
the synthetic control is meant to detect. The user can opt into
the full-matrix variant via :py:attr:`CLUSTERSCConfig.project_denoised`,
in which case HSVT is also applied to :math:`(Y_0)` for the
projection step.

Let :math:`Y_0^- = U \Sigma V^\top` be the SVD of the pre-period
donor matrix. The rank-:math:`r` hard truncation

.. math::

   \widetilde M^- = \mathrm{HSVT}_r(Y_0^-)
                  = \sum_{i = 1}^{r} \sigma_i u_i v_i^\top

isolates the low-rank signal :math:`M^-`. The truncation rank can
be chosen three ways:

* **Cumulative variance** (default; paper Section 6.1) -- smallest
  :math:`r` with :math:`\sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2
  \ge` ``cumvar_threshold`` (default :math:`0.95`).
* **Explicit rank.** Caller passes :math:`r` directly via the
  ``rank`` parameter; the dispatcher promotes
  ``rank_method="fixed"`` automatically.
* **USVT** (Chatterjee 2015; Donoho-Gavish 2014). Universal
  threshold; preserved for back-compatibility with Amjad et al.
  2018.

For the data-driven rules (``"cumvar"`` and ``"usvt"``), the
spectral comparison is computed on the column-centred donor
matrix by default (``standardize_for_rank=True``). Each donor is
demeaned over time; the variance scale is left untouched.
Otherwise the leading singular value of an uncentered panel
(e.g. cigarette sales in absolute units) absorbs the overall
scale and dwarfs the remaining components, leaving
``cumvar_threshold`` unable to discriminate. Full z-scoring
(also dividing by the donor standard deviations) is avoided
because it equalises donor variances and artificially inflates
the rank that a cumvar threshold picks. Centring is *only* used
for rank picking -- the HSVT step itself still consumes the raw
matrix so the counterfactual is returned in original units.

The same rank is applied to the full :math:`Y_0` so that the
post-period projection (Algorithm 4 Step 5) consumes the denoised
matrix :math:`\widetilde M = \mathrm{HSVT}_r(Y_0)`.

**Donor clustering** (Algorithm 3). When ``clustering=True`` the
estimator clusters donors on the rows of
:math:`\widetilde U = U \Sigma_r`. With :math:`k` chosen by the
silhouette coefficient (Rousseeuw 1987) over
:math:`k \in [2, k_{\max}]`, run :math:`k`-means and embed the
treated unit via :math:`\tilde u = V_r^\top y_1^-`; the donor pool
becomes the cluster minimising :math:`\|c_\ell - \tilde u\|_2`
(Algorithm 4 Step 2). The selected donor sub-matrix is denoised
again at the same rank before the weight step.

**OLS weights** (Algorithm 2 Step 3). Drop the simplex constraints
of canonical synthetic control and solve

.. math::

   \widehat f = \arg\min_{f \in \mathbb{R}^J}
                \bigl\| \widetilde M^- f - y_1^- \bigr\|_2^2
              = (\widetilde M^-)^{+} y_1^-,

with :math:`(\widetilde M^-)^{+}` the Moore-Penrose pseudo-inverse.
Appendix E of the paper compares this OLS path to ridge / lasso
variants; mlsynth exposes optional elastic-net knobs
(``lambda_penalty``, ``p``, ``q``) for the same purpose.

The counterfactual is

.. math::

   \widehat y^0_1 = Y_0\, \widehat f,
   \qquad
   \widehat{ATT} = \frac{1}{T - T_0} \sum_{t > T_0}
                   \bigl(y_{1, t} - (Y_0\, \widehat f)_t\bigr).

Algorithm 4 Step 5 of Rho et al. (2025) writes the projection
through the *denoised* matrix :math:`\widetilde M`. For the OLS
solver the two are mathematically identical -- :math:`\widehat f`
lies in the column space of :math:`V_r`, so the discarded
high-order components annihilate. They differ for the Bayesian and
SIMPLEX solvers below, where :math:`\widehat f` is not constrained
to the rank-:math:`r` subspace. Set ``project_denoised=True`` to
recover the paper-strict projection.

mlsynth extensions
^^^^^^^^^^^^^^^^^^

Two paper-extensible weight solvers live alongside the OLS default:

* **Bayesian PCR** (``estimator="bayesian"``). This is the Bayesian
  Robust Synthetic Control of Amjad, Shah & Shen [Amjad2018]_: replace
  the point-estimate OLS with a Gaussian posterior over the weights
  (Bayesian linear regression on the HSVT-denoised donors), which
  yields calibrated uncertainty directly rather than by resampling.
  Concretely,

  .. math::

     f \mid y_1^-, \widetilde M^- \sim
       \mathcal N\bigl(\mu_n, \Sigma_n\bigr),
     \quad
     \Sigma_n = \bigl( \sigma_e^{-2} (\widetilde M^-)^\top
                       \widetilde M^- + \alpha_0 I \bigr)^{-1},
     \quad
     \mu_n = \sigma_e^{-2} \Sigma_n (\widetilde M^-)^\top y_1^-,

  with :math:`\sigma_e^2 = \mathrm{Var}(y_1^-)` and prior precision
  :math:`\alpha_0`. ``n_bayes_samples`` posterior draws are
  propagated through :math:`\widetilde M` to yield per-period
  credible bands at level :math:`1 - \alpha`. Aggregated over the
  post period, the implied ATT credible interval is reported on
  :py:class:`CLUSTERSCInference`.

* **Convex PCR** (``pcr_objective="SIMPLEX"``). Keep the HSVT
  denoising of the donor matrix but solve the classical
  Abadie-Diamond-Hainmueller (2010) program,

  .. math::

     \widehat f = \arg\min_{f \in \Delta_J}
                    \bigl\| \widetilde M^- f - y_1^- \bigr\|_2^2,
     \qquad
     \Delta_J = \bigl\{ f \in \mathbb{R}_{\ge 0}^J :
                        \mathbf 1^\top f = 1 \bigr\}.

  Useful when the user wants non-extrapolation and an interpretable
  convex-combination donor weighting on top of the PCR denoising.

RPCA-SC family (Bayani 2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RPCA family follows the same five-step skeleton as PCR-SC but
swaps the feature step, the denoising step, and the weight
constraints. Algorithm 4 of the paper is the orchestrator.

**Step 1 -- Functional PCA on pre-period trajectories.** Each unit's
pre-period series :math:`y_j^- (t)` is smoothed via a cubic B-spline
expansion and projected onto the FPC basis
:math:`\{\phi_k\}_{k=1}^K` (Li, Wang & Carroll 2010):

.. math::

   y_j^-(t) \approx \mu(t)
                  + \sum_{k=1}^{K} \xi_{jk}\, \phi_k(t),
   \qquad
   \widehat \xi_{jk} =
       \int_a^b (y_j^-(t) - \widehat \mu(t))\, \widehat \phi_k(t)\, dt.

The rank :math:`K` is the smallest integer whose cumulative
spectral energy meets ``fpca_cumvar`` (paper default :math:`0.95`).
The scores are standardised before clustering.

**Step 2 -- :math:`k`-means clustering.** Apply Hartigan-Wong
(1979) :math:`k`-means to the FPC scores with :math:`k` chosen by
the silhouette coefficient

.. math::

   s(j) = \frac{b(j) - a(j)}{\max\{a(j), b(j)\}},

where :math:`a(j)` is the mean distance from unit :math:`j` to other
members of its own cluster and :math:`b(j)` the mean distance to
the closest neighbouring cluster. The donor pool
:math:`\mathcal{D} \subseteq \{2, \dots, J + 1\}` is the set of
non-treated units sharing the treated unit's cluster.

**Step 3 -- Robust PCA on the donor pool.** Stack the cluster donor
outcomes into :math:`Y_{-1} \in \mathbb{R}^{|\mathcal{D}| \times T}`.
mlsynth offers two robust decompositions :math:`Y_{-1} = L + S`.

*PCP* (Candes et al. 2011) replaces the NP-hard
:math:`\min_{L, S} \mathrm{rank}(L) + \lambda \|S\|_0` with the
convex relaxation

.. math::

   \min_{L, S} \; \| L \|_* + \lambda \| S \|_1
   \quad \text{s.t.} \quad Y_{-1} = L + S,

solved by ADMM with default penalties
:math:`\lambda = 1 / \sqrt{\max(|\mathcal D|, T)}` and
:math:`\mu = |\mathcal D| T / (4 \sum_{i, t} |Y_{-1, it}|)` per
Bayani Section 2.4. Each iteration alternates a singular-value
soft-threshold update on :math:`L`, an entry-wise soft-threshold
update on :math:`S`, and the standard dual ascent on the multiplier
:math:`\Lambda`.

*HQF* (Wang et al. 2023) instead factors :math:`L = UV` directly and
alternates two Tikhonov-regularised least-squares updates,

.. math::

   U \leftarrow \bigl((Y_{-1} - S) V^\top + \lambda U_{\text{prev}}\bigr)
                    (V V^\top + \lambda I)^{-1},
   \quad
   V \leftarrow (U^\top U + \lambda I)^{-1}
                    (U^\top (Y_{-1} - S) + \lambda V_{\text{prev}}),

with the sparse component updated by a median-absolute-deviation
threshold on the residual (controlled by ``hqf_ip``). The rank
defaults to the smallest :math:`r` whose cumulative singular-value
energy meets ``hqf_cumvar`` (Bayani uses :math:`0.999`).

**Step 4 -- Non-negative least squares.** Let
:math:`L^- \in \mathbb{R}^{|\mathcal{D}| \times T_0}` be the
pre-period slice of the low-rank component. The weights solve

.. math::

   \widehat \beta = \arg\min_{\beta \ge 0}
                    \bigl\| y_1^- - (L^-)^\top \beta \bigr\|_2^2.

The simplex constraint of canonical synthetic control is
deliberately dropped: the clustering step has already restricted
the donor pool to behaviourally similar units, so non-negativity
suffices to keep the counterfactual interpretable (Bayani 2021,
Section 2.4).

**Step 5 -- Projection.** Counterfactual and ATT come from the
denoised donor matrix,

.. math::

   \widehat y^0_1 = L^\top \widehat \beta,
   \qquad
   \widehat{ATT} = \frac{1}{T - T_0}
                    \sum_{t > T_0}
                    \bigl(y_{1, t} - (L^\top \widehat \beta)_t\bigr).

RPCA-SC tuning via leave-one-time-out cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayani (2021) takes the Candes-Li-Ma-Wright (2011) optimal-recovery
value :math:`\lambda^\star = 1/\sqrt{\max(|\mathcal D|, T)}` for
PCP, and chooses the HQF factorisation rank by an explained-variance
threshold. Both defaults are conservative for *L/S decomposition
identifiability* but can be poor for *counterfactual prediction*.
On the California Proposition 99 panel the Candes default leaves
PCP under-regularised by roughly 2x and HQF stops at a rank that
loses ATT magnitude.

mlsynth therefore exposes two opt-in cross-validation tuners
(:py:attr:`CLUSTERSCConfig.cv_lambda` for PCP,
:py:attr:`CLUSTERSCConfig.cv_hqf_rank` for HQF). The same
algorithm drives both:

Let :math:`L \in \mathbb{R}^{|\mathcal D| \times T}` be the low-rank
component returned by the robust PCA solver at candidate
hyperparameter :math:`\theta`. The pre-period slice
:math:`L^- \in \mathbb{R}^{|\mathcal D| \times T_0}` is the design
matrix for the NNLS weight step. Define the leave-one-time-period-out
mean squared error

.. math::

   \mathrm{CV}(\theta) = \frac{1}{T_0} \sum_{t = 1}^{T_0}
                         \biggl( y_{1, t}
                                 - (L^-_{:, -t})^\top \widehat \beta^{(-t)}
                         \biggr)^2,
   \quad
   \widehat \beta^{(-t)} = \arg\min_{\beta \ge 0}
                           \bigl\| y_{1, -t}^-
                                  - (L^-_{:, -t})^\top \beta \bigr\|_2^2,

where the :math:`-t` subscript drops the :math:`t`-th pre-period
column. The donor matrix :math:`Y_{-1}` is fully observed at every
period so the RPCA decomposition is run once per :math:`\theta`,
not :math:`T_0` times. The chosen :math:`\widehat \theta` minimises
:math:`\mathrm{CV}(\theta)` over a grid:

* For PCP, the grid is ``cv_lambda_multipliers`` :math:`\times
  \lambda^\star` with default multipliers
  :math:`(0.5, 1, 2, 3, 5, 8, 12)`.
* For HQF, the grid is the integer ranks
  :math:`\{1, 2, \dots, \min(|\mathcal D|, T_0 - 1)\}`.

On the Proposition 99 panel, ``cv_lambda=True`` picks
:math:`\widehat \lambda = 2 \lambda^\star` and cuts pre-period RMSE
in half (2.11 → 1.08, ATT moving from −15.5 to −17.7);
``cv_hqf_rank=True`` picks :math:`\widehat r = 8` (vs. cumvar
default of 4) with ATT moving from −12.8 to −16.6. Both ATTs land
in the canonical SCM / RSC range of −19 to −24.

When the assumptions bind: practical diagnostics
-------------------------------------------------

Assumptions 1-4 above (Agarwal et al. 2021, transcribed in the
PCR-SC family) are written as structural conditions. Here is how
each shows up in real data and what to check in the
:py:class:`CLUSTERSCResults` container before trusting the
counterfactual.

(a) **Low-rank latent-factor signal (A1).** The donor pool's
    untreated mean :math:`M` is approximately low rank. If the
    panel is genuinely full-rank or the spectrum decays slowly,
    HSVT throws away signal and the OLS step over-fits whatever
    is left.

    *Plausibly violated when* donor series are idiosyncratic in a
    way that no factor model can absorb -- think 200 firm-level
    time series where each firm has its own unrelated business
    cycle. *Diagnostic*: compute the spectral-energy share of the
    top-:math:`r` singular values of the column-centred donor
    matrix; if you need :math:`r \ge \min(T_0, J) / 2` to capture
    :math:`90\%` of the energy, the low-rank story is failing.
    On the Prop 99 panel the top-1 singular value carries
    :math:`\approx 99\%` of the energy -- the exact regime where
    PCR-SC dominates. When the spectrum decays slowly, fall back
    to canonical SC (:doc:`scm`, :doc:`tssc`) or to a
    balancing-aware estimator (:doc:`microsynth` for unit-level
    data; :doc:`fma` for factor-model-aware estimation).

(b) **Error-in-variables donors (A2).** Donor entries are
    noisy / partially-observed views of the signal. The HSVT step
    is precisely what restores consistency when this assumption
    holds. The condition fails *softly* when the noise is sparse
    and heavy-tailed instead of Gaussian, *hard* when missingness
    is non-random (donor entries missing **because** the donor
    deviates from the signal).

    *Plausibly violated when* you have a panel with structural
    outliers (a donor with a one-time policy shock spike) rather
    than i.i.d. Gaussian noise. *Diagnostic*: residualise each
    donor against the rank-:math:`r` HSVT reconstruction and
    histogram the residuals; sparse heavy tails are a red flag.
    The fix is the **RPCA-SC family** -- robust :math:`L + S`
    decomposition explicitly separates the low-rank signal from
    the sparse outliers, so heavy-tailed donor noise is absorbed
    into :math:`S` rather than contaminating :math:`L`.

(c) **Approximate linear SC (A3).** The treated signal lies
    (approximately) in the span of the denoised donor signals.
    PCR-SC uses unconstrained OLS so this relaxes the canonical
    convex-hull restriction -- but the model-mismatch term
    :math:`\phi_t` (the gap between the true treated mean and the
    closest linear combination of donor signals) still bounds the
    finite-sample error.

    *Plausibly violated when* the treated unit is structurally
    outside the donor pool -- coastal vs. interior states, a
    tech-led economy with only commodity-led donors. *Diagnostic*:
    inspect ``res.pcr.pre_rmse`` against the donor noise floor
    (the mean leave-one-out pre-RMSE across donors). If treated
    pre-RMSE is substantially higher than the donor noise floor,
    the mismatch :math:`\phi_t` is large and PCR-SC's
    unbiasedness argument breaks. For the structurally outside-
    hull case, switch to :doc:`iscm` (whose A2(b) mechanism
    identifies the effect through donors that use the treated
    unit as a *donor*).

(d) **Independent noise sources (A4).** Response noise, donor
    noise, and the missingness pattern are mutually independent.
    The PCR-SC consistency results assume the missingness pattern
    is data-independent.

    *Plausibly violated when* donor observations are missing
    **because** of the outcome value (e.g. countries stopping
    reporting GDP during recessions). *Diagnostic*: cross-tabulate
    missingness with the pre-period outcome quantiles; if missing
    entries are concentrated in the tails, the assumption fails.
    Multiple imputation pre-step or a missing-data-aware estimator
    is the fix; mlsynth's :doc:`snn` is built for missingness-
    informative panels.

(e) **Cluster structure (Rho et al. 2025, ClusterSC-specific).**
    The donor pool decomposes into latent subgroups distinguishable
    on the right-singular-vector embedding. Clustering only helps
    if such subgroups actually exist and the silhouette statistic
    can find them.

    *Plausibly violated when* the donor pool is genuinely
    homogeneous -- a tight set of comparable units already
    pre-screened by the analyst. *Diagnostic*: read
    ``res.pcr.metadata['k_clusters']``; if the silhouette picks
    :math:`k = 1`, clustering adds no value and you should set
    ``clustering=False`` to use the full pool. If :math:`k > 1`
    is picked but the post-fit ATT moves substantially when you
    flip ``clustering`` off and on, the cluster structure is
    *spurious* (cluster boundaries are within the noise floor)
    and the un-clustered fit is the more honest one.

(f) **Functional smoothness (Bayani 2021, RPCA-SC-specific).**
    Pre-period trajectories admit a parsimonious FPCA basis. If
    trajectories are dominated by high-frequency jagged noise the
    cubic-spline FPCA pre-step throws away most of the signal.

    *Plausibly violated when* the outcome is noisy at the
    observation frequency (daily stock prices, hourly sensor
    readings). *Diagnostic*: read
    ``res.rpca.metadata['fpca_components']``; if the silhouette
    or cumulative-variance step keeps a large number of FPC
    components, the smoothness assumption is failing. Aggregate
    to a coarser time grid (weekly, daily) before refitting, or
    move to a stationary-cycle estimator (:doc:`sbc`) that
    handles unsmoothed series natively.

(g) **NNLS-friendly truth (Bayani 2021, RPCA-SC-specific).**
    RPCA-SC ends with a non-negative least-squares step. If the
    true counterfactual is best represented as an *extrapolation*
    (some donor weights would naturally be negative), NNLS
    cannot reach it.

    *Plausibly violated when* the treated unit is at the edge of
    the donor distribution and the un-clustered PCR-SC OLS fit
    returns substantial negative weights. *Diagnostic*: refit
    with PCR-SC + OLS (``method="pcr"``, ``pcr_objective="OLS"``)
    and inspect the weight vector. If many large negative weights
    appear, the convex-combination restriction is binding and
    RPCA-SC is throwing away identification.

When to use PCR-SC, RPCA-SC, or neither
---------------------------------------

The four papers behind CLUSTERSC chip at different parts of the
canonical SC pipeline. The decision logic for picking among them:

**Reach for PCR-SC + clustering (the CLUSTERSC default) when:**

* The donor pool is **large and noisy** (:math:`J \gtrsim T_0`,
  disaggregated panels with hundreds of donors), and the donor
  matrix has a clear low-rank spectrum. This is the Rho et al.
  (2025) regime -- the empirical case studies are individual-
  level health records and disaggregated economic panels.
* The treated unit comes from a **plausible subgroup** of the
  donor pool that you cannot easily isolate by hand (similar
  patient phenotype, similar state-economy composition). The
  silhouette-driven :math:`k`-means step formalises the
  subgroup decision.
* You're willing to **let weights be negative** in the
  un-clustered OLS solve (Assumption 3 above) for tighter
  pre-fit and lower bias -- and you accept that "California =
  +0.4 Utah +0.3 Montana -0.1 Tennessee" is a defensible weight
  story for your application.

**Reach for PCR-SC without clustering (i.e., classic Amjad-Shah-Shen
RSC) when:**

* You have a **moderate-size donor pool** that you have already
  pre-screened, and you mostly want the HSVT denoising step to
  protect against donor-matrix noise / missingness. Pre-screening
  has done the work that clustering would do.
* You want the **Shen-Ding-Sekhon-Yu (2023) closed-form CIs**.
  These are wired in for ``pcr_objective="OLS"`` and give you
  proper frequentist HZ/VT/DR-source standard errors without a
  bootstrap; the Bayesian path delivers credible bands.
* The empirical setting is one of the **canonical aggregate SC
  case studies** (Prop 99, Basque, etc.) where the donor pool is
  small (J ≤ 40) and a hand-picked subgroup already exists.

**Reach for RPCA-SC when:**

* The donor matrix has **sparse heavy-tailed outliers** rather
  than uniform Gaussian noise -- a few donors with one-time
  policy shocks, structural breaks, or recording errors. The
  :math:`L + S` decomposition explicitly absorbs these into
  :math:`S`, leaving a clean :math:`L` for the weight fit.
  PCR-SC's HSVT is an :math:`L_2`-based denoiser and is
  *less* robust to this exact regime.
* Pre-period trajectories are **smooth functional curves**
  (annual GDP, monthly population) where the FPCA basis is a
  natural language for the donor pool. RPCA-SC's Step 1 was
  built for this.
* You want **non-negative interpretable weights** -- the NNLS
  step at the end produces a sparse convex-combination story
  similar to canonical SC, while still benefiting from the
  robust-PCA denoising. The German-reunification application
  is the canonical case (Norway 0.48, France 0.35, New Zealand
  0.30, Austria 0.02).
* You want **Cattaneo-Feng-Titiunik (2021) prediction intervals**
  -- mlsynth's CFT port runs on RPCA-SC, providing in-sample
  bootstrap + Hoeffding out-of-sample bands.

**Use PCR-SC over RPCA-SC when:**

* The noise is Gaussian / sub-Gaussian (PCR-SC's HSVT is
  optimal under this), the donor matrix is high-dimensional
  (clustering helps), and you want negative weights / closed-
  form Shen CIs.
* You need speed -- PCR-SC is **one SVD + one OLS**, while
  RPCA-SC's PCP is an ADMM loop and HQF is an iterative
  factorisation. Differences are small on Prop99-size panels;
  they bite on disaggregated panels with thousands of donors.

**Use RPCA-SC over PCR-SC when:**

* The donor matrix has visible sparse outliers (one or two
  donors with a single shock period that would dominate the
  HSVT spectrum). RPCA's :math:`L + S` decomposition is built
  for exactly that pattern.
* Trajectories are smooth and you want the **donor-pool
  reduction** that FPCA + :math:`k`-means delivers ahead of
  the weight fit (this is what selects the 11-economy
  West-Germany cluster from the 17-country OECD pool).
* You want a **non-negative interpretable** weight vector for
  policy storytelling.

**Do not use either family when:**

* **The donor matrix has no low-rank structure.** Both PCR-SC
  and RPCA-SC depend on this. With high-dimensional donor
  pools where the spectrum decays slowly, switch to a
  balancing-aware estimator (:doc:`microsynth` if you have
  user-level data; :doc:`fma` if a factor-model fit is
  defensible).
* **The treated unit is structurally outside the donor convex
  hull** in a way that no linear combination can capture --
  even with negative weights, the pre-RMSE stays large. Switch
  to :doc:`iscm`, whose A2(b) mechanism identifies the effect
  through donors that use the treated unit as a positive-
  weight donor in their own synthetic controls.
* **Distributional questions** (Lorenz curves, QTEs, tail
  changes). PCR-SC and RPCA-SC target the mean ATT. Use
  :doc:`dsc` instead.
* **Continuous or multi-valued treatment.** The CLUSTERSC
  pipeline encodes a single binary treatment indicator.
  Continuous dose belongs in :doc:`ctsc`.
* **Spillovers / interference across donors.** The low-rank
  signal model assumes donors are untreated and independent of
  the treatment of unit 1. Spillovers break this; use
  :doc:`spillsynth` or :doc:`spsydid`.
* **Tiny donor pool** (:math:`J \le 10`) and a tight canonical-
  SC pre-fit. The denoising and clustering machinery is
  overkill; both add variance without identification gain.
  Use :doc:`scm`, :doc:`tssc`, or :doc:`fdid`.
* **Staggered adoption with a long mixed-treatment pre-period.**
  CLUSTERSC assumes a common pre-period free of treatment.
  Use :doc:`fect` or :doc:`sdid`.
* **You need the donor weight vector to be a single sparse
  convex combination** as the headline story (and you're not
  willing to use RPCA-SC's NNLS variant). The PCR-SC
  default is the unrestricted OLS solve; the SIMPLEX
  variant re-imposes the convex hull at the cost of the
  denoising-bias gain. If the sparse convex combination is
  the deliverable, :doc:`scm` or :doc:`tssc` are the more
  honest default.

Inference
^^^^^^^^^

Three inference families are wired into :py:class:`CLUSTERSCInference`:

* **Frequentist PCR -- Shen-Ding-Sekhon-Yu (2023) closed-form CIs.**
  Default on for ``estimator="frequentist"`` and
  ``pcr_objective="OLS"``. See the next subsection.
* **Bayesian PCR -- posterior credible interval.** Computed when
  ``estimator="bayesian"`` from posterior draws of the counterfactual
  (the Bayesian Robust Synthetic Control of Amjad, Shah & Shen
  [Amjad2018]_).
* **RPCA-SC -- Cattaneo-Feng-Titiunik (2021) prediction
  intervals.** Opt-in via ``CLUSTERSCConfig.compute_cft_pi``. See
  the dedicated subsection below.

Shen-Ding-Sekhon-Yu (2023) frequentist CIs for OLS PCR
""""""""""""""""""""""""""""""""""""""""""""""""""""""

For the symmetric estimator class -- OLS minimum :math:`\ell_2`-norm,
PCR, and ridge -- Theorem 1 of Shen et al. (2023) shows that the
horizontal (HZ) and vertical (VT) regression formulations give
algebraically identical point estimates. The two formulations
nevertheless quantify uncertainty against *different* generative
models:

* **HZ model** (Assumption 1). Each donor's post-period outcome is a
  noisy linear combination of its own pre-period values:

  .. math::

     Y_{i, T} = \sum_{t \le T_0} \alpha^*_t Y_{i, t} + \varepsilon_{i, T},
     \quad i = 1, \dots, N_0.

  The randomness lives in the **cross-sectional** dimension.

* **VT model** (Assumption 2). The treated unit's pre-period outcome
  is a noisy linear combination of the donors' pre-period values:

  .. math::

     Y_{N, t} = \sum_{i \le N_0} \beta^*_i Y_{i, t} + \varepsilon_{N, t},
     \quad t = 1, \dots, T_0.

  The randomness lives in the **time-series** dimension.

* **DR model** (Assumption 3). Both sources of randomness are present.

Each model yields a distinct estimand and a distinct asymptotic
variance for the same point estimate
:math:`\widehat Y_{N, T}(0)` (Theorem 3). With rank-:math:`k` HSVT
projections :math:`H^u_\perp = I - U_k U_k^\top` (donor-space) and
:math:`H^v_\perp = I - V_k^\top V_k` (time-space), and the
homoskedastic variance plug-ins (paper eq 19):

.. math::

   \widehat \sigma^2_{\mathrm{hz}} =
       \frac{\| H^u_\perp y_T \|_2^2}{N_0 - R},
   \qquad
   \widehat \sigma^2_{\mathrm{vt}} =
       \frac{\| H^v_\perp y_N \|_2^2}{T_0 - R},

(where :math:`R = \mathrm{rank}(Y_0)` after truncation), the per-period
variance estimators are

.. math::

   \widehat v_{\mathrm{hz}} = \widehat \sigma^2_{\mathrm{hz}}\,
                              \| \widehat \beta \|_2^2,
   \quad
   \widehat v_{\mathrm{vt}} = \widehat \sigma^2_{\mathrm{vt}}\,
                              \| \widehat \alpha \|_2^2,
   \quad
   \widehat v_{\mathrm{dr}} = \max\!\bigl(0,\,
                              \widehat v_{\mathrm{hz}}
                            + \widehat v_{\mathrm{vt}}
                            - \mathrm{tr}\widehat A\bigr),

with :math:`\widehat A = \widehat \sigma^2_{\mathrm{hz}}
\widehat \sigma^2_{\mathrm{vt}}\, Y_0^{+} (Y_0^\top)^{+}` the
interaction term. The :math:`(1 - \alpha)` CI under source
:math:`s \in \{\mathrm{hz}, \mathrm{vt}, \mathrm{dr}\}` is

.. math::

   \widehat Y_{N, T}(0) \pm z_{\alpha/2}\, \sqrt{\widehat v_s}.

mlsynth also ports the **jackknife** and **HRK (Hartley-Rao-Kish)**
variance estimators from var.py in the authors' reference repository.
The HRK estimator is only valid when
:math:`\max_i (1 - (H_\perp)_{ii}) < 1/2` for both projections;
mlsynth checks this and raises if violated.

For multi-period extrapolation the procedure runs per post-period:
at each :math:`t > T_0` the donor outcomes :math:`y_t` change but
the projections, weights, and per-period variances are recomputed
from the same fitted weight pair. The ATT is the mean of per-period
gaps :math:`\widehat{ATT} = (T - T_0)^{-1} \sum_{t > T_0} \widehat \tau_t`,
and its variance is aggregated assuming independence across
post-periods:

.. math::

   \widehat v_s(\widehat{ATT}) =
       \frac{1}{T_1} \cdot
       \frac{1}{T_1} \sum_{t > T_0} \widehat v_s(t),
   \qquad
   T_1 = T - T_0.

The independence assumption is the standard first-pass; serially
correlated shocks would inflate the true variance.

Config knobs: :py:attr:`CLUSTERSCConfig.compute_shen_ci` (default
True) toggles the inference; :py:attr:`CLUSTERSCConfig.shen_variance`
chooses ``"homoskedastic"`` (default), ``"jackknife"``, or
``"hrk"``. The output :py:class:`CLUSTERSCInference.shen` carries
per-period gaps, per-period CIs under each source, and the
aggregated ATT CIs.

Cattaneo-Feng-Titiunik (2021) prediction intervals for RPCA-SC
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Bayani (2021) leaves inference unaddressed and the Shen et al. (2023)
closed-form CIs do not apply to the asymmetric NNLS weights of
RPCA-SC. For uncertainty quantification mlsynth ships a focused port
of *Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021), "Prediction
Intervals for Synthetic Control Methods"* (JASA 116(536):1865-1880),
opt-in via :py:attr:`CLUSTERSCConfig.compute_cft_pi`.

The CFT prediction error at post-period :math:`t > T_0` decomposes as

.. math::

   y_t - \widehat y^0_t
       = \underbrace{\bigl\langle p_t, w^* - \widehat w \bigr\rangle}_{
              \text{in-sample } u_t}
       + \underbrace{e_t}_{\text{out-of-sample shock}},

where :math:`p_t` is the loading at :math:`t` (for RPCA-SC the
denoised donor row :math:`L_t`), :math:`w^*` is the population
weight, and :math:`\widehat w` is the NNLS estimate. The two
components are quantified separately:

* **In-sample component** :math:`M_w(t, \alpha/2)`. The paper's
  reference implementation solves a constrained
  :math:`\sup / \inf` over the "compatible set" of weights via an
  ECOS LP at each post-period. mlsynth replaces that with an
  HC1-scaled parametric bootstrap of the pre-period residuals: at
  each of ``cft_sims`` draws,

  .. math::

     \widehat u^*_t = \sqrt{\tfrac{T_0}{T_0 - 1}} \cdot
                     \widehat u^-_{\sigma(t)},
     \quad \sigma \sim \mathrm{Unif}\{1, \dots, T_0\},

  we perturb the treated pre-period outcome
  :math:`y^*_t = \widehat y^0_t + \widehat u^*_t`, refit the full
  RPCA-SC pipeline, and collect the resulting counterfactual
  :math:`\widehat y^{0, *}` at every period. The asymmetric
  :math:`(\alpha/2, 1 - \alpha/2)` empirical quantiles of
  :math:`\widehat y^{0, *}_t - \widehat y^0_t` form the in-sample
  band per post-period. This is equivalent to the ECOS-based bound
  under regularity conditions and avoids pulling in ``ecos`` /
  ``dask`` / ``plotnine`` as hard dependencies.

* **Out-of-sample component** :math:`M_e(t, \alpha/2)`. Under
  sub-Gaussian post-period shocks with scale parameter
  :math:`\sigma_e`, Hoeffding's inequality gives the tail bound

  .. math::

     M_e(t, \alpha/2) = \sqrt{-2 \log \alpha} \; \widehat \sigma_e,
     \qquad
     \widehat \sigma_e = \mathrm{sd}\!\bigl(\widehat u^-\bigr).

  This is the ``gaussian`` variant of the paper's three
  out-of-sample options (``gaussian`` / ``ls`` / ``qreg``); the
  other two are deferred.

The combined :math:`(1 - \alpha)` PI on the counterfactual at
:math:`t` is

.. math::

   \widehat y^0_t \pm \bigl[ M_w(t, \alpha/2) + M_e(t, \alpha/2) \bigr],

which inverts to the PI on the per-period treatment effect
:math:`\widehat \tau_t = y_t - \widehat y^0_t`.

For the ATT, the in-sample component aggregates by storing the
*post-period mean of the counterfactual* at each bootstrap draw and
taking quantiles. The out-of-sample component shrinks by
:math:`\sqrt{T_1}` under post-period shock independence:

.. math::

   M_e^{\mathrm{ATT}}(\alpha/2)
       = \frac{\sqrt{-2 \log \alpha} \; \widehat \sigma_e}{\sqrt{T_1}}.

Config knobs: :py:attr:`CLUSTERSCConfig.compute_cft_pi` (default
False -- the bootstrap costs ``cft_sims`` full pipeline refits,
roughly 0.3-0.5 s each on a moderate panel),
:py:attr:`CLUSTERSCConfig.cft_sims` (default 200),
:py:attr:`CLUSTERSCConfig.cft_alpha` (default 0.05), and
:py:attr:`CLUSTERSCConfig.cft_e_method` (currently only
``"gaussian"``). The output
:py:class:`CLUSTERSCInference.cft` carries the per-period gaps,
per-period PIs, in-sample bootstrap bands, the Hoeffding constant,
and the aggregated ATT PI.

Core API
--------

.. automodule:: mlsynth.estimators.clustersc
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.CLUSTERSCConfig
   :members:
   :undoc-members:

Helper Modules
--------------

PCR-SC pipeline
^^^^^^^^^^^^^^^

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.hsvt
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.clustering
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.frequentist
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.bayesian
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.convex
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.pcr.pipeline
   :members:
   :undoc-members:

RPCA-SC pipeline
^^^^^^^^^^^^^^^^

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.fpca
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.clustering
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.pcp
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.hqf
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.weights
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.tuning
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.rpca.pipeline
   :members:
   :undoc-members:

Shared utilities
^^^^^^^^^^^^^^^^

.. automodule:: mlsynth.utils.clustersc_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.structures
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.clustersc_helpers.plotter
   :members:
   :undoc-members:

Example
-------

A self-contained one-draw Monte Carlo at a two-factor DGP, fitting
both PCR-SC and RPCA-SC in parallel and inspecting their fits via
the frozen :py:class:`CLUSTERSCResults` container.

.. code-block:: python

   """One draw of a two-factor CLUSTERSC simulation."""

   import numpy as np
   import pandas as pd

   from mlsynth import CLUSTERSC


   # ---------------------------------------------------------------------
   # 1. Simulate one panel from a two-factor DGP
   # ---------------------------------------------------------------------

   rng = np.random.default_rng(0)
   J = 12             # control units
   T_pre = 14
   T_post = 6
   T = T_pre + T_post
   r_true = 2
   tau_true = 1.0     # additive treatment effect on the treated unit

   F = rng.standard_normal((T, r_true))
   lam = rng.standard_normal((J + 1, r_true))
   eps = rng.standard_normal((T, J + 1)) * 0.4
   Y = F @ lam.T + eps
   Y[T_pre:, 0] += tau_true                              # unit 0 treated

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
   # 2. Fit CLUSTERSC with both families and Bayesian PCR inference
   # ---------------------------------------------------------------------

   res = CLUSTERSC({
       "df": df,
       "outcome": "y",
       "treat": "D",
       "unitid": "unit",
       "time": "time",
       "method": "both",
       "primary": "pcr",
       "pcr_objective": "OLS",
       "estimator": "bayesian",
       "rpca_method": "HQF",
       "alpha": 0.10,
       # No cluster structure in the synthetic panel; pool donors for
       # the RPCA branch so its Algorithm 4 Step 2 does not isolate
       # the treated row.
       "k_clusters": 1,
   }).fit()


   # ---------------------------------------------------------------------
   # 3. Inspect the output
   # ---------------------------------------------------------------------

   print(f"true tau              = {tau_true:+.3f}")
   print(f"selected variant      = {res.selected_variant}")
   print(f"primary ATT           = {res.att:+.3f}")
   print(f"PCR  ATT              = {res.pcr.att:+.3f}")
   print(f"RPCA ATT              = {res.rpca.att:+.3f}")
   print(f"PCR  pre-RMSE         = {res.pcr.pre_rmse:.4f}")
   print(f"RPCA pre-RMSE         = {res.rpca.pre_rmse:.4f}")
   print(f"HSVT rank (PCR)       = {res.pcr.metadata['rank']}")
   print(f"HQF  rank (RPCA)      = {res.rpca.metadata['hqf_rank']}")
   if res.inference.method == "bayesian_credible":
       lo, hi = res.inference.credible_interval
       print(f"Bayesian 90% CrI ATT  = [{lo:+.3f}, {hi:+.3f}]")

Empirical example: California Proposition 99
--------------------------------------------

The PCR-SC family also runs without clustering, in which case it
reduces to Amjad-Shah-Shen (2018) Robust Synthetic Control on the
full donor pool. Below: PCR with explicit rank :math:`r = 4`, then
inspect the Shen-Ding-Sekhon-Yu (2023) frequentist CIs that the
default ``compute_shen_ci=True`` produces.

.. code-block:: python

   from mlsynth import CLUSTERSC
   import pandas as pd

   file = (
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df = pd.read_csv(file)

   res = CLUSTERSC({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",
       "unitid": "state",
       "time": "year",
       "method": "pcr",
       "clustering": False,
       "rank": 4,                 # explicit HSVT rank
       "display_graphs": True,
   }).fit()

   print(f"ATT             = {res.att:+.3f}")
   shen = res.inference.shen
   print(f"95% CI (VT src) = [{shen.att_ci_vt[0]:+.2f}, {shen.att_ci_vt[1]:+.2f}]")
   print(f"95% CI (HZ src) = [{shen.att_ci_hz[0]:+.2f}, {shen.att_ci_hz[1]:+.2f}]")

Same panel, RPCA-SC with cross-validated PCP :math:`\lambda`. The
``cv_lambda=True`` flag triggers leave-one-time-out CV over
``cv_lambda_multipliers`` :math:`\times \lambda^\star` to pick the
prediction-optimal sparsity penalty.

.. code-block:: python

   res = CLUSTERSC({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",
       "unitid": "state",
       "time": "year",
       "method": "rpca",
       "rpca_method": "PCP",
       "cv_lambda": True,
       "k_clusters": 1,            # weak cluster structure on this panel
   }).fit()

   md = res.rpca.metadata
   print(f"PCP lambda chosen by CV = {md['pcp_lambda']:.4f} "
         f"(best of grid {[round(x,3) for x in md['cv_lambda_grid']]})")
   print(f"pre-RMSE  = {res.rpca.pre_rmse:.3f}")
   print(f"ATT       = {res.rpca.att:+.3f}")

Same fit with Cattaneo-Feng-Titiunik (2021) prediction intervals
(opt-in via ``compute_cft_pi=True``; the bootstrap reruns the full
pipeline ``cft_sims`` times, so this takes ~30s on Prop 99 with
the default ``cft_sims=200``).

.. code-block:: python

   res = CLUSTERSC({
       "df": df,
       "outcome": "cigsale",
       "treat": "Proposition 99",
       "unitid": "state",
       "time": "year",
       "method": "rpca",
       "rpca_method": "PCP",
       "cv_lambda": True,
       "k_clusters": 1,
       "compute_cft_pi": True,
       "cft_sims": 200,
       "cft_alpha": 0.05,
   }).fit()

   cft = res.inference.cft
   print(f"ATT                = {res.att:+.3f}")
   print(f"95% CFT PI on ATT  = [{cft.att_pi[0]:+.2f}, {cft.att_pi[1]:+.2f}]")
   print(f"sigma_e (pre-RMSE) = {cft.sigma_e:.3f}")
   print(f"out-of-sample eps  = {cft.out_of_sample_eps:.3f}")
   for ti in range(len(cft.per_period_gap)):
       lo, hi = cft.per_period_pi[ti]
       print(f"  t={res.inputs.T0+ti}:  tau_t={cft.per_period_gap[ti]:+.2f}  "
             f"PI=[{lo:+.2f}, {hi:+.2f}]")

Empirical validation: West German reunification (Bayani 2021)
-------------------------------------------------------------

Bayani (2021, Section 3) validates RPCA-SC on the German
reunification panel of Abadie, Diamond & Hainmueller (2015): the
1990 reunification is treated as an intervention on West German
per-capita GDP, with 16 OECD economies as the candidate donor pool.
It is the canonical empirical benchmark for the RPCA-SC family --
the analogue of Proposition 99 for PCR-SC -- and mlsynth reproduces
it from the bundled ``german_reunification.csv``. RPCA-SC uses only
the outcome series (no auxiliary covariates), in contrast to the
five predictors Abadie et al. (2015) used.

.. code-block:: python

   from mlsynth import CLUSTERSC
   import pandas as pd

   file = (
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/german_reunification.csv"
   )
   df = pd.read_csv(file)

   res = CLUSTERSC({
       "df": df,
       "outcome": "gdp",
       "treat": "Reunification",
       "unitid": "country",
       "time": "year",
       "method": "rpca",
       "rpca_method": "PCP",     # convex RPCA, as in Bayani (2021)
       "compute_cft_pi": True,
       "cft_sims": 200,
       "cft_alpha": 0.05,
       "display_graphs": True,
   }).fit()

   cft = res.inference.cft
   gap = res.inputs.treated_outcome - res.rpca.counterfactual
   weights = {k: round(v, 2) for k, v in res.rpca.donor_weights.items()
              if abs(v) > 1e-3}
   print(f"donor pool (clustered) = {len(res.rpca.donor_weights)}")
   print(f"positive weights       = {weights}")
   print(f"pre-RMSE (1960-1989)   = {res.rpca.pre_rmse:.1f}")
   print(f"reunification ATT      = {res.rpca.att:+.0f}")
   print(f"gap in 2003            = {gap[-1]:+.0f}")
   print(f"95% CFT PI on ATT      = [{cft.att_pi[0]:+.0f}, {cft.att_pi[1]:+.0f}]")

The FPCA + :math:`k`-means step (Steps 1-2) selects an 11-economy
cluster for West Germany -- Australia, Austria, Belgium, Denmark,
France, Italy, Japan, the Netherlands, New Zealand, Norway and the
UK -- excluding the level outliers (the USA, Switzerland) and the
lower-income periphery (Greece, Portugal, Spain), exactly the donor
pool of Bayani's Table 2. PCP then reproduces that table's weights
to two digits: Norway :math:`\approx 0.48`, France
:math:`\approx 0.35`, New Zealand :math:`\approx 0.30` and Austria
:math:`\approx 0.02`, with the remaining donors at zero. The average
post-1990 gap is :math:`\approx -1500` per-capita units with a
1960-1989 fit RMSE near :math:`90`, and the per-period gap widens to
about :math:`-3730` by 2003 -- tracing the Robust PCA Synthetic
Control trajectory in Bayani's Figure 4 (a small positive blip in
1991-1992 followed by a steadily widening negative gap). The
Cattaneo-Feng-Titiunik (2021) 95% prediction interval on the ATT is
about :math:`[-1765, -1280]`, excluding zero (its width tracks
``cft_sims`` and ``random_state``); the HQF decomposition lands near
:math:`-1920`. The conclusion echoes Abadie et al. (2015):
reunification depressed West German per-capita GDP relative to its
synthetic counterpart.

Simulation studies
------------------

The three families documented here were each validated by their
authors on a purpose-built simulation. mlsynth ships these designs
verbatim where reproducible so the estimators can be regression-
tested against the originals.

PCR-SC -- missing-data robustness (Agarwal et al. 2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Agarwal et al. (2021, Section 4.5) do not use a synthetic DGP;
their validation is a *missing-data robustness study* on two
canonical panels -- Basque terrorism (Abadie-Gardeazabal 2003) and
California Proposition 99 (Abadie et al. 2010). For each panel they

* take the widely accepted published counterfactual as ground truth;
* confirm the donor matrix is near rank-one -- over :math:`99\%` of
  the spectral energy sits in the top singular value, exactly the
  low-rank regime in which the Theorem 3.1 / 3.2 error bounds bite;
* randomly obfuscate :math:`5\%`-:math:`20\%` of the donor entries
  and re-fit PCR (HSVT + OLS) on the **outcome series only**.

The finding: PCR tracks the published baseline across all missing-
data levels, whereas classical convex SC degrades sharply and plain
OLS overfits the pre-period noise (it even flips the sign of the
Basque effect). This is the empirical face of the HSVT denoising
step -- PCR recovers the same conclusion with neither auxiliary
covariates nor complete data. mlsynth reproduces the headline check
on Proposition 99: the singular spectrum is near rank-one and a
rank-1 HSVT PCR fit matches the canonical estimate.

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import CLUSTERSC

   file = (
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/smoking_data.csv"
   )
   df = pd.read_csv(file)

   wide = df.pivot(index="year", columns="state", values="cigsale")
   sv = np.linalg.svd(wide.values, compute_uv=False)
   print(f"top-SV energy share = {sv[0]**2 / (sv**2).sum():.3f}")  # ~0.99

   res = CLUSTERSC({
       "df": df, "outcome": "cigsale", "treat": "Proposition 99",
       "unitid": "state", "time": "year",
       "method": "pcr", "clustering": False, "rank": 1,
   }).fit()
   print(f"rank-1 PCR ATT = {res.att:+.2f}")

PCR-SC -- de-noising and the Bayesian posterior (Amjad-Shah-Shen 2018)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Amjad-Shah-Shen ([Amjad2018]_, Section 5.3) introduce the HSVT
de-noising step that PCR-SC is built on, and validate it on a
fully synthetic panel where the *true* mean is known. Each unit
:math:`i` draws a latent feature :math:`\theta_i \sim U[0, 1]`,
time is :math:`\rho_t = t`, and the mean is

.. math::

   m_{it} = \theta_i\bigl(1 + 0.3\tfrac{\rho_t}{T} e^{\rho_t/T}\bigr)
   + \cos\tfrac{f_1\pi}{180} + 0.5\sin\tfrac{f_2\pi}{180}
   + 1.5\cos\tfrac{f_3\pi}{180} - 0.5\sin\tfrac{f_4\pi}{180},

with periodicities :math:`f_1 = \rho_t \bmod 360`,
:math:`f_2 = \rho_t \bmod 180`, :math:`f_3 = 2\rho_t \bmod 360`,
:math:`f_4 = 2\rho_t \bmod 180`; the observed
:math:`X_{it} = m_{it} + \mathcal N(0, \sigma^2)`. They use
:math:`N = 100`, :math:`T = 2000`, intervention at
:math:`t = 1600`. Two findings drive the paper:

* **Training error tracks generalization error (Table 1).** The
  pre-intervention MSE of the estimated mean closely matches the
  post-intervention MSE at every noise level -- the de-noised fit
  generalizes.
* **De-noising beats no de-noising (Table 2).** The HSVT step lowers
  the generalization error consistently versus regressing on the raw
  donor matrix.

mlsynth reproduces both: the pre/post MSE ratio is :math:`\approx 1`
across :math:`\sigma^2`, and rank-4 HSVT de-noising cuts the
generalization error several-fold relative to the full-rank (no
de-noising) fit. The paper's Bayesian counterpart (Figure 12) trades
the inner OLS for a Gaussian posterior over the weights; mlsynth's
``estimator="bayesian"`` path now tracks the frequentist point
estimate to within Monte-Carlo error. Three implementation choices
make that hold (in
:py:func:`mlsynth.utils.clustersc_helpers.pcr.bayesian.solve_bayesian`):
the observation noise :math:`\sigma^2` is estimated from the *pre-period
fit residual* (not the total target variance, which conflates signal
with noise); the point counterfactual is the *posterior-mean
projection* (not a Monte-Carlo median of draws); and both the point
estimate and the credible band are projected through the *de-noised*
rank-:math:`r` donor matrix, so the band reflects the signal subspace
rather than raw-donor noise in the weight null space.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import CLUSTERSC

   N, T, T0 = 100, 2000, 1600
   rng = np.random.default_rng(0)
   theta = rng.uniform(0, 1, N)
   rho = np.arange(1, T + 1).astype(float)
   time_term = (np.cos((rho % 360) * np.pi / 180)
                + 0.5 * np.sin((rho % 180) * np.pi / 180)
                + 1.5 * np.cos(((2 * rho) % 360) * np.pi / 180)
                - 0.5 * np.sin(((2 * rho) % 180) * np.pi / 180))
   a = 1.0 + 0.3 * (rho / T) * np.exp(rho / T)
   M = np.outer(theta, a) + np.outer(np.ones(N), time_term)   # true means (N, T)
   m_true = M[0]                                              # treated unit's truth

   def fit(X, rank, estimator="frequentist"):
       df = pd.DataFrame(
           [(j, t, float(X[j, t]), int(j == 0 and t >= T0))
            for j in range(N) for t in range(T)],
           columns=["unit", "time", "y", "D"])
       res = CLUSTERSC({
           "df": df, "outcome": "y", "treat": "D", "unitid": "unit",
           "time": "time", "method": "pcr", "estimator": estimator,
           "clustering": False, "rank": rank, "rank_method": "fixed",
           "standardize_for_rank": False, "compute_shen_ci": False,
       }).fit()
       return np.asarray(res.pcr.counterfactual)

   X = M + rng.normal(0, np.sqrt(1.9), (N, T))                # sigma^2 = 1.9
   cf4 = fit(X, rank=4)                  # de-noised (4 singular values)
   cf_full = fit(X, rank=N - 1)          # no de-noising (full rank)
   cf_b = fit(X, rank=4, estimator="bayesian")

   mse = lambda u, v: float(np.mean((u - v) ** 2))
   print(f"Table 1  train(pre)={mse(cf4[:T0], m_true[:T0]):.4f}  "
         f"gen(post)={mse(cf4[T0:], m_true[T0:]):.4f}")          # ~equal
   print(f"Table 2  de-noised gen={mse(cf4[T0:], m_true[T0:]):.4f}  "
         f"no-de-noising gen={mse(cf_full[T0:], m_true[T0:]):.4f}")
   print(f"Bayesian gen={mse(cf_b[T0:], m_true[T0:]):.4f}  (tracks frequentist)")

A single draw at :math:`\sigma^2 = 1.9` gives training and
generalization MSE both :math:`\approx 0.02`, a no-de-noising
generalization error :math:`\approx 6\times` larger, and a Bayesian
generalization error indistinguishable from the frequentist -- the
de-noising and posterior machinery behaving as Amjad-Shah-Shen
describe.

PCR-SC -- donor-selection gains (Rho et al. 2025, ClusterSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rho et al. (2025, Section 6.1) build a synthetic panel with two
latent subgroups to show that clustering the donor pool *before*
fitting SC tightens the post-intervention error (their Theorems
5.11 / 5.13). The DGP, sized after disaggregated SC applications,
uses :math:`T = 10`, :math:`T_0 = 8`, :math:`n \in \{1000, 2000\}`
donors split evenly into groups :math:`A` and :math:`B`. Each group
is a sum of :math:`r_S` sinusoids

.. math::

   v_{i,t} = \alpha_i \sin(2\pi \omega_i t + \phi_i),
   \qquad
   S_{i,t} = \sum_{i=1}^{r_S} w_i\, v_{i,t},
   \quad w_i \sim \mathrm{Unif}[0, 1],

with group-specific frequency/magnitude laws --
:math:`\alpha_i \sim \mathrm{Beta}(2, 2)`,
:math:`\omega_i \sim \mathrm{Unif}(1, 3)` for :math:`A` versus
:math:`\alpha_i \sim \mathrm{Beta}(2, 5)`,
:math:`\omega_i \sim \mathrm{Unif}(3, 6)` for :math:`B`, and
:math:`\phi_i \sim \mathcal N(0, 1)` for both -- then additive noise
:math:`E_{i,t} \sim \mathcal N(0, s^2)` swept over
:math:`s \in \{0.10, 0.15, \dots, 0.40\}`. Over 500 datasets, a
leave-one-out placebo test on :math:`30\%` of group :math:`A`
compares ClusterSC against full-pool SC via the per-target
improvement :math:`I_i = \mathrm{MSE}_{\text{SC}} -
\mathrm{MSE}_{\text{ClusterSC}}`. The median improvement is positive
at every noise level and **grows with** :math:`s`: once noise blurs
the latent structure, restricting to the treated unit's cluster pays
off. mlsynth's ``clustering=True`` (the default) is exactly this
donor-selection step ahead of the PCR weight solve.

mlsynth's PCR-SC was validated head to head against the authors'
own reference implementation (the ``syclib`` library released with
the paper) on this exact DGP and on the paper's empirical
house-price-index panel. Running both pipelines -- SVD
:math:`k`-means clustering, HSVT denoising, then an OLS weight solve
-- on the same targets, the two produce **near-identical
counterfactuals**: the per-target counterfactual trajectories
correlate at :math:`\ge 0.99` on the synthetic sine panel and at
:math:`\ge 0.999` on the house-price-index panel, with matching
cluster assignment / donor-pool sizes and pre- and post-intervention
MSEs in the same range. The two residual differences are documented
design choices, not discrepancies: mlsynth denoises the *pre-period*
donor block for the weight fit (the Amjad-Shah-Shen [Amjad2018]_
convention, ``project_denoised=False`` by default) where the
reference code denoises the full :math:`Y_0` once, and mlsynth's
clustering features come from the pre-period SVD rather than the full
panel. Set ``project_denoised=True`` to match the reference code's
projection convention.

The synthetic DGP is self-contained -- one draw, one treated
subgroup-:math:`A` unit, donors clustered before the PCR fit:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import CLUSTERSC


   def _sine_panel(n, T, noise, n_signals, alpha_beta, omega_lo, omega_hi, rng):
       """One subgroup: n units, each a random mix of n_signals sinusoids + noise."""
       basis = np.zeros((n_signals, T))
       grid = np.arange(int(T * 1.2)) * 10 * np.pi
       for i in range(n_signals):
           alpha = rng.beta(*alpha_beta)
           omega = rng.uniform(omega_lo, omega_hi)
           phi = rng.normal(0, 1)
           wave = alpha * np.sin(2 * np.pi * omega * grid / 360 + phi)
           basis[i] = wave[int(0.2 * T):]                  # drop first 20%
       W = rng.uniform(0, 1, (n, n_signals))
       return W @ basis + rng.normal(0, noise, (n, T))      # (n, T)


   rng = np.random.default_rng(0)
   T, T0, rS, nA, nB, noise = 10, 8, 3, 60, 60, 0.30
   A = _sine_panel(nA, T, noise, rS, (2, 2), 1, 3, rng)     # subgroup A
   B = _sine_panel(nB, T, noise, rS, (2, 5), 3, 6, rng)     # subgroup B
   panel = np.vstack([A, B])                                # (nA + nB, T)

   # Treat one subgroup-A unit; everyone else is a donor. Add a known effect.
   tau_true, treated = 5.0, 0
   panel[treated, T0:] += tau_true
   df = pd.DataFrame([
       {"unit": j, "time": t, "y": float(panel[j, t]),
        "D": int(j == treated and t >= T0)}
       for j in range(nA + nB) for t in range(T)
   ])

   res = CLUSTERSC({
       "df": df, "outcome": "y", "treat": "D", "unitid": "unit", "time": "time",
       "method": "pcr", "pcr_objective": "OLS", "clustering": True,
       "k_clusters": 2, "rank": rS, "rank_method": "fixed",
   }).fit()

   n_sel = len(res.pcr.donor_weights)
   print(f"donors selected by clustering = {n_sel} of {nA + nB - 1}")  # ~ subgroup A
   print(f"selected (mostly) subgroup A  = {n_sel <= nA}")             # True
   print(f"pre-RMSE                      = {res.pcr.pre_rmse:.3f}")
   print(f"ATT (true {tau_true:+.1f})             = {res.pcr.att:+.3f}")  # ~ +5

The clustering step isolates the treated unit's subgroup
(:math:`\approx n_A - 1` donors), the pre-period fit is tight, and
the OLS weight solve recovers the planted effect -- matching the
reference ClusterSC+OLS output to the differences noted above.

RPCA-SC -- two-process recovery under noise and missingness (Bayani 2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayani (2021, Section 4) generates :math:`N_1 = N_2 = 100` units
from two deterministic processes plus i.i.d. Gaussian noise
:math:`\epsilon_t \sim \mathcal N(0, \sigma^2)` over
:math:`t \in [0, T]` with :math:`T = 250` and intervention at
:math:`T_0 = 150`:

.. math::

   f_1(t) &= 0.3\,(t \bmod (T{+}1))
             - (t \bmod 10)\sin(t/\pi)
             + (t \bmod 10)\cos(t/\pi) + \epsilon_t, \\
   f_2(t) &= \log(t) + 4\sin(t/\pi) + 4\cos(t/\pi) + \epsilon_t,

with the noise variance swept over
:math:`\sigma^2 \in \{1, 4, 9, 16, 25\}`. The study has two halves:

* **Clustering recovery (Steps 1-2).** FPCA over the pre-period plus
  :math:`k`-means recovers the two latent processes with :math:`100\%`
  accuracy at every noise level; the first FPC score alone explains
  over :math:`95\%` of the variation and the silhouette statistic
  selects :math:`k = 2`.
* **Counterfactual accuracy (Steps 3-5).** Treating the noiseless
  mean of :math:`f_1` as the target and its 100 noisy realisations as
  the donor pool, RPCA extracts the low-rank component and projects
  the counterfactual. The estimation RMSPE stays low and pre-/post-
  intervention errors stay close even at high noise:

  .. list-table:: Bayani (2021) Table 3 -- RMSPE, fully observed
     :header-rows: 1
     :widths: 20 40 40

     * - :math:`\sigma^2`
       - Pre-intervention
       - Post-intervention
     * - 1
       - 0.09
       - 0.13
     * - 4
       - 0.19
       - 0.25
     * - 9
       - 0.29
       - 0.38
     * - 16
       - 0.39
       - 0.51
     * - 25
       - 0.49
       - 0.64

  Repeating with :math:`30\%` of entries removed at random (Table 4)
  inflates the errors -- the robust PCA step also imputes -- but the
  counterfactual still tracks the true mean (e.g. post-intervention
  RMSPE :math:`0.65` at :math:`\sigma^2 = 1`, rising to :math:`2.59`
  at :math:`\sigma^2 = 25`). Bayani's design uses the convex (PCP)
  decomposition; the HQF solver is a later mlsynth addition.

A compact one-draw reproduction of the clustering half:

.. code-block:: python

   import numpy as np, pandas as pd
   from mlsynth import CLUSTERSC

   rng = np.random.default_rng(0)
   T, T0, sigma2 = 250, 150, 9.0
   t = np.arange(1, T + 1)

   def f1(): return (0.3 * (t % (T + 1)) - (t % 10) * np.sin(t / np.pi)
                     + (t % 10) * np.cos(t / np.pi))
   def f2(): return np.log(t) + 4 * np.sin(t / np.pi) + 4 * np.cos(t / np.pi)

   rows = []
   for j in range(200):                       # 100 of each process
       base = f1() if j < 100 else f2()
       y = base + rng.normal(0, np.sqrt(sigma2), size=T)
       treated = (j == 0)                      # one f1 unit is treated
       if treated:
           y[T0:] += 5.0
       rows += [{"unit": j, "time": int(tt), "y": float(yi),
                 "D": int(treated and tt > T0)}
                for tt, yi in zip(t, y)]
   df = pd.DataFrame(rows)

   res = CLUSTERSC({
       "df": df, "outcome": "y", "treat": "D",
       "unitid": "unit", "time": "time",
       "method": "rpca", "rpca_method": "PCP",
   }).fit()
   print(f"donor pool size = {len(res.rpca.donor_weights)}")  # treated unit's cluster (f1 subgroup)
   print(f"pre-RMSE        = {res.rpca.pre_rmse:.2f}")
   print(f"ATT (true 5.0)  = {res.rpca.att:+.2f}")

References
----------

Abadie, A., & Gardeazabal, J. (2003). "The Economic Costs of
Conflict: A Case Study of the Basque Country." *American Economic
Review* 93(1):113-132.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect
of California's Tobacco Control Program." *Journal of the American
Statistical Association* 105(490):493-505.

Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative
Politics and the Synthetic Control Method." *American Journal of
Political Science* 59(2):495-510.

Agarwal, A., Shah, D., Shen, D., & Song, D. (2021). "On Robustness
of Principal Component Regression." *Journal of the American
Statistical Association* 116(536):1731-1745.

Amjad, M., Shah, D., & Shen, D. (2018). "Robust Synthetic Control."
*Journal of Machine Learning Research* 19(22):1-51.

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K.
(2021). "Matrix Completion Methods for Causal Panel Data Models."
*Journal of the American Statistical Association* 116(536):
1716-1730.

Bayani, M. (2021). "Robust PCA Synthetic Control."
arXiv:2108.12542.

Bayani, M. (2022). "Essays on Machine Learning Methods in
Economics." Chapter 1, CUNY Academic Works.

Candes, E. J., Li, X., Ma, Y., & Wright, J. (2011). "Robust
Principal Component Analysis?" *Journal of the ACM* 58(3):11.

Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction
Intervals for Synthetic Control Methods." *Journal of the American
Statistical Association* 116(536):1865-1880. Reference
implementation: https://github.com/nppackages/scpi.

Chatterjee, S. (2015). "Matrix Estimation by Universal Singular
Value Thresholding." *Annals of Statistics* 43(1):177-214.

Hartigan, J. A., & Wong, M. A. (1979). "Algorithm AS 136: A K-means
Clustering Algorithm." *Applied Statistics* 28(1):100-108.

Li, Y., Wang, N., & Carroll, R. J. (2010). "Generalized Functional
Linear Models with Semiparametric Single-index Interactions."
*Journal of the American Statistical Association* 105(490):621-633.

Rho, S., Tang, A., Bergam, N., Cummings, R., & Misra, V. (2025).
"ClusterSC: Advancing Synthetic Control with Donor Selection."
arXiv:2503.21629.

Rousseeuw, P. J. (1987). "Silhouettes: A Graphical Aid to the
Interpretation and Validation of Cluster Analysis." *Journal of
Computational and Applied Mathematics* 20:53-65.

Shen, D., Ding, P., Sekhon, J., & Yu, B. (2023). "Same Root
Different Leaves: Time Series and Cross-Sectional Methods in Panel
Data." *Econometrica* 91(6):2125-2154. Reference implementation:
https://github.com/deshen24/panel-data-regressions.

Wang, Z., Li, X. P., So, H. C., & Liu, Z. (2023). "Robust PCA via
Non-convex Half-quadratic Regularization." *Signal Processing*
204:108816.

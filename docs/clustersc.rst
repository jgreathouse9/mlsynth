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
  exposes two extensions on top of Algorithm 2: a Bayesian posterior
  over the weights (Bayani 2022) and a simplex-constrained variant
  that retains Abadie-Diamond-Hainmueller weights.

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

**HSVT denoising.** Let :math:`Y_0^- = U \Sigma V^\top` be the SVD
of the pre-period donor matrix. The rank-:math:`r` hard truncation

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
(``lambda_penalty``, ``p``, ``q``) for the same purpose. The
counterfactual and ATT come from projecting through the *denoised*
donor matrix in both periods,

.. math::

   \widehat y^0_1 = \widetilde M\, \widehat f,
   \qquad
   \widehat{ATT} = \frac{1}{T - T_0} \sum_{t > T_0}
                   \bigl(y_{1, t} - (\widetilde M\, \widehat f)_t\bigr).

mlsynth extensions
^^^^^^^^^^^^^^^^^^

Two paper-extensible weight solvers live alongside the OLS default:

* **Bayesian PCR** (``estimator="bayesian"``, Bayani 2022 Ch. 1).
  Replace OLS with the Gaussian posterior

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

Inference
^^^^^^^^^

Frequentist PCR (the paper's Algorithm 2) ships no built-in
inference -- the per-period prediction error decomposition in
Lemma B.10 of Rho et al. (2025) makes shrinkage to zero impossible
without additional assumptions, so mlsynth refrains from quoting a
spurious CI. :py:attr:`CLUSTERSCInference.method` records
``"none"`` in that case.

For Bayesian PCR (``estimator="bayesian"``), mlsynth returns a
:math:`(1 - \alpha)` posterior credible interval for the ATT,
derived from the per-period draws of the counterfactual against
which the treated post-period mean is differenced.

For RPCA-SC, only the point estimate is returned. The original
paper relies on permutation-based placebo tests for inference
(Bayani 2021, Section 3); these can be assembled outside the
estimator from the returned counterfactual.

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
full donor pool. The example below fits California Proposition 99
cigarette-sales data with HSVT truncated to rank :math:`r = 4`.

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

References
----------

Abadie, A., & Gardeazabal, J. (2003). "The Economic Costs of
Conflict: A Case Study of the Basque Country." *American Economic
Review* 93(1):113-132.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect
of California's Tobacco Control Program." *Journal of the American
Statistical Association* 105(490):493-505.

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

Wang, Z., Li, X. P., So, H. C., & Liu, Z. (2023). "Robust PCA via
Non-convex Half-quadratic Regularization." *Signal Processing*
204:108816.

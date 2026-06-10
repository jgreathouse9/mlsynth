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

R-faithful (NSC.R, ``fn_cv``): for each donor :math:`j`, fit NSC
weights on its **pre-period** matching vector using the other
:math:`J - 1` donors plus one randomly drawn extra donor (the pool
size stays at :math:`J` so the eigenvalue scaling of
:math:`(a^*, b^*)` is comparable to the final treated-unit fit).
The score is the **post-period** MSPE of donor :math:`j`'s held-out
outcomes against that weighted combination. The (a, b) selected
minimises the average held-out MSPE across donors. Coordinate
descent:

1. Initialise :math:`b^* = 0`.
2. Sweep :math:`a^*` over the grid; pick the minimiser.
3. Hold :math:`a^*`; sweep :math:`b^*`; pick the minimiser.
4. Iterate until :math:`(a^*, b^*)` stops moving.

The extra-donor draw uses ``seed`` (default ``123`` to match the
reference R script's ``set.seed(123)``). A previous in-sample
"controls" target (predicting the donor's *pre*-period from the
others, i.e. evaluating on the same data the fit used) was removed
in favour of this proper held-out score; the legacy
``cv_target = "treated"`` option no longer validates.

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

Assumptions (Tian 2023)
-----------------------

NSC inherits the interactive-fixed-effects (IFE) model of
Abadie-Diamond-Hainmueller (2010) and adds the structural
conditions needed for the synthetic-control bias to vanish under a
**nonlinear** outcome. The paper's formal assumptions:

**A1 (IFE for the untreated latent).** Each unit's untreated latent
outcome is :math:`Y_{it}^{0*} = X_i' \beta_t + \mu_i' \lambda_t +
\varepsilon_{it}`, where :math:`X_i` are observed predictors,
:math:`\mu_i` unobserved loadings, and :math:`\varepsilon_{it}` an
idiosyncratic shock.

**A2 (transitory shocks).** Shocks :math:`\varepsilon_{it}` are
independent across :math:`i, t`, have zero conditional mean given
:math:`(X_j, \mu_j, D_{js})` for all :math:`j, s`, and bounded
:math:`p`-th moments for some even :math:`p \ge 2`.

**A4 (factor non-degeneracy).** The smallest eigenvalue of
:math:`T_0^{-1} \sum_t \lambda_t \lambda_t'` is bounded away from
zero. Translation: the unobserved factors carry persistent
identifying variation across the pre-period.

**A5 (overlap / continuous support).** The composite predictor
:math:`H = [X', \mu']` has a density bounded away from zero on a
compact convex support, units are iid draws from that
distribution, and the treatment-assignment probability is
non-degenerate. Practical reading: for almost every value of the
treated unit's predictors there is a control with near-identical
predictors *in the population* -- with enough donors, some of
those near-twins land in the sample.

**A6 (smooth nonlinear link).** The observed outcome is
:math:`Y_{it}^0 = F(X_i' \beta_t + \mu_i' \lambda_t +
\varepsilon_{it})` with :math:`F` strictly monotonic and
:math:`G(\cdot) = \mathbb{E}_\varepsilon[F(\cdot)]` smooth. The
paper does **not** require the analyst to know :math:`F` or
:math:`G` -- only that they exist with these regularity
properties.

**A7 (nearest-:math:`M` weighting).** The synthetic control uses
only the :math:`M > k + T_0` nearest neighbours (in observed
predictors and pre-period outcomes) of the treated unit; donors
outside that neighbourhood get weight zero.

Under A2, A4, A5, A6, A7 the NSC estimator is asymptotically
unbiased as :math:`T_0 \to \infty` provided the donor pool grows
**super-polynomially** in :math:`T_0` (Theorem 2:
:math:`J = O(T_0^{b(T_0)})` with :math:`b'(\cdot) > 0`). The fast
growth is what makes Assumption 5's near-twins *available in
finite samples*, so the local-neighbourhood matching in A7 is
non-empty.

When the assumptions bind: practical diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(a) **IFE / smooth-link DGP for the untreated outcome (A1, A6).**
    The untreated potential outcome is generated by a smooth
    monotone function of a low-rank factor model.

    *Plausibly violated when* the outcome is genuinely discrete
    (binary, ordinal, low-count) -- ``F`` is then non-monotonic
    *between intervals*, A6 is violated, and matching on observed
    pre-period outcomes no longer implies matching on the latent
    predictors (paper's Remark 8). *Diagnostic*: histogram the
    outcome; if it heaps on a small number of values, switch to a
    discrete-choice or DSC-style framework
    (:doc:`dsc` for the distributional question).

(b) **Convex-hull / near-twin availability in the sample (A5).**
    NSC dropped the non-negativity restriction *precisely* because
    affine weights tolerate the treated unit lying *outside* the
    donor convex hull -- nearest neighbours can be reached with
    negative weights on far donors. But identification still
    requires the **population** density to put mass near the
    treated unit's predictors.

    *Plausibly violated when* the treated unit is qualitatively
    different from every potential donor -- a coastal mega-state
    against only interior small states, a tech-led economy with
    only commodity-led donors. *Diagnostic*: read
    ``res.pre_rmse``; a large pre-period RMSE despite the
    eigenvalue-scaled L1/L2 freedom signals no near-twin in the
    sample. Increasing the donor pool is the only fix -- NSC does
    not invent population mass that is not there.

(c) **Donor pool size grows with** :math:`T_0` **(Theorem 2).**
    Bias vanishes asymptotically only if :math:`J` grows
    super-polynomially in :math:`T_0`. In finite samples, the
    actionable reading is: **more donors are unambiguously better
    in the nonlinear case** because they make the nearest-:math:`M`
    matching tighter (eq. 5).

    *Plausibly violated when* :math:`J` is small relative to
    :math:`T_0` -- a country-level study with a dozen donors and
    decades of pre-period. *Diagnostic*: re-run NSC after dropping
    the bottom-half of donors by closeness; if ATT shifts
    substantially, the small donor pool was binding. Cross-check
    against the Monte Carlo table below, which shows bias falling
    almost in half when :math:`J` doubles from 25 to 50.

(d) **Pre-period factor non-degeneracy (A4).**
    The time factors :math:`\lambda_t` must move enough across the
    pre-period for the unobserved loadings to be identified.

    *Plausibly violated when* the pre-period is flat or
    co-trending across all units (e.g. all donors follow the same
    secular trend with no idiosyncratic shocks). *Diagnostic*: an
    SVD on the donor pre-matrix should show a clear spectrum, not
    a single dominant component drowning out the rest.

(e) **Outcome is monotone in the latent index (A6).**
    The link function :math:`F` is strictly increasing in its
    argument. Many transformations satisfy this (logistic, square,
    log) but capped or saturating outcomes do not (e.g. an outcome
    with a true ceiling that some donors but not the treated unit
    hit, making the *same* latent value produce different observed
    outcomes).

    *Plausibly violated when* the outcome has a hard floor or
    ceiling that some donors are pressed against. *Diagnostic*:
    inspect the empirical distribution of donor outcomes against
    its theoretical bounds; if a non-trivial fraction sit at the
    boundary, A6 fails for those donors and they should be dropped
    or re-coded.

(f) **Doudchenko-Imbens variance reflects a normal limit.**
    The closed-form CI assumes the gap is approximately normal
    with the leave-one-control variance estimate. With very small
    :math:`J` or heavy-tailed outcomes the normal approximation
    breaks.

    *Plausibly violated when* :math:`J < 10` or the donor outcomes
    are heavy-tailed. *Diagnostic*: bootstrap the post-period gap
    by resampling donors; if the bootstrap distribution is visibly
    skewed or shows heavy tails, treat the closed-form CI as a
    guide and report the bootstrap CI alongside.

When to use NSC -- and when not to
----------------------------------

**Reach for NSC when:**

* The outcome is a **smooth nonlinear** function of latent
  predictors -- bounded growth metrics, log-GDP, count outcomes
  modelled continuously, share variables on :math:`(0, 1)`.
* The treated unit sits at or near the **boundary** of the donor
  predictor distribution -- canonical SC's non-negativity
  restriction would force interpolation bias from far donors, and
  affine weights with the L1 closeness penalty can reach the
  treated unit by **extrapolating** through the closest neighbours.
* You have a **large donor pool** relative to the pre-period
  (:math:`J \gg T_0`) so that nearest-:math:`M` matching is sharp.
* You're willing to **trade interpretability** for bias: NSC
  returns weights that can be negative, which means "subtract some
  donors", a step beyond the canonical SC story of "a convex
  combination of similar units".

**Do not use NSC when:**

* **Outcome is binary, ordinal, or low-count.** A6 requires a
  strictly monotonic link :math:`F`; discrete outcomes need
  discrete-choice or distributional methods (:doc:`dsc`) instead.
* **You need genuinely sparse, non-negative weights.** NSC's whole
  premise is dropping non-negativity. If the policy story is
  "California is a convex combination of these four states", use
  *canonical SCM* or :doc:`tssc` -- NSC's negative weights are an
  identification gain but a rhetorical loss.
* **Very small donor pool** (:math:`J \le 10` or so). Theorem 2
  fails, the leave-one-control inference becomes unstable, and the
  L1-anchored nearest-neighbour story degenerates.
* **Outcome with hard floors or ceilings binding for some donors.**
  Bounded growth where multiple donors are pinned at the boundary
  violates A6 for those donors. Drop or re-code them, or move to
  a censored-regression-aware estimator.
* **Pre-period too short to identify the factor structure.** With
  :math:`T_0 \lesssim r` (factor count), A4 is at the edge and the
  cross-validation grid is choosing essentially noise. Use
  :doc:`fdid` (which is designed for short panels) or :doc:`fma`
  (which jointly estimates a low-rank factor model).
* **You suspect interference / spillovers across units.** NSC
  inherits SUTVA from canonical SC; a treated unit influencing
  donors breaks A2's mean-independence. Switch to
  :doc:`spillsynth` or :doc:`spsydid`.
* **Continuous treatment.** NSC encodes a single binary
  intervention. Continuous dose belongs in :doc:`ctsc`.

Monte Carlo: NSC vs OSC bias under nonlinearity
-----------------------------------------------

The paper's Section 4 simulation: a rank-2 IFE latent process
rescaled to :math:`[0, 1]` and raised to the power :math:`r`
(:math:`r = 1` linear; :math:`r = 2` nonlinear), with a small
post-period treatment effect grid. The table below reproduces the
qualitative finding -- NSC beats OSC (the non-negativity-restricted
original) on bias in the nonlinear regime, and matches it in the
linear one.

.. note::

   **Verification.** NSC is validated two ways: a **cross-validation**
   against Tian's own R implementation on Proposition 99 (weights match
   to correlation 0.989; effect path :math:`-9.1/-22.6/-27.0` vs the
   paper's :math:`-9.5/-24.5/-28.7`) and the **Path-B** nonlinear Monte
   Carlo (near-nominal coverage; error shrinking with the donor pool).
   See the dedicated page :doc:`replications/nsc`; durable checks
   ``nsc_prop99`` and ``nsc_mc``.

.. code-block:: python

   """Tian (2023) Section 4 DGP: NSC vs OSC bias as nonlinearity rises.

   This runs a handful of replications (per (J, T0, r) cell). The full
   paper uses 5000 reps; this snippet keeps it tractable while still
   reproducing the qualitative ordering."""

   import numpy as np
   import pandas as pd
   from mlsynth import NSC

   def gen_panel(J, T0, T_post, r, seed):
       rng = np.random.default_rng(seed)
       T = T0 + T_post
       X = rng.uniform(0.0, 2 * np.sqrt(3.0), size=(J + 1, 2))
       mu = rng.uniform(0.0, 2 * np.sqrt(3.0), size=(J + 1, 4))
       beta_t = rng.normal(10.0, 1.0, size=(T, 2))
       lam_t = rng.normal(10.0, 1.0, size=(T, 4))
       eps = rng.normal(0.0, 1.0, size=(T, J + 1))
       Y_star = (X @ beta_t.T).T + (mu @ lam_t.T).T + eps
       Yn = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
       return Yn ** r           # (T, J+1) untreated outcomes

   def att_bias(Y0, T0, J, seed):
       T_post = Y0.shape[0] - T0
       tau = np.linspace(0.02, 0.2, T_post)
       Y = Y0.copy(); Y[T0:, 0] += tau
       rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
                "D": int(j == 0 and t >= T0)}
               for j in range(J + 1) for t in range(Y.shape[0])]
       df = pd.DataFrame(rows)
       res = NSC({"df": df, "outcome": "y", "treat": "D",
                  "unitid": "unit", "time": "time",
                  "cv_grid_size": 0.1, "run_inference": False,
                  "display_graphs": False, "seed": seed}).fit()
       # Per-period |bias| averaged across post-period, ×100 (paper scale).
       return float(np.mean(np.abs(res.gap[T0:] - tau)) * 100.0)

   for (J, T0) in [(25, 15), (50, 15), (50, 30)]:
       for r in (1, 2):
           biases = []
           for rep in range(20):
               Y0 = gen_panel(J=J, T0=T0, T_post=10, r=r, seed=10 * rep + 1)
               biases.append(att_bias(Y0, T0=T0, J=J, seed=10 * rep + 2))
           print(f"  J={J:>3d}, T0={T0:>2d}, r={r}: "
                 f"|bias|x100 mean = {np.mean(biases):.2f}, "
                 f"SD = {np.std(biases, ddof=1):.2f}")

Representative output at one cell (mlsynth, 20 reps; the paper
uses 5000):

.. code-block:: text

   J=50, T0=15, r=1: |bias|x100 mean = 0.82, SD = 0.28   (paper NSC: 0.77)
   J=50, T0=15, r=2: |bias|x100 mean = 0.87, SD = 0.51   (paper NSC: 0.74)

The qualitative paper findings to look for as you fill in the
remaining cells:

* **Bias falls as** :math:`J` **grows** -- direct evidence of the
  Theorem 2 mechanism: more donors makes the near-twin pool denser
  and the nearest-:math:`M` matching tighter. Paper Table 1: NSC
  bias drops from 0.99 (J=25) to 0.77 (J=50) in the linear case
  and from 0.92 to 0.74 in the nonlinear case at T0=15.
* **NSC matches OSC in the linear case** and **wins in the
  nonlinear case**. Paper Table 1: at (J=25, T0=30), OSC bias is
  1.79 (r=1) and 1.40 (r=2); NSC stays at 0.91 and 0.87. The
  non-negativity restriction in OSC pays an extrapolation penalty
  in the nonlinear regime that NSC's affine + L1-anchored
  weighting avoids.
* **Coverage of the closed-form CI sits at 0.93-0.95** across the
  paper's :math:`(J, T_0, r)` grid -- the Doudchenko-Imbens
  variance estimator is well-calibrated for this DGP. mlsynth's
  CI reproduces that coverage band.

Empirical: Proposition 99 (California, 1989-2000)
-------------------------------------------------

Revisits the Abadie-Diamond-Hainmueller (2010) tobacco-control
case study. mlsynth's NSC, R-faithful to the reference
implementation, recovers the paper's published
:math:`(a^*, b^*) = (0.3, 0.7)` and produces per-year gaps within
a pack or two of the paper's reported numbers.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import NSC

   df = pd.read_stata("synth_smoking.dta")            # Abadie-Diamond-Hainmueller (2010)
   df = df.sort_values(["state", "year"]).reset_index(drop=True)
   df["treatment"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)

   res = NSC({
       "df": df, "outcome": "cigsale", "treat": "treatment",
       "unitid": "state", "time": "year",
       "cv_grid_size": 0.1, "run_inference": True,
       "display_graphs": False, "seed": 42,
   }).fit()

   print(f"selected (a*, b*) = ({res.design.a_star}, {res.design.b_star})")
   print(f"pre-RMSE          = {res.pre_rmse:.4f}")
   print(f"ATT (1989-2000)   = {res.att:+.4f}  packs/capita")
   print(f"ATT 95% CI        = ({res.inference_detail.att_lower:+.2f}, "
         f"{res.inference_detail.att_upper:+.2f})")

   gaps = pd.DataFrame({
       "year": res.inputs.time_labels.astype(int),
       "gap": res.gap,
       "ci_low": res.inference_detail.gap_lower,
       "ci_high": res.inference_detail.gap_upper,
   })
   for y in (1990, 1995, 2000):
       row = gaps.loc[gaps.year == y].iloc[0]
       print(f"  {y}: gap = {row.gap:+6.2f}  CI=({row.ci_low:+.2f}, {row.ci_high:+.2f})")

Output::

   selected (a*, b*) = (0.3, 0.7)        # matches the paper exactly
   pre-RMSE          = 1.2450
   ATT (1989-2000)   = -19.1313  packs/capita
   ATT 95% CI        = (-25.51, -12.75)
     1990: gap =  -9.05  CI=(-26.38, +8.27)        # paper: -9.5
     1995: gap = -22.62  CI=(-46.03, +0.78)        # paper: -24.5
     2000: gap = -27.01  CI=(-54.31, +0.29)        # paper: -28.7

Two things are worth flagging about this replication.

* **The selected** :math:`(a^*, b^*) = (0.3, 0.7)` **is grid-stable
  across seeds 42, 789, 1000, 2024** and one grid step away from
  the paper for other seeds (e.g. seed 123 lands on
  :math:`(0.2, 0.8)`). This jitter is the same RNG noise the
  reference R script exhibits across ``set.seed()`` values --
  unavoidable when the CV burns ``J`` random draws per grid point
  and Python's RNG differs from R's.
* **Sparsity comparison vs OSC**: the original SC concentrates
  weight on Utah (0.385), Montana (0.271), Nevada (0.186). NSC
  spreads weight across roughly 20 states -- the L1-anchored
  affine fit is recruiting more near-neighbour Western states
  with smaller weights (some negative -- Alabama -0.015,
  Tennessee -0.071 in the paper's Table 2). The pre-period fit is
  similar (NSC pre-RMSE 1.25 vs OSC ~2.0) but the spread weights
  reduce the variance of the post-period counterfactual.

The take-away mirrors the paper's: NSC's affine + L1-anchored
formulation finds a tighter pre-period fit by pulling in
close-neighbour Western states (with some negative weights to
correct for the few far donors it admits), which then translates
into a smoother and slightly larger counterfactual decline. The
ATT of ~-19 packs/capita over 1989-2000 is in the same ballpark
as the canonical Abadie et al. result while using a strictly
larger weight space.

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

.. note::

   ``NSC.fit()`` returns an :class:`~mlsynth.config_models.EffectResult` on the
   standardized two-family contract: ``res.att`` / ``res.att_ci`` /
   ``res.counterfactual`` / ``res.gap`` / ``res.donor_weights`` /
   ``res.pre_rmse`` resolve through the standardized sub-models. The rich
   Doudchenko-Imbens per-period bands are on ``res.inference_detail`` (the bare
   ``res.inference`` slot is reserved for the standardized ATT-level
   :class:`~mlsynth.config_models.InferenceResults`).

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
   print(f"95% CI ATT  = [{results.inference_detail.att_lower:+.3f}, "
         f"{results.inference_detail.att_upper:+.3f}]")
   print(f"p-value     = {results.inference_detail.p_value:.3f}")

   # Per-period diagnostics:
   import pandas as pd
   print(pd.DataFrame({
       "t": np.arange(T),
       "gap": results.gap,
       "ci_low": results.inference_detail.gap_lower,
       "ci_high": results.inference_detail.gap_upper,
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

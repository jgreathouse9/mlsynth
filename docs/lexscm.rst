Lexicographic Synthetic Control (LEXSCM)
========================================

.. currentmodule:: mlsynth

Overview
--------

LEXSCM is a tool for **synthetic experimental design**: given a panel of
units, a subset of which are *eligible* for treatment, it chooses which
:math:`m` of them to actually treat so that the rest of the panel forms a
credible synthetic control for the treated group, *and* so that the
resulting experiment has enough statistical power to detect the effects
the analyst cares about. It is the design-stage analogue of the
estimation-stage synthetic control methods elsewhere in mlsynth: instead
of taking the treated unit as given and building a donor combination, it
takes the donor pool as given and chooses the treated combination.

The method follows Abadie & Zhao, *Synthetic Controls for Experimental
Design* [ABADIE2024]_, with the power / minimum-detectable-effect (MDE)
machinery drawn from the synthetic-experimental-design work of
Vives-i-Bastida (2022). LEXSCM optimizes two objectives in a strict
priority order ("lexicographically"):

#. **Validity** -- the treated combination reproduces the population
   trajectory on the pre-treatment predictors (small *imbalance*);
#. **Power** -- subject to validity, the design detects the smallest
   possible effect (small *minimum detectable effect*).

The pipeline has four stages, each in its own helper module:

#. **Treated-tuple selection** (:mod:`~mlsynth.utils.fast_scm_helpers.lexsearch`)
   -- search the :math:`\binom{M}{m}` candidate treated combinations for
   the most balanced ones, under an optional budget.
#. **Control fit** (:mod:`~mlsynth.utils.fast_scm_helpers.fast_scm_control`)
   -- build a synthetic control for each candidate treated group and
   score its pre-treatment fit.
#. **Power analysis** (:mod:`~mlsynth.utils.fast_scm_helpers.lexpower`)
   -- a moving-block placebo MDE curve over a grid of post-treatment
   horizons.
#. **Recommendation** (:mod:`~mlsynth.utils.fast_scm_helpers.lexselect`)
   -- a lexicographic rule (validity gate :math:`\to` power
   :math:`\to` stability :math:`\to` cost) returning a single recommended
   design plus a Pareto frontier.

Mathematical Formulation
------------------------

Setup and notation
^^^^^^^^^^^^^^^^^^^

We observe :math:`J` units over a pre-treatment period of length
:math:`T_0`. Stack the pre-period outcomes as
:math:`Y \in \mathbb{R}^{T_0 \times J}` (rows = time, columns = units) and,
optionally, :math:`K` time-invariant or pre-period covariates as
:math:`Z \in \mathbb{R}^{K \times J}`. The two are stacked vertically into
the **predictor matrix**

.. math::

   X = \begin{bmatrix} Y \\ Z \end{bmatrix}
       \in \mathbb{R}^{(T_0 + K) \times J},

so each column :math:`X_{\cdot, j}` is unit :math:`j`'s predictor profile.
A subset :math:`\mathcal{C} \subseteq \{1, \dots, J\}` of size
:math:`M = |\mathcal{C}|` is flagged as **treatment-eligible** (the
``candidate_col``); the design must pick :math:`m` of these.

A **population weighting vector** :math:`f \in \mathbb{R}^J`,
:math:`f \ge 0`, :math:`\mathbf{1}^\top f = 1` (uniform :math:`1/J` by
default, or a ``weight_col`` such as population or revenue) defines the
estimand's target trajectory -- the :math:`f`-weighted average unit

.. math::

   \bar{X} = X f, \qquad \bar{X}_t = X_{t, \cdot}\, f .

The pre-period predictor rows are split into an **estimation window**
:math:`E` (the first :math:`\lfloor \texttt{frac\_E} \cdot T_0 \rfloor`
time rows, *plus* all covariate rows) and a held-out **blank window**
:math:`B` (the remaining pre-period time rows). :math:`E` is where the
design is fit; :math:`B` is a pre-treatment "dress rehearsal" with no
treatment, used to validate fit and to calibrate power.

Over :math:`E` the predictors are **row-standardized against the
population target**:

.. math::

   \widetilde{X}_{t, j} = \frac{X_{t, j} - \bar{X}_t}{\sigma_t},
   \qquad
   \sigma_t = \operatorname{sd}_j\!\bigl(X_{t, \cdot}\bigr),

with :math:`\sigma_t` floored away from zero. Centring on
:math:`\bar{X}_t` puts the population target at the origin; scaling by the
cross-sectional spread :math:`\sigma_t` makes mixed-scale predictors
(outcomes in dollars, covariates in percent) commensurable so no single
predictor dominates the fit. The :math:`J \times J` **Gram matrix**

.. math::

   G = \widetilde{X}_E^\top \widetilde{X}_E \succeq 0

summarizes all pairwise predictor inner products over :math:`E`.

Identifying assumptions
^^^^^^^^^^^^^^^^^^^^^^^^

LEXSCM inherits the design-based identification of Abadie & Zhao
[ABADIE2024]_. The untreated potential outcomes are assumed to follow an
(approximate) linear factor model

.. math::

   Y_{j, t}(0) = \delta_t + \theta_t^\top \mu_j + \varepsilon_{j, t},

with common time effects :math:`\delta_t`, latent time factors
:math:`\theta_t`, unit loadings :math:`\mu_j`, and mean-zero transitory
shocks :math:`\varepsilon_{j, t}`.

*Assumption 1 (approximate balance).* There exist treatment weights
:math:`w` on the chosen treated set :math:`S` and control weights
:math:`v` on the remainder such that the synthetic treated and synthetic
control reproduce the population target on the pre-period predictors,
:math:`\bar{X} - \sum_{j \in S} w_j X_j \approx 0`. *Remark.* This is the
design analogue of the SCM convex-hull condition, and it is the only
substantive requirement. Crucially it is **not imposed as an axiom**: the
*achieved* imbalance :math:`\lVert \bar{X} - \sum_j w_j X_j\rVert` is a
measurable goodness-of-fit quantity, reported for every design, on which
the validity of the bias bound and the inference is *conditional* (Abadie
& Zhao, p.13). The analyst checks the :math:`\approx 0` condition rather
than assuming it.

*Assumption 2 (factor structure controls bias).* Under the linear factor
model, matching the treated and synthetic-control trajectories on a long
enough pre-period drives the latent loadings :math:`\mu_j` into alignment,
so the design-based bias of the average treatment effect on the treated is
bounded by the achieved imbalance and shrinks as :math:`T_0` grows and the
fit tightens. *Remark.* This is why Stage 1 minimizes imbalance rather than
any in-sample treatment contrast: imbalance is the quantity the bias bound
is written in.

*Assumption 3 (placebo exchangeability / weak stationarity).* On the blank
window :math:`B` no treatment has occurred, so the treated-minus-control
gap is pure noise; that gap process is weakly stationary and free of
anticipation, so its serial-dependence structure carries over to the
post-treatment window. *Remark.* This is what licenses the Stage 3 power
analysis: the post-treatment null distribution of the test statistic is
reconstructed by **moving-block resampling** the blank-window gaps, which
preserves autocorrelation -- the time-series-robust inference of Abadie &
Zhao rather than an i.i.d. permutation.

Stage 1 -- Treated-tuple selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a treated set :math:`S` with :math:`|S| = m` and simplex weights
:math:`w \in \Delta(S) = \{w \ge 0 : \mathbf{1}^\top w = 1\}`, the
synthetic treated unit is :math:`\sum_{j \in S} w_j X_j`. Its
**imbalance** against the population target is, over the estimation window,

.. math::

   L(S) = \min_{w \in \Delta(S)}
          \Bigl\lVert \sum_{j \in S} w_j \widetilde{X}_{E, j}
          \Bigr\rVert_2^2
        = \min_{w \in \Delta(S)} w^\top G_{SS}\, w ,

where :math:`G_{SS}` is the :math:`m \times m` sub-Gram matrix on the rows
and columns of :math:`S`. Because the design is :math:`f`-centred the
population target sits at the origin, so :math:`L(S)` is the **squared
distance from the population centroid to the convex hull of the selected
donors**, and :math:`\sqrt{L(S)}` is the achieved imbalance of
Assumption 1. Stage 1 returns the ``top_K`` sets of smallest :math:`L(S)`,
subject to the budget below.

The inner simplex quadratic program is solved by an **Away-step
Frank-Wolfe** routine (pure NumPy), batched so thousands of candidate
tuples are scored in one vectorized sweep; the returned designs are then
re-solved to tight tolerance.

Why a search and not a full enumeration or an exact MIP
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

The obvious approach -- enumerate every treated combination and keep the
best -- is defeated by combinatorics. With :math:`M` eligible markets and
:math:`m` to treat there are :math:`\binom{M}{m}` combinations. For a
modest design with :math:`M = 120` eligible markets choosing :math:`m = 4`
that is already

.. math::

   \binom{120}{4} = 8{,}214{,}570

candidate tuples, and the count explodes super-polynomially in :math:`m`:
:math:`\binom{120}{6} \approx 3.8 \times 10^{9}` and
:math:`\binom{200}{6} \approx 8.2 \times 10^{10}`. Scoring a simplex QP for
each is hopeless past small cases.

The natural fallback -- a branch-and-bound integer program with a convex
relaxation lower bound to prune -- *also* fails here, for a structural
reason specific to this objective. The population target is the
:math:`f`-weighted **centroid** of the candidate predictors, so it lies
*inside* the convex hull of the full candidate set. Any convex
(cardinality-free) relaxation of "distance from the origin to the hull of
a chosen subset" is therefore :math:`\approx 0` over the entire upper half
of the branch-and-bound tree: the relaxation cannot certify that a partial
selection is bad, so it cannot prune. An exact MIP would degenerate into
near-exhaustive enumeration with extra overhead.

Both failures are also *unnecessary*. Abadie & Zhao [ABADIE2024]_ require
only that the chosen design be feasible and **approximately balanced**;
validity is conditional on the *achieved* imbalance, a reported quantity,
not on a certificate of global optimality. LEXSCM therefore:

* **Enumerates exactly** when :math:`\binom{M}{m} \le \texttt{enumerate\_max}`
  (default :math:`3{,}000{,}000`). This is the gold standard -- it returns
  the certified global ``top_K`` and reports termination status
  ``OPTIMAL``.
* **Runs a strengthened multi-start local search** otherwise: greedy
  construction from diverse seeds, best-improvement swap descent to a local
  optimum, and basin-hopping *kicks* to escape it. In Monte Carlo this
  lands on the exact optimum 83-100% of the time and within roughly
  :math:`1\%` (mean) / :math:`7\%` (worst case) of the minimal imbalance --
  immaterial under the approximate-balance criterion. Termination status is
  ``FEASIBLE``.

Because the local search has no MIP optimality gap, LEXSCM reports a
**consensus diagnostic** in its place: the fraction of independent starts
that converged to the incumbent (``consensus_rate``), the number of
distinct local optima seen, and the incumbent-improvement trail. High
consensus across many random starts is the practical confidence signal
that the incumbent is the global optimum. (A convex hull lower bound is
*also* reported, but only as advisory information -- for the reason above
it is :math:`\approx 0` and is deliberately **not** turned into an
optimality gap.)

Budget constraints
""""""""""""""""""

When a per-unit treatment cost :math:`c_j` is supplied (``unit_cost_col``,
constant within unit) and a total ``budget`` :math:`B_{\max}`, every
returned design must satisfy the hard knapsack constraint

.. math::

   \sum_{j \in S} c_j \le B_{\max}.

Two mechanisms enforce it. First, a **sound feasibility presolve** removes
any eligible unit :math:`i` that cannot belong to *any* budget-feasible
:math:`m`-tuple, i.e. whenever

.. math::

   c_i + \sum_{\text{(}m-1\text{) cheapest other candidates}} c_j
        \;>\; B_{\max}.

This only ever drops provably impossible units; it never touches the
objective, so it cannot change which feasible design is optimal. Second,
the search itself respects the budget at every step: exact enumeration
filters combinations by their cost sum, and the local search rejects any
greedy addition, swap, or kick that would breach :math:`B_{\max}`. If no
:math:`m`-tuple fits the budget the stage reports ``INFEASIBLE``.

Stage 1 returns a solver-style diagnostics block (``stats``) with the
problem definition, presolve removals, search method and number of subsets
evaluated, the incumbent and worst-in-pool imbalance, the advisory
relaxation bound, the solution pool of ``top_K`` tuples, and termination
status / runtime.

Stage 2 -- Control fit
^^^^^^^^^^^^^^^^^^^^^^^

For each candidate treated set :math:`S` with weights :math:`w`, the
synthetic treated trajectory is :math:`X_{\cdot, S}\, w`. A synthetic
control is then built from the **non-treated** units: control weights
:math:`v` solve a ridge-penalized quadratic program (penalty
``lambda_penalty``) matching the synthetic treated over :math:`E`, subject
to an exclusion constraint :math:`v_j = 0` for all :math:`j \in S` (a
treated unit cannot also be its own control). The per-period gap is

.. math::

   e_t = (X_{\cdot, S}\, w)_t - (X v)_t .

Pre-treatment fit is summarized by the normalized mean-squared error
(NMSE) of the synthetic treated against the target on both windows
(``nmse_E``, ``nmse_B``). The **blank-window gaps**
:math:`e_B = \{e_t : t \in B\}` (``residuals_B``) are, under Assumption 3,
pure placebo noise; they are the raw material for the power analysis. This
stage is identical to the pre-rebuild control path -- only the search and
power stages around it changed.

Stage 3 -- Power analysis (minimum detectable effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 3 asks: *how large must a treatment effect be for this design to
detect it?* The answer is computed at each horizon :math:`h` in
``n_post_grid`` (default :math:`h = 2, \dots, 8`) from the placebo
residuals, using one consistent resampling model for both the null and the
alternative.

**Moving-block resampling.** Let :math:`\sigma` be the standard deviation
of the placebo residual pool. To build a horizon-:math:`h` window we
sample contiguous blocks of length :math:`\ell \approx L^{1/3}` (where
:math:`L` is the residual series length) with wraparound and concatenate
them to length :math:`h`. This preserves the within-series autocorrelation
of the gap process (Assumption 3). When several placebo series are
supplied (the candidate's blank gaps plus donor-unit placebo gaps), each
draw first picks a series at random, reproducing the cross-unit placebo
distribution.

**Test statistic and critical value.** The statistic is the mean absolute
gap over the window,

.. math::

   S(e) = \frac{1}{h} \sum_{t=1}^{h} \lvert e_t \rvert .

Resampling :math:`n_\text{null}` placebo windows gives the null
distribution; its :math:`(1 - \alpha)` quantile is the critical value
:math:`c_\alpha`.

**Power and the MDE.** The power of a constant additive effect
:math:`\tau` is estimated by resampling the *same* residual structure,
adding the shift, and measuring exceedances,

.. math::

   \operatorname{power}(\tau)
     = \Pr\bigl[\, S(e + \tau) \ge c_\alpha \,\bigr],

so the null and the alternative share one residual model -- they differ
only by the location shift :math:`\tau`. The **minimum detectable effect**
is the smallest effect reaching the target power,

.. math::

   \mathrm{MDE}(h) = \min\bigl\{\tau \ge 0 :
                     \operatorname{power}(\tau) \ge \texttt{power\_target}\bigr\},

found on an adaptive grid in standard-deviation units
:math:`\tau / \sigma \in [0, \texttt{max\_sd}]` (default cap :math:`8`),
with linear interpolation for a finer value. The MDE is reported both in
**effect-size units** (``mde_sd`` :math:`= \mathrm{MDE}/\sigma`) and in
**outcome units** (``mde_abs`` :math:`= \mathrm{MDE}`); if the target power
is never reached within the grid the horizon returns :math:`\infty`
(infeasible).

Reporting the MDE in standard-deviation units -- rather than relative to a
fitted outcome level -- is both the natural effect-size scale (matching
Vives-i-Bastida's "detect effects larger than :math:`0.1` s.d."
convention) and numerically robust: there is no division by a fragile mean,
so zero-mean or near-zero outcomes do not blow the calculation up.

**Detectability curve and horizon collapse.** Sweeping the horizon grid
yields the detectability curve :math:`h \mapsto \mathrm{MDE}(h)`. The
estimator collapses it to a single representative MDE per the
``mde_horizon`` setting:

* ``"late"`` (default) -- the MDE at the **longest** horizon; the
  conservative choice for a sustained-exposure experiment.
* ``"early_min"`` -- the **smallest** MDE across horizons (most
  optimistic detectability).
* ``"early_mean"`` -- the **mean** MDE across feasible horizons.

Stage 4 -- Final recommendation (lexicographic selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final stage turns the per-design metrics -- imbalance (validity),
``mde_sd`` (power), ``stability`` :math:`=` out-of-sample
:math:`\text{NMSE}_B` (fit robustness), and ``total_cost`` -- into one
recommendation, applying the method's priorities in strict order:

#. **Validity gate.** Keep only designs whose imbalance is within a
   relative slack of the best achievable balance,

   .. math::

      \text{imbalance}(S) \le
      \bigl(1 + \texttt{imbalance\_tol}\bigr)\cdot
      \min_{S'} \text{imbalance}(S') ,

   (default ``imbalance_tol`` :math:`= 0.25`). This is the set of designs
   that satisfy Assumption 1 well enough to be trustworthy. (A degenerate
   tolerance that would empty the gate falls back to the single
   best-balanced design.)

#. **Power.** Among gated designs with a *feasible* MDE, choose the
   smallest ``mde_sd`` -- the most detectable valid design.

#. **Tie-breaks.** Break ties by better out-of-sample stability
   (:math:`\text{NMSE}_B`), then by lower ``total_cost``.

The recommendation is returned as a tuple of fields:
``winner`` (the chosen design), ``shortlist`` (the top
``max_shortlist`` ranked designs), ``pareto_ids`` (the Pareto frontier on
imbalance :math:`\downarrow` vs. ``mde_sd`` :math:`\downarrow`, always
exposed for transparency), a ``status``, a human-readable
``explanation``, and a per-design ``table`` (which becomes
``results.summary``). The status is one of:

* ``OK`` -- a valid, adequately powered design was found;
* ``POWER_NOT_ESTABLISHED`` -- no gated design reached the power target
  within the effect grid, so the best-balanced design is recommended and
  the power caveat is flagged (the pipeline degrades gracefully rather than
  crashing);
* ``EMPTY`` -- no candidate designs were supplied.

Core API
--------

.. automodule:: mlsynth.estimators.lexscm
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.LEXSCMConfig
   :members:
   :undoc-members:

Helper Modules
--------------

Treated-tuple selection (Stage 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exact enumeration / multi-start local search over treated
:math:`m`-tuples, with the budget presolve and the away-step Frank-Wolfe
simplex solver.

.. automodule:: mlsynth.utils.fast_scm_helpers.lexsearch
   :members: select_treated_designs, TreatedDesign
   :undoc-members:

Control fit (Stage 2)
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_control
   :members: evaluate_candidates
   :undoc-members:

Power / MDE (Stage 3)
^^^^^^^^^^^^^^^^^^^^^^

Moving-block placebo minimum-detectable-effect analysis.

.. automodule:: mlsynth.utils.fast_scm_helpers.lexpower
   :members: detectability_curve, compute_mde, block_resample_windows
   :undoc-members:

Recommendation (Stage 4)
^^^^^^^^^^^^^^^^^^^^^^^^^

Lexicographic design selection.

.. automodule:: mlsynth.utils.fast_scm_helpers.lexselect
   :members: select_design, DesignMetrics, Recommendation
   :undoc-members:

Data preparation and matrix construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for constructing the design matrices, the population target, the
estimation/blank split, and the Gram matrix.

.. automodule:: mlsynth.utils.fast_scm_helpers.fast_scm_setup
   :members: prepare_experiment_inputs, split_periods, build_X_tilde,
       build_candidate_mask, build_f_vector, build_Y_matrix, build_Z_matrix
   :undoc-members:

Example: choosing treated markets under a budget
================================================

This end-to-end example simulates a panel of sales markets, flags a subset
as treatment-eligible with heterogeneous treatment costs, and lets LEXSCM
choose which :math:`m = 3` markets to treat under a fixed budget -- then
inspects the solver-style search diagnostics and the lexicographic
recommendation.

Generate a synthetic panel
--------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd

    from mlsynth import LEXSCM


    def generate_synthetic_sales_panel(
        n_units: int = 100,
        n_time_periods: int = 100,
        n_candidates: int = 40,
        treatment_start: int = 80,
        seed: int = 42,
        sigma: float = 3.0,
        sales_scale: float = 100.0,
    ) -> pd.DataFrame:

        np.random.seed(seed)

        unit_fe = np.random.normal(0, 8, size=n_units)
        unit_trend = np.random.normal(0.3, 0.15, size=n_units)
        unit_sensitivity = np.random.uniform(0.8, 1.2, size=n_units)

        t = np.arange(n_time_periods)
        common_factor = (
            1.5 * t + 120
            + 12 * np.sin(2 * np.pi * t / 52)
            + np.random.normal(0, 3.5, n_time_periods)
        )

        sales = np.zeros((n_time_periods, n_units))
        for j in range(n_units):
            sales[:, j] = (
                sales_scale
                + common_factor * unit_sensitivity[j]
                + unit_fe[j]
                + unit_trend[j] * t
                + np.random.normal(0, sigma, n_time_periods)
            )
        sales = np.maximum(sales, 5.0)

        # heterogeneous per-market treatment costs
        base_cost = np.random.lognormal(mean=12, sigma=0.6, size=n_units) * 5
        size_factor = np.random.uniform(0.7, 1.4, n_units)
        treatment_cost = np.round((base_cost * size_factor) / 10) * 10

        unit_ids = np.repeat(np.arange(n_units), n_time_periods)
        time_ids = np.tile(np.arange(n_time_periods), n_units)
        sales_flat = sales.ravel(order="F")

        df = pd.DataFrame({
            "unitid": unit_ids,
            "time": time_ids,
            "sales": sales_flat,
            "treatment_cost": np.repeat(treatment_cost, n_time_periods),
        })

        candidate_mask = np.zeros(n_units, dtype=bool)
        candidate_idx = np.random.choice(n_units, size=n_candidates, replace=False)
        candidate_mask[candidate_idx] = True
        df["candidate"] = np.repeat(candidate_mask, n_time_periods)
        df["post"] = (df["time"] >= treatment_start).astype(int)

        return df

Run LEXSCM and inspect the design
---------------------------------

.. code-block:: python

    df = generate_synthetic_sales_panel(
        n_units=120,
        n_candidates=40,
        n_time_periods=100,
        treatment_start=90,
        seed=4545,
    )

    config = {
        "df": df,
        "outcome": "sales",
        "unitid": "unitid",
        "time": "time",
        "candidate_col": "candidate",
        "m": 3,                         # treat 3 of the 40 eligible markets
        "post_col": "post",
        "unit_cost_col": "treatment_cost",
        "budget": 4_000_000,            # hard knapsack cap on selected costs
        "lambda_penalty": 0.5,
        "mde_horizon": "late",          # conservative MDE at the longest horizon
        "power_target": 0.8,
        "imbalance_tol": 0.25,          # validity gate: within 25% of best balance
        "top_K": 20,
    }

    results = LEXSCM(config).fit()

    # --- Stage 1: solver-style search diagnostics ---
    stats = results.bnb_metadata["stats"]
    print("status        :", stats["termination"]["status"])      # OPTIMAL / FEASIBLE
    print("method        :", stats["search"]["method"])
    print("subsets scored :", stats["search"]["subsets_evaluated"])
    print("C(M,m)        :", stats["problem"]["feasible_region_C(M,m)"])
    print("best imbalance :", round(stats["incumbent"]["imbalance"], 4))

    # --- Stage 4: lexicographic recommendation tuple ---
    rec = results.bnb_metadata["recommendation"]
    print("rec status    :", rec["status"])      # OK / POWER_NOT_ESTABLISHED
    print("winner        :", rec["winner"])
    print("pareto ids    :", rec["pareto_ids"])
    print(rec["explanation"])

    # --- the recommended design ---
    print(results.summary)                        # ranked per-design table
    best = results.best_candidate
    print("treated weights:", best.treated_weight_dict)
    print("control weights:", best.control_weight_dict)

With :math:`M = 40` eligible markets and :math:`m = 3`,
:math:`\binom{40}{3} = 9{,}880` is well under ``enumerate_max``, so the
search **enumerates exactly** and reports ``status = OPTIMAL`` -- a
certified global ``top_K``. Raising :math:`M` and :math:`m` until
:math:`\binom{M}{m}` exceeds the cap flips the search to the multi-start
local solver (``status = FEASIBLE``), at which point the ``consensus``
block under ``stats["search"]`` reports how many independent starts agreed
on the incumbent.

References
----------

Abadie, A., & Zhao, J. (2024/2026). "Synthetic Controls for Experimental
Design." *arXiv Working Paper* 2108.02196. (The February 2026 revision
develops the design-based bias bound and the approximate-balance criterion
used here.) https://arxiv.org/abs/2108.02196

Vives-i-Bastida, J. (2022). "Synthetic Experimental Design." Working paper
-- source of the standard-deviation MDE scale and the power-analysis
conventions used in Stage 3.

Vives-i-Bastida, J. (2022). "Predictor Selection for Synthetic Controls."
*Working paper*, arXiv:2203.11576.

Doudchenko, N., Khosravi, K., Pouget-Abadie, J., Lahaie, S., et al.
"Synthetic Design: An Optimization Approach to Experimental Design with
Synthetic Controls." -- related optimization view of synthetic
experimental design.

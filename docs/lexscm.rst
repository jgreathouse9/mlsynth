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

How a single tuple is built: the inner simplex QP
"""""""""""""""""""""""""""""""""""""""""""""""""

"Building" a tuple :math:`S` means solving its inner problem
:math:`\min_{w \in \Delta(S)} w^\top G_{SS}\, w` for the **optimal
treatment weights** :math:`w(S)`; the synthetic treated unit is then
:math:`\sum_{j \in S} w_j(S)\, X_j` and the design's achieved imbalance is
:math:`\sqrt{w(S)^\top G_{SS}\, w(S)}`. This convex quadratic program over
the probability simplex is solved by an **Away-step Frank-Wolfe (AFW)**
routine in pure NumPy (``_afw_single``), chosen because every iterate stays
exactly on the simplex (no projection step) and the *away* move removes the
zig-zagging that plain Frank-Wolfe suffers near a face-constrained optimum
-- so the support of :math:`w` (which donors carry positive weight) sharpens
in a handful of iterations.

Write :math:`Q = G_{SS}`, :math:`f(w) = w^\top Q w`,
:math:`\nabla f(w) = 2 Q w`. From a vertex start
:math:`w^{(0)} = e_{\arg\min_i Q_{ii}}` (the single donor closest to the
target), each iteration:

#. **Pick two vertices.** The Frank-Wolfe vertex
   :math:`s = \arg\min_i [\nabla f(w)]_i` and the away vertex
   :math:`a = \arg\max_{i \in \operatorname{supp}(w)} [\nabla f(w)]_i` --
   the currently active donor the gradient most wants to shed.
#. **Choose the direction.** Compare the FW direction
   :math:`d_{\mathrm{FW}} = e_s - w` against the away direction
   :math:`d_{\mathrm{AW}} = w - e_a` by :math:`\langle \nabla f, d\rangle`
   and take whichever descends more. The away step's maximal feasible
   length is :math:`\gamma_{\max} = w_a / (1 - w_a)`; the FW step caps at
   :math:`1`.
#. **Exact line search.** Because :math:`f` is quadratic the optimal step
   along :math:`d` is closed-form,

   .. math::

      \gamma^\star = \operatorname{clip}\!\Bigl(
        -\frac{w^\top Q d}{d^\top Q d},\; 0,\; \gamma_{\max}\Bigr),

   then :math:`w \leftarrow w + \gamma^\star d`, dropping :math:`a` from the
   active set when a full away step empties it.
#. **Certified lower bound.** The Frank-Wolfe gap gives a running lower
   bound :math:`f(w) + \min_i[\nabla f(w)]_i - \nabla f(w)^\top w \le
   f(w^\star)` on the tuple's true minimal loss; iteration stops once the
   duality gap :math:`\nabla f(w)^\top w - \min_i[\nabla f(w)]_i` drops
   below ``tol``.

**Two-pass precision.** Scoring every candidate tuple to convergence would
be wasteful, so Stage 1 runs AFW at two fidelities. A **vectorized
batched** AFW (``_afw_batched``, ``iters=80``, all :math:`m \times m`
problems advanced together with ``einsum``) ranks thousands of tuples in
one sweep; only the surviving ``top_K`` are **re-solved** by the scalar
``_afw_single`` at ``iters=600``, ``tol=1e-14`` to pin :math:`w(S)` and the
loss to full precision. Each returned
:class:`~mlsynth.utils.fast_scm_helpers.lexsearch.TreatedDesign` then
carries its ``indices``, the high-precision ``weights`` :math:`w(S)`,
``loss`` :math:`= L(S)`, ``imbalance`` :math:`= \sqrt{L(S)}`,
``total_cost``, and a label-keyed ``weight_dict``.

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

Building tuples in the heuristic regime
"""""""""""""""""""""""""""""""""""""""

When :math:`\binom{M}{m}` exceeds ``enumerate_max`` the same per-tuple QP is
reused, but the *set* of tuples scored is grown adaptively by a multi-start
local search (``_local_search``):

* **Seeding.** :math:`2 \times` ``n_starts`` seeds: the ``n_starts`` units
  with the smallest :math:`G_{jj}` (single donors already nearest the
  population centroid -- the cheapest places to start near balance) plus
  ``n_starts`` uniformly random units for diversity.
* **Greedy construction.** From a seed, repeatedly add the candidate that
  most lowers the batched loss of the partial tuple until :math:`|S| = m`,
  skipping any addition that would breach the budget.
* **Best-improvement descent.** Score the full **1-swap** neighbourhood
  (replace one member with one non-member), move to the steepest improving
  swap, and repeat to a local optimum.
* **Basin-hopping kicks.** Perturb the incumbent with a random **2-swap**
  ``kick`` and re-descend, keeping the better basin; repeated ``n_kicks``
  (4) times per start.

Every distinct tuple ever scored is cached, and the global ``top_K`` of
that cache is returned and re-solved to full precision exactly as in the
exact path. The ``consensus`` block -- how many independent starts' final
optima coincide with the incumbent -- is the confidence diagnostic that
stands in for the absent MIP gap.

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

Stage 3 asks: *how large must a sustained treatment effect be for this
design to detect it?* The answer is a **minimum-detectable-effect (MDE)
curve over post-treatment horizon lengths**, computed entirely from the
Stage 2 placebo residuals with one consistent resampling model for both the
null and the alternative.

What "at each horizon" means -- and what it does not
""""""""""""""""""""""""""""""""""""""""""""""""""""

The curve is indexed by the **length** :math:`h` of the post-treatment
window (``n_post_grid``, default :math:`h = 2, \dots, 8`), *not* by calendar
period. There is no "MDE at period :math:`t = 91`". Instead, for each
candidate duration :math:`h`, the analysis *manufactures* many synthetic
:math:`h`-long gap windows by resampling the placebo residuals and asks what
constant effect a window of that length could detect. Reading the curve as
:math:`h` grows tells you how detectability improves the longer the
experiment runs.

Moving-block resampling
"""""""""""""""""""""""

Let the placebo pool be one or more residual series -- the chosen design's
blank-window gaps :math:`e_B`, optionally pooled with donor-unit placebo
gaps. The scale :math:`\sigma` is the standard deviation of the **pooled**
residuals (all series concatenated, ``ddof=1``, floored at
:math:`10^{-12}`). The block length defaults to

.. math::

   \ell = \max\!\bigl(1,\ \min(h,\ \operatorname{round}(\tilde L^{1/3}))\bigr),

where :math:`\tilde L` is the **median** series length: floored at 1, and
**capped at the horizon** :math:`h` so a block never exceeds the window it
fills. To draw one length-:math:`h` window
(:func:`~mlsynth.utils.fast_scm_helpers.lexpower.block_resample_windows`):
pick a placebo series uniformly at random, then repeatedly cut contiguous
blocks of length :math:`\ell` from a random offset **with wraparound**
(``np.take(..., mode="wrap")``), concatenate them, and truncate to
:math:`h`. Wrapping preserves the within-series autocorrelation
(Assumption 3); the random series choice reproduces the cross-unit placebo
distribution.

Test statistic and critical value
"""""""""""""""""""""""""""""""""

The statistic is the mean absolute gap over the window,

.. math::

   S(e) = \frac{1}{h} \sum_{t=1}^{h} \lvert e_t \rvert .

Resampling :math:`n_\text{null}` (default 4000) placebo windows gives the
null distribution; its :math:`(1 - \alpha)` **empirical quantile** is the
critical value :math:`c_\alpha`.

Power, the effect grid, and the MDE
"""""""""""""""""""""""""""""""""""

A treatment effect is modelled as a **constant additive shift** :math:`\tau`
applied to *every* point of a resampled window -- the homogeneous-effect
working assumption behind the MDE. Its power is estimated from a **fresh**
draw of :math:`n_\text{power}` (default 2000) windows (same residual model,
new randomness), so the null and the alternative differ only by the shift:

.. math::

   \operatorname{power}(\tau)
     = \Pr\bigl[\, S(e + \tau) \ge c_\alpha \,\bigr]
     \approx \frac{1}{n_\text{power}}
       \sum_{b} \mathbf{1}\!\bigl\{ S(e^{(b)} + \tau) \ge c_\alpha \bigr\}.

The effect is swept on a grid of :math:`n_\text{grid} = 64` points in
**standard-deviation units**, :math:`\tau / \sigma \in [0, \texttt{max\_sd}]`
(default cap :math:`8`). The search walks the grid until
:math:`\operatorname{power}(\tau)` first reaches ``power_target``, then
**linearly interpolates** between the last sub-threshold point
:math:`(g_0, p_0)` and the crossing point :math:`(g, p)` for a finer value:

.. math::

   \frac{\mathrm{MDE}(h)}{\sigma}
     = g_0 + (\texttt{power\_target} - p_0)\,\frac{g - g_0}{p - p_0} .

If the grid is exhausted without reaching the target, the horizon is
**infeasible** and returns :math:`\infty`.

Three reported scales
"""""""""""""""""""""

Each horizon reports the MDE on three scales:

* ``mde_sd`` :math:`= \mathrm{MDE}(h)/\sigma` -- **effect-size units**, the
  primary scale (matching Vives-i-Bastida's "detect effects larger than
  :math:`0.1` s.d." convention) and numerically robust: nothing is divided
  by a fragile mean, so zero-mean or near-zero outcomes cannot blow the
  calculation up;
* ``mde_abs`` :math:`= \mathrm{MDE}(h)` -- **outcome units**;
* ``mde_pct`` -- the **manager-facing percentage**,
  :math:`100 \cdot \mathrm{MDE}(h) / \lvert\text{baseline}\rvert`, where the
  baseline is the counterfactual level over the *matching* window,
  :math:`\text{baseline} = \operatorname{mean}\bigl(\texttt{synthetic\_treated}
  [-h:]\bigr)`. This is **guarded**: when
  :math:`\lvert\text{baseline}\rvert` falls below a floor (default one
  :math:`\sigma`) the percentage is returned as ``NaN`` *deliberately*, so a
  near-zero or sign-flipping level cannot manufacture a spurious "we can
  detect a 0.3% effect."

Detectability curve and horizon collapse
""""""""""""""""""""""""""""""""""""""""

:func:`~mlsynth.utils.fast_scm_helpers.lexpower.detectability_curve` sweeps
the horizon grid and returns ``curve_sd``
(:math:`h \mapsto \mathrm{MDE}(h)/\sigma`), ``curve_pct`` (the percentage
curve), the per-horizon ``details``, and ``min_horizon_mde_le_0p1sd`` -- the
shortest horizon whose MDE falls to :math:`0.1\sigma`, a quick read on how
long the experiment must run to clear the conventional effect-size bar.
Stage 4 then collapses the curve to a single representative MDE per the
``mde_horizon`` setting:

* ``"late"`` (default) -- the MDE at the **longest** horizon; the
  conservative choice for a sustained-exposure experiment.
* ``"early_min"`` -- the **smallest** MDE across feasible horizons (most
  optimistic detectability).
* ``"early_mean"`` -- the **mean** MDE across feasible horizons (the
  percentage averaged only over horizons where it is defined).

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

Worked numeric walkthrough
--------------------------

To make the two core mechanisms concrete, here is a single tuple built and
powered end to end on a deliberately tiny panel (:math:`T_0 = 6` pre-period
rows, :math:`J = 5` units, treat :math:`m = 2`). Every number below is the
actual helper output, not an illustration.

Building the tuple (Stage 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After :math:`f`-centring and row-standardising the predictors, the Gram
matrix's diagonal -- each unit's squared distance to the population centroid
-- is

.. math::

   \operatorname{diag}(G) = (5.77,\ 3.64,\ 6.08,\ 6.27,\ 8.24).

Unit 1 (0-indexed) sits closest to the target, so it seeds the AFW vertex
start. With :math:`\binom{5}{2} = 10` the search **enumerates exactly**
(status ``OPTIMAL``, 10 subsets scored). The winning tuple is
:math:`S = \{0, 1\}`, and its inner simplex QP returns treatment weights

.. math::

   w(S) = (0.4098,\ 0.5902), \qquad
   L(S) = w^\top G_{SS}\, w = 1.6481, \qquad
   \sqrt{L(S)} = 1.2838 .

The standalone high-precision re-solve reproduces this loss with a
Frank-Wolfe lower bound equal to it to :math:`10^{-6}` -- the QP is at its
certified optimum. The synthetic treated unit is
:math:`0.41\,X_0 + 0.59\,X_1`, and :math:`1.2838` is the achieved imbalance
on which the bias bound (Assumption 1-2) is conditional.

Powering the design (Stage 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take this design's blank-window placebo gaps to be a length-:math:`L = 24`
series with :math:`\sigma = 0.8695`. The auto block length is
:math:`\ell = \min\bigl(h, \operatorname{round}(24^{1/3})\bigr) =
\min(h, 3)`, so at :math:`h = 2` it is capped to :math:`2` and at
:math:`h \ge 3` it is :math:`3`. With a counterfactual level of
:math:`\text{baseline} = 12.0`, the per-horizon MDE is:

.. list-table::
   :header-rows: 1

   * - :math:`h`
     - :math:`\ell`
     - :math:`c_\alpha`
     - ``mde_sd``
     - ``mde_abs``
     - ``mde_pct``
   * - 2
     - 2
     - 1.245
     - 2.037
     - 1.771
     - 14.8%
   * - 4
     - 3
     - 1.108
     - 1.672
     - 1.454
     - 12.1%
   * - 8
     - 3
     - 0.981
     - 1.282
     - 1.115
     - 9.3%

Reading down the table *is* the detectability curve: as the horizon
lengthens, averaging over more periods shrinks both the critical value
:math:`c_\alpha` and the MDE, so a sustained effect of
:math:`\approx 9\%` of the counterfactual becomes detectable by
:math:`h = 8`, versus :math:`\approx 15\%` at :math:`h = 2`. Under
``mde_horizon="late"`` Stage 4 carries the :math:`h = 8` row
(``mde_sd`` :math:`= 1.28`) as this design's power score; under
``"early_min"`` it would carry the smallest feasible ``mde_sd``.

Standardized Post-Fit and Power Analysis
----------------------------------------

LEXSCM's Stage 3 already produces the lexicographic MDE curve used to **rank
designs against each other**; the standardized post-fit attached to
``LEXSCMResults.post_fit`` is the complementary surface used **once a design
has been chosen** -- it is the same
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` object SYNDES,
MAREX, and PANGEO expose, so downstream consumers (dashboards, paper-style
reports, comparison tables) read identical fields across the family.

.. code-block:: python

   pf = res.post_fit                          # SyntheticControlPostFit
   pf.ate, pf.ate_percent, pf.total_effect    # treatment-effect scalars
   pf.rmse_fit, pf.rmse_blank, pf.rmse_post   # pre / blank / post fit quality
   pf.p_value, pf.ci_lower, pf.ci_upper       # conformal inference
   pf.power                                   # PowerAnalysis (see below)

The trajectories ``pf.treated_series`` and ``pf.control_series`` are the
**winning candidate**'s ``predictions.synthetic_treated`` and
``predictions.synthetic_control`` -- per-unit weighted aggregates of the
treated and control donors over the full timeline. The phase boundaries
``(n_fit, n_blank, n_post)`` line up with the same E / B / post split LEXSCM
uses internally (so ``n_fit + n_blank = T0`` and the 30%-of-pre-tail
holdout convention applies via ``frac_E = 0.7``).

Where the unified power analysis fits in
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 3's :func:`~mlsynth.utils.fast_scm_helpers.lexpower.detectability_curve`
is the **design-selection** workhorse -- it converts the moving-block placebo
null on the B window into a per-horizon MDE used by Stage 4's lexicographic
selector. The post-fit power analysis is the **post-selection** companion:
analytical Gaussian + AR(1) variance inflation on the realised gap residuals,
matching the MAREX / SYNDES / PANGEO power surfaces exactly so the same
diagnostic table can be produced for every family member.

.. math::

   \mathrm{MDE}(T) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr) \cdot
       \hat\sigma_{\text{placebo}} \cdot \sqrt{\mathrm{VIF}(T, \hat\rho)},

with :math:`\hat\sigma_{\text{placebo}}` the SD of the gap on the B (blank /
holdout) window, :math:`\hat\rho` the lag-1 autocorrelation of those
residuals clipped to :math:`(-0.99, 0.99)`, and
:math:`\mathrm{VIF}(T, \rho) = \tfrac{1}{T}\bigl(1 + 2\sum_{k=1}^{T-1}
(1-k/T)\rho^k\bigr)` the standard AR(1) variance-inflation factor (textbook
:math:`1/T` when :math:`\rho = 0`). See :doc:`marex` for the full derivation;
the same module powers all three estimators.

.. code-block:: python

   p = res.post_fit.power                      # PowerAnalysis
   p.headline.mde_absolute                     # MDE at the realised T_post
   p.headline.mde_pct                          # ... as % of post-period baseline
   p.headline.power_at_observed                # power to detect res.post_fit.ate
   p.curve                                     # tuple of MDEPoint per horizon
   p.sigma_placebo                             # σ̂ used (B window in LEXSCM)
   p.serial_correlation                        # ρ̂ AR(1) of the B residuals

Two MDEs, complementary roles
"""""""""""""""""""""""""""""

* **Stage 3 MDE** (``best_candidate.mde_results``) -- moving-block placebo
  null on the B window, used to *rank designs against each other*. Aggregated
  to a representative scalar by ``mde_horizon`` (``late`` / ``early_min`` /
  ``early_mean``) and consumed by Stage 4's lexicographic gate.
* **Post-fit MDE** (``res.post_fit.power``) -- analytical Gaussian +
  AR(1) MDE consumed *after* a design has been chosen, on the same surface
  that MAREX / SYNDES / PANGEO produce. Use this when reporting a single
  detectability number alongside the realised ATE / CI.

Power-analysis failures (e.g. degenerate B-window residuals) never break a
fit; ``res.post_fit.power`` is simply left as ``None``. To compute on a
non-default horizon grid or significance level call
:func:`~mlsynth.utils.post_fit.compute_power_analysis` directly.

Verification
------------

LEXSCM is validated against the synthetic-experimental-design framework it
implements (Abadie & Zhao 2026; Vives-i-Bastida 2022). **Path A:** the Walmart
45-store placebo design tracks pre-period to ~2.7% of mean sales and yields a
placebo effect of ~0.9% whose permutation test fails to reject (CI covers zero)
-- the paper's "no spurious effect" result. **Path B:** on the paper's exact
Section-5 linear-factor DGP (``J = 15``, ``T_E = 20``) the design recovers the
planted effect with MAE far below its scale (~0.24 for the single-treated-unit
design, falling to ~0.16 at ``m = 2``). Pinned in
``benchmarks/cases/lexscm_walmart.py`` and ``lexscm_design_mc.py``; see the
dedicated page :doc:`replications/lexscm`.

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

Result Containers
-----------------

``LEXSCM.fit()`` returns a
:class:`~mlsynth.utils.fast_scm_helpers.structure.LEXSCMResults`, which is a
:class:`~mlsynth.config_models.DesignResult` -- the experimental-design half of
mlsynth's two-family result contract. LEXSCM *chooses which units to treat*
before any intervention, so it returns a **design** that resolves to an effect
report. The standardized design fields are:

* ``res.report`` -- an :class:`~mlsynth.config_models.EffectResult` built from the
  realized post-fit (the same flat ``att`` / ``counterfactual`` / ``gap`` /
  ``att_ci`` surface every effect estimator exposes);
* ``res.selected_units`` -- the treated units chosen by the design;
* ``res.assignment`` -- the treated / control split;
* ``res.design_weights`` -- the synthetic-control weights implied by the design
  (a :class:`~mlsynth.config_models.WeightsResults`);
* ``res.power`` -- the design's MDE / power analysis;
* ``res.metadata`` -- the lexicographic recommendation diagnostics.

Alongside the contract fields it keeps the LEXSCM-specific search structure: the
winning :class:`~mlsynth.utils.fast_scm_helpers.structure.SEDCandidate`
(``res.best_candidate``), the full candidate shortlist (``res.summary`` /
``res.all_candidates``), the Stage-1 branch-and-bound metadata
(``res.bnb_metadata``), and the time / unit metadata blocks. ``res.post_fit`` is
the standardized
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` shared across the
MAREX family (LEXSCM / MAREX / SYNDES / PANGEO): same ATE / RMSE / SMD /
inference / power surface, regardless of which estimator produced the design
(``res.report`` is the contract-standard view of the same realization).

.. autoclass:: mlsynth.utils.fast_scm_helpers.structure.LEXSCMResults
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.post_fit.SyntheticControlPostFit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.post_fit.PowerAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.post_fit.MDEPoint
   :members:
   :undoc-members:
   :show-inheritance:

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

Standardized post-fit (shared across the MAREX family) -- the
:func:`~mlsynth.utils.post_fit.compute_post_fit` and
:func:`~mlsynth.utils.post_fit.compute_power_analysis` helpers that
populate ``results.post_fit`` live outside this package so SYNDES, MAREX,
and PANGEO all consume the same diagnostics module:

.. automodule:: mlsynth.utils.post_fit
   :members:
   :undoc-members:
   :show-inheritance:

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

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental
Design." *arXiv Working Paper* 2108.02196. (The February 2026 revision
develops the design-based bias bound and the approximate-balance criterion
used here.) https://arxiv.org/abs/2108.02196

Vives-i-Bastida, J. (2022). "Synthetic Experimental Design for a UBI pilot study." Working paper (https://ivalua.cat/sites/default/files/2023-03/Vives-i-Bastida_2022_anon.pdf)

Doudchenko, N., Khosravi, K., Pouget-Abadie, J., Lahaie, S., et al.
"Synthetic Design: An Optimization Approach to Experimental Design with
Synthetic Controls." -- related optimization view of synthetic
experimental design.

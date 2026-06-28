Lexicographic Synthetic Control (LEXSCM)
========================================

.. currentmodule:: mlsynth

Overview
--------

LEXSCM is a tool for synthetic experimental design: given a panel of
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

#. Validity -- the treated combination reproduces the population
   trajectory on the pre-treatment predictors (small *imbalance*);
#. Power -- subject to validity, the design detects the smallest
   possible effect (small *minimum detectable effect*).

The pipeline has four stages, each in its own helper module:

#. Treated-tuple selection (:mod:`~mlsynth.utils.fast_scm_helpers.lexsearch`)
   -- search the :math:`\binom{M}{m}` candidate treated combinations for
   the most balanced ones, under an optional budget.
#. Control fit (:mod:`~mlsynth.utils.fast_scm_helpers.fast_scm_control`)
   -- build a synthetic control for each candidate treated group and
   score its pre-treatment fit.
#. Power analysis (:mod:`~mlsynth.utils.fast_scm_helpers.lexpower`)
   -- a moving-block placebo MDE curve over a grid of post-treatment
   horizons.
#. Recommendation (:mod:`~mlsynth.utils.fast_scm_helpers.lexselect`)
   -- a lexicographic rule (validity gate :math:`\to` power
   :math:`\to` stability :math:`\to` cost) returning a single recommended
   design plus a Pareto frontier.

When to use this estimator
--------------------------

Reach for LEXSCM before an experiment, not after one. The typical setting is a
geo-experiment or marketing test: you have a panel of markets, only some of
which can be treated (a budget, an operational constraint, or a policy makes the
rest off-limits), and you must decide *which* of the eligible markets to treat.
Treat too few or the wrong ones and the remaining markets cannot reconstruct the
treated trajectory, so the effect is biased; treat markets whose gap is too
noisy and the experiment cannot detect the lift you care about. LEXSCM chooses
the treated combination so that the untreated markets form a credible synthetic
control (validity) and the resulting design has the power to detect a
meaningfully small effect (power), in that strict priority order.

A concrete example: a retailer plans a price promotion and can run it in at most
three of its forty largest metros. LEXSCM searches the eligible metros, returns
the three whose combined sales path the rest of the chain can reproduce most
closely on the pre-period, and reports the minimum lift that design could detect
over a planned eight-week window -- together with a budget gate, geography
(spillover) exclusions, and coverage quotas so the chosen markets are
non-adjacent and spread across regions. Once the promotion runs, the same object
realizes into a standard effect report.

Notation
--------

There are :math:`N` units :math:`\mathcal{N} \coloneqq \{1, \dots, N\}`,
observed over the pre-treatment window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` of length
:math:`T_0`, with :math:`\mathcal{T} \coloneqq \{1, \dots, T\}` 1-indexed and the
intervention taking effect after :math:`T_0` (the post-period
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`; the design phase
uses only :math:`\mathcal{T}_1`). Unit :math:`j`'s outcome at time :math:`t` is
:math:`y_{jt}`. A subset :math:`\mathcal{E} \subseteq \mathcal{N}` of size
:math:`M = |\mathcal{E}|` is flagged as treatment-eligible (the
``candidate_col``); the design must pick :math:`m` of these as the treated set
:math:`\mathcal{S} \subseteq \mathcal{E}`, :math:`|\mathcal{S}| = m`. Treatment
weights on the chosen treated set are :math:`\mathbf{w}` and control weights on
the remainder are :math:`\mathbf{v}`, both on a simplex; an optimiser carries the
star superscript :math:`\mathbf{w}^\ast`. The per-period gap between the
synthetic treated and synthetic control is :math:`e_t`; a sustained treatment
effect is :math:`\tau` (the quantity the power analysis sizes), and :math:`\alpha`
is the significance level. The interference graph is
:math:`\mathbf{A}` with conflict-neighbours :math:`\mathcal{A}(\mathcal{S})`
(below).

Mathematical Formulation
------------------------

Setup and notation
^^^^^^^^^^^^^^^^^^^

We observe :math:`N` units :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` over a
pre-treatment period :math:`\mathcal{T}_1` of length :math:`T_0`. Stack the
pre-period outcomes as
:math:`\mathbf{Y} \in \mathbb{R}^{T_0 \times N}` (rows = time, columns = units)
and, optionally, :math:`K` time-invariant or pre-period covariates as
:math:`\mathbf{Z} \in \mathbb{R}^{K \times N}`. The two are stacked vertically
into the predictor matrix

.. math::

   \mathbf{X} = \begin{bmatrix} \mathbf{Y} \\ \mathbf{Z} \end{bmatrix}
       \in \mathbb{R}^{(T_0 + K) \times N},

so each column :math:`\mathbf{x}_j` is unit :math:`j`'s predictor profile.
A subset :math:`\mathcal{E} \subseteq \mathcal{N}` of size
:math:`M = |\mathcal{E}|` is flagged as treatment-eligible (the
``candidate_col``); the design must pick :math:`m` of these.

A population weighting vector :math:`\mathbf{f} \in \mathbb{R}^N`,
:math:`\mathbf{f} \ge 0`, :math:`\mathbf{1}^\top \mathbf{f} = 1` (uniform
:math:`1/N` by default, or a ``weight_col`` such as population or revenue)
defines the estimand's target trajectory -- the :math:`\mathbf{f}`-weighted
average unit

.. math::

   \bar{\mathbf{x}} \coloneqq \mathbf{X} \mathbf{f},
   \qquad \bar{x}_t = \mathbf{X}_{t, \cdot}\, \mathbf{f} .

The pre-period predictor rows are split into an estimation window
:math:`E` (the first :math:`\lfloor \texttt{frac\_E} \cdot T_0 \rfloor`
time rows, *plus* all covariate rows) and a held-out blank window
:math:`B` (the remaining pre-period time rows). :math:`E` is where the
design is fit; :math:`B` is a pre-treatment "dress rehearsal" with no
treatment, used to validate fit and to calibrate power.

Over :math:`E` the predictors are row-standardized against the
population target:

.. math::

   \widetilde{X}_{t, j} = \frac{X_{t, j} - \bar{x}_t}{\sigma_t},
   \qquad
   \sigma_t = \operatorname{sd}_j\!\bigl(X_{t, \cdot}\bigr),

with :math:`\sigma_t` floored away from zero. Centring on
:math:`\bar{x}_t` puts the population target at the origin; scaling by the
cross-sectional spread :math:`\sigma_t` makes mixed-scale predictors
(outcomes in dollars, covariates in percent) commensurable so no single
predictor dominates the fit. The :math:`N \times N` Gram matrix

.. math::

   \mathbf{G} \coloneqq \widetilde{\mathbf{X}}_E^\top \widetilde{\mathbf{X}}_E
   \succeq 0

summarizes all pairwise predictor inner products over :math:`E`.

Identifying assumptions
^^^^^^^^^^^^^^^^^^^^^^^^

LEXSCM inherits the design-based identification of Abadie & Zhao
[ABADIE2024]_. The untreated potential outcomes are assumed to follow an
(approximate) linear factor model

.. math::

   y_{jt}^N = \delta_t + \theta_t^\top \mu_j + \varepsilon_{j, t},

with common time effects :math:`\delta_t`, latent time factors
:math:`\theta_t`, unit loadings :math:`\mu_j`, and mean-zero transitory
shocks :math:`\varepsilon_{j, t}`.

1. Approximate balance. There exist treatment weights
   :math:`\mathbf{w}` on the chosen treated set :math:`\mathcal{S}` and control
   weights :math:`\mathbf{v}` on the remainder such that the synthetic treated and
   synthetic control reproduce the population target on the pre-period
   predictors,
   :math:`\bar{\mathbf{x}} - \sum_{j \in \mathcal{S}} w_j \mathbf{x}_j \approx 0`.

   *Remark.* This is the design analogue of the SCM convex-hull condition, and it
   is the only substantive requirement. Crucially it is not imposed as an
   axiom: the *achieved* imbalance
   :math:`\lVert \bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j\rVert` is a
   measurable goodness-of-fit quantity, reported for every design, on which
   the validity of the bias bound and the inference is *conditional* (Abadie
   & Zhao, p.13). The analyst checks the :math:`\approx 0` condition rather
   than assuming it.

2. Factor structure controls bias. Under the linear factor
   model, matching the treated and synthetic-control trajectories on a long
   enough pre-period drives the latent loadings :math:`\mu_j` into alignment,
   so the design-based bias of the average treatment effect on the treated is
   bounded by the achieved imbalance and shrinks as :math:`T_0` grows and the
   fit tightens.

   *Remark.* This is why Stage 1 minimizes imbalance rather than
   any in-sample treatment contrast: imbalance is the quantity the bias bound
   is written in.

3. Placebo exchangeability / weak stationarity. On the blank
   window :math:`B` no treatment has occurred, so the treated-minus-control
   gap is pure noise; that gap process is weakly stationary and free of
   anticipation, so its serial-dependence structure carries over to the
   post-treatment window.

   *Remark.* This is what licenses the Stage 3 power
   analysis: the post-treatment null distribution of the test statistic is
   reconstructed by moving-block resampling the blank-window gaps, which
   preserves autocorrelation -- the time-series-robust inference of Abadie &
   Zhao rather than an i.i.d. permutation.

Stage 1 -- Treated-tuple selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a treated set :math:`\mathcal{S}` with :math:`|\mathcal{S}| = m` and simplex
weights :math:`\mathbf{w} \in \Delta(\mathcal{S}) \coloneqq \{\mathbf{w} \ge 0 :
\mathbf{1}^\top \mathbf{w} = 1\}`, the synthetic treated unit is
:math:`\sum_{j \in \mathcal{S}} w_j \mathbf{x}_j`. Its imbalance against the
population target is, over the estimation window,

.. math::

   L(\mathcal{S}) = \min_{\mathbf{w} \in \Delta(\mathcal{S})}
          \Bigl\lVert \sum_{j \in \mathcal{S}} w_j \widetilde{\mathbf{x}}_{E, j}
          \Bigr\rVert_2^2
        = \min_{\mathbf{w} \in \Delta(\mathcal{S})}
          \mathbf{w}^\top \mathbf{G}_{\mathcal{S}\mathcal{S}}\, \mathbf{w} ,

where :math:`\mathbf{G}_{\mathcal{S}\mathcal{S}}` is the :math:`m \times m`
sub-Gram matrix on the rows and columns of :math:`\mathcal{S}`. Because the
design is :math:`\mathbf{f}`-centred the population target sits at the origin,
so :math:`L(\mathcal{S})` is the squared distance from the population centroid
to the convex hull of the selected donors, and :math:`\sqrt{L(\mathcal{S})}`
is the achieved imbalance of Assumption 1. Stage 1 returns the ``top_K`` sets of
smallest :math:`L(\mathcal{S})`, subject to the budget below.

Weakly targeted designs (``targeting_penalty``)
"""""""""""""""""""""""""""""""""""""""""""""""

By default Stage 1 is *fully targeted*: the weights :math:`\mathbf{w}` are free
to contort onto the population mean (Abadie & Zhao's representative-experiment
goal, so the estimand is the population ATE :math:`\tau_t = \sum_j f_j(Y^I_{jt}
- Y^N_{jt})`). In practice you often treat the chosen markets as a group --
equal- or population-weighted -- not with bespoke fractional weights, and you
may prefer the treated group to look like *itself* rather than the population.
``targeting_penalty`` :math:`= \gamma \ge 0` adds an anchor toward the group's
own equal-weight aggregate,

.. math::

   \min_{\mathbf{w} \in \Delta(\mathcal{S})}\ \mathbf{w}^\top
   \mathbf{G}_{\mathcal{S}\mathcal{S}}\, \mathbf{w}
   \;+\; \gamma\,\bigl\lVert \mathbf{w} - \tfrac{1}{m}\mathbf{1} \bigr\rVert_2^2 .

On the simplex :math:`\mathbf{1}^\top\mathbf{w}=1`, so :math:`\lVert \mathbf{w}
- \tfrac1m\mathbf{1}\rVert_2^2 = \mathbf{w}^\top\mathbf{w} - \tfrac1m` and the
penalty is exactly a diagonal ridge, :math:`\min_{\mathbf{w}\in\Delta}
\mathbf{w}^\top(\mathbf{G}_{\mathcal{S}\mathcal{S}} + \gamma\mathbf{I})\mathbf{w}`.
The reported imbalance stays the *true* targeting distance
:math:`\sqrt{\mathbf{w}^\top\mathbf{G}_{\mathcal{S}\mathcal{S}}\mathbf{w}}` at the
penalized weights. :math:`\gamma = 0` (default) is the fully targeted design;
:math:`\gamma \to \infty` selects the best equal-weight :math:`m`-tuple; in
between is *weakly targeted*, sliding the estimand from the population ATE toward
the treated group's own ATT. This is also the mechanism that discourages
idiosyncratic treated sets: free weights can hit the population mean even for an
odd set, whereas the anchor favours sets that sit near the population naturally
(with near-equal weights), which the donor pool can also reconstruct.

How a single tuple is built: the inner simplex QP
"""""""""""""""""""""""""""""""""""""""""""""""""

"Building" a tuple :math:`\mathcal{S}` means solving its inner problem
:math:`\min_{\mathbf{w} \in \Delta(\mathcal{S})} \mathbf{w}^\top
\mathbf{G}_{\mathcal{S}\mathcal{S}}\, \mathbf{w}` for the optimal treatment
weights :math:`\mathbf{w}(\mathcal{S})`; the synthetic treated unit is then
:math:`\sum_{j \in \mathcal{S}} w_j(\mathcal{S})\, \mathbf{x}_j` and the design's
achieved imbalance is :math:`\sqrt{\mathbf{w}(\mathcal{S})^\top
\mathbf{G}_{\mathcal{S}\mathcal{S}}\, \mathbf{w}(\mathcal{S})}`. This convex
quadratic program over the probability simplex is solved by an Away-step
Frank-Wolfe (AFW)
routine in pure NumPy (``_afw_single``), chosen because every iterate stays
exactly on the simplex (no projection step) and the *away* move removes the
zig-zagging that plain Frank-Wolfe suffers near a face-constrained optimum
-- so the support of :math:`\mathbf{w}` (which donors carry positive weight)
sharpens in a handful of iterations.

Write :math:`\mathbf{Q} = \mathbf{G}_{\mathcal{S}\mathcal{S}}`,
:math:`f(\mathbf{w}) = \mathbf{w}^\top \mathbf{Q} \mathbf{w}`,
:math:`\nabla f(\mathbf{w}) = 2 \mathbf{Q} \mathbf{w}`. From a vertex start
:math:`\mathbf{w}^{(0)} = \mathbf{e}_{\operatorname*{argmin}_i Q_{ii}}` (the single donor
closest to the target), each iteration (all vectors below are the iterate
:math:`\mathbf{w}` and its simplex vertices :math:`\mathbf{e}_i`):

#. Pick two vertices. The Frank-Wolfe vertex
   :math:`s = \operatorname*{argmin}_i [\nabla f(\mathbf{w})]_i` and the away vertex
   :math:`a = \operatorname*{argmax}_{i \in \operatorname{supp}(\mathbf{w})}
   [\nabla f(\mathbf{w})]_i` -- the currently active donor the gradient most
   wants to shed.
#. Choose the direction. Compare the FW direction
   :math:`\mathbf{d}_{\mathrm{FW}} = \mathbf{e}_s - \mathbf{w}` against the away
   direction :math:`\mathbf{d}_{\mathrm{AW}} = \mathbf{w} - \mathbf{e}_a` by
   :math:`\langle \nabla f, \mathbf{d}\rangle` and take whichever descends more.
   The away step's maximal feasible length is
   :math:`\gamma_{\max} = w_a / (1 - w_a)`; the FW step caps at :math:`1`.
#. Exact line search. Because :math:`f` is quadratic the optimal step
   along :math:`\mathbf{d}` is closed-form,

   .. math::

      \gamma^\ast = \operatorname{clip}\!\Bigl(
        -\frac{\mathbf{w}^\top \mathbf{Q} \mathbf{d}}{\mathbf{d}^\top
        \mathbf{Q} \mathbf{d}},\; 0,\; \gamma_{\max}\Bigr),

   then :math:`\mathbf{w} \leftarrow \mathbf{w} + \gamma^\ast \mathbf{d}`,
   dropping :math:`a` from the active set when a full away step empties it.
#. Certified lower bound. The Frank-Wolfe gap gives a running lower
   bound :math:`f(\mathbf{w}) + \min_i[\nabla f(\mathbf{w})]_i -
   \nabla f(\mathbf{w})^\top \mathbf{w} \le f(\mathbf{w}^\ast)` on the tuple's
   true minimal loss; iteration stops once the duality gap
   :math:`\nabla f(\mathbf{w})^\top \mathbf{w} - \min_i[\nabla f(\mathbf{w})]_i`
   drops below ``tol``.

Two-pass precision. Scoring every candidate tuple to convergence would
be wasteful, so Stage 1 runs AFW at two fidelities. A vectorized
batched AFW (``_afw_batched``, ``iters=80``, all :math:`m \times m`
problems advanced together with ``einsum``) ranks thousands of tuples in
one sweep; only the surviving ``top_K`` are re-solved by the scalar
``_afw_single`` at ``iters=600``, ``tol=1e-14`` to pin
:math:`\mathbf{w}(\mathcal{S})` and the loss to full precision. Each returned
:class:`~mlsynth.utils.fast_scm_helpers.lexsearch.TreatedDesign` then
carries its ``indices``, the high-precision ``weights``
:math:`\mathbf{w}(\mathcal{S})`, ``loss`` :math:`= L(\mathcal{S})`,
``imbalance`` :math:`= \sqrt{L(\mathcal{S})}`, ``total_cost``, and a
label-keyed ``weight_dict``.

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
:math:`f`-weighted centroid of the candidate predictors, so it lies
*inside* the convex hull of the full candidate set. Any convex
(cardinality-free) relaxation of "distance from the origin to the hull of
a chosen subset" is therefore :math:`\approx 0` over the entire upper half
of the branch-and-bound tree: the relaxation cannot certify that a partial
selection is bad, so it cannot prune. An exact MIP would degenerate into
near-exhaustive enumeration with extra overhead.

Both failures are also *unnecessary*. Abadie & Zhao [ABADIE2024]_ require
only that the chosen design be feasible and approximately balanced;
validity is conditional on the *achieved* imbalance, a reported quantity,
not on a certificate of global optimality. LEXSCM therefore:

* Enumerates exactly when :math:`\binom{M}{m} \le \texttt{enumerate\_max}`
  (default :math:`3{,}000{,}000`). This is the gold standard -- it returns
  the certified global ``top_K`` and reports termination status
  ``OPTIMAL``.
* Runs a strengthened multi-start local search otherwise: greedy
  construction from diverse seeds, best-improvement swap descent to a local
  optimum, and basin-hopping *kicks* to escape it. In Monte Carlo this
  lands on the exact optimum 83-100% of the time and within roughly
  :math:`1\%` (mean) / :math:`7\%` (worst case) of the minimal imbalance --
  immaterial under the approximate-balance criterion. Termination status is
  ``FEASIBLE``.

Because the local search has no MIP optimality gap, LEXSCM reports a
consensus diagnostic in its place: the fraction of independent starts
that converged to the incumbent (``consensus_rate``), the number of
distinct local optima seen, and the incumbent-improvement trail. High
consensus across many random starts is the practical confidence signal
that the incumbent is the global optimum. (A convex hull lower bound is
*also* reported, but only as advisory information -- for the reason above
it is :math:`\approx 0` and is deliberately not turned into an
optimality gap.)

Building tuples in the heuristic regime
"""""""""""""""""""""""""""""""""""""""

When :math:`\binom{M}{m}` exceeds ``enumerate_max`` the same per-tuple QP is
reused, but the *set* of tuples scored is grown adaptively by a multi-start
local search (``_local_search``):

* Seeding. :math:`2 \times` ``n_starts`` seeds: the ``n_starts`` units
  with the smallest :math:`G_{jj}` (single donors already nearest the
  population centroid -- the cheapest places to start near balance) plus
  ``n_starts`` uniformly random units for diversity.
* Greedy construction. From a seed, repeatedly add the candidate that
  most lowers the batched loss of the partial tuple until
  :math:`|\mathcal{S}| = m`, skipping any addition that would breach the budget.
* Best-improvement descent. Score the full 1-swap neighbourhood
  (replace one member with one non-member), move to the steepest improving
  swap, and repeat to a local optimum.
* Basin-hopping kicks. Perturb the incumbent with a random 2-swap
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

   \sum_{j \in \mathcal{S}} c_j \le B_{\max}.

Two mechanisms enforce it. First, a sound feasibility presolve removes
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

For each candidate treated set :math:`\mathcal{S}` with weights
:math:`\mathbf{w}`, the synthetic treated trajectory is
:math:`\mathbf{X}_{\cdot, \mathcal{S}}\, \mathbf{w}`. A synthetic control is then
built from the non-treated units: control weights :math:`\mathbf{v}` solve a
ridge-penalized quadratic program (penalty ``lambda_penalty``) matching the
synthetic treated over :math:`E`, subject to an exclusion constraint
:math:`v_j = 0` for all :math:`j \in \mathcal{S}` (a treated unit cannot also be
its own control; the spillover ``adjacency`` widens this to
:math:`\mathcal{S} \cup \mathcal{A}(\mathcal{S})`). The per-period gap is

.. math::

   e_t = (\mathbf{X}_{\cdot, \mathcal{S}}\, \mathbf{w})_t
         - (\mathbf{X} \mathbf{v})_t .

Pre-treatment fit is summarized by the normalized mean-squared error
(NMSE) of the synthetic treated against the target on both windows
(``nmse_E``, ``nmse_B``). The blank-window gaps
:math:`e_B = \{e_t : t \in B\}` (``residuals_B``) are, under Assumption 3,
pure placebo noise; they are the raw material for the power analysis. This
stage is identical to the pre-rebuild control path -- only the search and
power stages around it changed.

Spillover / interference exclusions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When treating a *set* of units, interference is a design concern, and
Vives-i-Bastida (2022) handles it with two exclusion criteria that LEXSCM
implements directly. Both read off one conflict graph, encoded by a
symmetric matrix :math:`\mathbf{A} \in \{0, 1\}^{N \times N}` with
:math:`A_{ij} = 1` iff units :math:`i` and :math:`j` interfere (zero diagonal).
:math:`\mathbf{A}` is supplied either as a ``cluster_col`` (:math:`A_{ij} = 1`
iff :math:`i, j` share a cluster, e.g. a state or province) or as an
``adjacency`` / spillover matrix (:math:`A_{ij} = 1` iff the entry exceeds
``spillover_threshold``); the two combine entrywise by logical OR. The matrix is
aligned to the IndexSet, the single source of truth for unit identity. Write the
conflict-neighbours of a treated set :math:`\mathcal{S}` as

.. math::

   \mathcal{A}(\mathcal{S}) \coloneqq
     \{\, k : A_{jk} = 1 \ \text{for some}\ j \in \mathcal{S} \,\}.

* "No interference" (Stage 1, a treatment criterion). The treated set
  :math:`\mathcal{S} = \operatorname{supp}(\mathbf{w})` must be conflict-free
  -- an *independent set* of :math:`\mathbf{A}`, :math:`A_{ij} = 0` for all
  :math:`i, j \in \mathcal{S}`. This restricts only the admissible supports, so
  it enters exactly where the cardinality constraint
  :math:`\lVert \mathbf{w} \rVert_0 = m` already lives -- a filter on the
  candidate tuples, not a term in the inner weight program -- and only *shrinks*
  the search.

* "Exclusion restriction" (Stage 2, a control criterion). The donor pool
  drops :math:`\mathcal{S} \cup \mathcal{A}(\mathcal{S})`: the Stage-2 program
  pins :math:`v_k = 0` for every :math:`k \in \mathcal{S} \cup
  \mathcal{A}(\mathcal{S})`, so a treated unit's spillover neighbours cannot
  enter its synthetic control and contaminate the counterfactual.

If no conflict-free :math:`m`-tuple exists (e.g. :math:`m` exceeds the number of
clusters), or the exclusions empty every donor pool, the fit raises
:class:`~mlsynth.exceptions.MlsynthConfigError` rather than returning a degenerate
design. With no ``cluster_col`` / ``adjacency`` supplied, :math:`\mathbf{A} =
\mathbf{0}` and the behaviour is exactly the unconstrained search.

.. code-block:: python

   # at most one treated unit per state, and no treated unit's
   # same-state neighbours used as its donors
   LEXSCM({"df": df, "outcome": "y", "unitid": "unit", "time": "year",
           "candidate_col": "eligible", "m": 2, "cluster_col": "state"}).fit()

Coverage quotas and size bands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two further treatment criteria from the same design checklist restrict *which*
units may be treated -- both, like the spillover "no interference" rule, act on
the admissible supports and never enter the inner weight program.

* Coverage / stratification. Give each unit a stratum (region / tier /
  segment) via ``stratum_col``. ``min_per_stratum`` requires at least that many
  treated units from every stratum that has a candidate ("test in every
  region"); ``max_per_stratum`` allows at most that many ("a quota"). Formally
  the treated set :math:`\mathcal{S}` must satisfy
  :math:`\texttt{min} \le |\{j \in \mathcal{S} : g(j) = s\}| \le \texttt{max}`
  for each candidate stratum :math:`s`, where :math:`g(j)` is unit :math:`j`'s
  stratum. (Setting ``max_per_stratum = 1`` mirrors a ``cluster_col`` quota on
  the treated side, but unlike the spillover rule it does not also clean the
  donor pool.)

* Treated-unit size bands. With ``size_col`` and ``min_size`` / ``max_size``,
  only units whose size lies in :math:`[\texttt{min\_size}, \texttt{max\_size}]`
  are eligible for treatment (they remain available as donors). The floor is
  a power / operational minimum; the ceiling encodes the convex-hull limit -- a
  unit far larger than the rest cannot be reproduced by a convex combination of
  the others (Vives-i-Bastida excludes big cities for exactly this reason).

Either constraint raises :class:`~mlsynth.exceptions.MlsynthConfigError` when it
is infeasible (e.g. ``min_per_stratum`` over more strata than :math:`m`, or fewer
than :math:`m` candidates inside the size band).

.. code-block:: python

   # cover every region, at most two per region, and only mid-sized markets
   LEXSCM({"df": df, "outcome": "y", "unitid": "unit", "time": "year",
           "candidate_col": "eligible", "m": 4,
           "stratum_col": "region", "min_per_stratum": 1, "max_per_stratum": 2,
           "size_col": "population", "min_size": 50_000, "max_size": 2_000_000}).fit()

Forced-in and forbidden markets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two hard market lists complete the treatment-criteria vocabulary, matching SYNDES
and GeoLift. ``to_be_treated`` pins specified markets into the treated set: every
candidate :math:`m`-tuple the search considers must contain them, which is useful
when a market is decided in advance (a client insists on testing it) and the
design only has to choose the rest around it. ``not_to_be_treated`` does the
opposite -- those markets are removed from the treatment pool but stay available
as donors -- for a market already running another campaign, with poor data, or
earmarked as a clean control. A forced-in market must be a treatment candidate
(``candidate_col`` true and within any size band), at most :math:`m` may be
forced, and a market cannot appear in both lists; each violation raises
:class:`~mlsynth.exceptions.MlsynthConfigError`. Forcing in a market that is hard
to synthesize costs imbalance, so check the achieved fit.

.. code-block:: python

   # always test DMA 501; never test DMA 635 (it stays an eligible donor)
   LEXSCM({"df": df, "outcome": "y", "unitid": "unit", "time": "year",
           "candidate_col": "eligible", "m": 4,
           "to_be_treated": [501], "not_to_be_treated": [635]}).fit()

Stage 3 -- Power analysis (minimum detectable effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 3 asks: *how large must a sustained treatment effect be for this
design to detect it?* The answer is a minimum-detectable-effect (MDE)
curve over post-treatment horizon lengths, computed entirely from the
Stage 2 placebo residuals with one consistent resampling model for both the
null and the alternative.

What "at each horizon" means -- and what it does not
""""""""""""""""""""""""""""""""""""""""""""""""""""

The curve is indexed by the length :math:`h` of the post-treatment
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
gaps. The scale :math:`\sigma` is the standard deviation of the pooled
residuals (all series concatenated, ``ddof=1``, floored at
:math:`10^{-12}`). The block length defaults to

.. math::

   \ell = \max\!\bigl(1,\ \min(h,\ \operatorname{round}(\tilde L^{1/3}))\bigr),

where :math:`\tilde L` is the median series length: floored at 1, and
capped at the horizon :math:`h` so a block never exceeds the window it
fills. To draw one length-:math:`h` window
(:func:`~mlsynth.utils.fast_scm_helpers.lexpower.block_resample_windows`):
pick a placebo series uniformly at random, then repeatedly cut contiguous
blocks of length :math:`\ell` from a random offset with wraparound
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
null distribution; its :math:`(1 - \alpha)` empirical quantile is the
critical value :math:`c_\alpha`.

Power, the effect grid, and the MDE
"""""""""""""""""""""""""""""""""""

A treatment effect is modelled as a constant additive shift :math:`\tau`
applied to *every* point of a resampled window -- the homogeneous-effect
working assumption behind the MDE. Its power is estimated from a fresh
draw of :math:`n_\text{power}` (default 2000) windows (same residual model,
new randomness), so the null and the alternative differ only by the shift:

.. math::

   \operatorname{power}(\tau)
     = \Pr\bigl[\, S(e + \tau) \ge c_\alpha \,\bigr]
     \approx \frac{1}{n_\text{power}}
       \sum_{b} \mathbf{1}\!\bigl\{ S(e^{(b)} + \tau) \ge c_\alpha \bigr\}.

The effect is swept on a grid of :math:`n_\text{grid} = 64` points in
standard-deviation units, :math:`\tau / \sigma \in [0, \texttt{max\_sd}]`
(default cap :math:`8`). The search walks the grid until
:math:`\operatorname{power}(\tau)` first reaches ``power_target``, then
linearly interpolates between the last sub-threshold point
:math:`(g_0, p_0)` and the crossing point :math:`(g, p)` for a finer value:

.. math::

   \frac{\mathrm{MDE}(h)}{\sigma}
     = g_0 + (\texttt{power\_target} - p_0)\,\frac{g - g_0}{p - p_0} .

If the grid is exhausted without reaching the target, the horizon is
infeasible and returns :math:`\infty`.

Three reported scales
"""""""""""""""""""""

Each horizon reports the MDE on three scales:

* ``mde_sd`` :math:`= \mathrm{MDE}(h)/\sigma` -- effect-size units, the
  primary scale (matching Vives-i-Bastida's "detect effects larger than
  :math:`0.1` s.d." convention) and numerically robust: nothing is divided
  by a fragile mean, so zero-mean or near-zero outcomes cannot blow the
  calculation up;
* ``mde_abs`` :math:`= \mathrm{MDE}(h)` -- outcome units;
* ``mde_pct`` -- the manager-facing percentage,
  :math:`100 \cdot \mathrm{MDE}(h) / \lvert\text{baseline}\rvert`, where the
  baseline is the counterfactual level over the *matching* window,
  :math:`\text{baseline} = \operatorname{mean}\bigl(\texttt{synthetic\_treated}
  [-h:]\bigr)`. This is guarded: when
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

* ``"late"`` (default) -- the MDE at the longest horizon; the
  conservative choice for a sustained-exposure experiment.
* ``"early_min"`` -- the smallest MDE across feasible horizons (most
  optimistic detectability).
* ``"early_mean"`` -- the mean MDE across feasible horizons (the
  percentage averaged only over horizons where it is defined).

Stage 4 -- Final recommendation (lexicographic selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final stage turns the per-design metrics -- imbalance (validity),
``mde_sd`` (power), ``stability`` :math:`=` out-of-sample
:math:`\text{NMSE}_B` (fit robustness), and ``total_cost`` -- into one
recommendation, applying the method's priorities in strict order:

#. Validity gate. Keep only designs whose imbalance is within a
   relative slack of the best achievable balance,

   .. math::

      \text{imbalance}(S) \le
      \bigl(1 + \texttt{imbalance\_tol}\bigr)\cdot
      \min_{S'} \text{imbalance}(S') ,

   (default ``imbalance_tol`` :math:`= 0.25`). This is the set of designs
   that satisfy Assumption 1 well enough to be trustworthy. (A degenerate
   tolerance that would empty the gate falls back to the single
   best-balanced design.)

#. Power. Among gated designs with a *feasible* MDE, choose the
   smallest ``mde_sd`` -- the most detectable valid design.

#. Tie-breaks. Break ties by better out-of-sample stability
   (:math:`\text{NMSE}_B`), then by lower ``total_cost``.

The recommendation is returned as a tuple of fields:
``winner`` (the chosen design), ``shortlist`` (the top
``max_shortlist`` ranked designs), ``pareto_ids`` (the Pareto frontier on
imbalance :math:`\downarrow` vs. ``mde_sd`` :math:`\downarrow`, always
exposed for transparency), a ``status``, a human-readable
``explanation``, and a per-design ``table`` (which becomes
``results.search.shortlist``). The status is one of:

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
rows, :math:`N = 5` units, treat :math:`m = 2`). Every number below is the
actual helper output, not an illustration.

Building the tuple (Stage 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After :math:`f`-centring and row-standardising the predictors, the Gram
matrix's diagonal -- each unit's squared distance to the population centroid
-- is

.. math::

   \operatorname{diag}(\mathbf{G}) = (5.77,\ 3.64,\ 6.08,\ 6.27,\ 8.24).

Unit 1 (0-indexed) sits closest to the target, so it seeds the AFW vertex
start. With :math:`\binom{5}{2} = 10` the search enumerates exactly
(status ``OPTIMAL``, 10 subsets scored). The winning tuple is
:math:`\mathcal{S} = \{0, 1\}`, and its inner simplex QP returns treatment weights

.. math::

   \mathbf{w}(\mathcal{S}) = (0.4098,\ 0.5902), \qquad
   L(\mathcal{S}) = \mathbf{w}^\top \mathbf{G}_{\mathcal{S}\mathcal{S}}\, \mathbf{w} = 1.6481, \qquad
   \sqrt{L(\mathcal{S})} = 1.2838 .

The standalone high-precision re-solve reproduces this loss with a
Frank-Wolfe lower bound equal to it to :math:`10^{-6}` -- the QP is at its
certified optimum. The synthetic treated unit is
:math:`0.41\,\mathbf{x}_0 + 0.59\,\mathbf{x}_1`, and :math:`1.2838` is the achieved imbalance
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

LEXSCM's Stage 3 already produces the lexicographic MDE curve used to rank
designs against each other; the standardized post-fit attached to
``LEXSCMResults.post_fit`` is the complementary surface used once a design
has been chosen -- it is the same
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` object SYNDES,
MAREX, and PANGEO expose, so downstream consumers (dashboards, paper-style
reports, comparison tables) read identical fields across the family.

.. code-block:: python

   pf = res.report.additional_outputs["post_fit"]   # SyntheticControlPostFit
   pf.ate, pf.ate_percent, pf.total_effect    # treatment-effect scalars
   pf.rmse_fit, pf.rmse_blank, pf.rmse_post   # pre / blank / post fit quality
   pf.p_value, pf.ci_lower, pf.ci_upper       # conformal inference
   pf.power                                   # PowerAnalysis (see below)

The trajectories ``pf.treated_series`` and ``pf.control_series`` are the
winning candidate's ``predictions.synthetic_treated`` and
``predictions.synthetic_control`` -- per-unit weighted aggregates of the
treated and control donors over the full timeline. The phase boundaries
``(n_fit, n_blank, n_post)`` line up with the same E / B / post split LEXSCM
uses internally (so ``n_fit + n_blank = T0`` and the 30%-of-pre-tail
holdout convention applies via ``frac_E = 0.7``).

Where the unified power analysis fits in
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stage 3's :func:`~mlsynth.utils.fast_scm_helpers.lexpower.detectability_curve`
is the design-selection workhorse -- it converts the moving-block placebo
null on the B window into a per-horizon MDE used by Stage 4's lexicographic
selector. The post-fit power analysis is the post-selection companion:
analytical Gaussian + AR(1) variance inflation on the realised gap residuals,
matching the MAREX / SYNDES / PANGEO power surfaces exactly so the same
diagnostic table can be produced for every family member.

.. math::

   \mathrm{MDE}(T) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr) \cdot
       \widehat{\sigma}_{\text{placebo}} \cdot \sqrt{\mathrm{VIF}(T, \widehat{\rho})},

with :math:`\widehat{\sigma}_{\text{placebo}}` the SD of the gap on the B (blank /
holdout) window, :math:`\widehat{\rho}` the lag-1 autocorrelation of those
residuals clipped to :math:`(-0.99, 0.99)`, and
:math:`\mathrm{VIF}(T, \rho) = \tfrac{1}{T}\bigl(1 + 2\sum_{k=1}^{T-1}
(1-k/T)\rho^k\bigr)` the standard AR(1) variance-inflation factor (textbook
:math:`1/T` when :math:`\rho = 0`). See :doc:`marex` for the full derivation;
the same module powers all three estimators.

.. code-block:: python

   p = res.power                               # PowerAnalysis (single source)
   p.headline.mde_absolute                     # MDE at the realised T_post
   p.headline.mde_pct                          # ... as % of post-period baseline
   p.headline.power_at_observed                # power to detect res.report.att
   p.curve                                     # tuple of MDEPoint per horizon
   p.sigma_placebo                             # σ̂ used (B window in LEXSCM)
   p.serial_correlation                        # ρ̂ AR(1) of the B residuals

Two MDEs, complementary roles
"""""""""""""""""""""""""""""

* Stage 3 MDE (``best_candidate.mde_results``) -- moving-block placebo
  null on the B window, used to *rank designs against each other*. Aggregated
  to a representative scalar by ``mde_horizon`` (``late`` / ``early_min`` /
  ``early_mean``) and consumed by Stage 4's lexicographic gate.
* Post-fit MDE (``res.power``) -- analytical Gaussian +
  AR(1) MDE consumed *after* a design has been chosen, on the same surface
  that MAREX / SYNDES / PANGEO produce. Use this when reporting a single
  detectability number alongside the realised ATE / CI.

Power-analysis failures (e.g. degenerate B-window residuals) never break a
fit; ``res.power`` is simply left as ``None``. To compute on a
non-default horizon grid or significance level call
:func:`~mlsynth.utils.post_fit.compute_power_analysis` directly.

Verification
------------

LEXSCM is validated against the synthetic-experimental-design framework it
implements (Abadie & Zhao 2026; Vives-i-Bastida 2022). Path A: the Walmart
45-store placebo design tracks pre-period to ~2.7% of mean sales and yields a
placebo effect of ~0.9% whose permutation test fails to reject (CI covers zero)
-- the paper's "no spurious effect" result. Path B: on the paper's exact
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
before any intervention, so it returns a design that resolves to an effect
report. The surface is small and grouped -- one obvious home for each thing.

Front door (the standardized contract):

* ``res.report`` -- the realized effect as an
  :class:`~mlsynth.config_models.EffectResult`: the *single source* for the ATT /
  CI / pre-fit (flat ``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` plus the
  standard sub-models). Its ``additional_outputs['post_fit']`` carries the full
  :class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` -- the MAREX-family
  shared bundle with per-period effects and covariate-balance SMDs --
  alongside ``ate_per_period`` and ``cumulative_effect``.
* ``res.power`` -- the design's MDE / power analysis (the *single source* for
  power; ``None`` if power analysis failed).
* ``res.selected_units`` / ``res.assignment`` / ``res.design_weights`` -- the
  chosen design (the treated units, the treated/control split, and the implied
  :class:`~mlsynth.config_models.WeightsResults`).
* ``res.metadata`` -- the lexicographic recommendation diagnostics.

Grouped detail:

* ``res.search`` (:class:`~mlsynth.utils.fast_scm_helpers.structure.LEXSCMSearch`)
  -- *how* the design was chosen: ``shortlist`` (the ranked table),
  ``candidates`` (every evaluated design), ``winner`` (the chosen
  :class:`~mlsynth.utils.fast_scm_helpers.structure.SEDCandidate` with its
  weights / predictions / losses / raw MDE curve), and ``selection`` (the
  Stage-1 treated-design selection diagnostics).
* ``res.panel`` (:class:`~mlsynth.utils.fast_scm_helpers.structure.LEXSCMPanel`)
  -- the panel structure: ``time``, ``units``, ``outcome``, ``population_mean``.

.. autoclass:: mlsynth.utils.fast_scm_helpers.structure.LEXSCMResults
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mlsynth.utils.fast_scm_helpers.structure.LEXSCMSearch
   :members:
   :undoc-members:

.. autoclass:: mlsynth.utils.fast_scm_helpers.structure.LEXSCMPanel
   :members:
   :undoc-members:

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
populate the post-fit bundle live outside this package so SYNDES, MAREX,
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
    stats = results.search.selection["stats"]
    print("status        :", stats["termination"]["status"])      # OPTIMAL / FEASIBLE
    print("method        :", stats["search"]["method"])
    print("subsets scored :", stats["search"]["subsets_evaluated"])
    print("C(M,m)        :", stats["problem"]["feasible_region_C(M,m)"])
    print("best imbalance :", round(stats["incumbent"]["imbalance"], 4))

    # --- Stage 4: lexicographic recommendation tuple ---
    rec = results.search.selection["recommendation"]
    print("rec status    :", rec["status"])      # OK / POWER_NOT_ESTABLISHED
    print("winner        :", rec["winner"])
    print("pareto ids    :", rec["pareto_ids"])
    print(rec["explanation"])

    # --- the recommended design ---
    print(results.search.shortlist)               # ranked per-design table
    best = results.search.winner
    print("treated weights:", best.treated_weight_dict)
    print("control weights:", best.control_weight_dict)

With :math:`M = 40` eligible markets and :math:`m = 3`,
:math:`\binom{40}{3} = 9{,}880` is well under ``enumerate_max``, so the
search enumerates exactly and reports ``status = OPTIMAL`` -- a
certified global ``top_K``. Raising :math:`M` and :math:`m` until
:math:`\binom{M}{m}` exceeds the cap flips the search to the multi-start
local solver (``status = FEASIBLE``), at which point the ``consensus``
block under ``stats["search"]`` reports how many independent starts agreed
on the incumbent.

Example: spillover-aware selection
==================================

Suppose the markets sit inside larger regions and treating two markets in the
same region would let the intervention spill across them (and onto each other's
donors). Assign every unit a ``region`` and pass it as ``cluster_col``: LEXSCM
then refuses to co-treat two markets from the same region (Stage 1) and
refuses to build a treated market's synthetic control from its own-region peers
(Stage 2).

.. code-block:: python

    import numpy as np

    df = generate_synthetic_sales_panel(n_units=120, n_candidates=40,
                                        n_time_periods=100, treatment_start=90,
                                        seed=4545)

    # assign each unit to one of 8 regions (constant within unit)
    rng = np.random.default_rng(0)
    region_of = rng.integers(0, 8, size=df["unitid"].nunique())
    df["region"] = df["unitid"].map(lambda u: f"R{region_of[u]}")

    config = {
        "df": df, "outcome": "sales", "unitid": "unitid", "time": "time",
        "candidate_col": "candidate", "m": 3, "post_col": "post",
        "cluster_col": "region",        # <-- the only new line vs the budget example
        "mde_horizon": "late", "top_K": 20,
    }
    results = LEXSCM(config).fit()

    treated = list(results.selected_units)
    treated_regions = {df.set_index("unitid")["region"].loc[int(u)] for u in treated}
    donor_regions = {df.set_index("unitid")["region"].loc[int(d)]
                     for d in results.design_weights.donor_weights}

    print("treated markets :", treated)
    print("treated regions :", treated_regions)       # all distinct -- one per region
    print("donor regions   :", donor_regions)
    assert treated_regions.isdisjoint(donor_regions)   # donors avoid treated regions

Instead of clusters you can hand LEXSCM a spillover/adjacency matrix -- a
``pandas.DataFrame`` indexed and columned by unit id, with a positive entry
wherever two units interfere -- via ``adjacency=...`` and an optional
``spillover_threshold`` (units conflict when the entry exceeds it). A
``cluster_col`` and an ``adjacency`` may be supplied together; their conflicts
combine. If the constraint admits no conflict-free ``m``-tuple (for instance
``m`` larger than the number of regions), the fit raises
:class:`~mlsynth.exceptions.MlsynthConfigError`.

A ready-made matrix ships with mlsynth -- you don't need a shapefile or GIS
toolchain. ``basedata/markets/dma_adjacency.csv`` is a 206 x 206 border matrix
over US Nielsen Designated Market Areas (built from the public TopoJSON via
shared-arc contiguity; see ``basedata/markets/build_dma_adjacency.py``), so
"no two treated markets may share a border" is one ``read_csv`` away:

.. code-block:: python

    import pandas as pd

    adj = pd.read_csv("basedata/markets/dma_adjacency.csv", index_col=0)
    meta = pd.read_csv("basedata/markets/dma_metadata.csv")          # state, lat, long
    se = [n for n in meta.loc[meta.state.isin({"FL","GA","AL","MS","SC","NC","TN"}),
                              "dma_name"] if n in adj.index]
    A = adj.loc[se, se]                                              # induced subgraph

    LEXSCM({"df": df, "outcome": "sales", "unitid": "dma", "time": "week",
            "candidate_col": "eligible", "m": 4, "adjacency": A}).fit()

On a spatially-correlated Southeast panel this is the difference between an
unconstrained design that co-treats a cluster of bordering Gulf-Coast markets
(Hattiesburg, Jackson, Mobile-Pensacola, Montgomery -- with Biloxi-Gulfport and
Dothan as *donors*, both bordering treated markets) and the spillover-aware
design, which spreads treatment across non-adjacent markets and keeps every
donor clear of a treated market.

Example: South & Midwest design with grouped factors
====================================================

This example exercises every selection constraint at once on real geography. We
take the South and Midwest DMAs from the bundled map, group them by census
division (South Atlantic, East/West South Central, East/West North Central),
and draw outcomes from a grouped linear factor model (after the grouped factor
structure of Liao, Shi & Zheng [RelaxSC]_): markets in the same division share
factor loadings, so they co-move -- the latent-group structure the
constraints are designed for. Each division is both a *stratum* (for coverage /
quota) and, through the DMA borders, a source of *spillover*.

Build the panel
---------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mlsynth import LEXSCM

    DIVISION = {  # census divisions across the South + Midwest
        "FL": "S. Atlantic", "GA": "S. Atlantic", "NC": "S. Atlantic",
        "SC": "S. Atlantic", "VA": "S. Atlantic", "WV": "S. Atlantic",
        "MD": "S. Atlantic",
        "KY": "E.S. Central", "TN": "E.S. Central", "MS": "E.S. Central",
        "AL": "E.S. Central",
        "AR": "W.S. Central", "LA": "W.S. Central", "OK": "W.S. Central",
        "TX": "W.S. Central",
        "OH": "E.N. Central", "IN": "E.N. Central", "IL": "E.N. Central",
        "MI": "E.N. Central", "WI": "E.N. Central",
        "MN": "W.N. Central", "IA": "W.N. Central", "MO": "W.N. Central",
        "ND": "W.N. Central", "SD": "W.N. Central", "NE": "W.N. Central",
        "KS": "W.N. Central",
    }

    meta = pd.read_csv("basedata/markets/dma_metadata.csv")
    adj = pd.read_csv("basedata/markets/dma_adjacency.csv", index_col=0)
    meta = meta[meta.state.isin(DIVISION)].copy()
    meta["division"] = meta.state.map(DIVISION)
    names = [n for n in meta.dma_name if n in adj.index]          # 145 DMAs
    meta = meta[meta.dma_name.isin(names)].reset_index(drop=True)
    A = adj.loc[names, names]                                     # border matrix

    # --- grouped linear factor model: same division -> shared loadings ---
    rng = np.random.default_rng(11)
    n, T, T_post, r = len(names), 60, 12, 4
    div = meta.set_index("dma_name").loc[names, "division"].values
    Lam_div = {d: rng.normal(size=r) for d in sorted(set(div))}
    Lam = np.array([Lam_div[d] for d in div]) + 0.15 * rng.normal(size=(n, r))
    F = np.cumsum(rng.normal(size=(T, r)), 0)                     # latent factors
    population = np.round(rng.lognormal(12.5, 0.8, n)).astype(int)
    cost = population * 5.0                       # $5/person intervention cost
    Y = 100 + rng.normal(0, 5, n) + F @ Lam.T + rng.normal(0, 1.0, (T, n))
    cand = sorted(rng.choice(n, 18, replace=False).tolist())     # eligible markets

    df = pd.DataFrame([
        {"market": names[j], "week": t, "sales": Y[t, j],
         "eligible": int(j in cand), "post": int(t >= T - T_post),
         "division": div[j], "population": int(population[j]),
         "cost": float(cost[j])}
        for j in range(n) for t in range(T)
    ])

    base = dict(df=df, outcome="sales", unitid="market", time="week",
                candidate_col="eligible", post_col="post", top_K=8, verbose=False)

Each constraint, one at a time
------------------------------

.. code-block:: python

    # 1. Spillover: no two treated markets share a border, and no donor borders a
    #    treated market (the DMA adjacency matrix).
    LEXSCM({**base, "m": 3, "adjacency": A}).fit()

    # 2. Coverage: at least one treated market in EVERY census division
    #    (m must be >= the 5 divisions).
    LEXSCM({**base, "m": 5, "stratum_col": "division", "min_per_stratum": 1}).fit()

    # 3. Quota: at most one treated market per division (a spread-out design).
    LEXSCM({**base, "m": 4, "stratum_col": "division", "max_per_stratum": 1}).fit()

    # 4. Size band: only treat markets at or above the median population.
    LEXSCM({**base, "m": 3, "size_col": "population",
            "min_size": int(np.median(population))}).fit()

    # 5. Compose constraints: cover all five divisions (min 1) AND at most two
    #    per division (max 2), at m=6 -- exactly one division carries two.
    LEXSCM({**base, "m": 6, "stratum_col": "division",
            "min_per_stratum": 1, "max_per_stratum": 2}).fit()

    # 6. With a size band on top, a division may lose all its eligible markets;
    #    coverage then only requires the divisions that still HAVE a large enough
    #    market (here East-South-Central drops out, so four divisions are covered).
    LEXSCM({**base, "m": 4, "stratum_col": "division", "min_per_stratum": 1,
            "size_col": "population", "min_size": int(np.median(population))}).fit()

With the grouped factors, an unconstrained design tends to concentrate treatment
in whichever division is easiest to balance; the coverage and quota constraints
force it to spread across divisions (so the experiment speaks to the whole
footprint), the adjacency rule keeps treated markets from contaminating each
other, and the size band drops markets too small to power -- or too large to
reproduce from the rest. All compose on the same Stage-1 admissible-tuple layer,
and an impossible combination (for example ``min_per_stratum=1`` with ``m`` below
the division count) raises :class:`~mlsynth.exceptions.MlsynthConfigError`.

Under a budget
--------------

Experiments cost money, and the cost is largely driven by market size -- a
bigger DMA means a bigger population to treat. Here ``cost`` is :math:`\$5` per
person, and the program carries a hard :math:`\$10\text{M}` knapsack
(``unit_cost_col`` + ``budget``) that composes with everything above.

.. code-block:: python

    BUDGET = 10_000_000
    bud = {**base, "unit_cost_col": "cost", "budget": BUDGET}

    # budget alone: the most balanced affordable design (here ~$6.9M of $10M)
    LEXSCM({**bud, "m": 4}).fit()

    # budget + coverage: one market per division AND total cost <= $10M -- the
    # knapsack pushes the design toward the *cheaper* (smaller) market in each
    # division (here ~$8.3M, all five divisions covered)
    LEXSCM({**bud, "m": 5, "stratum_col": "division", "min_per_stratum": 1}).fit()

The constraints can genuinely conflict: requiring coverage of all five
divisions *and* only above-median markets *and* a $10M cap asks for five large
(expensive) markets that the budget cannot afford, so the fit fails loudly
rather than silently dropping a criterion.

.. code-block:: python

    from mlsynth.exceptions import MlsynthConfigError

    try:
        LEXSCM({**bud, "m": 5, "stratum_col": "division", "min_per_stratum": 1,
                "size_col": "population",
                "min_size": int(np.median(population))}).fit()
    except MlsynthConfigError as e:
        print(e)

prints the binding constraint and by how much, not just "infeasible"::

    LEXSCM design is infeasible -- the binding constraint(s):
      - budget: the 5 cheapest eligible markets cost $10,100,765, over the
        $10,000,000 budget by $100,765. Raise the budget to >= $10,100,765,
        reduce m, or relax the size band.

Every infeasibility -- candidate pool, budget, coverage, quota, or spillover --
is audited up front and reported in this same ``have vs need -> minimal fix``
shape, and all binding constraints are listed together (so you fix them in
one pass rather than one error at a time). The audit only *reports*: it never
silently relaxes a constraint you set. All of them raise the one
:class:`~mlsynth.exceptions.MlsynthConfigError`, so a caller catches a single
type and surfaces exactly which design ask was impossible and the smallest change
that would satisfy it.

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

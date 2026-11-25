Parallel-Trends Supergeo Design (PANGEO)
=========================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

PANGEO is a tool for designing a geo experiment — deciding, *before*
you run it, which geographic markets to treat and which to hold out, so
that the after-the-fact comparison is as clean as possible. Use it when:

- you can assign treatment at the geo level (turn an ad campaign,
  price, or feature on in some markets and not others);
- you have a panel of pre-period history (weekly/monthly sales,
  conversions, GMV…) for every candidate geo;
- the number of geos is modest (tens, not thousands), as is typical
  with DMAs/regions; and
- you want the eventual treatment-effect estimate to be precise —
  i.e. you want treated and control markets that already move together.

A worked geo-experiment. A brand wants to measure the incremental
sales from a new TV/CTV campaign. Ads are bought at the DMA level, so the
experimental units are ~50–210 large, heterogeneous markets — not
exchangeable shoppers. The plan: run the campaign in a *treatment* set of
DMAs, withhold it in a *control* set, and read the post-launch sales gap
as the lift. The whole experiment lives or dies on the split: if the
treated DMAs were already trending differently from the controls, the
post-launch gap mixes the campaign effect with that pre-existing
divergence. PANGEO reads the DMAs' pre-period sales panel and chooses the
treatment/control split (bundling DMAs into balanced *supergeos*) so the
two sides' sales trajectories run parallel beforehand — turning the
post-period gap into a clean read on the campaign.

PANGEO is two stages:

#. Design (pre-period data only). Each arm's geos are grouped into
   composite *supergeos* and formed into balanced pairs --- with no geo
   trimmed --- so that pre-period parallelism is maximised. By default this
   partition is found by a fast clustering heuristic (the OSD analogue of
   Shaw 2025); an exact set-partitioning mixed-integer program is available
   as an opt-out (``fast=False``). Both optimise the same parallelism
   objective. A power analysis reports the minimum detectable effect (MDE)
   implied by the chosen supergeo size :math:`Q`.

#. Evaluation (after the experiment). The *same* design is scored
   against the realised outcomes with the Augmented
   Difference-in-Differences estimator of Li & Van den Bulte (2022),
   giving the ATT, percent ATT, and CIs at the arm and program levels.

The two stages share one quantity: both the design objective and the
standard error of the realised effect are governed by the variance of the
supergeo *gap* residual. Minimising non-parallelism simultaneously
minimises the MDE and tightens the CI — optimising parallelism is
optimising inferential precision.

This is a principled deviation from Google's Supergeo Design. Chen et
al. (2023) and OSD (Shaw 2025) match supergeos on a scalar summary —
the summed baseline response, or a few covariate totals — which collapses
the time dimension. PANGEO matches on the full pre-period trajectory.
That difference is not cosmetic: the downstream analysis is a
difference-in-differences, which *differences trajectories over time*, so
two markets with identical totals but different seasonal shapes are not
interchangeable for it even though scalar matching scores them as a
perfect match. In trending, seasonal data — which is essentially all
geo-marketing data — matching on shape, not on a single number is
what makes the post-period comparison valid. (The simulation at the end
of this page quantifies the gap: when geos share a baseline mean but
differ in shape, PANGEO recovers the effect ~30× more precisely than a
scalar match.)

When *not* to use it. If the assignment is already fixed (an
observational study, or a campaign that already ran in specific markets),
there is no design to choose — use an estimation-stage method
(:doc:`tssc`, :doc:`sbc`, :doc:`fdid`). Large geo pools (hundreds of
markets) are handled by the default clustering partition (see *Forming the
supergeos* below); only the exact opt-out program is scale-limited. And if
the outcome is plausibly stationary with no trend or seasonality, scalar
matching is already adequate and simpler.

What is a supergeo?
^^^^^^^^^^^^^^^^^^^

Geo experiments differ from ordinary A/B tests in one decisive way: the
experimental units are a *small* number of *large, heterogeneous* aggregates
--- markets, regions, DMAs ---, not many exchangeable individuals.
Randomising treatment across a handful of dissimilar markets routinely
produces treatment and control groups with very different baseline
characteristics, and the resulting post-randomisation bias does not average
away over the single assignment a practitioner actually runs (Abadie & Zhao
2026). Classic matched-pair designs help, but with heterogeneous geos there
may be *no* good one-to-one match for a given market.

A supergeo resolves this by relaxing the unit of matching. Instead of
insisting that single geos match, geos are pooled into composite aggregates:
a supergeo is simply a bundle of geos treated as one unit, with outcome equal
to their (population-weighted) mean. Composite units can be made comparable
even when their constituents are not --- a small, noisy market combined with
a complementary one can, in aggregate, track another composite closely. The
design then pairs *supergeos*, randomises treatment within each pair, and ---
unlike trimming-based approaches --- assigns every geo to some supergeo,
so the experiment spans the entire market with nothing discarded (Chen,
Doudchenko, Jiang, Stein & Ying 2023).

PANGEO keeps this structure but changes *what* the supergeos are matched on.
Supergeo Design and OSD match on a scalar summary (the summed response, or a
few covariate totals), which collapses the time dimension; PANGEO matches on
the full pre-treatment trajectory, choosing pairs whose aggregate paths
run as parallel as possible. The reason is that the downstream
difference-in-differences analysis differences trajectories: two markets with
identical totals but different seasonal shapes are not interchangeable for
it, even though scalar matching treats them as equivalent.

Setup and notation
------------------

Let :math:`y_{it}` denote the outcome of geo
:math:`i \in \mathcal{N} \coloneqq \{1,\dots,N\}` in period
:math:`t \in \mathcal{T} \coloneqq \{1,\dots,T\}`. The first :math:`T_0`
periods are the pre-treatment (design) window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` and the
remaining :math:`|\mathcal{T}_2| = T - T_0` periods are the experimental window
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`.
A single categorical column assigns each geo to an arm; arms occupy
disjoint geo pools :math:`\mathcal{N}_a` and are designed independently, so
the exposition below fixes one arm and drops the arm subscript.

Throughout we maintain the linear factor model used by both the synthetic-
control and DiD literatures (Abadie, Diamond & Hainmueller 2010; Li &
Van den Bulte 2022) for the no-treatment potential outcome,

.. math::
   :label: factor

   y_{it}^{N} = \delta_t + \boldsymbol{\theta}_t^{\top} \mathbf{z}_i
                + \boldsymbol{\lambda}_t^{\top}\boldsymbol{\mu}_i
                + \varepsilon_{it},

where :math:`\delta_t` is a common time effect, :math:`\mathbf{z}_i` are
observed covariates with time-varying loadings :math:`\boldsymbol{\theta}_t`,
:math:`\boldsymbol{\mu}_i` are unobserved factor loadings with factors
:math:`\boldsymbol{\lambda}_t`, and :math:`\varepsilon_{it}` is mean-zero
idiosyncratic noise.

A supergeo is a set :math:`\mathcal{S}` of same-arm geos with aggregate
trajectory

.. math::

   \bar y_{\mathcal{S},t}
   \coloneqq \frac{\sum_{i\in \mathcal{S}}\omega_i\,y_{it}}{\sum_{i\in \mathcal{S}}\omega_i},

where :math:`\omega_i>0` are aggregation weights (the ``weight_col``
population, or :math:`\omega_i\equiv 1`). A pair
:math:`p=(\mathcal{A}_p,\mathcal{B}_p)` consists of two disjoint supergeos with
:math:`|\mathcal{A}_p|,|\mathcal{B}_p|\le Q`; :math:`\mathcal{A}_p` is the
treatment half and :math:`\mathcal{B}_p` the control half. Its gap is

.. math::
   :label: gap

   g_{p,t} \coloneqq \bar y_{\mathcal{A}_p,t} - \bar y_{\mathcal{B}_p,t}.

Under :eq:`factor` the common time effect cancels and

.. math::
   :label: gapdecomp

   g_{p,t}
       = \boldsymbol{\lambda}_t^{\top}\big(\bar{\boldsymbol{\mu}}_{\mathcal{A}_p}-\bar{\boldsymbol{\mu}}_{\mathcal{B}_p}\big)
       + \boldsymbol{\theta}_t^{\top}\big(\bar{\mathbf{z}}_{\mathcal{A}_p}-\bar{\mathbf{z}}_{\mathcal{B}_p}\big)
       + \big(\bar\varepsilon_{\mathcal{A}_p,t}-\bar\varepsilon_{\mathcal{B}_p,t}\big),

with :math:`\bar{\boldsymbol{\mu}}_{\mathcal{S}}, \bar{\mathbf{z}}_{\mathcal{S}}`
the weighted means over :math:`\mathcal{S}`. The
pair exhibits parallel trends precisely when the loadings are balanced,
:math:`\bar{\boldsymbol{\mu}}_{\mathcal{A}_p}=\bar{\boldsymbol{\mu}}_{\mathcal{B}_p}`
(and :math:`\bar{\mathbf{z}}_{\mathcal{A}_p}=\bar{\mathbf{z}}_{\mathcal{B}_p}`);
the gap is then constant in expectation and a difference-in-differences
comparison within the pair is unbiased.

Identifying assumptions
-----------------------

PANGEO can make a good experiment *likely*, but it cannot manufacture one
that the data do not support. Three assumptions underpin it, and each
points at a way the method can fail or stall.

1. Parallel trends (the crux). The design maximises parallelism in the
   pre-period; the validity of the Stage-2 effect estimate rests on that
   parallelism persisting into the post-period absent treatment — i.e.
   the treatment and control supergeos *would have continued to move
   together* had the campaign never launched. This is exactly the
   difference-in-differences parallel-trends assumption, and it is the
   assumption PANGEO is organised around.

   *Remark.* PANGEO *optimises* pre-period parallelism (making the
   assumption as plausible as the data allow) but cannot guarantee it holds
   out-of-sample. If a shock hits the treated markets, a competitor reacts
   only there, or the pre-period co-movement was coincidental, the
   post-period gap diverges on its own and the ATT is biased — the same
   Achilles' heel as any DiD. The achieved parallelism :math:`R^2` (low
   values mean no balanced design exists), the reported MDE, and a placebo /
   blank-window check on the held-out pre-period are the diagnostics that
   flag the risk.

2. A linear factor structure for the no-treatment outcomes
   (Eq. :eq:`factor`). The gap decomposition that makes "match on trajectory"
   equivalent to "balance the factor loadings" relies on this model. It is
   the standard synthetic-control / interactive-fixed-effects assumption.

   *Remark.* The model is mild for sales-like panels, but a wildly
   non-factor outcome (e.g. one driven by an idiosyncratic, unit-specific
   regime change) is not balanceable by any partition — no split of the geos
   can render :eq:`gapdecomp` constant in expectation.

3. A modest, designable geo pool. Each arm needs enough geos to form
   at least one supergeo pair, and the geos must be heterogeneous-but-
   matchable.

   *Remark.* A pool that cannot form a balanced pair (e.g. two wildly
   different markets) has no good design to find. Scale itself is not a
   barrier --- the default clustering partition handles large pools (see
   *Forming the supergeos* below) --- only the exact opt-out program is
   scale-limited.

When PANGEO Fails or Stalls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The concrete failure / stall modes:

- Parallel trends breaks post-launch — the dominant risk, above. No
  design fixes it; only the diagnostics warn of it.
- No matchable structure. If the geos are so heterogeneous that *no*
  partition achieves high parallelism, PANGEO still returns the best
  feasible design, but the parallelism :math:`R^2` stays low and the MDE
  blows up — a signal that a geo experiment here is underpowered and the
  read will be noisy regardless of split.
- The exact program stalls. Set partitioning is NP-hard, so the exact
  opt-out program (``fast=False``) can be slow or intractable with many geos
  and a large supergeo size :math:`Q`. The default clustering partition
  avoids this --- it is :math:`O(n\log n)` and, under cluster structure,
  matches the exact optimum (see *Forming the supergeos*). If you
  nonetheless need the exact program, cap :math:`Q` (smaller supergeos), use
  the automatic :math:`Q` selection, or raise ``min_pairs``.
- Too few geos / arms. A pool that cannot form a balanced pair (e.g.
  two wildly different markets) has no good design to find.

In short: PANGEO improves the *plausibility* of parallel trends by
construction and *quantifies the residual risk* (parallelism
:math:`R^2`, MDE), but it inherits DiD's identifying assumption instead of
removing it. Treat a low parallelism :math:`R^2` or a large MDE as
the design telling you the experiment is fragile.

Stage 1 --- the supergeo design
-------------------------------

The parallelism objective
^^^^^^^^^^^^^^^^^^^^^^^^^^

The pre-treatment window is split into an estimation window
:math:`\mathcal E` (the first :math:`\lfloor \kappa T_0\rfloor` periods,
:math:`\kappa=` ``frac_E``, default :math:`0.7`) and a held-out blank
window :math:`\mathcal B=\{1,\dots,T_0\}\setminus\mathcal E`. A pair is
scored by the variance of its *level-removed* gap over the estimation
window,

.. math::
   :label: score

   c(p) \coloneqq \sum_{t\in\mathcal E}\big(g_{p,t}-\bar g_p\big)^2,
   \qquad
   \bar g_p \coloneqq \frac{1}{|\mathcal E|}\sum_{t\in\mathcal E} g_{p,t},

which is exactly the pre-period residual sum of squares of a
difference-in-differences fit (cf.
:func:`mlsynth.utils.selector_helpers._did_from_mean`). Taking expectations
under :eq:`gapdecomp` with balanced covariates,

.. math::

   \mathbb E\,c(p)
     = \big(\bar{\boldsymbol{\mu}}_{A_p}-\bar{\boldsymbol{\mu}}_{B_p}\big)^{\top}
       \Big[\textstyle\sum_{t\in\mathcal E}(\boldsymbol{\lambda}_t-\bar{\boldsymbol{\lambda}})
            (\boldsymbol{\lambda}_t-\bar{\boldsymbol{\lambda}})^{\top}\Big]
       \big(\bar{\boldsymbol{\mu}}_{A_p}-\bar{\boldsymbol{\mu}}_{B_p}\big)
     \;+\; \mathbb E\!\sum_{t\in\mathcal E}
       \big(\bar\varepsilon_{A_p,t}-\bar\varepsilon_{B_p,t}-\overline{\cdot}\big)^2 .

The first term is a positive-definite quadratic form in the loading
imbalance, so minimising :eq:`score` drives
:math:`\bar{\boldsymbol{\mu}}_{A_p}\to\bar{\boldsymbol{\mu}}_{B_p}` --- it balances the *unobserved*
factor loadings, which is what parallel-trends DiD requires. The
time-constant component of the loading difference is absorbed by the level
shift :math:`\bar g_p` and never penalised: two supergeos may differ
arbitrarily in level yet match perfectly in *shape*. Scalar sum-matching,
by contrast, collapses the time dimension and is blind to shape.

Forming the supergeos
^^^^^^^^^^^^^^^^^^^^^^

Given the per-pair cost :eq:`score`, the design must partition each arm's geos
into supergeo pairs of minimum total non-parallelism. PANGEO offers two solvers
for this, and they optimise the *same* objective :eq:`score`. The econometric
content of the design --- balancing the factor loadings of :eq:`gapdecomp`,
i.e. making the supergeo halves move in parallel --- is therefore identical
whichever solver runs; they differ only in how they search. Both run
independently within each arm (arms hold disjoint geos), and both produce an
exact cover: every geo is assigned to exactly one supergeo pair, with nothing
trimmed.

Clustering partition (default, ``fast=True``). Geo-experiment panels almost
always have cluster structure: markets fall into a handful of latent types
that move together up to a level shift (shared seasonality, regional demand,
category-level trends). When that structure is present the best supergeos are
just groups of same-type geos, which can be found by clustering, not by
combinatorial search. The default solver --- an analogue of OSD (Shaw 2025)
for the trajectory objective --- forms the supergeos in five plain steps:

#. Level removal. Each geo's pre-period trajectory has its own time-mean
   subtracted, leaving the *shape* :math:`y_{it}-\bar y_i`. This is the same
   demeaning as the score :eq:`score`: two geos moving in parallel at any
   level become identical, so the grouping targets parallel trends instead of
   matching levels.
#. Embedding. The shapes are projected onto their leading principal
   components (denoising; under the factor model :eq:`factor` the shapes span
   the factor space, which PCA recovers).
#. Clustering. Hierarchical (Ward) linkage orders the geos so that
   parallel-moving markets are adjacent.
#. Size-bounded grouping. The ordering is cut into contiguous groups of size
   :math:`2,\dots,2Q` --- each splittable into two halves of size :math:`\le
   Q` --- giving the exact cover. ``min_pairs`` caps the group size so that at
   least that many pairs form.
#. Per-group split. Within each group the treatment and control halves are
   the split minimising :eq:`score` --- the same best split the exact program
   uses.

A handful of candidate groupings are generated (varying the linkage rule and a
small embedding perturbation, as in OSD) and the one with the lowest total
:eq:`score` is kept; ``fast_candidates`` sets how many. The cost is
:math:`O(n\log n)`, against the exponential search below.

*Remark.* The clustering partition is a heuristic: its total cost is
:math:`\ge` the exact optimum, with equality when the trajectory clusters are
clean. Checked against a structure-aware oracle (group by the true latent
clusters, then split each exactly), on clustered panels it *matches* the exact
optimum --- e.g. at :math:`N=60` geos with supergeos up to :math:`Q=6` it
recovers the oracle design in well under a second, a regime where the exact
program cannot even begin (it enumerates :math:`O(N^{2Q})` candidate subsets).
This is why it is the default. On data with no cluster structure the gap to the
optimum can be large; there the exact program is preferable when affordable.

The exact set-partitioning program (``fast=False``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exact solver enumerates every admissible pair and selects the optimal
exact cover by mixed-integer programming. Let :math:`\mathcal F` be the family
of admissible pairs: every subset
of the arm's geos of size :math:`2,\dots,2Q` that can be split into two
halves each of size :math:`\le Q`, each subset scored at its best such split
by :eq:`score`. Let :math:`\mathbf{M}\in\{0,1\}^{N\times|\mathcal F|}` be the geo-by-
pair incidence matrix (:math:`M_{iG}=1` iff geo :math:`i\in G`) and
:math:`c_G` the score of pair :math:`G`. The design solves the
set-partitioning program

.. math::
   :label: mip

   \min_{\mathbf{x}\in\{0,1\}^{|\mathcal F|}} \sum_{G\in\mathcal F} c_G\,x_G
   \quad\text{s.t.}\quad
   \mathbf{M}\mathbf{x} = \mathbf 1 \ \ (\text{exact cover}),\qquad
   \mathbf 1^{\top} \mathbf{x} \ge \kappa_{\min}\ \ (\text{minimum pairs}),

solved with HiGHS, through the standalone ``highspy`` interface when it is
installed and through ``cvxpy`` otherwise (``mip_backend``: ``"auto"``,
``"highspy"`` or ``"cvxpy"``; the solver diagnostics record which one ran).
Both state the same program and solve it to the optimum: the relative and
absolute MIP gaps are 0 unless ``mip_rel_gap`` or ``mip_abs_gap`` states
others. HiGHS's own default relative gap, :math:`10^{-4}`, stops at any design
within 0.01% of its dual bound and reports it optimal; on 20 markets with one
outlying market it returned a design costing 281,977,490 against the optimum
281,971,264. Two more steps keep the answer the solver's best. The costs are
multiplied by a power of two that puts the largest in
:math:`[2^{29}, 2^{30})`, since HiGHS's tolerances are absolute: the same
instance with its costs multiplied by :math:`10^{-12}` gave a design 368
dearer in the original units, with both gaps at 0. And a solve that does not
end optimal is repeated once with presolve off, since HiGHS's presolve has
returned "Solve error" on a feasible model the simplex solved without it.
The exact-cover
constraint :math:`\mathbf{M}\mathbf{x}=\mathbf 1` assigns every geo to exactly one chosen
pair (no geo is trimmed). Because each :math:`c_G` is *precomputed* offline,
the objective is linear in :math:`\mathbf{x}` --- the program is a mixed-integer
*linear* program regardless of the (possibly nonlinear) per-pair cost,
which is what keeps it tractable. Within each chosen pair the two halves are
the score-minimising split, and a fair coin, drawn independently in each pair
after the pairs are chosen, decides which half is treated, as in the
supergeo design of Chen et al. (2023, section 3). The coins come from ``assignment_seed`` (default 0): the pairs do
not depend on it, and the same seed on the same pre-period panel labels the
pairs the same way, so a run with ``post_col`` after the experiment reads the
assignment the experiment ran. Each pair's coin is on
``SupergeoPair.first_half_treated`` (whether the half holding the pair's first
unit is treated) and the seed on ``results.assignment_seed``. These coins are
the randomization the permutation test below inverts.

Which pairs reach the MIP
^^^^^^^^^^^^^^^^^^^^^^^^^

The family :math:`\mathcal F` grows like :math:`N^{2Q}`: 20 geos at
:math:`Q=3` give 60,439 pairs. Handed all of them, HiGHS took 175 s on the
seasonal panel below, of which its LP relaxation took 0.2 s. The exact path
therefore reaches the optimum of :eq:`mip` in three steps
(``mlsynth.utils.pangeo_helpers.columns``).

#. Scoring. Every pair is scored at its best split in compiled code. A split's
   cost is the residual sum of squares of fitting a level to the gap between
   its halves' mean paths, a least squares problem, and it is computed the way
   such problems are solved: each geo's level is removed first (the two-pass
   algorithm; Chan, Golub and LeVeque, 1983; Bjorck, 2015, section 2.1), the
   gap is formed directly from the centred paths, its level is removed again,
   and its squares are summed. The covariate term is the difference of the
   halves' arm-centred covariate means, formed the same way. Removing the levels
   first is exact, since the fitted level absorbs any constant added to a geo's
   path. The cost is not formed from the cross-product matrix of the paths
   :math:`\mathbf{A}^{\top}\mathbf{A}` as a quadratic form: that is the method of
   normal equations, whose error grows with the square of the problem's
   conditioning (Golub and Van Loan, 2013, section 5.3.7; Bjorck, 1996,
   section 2.2.1). Geos that share a large common factor and run nearly
   parallel are exactly the case it fails on. On panels of such geos in
   clusters of two (factor :math:`10^6`, levels :math:`10^7`, noise
   :math:`10^{-2}`), that form gave the costs that decide the design relative
   errors near 0.4, some of them negative, and a design above the exact
   optimum on 14 of 16 panels, by up to 154%. Each split also carries a lower
   and an upper bound on its exact cost, the cost in exact arithmetic on the
   panel's values, from the rounding error bounds of Rump (2010, Theorem 2.1).
#. Bound. The LP relaxation of :eq:`mip` is solved by column generation: it
   starts from a few pairs and adds the pairs of negative reduced cost
   :math:`\bar c_G = c_G - \sum_{i\in G}\pi_i - \sigma` until none is left,
   with :math:`\pi_i` the dual of geo :math:`i`'s cover row and
   :math:`\sigma\ge 0` that of the row
   :math:`\mathbf 1^{\top}\mathbf{x}\ge k`. Here
   :math:`k=\max(\kappa_{\min},\lceil N/2Q\rceil)`, the fewest pairs a design
   can have, since no pair holds more than :math:`2Q` geos (a Chvatal-Gomory
   cut). For every design :math:`\mathbf{x}` and every pair :math:`H` in it,

   .. math::

      \sum_G c_G x_G \;\ge\; L + \bar c_H,\qquad
      L = \sum_i \pi_i + \sigma k + \sum_{G} \min(\bar c_G, 0),

   whatever the duals (Geoffrion, 1974; Lubbecke and Desrosiers, 2005). The
   bound holds for any duals only if it is evaluated without rounding against
   it (Jansson, 2004; Neumaier and Shcherbina, 2004), so the reduced costs are
   formed from the costs' lower bounds less their own rounding, and :math:`L`
   is summed with its rounding subtracted and rounded down.
#. Elimination. A pair with :math:`\bar c_G > z - L` is in no design cheaper
   than :math:`z`, so the MIP needs only the pairs with
   :math:`\bar c_G \le z - L` (reduced-cost fixing; Nemhauser and Wolsey, 1988,
   section II.1.4). The first MIP takes the pairs of least reduced cost and a
   fixed design's pairs. When the best design so far costs :math:`z` and a pair
   left out has :math:`\bar c_G \le z - L`, the next MIP takes the pairs up to
   the smaller of :math:`z - L` and the reduced cost of a set four times larger,
   so a first design far above the optimum does not admit almost every pair at
   once (at 24 geos it admitted 189,950 of 190,026, and the MIP took 795 s). The
   search ends when every pair left out has :math:`\bar c_G > z - L`, or the MIP
   has every pair. With :math:`z` the sum of the design's upper bounds and the
   margin compared exactly, every design whose exact cost is at most the
   returned design's is in the last MIP; that MIP's optimum is HiGHS's, on the
   computed costs, to its tolerances.
#. Search before the MIP. Before any MIP, a depth-first search over exact
   covers (Knuth, 2000) looks for the design and its proof together
   (``mlsynth.utils.pangeo_helpers.cover``). It branches on the geo with the
   fewest pairs that still fit it, tries those pairs cheapest first, and drops
   a partial design whose bound reaches the best design's upper bound. The
   bound is the chosen pairs' lower bounds, plus the duals :math:`\pi_i` of the
   geos still uncovered, plus the number of pairs still to place times
   :math:`\sigma + \min_G \bar c_G`, taken at the fewest or the most pairs
   that can still be placed, whichever is smaller: the Lagrangian bound of what
   remains, at the root duals. Every comparison subtracts its own rounding
   error. It searches only the pairs the elimination keeps for the upper bound
   :math:`z` of a first design, found by a short search over the pairs of least
   reduced cost. A search that ends within its node cap (1.5 million pairs
   placed) has proved its design optimal, and no MIP runs. A search the cap
   stops leaves the MIPs to run as they would without it. The search runs with
   ``mip_backend="auto"``; naming ``"highspy"`` or ``"cvxpy"`` asks for that
   MIP and skips it.

The LP bound is weak on this problem: on the seasonal panels below it lies 14
to 33% under the optimum, and subset-row cuts (Jepsen, Petersen, Spoorendonk
and Pisinger, 2008) moved the gap at 20 geos and :math:`Q=3` only from 42.8 to
39.6%. When :math:`2Q` does not divide :math:`N`, the LP covers the arm with a
fractional :math:`N/2Q` of the cheap largest pairs, and the cut on
:math:`\mathbf 1^{\top}\mathbf{x}` halved that gap, to 22%. Even so, the last
MIP holds at most one pair in twelve, and at 28 geos one in 137. One arm of the
seasonal panel (``make_seasonal_sales_panel``, 60 periods, a 42-period
estimation window), with the optimum the same in every row both paths ran:

.. list-table::
   :header-rows: 1

   * - :math:`Q`
     - geos
     - pairs
     - every pair to the MIP
     - pairs in the last MIP
     - this path
   * - 2
     - 16
     - 2,500
     - 0.87 s
     - 208
     - 0.12 s
   * - 2
     - 18
     - 4,029
     - 2.4 s
     - 209
     - 0.75 s
   * - 2
     - 20
     - 6,175
     - 2.9 s
     - 210
     - 0.60 s
   * - 3
     - 14
     - 6,461
     - 4.7 s
     - 207
     - 0.08 s
   * - 3
     - 16
     - 14,876
     - 13.8 s
     - 208
     - 0.16 s
   * - 3
     - 18
     - 31,161
     - 40.4 s
     - 209
     - 0.25 s
   * - 3
     - 20
     - 60,439
     - 179.5 s
     - 643
     - 5.7 s
   * - 2
     - 30
     - 31,900
     - not run
     - 396
     - 6.7 s
   * - 3
     - 24
     - 190,026
     - not run
     - 1,100
     - 22.6 s
   * - 3
     - 28
     - 499,149
     - not run
     - 3,654
     - 56.5 s

The full path's time is enumeration in Python plus the MIP over every pair;
HiGHS's time on it varies between runs (13 s and 40 s at 16 geos). Scoring every pair
from the centred data takes 0.9 s of the 56.5 s at 28 geos; the MIPs take 55 s.

The search before the MIP replaces those MIPs where it finishes. On the same
panel (one arm, 42 estimation periods, single-threaded), the whole solve,
scoring and LP included, with the same optimum from both:

.. list-table::
   :header-rows: 1

   * - :math:`Q`
     - geos
     - search first
     - MIPs only
   * - 3
     - 24
     - 0.9 s, proved
     - 21.1 s
   * - 3
     - 28
     - 3.7 s, proved
     - 46.8 s
   * - 3
     - 32
     - 17.7 s, proved
     - 75.8 s
   * - 2
     - 30
     - 0.2 s, proved
     - 5.6 s
   * - 2
     - 40
     - 26.9 s, capped
     - 18.7 s
   * - 2
     - 50
     - 35.6 s, capped
     - 27.0 s

Its bound does not tighten as it goes deeper, where HiGHS re-solves the LP and
adds cuts at every node, so it loses where many designs lie between the bound
and the optimum. At :math:`Q=2` and 50 geos the root bound is 8.4% under the
optimum, and the proof needed 10 million nodes even when started from the
optimum (32.8 s). The node cap bounds what such an arm costs before the MIPs
take over, 8 to 9 s on the two capped arms above. A capped search does not
pass its design to the MIPs: on the 50-geo arm that made HiGHS take 58 to 64 s
over the same pairs, against 26 to 29 s without it.

``results.mip_reports`` records the pairs scored (``columns_scored``), the
pairs in the last MIP or the search (``columns_solved``), the bound :math:`L`
(``lagrangian_bound``), and what the search did (``search``: ``"proved"``,
``"node_limit"``, ``"off"``, or ``"unavailable"`` without numba) with the pairs
it placed (``search_nodes``). When it proved the design, ``backend`` is
``"search"``, ``dual_bound`` a lower bound on the optimum and ``proven_gap``
the certified gap. The per-arm diagnostics add the LP's rounds and value,
the MIPs solved, and the seconds spent scoring, on the LP and in the MIPs. An
arm larger than 20 million pairs raises and names ``fast=True``.

Per-pair objectives
^^^^^^^^^^^^^^^^^^^^

The ``objective`` argument selects the per-pair cost :math:`c_G`; every
choice leaves :eq:`mip` a linear program. Writing :math:`g_t` for the gap
of a candidate split and :math:`\bar g` for its estimation-window mean,

* ``"ss_res"`` (default) --- the absolute residual sum of squares
  :math:`\sum_t (g_t-\bar g)^2`. Scale-dependent, so high-amplitude pairs
  weigh more and the design prioritises making large markets parallel.
* ``"r2"`` --- the scale-free criterion
  :math:`1-R^2 = \sum_t(g_t-\bar g)^2 / \sum_t(\bar Y_{A,t}-\overline{\bar Y_A})^2`,
  so every pair counts equally (FDID's :math:`R^2` criterion, optimised
  *exactly* by the program, not greedily).
* ``"weighted"`` --- a recency-weighted residual SS
  :math:`\sum_t w_t (g_t-\bar g_w)^2`, the level removed at the weighted mean
  :math:`\bar g_w`, with weights :math:`w_t=\rho_{\mathrm{dec}}^{\,T_0-1-t}`
  (``recency_decay``), up-weighting the recent pre-period closest to the
  experiment.
* ``"supergeo_total"`` --- the objective of the Supergeo paper (Chen,
  Doudchenko, Jiang, Stein and Ying 2023), the squared difference between
  the halves' totals over the estimation window :math:`\mathcal E`,
  :math:`\big(\sum_{i\in A}\sum_{t\in\mathcal E} Y_{it} - \sum_{i\in B}\sum_{t\in\mathcal E} Y_{it}\big)^2`.
  Sums over markets, no level shift and no trajectory: the level is what
  it balances, and two markets with equal totals score as a perfect match
  whatever their shapes. It is the baseline PANGEO's trajectory objectives
  are compared against. It is available for pairs (``max_supergeo_size=1``)
  on the exact path (``fast=False``); with pairs the optimal design pairs
  the markets in sorted order of their totals, neighbour with neighbour, and
  PANGEO designs it that way (see Pairs, below).

The per-pair ``gap_variance`` and ``parallelism_r2`` reported on the result
are always the unweighted quantities of :eq:`score`, so designs from
different objectives are comparable on a common yardstick.

Pairs: a closed form and a matching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`Q=1` every supergeo is one geo, so a design of an arm is a
perfect matching of its geos, and :eq:`mip` is the minimum-weight perfect
matching problem (Edmonds 1965). On the exact path (``fast=False``) with
``mip_backend="auto"``, PANGEO solves it directly, in one of two ways.

* Under ``"supergeo_total"`` with no covariates, the cost of a pair is the
  squared difference of two scalars, the geos' totals over :math:`\mathcal E`.
  For totals :math:`a\le b\le c\le d`, the neighbour pairing costs
  :math:`(b-a)^2+(d-c)^2`, no more than either the crossing pairing
  :math:`(c-a)^2+(d-b)^2` or the nested one :math:`(d-a)^2+(c-b)^2`. A design
  with two crossing or nested pairs can therefore be uncrossed at no extra
  cost, and the optimum pairs the geos in sorted order of their totals,
  neighbour with neighbour. The design is a sort, with no program to solve.
  It is also the most precise randomized design when the totals order the
  geos' expected outcomes: among stratified randomizations that treat each
  unit with probability one half, the difference in means has the least MSE
  under pairs of neighbours in that order (Bai 2022, Theorems 1 and 2).
* Under any other objective, or with covariates, the cost of a pair depends
  on whole trajectories and there is no closed form. The design is solved by
  Edmonds' blossom algorithm, which returns with the pairing the dual
  solution that proves it optimal. PANGEO checks that proof in exact
  arithmetic before it returns, and records it in the solver diagnostics
  (``proved_optimal``, ``certificate_violation``, ``certificate_tolerance``).

An explicit ``mip_backend="highspy"`` or ``"cvxpy"`` solves :eq:`mip`
instead. The solver diagnostics' ``path`` records which route ran:
``closed_form_sorted_pairs``, ``blossom_matching`` or ``exact_mip``. On the
174 geos of google/trimmed_match's example, the sorted design takes 40 to
57 ms against 0.71 to 0.93 s for the MIP, at the same objective value.

An arm with an odd number of geos has no perfect matching, so one geo is
left out: the one whose removal lets the rest pair at the least cost. Under
``"supergeo_total"`` without covariates it sits at an even position of the
sorted totals (leaving out an odd position is never better than leaving out
the one before it), so one scan of the sorted order finds it. Under every
other cost, the matcher adds a dummy geo joined to every geo at cost 0, and
the geo paired with the dummy is the one left out (Kaibel and Walter 2017).
The geo left out is recorded on ``ArmDesign.excluded_units`` and
``results.excluded_units``, named in a warning, and kept out of
``results.assignment``; it takes no part in the experiment or its analysis.
An explicit MIP backend and the fast path have no such step, and an odd arm
raises there.

Supergeo size :math:`Q` and automatic selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``max_supergeo_size`` :math:`=Q=1` recovers the classic
matched-pairs design; :math:`Q>1` permits composite supergeos when no single
geo matches another well, without trimming. :math:`Q` is a granularity knob
with an interior optimum: too small and no parallel matches exist
(singleton geos are too noisy); too large and the arm yields few, coarse
pairs. The program-level MDE is *not* monotone in :math:`Q` and is *not*
tracked by the parallelism :math:`R^2` (which is scale-free and rises with
:math:`Q`); only the absolute residual variance that drives power matters.

Consequently, if ``max_supergeo_size`` is left unset, PANGEO selects
:math:`Q` automatically: it solves :eq:`mip` for every feasible
:math:`Q\in\{1,\dots,\min(\lceil N/2\rceil, 6)\}` and returns the design
with the smallest mean program MDE. The full sweep --- each :math:`Q`'s
program-pair count, mean program MDE, and the :math:`2/2^{P}`
randomisation-inference p-value floor for :math:`P` pairs --- is recorded in
``results.metadata["q_sweep"]`` and the choice in
``results.metadata["q_selected"]``, so the decision is auditable and may be
overridden with an explicit :math:`Q`.

The default selection rule (``q_selection="mde_min"``) minimises the mean
program MDE. That rule sees only one axis --- power --- and can pick a
:math:`Q` whose MDE edge is within the sampling noise of a design with more
pairs. The alternative ``q_selection="pareto_1se"`` treats the choice as the
two-objective problem it is: minimise the MDE *and* maximise the pair count
:math:`P` (more pairs give a finer randomisation reference and a lower
:math:`2/2^{P}` p-value floor). It keeps only the Pareto-efficient
:math:`Q` on (MDE :math:`\downarrow`, :math:`P` :math:`\uparrow`), then a
one-standard-error tie-break returns the *largest* :math:`P` whose MDE is
within one SE of the frontier's best --- spending pairs on power only when the
gain exceeds its own noise. The SE is the deterministic small-sample estimate
:math:`\mathrm{MDE}/\sqrt{2(B-1)}` on the :math:`B`-period blank window, and an
optional ``q_min_pairs`` sets a hard inference floor. Setting
``compute_q_sweep=True`` records the full sweep even when :math:`Q` is fixed,
so a chosen :math:`Q` can be audited against the alternatives.

Solver diagnostics. Every design records how the partition solver ran in
``results.metadata["solver_diagnostics"]`` (per arm). The clustering
(``fast=True``) path reports the candidate groupings tried and feasible, the
winning candidate and its linkage, and each candidate's total score; the exact
set-partitioning path (``fast=False``) reports the number of candidate
supergeo pairs :math:`|\mathcal F|`, the MIP objective, optimality gap and
dual bound, node/iteration counts, and the solve time. The same path puts one
typed record per arm on ``results.mip_reports``
(:class:`~mlsynth.utils.pangeo_helpers.structures.MipSolveReport`): the
status, whether the solve was repeated with presolve off, the gaps asked for
and whether the solver could be given them (through ``cvxpy``, GLPK_MI and CBC
cannot), and the proven gap and dual bound. When no exact cover exists the
raised error names the structural obstruction (units that appear in no
admissible pair, or an odd-arm/even-pair parity clash) and whether the solve
was repeated, not a generic failure.

Balancing baseline covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallelism is level-blind: by :eq:`score` the level shift
:math:`\bar g_p` absorbs any time-constant gap, so a baseline characteristic
(population, income) that merely shifts a market's level is differenced out
and never enters the trajectory score. This is correct for parallel-trends
DiD but says nothing about balance on such characteristics --- the role of
OSD's scalar covariate matching. PANGEO restores it with a standardised
mean-difference penalty appended to :eq:`score`,

.. math::
   :label: covpen

   c(p) \;\longmapsto\; c(p)
     \;+\; \sum_{m} \omega^{\mathrm{cov}}_m
       \Big(\frac{\bar c_{A_p,m}-\bar c_{B_p,m}}{s_m}\Big)^2,

the weighted squared standardised mean difference (SMD) between the halves'
covariate means, where :math:`s_m` is the cross-geo standard deviation
(``standardize_covariates``, default ``True``) and :math:`\omega^{\mathrm{cov}}_m`
a per-covariate weight (``covariate_weights``, default :math:`1`). Because
:eq:`covpen` is precomputed it preserves linearity in :eq:`mip`. Larger
weights buy tighter covariate balance at the cost of some parallelism; the
achieved per-pair SMDs are reported in ``SupergeoPair.covariate_smd``. Pass
``covariates=[...]`` (baseline columns, each reduced to its per-geo mean) to
enable; with no covariates the design is unchanged. This is also the
Abadie & Zhao (2026, Thm. 1) prescription --- moving structure from the
unobserved :math:`\boldsymbol{\mu}_i` into the observed :math:`\mathbf{z}_i` lowers the
estimator's bias --- and the Stage-2 device for restoring inferential
validity (below).

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, covariates=True)

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
       "covariates": ["population", "income"],
       "covariate_weights": {"population": 5.0, "income": 5.0},
   }).fit()

   for arm, design in res.arm_designs.items():
       for p in design.pairs:
           print(p.treatment, p.control, p.parallelism_r2, p.covariate_smd)

Power and the minimum detectable effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because power and the design objective are governed by the same supergeo
gap residual, :meth:`mlsynth.PANGEO.fit` returns a power analysis
(``results.power``). For pair :math:`p` the per-period noise is estimated
honestly on the held-out blank window :math:`\mathcal B` (out of sample
with respect to the optimisation) as the residual of the *same counterfactual
model used at evaluation* (:eq:`adid`) --- fit on the estimation window
:math:`\mathcal E`, evaluated on :math:`\mathcal B`:

.. math::

   \widehat\sigma_p^2
     \coloneqq \frac{1}{|\mathcal B|-1}\sum_{t\in\mathcal B} \widehat e_{p,t}^2 .

Using the evaluation model here (the augmented-DiD residual by default,
or the plain level-removed gap when ``att_augment=False``), not a
fixed recipe keeps the projected MDE and the realised standard error
(:eq:`adidvar`) coherent. The :math:`X`-period effect for the pair then has
variance :math:`\widehat\sigma_p^2\,[f(X,\rho)+f(T_0,\rho)]`, where

.. math::

   f(n,\rho) = \frac{1}{n}\Big(1 + 2\sum_{k=1}^{n-1}(1-\tfrac{k}{n})\rho^{k}\Big)

is the variance-inflation factor of the mean of :math:`n` AR(1)-correlated
periods and :math:`\rho` is the pooled lag-1 autocorrelation of the blank
residuals. Serial correlation is decisive: weekly sales are highly
autocorrelated, so :math:`X` post weeks are worth far fewer than :math:`X`
independent observations and adding post periods yields sharply diminishing
returns --- the trap a naive i.i.d. power calculation falls into.

The program-level effect is the treated-size-weighted average of the pair
effects, with weights
:math:`w_p = (\sum_{i\in A_p}\omega_i)/\sum_{q}\sum_{i\in A_q}\omega_i`, and
(treating pairs as independent across the program)

.. math::

   \widehat{\operatorname{Var}}(\widehat\tau_{\mathrm{prog}})
     = \sum_p w_p^2\,\widehat\sigma_p^2\,\big[f(X,\rho)+f(T_0,\rho)\big],
   \qquad
   \mathrm{MDE}(X) = \big(z_{1-\alpha/2}+z_{1-\beta}\big)\,
       \sqrt{\widehat{\operatorname{Var}}(\widehat\tau_{\mathrm{prog}})}.

The program level is the headline: small arms are individually
under-powered (with :math:`P` pairs a pure within-pair randomisation test
has a hard p-value floor of :math:`2/2^{P}`, so one needs :math:`P\ge 6` to
reach :math:`p<0.05`), whereas pooling across arms gives the program an
effective sample size equal to the total pair count and routinely detects
effects several points smaller than any one arm. Per-arm curves are stored
in ``results.power.arms``. The MDE is reported in outcome units and as a
percent of the treated baseline, by default at :math:`1-\beta=0.80` power
for horizons :math:`X=2,\dots,12`; ``power_target``, ``power_alpha`` and
``power_post_periods`` configure this and ``compute_power=False`` skips it.

.. code-block:: python

   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "max_supergeo_size": 3,
   }).fit()

   pw = res.power
   print(f"serial correlation rho = {pw.serial_correlation:.2f}")
   print(pw.summary())                       # MDE % by horizon: program + arms
   print(pw.program.mde_pct_by_horizon()[8]) # detectable % lift after 8 weeks
   print(pw.power_for_effect(effect_pct=5.0, post_periods=8))  # invert: power

Stage 2 --- evaluation by Augmented DiD
---------------------------------------

The estimator
^^^^^^^^^^^^^

Once the experiment has run, pass a ``post_col`` (a :math:`0/1` indicator of
post-treatment periods, as in LEXSCM). The design is rebuilt on the pre rows
alone --- so it is identical to the design-only result --- and
``results.effects`` carries the realised ATT at the arm and program
levels using the Augmented Difference-in-Differences estimator of Li &
Van den Bulte (2022). Both the pre and the post rows are read through
:func:`mlsynth.utils.datautils.geoex_dataprep`, which refuses a duplicate or
absent unit-period and a missing or non-finite outcome among the design's
units with ``MlsynthDataError``; a geo that appears only in the post rows is
ignored.

Fix a level (an arm, or the program) and write :math:`y^{T}_t` for its
treated supergeo aggregate and :math:`y^{C}_t` for its control supergeo
aggregate, both treated-size-weighted across the level's pairs. The
counterfactual is the pre-period least-squares projection

.. math::
   :label: adid

   y^{T}_t = \delta_1 + \delta_2\,y^{C}_t + \gamma\,t + e_t,
   \qquad t = 1,\dots,T_0 .

This *augments* plain DiD in two ways: the control scale :math:`\delta_2` is
estimated, not fixed at :math:`1`, and a linear time trend
:math:`\gamma t` is included (``att_augment`` and ``att_trend``, both default
``True``). With regressor :math:`\mathbf{x}_t=(1,\,y^{C}_t,\,t)^{\top}` and OLS
estimate :math:`\widehat{\boldsymbol{\delta}}`, the per-period effect and the ATT are

.. math::

   \widehat u_t \coloneqq y^{T}_t - \mathbf{x}_t^{\top}\widehat{\boldsymbol{\delta}},
   \qquad
   \widehat\tau \coloneqq \frac{1}{T_{\mathrm{post}}}
       \sum_{t=T_0+1}^{T} \widehat u_t .

The percent ATT is taken relative to the post-period counterfactual
(cf. :func:`mlsynth.utils.resultutils.effects.calculate`), not the
pre-treatment baseline:

.. math::

   \widehat\tau_{\%} \coloneqq 100\times\frac{\widehat\tau}{\bar y^{0}_{\mathrm{post}}},
   \qquad
   \bar y^{0}_{\mathrm{post}}
     \coloneqq \frac{1}{T_{\mathrm{post}}}\sum_{t=T_0+1}^{T} \mathbf{x}_t^{\top}\widehat{\boldsymbol{\delta}}
     = \frac{1}{T_{\mathrm{post}}}\sum_{t=T_0+1}^{T}\big(y^{T}_t-\widehat u_t\big).

Inference
^^^^^^^^^

Li & Van den Bulte (2022, Prop. 3.1--3.3) show
:math:`\sqrt{T_{\mathrm{post}}}\,(\widehat\tau-\tau)\xrightarrow{d}
N(0,\Sigma_1+\Sigma_2)`, where :math:`\Sigma_1` is the variance from
estimating :math:`\boldsymbol{\delta}` and :math:`\Sigma_2` from averaging the
post-period errors. Their Web Appendix C.13 gives the
prediction-variance estimator

.. math::
   :label: adidvar

   \widehat{\operatorname{Var}}(\widehat\tau)
     = \widehat\omega^2\Big[\,
         \bar{\mathbf{x}}_{\mathrm{post}}^{\top}
         \Big(\textstyle\sum_{t=1}^{T_0} \mathbf{x}_t \mathbf{x}_t^{\top}\Big)^{-1}
         \bar{\mathbf{x}}_{\mathrm{post}}
         \;+\; \frac{1}{T_{\mathrm{post}}}\Big],

with :math:`\bar{\mathbf{x}}_{\mathrm{post}}=T_{\mathrm{post}}^{-1}\sum_{t>T_0}\mathbf{x}_t`.
The first bracketed term is :math:`\Sigma_1` (it inflates automatically when
the post-period control drifts outside its pre-period range, pricing the
extrapolation uncertainty) and the second is :math:`\Sigma_2`. The residual
variance :math:`\widehat\omega^2` is estimated over the long pre-period as a
Newey--West/Bartlett long-run variance with truncation lag
:math:`\lfloor T_0^{1/4}\rfloor` (Li & Van den Bulte's
:math:`O(T^{1/4})` rule); lag :math:`0` is the i.i.d. case
:math:`\widehat\omega^2=\widehat e^{\top}\widehat e/(T_0-k)` for :math:`k` regressors.
The confidence interval is
:math:`\widehat\tau \pm z_{1-\alpha/2}\sqrt{\widehat{\operatorname{Var}}(\widehat\tau)}`
and the p-value is the two-sided normal test of :math:`\tau=0`.

Design-based inference. The Augmented-DiD interval above is *model-based*:
it is valid when the regression residual is weakly-dependent stationary.
Because treatment is randomised *within* each supergeo pair, PANGEO also
reports an assumption-light *design-based* companion at
``results.effects.randomization`` (per arm and for the program). Let
:math:`d_k` be the antisymmetric within-pair difference-in-differences
contrast --- treated-minus-control, level-removed --- which negates exactly
when a pair's assignment is flipped. Under the sharp null of no effect the
only randomness is the assignment, so the null distribution of the
weighted-mean statistic is generated by the :math:`2^{P}` sign flips
:math:`\{\sum_k w_k s_k d_k : s\in\{\pm1\}^{P}\}`; the two-sided permutation
p-value is the share with :math:`|\text{stat}|\ge|\text{observed}|`
(enumerated exactly for :math:`P\le 14` pairs, otherwise a fixed-seed Monte
Carlo). The companion matched-pair (pair-clustered) standard error
:math:`\sqrt{\sum_k w_k^2 (d_k-\widehat\tau)^2}\,P/(P-1)` --- which reduces to
:math:`\mathrm{sd}(d_k)/\sqrt{P}` under equal weights --- gives a
:math:`t_{P-1}` interval that respects the pairing. The permutation p-value
cannot fall below :math:`2/2^{P}`, so a design with few pairs honestly reports
weak design-based evidence even when the model-based SE is tight --- the same
:math:`P`-versus-power tension the ``q_sweep`` p-value floor surfaces at the
planning stage.

Why this estimator suits the supergeo gap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Li & Van den Bulte's regularity conditions (Assumptions C2--C3) explicitly
admit trend and unit-root (integrated) common factors :math:`\boldsymbol{\lambda}_t`
--- the regimes under which naive i.i.d. standard errors collapse. The
mechanism is the augmentation: regressing the treated aggregate on a
*scaled* control is a cointegrating regression, and a single
:math:`\delta_2` cancels a shared integrated factor in :eq:`gapdecomp`,
while :math:`\gamma t` absorbs deterministic drift. The validity condition
reduces to a single requirement --- that the regression residual
:math:`e_t` be (weakly dependent) stationary --- which the augmentation and
trend deliver. The design's parallelism is retained throughout; it minimises
the residual variance, which by :eq:`adidvar` directly tightens the standard
error --- and, because the power analysis reads the *same* held-out residual,
the planning MDE as well.

Plain DiD as an option. Setting ``att_augment=False`` (and optionally
``att_trend=False``) recovers Li & Van den Bulte's ordinary
difference-in-differences --- ``y^{T}_t - y^{C}_t = \delta_1 [+ \gamma t] +
e_t`` with the control coefficient fixed at one --- and the power analysis
follows suit, so the two stages stay coherent. A head-to-head Monte-Carlo
comparison (R\ :sup:`2` design + plain DiD versus the augmented defaults)
found augmented DiD both more precise (lower realised MDE) and
better-covering across the stationary, trend-plus-seasonal and
integrated-factor regimes, because plain DiD has no mechanism to absorb a
control-scale mismatch or a trend and leaves that structure in its residual.
The augmented estimator is therefore the default; plain DiD remains available
for settings where its textbook simplicity is preferred.

The result object
^^^^^^^^^^^^^^^^^

``fit()`` returns a :class:`~mlsynth.config_models.DesignResult` --- the
experimental-design half of mlsynth's two-family result contract --- because
PANGEO chooses an assignment *before* any intervention. The chosen treated
markets are on ``results.selected_units`` and the full map on
``results.assignment``; the supergeo pairs, achieved parallelism, power/MDE and
solver diagnostics are on ``results.arm_designs``, ``results.power`` and
``results.metadata``. When the panel carries a ``post_col``, the design
*resolves* to an effect report: ``results.report`` is a standard
:class:`~mlsynth.config_models.EffectResult` exposing the program ATT through
the same flat surface as any estimation-stage method
(``.att``, ``.counterfactual``, ``.gap``, ``.att_ci``), while the richer
per-pair, per-arm and design-based numbers remain on ``results.effects``.

Validity envelope (smoke tests)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Monte-Carlo study over the bundled simulator confirms that validity hinges
on residual stationarity, not on the interval recipe. The simulator can
place the unobserved factor on an i.i.d., AR(1) or random-walk process
(``factor``) and toggle the seasonal amplitude (``season_amp``) and per-geo
trend (``trend_sd``):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Gap structure
     - Program coverage (nominal 0.95)
     - Type-I (nominal 0.05)
   * - stationary i.i.d. factor (paper DGP)
     - 0.93
     - 0.07
   * - + linear trend + seasonality
     - 0.87
     - 0.13
   * - + integrated factor (random walk)
     - 0.60
     - 0.40

The point estimate is unbiased in every regime. On a stationary gap ---
matching Li & Van den Bulte's factor-model design --- the prediction-
variance interval is at its nominal rate; the augmentation and trend
regressor recover most of the coverage lost to a deterministic trend and
seasonality; and the adversarial random-walk-plus-seasonality gap, where two
integrated factors and amplitude-heterogeneous seasonality exceed what a
single :math:`\delta_2` can cointegrate, marks the honest assumption
boundary. In practice the fitted :math:`\widehat\delta_2` (reported as
``AttEstimate.scale``) and the residual diagnose whether the assumption
holds; if a single scale cannot flatten the gap, add covariate or seasonal
regressors before trusting the interval.

Because the power analysis now uses the evaluation model's held-out
residual, the planning MDE is calibrated to the realised standard error:
on the stationary gap the projected MDE matches the realised value to within
roughly 7% (ratio :math:`\approx 0.93`), and on integrated gaps it is
conservative (over-states the MDE), the safe direction. These experiments
live in ``mlsynth/tests/test_pangeo.py`` (``TestADIDInference``).

.. code-block:: python

   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=104, seed=0, n_post=8)
   res = PANGEO({
       "df": df, "outcome": "sales", "arm": "arm",
       "unitid": "unit", "time": "time", "post_col": "post_col",
       "max_supergeo_size": 3,
       "att_augment": True, "att_trend": True,   # Augmented DiD (defaults)
   }).fit()

   print(res.effects.summary())           # program + per-arm ATT, SE, CI, p
   pe = res.effects.program
   print(f"program ATT = {pe.att_pct:.1f}% "
         f"[{pe.ci_lower_pct:.1f}, {pe.ci_upper_pct:.1f}], "
         f"p={pe.p_value:.3f}, scale delta_2={pe.scale:.2f}")

Verification
------------

Two benchmark cases pin PANGEO. ``pangeo_supergeo_mc``
(`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pangeo_supergeo_mc.py>`__)
reproduces the trajectory-matching simulation below. ``pangeo_trimmed_match_pairs``
(`source <https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/pangeo_trimmed_match_pairs.py>`__)
cross-validates the pairs design under ``objective="supergeo_total"`` against
google/trimmed_match's own pairing, run live in its own environment on the
package's example data: the same 87 pairs. See
:doc:`replications/pangeo_trimmed_match`.

Core API
--------

.. automodule:: mlsynth.estimators.pangeo
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.PANGEOConfig
   :members:
   :undoc-members:

Helper Modules
--------------

.. automodule:: mlsynth.utils.pangeo_helpers.parallelism
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.mip
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.pipeline
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.power
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.effects
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.pangeo_helpers.structures
   :members:
   :undoc-members:

Example
-------

A seasonal, multi-arm sales panel (the bundled simulator), designed into
parallel supergeo pairs. With ``display_graphs=True`` PANGEO plots, per arm,
the observed treated supergeo aggregate against the augmented-DiD
counterfactual prediction (:func:`mlsynth.utils.pangeo_helpers.effects.adid_counterfactual`):
the in-sample pre-period fit when designing, and -- once ``post_col`` data
are supplied -- the counterfactual extended past the treatment date, so the
post-window gap is the estimated effect.

.. code-block:: python

   from mlsynth import PANGEO
   from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel

   # 3 arms (non-overlapping geos), 6 geos each, 156 weeks of history.
   df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                  T=156, seed=0)

   res = PANGEO({
       "df": df,
       "outcome": "sales",
       "arm": "arm",                # single categorical arm column
       "unitid": "unit",
       "time": "time",
       "max_supergeo_size": 3,      # Q
   }).fit()

   for arm, design in res.arm_designs.items():
       print(f"Arm {arm}: {len(design.pairs)} pair(s), "
             f"parallel-trends R^2 = {design.mean_parallelism_r2:.3f}")
       for p in design.pairs:
           print(f"   T={p.treatment}  C={p.control}  R^2={p.parallelism_r2:.3f}")

   # res.assignment maps every geo -> 'treatment' / 'control'.

On the simulated data this returns designs with parallel-trends :math:`R^2`
around 0.90--0.98 --- roughly 10--35x more parallel than a random
treatment/control split of the same geos.

Simulation: Trajectory Matching vs. a Scalar Supergeo
-----------------------------------------------------

The supergeo design of Chen et al. (2023) matches (super)geos on a
scalar summary of baseline response (its variance term is
:math:`\sum_k (Z_{G_{k,+}} - Z_{G_{k,-}})^2`, a sum over scalar baseline
differences). PANGEO carries this into the panel setting: it matches
on the full pre-treatment *trajectory* (level-removed parallelism), so it
can separate geos that look identical on a scalar yet move differently
over time. The self-contained Monte Carlo below makes the gap concrete —
adapting the paper's RMSE comparison (supergeo vs. matched pairs) to a
panel where every geo has the same pre-period mean but a distinct
trajectory shape, a setting in which scalar matching is by construction
blind. Six geos keep the MIP instantaneous.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import PANGEO

   def make_panel(rng, T_pre=20, S=6, level=100.0, noise=0.6):
       """Six geos = three parallel pairs (up-trend, down-trend, cycle);
       each shape is demeaned over the pre-period so all pre-means match."""
       T = T_pre + S
       t = np.arange(T)
       up = (t - t.mean()) / t.std()
       cyc = np.sin(2 * np.pi * t / 5.0)
       shapes = [5 * up, 5 * up, -5 * up, -5 * up, 5 * cyc, 5 * cyc]
       cols = []
       for s in shapes:
           s = s - s[:T_pre].mean()              # equal pre-means => scalar-blind
           cols.append(level + s + rng.normal(0, noise, T))
       return np.column_stack(cols), T_pre

   def did(Y, T_pre, treated, control, tau):
       Yo = Y.copy()
       Yo[T_pre:, treated] += tau                # inject the true effect
       t_eff = Yo[T_pre:, treated].mean() - Yo[:T_pre, treated].mean()
       c_eff = Yo[T_pre:, control].mean() - Yo[:T_pre, control].mean()
       return t_eff - c_eff

   def pangeo_design(Y, T_pre):
       df = pd.DataFrame(
           {"geo": f"g{g}", "t": int(t), "y": Y[t, g], "arm": "A"}
           for g in range(6) for t in range(T_pre)
       )
       a = PANGEO({"df": df, "outcome": "y", "unitid": "geo", "time": "t",
                   "arm": "arm", "max_supergeo_size": 1,
                   "compute_power": False, "display_graphs": False}).fit().assignment
       T = [int(g[1:]) for g, v in a.items() if v == "treatment"]
       C = [int(g[1:]) for g, v in a.items() if v == "control"]
       return T, C

   def scalar_match(Y, T_pre, rng):              # Google-style: pair on scalar mean
       order = np.argsort(Y[:T_pre].mean(0))
       T, C = [], []
       for k in range(0, 6, 2):
           a, b = order[k], order[k + 1]
           if rng.random() < 0.5:
               T.append(a); C.append(b)
           else:
               T.append(b); C.append(a)
       return T, C

   tau, R = 4.0, 60
   rng = np.random.default_rng(1)
   errs = {"PANGEO (trajectory)": [], "scalar matched-pairs": []}
   for _ in range(R):
       Y, T_pre = make_panel(rng)
       Tp, Cp = pangeo_design(Y, T_pre)
       errs["PANGEO (trajectory)"].append(did(Y, T_pre, Tp, Cp, tau) - tau)
       Ts, Cs = scalar_match(Y, T_pre, rng)
       errs["scalar matched-pairs"].append(did(Y, T_pre, Ts, Cs, tau) - tau)

   for name, e in errs.items():
       e = np.array(e)
       print(f"{name:22s} RMSE = {np.sqrt((e ** 2).mean()):.2f}")

Because all geos share a pre-period mean, the scalar match pairs them
essentially at random, and the difference-in-differences estimate
inherits the up/down/cycle shape mismatch (RMSE ≈ 6 against a true effect
of 4). PANGEO reads the trajectory shape, recovers the three parallel
pairs, and estimates the effect about 30× more precisely (RMSE ≈ 0.2).
That is the supergeo idea carried into the panel world: match on *how the
series move*, not on a single number.

References
----------

Chen, A., Doudchenko, N., Jiang, S., Stein, C., & Ying, B. (2023).
"Supergeo Design: Generalized Matching for Geographic Experiments."
arXiv:2301.12044.

Bai, Y. (2022). "Optimality of Matched-Pair Designs in Randomized
Controlled Trials." *American Economic Review* 112(12):3911-3940.

Kaibel, V., & Walter, M. (2017). "A Note on Matchings Constructed during
Edmonds' Weighted Perfect Matching Algorithm." arXiv:1703.09505.

Edmonds, J. (1965). "Paths, Trees, and Flowers." *Canadian Journal of
Mathematics* 17:449-467.

Edmonds, J. (1965). "Maximum Matching and a Polyhedron with 0,1-Vertices."
*Journal of Research of the National Bureau of Standards* 69B:125-130.

Shaw, C. (2025). "Optimized Supergeo Design: A Scalable Framework for
Geographic Marketing Experiments." arXiv:2506.20499.

Li, K. T. (2023). "Frontiers: A Simple Forward Difference-in-Differences
Method." *Marketing Science* 43(2):267-279.

Li, K. T., & Van den Bulte, C. (2022). "Augmented Difference-in-Differences."
*Marketing Science* 42(4):746-767.

Geoffrion, A. M. (1974). "Lagrangean Relaxation for Integer Programming."
*Mathematical Programming Study* 2:82-114.

Lubbecke, M. E., & Desrosiers, J. (2005). "Selected Topics in Column
Generation." *Operations Research* 53(6):1007-1023.

Nemhauser, G. L., & Wolsey, L. A. (1988). *Integer and Combinatorial
Optimization.* Wiley.

Knuth, D. E. (2000). "Dancing Links." In J. Davies, B. Roscoe and J. Woodcock
(eds.), *Millennial Perspectives in Computer Science*, Palgrave, 187-214.

Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing
the Sample Variance: Analysis and Recommendations." *The American Statistician*
37(3):241-247.

Bjorck, A. (1996). *Numerical Methods for Least Squares Problems.* SIAM.

Bjorck, A. (2015). *Numerical Methods in Matrix Computations.* Springer.

Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns
Hopkins University Press.

Rump, S. M. (2010). "Verification Methods: Rigorous Results Using
Floating-Point Arithmetic." *Acta Numerica* 19:287-449.

Jansson, C. (2004). "Rigorous Lower and Upper Bounds in Linear Programming."
*SIAM Journal on Optimization* 14(3):914-935.

Neumaier, A., & Shcherbina, O. (2004). "Safe Bounds in Linear and Mixed-Integer
Linear Programming." *Mathematical Programming* 99:283-296.

Chvatal, V. (1973). "Edmonds Polytopes and a Hierarchy of Combinatorial
Problems." *Discrete Mathematics* 4(4):305-337.

Jepsen, M., Petersen, B., Spoorendonk, S., & Pisinger, D. (2008).
"Subset-Row Inequalities Applied to the Vehicle-Routing Problem with Time
Windows." *Operations Research* 56(2):497-511.

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."
*Working paper.*

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control
Methods for Comparative Case Studies." *Journal of the American
Statistical Association* 105(490):493-505.

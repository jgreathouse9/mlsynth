Synthetic Controls for Experimental Design (MAREX)
==================================================

.. currentmodule:: mlsynth

When to Use This Estimator
--------------------------

The estimators elsewhere in ``mlsynth`` are *retrospective*: a treatment has
already happened and you reweight donors to reconstruct the treated unit's
counterfactual. MAREX, due to Abadie and Zhao (2026) [ABADIE2024]_, is
*prospective* — it designs an experiment. Before any treatment is assigned,
and using only pre-experimental data, it chooses which aggregate units to
treat and which to hold out as controls, so that the experiment you are
about to run yields a credible estimate.

The motivating setting is a firm (say a ride-sharing company) that wants to test
a new policy but can only deploy it in a *few* whole markets. A within-market
A/B test is contaminated by interference (treated and control drivers compete);
randomizing whole markets to treatment is unbiased *ex ante* but, with a handful
of large units, routinely produces treated and control groups with very
different baselines, so any single realisation is badly off. MAREX instead picks
the treated and control markets so their pre-experiment predictors match the
population — a non-randomized design that, the paper shows, substantially
reduces estimation bias relative to randomization.

Reach for MAREX when:

* Units are large aggregates (markets, regions, stores) and only one or a
  few can be treated.
* You control the assignment and want to choose it well, not estimate after
  the fact.
* Interference or equity rules out within-unit randomization, forcing
  whole-unit treatment.

Notation
--------

There are :math:`N` units :math:`\mathcal{N} \coloneqq \{1, \dots, N\}` and
:math:`T` periods :math:`t \in \mathcal{T} \coloneqq \{1, \dots, T\}`,
1-indexed; the experiment takes effect after period :math:`T_0`, splitting
:math:`\mathcal{T}` into the pre-experiment window
:math:`\mathcal{T}_1 \coloneqq \{t \in \mathcal{T} : t \le T_0\}` (of length
:math:`T_0`) and the experimental window
:math:`\mathcal{T}_2 \coloneqq \{t \in \mathcal{T} : t > T_0\}`. Because MAREX
designs an experiment instead of reweighting around one already-treated unit,
units are indexed generically by :math:`i, j \in \mathcal{N}` with no forced
treated unit. Each unit has a pre-intervention predictor vector
:math:`\mathbf{x}_j` (pre-period outcomes and optional covariates);
:math:`\bar{\mathbf{x}} = \sum_j f_j \mathbf{x}_j` is the population predictor
mean for known weights :math:`f_j` (e.g. market shares, or :math:`1/N`). The
experimenter chooses treated weights :math:`\mathbf{w}` and control weights
:math:`\mathbf{v}`, both on the simplex, and *disjoint*:

.. math::

   \sum_j w_j = 1,\quad \sum_j v_j = 1,\quad w_j, v_j \ge 0,\quad w_j v_j = 0
   \;\;\forall j.

Units with :math:`w_j > 0` are treated; among the rest, units with
:math:`v_j > 0` form the synthetic control. Writing :math:`y_{jt}` for the
observed outcome (treated units realise :math:`y_{jt}^I` over
:math:`\mathcal{T}_2`, everyone else :math:`y_{jt}^N`), the design estimator of
the average effect is

.. math::

   \widehat{\tau}_t(\mathbf{w}, \mathbf{v}) = \sum_j w_j y_{jt} - \sum_j v_j y_{jt},
   \qquad t \in \mathcal{T}_2.

Assumptions
-----------

Assumption 1 (linear factor model). Potential outcomes follow

.. math::

   y_{jt}^N = \delta_t + \boldsymbol{\theta}_t^\top \mathbf{z}_j
   + \boldsymbol{\lambda}_t^\top \boldsymbol{\mu}_j + \varepsilon_{jt},
   \qquad
   y_{jt}^I = \upsilon_t + \boldsymbol{\gamma}_t^\top \mathbf{z}_j
   + \boldsymbol{\eta}_t^\top \boldsymbol{\mu}_j + \xi_{jt},

with observed covariates :math:`\mathbf{z}_j`, unobserved factors
:math:`\boldsymbol{\mu}_j`, and mean-zero idiosyncratic noise.

*Remark.* This is the interactive-fixed-effects model of the SC literature
(Abadie-Diamond-Hainmueller 2010), extended with a *separate* factor structure
for the treated potential outcome — necessary because a design must choose a
treatment group, not just a comparison group.

Assumption 2 (regularity). The factor loadings are non-degenerate
(:math:`F \le T_E`, smallest eigenvalue bounded below) and the noise is
i.i.d. sub-Gaussian with common variance, independent across the two potential
outcomes; dependence *across units* is allowed.

*Remark.* These conditions keep the factor structure recoverable from the
pre-experiment window and the noise well-behaved, so the population predictor
mean :math:`\bar{\mathbf{x}}` is a meaningful matching target, not an
artifact of a degenerate loading matrix.

Assumption 3 / 4 (fit quality). A weight vector reproducing the population
predictor means exists exactly (Assumption 3), or approximately within a
tolerance :math:`d` (Assumption 4). This is the design-time analogue of "the
treated unit lies in the convex hull of the donors."

*Remark.* Under these conditions Abadie & Zhao bound the bias of
:math:`\widehat{\tau}_t(\mathbf{w}, \mathbf{v})` and develop the permutation test
below; the better the pre-experiment match, the smaller the bias.

Mathematical Formulation
------------------------

The Design Optimization
^^^^^^^^^^^^^^^^^^^^^^^

MAREX chooses :math:`\mathbf{w}, \mathbf{v}` (and a binary selection mask
:math:`\mathbf{z}`, with :math:`w_j \le z_j`, :math:`v_j \le 1 - z_j`, so a unit
is treated *or* a control, never both) to match the population predictor mean.
The ``base`` design minimises

.. math::

   \min_{\mathbf{w}, \mathbf{v}, \mathbf{z}}\;
   \Bigl\| \bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j \Bigr\|_2^2
   + \Bigl\| \bar{\mathbf{x}} - \sum_j v_j \mathbf{x}_j \Bigr\|_2^2
   \quad \text{s.t. the simplex / disjointness / cardinality constraints,}

with the number of treated units pinned by ``m_eq`` (exactly) or bounded by
``m_min``/``m_max``. This is a mixed-integer quadratic program (the binary
``z``); ``mlsynth`` solves it with SCIP by default, or — via ``relaxed=True`` —
relaxes ``z`` to :math:`[0, 1]`, solves the QP, and discretises post hoc.

From SCIP's solution MAREX keeps only the selection. SCIP counts a binary
within its integrality tolerance of an integer as integral, so a selection
variable left near :math:`10^{-6}` rounds to "not treated" while still allowing
a treated weight of that size, and a market can then carry weight on both sides
of the design. On one five-market test panel SCIP's point did exactly that: it
used a market as a treated unit with weight :math:`10^{-6}` and as a control
with weight 0.146, and reached an objective below that of any feasible design.
MAREX therefore fixes the selection at its rounding and refits the weights with
the selection held constant, which is a convex QP, so the weights and objective
it reports belong to the design the selection defines.

Most of the MIQP's cost is SCIP *proving* a design optimal, which grows steeply
with the number of markets and treated units. Two options manage that. A
``warm_start`` (a list of treated unit labels) seeds the search with a known
good design — for instance LEXSCM's top candidate, which solves a near-identical
problem by lexicographic search, not by proof:
``MAREX(..., warm_start=lexscm_warm_start(lex.fit()))`` (the helper
``lexscm_warm_start`` lives in ``mlsynth.utils.marex_helpers.warmstart``). The
seed is only a hint,
so a solve that runs to completion returns the identical proven optimum; it just
starts the branch-and-bound from a strong incumbent instead of hunting for one.
Paired with ``time_limit`` (seconds), MAREX then returns the best design found
within the budget — typically the seed, confirmed or refined — instead of paying
the full optimality proof. Both default to off and apply to the exact MIQP only.
For every design, ``backend="exact"`` skips the MIQP and finds the same optimum
by searching treated sets directly (:ref:`marex-search-backend`).

``mlsynth`` exposes four objective variants through ``design`` (clear names
that map to the paper's formulations):

* ``"standard"`` — match each predictor mean with both synthetic units
  (formulation 5);
* ``"weakly_targeted"`` — match the treated synthetic to the mean and softly tie
  the control synthetic to it (weight ``beta``);
* ``"penalized"`` — ``standard`` plus a distance penalty that down-weights
  units far from the population mean (``lambda1`` / ``lambda2``);
* ``"unit_penalized"`` — ``standard`` plus unit-level penalties (``xi`` /
  ``lambda1_unit``).

The ``xi`` term is formulation (10): it charges each candidate treated unit for
how well the remaining controls reproduce that unit, weighted by how much of
the synthetic treated it carries. The charge for a given unit is a minimum over
the control weights fitted to it alone, so it does not depend on the design
being chosen; ``mlsynth`` solves those per-unit problems once and the penalty is
then linear in the treated weights. Writing it as the treated weight times the
joint design's own fit to that unit gives a different objective, and one that is
not convex in the two weight vectors together.

The paper's control-side counterpart ``lambda2_unit`` is accepted only as
``0``. It pairs the treated weights against the control weights through a
distance matrix, which is bilinear in two decision variables; ``lambda1_unit``
charges the treated side against the population target, and the ``penalized``
design charges both sides against it separately.

Control penalty
^^^^^^^^^^^^^^^

The ``weakly_targeted`` design takes one more term, ``control_penalty``
(:math:`\lambda`), from Vives-i-Bastida (2022, program 1):

.. math::

   \Bigl\| \bar{\mathbf{x}} - \mathbf{u} \Bigr\|_2^2
   + \beta \Bigl\| \mathbf{u} - \sum_j v_j \mathbf{x}_j \Bigr\|_2^2
   + \beta \lambda \sum_j v_j \bigl\| \mathbf{u} - \mathbf{x}_j \bigr\|_2^2,
   \qquad \mathbf{u} = \sum_j w_j \mathbf{x}_j.

The last term is Abadie and L'Hour's (2021) penalty on the controls: each
control market is charged its distance from the synthetic treated unit, so the
synthetic control is built from a few nearby markets instead of many distant
ones that average out. Vives-i-Bastida's design for the Catalan UBI pilot uses
:math:`\lambda = 0.1`. At ``control_penalty=0``, the default, the design is
the weakly-targeted design as before.

As written, the term multiplies a control weight by a function of the treated
weights. Because the control weights sum to one it equals
:math:`\|\mathbf{u}\|^2 - 2\,\mathbf{u}^\top \mathbf{s} + \sum_j v_j \|\mathbf{x}_j\|^2`
with :math:`\mathbf{s} = \sum_j v_j \mathbf{x}_j`, a quadratic in the two
synthetic units plus a linear term in :math:`\mathbf{v}`. The whole objective is
then convex in :math:`(\mathbf{w}, \mathbf{v})` exactly when
:math:`\beta \lambda (1 + \lambda) \le 1`, and MAREX refuses a configuration
past that bound when it is built, since neither backend could certify a
minimum there. At :math:`\beta = 1` the bound is :math:`\lambda \le 0.618`; as
:math:`\beta \to 0`, the lexicographic limit of Vives-i-Bastida's Algorithm 1,
it places no limit on :math:`\lambda`. The MIQP receives the objective as a
sum of squares in the centred synthetic units; the exact backend handles it as
described under :ref:`marex-search-backend`.

Covariates
^^^^^^^^^^

By default the design matches on pre-period outcomes. Passing ``covariates``
(column names) appends them to the predictor vector,
:math:`\mathbf{x}_j = [\mathbf{y}^E_j ; \mathbf{z}_j]`, exactly as in the paper — the synthetic treated and
control are then balanced on both pre-period outcomes *and* covariates (with an
optional ``covariate_weight`` scale). With ``standardize=True`` every predictor
column is first divided by its standard deviation across units, and the
covariate columns are then multiplied by ``covariate_weight``, so a weight of
0.1 makes each covariate count one tenth as much as one period of the outcome.
This is the order of Vives-i-Bastida (2022, Algorithm 1), whose design puts the
covariates at 1/10 of the outcomes' weight. ``covariate_weight`` also takes a
mapping from covariate name to weight, so each covariate can carry its own
(Vives-i-Bastida's diagonal :math:`\mathbf{H}`): for example
``covariate_weight={"age": 0.1, "income": 0.3}``. The mapping must name exactly
the listed covariates, every weight must be non-negative, and a weight of 0
leaves that covariate out of the match. A column that does not vary across
units is left as it is: the weights on each side sum to one, so such a column
adds nothing to the objective. When the pre-period is long, the outcomes
already encode the covariates' contribution, so covariates matter most when few
pre-periods are available.

Each covariate enters as its per-unit mean over the fitting periods, the first
``T0 - blank_periods`` pre-periods, the same window the outcomes are matched
on. A covariate that does not change over time is its value. A covariate that
does change is averaged over the fitting window only: the blank periods are
kept out of the choice of the design so that their gaps can serve as the
placebo distribution for inference, and their covariate values are kept out
with them. Vives-i-Bastida (2022) averages the covariates over the fitting
years in the same way. The balance diagnostics below use the same means.

Clustering, Costs, and Budgets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passing a ``cluster`` column solves the design *within* each cluster (one or a
few treated units per cluster), which better approximates the population
predictor distribution and limits interpolation bias (paper OA.1). Per-unit
``costs`` and a ``budget`` (scalar or per-cluster) add a knapsack constraint
:math:`\sum_j c_j w_j \le B`, so the chosen treatment group respects a spend cap.

Geographic design restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAREX already carries most geographic structure natively. Its ``cluster`` field
*is* the paper's "region as a distinct experimental design" (p.13: weather
patterns dependent across cities in the same region motivate treating each region
separately); ``m_min`` / ``m_max`` per cluster is a stratum quota ("at least /
at most this many treated per region"); ``costs`` / ``budget`` is the cost bound;
and because each cluster's control synthetic :math:`v_{\cdot,k}` is built only
from cluster-``k`` members, donors are automatically same-region. The same
restriction vocabulary SYNDES and LEXSCM expose then adds the rest, as
constraints on the MIP:

* ``to_be_treated`` / ``not_to_be_treated`` -- force a market in (one leadership
  already committed to) or out (a regulated market, or one already under another
  test); a forbidden market stays a donor.
* ``adjacency`` + ``spillover_threshold`` -- no two treated markets may share a
  border (:math:`\sum_k z_{ik} + \sum_k z_{jk} \le 1`), for when one test
  market's campaign would bleed into a neighbouring one and bias the lift.
* ``size_col`` + ``min_size`` / ``max_size`` -- a treated-unit size band (the
  floor is a power minimum, the ceiling is synthesizability); out-of-band markets
  stay donors.
* ``cluster_col`` -- at most one treated market per cluster value
  (:math:`\sum_k z_{ik} + \sum_k z_{jk} \le 1` for same-cluster pairs), a
  no-two-from-one-cluster rule for markets that share a media buy or retail
  footprint. This is distinct from the ``cluster`` design grouping above: it adds
  conflict constraints to the treated set instead of splitting the objective.
* ``stratum_col`` + ``min_per_stratum`` / ``max_per_stratum`` -- a coverage quota
  on the treated set of one design (at least / at most this many treated per
  stratum). Use this when the design is a single cluster but the read still has to
  span regions; when each region is its own design, ``cluster`` with per-cluster
  ``m_min`` / ``m_max`` expresses the same quota.
* ``exclude_bordering_donors`` -- drop a treated market's bordering neighbours
  from its (within-cluster) control pool, so a market partly treated by spillover
  never sits in its counterfactual. Requires ``adjacency``.

These are supported in the exact MIQP and in the search backend
(:ref:`marex-search-backend`), not with ``relaxed=True``, whose post-hoc
rounding cannot guarantee them. An over-constrained design -- e.g. forbidding
every market of a region, leaving its required treated synthetic with no member
-- raises a translated :class:`~mlsynth.exceptions.MlsynthEstimationError` naming
the restrictions. The example below runs on the bundled real US DMA contiguity
map and Census regions, with a region-grouped linear factor sales model (after
Liao, Shi & Zheng [RelaxSC]_); the geography is real, the sales reproducible.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MAREX

   base = ("https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
           "refs/heads/main/basedata/markets")
   adj = pd.read_csv(f"{base}/dma_adjacency.csv", index_col=0)
   meta = pd.read_csv(f"{base}/dma_metadata.csv")
   CDC = {**{s: "Midwest" for s in ["OH", "IN", "MI", "KY"]},
          **{s: "South" for s in ["WV", "VA", "TN", "GA", "FL", "SC"]}}
   meta = meta[meta["state"].isin(CDC)].copy()
   meta["region"] = meta["state"].map(CDC)
   names = [n for n in meta["dma_name"] if n in adj.index]
   meta = meta[meta["dma_name"].isin(names)].reset_index(drop=True)
   adj = adj.loc[names, names]

   rng = np.random.default_rng(7)
   reg = meta["region"].to_numpy(); n, r, T, T0 = len(names), 3, 30, 24
   Lam = (np.array([{g: rng.normal(size=r) for g in sorted(set(reg))}[g] for g in reg])
          + 0.15 * rng.normal(size=(n, r)))
   F = np.cumsum(rng.normal(size=(T, r)), axis=0)
   pop = np.round(rng.lognormal(12.5, 0.7, n)).astype(int)
   Y = 100 + rng.normal(0, 5, n) + F @ Lam.T + rng.normal(0, 1, (T, n))
   df = pd.DataFrame([{"dma": names[j], "week": t, "sales": float(Y[t, j]),
                       "region": reg[j], "population": int(pop[j])}
                      for j in range(n) for t in range(T)])

   B = dict(df=df, outcome="sales", unitid="dma", time="week",
            cluster="region", T0=T0)                    # cluster = Census region

   MAREX({**B, "m_max": 2}).fit()                       # native quota: <=2 per region
   MAREX({**B, "m_max": 2, "to_be_treated": ["Atlanta, GA"],
          "not_to_be_treated": ["Miami-Fort Lauderdale, FL"]}).fit()
   MAREX({**B, "m_max": 2, "adjacency": adj,
          "spillover_threshold": 0.5}).fit()            # no two treated border
   MAREX({**B, "m_max": 2, "size_col": "population",
          "min_size": 200_000, "max_size": 3_000_000}).fit()
   MAREX({**B, "m_max": 1, "adjacency": adj, "spillover_threshold": 0.5,
          "exclude_bordering_donors": True}).fit()      # non-bordering donors

Across groups: cluster vs. quotas vs. restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three mechanisms touch on grouping/geography; they are distinct, not
interchangeable, and ``cluster`` is *not* made obsolete by the restrictions:

* A *separate experiment per group* -- each group's own representativeness
  target and donor pool: ``cluster``. This is the design's objective, not a
  constraint: each cluster ``k`` reconstructs its own predictor mean
  :math:`\overline{X}_k`. It is the SYNDES analog of that estimator's ``arm``
  (:doc:`syndes`), but baked into the objective, not run as separate
  solves -- which is why, in MAREX, the restrictions compose *with* ``cluster``
  (they apply within each cluster) and the per-cluster cardinality ``m_min`` /
  ``m_max`` *is* the stratum quota.
* *One* design with geographic / forcing limits within those clusters: the
  restriction suite above (force in/out, border conflict, size band, donor
  exclusion).

So a stratum quota is not an alternative to ``cluster``; in MAREX it is a
property *of* the clustering. Drop ``cluster`` (a single global cluster) and you
recover one design against the whole-population mean -- a different estimand, not
a constrained version of the clustered one.

.. _marex-search-backend:

Solving the design by search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``backend`` chooses how the design is solved. The default, ``"mip"``, hands the
mixed-integer program above to SCIP. ``"exact"`` searches over treated sets
directly and returns the same optimal design. It calls no mixed-integer solver,
so it also runs on a base install, without the ``design`` extra that provides
``pyscipopt``.

The search works because fixing the treated set splits the problem in two. In
one cluster, write :math:`\mathcal{S}` for the treated markets,
:math:`\mathcal{N}_0(\mathcal{S})` for the cluster's other markets, and
:math:`\Delta^{\mathcal{S}}` for the weight vectors on :math:`\mathcal{S}` that
are non-negative and sum to one. In the ``standard`` design the treated weights
live on :math:`\mathcal{S}` and the control weights on
:math:`\mathcal{N}_0(\mathcal{S})`, and no term of the objective involves both,
so once :math:`\mathcal{S}` is named the design is two separate
synthetic-control fits:

.. math::

   f(\mathcal{S}) =
     \underbrace{\min_{\mathbf{w} \in \Delta^{\mathcal{S}}}
       \Bigl\| \bar{\mathbf{x}} - \sum_{j \in \mathcal{S}} w_j \mathbf{x}_j
       \Bigr\|_2^2}_{g(\mathcal{S})}
     \;+\;
     \underbrace{\min_{\mathbf{v} \in \Delta^{\mathcal{N}_0(\mathcal{S})}}
       \Bigl\| \bar{\mathbf{x}}
         - \sum_{j \in \mathcal{N}_0(\mathcal{S})} v_j \mathbf{x}_j
       \Bigr\|_2^2}_{h(\mathcal{S})}.

Each fit is a small quadratic program that ``mlsynth`` solves exactly with an
active-set method. The ``penalized`` and ``unit_penalized`` designs add
penalties that are non-negative and linear in :math:`\mathbf{w}` or in
:math:`\mathbf{v}`, and the split survives them. The optimal design is the
:math:`\mathcal{S}` with the smallest :math:`f(\mathcal{S})` among the treated
sets the cardinality (``m_eq``, or ``m_min`` and ``m_max``) allows, and the
binary variables of the MIQP are one way of searching those sets.

Fitting every candidate is what Abadie and Zhao's own cardinality routine does
(``Synthetic_Experiment_Cardinality_Constraint`` in their code). It is exact,
but it pays two fits per candidate, and :math:`m` treated among :math:`N`
markets is :math:`\binom{N}{m}` candidates. The search fits few of them, because
:math:`g(\mathcal{S})` has a lower bound that costs much less than the fit. The
weights sum to one, so :math:`\bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j =
-\sum_j w_j (\mathbf{x}_j - \bar{\mathbf{x}})`: the fit is unchanged when every
predictor vector is measured from the target. Let
:math:`\mathbf{G}_{\mathcal{S}}` hold the inner products of those centred
vectors for the markets in :math:`\mathcal{S}`. Dropping only the requirement
that the weights be non-negative leaves a problem with a closed-form answer,

.. math::

   g(\mathcal{S}) \;\ge\; \min_{\mathbf{1}^\top \mathbf{w} = 1}
   \mathbf{w}^\top \mathbf{G}_{\mathcal{S}} \mathbf{w}
   \;=\; \frac{1}{\mathbf{1}^\top \mathbf{G}_{\mathcal{S}}^{-1} \mathbf{1}},

with equality when the minimiser has no negative weight. When the design adds a
penalty :math:`\mathbf{a}^\top \mathbf{w}` to the treated fit, the bound becomes
:math:`(2 + t)^2 / (4s) - u/4`, with
:math:`s = \mathbf{1}^\top \mathbf{G}_{\mathcal{S}}^{-1} \mathbf{1}`,
:math:`t = \mathbf{1}^\top \mathbf{G}_{\mathcal{S}}^{-1} \mathbf{a}` and
:math:`u = \mathbf{a}^\top \mathbf{G}_{\mathcal{S}}^{-1} \mathbf{a}`. When
:math:`\mathbf{G}_{\mathcal{S}}` is singular (more treated markets than
predictors, or a market that matches the target exactly) the bound is set to
zero, which discards nothing. The bound is one small linear solve, computed for
thousands of candidates in a single call. Since
:math:`f(\mathcal{S}) \ge g(\mathcal{S})`, a candidate whose bound already
exceeds the best design found so far cannot be optimal, and the search discards
it without fitting it.

The search reads the candidates once, keeps the 512 with the smallest bounds,
and fits those in order of bound; the best of them becomes the incumbent. If
every kept bound is still under the incumbent, a candidate outside the 512 could
yet win, and a second pass fits every candidate whose bound is under the
incumbent. Each candidate is then either fitted or excluded by its bound, so the
design returned is the optimum of the program SCIP solves. The design and
everything computed from it, inference and the post-fit included, go through
the same code for both backends.

The geographic restrictions carry over, within each cluster. ``to_be_treated``,
``not_to_be_treated``, the size band and the stratum quotas decide which treated
sets exist, so the search generates only the sets they allow (with the
enumeration of SYNDES's search, :doc:`syndes`) and counts them exactly. Border
and same-cluster conflicts are checked on each candidate.
``exclude_bordering_donors`` removes a candidate's bordering markets from its
control pool, which changes :math:`h(\mathcal{S})` and leaves
:math:`g(\mathcal{S})` and its bound alone. The clusters share no weights, so
each is searched on its own; a border conflict between markets of two clusters,
or a stratum quota that counts markets in two clusters, would tie those searches
together, and ``backend="exact"`` refuses it. With ``top_K > 1`` each cluster's
search keeps its ``top_K`` best treated sets, and the pool is the ``top_K``
smallest totals of one choice per cluster; the MIQP builds the same pool by
re-solving once per design, with a cut excluding the designs already found.

The ``weakly_targeted`` design ties the two sides together. Its objective is

.. math::

   f(\mathcal{S}) =
     \min_{\mathbf{w} \in \Delta^{\mathcal{S}},\;
           \mathbf{v} \in \Delta^{\mathcal{N}_0(\mathcal{S})}}
     \Bigl\| \bar{\mathbf{x}} - \sum_{j \in \mathcal{S}} w_j \mathbf{x}_j
     \Bigr\|_2^2
     + \beta \Bigl\| \sum_{j \in \mathcal{S}} w_j \mathbf{x}_j
       - \sum_{j \in \mathcal{N}_0(\mathcal{S})} v_j \mathbf{x}_j
     \Bigr\|_2^2,

and its second term involves both sets of weights, so naming
:math:`\mathcal{S}` no longer leaves two separate fits. It leaves a convex
problem in the treated weights alone,
:math:`\phi(\mathbf{w}) = \|\bar{\mathbf{x}} - \mathbf{u}\|^2 + \beta\, d(\mathbf{u})^2`
with :math:`\mathbf{u} = \sum_{j \in \mathcal{S}} w_j \mathbf{x}_j` the synthetic
treated and :math:`d(\mathbf{u})` its distance to the convex hull of the control
markets, since for a given synthetic treated the best synthetic control is the
nearest point of that hull.

The search minimises :math:`\phi` one face of that hull at a time. The nearest
point of a face's affine hull moves linearly with :math:`\mathbf{u}`, so on the
face where the current synthetic control sits, :math:`\phi` is a quadratic in
:math:`\mathbf{w}`, and minimising it over the treated simplex is a single
synthetic-control fit with a stacked design. That fit is the step. The search
keeps a step only if :math:`\phi`, with the controls refitted exactly, goes
down; it halves the step otherwise, and if no halving helps it takes one
alternating step (the treated fit to
:math:`(\bar{\mathbf{x}} + \beta \mathbf{c}) / (1 + \beta)` for the current
synthetic control :math:`\mathbf{c}`, then the controls). Because each step
solves its face exactly, the number of steps does not grow with :math:`\beta`:
across 400 random problems it was never more than three, at :math:`\beta` from 0 to
100. Plain alternation between the two fits converges at the rate
:math:`\beta / (1 + \beta)` and needed a median of 1,750 steps per candidate at
:math:`\beta = 100` on GeoLift's forty markets.

The fit stops on a certificate, not on a small step. The Frank-Wolfe gap (Frank
and Wolfe 1956), the improvement the best vertex of each simplex offers along
the gradient, bounds the distance to the minimum from above, and the fit stops
when the gap is under :math:`10^{-10}` of the objective. A fit whose gap is
still open after 20,000 steps raises with a message naming ``backend="mip"``,
since ranking candidates on an uncertified value could discard the optimal
design. The same gap gives a certified lower bound at every step, the objective
less the gap. The search hands each fit the objective of the best design found
so far, and the fit stops and discards its candidate as soon as that lower bound
is above it.

The coupling term is non-negative, so :math:`f(\mathcal{S}) \ge g(\mathcal{S})`
still holds and the bound and both passes prune as they do for the other
designs. At large :math:`\beta` that bound prunes little, because the coupling
term then dominates the objective and a bound on the treated fit cannot see it.
A second bound sees it. The synthetic treated lies in the convex hull of the
treated markets (the set of their weighted averages with non-negative weights
summing to one), and the synthetic control in the convex hull of the control
pool, so the coupling term is at least :math:`\beta` times the squared distance
between the two hulls:

.. math::

   f(\mathcal{S}) \;\ge\; g(\mathcal{S}) + \beta\,
   \operatorname{dist}\bigl(\operatorname{conv}\{\mathbf{x}_j\}_{j \in \mathcal{S}},\;
   \operatorname{conv}\{\mathbf{x}_j\}_{j \in \mathcal{N}_0(\mathcal{S})}\bigr)^2.

Wolfe's (1976) nearest-point method finds the distance by walking toward the
closest pair of points of the two hulls. Every step offers a direction
:math:`\mathbf{z}`, and any direction bounds the distance from below (his
Theorem 2.2):

.. math::

   \operatorname{dist} \;\ge\;
   \frac{\min_{i \in \mathcal{S}} \mathbf{z}^\top \mathbf{x}_i
          - \max_{j \in \mathcal{N}_0(\mathcal{S})} \mathbf{z}^\top \mathbf{x}_j}
         {\|\mathbf{z}\|_2},

the gap between the two hulls' shadows on the line through :math:`\mathbf{z}`.
The search runs the method on 512 candidates at a time, stops each one as soon
as its bound clears the best design found so far, and fits only the candidates
it cannot discard; until a first design exists there is nothing to clear, so
the first eight are fitted unscreened. The bound reads the data only through
the products :math:`\mathbf{z}^\top \mathbf{x}_j`, and the search evaluates it
from the data with the rounding of those products bounded in advance (Higham
2002, section 3.1), so whatever the iteration did, the bound cannot exceed the
distance. It takes the closed-form treated bound in place of
:math:`g(\mathcal{S})`, so pricing a candidate needs no fit.

On GeoLift's forty markets with four treated, the search fitted 11 of the
91,390 candidates at :math:`\beta = 1` and took 0.13 seconds, against 234 fits
and 0.92 seconds without the coupling bound. At :math:`\beta = 100` it fitted
677 and took 16 seconds, against 82,296 fits and four minutes. Both searches
return the same design with or without it. Each fit the coupling bound leaves
still stops on the cutoff, usually after one or two steps. Before the face steps and the cutoff, the same
searches took 25 seconds and, by extrapolation from the measured cost per
candidate, about five days. SCIP, given 30 minutes on each of the same two
problems through ``backend="mip"``, stopped at designs whose objectives were 2.1
times the optimum at :math:`\beta = 1` and 2.3 times it at :math:`\beta = 100`,
without a proof of optimality for either.

With a ``control_penalty`` :math:`\lambda > 0` the same fit runs on the
penalised objective. For a given synthetic treated :math:`\mathbf{u}` the best
controls are a synthetic-control fit to :math:`(1 + \lambda)\mathbf{u}` with
each market charged :math:`\lambda \|\mathbf{x}_j\|^2` (all points measured from
the target), so they remain one exact simplex fit. On a face the minimum over
the controls is again a quadratic in :math:`\mathbf{u}`, now with a linear part
that moves along the face, and the step is a stacked fit with a linear term. The
Frank-Wolfe gap and the cutoff carry over unchanged, since the objective is
convex under the bound above. Measured over 40 random problems per setting, at
:math:`\beta` from 0.01 to 100 and :math:`\lambda` up to 0.99 of the bound, the
fit certified in at most four steps, none of them a fallback sweep, in about two
milliseconds. The penalty is non-negative, so the treated-fit bound still
prunes, and it is at least :math:`\beta\lambda` times the squared coupling
term by convexity of the square, so the coupling bound scales by
:math:`1 + \lambda`. It sees less of this objective: the penalty also charges
the spread of the controls around the synthetic control, which no distance
between the hulls measures. At :math:`\beta = 100` and :math:`\lambda = 0.0099`,
just inside the convexity bound, the coupling bound reached a median 41 percent
of a candidate's objective on 300 sampled candidates, and the search fitted
49,899 of the 91,390 and took 9.6 minutes, against 87,499 fits and 15.4 minutes
without it.

Abadie and Zhao's enumeration solves two separate synthetic-control problems per
candidate (supplementary appendix, Section OA.6), and they solve this design by
their quadratic-programming route instead: a non-convex quadratically
constrained program passed to Gurobi. They report numerical errors of up to
0.001 in the weights from it, round the weights to two digits, and set weights
at or below 0.01 to zero. The search solves the same program and returns its
weights unrounded.

``backend="exact"`` refuses, when the configuration is built and with a message
naming ``backend="mip"``:

* a negative ``beta``, which makes the weakly-targeted coupling concave, so the
  design has no minimum to certify;
* ``costs`` with a ``budget``, which caps a weighted sum of the treated weights,
  a second constraint on the treated fit that the search's simplex solver does
  not take;
* ``relaxed=True``, and the branch-and-bound settings ``solver``,
  ``warm_start`` and ``time_limit``, which have nothing to act on.

Above twenty million admissible treated sets in a cluster, pricing every set
stops being affordable. The ``standard``, ``penalized`` and ``unit_penalized``
designs are then searched by a certified branch and bound
(:ref:`marex-branch-and-bound`), which returns the same optimum; the
``weakly_targeted`` design is refused, with the same pointer to the MIQP. Five
treated among forty markets is 658,008 candidates. With no treated count set, every size from
one to :math:`N - 1` is a candidate, :math:`2^N - 2` sets in all, which passes
the limit at 25 markets. The count is of the sets the restrictions allow, so
forcing markets in or out raises the cluster size the search can take.

On the Walmart design below (45 stores, two treated), the whole fit, inference
included, takes 0.03 seconds with ``backend="exact"`` and returns the same two
stores and the same p-value as the MIQP, whose design step took between 46 and
83 seconds on the same data. On simulated one-cluster panels of 12 to 40
markets with three to five treated, three panels per size and both backends on
one core, the search took 9 to 524 milliseconds and SCIP 0.5 to 249 seconds.
The geometric-mean speedup per size was 48 at 12 markets, 96 at 16, 121 at 20,
and between 345 and 482 from 25 to 40 markets. Every panel returned SCIP's
treated set, with an objective at or below SCIP's. From 25 markets on, the
search's time grows about as fast as the number of candidates, under a
microsecond each at forty markets; SCIP's follows its proof of optimality.
From 12 to 40 markets SCIP's median time grew 235-fold and the search's 29-fold,
though above 25 markets the search's grew faster and the speedup eased from its
peak. The script and its results are in
`benchmarks/studies/marex_exact_backend
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/marex_exact_backend>`_.

``res.study.backend`` records which backend ran, and ``res.globres.objective``
the design objective it reached, which is the number to compare across
backends. The two reach the same objective. Both report weights fitted to the
selected treated set, the search by an exact active-set fit and the MIQP by the
refit above, so on the test panels the two agree to within
:math:`5 \times 10^{-6}` in the weights and :math:`3 \times 10^{-9}` relative in
the objective. Where the objective is flat in some direction the optimal weights
are not unique, and the two can then return different weights at the same
objective.

.. _marex-certificate:

Certifying the search
^^^^^^^^^^^^^^^^^^^^^

The search prices and fits candidates in floating-point arithmetic, which rounds
every operation, so its design is optimal up to rounding that the search does
not bound. With ``certify=True`` (and ``backend="exact"``) the result also
carries ``res.certificate``, two numbers with

.. math::

   \texttt{lower} \;\le\; f^\star \;\le\; f(\text{design}) \;\le\; \texttt{upper},

where :math:`f^\star` is the smallest objective any admissible design attains
and :math:`f(\text{design})` is the returned design's, both evaluated exactly on
the stored data. ``res.certificate.gap`` is ``upper - lower``: the returned
design is within that distance of the optimum, whatever the rounding did.

.. code-block:: python

   res = MAREX(MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                           T0=T0, m_eq=4, backend="exact",
                           certify=True)).fit()
   res.certificate.lower, res.certificate.upper, res.certificate.gap

The construction is Neumaier and Shcherbina's (2004) rigorous bound for
linear programs: any dual point gives a valid bound, so a floating-point
solution needs only to be evaluated with its rounding under control. They use
interval arithmetic; the certificate uses exact arithmetic, and bounds in
advance the rounding of the one step it batches. It rests on three facts.

1. Exact arithmetic. Every finite double is an integer divided by a power of
   two, so the data, and any sum or product of them, can be held exactly as
   integers. The certificate computes its bounds as exact fractions and rounds
   only the final two numbers, ``lower`` down and ``upper`` up.

2. Bounds on one fit. Any weights on the simplex give an upper bound on
   :math:`g(\mathcal{S})`: the objective at those weights. With
   :math:`\mathbf{b}_j = \mathbf{x}_j - \bar{\mathbf{x}}` the centred predictor
   vectors and :math:`\mathbf{a}` the design's penalty (zero for
   ``standard``), any vector :math:`\mathbf{y}` gives a lower bound, the dual
   of the fit (in squared form, the bound of Wolfe 1976, Theorem 2.2).
   Expanding
   :math:`\|\sum_j w_j \mathbf{b}_j - \mathbf{y}\|_2^2 \ge 0` gives
   :math:`\|\sum_j w_j \mathbf{b}_j\|_2^2 \ge \sum_j w_j\, 2 \mathbf{y}^\top
   \mathbf{b}_j - \|\mathbf{y}\|_2^2`, and on the simplex a weighted average is
   at least its smallest term, so

   .. math::

      g(\mathcal{S}) \;\ge\; \min_{j \in \mathcal{S}}
        \bigl(2 \mathbf{y}^\top \mathbf{b}_j + a_j\bigr)
        - \|\mathbf{y}\|_2^2 .

   At :math:`\mathbf{y} = \sum_j w_j \mathbf{b}_j`, with :math:`\mathbf{w}` the
   fitted weights, the two bounds differ by the fit's own error and meet at the
   optimum. The certificate computes both exactly for the design's treated fit
   and control fit.

3. The search's bound, made rigorous. The dual bound holds for every
   :math:`\mathbf{y}`, so it can be taken at the closed-form point the search
   already computes, where it equals the closed form above. Evaluated in
   floating point it reads the data through the inner products
   :math:`\mathbf{G}_{\mathcal{S}}`, and each rounding on the way has an a
   priori bound (Higham 2002, sec. 3.1): the inner products, the subtraction of
   the target, and the products with the point. The certificate subtracts all
   of them and rounds down, so every candidate gets a bound the arithmetic
   cannot push above its fit, at about the cost of the search's own.

Per cluster, the certificate prices every admissible candidate again with that
bound plus a floor on its control fit: the control pool is part of the cluster,
and a fit over more markets is no worse, so :math:`h(\mathcal{S})` is at least
the exact lower bound of the fit over the whole cluster. A candidate priced at
or above ``upper`` cannot beat the design. The rest are fitted and bracketed
exactly, and a candidate whose exact lower bound still does not clear ``upper``
is listed in ``res.certificate.clusters[label].unresolved``, by unit label:
it might be as good as the design. A set that ties the design exactly is always
listed, and ``lower`` is the smallest exact lower bound among the design and
the listed sets. An empty list means every other treated set has an objective
above ``upper``.
The clusters are searched apart, so the certificate of the whole design sums
theirs.

``res.globres.objective`` is the floating-point value of the same design, and
it agrees with the bracket to within its own rounding. The ``weakly_targeted``
design is refused, since its coupling term ties the treated and control fits
together and the split the certificate relies on does not exist.

On GeoLift's 40 markets as one cluster (90 pre-period days, the ``standard``
design), the certificate confirmed the search's design for every treated count
from two to seven, with a relative gap of at most :math:`4 \times 10^{-13}`
and no unresolved set. The certified bound cleared all but at most four
candidates per count without an exact fit. It cost 1.7 to 1.95 times the search
from three treated up, and 2.4 times at two, where the whole search takes ten
milliseconds; at seven treated, 18.6 million candidates, the search took 27
seconds and the certificate 53.


.. _marex-branch-and-bound:

Large clusters: a certified branch and bound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eight treated among forty markets is 77 million treated sets, ten is 848
million, and the number grows as :math:`\binom{N}{m}`. Above twenty million sets
in a cluster, ``backend="exact"`` stops pricing every set and searches the
``standard``, ``penalized`` and ``unit_penalized`` designs by branch and bound.
Nothing changes at or under the limit, and nothing needs to be set: the search
returns the optimum the enumeration would, and with ``certify=True`` the
certificate of the previous section, which it produces as it goes.

A branch and bound organises the treated sets as a tree. A node fixes some
markets into the treated set (:math:`\mathcal{I}`) and some out of it
(:math:`\mathcal{O}`), and leaves the rest free (:math:`\mathcal{F}`); below it
lie the admissible treated sets of size :math:`m` that contain
:math:`\mathcal{I}` and avoid :math:`\mathcal{O}`. At each node the search does
one of two things. It prices the node's sets one at a time, exactly as the
enumeration prices a cluster, when there are few enough of them. Otherwise it
computes a bound, a number no design below the node can go under: if the bound
is above the best design found so far, the whole node is discarded unseen, and
if not, the node is split on one free market into a node that treats it and one
that does not. It bounds a node only when pricing the node's sets would take
longer than computing the bound, at about a microsecond per set and a few
seconds per bound at forty markets.

The bound has to see the products of the weights. Relaxing the choice of
treated markets to fractions between zero and one is not enough: spreading the
weights evenly over every market is an average of designs, and it scores
exactly zero, because the centred predictors sum to zero. The bound used here,
persp+N, works with the matrix of weight products. For each side of the design
(treated and control) it keeps the weights :math:`\mathbf{x}` on the markets
that side can use and a matrix :math:`\mathbf{X}` standing for
:math:`\mathbf{x}\mathbf{x}^\top`, with :math:`z_j` the fraction of market
:math:`j` that is treated, and requires

.. math::

   \begin{pmatrix} 1 & \mathbf{x}^\top \\ \mathbf{x} & \mathbf{X} \end{pmatrix}
   \succeq 0, \qquad \mathbf{X} \ge 0, \qquad
   \mathbf{X}\mathbf{1} = \mathbf{x}, \qquad \mathbf{1}^\top \mathbf{x} = 1,
   \qquad
   \begin{pmatrix} z_j & x_j \\ x_j & X_{jj} \end{pmatrix} \succeq 0,
   \qquad x_j \le z_j,

for every free market (with :math:`1 - z_j` in place of :math:`z_j` on the
control side), and :math:`\sum_{j \in \mathcal{F}} z_j = m - |\mathcal{I}|`.
The objective is :math:`\langle \mathbf{G}, \mathbf{X} \rangle +
\mathbf{a}^\top \mathbf{x}` per side. Every design meets every constraint with
:math:`\mathbf{X} = \mathbf{x}\mathbf{x}^\top`, so the minimum is a lower bound
on every design below the node. The 2x2 blocks are the perspective constraint
:math:`X_{jj} z_j \ge x_j^2` of Frangioni and Gentile (2006) and Günlük and
Linderoth (2010): a market carrying weight must be treated, and its weight
concentrates the second moments on few markets. With the semidefinite block they
form the relaxation of Dong, Chen and Linderoth as given in Wei, Gómez and
Küçükyavuz (2022, eq. 24), the primal of the best-diagonal program of Zheng, Sun
and Li (2014). :math:`\mathbf{X} \ge 0` is the entrywise half of the doubly
nonnegative relaxation of Burer's (2009) completely positive form, and it
carries most of the bound: dropping it from the full doubly nonnegative
relaxation took that bound from 99 percent of the optimum to 29.5 percent at
the root of GeoLift's tree with four treated.

A solver returns the bound in floating point, and the search does not take it on
trust. It reads the solver's multipliers, clips each to the sign its constraint
requires, and assembles the Lagrangian they define in exact arithmetic from the
exact Gram matrix of the data. Every design satisfies
:math:`\operatorname{tr}\begin{pmatrix} 1 & \mathbf{x}^\top \\ \mathbf{x} &
\mathbf{x}\mathbf{x}^\top \end{pmatrix} \le 2` on each side, so the bound is
the Lagrangian's constant, plus twice the smallest eigenvalue of its matrix when
that is negative, plus the :math:`m - |\mathcal{I}|` smallest coefficients of the
treated fractions. The smallest eigenvalue is bounded below exactly, from a
floating-point Cholesky factor of the shifted matrix and its exact residual.
This is the rigorous bound of Jansson, Chaykin and Keil (2007) for semidefinite
programs, with exact arithmetic in place of their interval arithmetic: any
multipliers give a valid bound, and accurate ones a tight one. A node whose
solve fails keeps the bound it had, and is split or priced like any other.

Every set the search discards is discarded on a certified lower bound above a
certified upper bound on a design already found, so it returns the
enumeration's optimum, and its certificate lists the sets that might tie it, as
the certificate above does. The restrictions carry over: forced and forbidden
units start the tree, a market treated at a node sends its conflict partners out
of the treated set, stratum quotas and forced units decide how many sets a node
holds, and every set priced passes the cluster's rules. Donor exclusions can
only raise a control fit, so every bound holds without them and every fitted
design honours them.

On one core, with GeoLift's 40 markets as one cluster over 90 days and
Walmart's 45 stores as the paper's design sees them, the search returns the
enumeration's design every time, certified with a relative gap of at most
:math:`1.5 \times 10^{-12}` and no unresolved set:

.. list-table:: The branch and bound against enumeration with its limit raised
   :header-rows: 1

   * - Panel
     - Treated
     - Treated sets
     - Branch and bound
     - Bounds computed
     - Enumeration
   * - GeoLift
     - 8
     - 76,904,685
     - 41.9 s
     - 7
     - 105.1 s
   * - GeoLift
     - 10
     - 847,660,528
     - 36.9 s
     - 8
     - 1,866.7 s
   * - Walmart
     - 6
     - 8,145,060
     - 14.0 s
     - 2
     - 7.5 s
   * - Walmart
     - 8
     - 215,553,195
     - 211.3 s
     - 22
     - 305.2 s

Below about ten million sets, enumeration is the faster of the two, which is why
it keeps every cluster under the limit. Through MAREX, GeoLift with ten treated,
which the exact backend refused before, takes 35.5 seconds with
``certify=True``. The runs are
`benchmarks/studies/marex_bnb/library_search.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/studies/marex_bnb/library_search.py>`_.

The search raises after a thousand bounds without closing its gap, giving the
bracket it reached and naming ``backend="mip"``. A bound's cost grows with
the cube of the markets on a side, so at a few hundred markets in one cluster
it is out of reach; clusters, restrictions or a smaller treated count bring each
search back within it. The weakly-targeted design couples the treated and
control fits, which the bound splits, so it is refused above the limit. The
tests are in ``mlsynth/tests/test_marex_branch.py``, and the study that chose
the bound, with the papers behind each piece, is in
`benchmarks/studies/marex_bnb
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/marex_bnb>`_.

Inference
^^^^^^^^^

When ``inference=True`` with ``blank_periods > 0``, the last few pre-experiment
periods are held out as blanks: there the synthetic treated minus synthetic
control is pure noise, so its distribution calibrates inference for the
post-period effect. MAREX reports a permutation p-value for the global null of
no effect, per-period p-values, and a confidence band from split-conformal
prediction (Lei et al. 2018; Vovk, Gammerman, and Shafer 2005) built on the
held-out blank periods -- rearranging only over the blank and post-intervention
periods, not over all periods as in Chernozhukov-Wuthrich-Zhu (2021) -- all on
:class:`~mlsynth.utils.marex_helpers.structures.MAREXInference`.

.. _marex-cumulative-band:

The total effect
^^^^^^^^^^^^^^^^

The band above bounds one period at a time. How much the intervention added
over the whole experiment is the total :math:`\sum_{t > T_0} \tau_t` over the
:math:`H` post periods, and its band is not arithmetic on the pointwise
endpoints: summing them assumes the period errors move together, and dividing
one by :math:`H` assumes they are independent. ``cumulative_band=True`` attaches
a band for the total to ``inference.cumulative_band``, on the aggregate design
and on every cluster, built in one of two ways chosen by ``cumulative_method``.
Both are mlsynth's own constructions: Abadie and Zhao's Theorem 3 covers one
period at a time, and their paper gives no band for the total.

.. code-block:: python

   res = MAREX(MAREXConfig(df=df, outcome="y", unitid="unit", time="time",
                           T0=T0, m_eq=3, inference=True, blank_periods=26,
                           cumulative_band=True,
                           cumulative_method="inversion")).fit()
   band = res.globres.inference.cumulative_band
   band.point, band.lower, band.upper

``point`` is MAREX's own estimate of the total, the sum of the post-period
effects, under both methods.

``cumulative_method="resample"``, the default, splits the blank window into
horizon-length sub-windows and resamples their effects into whole paths
(``cumulative_block`` sets the block length, ``cumulative_n_sim`` the number of
paths); the half-width is a quantile of the accumulated totals. It uses only
errors observed before the post period, and every band built that way meets a
floor set by how many of them there are. With :math:`n` blank errors, suppose
one error, placed at a period chosen at random, is far larger than the rest.
With probability :math:`H/(n+H)` it lands in the post window, where no blank
error reveals it. A band that covers the total for every exchangeable error law
(one whose distribution does not change when the periods are reordered) must
then be unbounded with probability at least :math:`1 - \alpha(n/H + 1)`. A band
that stays finite needs :math:`n \ge H(1/\alpha - 1)`, which is 247 blank
periods for a 95% band over 13 post periods.

``cumulative_method="inversion"`` is not subject to that floor because it reads
the post outcomes, which an experiment observes: they are the counterfactual
errors plus the effect. It inverts the test of Chernozhukov, Wüthrich and Zhu
(2021). For a candidate per-period effect :math:`\theta`, MAREX subtracts
:math:`\theta` from the synthetic treated series in each post period, refits a
synthetic control for that series on every period from the units the design
left untreated, and takes the residuals :math:`\hat u_t`. The statistic is the
size of their post sum,

.. math::

   S(\hat{\mathbf{u}}) = \frac{\bigl|\sum_{t > T_0} \hat u_t\bigr|}{\sqrt{H}},

and the p-value is the share of cyclic shifts of the residual path, taken over
the blank and post periods, whose post block gives a statistic at least as
large. A cyclic shift moves every residual :math:`k` periods along and wraps
the end around to the start, so it keeps the order in which errors follow one
another. The band is :math:`H` times the candidates with :math:`p > \alpha`, and
its endpoints need not be symmetric about ``point``.

Assumptions
"""""""""""

1. The errors are stationary and weakly dependent over the blank and post
   periods: their joint law does not drift, and periods far apart are close to
   independent.

   Remark. This is the condition under which cyclic shifts give an
   approximately valid test (Chernozhukov, Wüthrich and Zhu 2021, Assumption
   2.2). Shifts preserve the serial dependence that random reorderings would
   destroy, at the price of a coarser p-value: there are only
   :math:`T_b + H` of them, so no p-value falls below :math:`1/(T_b + H)`. With
   fewer than :math:`1/\alpha` blank and post periods nothing is rejected, the
   band is :math:`(-\infty, \infty)`, and MAREX warns. A 95% band needs at least
   20.

2. The null is a constant effect, and the band is read as a band for the total.

   Remark. A one-dimensional band needs a one-dimensional null, so the test
   imposes :math:`\tau_t = \theta` in every post period. The statistic depends
   on the residuals only through their sum, so its observed value depends on the
   effect path only through the total, and the band covers the total when the
   effect grows or decays as well. With the default statistic of the general
   routine, the mean absolute residual, that fails: on simulated MAREX designs
   with independent factors and a ramp-shaped effect the 95% band covered the
   total 11% of the time, against 99.7% for the signed sum. What breaks the signed sum is a single
   error so large that it dominates the post sum; heavy tails alone did not, in
   a search over Cauchy errors.

3. The fitting periods stay out of the reference set.

   Remark. MAREX chooses the treated set on the fitting periods, and the refit
   residuals there are smaller, by about 30% in mean square on the simulated
   designs, than on the blank and post periods, which behave alike. A reference
   that includes them understates the spread of a 13-period post sum, by a
   factor of 1.17 with independent factors and 1.59 with factors autocorrelated
   at 0.8, and the band under-covers. The refit still matches on every period,
   which lets it track a trend the controls share; only the reference starts at
   the blank window. Coverage of the total at 95% / 80%, with the median width
   of the 95% band, over 300 MAREX designs with 40 fitting, 26 blank and 13 post
   periods:

   .. list-table::
      :header-rows: 1
      :widths: 30 30 30

      * - factors
        - every period referenced
        - blank and post referenced
      * - independent
        - 0.950 / 0.773 (25.0)
        - 0.967 / 0.827 (37.5)
      * - autocorrelated at 0.5
        - 0.927 / 0.767 (30.0)
        - 0.973 / 0.787 (45.0)
      * - autocorrelated at 0.8
        - 0.910 / 0.730 (42.5)
        - 0.967 / 0.777 (60.0)
      * - trending
        - 0.983 / 0.833 (32.5)
        - 0.983 / 0.807 (38.8)

   Whitening the residuals by an autoregressive model before shifting them,
   which Chernozhukov, Wüthrich and Zhu propose for serially dependent errors,
   did not close the gap: the difference is between two stretches of the
   series, not a stationary dependence a filter can remove.

For a cluster, the refit draws on the cluster's own unselected members, the
units its control was built from; for the aggregate design, on every untreated
unit. Each candidate costs one simplex fit, and the search brackets and bisects
each endpoint, so the band costs some tens of fits per design and no extra
solve. ``cumulative_block``, ``cumulative_n_sim`` and ``cumulative_seed`` apply
to ``"resample"`` alone. The band is a
:class:`~mlsynth.utils.conformal.CumulativeInversionBand`; its
``reference_start`` is the first blank period and its ``n_reference`` is
:math:`T_b + H`. The study behind these choices, with the floor above and its
proof, is in `benchmarks/studies/marex_cumulative_conformal
<https://github.com/jgreathouse9/mlsynth/tree/main/benchmarks/studies/marex_cumulative_conformal>`_,
and the tests are in ``mlsynth/tests/test_marex_cumulative_inversion.py``.

Standardized Post-Fit and Power Analysis
----------------------------------------

Every call to :meth:`MAREX.fit` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` to ``res.post_fit``.
This is the single, estimator-agnostic surface for the diagnostic numbers a
consumer of the design typically needs: effects, fit RMSEs, conformal /
permutation inference, covariate balance (when covariates were used), and
power analysis. It is computed by
:func:`~mlsynth.utils.post_fit.compute_post_fit` from MAREX's own
``synthetic_treated`` / ``synthetic_control`` trajectories and weight vectors,
so by construction it agrees with what the underlying optimization produced.

.. code-block:: python

   pf = res.post_fit                          # SyntheticControlPostFit
   pf.ate, pf.ate_percent, pf.total_effect    # treatment-effect scalars
   pf.rmse_fit, pf.rmse_blank, pf.rmse_post   # fit quality, per phase
   pf.p_value, pf.ci_lower, pf.ci_upper       # inference (when computed)
   pf.covariate_smd                           # treated-vs-control SMD dict
   pf.covariate_smd_treated_vs_pop            # treated-vs-population
   pf.covariate_smd_control_vs_pop            # control-vs-population
   pf.power                                   # PowerAnalysis (see below)

Three Standardized Mean Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``covariates=[...]`` is set, the post-fit reports the three covariate
balance diagnostics that match the structure of Abadie & Zhao's objective.
Each is a per-covariate signed dict (``covariate_smd_*``) plus two summary
scalars (max absolute SMD, sum of squared SMDs). With :math:`\bar{\mathbf{x}}`
the population covariate aggregate,
:math:`\mathbf{x}_w \coloneqq \sum_j w_j \mathbf{x}_j`,
:math:`\mathbf{x}_v \coloneqq \sum_j v_j \mathbf{x}_j`, and :math:`s_m` the
cross-unit standard deviation of covariate :math:`m`, each comparison is the
unit-free vector

.. math::

   \mathrm{SMD}_m^{(a,b)} = \frac{x_a[m] - x_b[m]}{s_m}.

The three pairs ``(a, b)`` reported are:

* ``covariate_smd``                — :math:`(\mathbf{x}_w, \mathbf{x}_v)`: synthetic treated vs
  synthetic control. The internal-validity check ("is the experiment
  apples-to-apples?").
* ``covariate_smd_treated_vs_pop`` — :math:`(\mathbf{x}_w, \bar{\mathbf{x}})`: synthetic treated vs
  population aggregate. Tracks the first term of MAREX's objective,
  :math:`\|\bar{\mathbf{x}} - \sum_j w_j \mathbf{x}_j\|^2`. Tells you whether the *chosen
  treated group* represents the population.
* ``covariate_smd_control_vs_pop`` — :math:`(\mathbf{x}_v, \bar{\mathbf{x}})`: synthetic control vs
  population aggregate. Tracks the second term of the objective. Tells you
  whether the *control set* represents the population.

A rule-of-thumb threshold of :math:`|\mathrm{SMD}| < 0.1` is conventionally
"well balanced"; below :math:`0.25` is acceptable; above is a red flag.

Power Analysis and Minimum Detectable Effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Power analysis answers the pre-experiment planning question: *given the
design I've chosen, how large a treatment effect can I detect with high
probability?* This is the dual of inference: inference asks "is the observed
effect distinguishable from noise?", power asks "what effect sizes would be?"

The paper's inference is built to stay valid under serial dependence and
non-stationarity; staying squarely within that framework, ``mlsynth`` also
reports a planning-oriented minimum detectable effect -- an analytical,
AR(1)-inflated Gaussian MDE computed from the same residual series the
permutation test draws on. Set ``inference=True`` and the result auto-populates
``res.post_fit.power`` — ``blank_periods`` defaults to
``max(1, floor(0.3 * T0))`` so you do not need to pick a scalar yourself
(matching the LEXSCM / SYNDES / PANGEO convention).

Where the noise standard deviation comes from
""""""""""""""""""""""""""""""""""""""""""""""

Under the linear factor model of Assumption 1, the per-period contrast
:math:`g_t \coloneqq \sum_j w_j y_{jt} - \sum_j v_j y_{jt}` has expectation zero
under the no-effect null. Its sample SD on the blank window
:math:`\mathcal{B}` (the held-out tail of the pre-period) is the natural
estimator of the noise scale:

.. math::

   \widehat{\sigma}_{\text{placebo}} = \sqrt{\frac{1}{|\mathcal{B}| - 1}
       \sum_{t \in \mathcal{B}} \bigl(g_t - \bar g\bigr)^2}.

When no blank window is carved out (``inference=False``) the pre-period gap
serves as the placebo proxy. The blank-window estimator is preferred because
it uses periods that played no role in fitting the weights — it is honest in
exactly the same sense Chernozhukov-Wuthrich-Zhu's conformal residuals are.

Serial correlation matters
""""""""""""""""""""""""""

Synthetic-control gap residuals are virtually always serially correlated:
the donor weighting absorbs the level but the persistent components of the
factor structure (business cycles, seasonality, slow trends) leak through.
Ignoring this systematically under-states the SE at long horizons. We model
it as an AR(1) process with lag-1 autocorrelation

.. math::

   \widehat{\rho} = \frac{\sum_t g_t g_{t-1}}{\sum_t g_t^2},

clipped to :math:`(-0.99, 0.99)` for numerical safety. The variance of the
mean of :math:`T` consecutive AR(1) periods, expressed as a multiple of
:math:`\sigma^2`, is the *variance inflation factor*

.. math::

   \mathrm{VIF}(T, \rho) =
   \frac{1}{T}\!\left(1 + 2 \sum_{k=1}^{T-1}\!\Bigl(1 - \frac{k}{T}\Bigr)\rho^k\right),

which collapses to the textbook :math:`1/T` when :math:`\rho = 0` and grows
substantially for :math:`\rho > 0.3`. The same formula is used by PANGEO's
power module.

The MDE formula
"""""""""""""""

Combining: the standard error of the mean of :math:`T` post-period contrasts
under :math:`H_0` is :math:`\mathrm{SE}(T) = \widehat{\sigma}_{\text{placebo}} \,
\sqrt{\mathrm{VIF}(T, \widehat{\rho})}`. For a two-sided test at level :math:`\alpha`
with target power :math:`1 - \beta`, the minimum detectable effect is

.. math::

   \mathrm{MDE}(T) = \bigl(z_{1-\alpha/2} + z_{1-\beta}\bigr) \cdot
   \widehat{\sigma}_{\text{placebo}} \cdot \sqrt{\mathrm{VIF}(T, \widehat{\rho})}.

The corresponding power to detect a *given* true effect :math:`\tau` at
horizon :math:`T` is

.. math::

   \pi(\tau, T) = \Phi\!\Bigl(\frac{|\tau|}{\mathrm{SE}(T)} - z_{1-\alpha/2}\Bigr)
                 + \Phi\!\Bigl(-\frac{|\tau|}{\mathrm{SE}(T)} - z_{1-\alpha/2}\Bigr),

which is reported as ``power_at_observed`` for each horizon point using the
realised :math:`\widehat{\tau}`.

What the surface looks like
"""""""""""""""""""""""""""

.. code-block:: python

   p = res.post_fit.power                  # PowerAnalysis dataclass

   p.headline.mde_absolute                 # MDE at the realised T_post
   p.headline.mde_pct                      # ... as % of post-period baseline
   p.headline.se                           # implied SE of mean(g_t) over T_post
   p.headline.power_at_observed            # power to detect res.post_fit.ate

   p.curve                                 # tuple of MDEPoint, one per horizon
   for pt in p.curve:
       print(pt.post_periods, pt.mde_absolute, pt.mde_pct, pt.power_at_observed)

   p.sigma_placebo                         # σ̂ used (from blank or pre window)
   p.serial_correlation                    # ρ̂ AR(1) of the placebo gaps
   p.placebo_bias                          # mean placebo gap; 0 for an unbiased design
   p.placebo_bias_pvalue                   # is that offset distinguishable from 0?
   p.n_placebo, p.sigma_ci                 # periods σ̂ rests on, and its interval
   p.baseline                              # mean(synthetic_control) on post window
   p.alpha, p.power_target                 # 0.05 / 0.80 by default
   p.method                                # "analytical_ar1_mean_gap"

   pt = p.headline
   pt.critical_value                       # t quantile the MDE is built at
   pt.mde_ci                               # MDE at the ends of p.sigma_ci

Test an observed effect against ``pt.critical_value``, not against 1.96. The
noise scale is estimated from the placebo window, so the pivot follows a
Student-t on that window's effective degrees of freedom; the Gaussian quantile
gives a nominal 5% test a true size above 10% on a short, persistent window.

The default horizon grid covers :math:`T \in \{1, 2, 4, 6, 8, 12\}` plus the
realised ``n_post``, so the table also doubles as a *"how long do I need to
run?"* answer — pick the smallest :math:`T` whose MDE drops below your
target effect size.

Practical reading
"""""""""""""""""

A typical MAREX run with ``T_post = 6``, ``blank_periods = 4`` and modest
serial correlation (:math:`\widehat{\rho} \approx 0.5`) on a Walmart-style sales
panel produces an MDE on the order of 0.05–0.15% of mean sales, well
below the 1–3% effect sizes typical marketing interventions aim for; this
is the quantitative substance of *"good designs are well-powered"*.
Conversely, an MDE much above the expected effect is a signal the design
needs more units (lower ``m_eq``/``m_max`` are typically *worse* for power)
or more post-periods (extend the experiment).

Opting out
""""""""""

The power computation is wrapped in a ``try/except`` in
:func:`~mlsynth.utils.marex_helpers.orchestration.solve_marex` — a power
analysis failure (e.g. degenerate residual variance) never breaks the fit,
``res.post_fit.power`` is just left as ``None``. To compute power on a
non-default horizon grid or significance level, call the free function
directly:

.. code-block:: python

   from mlsynth.utils.post_fit import compute_power_analysis

   alt = compute_power_analysis(
       res.post_fit, alpha=0.10, power_target=0.90,
       post_grid=[2, 4, 8, 16, 32, 52],     # weekly horizons out to a year
   )

Monte Carlo: Recovering the Treatment Effect
--------------------------------------------

The block below replicates the qualitative finding of the paper's simulation
study (Section 5) using ``mlsynth``'s own reimplementation of the linear-factor
DGP. A sample is drawn, the design is fit on the pre-period, the treated units
realise :math:`y^I` in the experiment, and the estimate is compared to the true
average effect.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mlsynth import MAREX
   from mlsynth.utils.marex_helpers.simulation import generate_marex_sample

   rng = np.random.default_rng(0)

   def design_mae(sample, **card):
       J, T = sample.Y_N.shape
       T0 = sample.T0
       df = pd.DataFrame(
           [{"unit": f"u{j}", "time": t, "y": float(sample.Y_N[j, t])}
            for j in range(J) for t in range(T)]
       )
       res = MAREX({"df": df, "outcome": "y", "unitid": "unit",
                    "time": "time", "T0": T0, **card}).fit()
       w = res.globres.treated_weights_agg
       v = res.globres.control_weights_agg
       treated = np.where(w > 1e-8)[0]
       Y_obs = sample.Y_N.copy()
       Y_obs[treated, T0:] = sample.Y_I[treated, T0:]   # experiment realises Y^I
       tau_hat = w @ Y_obs[:, T0:] - v @ Y_obs[:, T0:]
       return np.mean(np.abs(tau_hat - sample.tau[T0:]))

   maes, scales = [], []
   for _ in range(5):
       s = generate_marex_sample(J=12, T=30, T0=25, rng=rng)
       maes.append(design_mae(s, m_min=1, m_max=11))     # Unconstrained
       scales.append(np.mean(np.abs(s.tau[s.T0:])))
   print(f"MAE {np.mean(maes):.2f}  vs  effect scale {np.mean(scales):.2f}")

The synthetic-control design recovers the average treatment effect with mean
absolute error far below the effect's own scale (≈ 4.4 vs. ≈ 14, i.e. under a
third), the central message of the paper's Table 2. Over the paper's full 1000
simulations the error also *decreases* as more units are allowed into the
treated group (the Unconstrained design is best), with the largest gains moving
from one to two or three treated units.

.. note::

   This is a Path-B replication: it reproduces the simulation study's
   conclusions from public DGPs and ``mlsynth`` code, with no dependency on the
   authors' replication package. It is locked in as
   :mod:`mlsynth.tests.test_marex_replication`.

Empirical Application: Walmart (Placebo Experiment)
---------------------------------------------------

We replicate the paper's empirical illustration (Section 4) on the Walmart
store-sales panel (``basedata/walmart_weekly_sales.csv``): weekly sales for
45 stores over 143 weeks (Feb 2010 – Oct 2012). Following the paper, we
design a placebo experiment with a fictitious intervention at week 129:
:math:`T_0 = 128` pre-experiment weeks, of which the first :math:`T_E = 100` are
the fitting period and the last 28 are blank, leaving 15 experimental weeks. The
design uses the constrained formulation with :math:`m = 2` treated stores,
uniform weights, and predictors normalised to unit variance (``standardize``).

.. code-block:: python

   import pandas as pd
   from mlsynth import MAREX

   df = pd.read_csv(
       "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
       "refs/heads/main/basedata/walmart_weekly_sales.csv"
   )

   res = MAREX({
       "df": df, "outcome": "sales", "unitid": "store", "time": "week",
       "T0": 128, "blank_periods": 28, "T_post": 15,   # TE=100, 28 blank, 15 post
       "m_eq": 2,                  # constrained design, two treated stores
       "design": "standard",
       "standardize": True,        # unit-variance predictors (paper's normalisation)
       "inference": True,
       "display_graph": True,
   }).fit()

   print("treated stores:", res.treated_units)              # [1, 15]
   print("placebo p-value:", round(res.globres.inference.global_p_value, 3))

Because the intervention is a placebo (no real effect), a correct design should
produce synthetic treated and control units that track closely and an estimated
effect near zero. ``mlsynth`` reproduces exactly that — and the paper's headline
number:

.. list-table:: Walmart placebo design (m = 2)
   :header-rows: 1
   :widths: 38 30 30

   * - Quantity
     - ``mlsynth``
     - Paper (Section 4)
   * - Pre-fit RMSE / mean sales
     - 2.2%
     - small (close tracking)
   * - Experimental ATT / mean sales
     - -1.0%
     - near zero
   * - Placebo permutation p-value
     - 0.937
     - 0.933
   * - Confidence band covers zero
     - yes (all post weeks)
     - yes

The synthetic treated and control units track to within ~2% of mean sales over
the fitting *and* blank periods, the estimated placebo effect is ~1% of sales,
and the permutation test fails to reject the null of no effect
(:math:`p = 0.937`, matching the paper's :math:`0.933`) — exactly the
"no spurious effect" result a good design should deliver on a placebo.

.. note::

   This uses the exact MIQP (``relaxed=False``, the default) with
   ``standardize=True``; the unit-variance normalisation is essential here
   because Walmart stores differ enormously in sales level, and without it the
   level differences dominate the match. The solve takes roughly a minute with
   the open-source SCIP solver (the paper used commercial Gurobi). With
   ``backend="exact"`` the whole fit takes about 0.03 seconds and returns the
   same two stores (:ref:`marex-search-backend`).

.. note::

   Cross-validated against the authors' code, without Gurobi. The durable
   benchmark ``benchmarks/cases/marex_walmart.py`` runs this design on the full
   45-store panel, additionally matching on the four store-level covariates the
   reference data carries (temperature, fuel price, CPI, unemployment -- the R
   code's "few covariates" configuration), and compares it cell-by-cell against a
   live run of SCDesign's own design code. The authors' published routine is a
   Gurobi non-convex MIQP, but their *constrained* (cardinality-:math:`K`)
   design -- ``Synthetic_Experiment_Cardinality_Constraint``, which enumerates
   the partitions of size :math:`\le K` and solves each treated/control synthetic
   control through the open ``quadprog`` path -- is exactly the design ``m_eq``
   solves and needs no commercial solver. MAREX (SCIP) and SCDesign (``quadprog``)
   select the same two treated stores with treated weights agreeing to
   :math:`2\times10^{-4}`, the same pre-period fit, and the same placebo effect to
   :math:`10^{-4}` of mean sales; see ``benchmarks/reference/marex_walmart/`` for
   the captured reference run.

Correspondence with the Authors' Code
-------------------------------------

The authors' R replication code (``Random_Data_Generator.R``,
``Synthetic_Experiments.R``, ``Different_optimization_methods.R``) maps directly
onto ``mlsynth``'s implementation, which was checked against it:

.. list-table:: Authors' R ↔ ``mlsynth`` MAREX
   :header-rows: 1
   :widths: 42 42

   * - Authors' R
     - ``mlsynth``
   * - DGP (``Random_Data_Generator.R``)
     - :func:`~mlsynth.utils.marex_helpers.simulation.generate_marex_sample`
   * - Formulation (5), Gurobi non-convex QCQP
     - ``design="standard"`` (MIQP with binary ``z``; same optimum)
   * - Penalization formulation
     - ``design="penalized"`` (identical :math:`\lambda` distance penalty)
   * - Cardinality formulation
     - ``m_eq`` / ``m_min`` / ``m_max``
   * - Cardinality routine (every partition, ``quadprog`` on each side)
     - ``backend="exact"`` (the same enumeration, with a bound that skips most
       of the fits)
   * - Predictors :math:`\mathbf{x} = [\mathbf{y}^E ; \mathbf{z}]`
     - ``covariates=[...]`` (matched on pre-outcomes + covariates)
   * - "treated = smaller set" swap
     - applied where either labelling is optimal; not where ``m_eq``, a cost
       bound or an asymmetric objective already settles it (see below)
   * - Exact permutation test (sum statistic)
     - permutation inference (mlsynth defaults to a mean statistic / sampled
       permutations)

The reference's labelling rule is kept, and only where it applies. The authors'
base program in ``SCdesign_LazyRun.R`` is two simplex rows and the disjointness
:math:`\mathbf{w}'\mathbf{v} = 0`, solved as a continuous non-convex QCQP
(``vtype = 'C'``) with no selection variable at all. Exchanging the two weight
vectors is feasible there at the same cost, so the labelling is free and
"treatment regions are the smaller set of regions" is the tie-break that
program needs. Their cardinality routine enumerates treated sets of every size
from 1 up to :math:`K`, defaulting to :math:`\lfloor N/2 \rfloor`, so the
treated group is never the majority there either.

``mlsynth``'s program can be either case, so the rule is applied on a test of
which. Swapping :math:`\mathbf{w}` and :math:`\mathbf{v}` is feasible at the
same cost when the objective is symmetric in them (the ``standard`` design),
no exact ``m_eq`` is set, the complement's size also satisfies ``m_min`` and
``m_max``, and there is no cost bound or geographic restriction -- those price
and restrict :math:`\mathbf{w}` and :math:`\mathbf{z}` alone. That case is
ordinary: ``m_min=1, m_max=N-1`` is exactly it, and it is the cell the Section 5
Monte Carlo compares against the authors' numbers. There the convention decides,
as it does for them.

Where the program does settle the labelling it is read off :math:`\mathbf{w}`,
the variable the program constrains. Applying the convention there instead
returned three treated markets for ``m_eq = 6`` on twelve, and they were the
control synthetic's markets -- the group :math:`\sum_j z_j = m` does not
constrain and :math:`\sum_j c_j w_j \le B` does not price. It also negated the
estimated effect, which is the treated synthetic minus the control one.

Driving ``mlsynth``'s ``solve_design`` on the authors' exact DGP and predictor
matrix recovers the average treatment effect to within its scale, and the effect
estimate degrades gracefully as the noise SD rises from 1 to 5 to 10 (the
figures 2-7 settings) — matching the paper's qualitative findings.

.. note::

   Two faithfulness details from their R code: ``rnorm(N, 0, noise.variance)``
   passes the value as a *standard deviation*, so the figures' "variance"
   1/5/10 are SDs; and the R code uses random population weights :math:`f_j`
   whereas the 2026 paper (and ``mlsynth``) use :math:`f_j = 1/J`. Also,
   the *unconstrained* ``standard`` design (formulation 5) is degenerate —
   many disjoint splits match :math:`\bar X` equally well, so the realised
   design (and hence a single ATE estimate) is solver-dependent; the
   cardinality-constrained design is the stable, recommended choice.

Example
-------

.. code-block:: python

   import pandas as pd
   from mlsynth import MAREX

   # long panel: one row per (market, period)
   res = MAREX({
       "df": df, "outcome": "revenue", "unitid": "market", "time": "week",
       "T0": 40,            # 40 pre-experiment weeks
       # Equivalently: pass a 0/1 column marking the experiment window
       # "post_col": "in_experiment",
       "m_eq": 2,           # treat exactly two markets
       "design": "standard",
       "inference": True,   # blank_periods defaults to floor(0.3 * T0) = 12
       "display_graph": True,
   }).fit()

   print("treated markets:", res.treated_units)
   print("global p-value:", res.globres.inference.global_p_value)
   for label, c in res.clusters.items():
       print(label, c.unit_weight_map["Treated"])

Verification
------------

Validated against Abadie & Zhao's Section 4 Walmart application (their reference
code is `jinglongzhao2/SCDesign <https://github.com/jinglongzhao2/SCDesign>`_):
on a 10-store subset of ``walmart_weekly_sales.csv`` MAREX's exact MIQP designs a
placebo experiment that tracks closely pre-period (pre-fit RMSE ~2.7% of mean
sales, matching LEXSCM) and yields a placebo effect indistinguishable from zero
(~1% of mean, CI covering zero) -- the paper's "no spurious effect" result. This
is an independent commit-stamped check (MAREX's own optimizer) complementing the
LEXSCM Walmart benchmark. See :doc:`replications/marex`; run it with
``python benchmarks/run_benchmarks.py marex_walmart``.

The exact backend (``backend="exact"``) is checked separately against a
brute-force enumeration of every treated set in R, across every design and
objective option it solves: `benchmarks/cases/marex_exact_bruteforce.py
<https://github.com/jgreathouse9/mlsynth/blob/main/benchmarks/cases/marex_exact_bruteforce.py>`_,
described in :doc:`replications/marex`.

.. note::

   The benchmark uses the exact MIQP (free SCIP), not the relaxed
   continuous-``z`` mode: the relaxation shares the design objective but drops the
   integrality that defines the selection, so its top-``m`` rounding is degenerate
   and non-deterministic for small treated counts. The authors' full 45-store
   MIQP uses Gurobi, so the validator is Path A on a subset, not a live R
   cross-validation.

Core API
--------

.. automodule:: mlsynth.estimators.scexp
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: mlsynth.config_models.MAREXConfig
   :members:
   :undoc-members:

Result Containers
-----------------

``MAREX.fit()`` returns a
:class:`~mlsynth.utils.marex_helpers.structures.MAREXResults`: a dict of
per-cluster :class:`~mlsynth.utils.marex_helpers.structures.MAREXClusterDesign`
objects, the aggregate
:class:`~mlsynth.utils.marex_helpers.structures.MAREXGlobalDesign`, the
:class:`~mlsynth.utils.marex_helpers.structures.MAREXStudy` hyperparameters, and
(optionally) :class:`~mlsynth.utils.marex_helpers.structures.MAREXInference`.

.. note::

   ``MAREX.fit()`` returns a :class:`~mlsynth.config_models.DesignResult` (the
   experimental-design family, *not* an ``EffectResult``): MAREX designs an
   experiment, so it exposes the standardized design surface --
   ``res.report`` (the realized effect as an
   :class:`~mlsynth.config_models.EffectResult`, the single source for ATT / CI
   / pre-fit; ``res.report.att`` / ``res.report.counterfactual`` / ...),
   ``res.selected_units`` / ``res.assignment`` (treated vs control),
   ``res.design_weights``, and ``res.power``. The MAREX-specific design detail
   stays on ``res.clusters`` / ``res.study`` / ``res.globres`` / ``res.post_fit``
   (the same ``SyntheticControlPostFit`` that ``res.report`` is built from).

.. automodule:: mlsynth.utils.marex_helpers.structures
   :members:
   :undoc-members:
   :show-inheritance:

In addition, ``MAREX.fit()`` attaches a
:class:`~mlsynth.utils.post_fit.SyntheticControlPostFit` as
``results.post_fit``: the standardized diagnostics container shared across the
MAREX family (LEXSCM, MAREX, SYNDES, PANGEO). It carries the ATE / total /
lift / per-period / cumulative effect summaries, the inference triple
(:math:`p`, CI), the pre-/blank-/post-period RMSEs, the three
standardized-mean-difference blocks (treated-vs-control, treated-vs-population,
control-vs-population), and — when a valid noise window exists — a
:class:`~mlsynth.utils.post_fit.PowerAnalysis` block with the headline MDE and
the MDE-versus-horizon curve.

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

.. automodule:: mlsynth.utils.marex_helpers.setup
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.formulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.optimization
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.inference
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.orchestration
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.simulation
   :members:
   :undoc-members:

.. automodule:: mlsynth.utils.marex_helpers.plotter
   :members:
   :undoc-members:

The shared post-fit module — :func:`~mlsynth.utils.post_fit.compute_smd`,
:func:`~mlsynth.utils.post_fit.compute_post_fit`, and
:func:`~mlsynth.utils.post_fit.compute_power_analysis` — lives outside the
``marex_helpers`` package so the other MAREX-family estimators (LEXSCM,
SYNDES, PANGEO) can call into the same one-source-of-truth diagnostics:

.. automodule:: mlsynth.utils.post_fit
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."
See [ABADIE2024]_.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
for Comparative Case Studies." *Journal of the American Statistical
Association* 105(490):493-505.

Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). "An Exact and Robust
Conformal Inference Method for Counterfactual and Synthetic Controls."
*Journal of the American Statistical Association* 116(536):1849-1864.

Vives-i-Bastida, J. (2022). "Synthetic Experimental Design for a UBI Pilot
Study." Report, MIT.

Abadie, A., & L'Hour, J. (2021). "A Penalized Synthetic Control Estimator for
Disaggregated Data." *Journal of the American Statistical Association*
116(536):1817-1834.

Wolfe, P. (1976). "Finding the Nearest Point in a Polytope." *Mathematical
Programming* 11(1):128-149.

Frank, M., & Wolfe, P. (1956). "An Algorithm for Quadratic Programming."
*Naval Research Logistics Quarterly* 3(1-2):95-110.

Neumaier, A., & Shcherbina, O. (2004). "Safe Bounds in Linear and
Mixed-Integer Linear Programming." *Mathematical Programming* 99:283-296.

Burer, S. (2009). "On the Copositive Representation of Binary and Continuous
Nonconvex Quadratic Programs." *Mathematical Programming* 120:479-495.

Frangioni, A., & Gentile, C. (2006). "Perspective Cuts for a Class of Convex
0-1 Mixed Integer Programs." *Mathematical Programming* 106:225-236.

Günlük, O., & Linderoth, J. (2010). "Perspective Reformulations of Mixed Integer
Nonlinear Programs with Indicator Variables." *Mathematical Programming*
124:183-205.

Jansson, C., Chaykin, D., & Keil, C. (2007). "Rigorous Error Bounds for the
Optimal Value in Semidefinite Programming." *SIAM Journal on Numerical
Analysis* 46(1):180-200.

Wei, L., Gómez, A., & Küçükyavuz, S. (2022). "Ideal Formulations for
Constrained Convex Optimization Problems with Indicator Variables."
*Mathematical Programming* 192:57-88.

Zheng, X., Sun, X., & Li, D. (2014). "Improving the Performance of MIQP Solvers
for Quadratic Programs with Cardinality and Minimum Threshold Constraints: A
Semidefinite Program Approach." *INFORMS Journal on Computing* 26(4):690-703.

Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd
ed. SIAM.

# Roadmap: carrying the supergeo study into PANGEO

What this supergeo study found, what Trimmed Match Design (Chen, Longfils and Remy,
arXiv 2105.07060) adds, and the sequence of changes to `mlsynth.PANGEO` that
follows. Each phase is its own branch and scope, per `CLAUDE.md`: tests first,
a `hypothesis` property per invariant, a mutant per defect it could reintroduce,
and a measurement that decides whether it ships. Benchmark cases are a separate
workstream (phase 7).

## Where PANGEO stands

- Objective: per supergeo pair (halves A and B of at most Q markets each), the
  DiD residual sum of squares of the mean trajectories,
  `min over splits sum_t [(Ybar_A - Ybar_B)_t - delta]^2`, plus an optional
  covariate SMD^2 term. A per-group cost, so a set-partitioning master.
- Default solver `fast=True`: level-removed trajectories, PCA, hierarchical
  clustering into size-bounded groups, best split per group (an OSD analog).
  For PANGEO's objective this is the right structure, since parallel trends asks
  for similar shapes; the study's finding that similarity clustering fails
  applies to the pretest-total objective, not here. It returns no gap.
- Exact solver `fast=False`: every admissible group of size 2..2Q enumerated
  (O(n^{2Q})), then an exact-cover MIP through highspy or cvxpy with default
  options.
- Power: closed-form MDE from the pairs' pre-period residual variance with an
  AR(1) inflation, the quantity the MIP minimised. A blank window (`frac_E`)
  holds out the tail of the pre-period, and a held-out residual is computed.

## What the study and the paper establish

From this study (base problem, groups of 2 to 4; numbers in `README.md`):

1. HiGHS's "Optimal" is optimal to about 1e-6; it was wrong on 17 of 25 windows
   of one panel. Its default relative gap of 1e-4 stops at a warm start that
   close. Its presolve returned "Solve error" on a feasible master (fixture
   `presolve_failure_master.json`). Fixed artificial prices fail after a cost
   rescale.
2. Column generation with exact pricing replaces enumeration; reduced-cost fixing
   leaves 422 to 3,284 of 251,125 groups at 50 markets.
3. The staged exact-cover search over the survivors, with the elimination redone
   at each better incumbent, proves 50-market optima in 8.0 s and 20.3 s and
   reaches 80 markets (566 s); branch and bound needed 260 s at 50.
4. The pretest optimum overfits: its imbalance sits far below the noise level.

From Trimmed Match Design:

5. Pair on one pretest window and evaluate on another (cross validation);
   evaluating on the pairing window under-estimates RMSE.
6. Power by simulation: random within-pair assignment, the estimator's RMSE on
   the evaluation window, minimum detectable effect RMSE (q_{1-alpha} + q_beta).
7. Rerandomization with balance checks; optional geo exclusion (optimal subset
   pairing with pseudo geos); Trimmed Match as the outlier-robust estimator.

## Solver options (decided 2026-11-22)

The author's decisions on PANGEO's solver options, which phases 2 to 4 build
against:

1. Shape. PANGEO takes TBR's shape: `search: "exact" | "heuristic"`, with
   `exact_gap` and `exact_time_limit`. Behind `"exact"` the library chooses the
   method, closed form first (CLAUDE.md invariant 8): the sorted pairing, then
   the blossom matcher, then the certified staged search. The solver
   diagnostics' `path` records which ran. A search stopped by the time limit
   returns its incumbent and the proven gap. The heuristic is the search's
   starting incumbent; with a stated gap there is no case for returning it as
   the answer, and phase 4 measures whether `"heuristic"` keeps a reason to
   exist at all.
2. `fast` and `mip_backend` are deprecated when the shape lands: accepted for
   one release with a warning that maps them onto it, then removed.
   `mip_backend="cvxpy"` maps to nothing. A cross-check route for tests and
   benchmarks stays as a private helper, not an option.
3. No HiGHS. The goal is a PANGEO with no `highspy` (and no cvxpy) dependency.
   The study already has the parts: `lpsx.py` (a warm-started dual simplex in
   numba) for the column-generation master, whose bound stays valid whatever
   LP solver produced the duals because the certificate re-checks it in exact
   arithmetic, and the exact-cover search (`exactnb`, `verify_nb`) for the
   final proof, which `certify_staged` already uses in place of the MIP. The
   gate for `lpsx`: the same bound as HiGHS on the study's master problems at
   comparable time, the presolve-failure fixture included.
4. Pairs under `fast=True` stay on the clustering heuristic for now
   (jgreathouse9/mlsynth#1115 moved only the exact path); they move with the
   shape in item 1.

## Phases

### Phase 0: measure PANGEO as it is (no code change)

On the 50-state panel, GeoLift, and simulated panels at 20 to 80 markets, Q = 2:
time and status of `fast=False`; the `fast=True` design's cost; and the proven
gap of the fast design, from a study-side port of the certificate to PANGEO's
score. Decides what phases 2 to 4 can gain: if the fast design sits within
noise of the optimum on real panels, phase 4 is a certificate, not a better
design.

### Phase 0b: the supergeo objective in PANGEO, solved the base way

PANGEO's objectives (`ss_res`, `r2`, `weighted`) all score the level-removed gap
between the halves' mean trajectories; none is the objective of Chen, Doudchenko,
Jiang, Stein and Ying (2023), which this study optimises. A comparison of the
study's methods with PANGEO is fair only on one objective, so PANGEO first gets
the paper's, solved by the route a practitioner would build from the paper:
enumerate every group, then an exact-cover MIP through cvxpy with HiGHS. Phases
1 to 4 are then measured against that base, and the difference is what the
research buys (time, scale, proven optimality), with the objective held fixed.

- Objective: `objective="supergeo_total"`, score(G) = min over non-empty proper
  A of (sum_{i in A} z_i - sum_{i in G minus A} z_i)^2, z_i market i's total over
  the estimation window (`frac_E`, the same window for every arm). Sums, not
  means (the paper's variance under a fixed budget is a sum over supergeos of
  the squared difference of the halves' totals); no level shift delta (the
  level is what the objective balances); no trajectory.
- Group size: the cap is on the group (2 to 4 markets, any split, one against
  three included), not on each half (PANGEO's Q). A test pins that a group of
  four may split one against three under this objective.
- Solver: `fast=False` enumerates the groups and solves the exact-cover MIP with
  cvxpy and HiGHS (`pangeo_helpers/mip.py`), PANGEO's options as they are, with a
  stated time limit. The base runs as shipped: its default relative gap (about
  1e-4), the presolve failure on a feasible master (`presolve_failure_master.json`)
  and HiGHS's optimality to about 1e-6 are recorded on the comparison, not
  fixed here (phase 1 fixes them).
- `fast=True` with this objective raises: PANGEO's fast path removes levels and
  clusters similar trajectories, which suits its own objective, and the study
  found similarity clustering fails for this one (good groups join large and
  small markets). The heuristic for this objective comes with phase 3.
- Defaults unchanged: a test pins `objective="ss_res"` and its designs on the
  existing fixtures.
- The comparison (study side, one session, same machine and threads): the 50-
  state panel, GeoLift and simulated panels at 20 to 180 markets; for each arm
  the time, the status, the design's cost, the proven gap (the base reports
  HiGHS's; the study's certificate reports its exact one), the exact cost of
  the base's design (re-costed in exact arithmetic), and where the base stops
  (enumeration memory or the time limit: about 8.5 million groups at 120
  markets and 43 million at 180). Then both designs and PANGEO's own on held-out
  windows: the variance of the effect estimate (phase 5) and the RMSE with
  injected effects, each design analysed by both estimators.
- Tests first: the new score against brute force on small panels (property),
  the MIP's design against brute force over partitions, failure tests for
  `fast=True` and for bad group sizes, mutants for the score (sums, no delta,
  the group cap).

Two steps, simplest first:

- 0b-1, pairs. Groups of exactly two (PANGEO's Q = 1), the objective above,
  solved by the cvxpy and HiGHS MIP. Pairs are not a partitioning problem: with
  the paper's scalar objective the optimal pairing pairs neighbours in sorted
  order (for sorted a <= b <= c <= d the crossing pairing costs at least
  (b - a)^2 + (d - c)^2 since each difference only grows, and the nested one
  (d - a)^2 + (c - b)^2 >= (b - a)^2 + (d - c)^2; an exchange argument extends
  it to n), with an O(n) dynamic program choosing the market left out when n is
  odd; with any per-pair cost (trajectories, PANGEO's DiD score) it is a minimum
  weight perfect matching, polynomial (Edmonds' blossom algorithm). The MIP's
  design must equal the sorted pairing: an exact oracle for the new objective and
  the cvxpy and HiGHS path. Trimmed Match Design is a pairs design, so this step
  is also cross-validated against google/trimmed_match's pairing on its example
  data (phase 7), the one external reference the comparison has.
- 0b-2, groups of 2 to 4, any split: the case that contains number
  partitioning, and the base against which phases 1 to 4 are measured.

The comparison then has three arms: pairs (polynomial), groups of 2 to 4 by the
MIP, groups of 2 to 4 by the study's certificate; it shows what supergeos buy
over pairs before it shows what the research buys over the MIP.

Own branch, library code; lands before phase 1.

### Phase 0c: the general solver on the closed-form case

With pairs (groups of exactly two) the paper's objective has a closed form: the
markets paired as neighbours in sorted order of their totals are optimal (an
exchange argument: for sorted a <= b <= c <= d the crossing pairing costs at least
(b - a)^2 + (d - c)^2, since each difference only grows, and the nested one
(d - a)^2 + (c - b)^2 >= (b - a)^2 + (d - c)^2). With any other per-pair cost
(trajectories, PANGEO's DiD score) pairs are a minimum weight perfect matching,
polynomial. Pairs are therefore the instance on which the general machinery's
overhead can be measured against a known answer and its known cost, before the
roadmap continues.

Measured on google/trimmed_match's 174 example geos (totals over 2020-01-01 to
2020-12-01), pairs only, one session, all at the exact optimum:

| route | time |
| --- | --- |
| sorting (Trimmed Match's create_geo_pairs) | 0.04 s |
| PANGEO, supergeo_total, cvxpy and HiGHS MIP over 15,051 candidate pairs | 0.76 to 1.24 s (MIP 0.48 to 0.80 s) |
| the study's certificate, lo = hi = 2, defaults | 5.89 to 7.33 s, of which the heuristic 5.4 to 5.7 s |
| the same, heuristic off | 0.37 to 0.41 s (root 0.21 s; proof 8,742 nodes) |

The default heuristic runs a fixed N/2 steps (87) although the sorted start is
already optimal; the proof itself is cheap because the start is optimal, and the
root bound is 0.32 of the optimum. Three steps, each general, each measured on
the known-answer families below and on groups of 2 to 4 so the hard case does
not slow:

1. Stop the heuristic when it stops paying: after k steps without improvement,
   or once the bound proves the gap asked for, counted in steps so the design
   stays a function of the inputs and the seed. On pairs it stops at once; on
   the 180-market panel the fixed budget was a third of the run, and on the 174
   geos with groups of 2 to 4 it took 7.4 s.
2. Tighten the bound with odd-set (blossom) inequalities. With pairs the master
   is the perfect matching LP, and Edmonds' blossom inequalities make it exact;
   Chandrasekaran, Vegh and Vempala ("The cutting plane method is polynomial for
   perfect matchings", Math. Oper. Res. 41 (2016) 23-48,
   doi:10.1287/moor.2015.0714) show the cutting plane loop converges in
   polynomial time when (a) violated blossoms are read off the odd cycles in the
   support of the half-integral LP solutions (no Padberg-Rao separation needed
   while solutions stay half-integral) and (b) only the cuts with positive duals
   are kept, a laminar family. Blossom inequalities have Chvatal rank 1, so the
   study's rank-1 Chvatal-Gomory cuts on 3 to 5 rows (rank1.py) are small
   blossoms already, and the parity cut for odd N is the blossom on the whole set;
   odd-cycle separation extends them to odd sets of any size, and pricing
   subtracts the duals of the cuts a group crosses. Target: the pairs bound equal
   to the optimum at the root, so the certificate ends without a search. Open
   question: how far such cuts lift the bound with groups of 2 to 4, where a
   triple covers an odd set and the argument weakens.

   Which cuts the pricing can carry. The master's columns are generated, so a
   cut is robust or non-robust in the sense of Uchoa, Pessoa and Moreno
   ("Optimizing with Column Generation: Advanced Branch-Cut-and-Price
   Algorithms", Part I, Cadernos do LOGIS 2024(3), section 4.3): a cut stated on
   the markets (the rows) gives a new group a coefficient that depends only on
   how many of the cut's markets it covers, and the scan prices it cheaply; a cut
   stated on the groups (the columns) needs the pricing to decide a new group's
   coefficient from the groups already in the cut. Borndörfer ("Aspects of Set
   Packing, Partitioning, and Covering", dissertation, TU Berlin, 1998, section
   3.3) separates clique, odd-cycle and aggregated odd-cycle inequalities on the
   fractional intersection graph for set partitioning; those are facets of the
   set packing polytope whatever the group sizes, and they are stated on columns,
   so in this master they are non-robust in the costly way. The rank-1 cut with
   multiplier 1/2 on an odd set S of markets is the bridge: a group's
   coefficient is floor(|group ∩ S| / 2), which for pairs is Edmonds' blossom
   inequality on S, and for groups of 2 to 4 its generalisation, priced from a
   count. Step 2 is therefore the study's rank-1 cuts (3 to 5 markets) extended
   to odd sets of any size, separated from the odd cycles of the LP solution as
   in Chandrasekaran-Vegh-Vempala, with limited memory to keep pricing cheap
   (Pecin, Pessoa, Poggi and Uchoa, "Improved branch-cut-and-price for capacitated
   vehicle routing", Math. Program. Comput. 9 (2017) 61-100). Borndörfer's
   preprocessing (section 3.2) lists the reductions beside it: reduced cost
   fixing (3.2.11), which the exact elimination already performs, and dominated
   columns and probing, which it does not. Rasmussen ("Optimisation-Based
   Solution Methods for Set Partitioning Models", dissertation, DTU, 2011,
   section 2.5) gives the conditions under which the set partitioning LP is
   integral (balanced and perfect matrices); the pairs master meets none of
   them, which is why its root bound is 0.32 of the optimum without odd-set cuts.

   Measured (study README, "Odd-set cuts"), and not built. Pairs: four cuts in
   one round lift the 174 geos' bound from 0.32 of the optimum to the optimum.
   Groups of 2 to 4: every odd-set cut closes the whole gap on 11 and 12 markets
   and at least 0.71 of it on 13, so the family is strong; at 30 markets exact
   separation by integer program closed 0.13 of the gap in 15 rounds and 253 s,
   gaining about 0.4% a round by the end, on one instance. The cost is the
   separation (a parity problem over groups of 3 and 4) and a pricing kernel for
   cuts on sets of any size. The open question is answered for these sizes: the
   cuts do not pay for groups of 2 to 4, and pairs need no bound.
3. Profile what remains: the root LP's warm start and the initial pool. The
   bound at every column generation iteration, z_RM + (N / lo) c* (Uchoa, Pessoa
   and Moreno, Theorem 2.8, pp. 29-30; it is the bound `_exact_bound` already
   computes at convergence), says when further pricing rounds stop moving the
   bound, the test for stopping the root early. Farley's bound (their p. 181)
   needs equal column costs and does not apply.

Known-answer families for every claim: pairs against sorting at 20 to 1,000
markets; pairs with trajectory costs against an exact minimum weight perfect
matching, from Blossom VI (Arkhipov and Kolmogorov, "Blossom VI: A practical
minimum weight perfect matching algorithm", arXiv:2604.20351, 2026; C++ at
github.com/Paul566/blossom-vi, nearly linear in practice on every family they
tested, faster than Blossom V where Blossom V is superlinear; built in a pinned
environment like the Trimmed Match reference) or networkx's Edmonds
implementation at a few hundred markets; groups of 2 to 4 with planted optima.

PANGEO side: with Q = 1 every objective is a minimum weight perfect matching, so
an exact matching (sorting for supergeo_total; Blossom VI or Edmonds for the
trajectory objectives) can replace the enumeration and the MIP. With an odd
number of markets no perfect matching exists: one market is dropped or put in a
triple, as Trimmed Match drops its largest geo; the choice is stated and tested.

Which market to drop has an exact answer. Kaibel and Walter ("A note on matchings
constructed during Edmonds' weighted perfect matching algorithm",
arXiv:1703.09505, 2017; earlier Lawler, "Combinatorial Optimization: Networks
and Matroids", Holt, Rinehart and Winston, 1976, chapter 6, problem 8.2, p. 247,
with the bipartite case proved in chapter 5, section 8, p. 205) prove that every
matching Edmonds' weighted perfect matching algorithm builds is of minimum weight among the
matchings of its cardinality, under mild restrictions on the algorithm's
choices. Run on an odd number of markets, the algorithm ends at a maximum
matching that is of minimum weight among maximum matchings: the optimal market
to leave out, with no auxiliary graph of doubled size. For supergeo_total the
same answer is a scan of the sorted order: with market i left out, the rest pair
as neighbours, so the cost is a prefix sum below i plus a suffix sum above it,
and only even positions i qualify. Trimmed Match's drop of the largest geo is
one point of that scan, compared against its minimum.

The same result covers pair exclusion. Trimmed Match's designs drop pairs in a
nested sequence, the worst pair each time; the matchings of Edmonds' algorithm
give, for each number of pairs k, the k pairs of least total cost. Those are not
nested in general (an augmenting path changes pairs already formed), so the two
answer different questions: the nested sequence keeps one pairing and trims it,
the cardinality-optimal one re-pairs at each k. The comparison on the 174 geos
reports both costs at every k.

### Phase 1: harden the exact-cover MIP (`pangeo_helpers/mip.py`)

- Relative and absolute MIP gap 0 (or a stated tolerance, reported).
- One retry with presolve off on any status other than optimal (study:
  `sg.run_lp`).
- Report the solver's status and its proven gap on the result
  (`MethodDetailsResults`), never only the design.
- Tests: the presolve fixture, the default-gap instance (20 markets, outlier),
  infeasible covers reported. Mutants for each.

Small, shared, no change to the answer when HiGHS behaves. Lands first.

### Phase 2: column generation and reduced-cost fixing for the exact path

- A numba pricing kernel for PANGEO's score (DiD residual of mean trajectories,
  best split with halves of at most Q), scanning groups of 2..2Q against the
  duals instead of materialising every candidate.
- Root column generation, the Lagrangian bound, and elimination: only the
  groups that can be in a design no dearer than the incumbent reach the MIP.
- Covariate term and weights carry over unchanged (they are part of a group's
  cost).
- Acceptance: the same optimum as full enumeration on every test panel
  (property against brute force at small N), and the MIP's column count and
  time measured against enumeration at Q = 2 and Q = 3.

### Phase 3: the staged exact search and a certificate

- Port `verify_nb` (one and several words) and `certify_staged` to PANGEO's
  score: capped stages on floating-point scores for incumbents, a final stage
  that proves, the elimination redone at each better incumbent, automatic caps
  and heuristic time.
- Exactness: the score involves means (division by half sizes) and a level
  shift; the certificate needs either exact rationals for the final stage's
  costs or a floating-point evaluation with a proven error bound folded into the
  search margin. Decide by measurement (the study's lesson: exact Fractions on
  947,976 groups cost 103 s; floats with a margin are the scalable path).
- Exposed as `search="exact"`; result carries `proved`, the bound, and the gap.
- Acceptance: optima matching phase 2's MIP and brute force; time on the phase 0
  panels.

### Phase 3b: the proof past 100 markets (added 2026-11-27)

Where it stands. The proof after the first MIP is the volume-bound exact-cover
tree (`pangeo_helpers/vtree.py`, #1183): a Lagrangian bound at every node from
the volume algorithm (Barahona and Anbil 2000, 2002), reduced-cost fixing at
every node, rounding bounded. Exact lazy pricing (#1184) keeps every iterate and
halves its time. Proofs from the optimum take seconds or less up to 50 markets
(8.5 s at 40 markets and Q = 3). At 120 markets and Q = 2 the proof does not
finish: from the start design, 3 million nodes in reduced-cost order reached a
design 8.86% above the root bound, inside the first of 4,911 first-level
branches, and an exact proof extrapolates to about 10^8 nodes.

What limits it, measured. The number of nodes, set by the relaxation: the root
LP over every group sits 8 to 26% below the optimum on arms with known optima,
and at 120 markets the bound closes about 0.7 points a level, so nodes are
pruned only near depth 11 to 12. Per node, the volume steps are 80 to 90% of the
time. Ruled out: more volume steps at shallow nodes (fewer nodes, never less
time: 21.6 to 52.3 s against 18.2 s at 40 markets and Q = 3), the COIN-OR Vol
step rules, the yellow-step and fix-to-one rules of Barahona and Ladanyi (2006),
small re-cover moves on the incumbent, and a semidefinite bound: a group's cost
is a quadratic form over its markets, but every point of the column LP maps to a
point of any SDP built from positive semidefiniteness and constraints each group
satisfies, with the same cost, so no such SDP beats the column LP (measured: 2
to 46% of the LP bound at 24 to 50 markets). The generalized BQP inequalities of
Barkel and Sotirov (arXiv 2605.02456, eqs. 7 to 8, p. 8) are implied by the
column LP for the same reason. Threads over the first-level branches are
deferred until these options are exhausted, so that a result does not depend on
the number of cores.

The work, in order, each measured by nodes and time to a fixed incumbent:

1. Tolerance. A stated relative gap: the search proves its design within it and
   reports the gap proved; exact stays the default. On arms with known optima,
   1% cut the proof 1.5x and 2% cut it 2x at 40 markets and Q = 3, every design
   inside its gap. At 120 markets, the 1% and 2% runs measure whether a proof
   finishes. An iterative cutoff (prove that no design lies below the root bound
   times 1.01^i, raising i; Lichau, Dupas and Francois, Ann. Oper. Res. 366
   (2026), p. 1196) is the same idea run upward.
2. Cuts on the column LP, in the node bound. Rank-1 Chvatal-Gomory cuts on the
   cover rows: for units C and weights rho, sum over groups of
   floor(sum of rho_i over the group's units in C) times x_G is at most
   floor(sum of rho_i) (Lichau et al., eq. 6, p. 1196). Three units with
   rho = 1/2 is the subset-row cut: the groups holding two or more of the three
   sum to at most 1. With all cuts in, rank-1 cuts closed 57 to 84% of the root
   gap left by column generation in Oliveira, Pessoa and Roboredo (4OR 23
   (2025), Table 2, p. 238). The capacity form, the groups meeting a unit set S
   sum to at least ceil(|S| / 2Q), is a rank-1 rounding of the cover rows too.
   Root clique cuts raised the 40-market root bound but nearly doubled the proof
   nodes, so the cuts go in at the root only, withdrawn in the tree (as Olariu
   and Frasinaru, Carpathian J. Math. 41 (2025), p. 471, do), and with a cap on
   the new cuts per group (Barkel and Sotirov, p. 17), and are judged by nodes
   saved, not by the root bound.

   Measured at the root, 2026-11-27. On the 50-market Q = 2 arm the root LP
   has no group at 1: 51 fractional groups at arbitrary values, one connected
   piece over all 50 markets, 2 to 7 fractional groups per market, and the
   cardinality row binding at 13, the optimum's group count (11 groups of four,
   2 of three). Subset-row cuts to saturation (24 cuts) close 2% of the gap
   (8.37% to 8.21%) and save 8% of the proof's nodes. Exact rank-1 separation
   (a MIP over the cover and cardinality rows' multipliers) finds a violated cut
   in each of 15 rounds, but each cut is dense (every kept group) and the 15
   close 9% (to 7.61%), tailing to 0.01 points a round. Rank-1 cuts are not the
   lever at Q = 2; on a 20-market Q = 4 arm subset-row cuts closed a quarter of
   the gap and saved 25% of the nodes, so they are kept for larger groups. At
   Q = 2, branching (item 3) and finding (item 4) come before further cut work.
3. Branching. Ryan-Foster branching on the pair of units with the largest
   fractional mass in the volume algorithm's averaged primal x-bar, same group
   or not (the pair rule of Olariu and Frasinaru, Prop. 4.2, pp. 469-470),
   against the present rule (the uncovered unit with the fewest open groups).
   Then strong branching, a few volume steps per candidate child scored by the
   product rule, with the number of candidates set by an estimate of tree size
   (You, Yang, Wang and Yin, SSRN 4630549, sections 3 and 5). Branching rules are
   compared with a fixed incumbent and cuts at the root only.
4. Finding. POPMUSIC over the whole incumbent: each pass splits its groups into
   disjoint clusters by the centroid of their markets' pretest paths, re-solves
   each cluster exactly with the tree, warm-started, and grows the cluster size
   until one cluster is the arm (Bijiga, Warwicker and Rebennack, Networks 2026,
   Algorithm 1, p. 12; after Ribeiro, Hansen, Taillard and Voss 2002). Re-covers
   of 48 markets reached the optimum at 50 markets where moves of 2 to 6 groups
   found nothing.

Measured, 2026-11-27: items 1, 3 and 4, and the relaxation. Every run is
single-threaded and alone on the machine, on the seasonal simulated arms
(`make_seasonal_sales_panel`, seed 0, 42 pretest periods), with the optimum as a
fixed incumbent unless stated. Node and time ratios are against the library's
tree on the same arm.

Finding (item 4). POPMUSIC as written above (clusters grown by 8 markets from
16, at most 3 passes per size, each cluster re-covered by the tree under 2
million nodes, seed 0) reached the 50-market Q = 2 optimum, 5.405856, from the
start design 5.875812 in 7.3 s, through clusters of 48 and 56 markets. At 120
markets and Q = 2, from the start design 10.032762, clusters of 16 to 72 markets
improved nothing: the start design is optimal inside every cluster those passes
formed. Clusters
of 80 reached 9.938, 88 reached 9.284, 96 to 104 reached 9.068 and 112 reached
9.041550 after 1,519 s, 8.85% above the root bound 8.306326 and below the best
design any earlier method found (9.042264). At this size finding is solved; the
proof is not.

Branching (item 3). Ryan-Foster branching on the pair with the largest
fractional mass in x-bar, or the mass nearest 1/2, with and without a cap on
pair levels and a minimum count of open groups, finds and proves every optimum,
but its best variant per arm against the unit rule is: 0.51x the nodes and
1.99x the time at 20 markets and Q = 4, 1.34x and 1.24x at 36 and Q = 3, 1.15x
and 1.13x at 40 and Q = 2, 0.99x and 1.01x at 40 and Q = 3, 1.26x and 1.20x at
50 and Q = 2. The unit rule is multi-node branching on one cover row, which Cho
and Linderoth find holds its own on dense set partitioning (30 wins of 55 on
dense small pure-partitioning instances, ICS 2015, Table 1, p. 129), and groups
of 2Q of n markets are dense. Choosing the unit by the gap mass of its open
groups, in place of their count, is never better.

Strong branching (item 3) has nothing to save here: a child that survives
fixing stops its volume steps once its bound passes the incumbent. Its zero-step
form does save nodes, as a stronger fixing rule. Taking group j removes every
open group that meets it, so at the parent's multipliers the child's bound is
L + max(rc_j, 0) plus the negative reduced costs of the open groups clashing
with j (each reduced cost taken at its upper bound, so the rounding stays on the
safe side). Dropping the children that bound rules out gave 0.75x the nodes and
0.77x the time at 40 markets and Q = 3, 0.82x and 0.88x at 60 and Q = 2, 0.80x
to 0.94x the nodes at the other arms (time 0.95x to 1.3x, on runs under half a
second), and at 120 markets
8% more time a node (169.9 against 156.7 us). It is a constant factor against a
tree that grows 12 to 14 times for each percent of gap, and is not built. A
build must copy the clashing groups' ids before the fixing loop, which compacts
the open list in place: a scratch version that read them from the list after
compaction proved a design that is not the optimum at 40 markets and Q = 2.
Column dominance (reduction 4 of Garfinkel and Nemhauser, as Cacao et al. state
it, p. 7: drop a group that costs at least some partition of its markets into
other groups, valid here since the cardinality row is sum x >= k) removes
nothing at Q = 2 (no group at 50 or 120 markets is dominated by a two-way split);
at 40 markets and Q = 3 it covers 6.2% of the pool but none of the groups
elimination keeps.

Tolerance and iterative cutoff (item 1). Rounds at the cutoff root bound times
1.01^i, each proving no design lies below its cutoff, cost 1.9x to 3.4x the
plain proof in total on the arms with known optima (40 markets Q = 3: 18 rounds,
22.9 s against 8.5 s), as expected, since the last round is the plain proof. The
gain is a proved lower bound that rises during the run. At 120 markets and
Q = 2 (12 million nodes a round at most):

| cutoff over the root bound | nodes | seconds |
|---|---|---|
| +1.00% | 503 | 0.06 |
| +2.01% | 7,065 | 1.3 |
| +3.03% | 87,963 | 19.9 |
| +4.06% | 1,229,329 | 258 |
| +5.10% | stopped at 12,000,000, 2 of 412 first-level branches done | 2,071 |

So after 40 minutes every design costs at least 8.643596, and against 9.041550
the gap is certified at 4.6%. A time-limited plain search certifies only the root
bound, since depth-first search keeps its shallowest open node there. At about
13 times the nodes per percent, a 3% certificate (the bound at 8.778) is hours
away and a 1% certificate (8.952) is out of reach.

The relaxation, measured. Three relaxations past the column LP:

- An LP over pairs of disjoint groups (plus at most one single group, which
  every design splits into) closes 0.0% of the gap at 36, 40 and 50 markets and
  0.8% at 20 markets and Q = 4: the root solution already decomposes into
  disjoint pairs.
- A group's cost is its between-halves scatter, so a design costs
  W(groups) - W(halves), W the within-cluster sum of squares (checked on every
  quadruple to 1e-13). With the Peng-Wei matrices of both partitions, psd,
  nonnegative, rows summing to 1, nested (Z_h - Z_g psd, Z_h <= 2 Z_g), and
  triangle cuts on both, the SDP sits 60 to 73% below the optimum at 24 to 40
  markets, against 15 to 32% for the column LP, and violates almost no cut.
- Rank-1 and subset-row cuts close 2 to 9% (above), root clique cuts 14%.

The cause is the cost. On the same panels and group shapes the clustering cost
of the MSSC literature (within-group scatter) leaves a column LP gap of 2.6%,
1.9% and 4.1% at 24, 32 and 40 markets, against 31.7%, 16.7% and 14.8% under
PANGEO's cost. Under PANGEO's cost a market lies in a median 75 to 146 groups
within the integrality gap (3 to 4 under the clustering cost), the root LP is
fractional everywhere in one piece over the arm (15 to 18 separate pieces under
the clustering cost), and its groups' markets lie as far apart as random markets
(0.88 to 0.95 of the mean distance, 0.24 to 0.33 under the clustering cost).
Every PANGEO group of four costs less than its best cover by two pairs (median
ratio 0.137). A group of four is cheap when two pairs have nearly equal sums,
however far apart its markets, so each market belongs to many nearly cancelling
groups, and the LP spreads its weight over them where a design must find
disjoint ones. That is the structure of vector number partitioning, where no
strong relaxation is known: the ILP relaxation is "natural, but not tight"
(Kojic, Comput. Math. Appl. 60 (2010), p. 2306), exact anticlustering stops near
30 to 70 items (Papenberg and Klau, Psychol. Methods 26 (2021), preprint
Table 1, p. 21), and complete Karmarkar-Karp prunes with a scalar bound that has
no vector form (Mertens, arXiv cs/9903011). The MSSC column LP is tight in
practice (du Merle et al., SIAM J. Sci. Comput. 21 (2000), p. 1498; Aloise,
Hansen and Liberti, Math. Program. 131 (2012), Tables 2 to 13, pp. 211 to 216)
because its cheap groups are local, and the Peng-Wei SDP weakens as clusters
multiply (Piccialli, Sudoso and Wiegele, SOS-SDP, arXiv 2104.11542v3, p. 29).

What follows. Past about 60 markets at Q = 2 the deliverable is the best design
with a certified gap: POPMUSIC for the design, the iterative cutoff for the
bound. A search over halves (once the halves are fixed, pairing them into groups
is a minimum-weight perfect matching, Edmonds) is the one exact idea left, and
it changes the algorithm, not the bound.

Built and measured after the certified gap, 2026-11-27. Items 1 and 4 are in
the library: the iterative cutoff as the certified gap of a capped proof
(#1188) and POPMUSIC before it (#1190).

POPMUSIC in the library. Inside `solve_exact` POPMUSIC starts from the capped
proof's design, not from the start design. On the 120-market Q = 2 arm with
every default it lowered the design from 9.440014 to 9.208466 with its whole
12 million node budget (17 passes, 47 subproblems, 6 of them improving), and
the certified gap fell from 8.44% to 6.13% against the dual bound 8.643596
(4,717 s single-threaded). The 9.041550 above came from the start design
10.032762 with the same settings: where POPMUSIC ends depends on where it
starts.

Two more relaxations, on the 40-market Q = 2 seasonal arm (optimum 4.889056,
column LP 4.165353, 14.80% below):

- Overlapping blocks. Blocks of r markets, each priced exactly over the
  patterns its markets can take, give a relaxation that is exact on every
  block (r-neighbourhood-exact in the sense of Eisenbrand, Fiorini, Rohwedder
  and Wei, arXiv 2607.09222, section 3.1, p. 7, and Lemma 4, p. 8). A
  subgradient method on the multipliers closed 11.1% of the gap with blocks of
  3 markets (49 blocks, bound 4.245502 after 873 s) and 11.5% with blocks of 4
  (20 blocks, 4.248411 after 4,174 s), 13.1% below the optimum either way:
  about 0.3 points of gap for each market added to a block. Column generation
  over the same blocks stalled on degenerate duals.
- RLT level 2 (Sherali and Adams, in Du and Pardalos (eds.), Handbook of
  Combinatorial Optimization, vol. 1, Kluwer 1998, pp. 479-532). For set
  partitioning the first two levels already capture published valid
  inequalities (p. 493), and a product of two groups that share a market is
  zero (p. 506), after Sherali and Lee (Discrete Appl. Math. 68 (1996)). The
  build keeps a variable for each pair of disjoint groups whose reduced costs
  fit in the gap and multiplies each cover row by each kept group: 199,142
  pairs and 50,621 equality rows. Its LP bound is 4.623153, 5.44% below the
  optimum, closing 63.3% of the gap, in 8,392 s with HiGHS's interior point
  method (the simplex stalled). On the 20-market Q = 4 arm it closes all of the
  31.60% gap in 0.4 s. It is the only relaxation measured here that closes most
  of the gap, but one root LP at 40 markets takes 2.3 hours and the pair count
  grows with the square of the pool, so as a node bound past 100 markets it is
  out of reach. It is the target a cheaper bound would have to approximate.

Grouped panels prove. When the markets fall into latent groups that share
their time paths or loadings, the column LP is close and the proof finishes,
at 120 markets too (Q = 2, every default, single-threaded):

| panel | markets | column LP below the optimum | result |
|---|---|---|---|
| exact-group factor model (4 groups, 4 factors, T0 = 60) | 40 | 1.66% | proved; RLT-2 closes all of it in 8 s |
| the same | 50 | 1.37% | proved; RLT-2 closes all of it in 1,004 s |
| grouped fixed effects (a time path per group plus noise), 3 groups of 18 / 18 / 24, T = 25, noise sd 0.5 / 1.0 | 60 | 4.31% / 4.58% | proved in 9 s / 27 s |
| the `classo` package's sample data (github.com/zhan-gao/classo), 18 / 18 / 24 markets of its three groups, T = 25 | 60 | 3.67% | proved |
| Liao, Shi and Zheng, "A Relaxation Approach to Synthetic Control" (arXiv 2508.01793v3), section 5.1, exact groups, J = 120, K = 4, r = 4 | 120 | 1.76% | proved in 10,306 s |
| the same, section 5.2, approximate groups | 120 | 1.93% | proved in 8,248 s |

Against 8 to 26% on the seasonal arms, this is the cause above seen from the
other side: when the groups are latent clusters, the cheap groups of four are
local, as under the clustering cost, and cancellation across distant markets
has little room. The general-loading design of the same section 5.3 (every
market's loading drawn on its own) was stopped by a machine restart and is not
measured.

### Note for phases 2 and 3: the trajectory cost through a Gram matrix

A group's trajectory cost, the squared distance between its halves' mean paths
after the best level shift, depends on the data only through inner products of
the markets' demeaned paths. With the N x N Gram matrix of those paths computed
once, a group's cost is a quadratic form in a weight vector over its 2 to 4
markets and the matching block of the matrix: every group scores without
touching the T periods (the profiled, batched evaluation of the slope-free TBR
and FDID work). Profiling out a slope as well as the level balances trends and
keeps the same form. The scalar window of the base problem then has a
counterpart: project the paths on the Gram matrix's leading eigenvectors, run
the window in that space, and fold the discarded eigenvalues into the safety
margin so the elimination stays exact. Totals cannot see trend, seasonality or
volatility (two markets with equal pretest totals can differ in all three), so
this is the path from the base problem's machinery to PANGEO's objective.

### Note for the trajectory objective: what the QUBO literature offers

With sums over markets and the level profiled out (M = I - 11'/T on a market's
path y_i), a split's trajectory cost is ||M sum_i s_i y_i||^2 = s' G s, s in
{-1, +1}^n the halves and G_ij = y_i' M y_j the Gram matrix of the note above: an
Ising problem with couplings G and no field, that is a QUBO. The base problem's
objective is the rank-one case G = z z' (number partitioning; Lucas, "Ising
formulations of many NP problems", Front. Phys. 2 (2014) 5); with trajectories G
has rank at most T - 1 (vector number partitioning, or vector balancing).

- Complexity. The rank-one QUBO is NP-hard (Cela and Punnen, "Complexity and
  polynomially solvable special cases of QUBO", ch. 3 of Punnen (ed.), The
  Quadratic Unconstrained Binary Optimization Problem, Springer 2022,
  doi:10.1007/978-3-031-04520-2, Theorem 3.10, after Cela, Klinz and Meyer, J.
  Comb. Optim. 12 (2006) 187-215, and Hammer, Hansen, Pardalos and Rader,
  Optimization 51 (2002) 511-537). The polynomial fixed-rank case is the
  maximisation of a positive semidefinite form, by enumerating a zonotope's
  vertices (Allemand, Fukuda, Liebling and Steiner, Math. Program. 91 (2001)
  49-52; Ferrez, Fukuda and Liebling, Eur. J. Oper. Res. 166 (2005) 35-50; ch. 3,
  section 3.3.2): the max-cut direction, not ours, which minimises one. What
  does carry over is the pseudo-polynomial algorithm for rational Q of fixed rank
  (ch. 3, section 3.5.1, after Cela, Klinz and Meyer 2006), the counterpart of
  the number-partitioning dynamic program; its table grows with the data's range
  to the power of the rank, so at most a few leading eigen-directions.
- Pricing. Column generation's pricing becomes a QUBO restricted to groups of 2
  to 4 with the duals as a linear term; with G precomputed a group scores in
  about 16 lookups. The scalar window's sort has no counterpart in R^T; the
  projection on G's leading eigenvectors with a proven margin (note above) is
  the route.
- Bounds. The semidefinite relaxation of min s' G s may sit near zero: G's rank
  is at most T - 1, far below the number of markets, which leaves the
  relaxation room, the same failure as the LP bound on number partitioning (0.27
  of the optimum on GeoLift). Measure before relying on it. The warm-startable
  bounds of Chalkis, Kleinert and Sofranac ("QUBO dual bounds via SDP plane
  projection method", arXiv:2312.15026), built on the quadratic convex
  reformulation for branch and bound, are the candidate to measure; Rehfeldt,
  Koch and Shinano ("Faster exact solution of sparse MaxCut and QUBO problems",
  Math. Program. Comput. 15 (2023) 445-470) is the exact branch-and-cut state of
  the art, whose reductions and separation assume sparsity that a Gram matrix
  does not have.
- Fixing. Autarkies and persistencies (Boros, ch. 5 of Punnen 2022) fix
  variables in every optimum from the roof dual: the QUBO counterpart of the
  reduced-cost elimination, to measure on the trajectory objective.
- Incumbents. QUBO heuristics supply incumbents to the exact certificate, never
  the proof: metaheuristics (Wang and Hao, ch. 9 of Punnen 2022: tabu search,
  path relinking), mean-field annealing (Veszeli and Vattay, PLoS ONE 17 (2022)
  e0273709), and recent hybrid solvers (VeloxQ, Pawlowski et al.,
  arXiv:2501.19221; QIS3, TuringQ, arXiv:2506.04596). Software: ch. 11 of
  Punnen 2022.
- Constraints. The whole design as one QUBO needs penalty terms for group
  membership, group size (2 to 4) and a budget, which distort the landscape and
  need tuning; Samanta, Wang, Hodges and Ferenczi ("An effective QUBO formulation
  for knapsack problems", IEEE QAI 2025, doi:10.1109/QAI63978.2025.00039) encode
  an inequality with two slack variables, the form a budget or size cap would
  take. The study keeps the constraints in the set-partitioning master instead.
- Random instances. The random QUBO (Natarajan, ch. 7 of Punnen 2022) and
  Mertens' random-cost picture of partitioning (phase 3) bear on how the
  proof's tree grows with T; check what the multidimensional partitioning
  literature says before claiming where the phase transition moves.
- Design, not optimisation: the Gram-Schmidt walk design (Harshaw, Savje,
  Spielman and Zhang, "Balancing covariates in randomized experiments with the
  Gram-Schmidt walk design", J. Amer. Statist. Assoc.) randomises while
  balancing vector covariates such as pretest paths, with a design-based
  variance bound; a comparison arm for phase 5.

### Phase 4: the fast design, certified

- `fast=True` keeps its design as the incumbent; the staged search then improves
  it or certifies it within a time limit and reports the proven gap. The fast
  path stays the default; the gap makes its quality visible.
- Acceptance: on the phase 0 panels, the gap reported in seconds.

### Phase 5: power analysis

The variance of the design's iROAS estimate has a closed form, and the window it
is evaluated on decides whether the number means anything.

- Closed form. Under a common iROAS, a fixed budget B and the ratio estimator,
  the randomization over the coin flips gives Var(theta_hat) = sum_k D_k^2 / B^2,
  D_k the difference between supergeo k's halves in the test window's response
  without treatment (Chen, Doudchenko, Jiang, Stein and Ying, "Supergeo design",
  arXiv:2301.12044, eq. 3). The minimum detectable iROAS is
  (z_{1-alpha} + z_{power}) sqrt(sum_k D_k^2) / B, about 2.49 sqrt(sum D_k^2) / B
  for a one-sided 5% test with 80% power; the budget for a target iROAS theta* is
  2.49 sqrt(sum D_k^2) / theta*.
- Held-out windows. The D_k of the test window are unknown at design time, and on
  the design's own window they are the design loss, which understates them: on
  the simulated 180-market panel (study README) sqrt(sum D_k^2) was 0.63 in sample
  and 583 at the median held-out 4-week window, a minimum detectable iROAS of
  0.001 against 0.99 (1.54 at the 95th percentile). Design on one window and
  evaluate on others (Chen, Longfils and Remy, arXiv:2105.07060, Algorithm 1).
  Report the median and a high percentile over the evaluation windows.
- Rolling re-design. The faithful backtest re-runs the design at each historical
  date and evaluates it on the next window, at the lag the real test has; one
  design evaluated on distant windows mixes in the decay of its balance. The
  design is reproducible (step-budgeted heuristic, seed), so each backtest design
  is too.
- Test length. On that panel the minimum detectable iROAS fell only from 1.03 at
  1 week to 0.84 at 26 weeks: the halves' differences drift (trends, seasons,
  persistent factors) and grow almost as fast as the budget. The power analysis
  reports the curve against length, and the estimators below are compared on it.
- Estimators by simulation. For the Trimmed Match estimator (phase 6), a
  pretest-adjusted estimator (difference in differences, or the TBR regression),
  and heterogeneous iROAS, there is no closed form: inject a known effect into
  each evaluation window, draw assignments, run the estimator, take the RMSE
  (TMD Algorithm 1). With few supergeos, critical values from the permutation
  distribution of sum_k A_k D_k (on the 180-market design, 47 supergeos, the
  permutation 95th percentile was 1.647 standard errors against the normal 1.645).
- Budget. The fixed-budget assumption holds when the change is proportional to
  each treated geo's spend with the ratio set from the realized assignment
  (Chen et al. 2023, Assumption 4); a full go-dark cuts the treated geos' whole
  spend, which varies with the assignment. On the simulated panel a full go-dark
  cut 1,835 +- 59 (half of all spend) against a cap of 1,468, in every assignment;
  the proportional cut met the cap exactly with a ratio of 0.80. Re-choosing the
  splits to balance spend narrowed the cut's spread and raised the iROAS error
  12%, so spend balance is a constraint for small designs, not an objective
  (phase 5e).
- Also from Trimmed Match Design: the maximum supergeo size (Q) and the number of
  pairs chosen by the cross-validated RMSE, rerandomization with recorded balance
  checks (TMD section 4.3), and optional geo exclusion as singleton columns with a
  bound on their number. Excluding geos changes the estimand to the included
  geos' spend-weighted iROAS and can bias it when their iROAS differs (Chen et al.
  2023, err_3); the result says which estimand it reports.
- Pair exclusion is valid under four conditions, and the library option checks
  or reports each:
  1. Design time only. Pairs are dropped before assignment, on pre-period data;
     dropping pairs after seeing post-period outcomes is a specification search
     and voids the inference.
  2. The estimand is restated. The effect reported is the included markets'
     spend-weighted iROAS, named on the result with the markets left out.
  3. The count is chosen out of sample. The number of pairs dropped is chosen by
     the cross-validated RMSE on held-out windows above, not by the in-sample
     cost, which falls with every pair dropped.
  4. The count is bounded and refused when it leaves no design: a stated cap on
     the share of pairs dropped, and an error when the exclusion drops every
     pair, more pairs than the markets form, or leaves fewer than `min_pairs`
     (the study's `exclusion.design_excluding` raises in each case).
  The study (README, "Pair exclusion") compares the two ways of choosing which
  pairs to keep, nested trimming and the cardinality-optimal pairs; the option
  waits for the cross-validated count of condition 3.
- Acceptance: the closed form against simulation on the same windows; a placebo
  study on held-out windows showing whether the cross-validated choices beat the
  in-sample ones in RMSE.

### Phase 5b: designs on a prognostic index (Bai 2022)

Bai, "Optimality of Matched-Pair Designs in Randomized Controlled Trials",
American Economic Review 112(12) (2022) 3911-3940.

- The result. Among stratified randomizations that treat each unit with
  probability one half, the MSE of the difference in means (conditional on the
  covariates) is minimised by ordering the units by the index
  g(x) = E[Y(1) + Y(0) | X = x] and pairing neighbours (Theorem 1), at any sample
  size and with heterogeneous effects. The ex post bias variance is
  (1/4n^2) sum over pairs of (g_a - g_b)^2, minimised by sorting (the
  Hardy-Littlewood-Polya rearrangement inequality, his Lemma B.1); any
  stratification is a mixture of matched-pair designs (Lemma 2), so larger strata
  never do better. With one covariate on which g is monotone, sorting on that
  covariate is optimal (Theorem 2).
- What it says about the pairs path. `supergeo_total`'s pair cost, the squared
  difference of estimation-window totals, is that variance with the pre-period
  total in place of g: the closed form of jgreathouse9/mlsynth#1115 is Bai's
  "MP base" design, and Theorems 1 and 2 are its citation (added to `pairs.py`
  and the docs page on the odd-case branch).
- A prognostic objective. The index is unknown, and with homogeneous effects its
  order is the order of E[Y(0) | X] (his eq. 9), which auxiliary data can
  estimate. In a geo panel the history is that data: fit a held-out late window
  of the pre-period on earlier windows and covariates, cross-validated (Trimmed
  Match Design's evaluation-window rule), and pair on the fitted index. One
  scalar per market again, so the design is a sort (CLAUDE.md invariant 8).
  Acceptance: the held-out MDE (phase 5) against `supergeo_total` and the
  covariate-penalty designs, on the phase 0 panels and on simulated panels where
  the index is planted.
- Inference. PANGEO's matched-pair SE is the "matched pairs" form, which is
  conservative unless effects are homogeneous (his Remark 6). The adjusted
  variance of Bai, Romano and Shaikh ("Inference in Experiments with Matched
  Pairs", JASA 2021; Bai's eqs. 13-14) adds a correction from adjacent pairs of
  pairs and is consistent; it needs the pairs in index order, which a sorted
  design provides. Test-first in `effects.py`, with the size of the test checked
  by simulation.
- Several covariates. In Bai's simulations on ten AEJ: Applied RCTs (Table 2,
  ratios to no stratification): pairs on the baseline outcome alone gave MSE
  0.762 and a test of size 4.8% at 5%; Mahalanobis pairs on all covariates gave
  MSE 0.549 but a size of 2.3%, the pairs too far apart for the asymptotics; sets
  of four on the same distance (pair, then pair the pairs by midpoints) restored
  the size to 5.1% at MSE 0.695. Regression adjustment after the fact did not
  recover what matching gained. If the covariate path stays, it matches in sets
  of four for inference.
- Limits. Bai's units are i.i.d. and assigned one at a time, the estimator is the
  difference in means, and the sample is even. A supergeo (Q >= 2) assigns
  several geos together, outside his Assumption 1, so nothing here says whether
  supergeos help; PANGEO's analysis is a DiD; geo panels are small and uneven in
  size. The results transfer most directly to Q = 1.

### Phase 5c: choosing the covariates by permutation (Anand and Talbot 2015)

Anand and Talbot, "Automatic Selection of Partitioning Variables for Small
Multiple Displays", IEEE Transactions on Visualization and Computer Graphics
(VIS 2015).

- The method. A candidate partitioning variable is scored by a statistic
  computed on each partition it induces, against the same statistic on random
  partitions of the same sizes (a randomized permutation test; Good,
  "Permutation Tests", Springer 2000). Each partition gets a z-score against its
  permutation null, with Chebyshev's inequality as the conservative bound when
  the observed value lies outside the simulated range, and the variable scores
  the largest |z|. Their criteria map onto the design problem: informative (the
  partitions differ from random ones), well supported (small partitions widen the
  null, so chance patterns score low) and parsimonious (more partitions widen it
  too).
- Use in PANGEO. Bai's index is unknown and his simulations show that covariates
  added without regard to their predictive power cost inference; the study found
  the pretest optimum overfits. A covariate (or a region variable for phase 6b
  and the study's groups within regions) enters the index, the blocking or the
  regions only if partitioning the markets by it separates the held-out
  pre-period outcome, or its residual on the outcome-only index, beyond what
  random partitions of the same sizes do. The statistic is the between-partition
  variance of that outcome; continuous covariates are binned as the paper does
  (equal-count bins), with the binning held fixed across permutations.
- Their stated limits (section 5), each with an answer here: correlated
  candidates both score high, so selection is sequential, each candidate scored
  on the residual after those already chosen; the Chebyshev bound is
  conservative, so with enough permutations the empirical p-value is reported
  instead; the maximum over partitions favours one strong partition, which suits
  a variable that isolates one group of markets but is compared against the mean
  |z| in the measurement.
- Acceptance: on simulated panels with planted predictive and irrelevant
  covariates, the selection keeps the planted ones and drops the rest at the
  stated rate; on the phase 0 panels, the held-out MDE of designs on the selected
  covariates against designs on the outcome alone and on all covariates.

### Phase 5d: blocked designs under a grouped factor model

The model, for market i in cluster c(i), with a primary KPI and secondary ones
of the same form:

    Y_it(0) = alpha_i + lambda_c(i)' F_t + beta_c(i)' W_c(i),t + eps_it,

F_t latent common factors, W_c,t observed cluster-level series such as weather
(heat, rain), and loadings and sensitivities shared within a cluster: the
grouped linear factor structure of Liao, Shi and Zheng that LEXSCM's and
SYNDES's examples already simulate.

- What a pair's difference carries. Two markets in one cluster share lambda,
  beta and W, so F and W cancel in their difference and only alpha_a - alpha_b
  and noise remain: the test window's factors and weather, unknown at design
  time, cannot enter the comparison. Two markets in different clusters carry
  (lambda_c - lambda_c')' F_t and the weather terms, a variance of
  (lambda_c - lambda_c')' Sigma_F (lambda_c - lambda_c') plus its weather
  analogue: a distance in several dimensions.
- The design. Block on the clusters; within a cluster the remaining imbalance is
  the scalar alpha_i, so Bai's result (phase 5b) applies block by block: sort
  each cluster on its level or prognostic index and pair neighbours, one closed
  form per cluster (CLAUDE.md invariant 8). A cluster with an odd count leaves
  one market; the leftovers are paired across clusters at the loading-distance
  cost, a matching in several dimensions solved by the blossom matcher, or left
  out (the odd case of the pairs path). The global minimum-weight matching on the
  full cost does both at once, since pairs within a cluster cost almost nothing;
  the closed form applies exactly when every cluster is even. The supergeo
  study's groups within regions (`certify_staged`'s region constraint) is the
  same blocking for Q >= 2.
- Weather, twice. At design time the cluster-level series define or sharpen the
  clusters (weather is regional), and a market's estimated sensitivity beta can
  enter its index. At analysis time the test window's weather is observed and
  not affected by treatment, so it enters the DiD/ADID as a control and absorbs
  a shock the design could not see; this differs from the regression adjustment
  of baseline covariates that Bai (Section V) finds does not recover what
  matching gains.
- Several KPIs. The blocking does not depend on the KPI: shared factors and
  weather cancel within a cluster for every KPI at once. The within-cluster index
  does: the primary KPI's index, or matched sets of four on the baselines of all
  KPIs (Bai, Section V.C, where pairing on one outcome did not help another).
- The clusters are checked, not assumed. Clusters from geography (census
  divisions, weather zones) and clusters estimated from co-movement (grouped
  fixed effects, Bonhomme and Manresa, Econometrica 83(3), 2015; a grouped
  factor model, Ando and Bai) both earn their blocks through phase 5c's
  permutation test: the partition separates the held-out outcome beyond random
  partitions of the same sizes. Estimated clusters are fitted to the pre-period
  and the test is what keeps them from overfitting it.
- Inference. Bai's optimality assumes independent units; under the factor model
  units in a cluster are dependent through F and W. Pairing within clusters
  removes that dependence from the pair differences, but with few clusters the
  randomization test permutes within blocks and the variance is checked by
  simulation.
- Acceptance: on panels simulated from the model with planted clusters, weather
  and loadings, the blocked design's held-out MDE (phase 5) against the
  unblocked sorted design, the trajectory objective and the matcher on the full
  cost; the permutation test recovers the planted partition and rejects a random
  one; the analysis with weather as a control against the analysis without it.

### Phase 5e: spend balance as a design constraint

The paper's variance, Var(theta_hat) = sum_k D_k^2 / B^2 (eq. 3), needs the
spend difference between the arms to be a constant B whatever the coin flips
(Assumption 2). A proportional heavy-up meets it by construction: the ratio r is
set after the draw so the treated geos' extra spend is B (Assumption 4). A go-dark
does not. It cuts each treated geo's whole spend, so the cut is the spend of
whichever half each coin treats.

- What a random B does. Write S_k+ and S_k- for the spend of pair k's halves over
  the test window, g_k = S_k+ - S_k- for their gap, and B0 = sum_k (S_k+ + S_k-) / 2
  for the mean cut. A go-dark cuts B(A) = B0 + sum_k A_k g_k / 2. Then:
  1. The cut ranges over [sum_k min(S_k+, S_k-), sum_k max(S_k+, S_k-)] and has
     variance sum_k g_k^2 / 4 over the coin flips.
  2. The error of the ratio estimator is sum_k A_k D_k / B(A). Its numerator and
     denominator share the coins, so to first order it has mean
     -sum_k D_k g_k / (2 B0^2). The bias is zero when the spend gaps are unrelated
     to the response gaps, and in ad data the two move together (the geo that
     spends more sells more).
  3. A business cap on the cut can be broken. On the simulated 180-market panel
     (phase 5, Budget) a full go-dark cut 1,835 +- 59 against a cap of 1,468, in
     every assignment.
  Under a heavy-up B is fixed, but r = B / (treated spend) still varies with the
  draw, and the heterogeneous-iROAS error err_3 = r sum_{treated} (theta_g - theta)
  S_g (Chen et al. 2023, eq. 5) scales with it. Spend balance narrows r's spread
  too.
- The constraints. Each is linear in the columns of the exact-cover program, so
  it adds rows or removes columns, and the objective stays sum_k D_k^2:
  1. A cap on every assignment's cut: sum_k max(S_k+, S_k-) x_k <= cap, one
     knapsack row. Since max(a, b) = (a + b) / 2 + |a - b| / 2 and the halves
     cover every market, the row is sum_k |g_k| x_k <= 2 cap - S, with S the
     total spend. A floor (the cut must reach a stated size for power) is the
     row sum_k min(S_k+, S_k-) x_k >= floor, the same gap row against S - 2 floor.
  2. A bound on the cut's spread: sum_k g_k^2 x_k <= 4 v, a row on the variance.
  3. A gap per pair: |g_k| <= tau (S_k+ + S_k-), applied by dropping the columns
     that break it, with no row. For pairs (Q = 1) these are forbidden pairs of
     markets.
  The measured case for a constraint over an objective: re-choosing the splits
  to balance spend narrowed the cut's spread and raised the iROAS error 12%
  (phase 5, Budget). The design keeps the response objective and imposes spend
  balance only as far as the cap, floor or gap requires.
- Solvers, closed form first (CLAUDE.md invariant 8). A design that is optimal
  without the constraints and meets them is optimal with them, so every path
  first solves the unconstrained problem (the sorted pairing, the matcher, the
  staged search) and checks the constraints on its answer. Only a design that
  breaks one goes further:
  - Pairs with a gap per pair only: the matcher with the forbidden pairs at a
    cost above any saving; a forbidden pair in its answer means no admissible
    pairing exists, which raises.
  - Pairs with a cap, floor or spread row: a matching with one budget row, which
    is NP-hard (Berger, Bonifaci, Grandoni and Schafer, "Budgeted matching and
    budgeted matroid intersection via the gasoline puzzle", Math. Program. 128
    (2011) 355-372, who give a PTAS). For each multiplier lambda, the cost
    D_k^2 + lambda |g_k| is a plain matching, so the matcher gives the
    Lagrangian bound and a feasible design at the multiplier where the row is
    met; the gap between the two is reported, and an exact answer comes from
    branching on pairs under that bound, or from the exact-cover program with
    the row.
  - Supergeos (Q >= 2): the row in the exact-cover program, and in the staged
    search (phase 3) as a bound on the cheapest completion's gap sum.
- Interface. A spend column in the panel; `spend_change` ("heavy_up",
  "go_dark", "partial_go_dark", the last cutting each treated geo by one fraction
  chosen after the draw so the cut is a fixed B, which restores Assumption 2);
  `budget_cap`, `budget_floor`, `max_cut_sd` and `max_pair_spend_gap`, each
  optional. The result reports the cut's range and standard deviation over the
  assignments, the first-order bias term on the design window, which constraints
  bind, and the proven gap when the Lagrangian path does not close.
- Failures. A cap below S / 2 admits no go-dark design: sum_k max(S_k+, S_k-)
  >= S / 2 for every design, and the mean cut is S / 2. A floor above S / 2
  admits none for the same reason, since sum_k min(S_k+, S_k-) <= S / 2. A gap
  bound that leaves a market with no admissible partner admits none either. Each
  raises MlsynthConfigError naming the bound and the value it would need.
- Acceptance:
  - brute force over every design of up to 12 markets with each constraint,
    against every solver path, and the unconstrained answer returned unchanged
    when it meets the constraints;
  - the Lagrangian bound never above the brute-force optimum;
  - on the simulated 180-market panel, the share of assignments over the cap
    (100% unconstrained) falls to 0, with the held-out MDE (phase 5) of the
    constrained design against the unconstrained one and against spend in the
    objective;
  - the bias term measured by simulation on panels where spend and response are
    correlated, against its first-order value, under go-dark and partial
    go-dark.

### Phase 6: Trimmed Match as an analysis estimator

A separate estimator, own branch: the trimmed ratio estimator of Chen and Au
("Robust causal inference for incremental return on ad spend with randomized geo
experiments", Annals of Applied Statistics), with its confidence interval, for
pairs and for supergeo pairs. Used by phase 5's simulations and by PANGEO's
analysis of a finished experiment.

- Reference implementation: google/trimmed_match (Apache 2.0, "for research
  purposes only"), a C++ core with a Python wrapper built by bazel, tested with
  Python 3.7. Its estimator is `trimmed_match.estimator.TrimmedMatch`, taking the
  per-pair `delta_response` and `delta_spend` (treatment minus control), with
  `Report(confidence=...)` returning the estimate and the interval. At commit
  0be855234ee3a589eaddc8b945e2a0d924ad8791 (2023-06-01) the package also holds:
  - design: `design/trimmed_match_design.py`, `TrimmedMatchGeoXDesign`, with
    `create_geo_pairs(use_cross_validation=True)` and
    `report_candidate_designs(budget_list, iroas_list, use_cross_validation,
    num_simulations, max_trim_rate, estimator)`, which returns the RMSE of the
    iROAS estimate for each budget, hypothesized iROAS and number of pairs, with
    the experiment's response, spend and share of total cost;
  - power: `design/matched_pairs_rmse.py`, `MatchedPairsRMSE(geox_type,
    geo_pairs_eval_data, budget, hypothesized_iroas, base_seed)` and its
    `report(num_simulations, max_trim_rate)`, with `GeoXType.GO_DARK` among the
    experiment types;
  - assignment: `design/geo_assignment.py`, `generate_balanced_random_assignment`
    (rerandomization with balance checks), `binomial_sign_test`,
    `trimmed_match_aa_test`, and `representativeness_test`, whose proxy for a
    representative assignment is a minimum response and a minimum number of geos
    in each group and a maximum share of either group's response in its largest
    geo;
  - post-analysis: `post_analysis/trimmed_match_post_analysis.py`
    (`prepare_data_for_post_analysis`, `calculate_experiment_results`);
  - data: `example_datasets/example_data_for_design.csv` (174 geos, daily
    response and cost over 364 days of 2020),
    `example_data_for_post_analysis.csv` (114 geos in 57 assigned pairs),
    `example_geo_pairs.csv` (172 geos: 67 treatment, 67 control, 38 filtered,
    the design's geo exclusion), and `simulation_script.py` (seeded simulated
    design and post-analysis datasets);
  - its own tests (`estimator_test.py`, `design/tests/`, `post_analysis/tests/`).
- Trimming and supergeos: Chen et al. (2023, section 4) show trimming poorly
  matched pairs can bias the estimate under heterogeneous iROAS, and that the
  supergeo design does not introduce that bias; the estimator reports how many
  pairs it trimmed, and phase 5 measures the bias on simulated heterogeneous
  effects.

### Phase 6b: an aggregate effect from regional designs

With `region=` the design keeps every supergeo inside one region (study README).
The analysis then owes one national number and one number per region, from the
same supergeos.

- Every supergeo is still a randomized unit. A coin picks which half of supergeo
  k is treated, so with X_k and Y_k the treated-minus-control spend and response,
  e_k(theta) = Y_k - theta X_k is symmetric about 0 at the true iROAS under no
  interference beyond spend and a common iROAS (Chen and Au, "Robust causal
  inference for incremental return on ad spend with randomized paired geo
  experiments", Ann. Appl. Stat. 16 (2022) 1-20, doi:10.1214/21-AOAS1493,
  Assumptions 0 and 1, Proposition 2). The region constraint limits which
  supergeos form and leaves the coin alone: the design is a stratified paired
  design.
- The estimand adds up. The iROAS is total incremental response over total
  incremental spend (Chen and Au, eq. 1.2), and the empirical estimator
  (eq. 3.3) splits exactly: theta_hat = sum_r Y_r / sum_r X_r = sum_r w_r
  theta_hat_r, with Y_r and X_r region r's sums, theta_hat_r = Y_r / X_r and
  w_r = X_r / sum_s X_s. The national iROAS is the spend-weighted average of the
  regional ones.
- The design objective is the aggregate's variance. Var(theta_hat) =
  sum_k D_k^2 / B^2 (phase 5) splits by region into sum_r cost_r / B^2, so the
  regional certificates minimize the national estimate's variance under the
  constraint, and (cost with regions - cost without) / B^2 is the variance the
  constraint costs. Report it with every regional design.
- Three estimators, by what is assumed:
  - one iROAS everywhere: pool every region's supergeos into one Trimmed Match
    fit (Chen and Au, eq. 5.4) with its interval (eqs. 5.6 and 5.7);
  - an iROAS per region (theta_g = theta_r): Trimmed Match within each region,
    combined with the weights w_r; a pooled fit would trim regions unevenly and
    shift their weights, the trimming bias of Chen et al. (2023, section 4);
  - which of the two: the signed-rank test on the residuals e_k(theta_hat) that
    Chen and Au use to check Assumption 1 (section 8), run within each region at
    the pooled estimate.
- The combined interval cannot come from inverting one theta. Var = sum_r w_r^2
  Var(theta_hat_r) holds when the spend differences are fixed (go-dark, or spend
  prespecified per geo: the cases of Chen and Au's Assumption 0); otherwise a
  bootstrap stratified by region.
- Lift in percent fits the same frame. If treatment multiplies the response by
  phi = 1 + lift, then e_k(phi) = R_T,k - phi R_C,k = phi (Z_1 - Z_2) A_k is
  symmetric about 0, so Trimmed Match applies with the control half's response
  in the place of spend; the empirical lift is sum_r R_T,r / sum_r R_C,r - 1,
  the regional lifts weighted by their control-side response.
- Build: a study function taking a regional design and post-period responses
  (and spend, when there is any) and returning the regional and pooled
  estimates (empirical and Trimmed Match, iROAS or lift), the spend-weighted
  aggregate and its interval, the per-region symmetry check and the variance
  the constraint costs. Test-first, against simulations with planted regional
  effects, equal and unequal across regions. Own branch, after phase 6.

### Phase 7: benchmarks and docs (separate workstream)

- `benchmarks/cases/trimmed_match.py`, cross-validation against the reference
  implementation: the estimate, the trimming and the interval on the README's
  toy example, on `example_data_for_post_analysis.csv` and on datasets from
  `simulation_script.py` at fixed seeds match google/trimmed_match, following the
  benchmarks' existing pattern for pinned references. A pinned installer (`benchmarks/reference/install_trimmed_match.sh`)
  builds the reference at a fixed commit (head at the time of writing:
  0be855234ee3a589eaddc8b945e2a0d924ad8791) into its own virtual environment,
  since it is tested with Python 3.7 and builds a C++ core with bazel while
  mlsynth runs on a current Python; the case runs it live through a subprocess,
  exchanging JSON, raises BenchmarkSkipped when that environment is absent (as
  `drosc_basque.py` does for R), and writes `benchmarks/reference/trimmed_match/`
  with the manifest, the provenance (reference commit, Python and bazel
  versions) and the cell-by-cell comparison. Building that environment is the
  first task, and the installer pins every dependency.
- `benchmarks/cases/pangeo_power.py`: the closed-form variance against
  simulation, and the power analysis against `MatchedPairsRMSE` and
  `report_candidate_designs` on the same pairs, evaluation window, budgets and
  hypothesized iROAS (`example_data_for_design.csv` and `example_geo_pairs.csv`),
  go-dark and heavy-up.
- `benchmarks/cases/pangeo_vs_tmd.py`: on `example_data_for_design.csv`, 174
  public geos, the certified supergeo design against the reference's
  cross-validated pairs, compared on held-out RMSE of the iROAS estimate at the
  same budget; a public panel for the supergeo study's own measurements.
- The datasets are Apache 2.0 and are read from the pinned clone at run time,
  with the commit recorded in the provenance.
- `benchmarks/cases/pangeo_certificate.py`: certified optima on fixed panels.
- `docs/replications/` pages for each, and the PANGEO docs page's Verification
  pointer, solver section and power-analysis section.
- The supergeo paper's results (Chen et al. 2023, Tables 1 to 3) inject synthetic
  effects into two private advertising datasets of the 210 US DMAs, so neither
  path A nor path B applies; the Trimmed Match case is cross-validation.

### Phase 8 (open): designs that survive losing markets

A design is fixed before the test, and markets can be lost during it: a data
outage, contamination from a national campaign, a market withdrawn by its owner.
Recoverable robustness (Büsing, "Recoverable Robustness in Combinatorial
Optimization", dissertation, TU Berlin, 2010; Cuvillier, 2011) models this in two
stages: choose a design now, then, once a scenario is revealed, repair it within
a stated recovery rule, and optimise the first-stage design against the worst
repaired outcome. Two of its recovery rules fit geo designs:

- exact subset recovery (chapter 4): the repair may only keep part of the
  design, so the groups containing a lost market are dropped;
- k-distance recovery (chapter 2): the affected groups may be re-formed, with at
  most k markets moved.

Its Γ-scenarios (at most Γ elements fail; sections 2.4, 3.4, 4.4) are "at most Γ
markets lost".

Two limits. The thesis solves knapsack, shortest path, minimum cut, matroid bases
and disjoint hitting set, not matching or set partitioning, so the framework
carries over and the algorithms do not. And the objective decides whether the
problem exists: under supergeo_total a dropped group removes its term from the
sum, so losing markets lowers the cost and the robust problem is trivial. It
binds only under an objective that charges for lost markets, the variance of the
estimator (phase 5) or a floor on the number of groups or on the share of the
population covered.

Measurement first, any time after phase 0b: on the certified designs of the
study's panels, remove the worst single market and the worst pair (N and
N(N-1)/2 scenarios), drop the groups they sit in, and report the change in
balance and in the closed-form variance. A small change closes the phase as a
note; a large one opens a design phase, starting from the Γ-scenario model and
the extended cover inequalities of the thesis's recoverable robust knapsack
(chapter 5).

Trimmed Match is a different mechanism: its estimator trims the pairs with
extreme effects after the test, and its design's pair exclusion chooses how many
pairs to keep. Neither repairs a design after a market is lost.

## Order and dependencies

Phase 0, then 0b (the paper's objective, solved by enumeration and the cvxpy and HiGHS MIP, the base every later phase is measured against), then 0c (the general solver against the closed form on pairs: heuristic stopping, odd-set cuts, the matching oracles), then 1 (independent, small), then 2 and 3 in sequence, then 4. Phase 3b follows 3, its items in their listed order, with threads after all of them. Phase 5's
closed form and backtest can start after phase 1 and run beside 2 to 4; its
Trimmed Match simulations follow phase 6, and phase 6b follows phase 6. The Trimmed Match benchmark (phase 7)
lands with phase 6. Phases 5b, 5c and 5d follow phase 5's held-out MDE, which
is their measurement, and 5d uses 5c's test on its clusters. Phase 5e follows phase 5's
closed form and backtest, needs a spend column, and adds its row to the exact paths of phases 2 and 3 as they land. 5b's inference change and its citation of the pairs path can
land any time. The rest of phase 7 after each phase lands. Phase 8 is open:
its measurement can run any time after phase 0b, and any design work waits on
phase 5's variance.

## Risks

- Q > 2 grows the groups to 2Q markets: pricing enumerates more groups per scan
  (at 50 markets, 15.9 million groups of 6) and each score enumerates more
  splits; phase 2 measures where that stops being affordable.
- The certificate past 62 markets costs minutes (study: 102 s at 64, 566 s at 80);
  at a few hundred markets a proof is out of reach (study: 85,547 survivors at
  210), and the deliverable there is a design with a proven gap.
- The objective overfits whatever the solver: phase 5 is what turns an optimal
  design into a better experiment.
- The column LP is 8 to 26% below the optimum on the arms measured, and the
  cuts, branching rules and relaxations measured in phase 3b close at most 14%
  of that gap: PANGEO's cost rewards cancellation, as vector number partitioning
  does. An exact proof stays out of reach past about 100 markets at Q = 2, and
  the deliverable there is a design with a certified gap (phase 3b: POPMUSIC
  and the iterative cutoff; 6.1% with every default at 120 markets, 4.6%
  against the best design known). Panels whose markets fall into latent groups
  are the exception: their column LP sits 1.4 to 4.6% below the optimum and
  the proof finishes at 120 markets.

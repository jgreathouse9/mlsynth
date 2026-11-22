# Supergeo base problem: certified search and branch, cut and price

An exploratory study, not a benchmark case or library code. Timings are
machine-dependent and were taken on one core.

## The problem

Chen, Doudchenko, Jiang, Stein and Ying (2023) partition `N` geos into
supergeos, groups of 2 to 4, and split each group into a treated side and a
control side. With `z_i` a geo's pretest total, a group `G` costs

    score(G) = min over non-empty proper A of (2 Z_A - Z_G)^2,

the squared imbalance of its best split, and a design minimises the sum over its
groups. It is a set-partitioning problem with one column per group: about 80
million columns at `N = 210`. It contains number partitioning, so it is strongly
NP-hard.

## Method

- Column generation over the set-partitioning master (`sg.py`), with exact
  pricing: a full scan of every group of 2 to 4 (`sg._scan`), or a
  meet-in-the-middle search that lists the signed halves of at most two geos,
  sorts them, and searches a window around each (`mitm.py`, `mitm_par.py`).
- The Lagrangian bound `sum(pi) + (N / lo) min(0, min reduced cost)`, valid for
  any duals, and reduced-cost elimination: a column whose reduced cost exceeds
  the gap between the incumbent and the bound cannot be in a better design.
- A parity cut for odd `N` (`parity.py`): halving and rounding the partition
  rows gives "at least one odd-size group", a Chvatal-Gomory cut over every row.
- Subset-row cuts on triples of rows (`sr.py`; Jepsen, Petersen, Spoorendonk and
  Pisinger 2008).
- An exact layer (`mitm_exact.py`, `exactnb.py`, `verify_nb.py`,
  `certify_exact.py`): floating point finds a superset under a proven rounding
  margin (Rump 2010, Sect. 2-3), and membership, the bound and the optimum are
  decided exactly, by `Fraction` or by error-free transformations (TwoSum,
  Dekker's TwoProduct, Shewchuk's expansions) in numba. The final optimum over
  the surviving columns is proven by an exact-cover search with
  fewest-fitting-columns branching.
- Branch and price with Ryan and Foster (1981) pair branching (`bnp.py`), and
  branch, cut and price (`bcp.py`): subset-row cuts at every node, and branching
  on the group of the fractional market with the fewest viable groups.

## Findings

1. Column generation is cheap; the bound is weak. On all 40 GeoLift markets
   (4 pretest weeks) the LP bound is 0.27 of the optimum. At `N = 210`
   (simulated, lognormal sizes) column generation converges in about 40 s and
   the bound is 2.8e-6 against a best design of 0.0039. A MIP over the 85,547
   surviving columns did not close that gap in 25 minutes.
2. The pretest optimum overfits. On GeoLift the certified optimum scored no
   better on the following 4 weeks than designs 100 to 10,000 times worse on the
   pretest.
3. The integrality gap has two kinds. Arithmetic cancellation (number
   partitioning) leaves 0.27 on GeoLift, and every violated subset-row cut lifts
   it only to 0.33. Parity (odd `N`, every geo near the same size) is closed
   exactly by the one parity cut: on the 27-geo panel the bound equals the
   optimum in 24 of 25 windows.
4. HiGHS's "Optimal" is optimal to about 1e-6. Over the 25 windows of the
   27-geo panel the exact search found HiGHS's design above the exact optimum in
   17, by up to 5e-5 relative. The certificate in `certify_exact.py` is exact
   from end to end, including the rescaling of the costs, which uses a power of
   4 so that `z * 2^k` is exact.
5. On small GeoLift subsets an outlier explains a weak root. On one 12-market
   subset a market 2.3 times any other has to sit one against three; the LP
   spreads it over two near-balanced groups whose leftovers cannot balance
   (root bound 0.004 of the optimum). Group branching on that market proves the
   optimum in 23 children, against 419 nodes of plain pair branching.

## Branch, cut and price on 12-market GeoLift subsets

Five subsets (seed 2026), every optimum checked against brute force over every
partition (`python -m benchmarks.studies.supergeo.scale 12 5`):

| subset | branch and price | branch, cut and price | root bound / optimum with cuts |
| --- | --- | --- | --- |
| 0 | 419 nodes, 44.1 s | 68 nodes, 17.1 s; 24 nodes, 6.3 s with a 2 s incumbent | 0.051 |
| 1 | 3 nodes, 0.5 s | 1 node, 0.2 s | 1.000 |
| 2 | 19 nodes, 2.0 s | 1 node, 0.3 s | 1.000 |
| 3 | 5 nodes, 0.9 s | 1 node, 0.7 s | 1.000 |
| 4 | 9 nodes, 1.1 s | 1 node, 0.5 s | 1.000 |

Without an incumbent every group of the outlier counts as viable, the count
exceeds the group-branching limit (64), and the search pair-branches until it
finds a design.

## Scaling to 16 and 20 markets

Three subsets each (seed 2026), branch, cut and price with a 2 s exact-repair
incumbent, every optimum checked against the exact certificate
(`python -m benchmarks.studies.supergeo.scale K 3 300`):

| K | subset | branch and price | branch, cut and price | root bound / optimum: plain, with cuts | group / pair branches |
| --- | --- | --- | --- | --- | --- |
| 16 | 0 | 3 nodes, 0.6 s | root, 2.6 s | 0.536, 1.000 | 0 / 0 |
| 16 | 1 | 121 nodes, 13.7 s | 35 nodes, 8.4 s | 0.246, 0.425 | 2 / 0 |
| 16 | 2 | 61 nodes, 6.3 s | 10 nodes, 4.4 s | 0.203, 0.671 | 1 / 0 |
| 20 | 0 | 1,337 nodes, 124.5 s | 346 nodes, 43.8 s | 0.055, 0.093 | 26 / 0 |
| 20 | 1 | 53 nodes, 6.0 s | 20 nodes, 6.1 s | 0.248, 0.401 | 3 / 0 |
| 20 | 2 | 161 nodes, 16.7 s | 55 nodes, 11.4 s | 0.222, 0.352 | 5 / 0 |

Branch, cut and price takes 2.7 to 3.9 times fewer nodes at 20 markets, and
group branching did all of its branching.

## Rank-1 cuts on 4 and 5 rows

Root bound over the exact optimum after separating to a fixed point, subset-row
cuts only against every family of `rank1.py` (3 rows with 1/2; 4 rows with 1/3
and with (2/3, 1/3, 1/3, 1/3); 5 rows with 1/3 and with 1/2)
(`python -m benchmarks.studies.supergeo.rank1_bounds`):

| instance | no cuts | subset-row cuts | every rank-1 family |
| --- | --- | --- | --- |
| 12 markets, subset 0 (outlier) | 0.004 | 0.051 | 1.000 (170 cuts, 1 s) |
| 16 markets, subset 0 | 0.536 | 1.000 | 1.000 |
| 16 markets, subset 1 | 0.246 | 0.425 | 0.702 (191 cuts, 3 s) |
| 16 markets, subset 2 | 0.203 | 0.671 | 0.961 (171 cuts, 2 s) |
| 20 markets, subset 0 | 0.055 | 0.093 | 0.173 (161 cuts, 4 s) |
| 20 markets, subset 1 | 0.248 | 0.401 | 0.634 (163 cuts, 1 s) |
| 20 markets, subset 2 | 0.222 | 0.352 | 0.426 (147 cuts, 2 s) |
| 40 markets (all GeoLift) | 0.273 | 0.326 | 0.357 (174 cuts, 13 s) |

The 4- and 5-row cuts close the coupling gap: the outlier subset, whose bound
subset-row cuts left at 0.051, becomes exact at the root. They close less of the
cancellation gap, and less as the number of markets grows: 0.31 of what
subset-row cuts left at 16 markets on average, 0.1 to 0.4 at 20, 0.05 at 40.

### Rank-1 cuts inside branch, cut and price

Every rank-1 family separated at every node (`BCP(families=rank1.FAMILIES)`)
against subset-row cuts alone, 2 s incumbent, optima checked against the exact
certificate (`python -m benchmarks.studies.supergeo.compare_cuts K 3`):

| instance | subset-row cuts | every rank-1 family |
| --- | --- | --- |
| 16 markets, subset 0 | root, 3.4 s | root, 3.4 s |
| 16 markets, subset 1 | 35 nodes, 10.9 s | 7 nodes, 6.8 s |
| 16 markets, subset 2 | 10 nodes, 4.1 s | 10 nodes, 5.5 s |
| 20 markets, subset 0 | 346 nodes, 43.0 s | 104 nodes, 84.4 s (850 cuts) |
| 20 markets, subset 1 | 20 nodes, 5.9 s | 5 nodes, 5.3 s |
| 20 markets, subset 2 | 55 nodes, 11.2 s | 21 nodes, 13.2 s (395 cuts) |

The rank-1 families cut nodes by up to 5 times. Time follows only when the cut
pool stays small: pricing evaluates every cut for every column, and separation
over row sets of 4 and 5 runs at every node, so on the hardest subset 850 cuts
made each node 6.5 times dearer than with subset-row cuts.

### Cut management

`BCP(..., purge_after=5, deep_depth=1, max_active=250)`: a cut is dropped after 5
LP solves with a zero dual, every family is separated only to depth 1 (subset-row
cuts below), and at most 250 cuts stay in the pool. Same subsets, optima checked
against the exact certificate:

| instance | subset-row cuts | rank-1 | rank-1 managed |
| --- | --- | --- | --- |
| 16 markets, subset 0 | root, 3.9 s | root, 3.3 s | root, 2.6 s |
| 16 markets, subset 1 | 35 nodes, 11.2 s | 7 nodes, 7.0 s | 7 nodes, 6.7 s |
| 16 markets, subset 2 | 10 nodes, 4.1 s | 10 nodes, 5.4 s | 10 nodes, 4.9 s |
| 20 markets, subset 0 | 346 nodes, 44.1 s | 104 nodes, 85.0 s (850 cuts) | 158 nodes, 44.7 s (largest pool 221) |
| 20 markets, subset 1 | 20 nodes, 5.9 s | 5 nodes, 5.4 s | 5 nodes, 5.0 s |
| 20 markets, subset 2 | 55 nodes, 12.1 s | 21 nodes, 13.5 s (395 cuts) | 21 nodes, 8.9 s (largest pool 184) |

Managed rank-1 cuts are the fastest or level on every instance and keep most of
the node reduction. On the hardest subset they give up part of it (158 nodes
against 104) for nodes six times cheaper, and match subset-row cuts in time with
under half their nodes.

### Tuning the depth limit and purge age

Grid over the depth to which every rank-1 family is separated (0, 1, 2, 3, every
depth) and the purge age (2, 5, 10, never), pool capped at 250, on 20-market
subsets 0 to 2; the best two and two baselines then on held-out subsets 3 to 5
(`python -m benchmarks.studies.supergeo.tune_cuts tune` / `validate`). All 75 runs
found the optimum the exact certificate gives.

| setting | tuning: time, nodes | held out: time, nodes |
| --- | --- | --- |
| every depth, purge 2 | 57.5 s, 130 | 42.6 s, 69 |
| depth 2, purge 2 | 58.7 s, 130 | 43.0 s, 69 |
| depth 1, purge 5 | 71.7 s, 184 | 46.2 s, 74 |
| subset-row cuts | 67.4 s, 421 | 46.5 s, 192 |
| every depth, never purge | 79.9 s, 130 | |
| depth 0, never purge | 118.8 s, 380 | |

The purge age sets the cost of a node: a cut dropped after 2 LP solves with a zero
dual is fastest at every depth, and never purging costs about 40% more time for
the same tree. The depth limit sets the tree: depth 0 loses the whole node
reduction and depth 1 about half of it, and from depth 2 the tree is the one
unrestricted separation gives. With purge age 2 no depth limit is needed
(`bcp.managed()`): 2.8 to 3.2 times fewer nodes than subset-row cuts, in 8 to 15%
less time. Per instance the time is mixed (held-out subset 3: 13.0 s against
9.2 s), so the gain is in total time and in nodes.

The managed property test found a defect in both searches: with integer sizes the
root LP can be 0, so the cost rescale fired at a later node while the incumbent
and the open bounds stayed in the old units (2/128 reported for an optimum of 2).
The scale is now fixed after the root.

## Starting heuristics

Cost of the starting design over the exact optimum
(`python -m benchmarks.studies.supergeo.heuristics`; `N = 210` left out for now):

| instance | pairs + repair 2 s | greedy | GRASP 2 s | GRASP + LP-guided MIP 1 s |
| --- | --- | --- | --- | --- |
| 12 markets, 5 subsets | 1 on all | 1 to 652 | 1 on all | 1 on all |
| 16 markets, 3 subsets | 1 on all | 4.5 to 4,813 | 1 on all | 1 on all |
| 20 markets, subset 0 | 1 | 197 | 1.026 | 1.026 |
| 20 markets, subset 1 | 1.064 | 1.064 | 1 | 1 |
| 20 markets, subset 2 | 1 | 4,480 | 1.274 | 1 |
| 40 markets (all GeoLift) | 53.7 | 932 | 2.14 | 2.14 |

The largest-first greedy alone is a poor design: the last markets it places are
forced together however badly they balance. Up to 20 markets the incumbent the
search already used was optimal on 10 of 11 instances, so the nodes in the
scaling tables above go to the proof, not the search for a design. At 40 markets
GRASP is 25 times closer to the optimum than pairs plus repair, and a 1 s MIP over
the root's columns did not improve it; the earlier restricted master over 5,000
low-reduced-cost columns found the 40-market optimum in 18 s. The cuts close less of the root gap
as `K` grows (all of it on four of five 12-market subsets; 0.09 to 0.40 of the
optimum at 20 markets), so the search depends more on the incumbent.

## Fifty markets: why branch, cut and price is slow

Two 50-market instances (`fifty.py`): the 50-state, 40-month panel of the SYNDES
benchmark (`basedata/urate_cps.csv`, months 34 to 37 summed) and 50 lognormal
sizes (seed 2026). Every run below used one thread per process, four processes
on four cores, 30 minutes; the profile ran alone. Measurements: `rca50.py`.
Certified optima: `fifty_optima.json`.

Incident. Both arms (subset-row cuts; `bcp.managed()`) hit the time limit on both
instances. The exact certificate solved both, in 24 s (lognormal) and 293 s
(50-state).

| instance | arm | nodes | incumbent / optimum | bound / optimum |
| --- | --- | --- | --- | --- |
| 50-state | subset-row | 3,371 | 48.9 | 0.758 |
| 50-state | tuned | 210 | 48.9 | 0.445 |
| lognormal | subset-row | 4,740 | optimum + 0.0621 | optimum - 0.0004 |
| lognormal | tuned | 166 | optimum + 0.0621 | optimum - 0.0009 |

Time to solve is nodes times time per node, and both factors are wrong.

Cause 1, the incumbent. The search starts from pairs plus repair (10 s), 48.9 times
the optimum on the 50-state panel. Best-first search finds a leaf only when a node
LP is integral, and in the four incident runs the incumbent never moved, so no node
with a bound below 48.9 times the optimum is pruned. GRASP is no better there
(55.7 times). The certificate found the optimum with a MIP over the columns that
survive reduced-cost elimination at the root.

Cause 2, no reduced-cost fixing. At the optimum's cost only 422 (50-state) and 486
(lognormal) of the 251,125 groups can be in a better design. The search still
prices all 251,125 at every LP (51% of a subset-row node) and keeps every column it
has priced in the master.

| start | universe | 50-state | lognormal |
| --- | --- | --- | --- |
| pairs + repair | every group | time limit, bound 0.758 | time limit |
| optimum | every group | time limit, 8,218 nodes, bound 0.973 | time limit, 4,849 nodes |
| pairs + repair | root survivors | time limit, 6,895 nodes, bound 0.868 | time limit, 18,682 nodes |
| optimum | root survivors | optimal, 9,059 nodes, 488 s | optimal, 22,405 nodes, 752 s |

Each cause is necessary: removing either one alone leaves both instances unsolved
at 30 minutes, and removing both solves both, at the certificate's optimum. The
fixed universe cuts the time per node 4.1 times (0.219 to 0.054 s, 50-state) and 11
times (0.371 to 0.034 s, lognormal); at the unrestricted rate the 9,059 nodes the
50-state search needs from the optimum take about 1,980 s, past the limit, where
the unrestricted run stopped with its bound at 0.973.

Cause 3, group branching on lognormal. `_viable` keeps a group when its loss is
below the incumbent. When one group carries nearly all of the cost (7,616 here),
almost every group passes: 179,257 of them, a median of 14,990 per market and over
64 on 49 of 50 markets, so group branching (at most 64 children) applies to one market and
the search branches on pairs (9,960 of 10,027 branches even over the survivors).
The reduced-cost test leaves 486 groups, at most 64 per market, on all 50. Its
effect on the node count is not measured yet. On the 50-state panel the two tests
agree (454 and 422 groups) and group branching already runs.

Cause 4, the tuned setting at 50 markets. A tuned node took 12.4 s against 1.15 s
(first 30 nodes, 50-state, alone):

| per node | subset-row | tuned |
| --- | --- | --- |
| separation | 0.001 s | 7.15 s |
| master LP | 0.30 s | 3.27 s, 2.47 s of it `Cut.coef` in Python (47 million calls) |
| pricing | 0.58 s | 1.67 s |
| branching | 0.15 s | 0.15 s |

Separation enumerates every row set of 4 and 5 among the fractional rows: 15,504
five-row sets at 20 markets, where the setting was tuned, and 2,118,760 at 50.

With causes 1 and 2 removed the tree still needs 9,059 and 22,405 LP nodes, at 34
to 54 ms each, where the certificate's exact search covers the same survivors in
7.5 million nodes in 7 s. The root LP is 0.135 of the 50-state optimum, so the LP
bound rises slowly under branching, and a node that costs a microsecond and
carries a weak bound wins over a node that costs 50 ms and carries the LP bound.

### After the fixes

Fixes 1 and 2 are in `bcp.py`: reduced-cost fixing at the root (`fix_columns`,
on by default) and a set-partitioning MIP over the fixed universe after the root
(`root_mip_seconds`, default 300; gap 0, since HiGHS's default relative gap of
1e-4 stops at a warm start that close). Same runs as the incident (four
processes on four cores, 30 minutes, pairs plus repair for 10 s first):

| instance | arm | before | after |
| --- | --- | --- | --- |
| 50-state | subset-row | time limit, incumbent 48.9 x optimum | optimal, 741 s, 8,968 nodes |
| lognormal | subset-row | time limit, incumbent optimum + 0.062 | optimal, 709 s, 23,020 nodes |
| 50-state | tuned | time limit | time limit (cause 4), bound 0.63 of the optimum |
| lognormal | tuned | time limit | time limit (cause 4) |

Both optima match the certificate. In every run the root MIP reached the
optimum, and the universe fell from 3,284 and 3,188 groups to 421 and 487.
Plunging dives were measured as the alternative to the MIP: 9.4 and 21.8 times
the 50-state optimum after 900 s, no improvement on lognormal.

The certificate's exact-cover search (`verify_nb`) over the root survivors, from
the same pairs-plus-repair start and with no MIP at all, is faster than either:

| instance | exact search | branch, cut and price after the fixes | certificate with the HiGHS MIP |
| --- | --- | --- | --- |
| 50-state | 40.9 s, 39.0 million nodes | 741 s | 293 s |
| lognormal | 18.0 s, 7.5 million nodes | 709 s | 24 s |

A search node costs about a microsecond and carries a weak bound; an LP node
costs 30 to 50 ms and carries the LP bound, and at 50 markets the cheap node
wins. The exact search holds a group as a 64-bit mask, so it stops at 62
markets; past that the LP tree is the route.

Three defects found on the way, each pre-existing:

- The cost rescale in `rank1.root_bound`, `sr.run`, `bcp` and `bnp` had no
  ceiling: an intermediate LP of 1.6e-30 (0 up to noise) scaled the costs by
  4^50, and HiGHS returned "Unknown". The rescale is now once and capped at
  costs of 1e10 (`rank1._rescale_power`).
- The artificials cost a fixed 1e7. After a rescale a column cost 9.3e9, a
  feasible root master covered a market artificially, the root was pruned as
  infeasible, and the search returned no design (sizes 6.1, 0.2, 7.9, 4.7, 2.4,
  7.7, 6.1, 1.0). The artificials now cost 1e3 times the dearest column, and
  when one is still in use a phase-1 LP decides between infeasible and "price
  too low" (`bnp.solve_feasible`).
- A test of the default MIP gap on 9 markets could not tell it from gap 0; the
  test now uses 20 markets, where it can.

Mutants (`tools/mutation/targets.toml`, targets `supergeo-bcp`, `supergeo-bnp`,
`supergeo-rank1`): 13 in all, 12 killed. `rescale-every-round` survives and is
recorded as equivalent: under the cap a second rescale cannot lift the costs past
1e10, so no test can see it.

## The staged certificate

`certify_exact.certify_staged` keeps the certificate's exact root bound and exact
elimination and drops the MIP. Each stage eliminates at the incumbent's exact cost
and runs the exact-cover search over the survivors, capped at 0.3, 1 and 3
million nodes; its best cover is the next incumbent, so the next stage has fewer
survivors. A final stage has no cap, and a stage that completes proves its
result. Alone on the machine, from a 1 s pairs-plus-repair start:

| instance | total | survivors by stage | search nodes | branch, cut and price | certificate with the MIP |
| --- | --- | --- | --- | --- | --- |
| 50-state | 8.0 s | 4,202, 941, 549 | 2.9 million | 260 s | 293 s |
| lognormal | 20.3 s | 4,308, 877, 509 | 1.5 million | 139.5 s | 24 s |

The search takes 3.4 s and 1.5 s of those totals. On lognormal most of the rest is
the exact elimination: one market of 1,182 against a median of 24.5 widens the
safe window to 188,825 groups at every stage.

Branch, cut and price needs about 8,000 LP nodes on these instances, set by the
root bound (0.135 of the 50-state optimum); a node costs about 30 ms after the
fixes (29.8 ms on the 50-state panel, 64% of it in the master LP). The exact
search's nodes cost about a microsecond with a weaker bound, and at 50 markets
that wins.

### Past 62 markets

`verify_nb` holds the uncovered set and each group as words of 63 bits above 62
markets (`_search_w`, the same tree as the one-word search). The staged
certificate on lognormal instances (seed 2026, 1 s heuristic), alone:

| markets | 3 capped stages | 6 capped stages | survivors by stage (6 stages) |
| --- | --- | --- | --- |
| 64 | 102 s | 119.6 s | 69,698, 2,107, 959, 917, 789 |
| 72 | 413 s | 521.9 s | 947,976, 25,256, 1,859, 1,346, 1,205, 1,194 |
| 80 | 566 s | 687.9 s | 138,889, 6,172, 1,883, 1,434, 1,408, 1,263, 1,104 |

After about three stages the incumbent is near the optimum: further stages shrink
the survivors little and each restarts the search, so they add time. What remains
is the proof, whose node count the weak bound sets. About half of the 72-market
time is outside the search: the exact costs (Fractions) of the first stage's
947,976 survivors and the arrays built from them. So staging should stop when the
survivors stop shrinking by more than about half, and the first stage's exact
bookkeeping is the next cost to cut.

### Float stages, automatic staging and heuristic time

At 72 markets the first stage's steps took: exact bound 24.0 s, window and exact
screening 22.8 s, exact costs in Fractions 102.6 s, search setup and 300,000 nodes
14.1 s. Capped stages now search on floating-point scores and only supply
incumbents; the final stage runs on exact costs and is the proof. Caps grow 3
times per stage, staging stops once the next survivors exceed `keep` times the
last stage's, and the heuristic gets max(1, N/24) s (at 72 markets 3 s leaves
62,899 first-stage survivors, 1 s 947,976). Alone, from the defaults:

| instance | before | keep 0.5 | keep 0.8 |
| --- | --- | --- | --- |
| 50-state | 8.0 s | 8.7 s | |
| lognormal 50 | 20.3 s | 21.8 s | |
| 64 | 102 s | 75.3 s | 89.4 s |
| 72 | 413 s | 266.6 s | 268.0 s |
| 80 | 566 s | 685.1 s | 590.6 s |

The heuristic runs for a time, so its design and the first stage's survivors vary
between runs (62,899 and 42,927 at 72 markets); differences of 15% are within
that. From 64 markets on, the final exact stage is most of the time (20 s, 160 s,
410 to 560 s), set by the 1,000 to 1,600 survivors it starts from and the weak
bound. `keep` stays 0.5.

### The exact stage on threads

`verify(..., workers=k)` cuts the exact-cover tree at the top into at least 64 k
subtrees, expanding the oldest node first and recording any complete cover met on
the way, then searches them in increasing order of node bound on k threads. The
subtree search is the serial one from a given node, compiled without the GIL, and
prunes on one shared double holding the incumbent's upper value. Two threads can
race on writing it and leave the larger of two values. Every value written is the
upper value of a recorded cover, so the race costs pruning and never a cover.
`certify_staged` runs its exact stage on min(4, CPU count) threads.

The final stage's search alone, on the inputs captured from one run of the
certificate at each size (lognormal instances, seed 2026; the 50-state panel),
4 cores, nothing else running:

| markets | survivors | this thread | 1 thread | 2 threads | 4 threads | nodes (serial, 4 threads) |
| --- | --- | --- | --- | --- | --- | --- |
| 50-state | 549 | 1.51 s | 1.75 s | 0.68 s | 0.38 s | 1.6, 1.2 million |
| 64 | 928 | 21.0 s | 16.0 s | 7.6 s | 3.75 s | 11.0, 10.9 million |
| 72 | 1,219 | 140.3 s | 114.9 s | 58.5 s | 32.7 s | 70.2, 73.4 million |
| 80 | 1,588 | 489.7 s | 422.2 s | 208.6 s | 79.7 s | 224, 159 million |

Every run proved the same optimum. One thread beats the serial walk at 64 markets
and above because the subtrees are searched in bound order. At 80 markets four
threads take 5.3 times less than one: a thread finds the better incumbent sooner
and the search visits 26% fewer nodes.

### Where the time goes at 72 markets

With the exact stage on four threads the 72-market certificate takes 113 to 129 s
(the heuristic varies between runs). A profile of one run (128.1 s):

| step | time |
| --- | --- |
| the safe window, five builds | 68.6 s |
| exact screening of the window (`members2`) | 9.9 s |
| exact stage's search | 33.2 s |
| capped stages' searches | 9.8 s |
| heuristic | 3.0 s |
| exact costs in Fractions | 0.6 s |

Exact rational arithmetic is no longer a cost: the capped stages search on
floating-point scores, so exact costs are computed only for the final stage's
survivors and the covers found. On 200,000 groups at 72 markets, exact costs as
Python integers over one common power of two (2^54) took 1.29 s, gmpy2's `mpq`
1.94 s and `Fraction` 10.73 s, with identical values; at 0.6 s a run, either
saves under a second.

The window is the cost. Its radius is set by the duals, not by the threshold: at
thresholds from 0 to 1.29 it holds 886,723 to 886,726 groups and takes 13.7 to
14.9 s to build, for 19 to 42,927 survivors. Each stage's threshold is at most the
last one's (the incumbent only improves), so each stage's survivors are a subset
of the last stage's and can be screened from them instead of from a new window.

### One window

The first stage now screens the window and each later stage screens the last
stage's rows (`_screen_rows`). On the 72-market run above the later stages'
screens took 0.1 s and 0.0 s (42,927 to 2,444 rows, 2,444 to 1,443) in place of
two window builds of 13.6 to 14.9 s each and their screens; the bound's window
and the first stage's stay (the bound's is built before m is known, and a window
at a lower threshold need not hold the first stage's survivors). The certificate
end to end from the defaults (heuristic max(1, N / 24) s, exact stage on four
threads), alone:

| instance | before (keep 0.5, one thread) | now | stages: survivors of rows screened |
| --- | --- | --- | --- |
| 50-state | 8.7 s | 5.2 s | 4,202 of 4,202; 941 of 4,202; 549 of 941 |
| 64 | 75.3 s | 28.1 s | 18,146 of 539,298; 1,586 of 18,146; 928 of 1,586 |
| 72 | 266.6 s | 79.5 s | 42,927 of 886,726; 2,444 of 42,927; 1,443 of 2,444 |
| 72, second run | | 78.6 s | 62,899 of 886,730; 3,079 of 62,899; 1,443 of 3,079; 1,219 of 1,443 |
| 80 | 685.1 s | 128.5 s | 56,426 of 1,200,904; 6,109 of 56,426; 2,202 of 6,109; 1,588 of 2,202 |

The same optima as before at every size. At 80 markets the exact stage is now
76.4 s of the 128.5 s, and the two window builds most of the rest.

### A stated gap, a balance standard, and 120 markets

`certify_staged(..., gap=g)` proves that no design costs less than (1 - g) times
the design it returns: it stops once the exact bound is within g, eliminates at
(1 - g) times the incumbent, and the exact stage searches below that. The result
reports `gap_proved` and `lb_proved`. `max_group_cost=t` allows only groups whose
imbalance is at most t, proves infeasibility when no partition meets it, and
reports the design's largest group cost. Both are exact; the docstrings give the
reasons and the citations (Mertens 1998 and 2001 on the random-cost landscape of
partitioning; Chen, Longfils and Remy on balance checks). The threaded search's
node limit now bounds the total over the threads (four threads on a 4e8 limit had
stopped at 1.6e9 nodes).

The exact proof's tree, estimated by random probes (Knuth 1975) at the best known
incumbent, and the gap the exact bound proves before the exact stage:

| markets | gap the bound proves | exact tree (nodes) | gap 1e-6 |
| --- | --- | --- | --- |
| 64 | 7.4e-8 | 4.9 million | no search |
| 80 | 8.2e-8 | 75 million | no search |
| 96 | 3.1e-7 | 8e9 | no search |
| 120 | 1.1e-6 | about 7e21 (at the start incumbent) | |

On lognormal panels the bound is within 1e-7 of the optimum and the exact proof
spends minutes (80 markets) to never (120) closing that sliver. At 120 markets
the certificate with g = 1e-6 took 147.4 s and proved a gap of 8.6e-7 from three
capped stages, with no exact search.

Similar-sized markets are the opposite case (120 draws of N(100, 10), seed 2026).
The bound is 1.8e-5 of the design, so no useful relative gap is provable (g =
1e-2 did not finish in 25 minutes). The design is nonetheless close to perfect in
absolute terms: its groups' halves differ by 0.12 median spacings between market
sizes (rms over 32 groups; 0.27 at the worst group), against 4.8 on the
lognormal panel. The optimum sits near zero, where near-perfect partitions are
plentiful (the easy phase of Mertens 2001), and a relative gap divides by it. For
such panels the certificate that means something is absolute: the design is
within its own cost, here 0.47 squared spacings, of perfect balance, since the
bound is at least 0.

`certify_staged(..., abs_gap=d)` proves the design within max(g UB, d) of the
optimum, d in the objective's units. Independent noise of standard deviation
sigma in each market's total adds N sigma^2 to any design's expected cost, which
sets the scale for d. On the similar-sized 120-market panel, from the defaults:

| d (squared median spacings) | time | design's cost | stages |
| --- | --- | --- | --- |
| 1.0 | 10.9 s | 0.471 | none (the heuristic's design is within d of 0) |
| 0.1 | 25.6 s | 0.0125 | one capped stage, 140,228 survivors |
| 0.01 | 35.2 s | 0.00298 | two capped stages, 156,339 and 14,593 survivors |

A tighter d sends the capped stages after a better design: at d = 0.01 the
design costs 160 times less than the heuristic's, its groups' halves differing
by 0.0097 median spacings (rms over 32 groups), and the bound (at least 0)
proves it within d with no exact stage.

### 120 markets: capped stages on threads, one window

Every step of the certificate at 120 markets, timed alone (lognormal panel with
gap 1e-6; similar-sized panel with absolute gap 0.01), before and after two
changes: the capped stages run on the threads of the exact stage
(`verify_rows(..., workers=)`), and one safe window serves the bound and the
first stage, built at a threshold guessed from the floating-point minimum
reduced cost (the stage builds its own when the exact one is above the guess).

| step | lognormal before | lognormal after | similar-sized before | after |
| --- | --- | --- | --- | --- |
| capped searches | 49.5 s | 10.3 s | 19.9 s | 2.2 s |
| safe window | 44.9 s (2 builds) | 27.7 s (1) | 0.3 s | 0.6 s (2) |
| exact screen | 14.8 s | 13.8 s | 0.4 s | 0.3 s |
| heuristic | 5.0 s | 5.0 s | 5.1 s | 5.0 s |
| root column generation | 2.5 s | 2.5 s | 2.6 s | 2.5 s |
| total | 117.6 s | 60.2 s | 28.5 s | 10.8 s |

On the captured first capped stage (563,453 rows, 300,000 nodes) one thread
took 21.0 s and four 6.1 s. Branching on the lowest uncovered market instead of
the market with the fewest fitting groups ran 17 to 35 times faster per node and
found no cover within the cap: it loses the early test for a market with no
fitting group, so the branching stays. On the similar-sized panel the guess fell
short and a second window was built, at 0.6 s in all.

What remains: on the lognormal panel the window (one thread, two passes over the
halves: count, then fill) and its exact screen (two passes over the same rows,
the bound's test and the stage's) are 69% of the run; on the similar-sized
panel the fixed heuristic budget (N / 24 s) is 46%.

### The window and the screen

The exact screen (`exactnb.members2`) now decides each row by a floating-point
evaluation of rc - theta with a proven error bound, 16 u (A^2 + P + |tau| +
|theta|) (Higham's summation bounds; the filter in front of an exact predicate of
Shewchuk 1997), and sends only the rows within the bound to the exact test. On
the 120-market window (3,768,084 rows) it decided every row at the first stage's
threshold and all but 120 at theta = 0 (the LP's own columns, rc near 0), in
0.04 s against 6.0 to 7.8 s, every decision matching the exact test. The window
is counted and filled in parallel (`mitm_exact._window_safe_par`): 3.8 s
against 14.5 s for the serial count and fill of its 24,088,508 rows (3,768,084
distinct; a group is emitted once per split that passes).

| 120 markets, alone | before | after |
| --- | --- | --- |
| lognormal, gap 1e-6: total | 60.2 s | 24.0 s |
| safe window | 27.7 s | 6.4 s |
| exact screen | 13.8 s | 0.1 s |
| capped searches | 10.3 s | 9.4 s |
| heuristic | 5.0 s | 5.0 s |
| root column generation | 2.5 s | 2.3 s |
| similar-sized, absolute gap 0.01: total | 10.8 s | 9.6 s |

From 117.6 s before the threaded capped stages to 24.0 s on the lognormal panel. What remains there
is the capped searches (39%), the window (27%, of which the de-duplication of the
24 million emitted rows is 2.2 s) and the fixed heuristic budget (21%); on the
similar-sized panel the heuristic is 53%.

### 180 markets of a simulated weekly panel

A simulated panel of 180 markets over 354 weeks, with the time-series structure of
an index-valued weekly panel: level near 100 (unit levels' sd 16), unit trends
(sd 19 points over the panel), seasonality at 35% of each series' variance whose
amplitude and phase differ by region (two regions, 30% and 70% of the markets:
amplitude 5.6 against 3.7, peak at 260 and 208 degrees), five persistent common
factors (variance shares 0.23, 0.16, 0.13, 0.06; lag-1 autocorrelation 0.89), and
AR(1) idiosyncratic noise (sd 3.8, coefficient 0.59) with Student-t innovations
(kurtosis of weekly changes 6.0, 2.2 moves beyond 4 sd per 1,000 unit-weeks).
Units are drawn jointly by a variance-corrected smoothed bootstrap and the factors
are AR(13) paths. The design input is each market's total over four weeks
(mean 405.7, sd 58.9).

The noise of a market's four-week total has sd 11.65 index points, 16.8 median
spacings between market totals, so noise adds N sigma^2 = 51,030 squared spacings
to any design's expected cost. The 7.5 s heuristic's design costs 0.82, 1.6e-5
of that.

| absolute gap d | time | design's cost | stages |
| --- | --- | --- | --- |
| 0.01 N sigma^2 (510) | 17.7 s | 0.82 | none |
| 0.001 N sigma^2 (51) | 17.7 s | 0.82 | none |
| 0.01 | 27.0 s | 0.0067 | one capped stage, 556,395 survivors |
| 0.001 | stopped after 10 minutes | | |

Any gap set from the noise is proved at once: the 17.7 s are root column
generation (9.7 s) and the heuristic's fixed budget (7.5 s). A design better than
the heuristic's changes the variance of the treated-minus-control comparison by
about one part in 60,000, so gaps below its cost do not bear on the experiment;
the run at d = 0.001 was stopped for that reason, not for the time.

### A reproducible design

Three runs of the certificate on that panel gave three designs, all within the
gap: the heuristic was seeded but stopped on the clock, so the steps it
completed depended on the machine's speed and load, and with the gap proved from
the heuristic's design that design is the result. Three designs certified as
equivalent tracked the population differently after the balanced weeks (mean
gap of the treated from the population, weeks 240 to 353: -0.56, +0.60 and -0.79
index points per market), so which one a run returned mattered.

The heuristic now runs a fixed number of steps, max(10, N // 2) by default, from
a seed (`heuristic_steps`, `seed`; the result records the steps that ran). Three
default runs on the 180-market panel, the third with three busy threads
competing for the machine, returned the same design (cost 0.825493, 90 steps).
In one session, alone, both arms run twice with every step timed, the time
budget took 21.7 and 22.2 s (82 and 84 steps) and the step budget 22.8 and
23.4 s: the 90 steps cost 1.1 to 1.5 s more, as documented (about N / 20 s
against N / 24 s). An earlier comparison, 24.1 s against the 17.7 s of the
table above, was across sessions: the root column generation, with identical
code and work (13 pricing scans, 16 LP solves), took 9.7 s in the earlier session
and 13.6 to 18.0 s in the later one (the old commit's root just as slow), and
the 120-market root moved from 2.3 to 2.6 s to 3.5 to 3.9 s, a slower host by
about 1.5 times. Timings in this README compare arms within one session.

### The heuristic stops when it stops paying

The fixed budget ran its N // 2 steps whatever they brought. On google/trimmed_match's
174 example geos in pairs (totals over 2020-01-01 to 2020-12-01), whose sorted
start is already optimal, the 87 steps were 5.4 of the certificate's 5.9 s.
Two rules now end the heuristic early, both counted in steps so the design stays
a function of the inputs and the seed (`lns.lns_dp`'s `patience` and `target`):

- a stall: `heuristic_patience` steps in a row without a cheaper design (30 by
  default);
- the bound: a design within the requested gap of the root LP bound b, at a cost
  at most max(b / (1 - g), b + d).

The result's `heuristic_stop` says which ended it ("budget", "stall", "bound").

The patience comes from per-step trajectories at four times the default budget,
seeds 0 and 1, on nine panels: the 50-state and 50-lognormal panels, two
120-market lognormal panels, two 120-market integer panels, the 174 geos in pairs
and in groups of 2 to 4, and the 180-market simulated panel. Inside the default
budget the longest run of steps without a cheaper design was 19 (the 180-market
panel, seed 0), so a patience of 30 leaves every one of those runs as it was; a
patience of 10 would have stopped that run at step 53 of 90 with a design 3.16
times as costly. Where the search keeps improving it improves to the end of a
budget four times the default (last improvement at step 596 to 706 of 696 to
720 on the 174 and 180-market panels), so the rules shorten easy runs and leave
the hard ones alone. The integer panels reach a design of cost 0, the root
bound, at step 12 to 22 of 60, and the bound rule stops them there.

End to end, one session, alone, the default against the rules (two runs each,
the same design and cost in every pair):

| panel | default | with the rules |
| --- | --- | --- |
| 174 geos in pairs, gap 0 | 5.15, 5.29 s (87 steps) | 2.05, 1.97 s (30 steps, stall) |
| 120 integer markets, gap 0 | 17.9, 16.9 s (60 steps) | 13.3, 15.5 s (22 steps, bound) |

On pairs the remaining 2.0 s are the root (about 0.2 s), 30 steps that cannot
improve, and the proof; the odd-set cuts of the roadmap's phase 0c step 2 would
make the bound equal the optimum at the root, and the bound rule would then stop
the heuristic before its first step.

### Reproducible threads

A run that needs search stages on several threads depended on thread timing as
well: each thread took the next subtree when free and pruned on the best design
any thread had found, so the incumbent a capped stage handed on, and the design
it led to, could differ from run to run. The stages now search their subtrees
in rounds by default (`deterministic=True`; `verify_nb._rounds`): within a
round each subtree prunes on the incumbent known when the round began and gets
its share of the node limit left, and the results merge in subtree order. The
search is then a function of its inputs, idle or under load.

A subtree cannot use a design found during its own round, and a round waits for
its slowest subtree, so rounds want many small subtrees. On the captured
80-market exact stage, four threads, in one session:

| schedule | subtrees a thread | subtrees a round, a thread | time | nodes |
| --- | --- | --- | --- | --- |
| dynamic | 64 | | 86.4 s | 157 million |
| dynamic | 1,024 | | 81.5 s | 138 million |
| rounds | 1,024 | 1 | 155.9 s | 147 million |
| rounds | 1,024 | 2 | 117.7 s | 156 million |
| rounds | 1,024 | 4 | 132.5 s | 184 million |
| rounds | 4,096 | 1 | 157.9 s | 132 million |
| rounds | 4,096 | 2 | 121.0 s | 137 million |
| rounds | 4,096 | 4 | 106.9 s | 153 million |
| rounds | 4,096 | 8 | 115.6 s | 175 million |

The default for rounds is the fastest of these, 4,096 subtrees a thread in
rounds of 4 a thread, 1.24 times the dynamic schedule. (In an earlier session,
rounds of 1 and of 4 over 64 subtrees a thread took 236 to 249 s and 376 s.)

A capped stage has a small node limit, and a round gives each subtree its share
of what is left; the search checks its count every 1,024 nodes, so a smaller
share is not honoured. Rounds therefore cut at most one subtree per 1,024 nodes
of the limit (and at least one a thread). Without that bound a capped stage of
2,000 nodes cut into 16,384 subtrees spent its budget on the frontier, found no
cover, and left the exact stage 1.28 million nodes from the start incumbent
against 1,216 with the bound (16 markets, four threads). The 120-market capped
stage (300,000 nodes, 292 subtrees) took 7.0 to 7.6 s in rounds against 4.0 s
dynamic, with the same cover as at every split tried. `deterministic=False`
gives the dynamic schedule; the timings in the sections above were taken on it.

A deterministic replacement was measured and rejected: the best partition into
runs of consecutive markets by size (a one-pass dynamic program) costs 70 to
1,000 times the heuristic's design on four of five panels (50-state 263.7
against 0.22; similar-sized 120: 691 against 0.47), because good supergeos join
large and small markets against medium ones.

### Groups within regions

`certify_staged(..., region=labels)` allows only groups whose markets share a
label, so a supergeo stays under common local conditions (seasons, weather).
The problem separates: one certificate per region on its own markets, the
design's cost and proven lower bound the sums over regions (the exact optimum
under the constraint), a relative gap holding for the sum when it holds in every
region, and the absolute gap shared in proportion to the regions' sizes. A
region with fewer than `lo` markets has no design, and the result proves it.

On the 180-market simulated panel, whose two regions (126 and 54 markets) differ
in seasonal amplitude and phase, with the absolute gap 0.001 N sigma^2 and four
threads, in one session:

| design | time | cost | supergeos | across regions | held-out sum of D_k^2, median (95th percentile) |
| --- | --- | --- | --- | --- | --- |
| national | 23.4 s | 0.83 | 47 | 29 | 277,600 (719,500) |
| by region | 13.3 s | 7.63 | 46 | 0 | 210,500 (494,200) |

D_k is the difference between supergeo k's halves (split as on the design
window) in a held-out four-week total, over the 27 windows from week 244, in
squared index points; the sum is the numerator of the iROAS estimate's variance
(PANGEO_ROADMAP.md, phase 5). The by-region design costs 9.2 times as much on
the design window, both far below the noise scale N sigma^2 = 51,030, and its
held-out variance is 24% lower (median of the per-window ratios 0.73), a minimum
detectable effect 13% lower: 29 of the national design's 47 supergeos join
markets whose seasons differ, and their halves drift apart after the design
window. The panel was simulated with region-specific seasons, so this measures
the case the constraint is for, on one panel.

## 210 markets

On the 210-market instance (`n210_duals.json`), from the best design found (the
steered repair, 221,075 in its units), the root bound is 7.3e-4 of the design and
85,547 groups survive the elimination with the market prices alone: 60,500 at
half the design's cost, 27,336 at a tenth, 8,291 at a hundredth. Subset-row cuts
raise the root bound from 160.92 to 161.99 before separation finds no more
violated cuts, so their duals cannot shrink the survivors by orders of
magnitude. The surviving groups have no band in size order (on the 50-state
panel their markets span a median of 26 of 49 rank positions): a group cancels a
large market with small ones. A proof at 210 is out of reach on these routes;
what moves there is the incumbent and the gap to the bound.

## Optimized Supergeo Design

Shaw (arXiv 2506.20499) generates candidate supergeos by PCA and hierarchical
clustering of covariates and chooses among them with a 30 s MILP that assigns
whole supergeos to treatment or control to minimise standardised mean
differences. It answers a different problem: no supergeo is split, and the
objective is arm-level balance across several covariates. Its guarantee is
qualitative (constants left unspecified). Two parts carry over: balancing effect
modifiers as well as the baseline, which here is a vector score per group
(min over splits of sum_k lambda_k (2 Z_A^k - Z_G^k)^2) and leaves the pricing,
fixing, exact search and certificate unchanged; and judging designs by the RMSE
of the effect estimate over simulated heterogeneous effects, since the pretest
optimum overfits. Similarity clustering does not carry over: the base problem's
good groups mix sizes.

## Pairs by Edmonds' blossom algorithm

A design of pairs costs the sum of its pairs' costs, so the best pairing of n
markets is a minimum-weight perfect matching on the complete graph of markets
(Edmonds 1965). `matching.py` is the primal-dual blossom algorithm in its
O(n^3) form, in Numba, written from Lawler (1976, chapter 6) and Galil (1986,
section 3). It is a clean-room implementation: Blossom VI (Kolmogorov 2009) and
networkx served only as reference solvers to compare answers against, and no
code was taken from either.

With an odd count, a dummy market joined to every market at cost 0 makes the
count even, and the market matched to it is the one left out. The rest is the
least-cost set of (n - 1) / 2 pairs, by the cardinality optimality of Edmonds'
intermediate matchings (Kaibel and Walter 2017).

`min_weight_matching(W)` returns the pairs, the market left out, the cost, and
the duals the algorithm ends with: one per market and one per odd set (a
blossom). The duals prove the pairing optimal by linear programming duality on
Edmonds' matching polytope. They are feasible, and their objective equals the
cost, so no pairing costs less. The call checks this in exact arithmetic
before it returns, and raises if the check fails. Integer costs whose sums are
exact in floating point get an exact certificate. Other costs get one exact to
rounding, with the tolerance stated on the result.

### The certificate check

The first check evaluated every pair in Python rationals and added each odd
set's dual into every pair inside it. With the scalar cost (z_a - z_b)^2 the
algorithm ends with about n / 2 nested odd sets, so that check ran in time
near n^3, and at 1000 markets it took 20 s against a 1.4 s solve.

The odd sets of Edmonds' algorithm are laminar: any two are nested or
disjoint. One vertex order lays each set out as an interval (Korte and Vygen
2018, section 2.2), so the dual sum over the sets holding a pair is a
two-dimensional cumulative sum. A floating pass, with a rounding bound that
counts the terms of each sum (Higham 2002, section 4.2), clears every pair
whose slack is plainly positive. Only the rest, close to n of them, are settled
in rationals, a floating-point filter in front of an exact test (Shewchuk
1997). The result equals the direct rational check, a property test compares
the two exactly, and a crossing family falls back to the direct check.

### Timings

One quiet session, one thread, best and worst of 3 to 5 runs. "Solve" is the
Numba solver alone; "check" is the exact certificate; Blossom VI is its
solver's own timer, on integer weights (the random costs as drawn; the 174 geos' costs capped at a greedy
matching's cost and rounded to a scale of 2^28, which keeps the optimum).

| instance | Numba solve | check before | check now | Blossom VI | networkx |
| --- | --- | --- | --- | --- | --- |
| Trimmed Match's 174 geos, pairs | 6.3-6.9 ms | 43-47 ms | 2.4-2.7 ms | 0.21-0.33 ms | 1.6 s |
| 174 markets, integer costs | 8.3-9.5 ms | 35-37 ms | 2.8-2.9 ms | 1.5-1.7 ms | 1.8 s |
| 400 markets, integer costs | 81-119 ms | 0.25-0.27 s | 12 ms | 14-19 ms | 24.5 s |
| 400 markets, scalar cost | 91-119 ms | 2.7-3.0 s | 39-44 ms | | |
| 1000 markets, integer costs | 1.20-1.23 s | 2.8-3.1 s | 76-79 ms | 0.15 s | |
| 1000 markets, scalar cost | 1.38-1.57 s | 19-20 s | 0.14-0.17 s | | |

Every cost equals the reference: the sorted pairing for the scalar cost, and
Blossom VI and networkx for integer costs. On the 174 geos the whole call,
certificate included, takes 9 to 11 ms against 0.72 to 0.83 s for PANGEO's
cvxpy MIP and 2.0 to 2.2 s for the staged certificate (after the stop rules).

The Numba solver is 5 to 26 times slower than Blossom VI's. Blossom VI starts
from a greedy fractional matching and keeps each blossom's edges in priority
queues (Kolmogorov 2009), while this solver is the dense O(n^3) form, which
scans every slack in each dual step. It is 200 to 300 times faster than
networkx, the pure-Python form of the same algorithm.

References: Edmonds, "Paths, trees, and flowers", Canad. J. Math. 17 (1965)
449-467; Lawler, "Combinatorial Optimization: Networks and Matroids", 1976;
Galil, ACM Computing Surveys 18 (1986) 23-38; Kolmogorov, "Blossom V", Math.
Prog. Comp. 1 (2009) 43-67; Kaibel and Walter, arXiv:1703.09505 (2017); Korte
and Vygen, "Combinatorial Optimization", 6th ed., 2018; Higham, "Accuracy and
Stability of Numerical Algorithms", 2nd ed., 2002; Shewchuk, Discrete Comput.
Geom. 18 (1997) 305-363.

## Tests

`python -m pytest benchmarks/studies/supergeo -q`. Brute force over every
partition (fixed instances and `hypothesis` properties) for branch and price
and branch, cut and price, with each of cuts and group branching switched off
in turn; exact-arithmetic oracles for the pricing window, the numba exact
kernel and the exact-cover search, including a constructed near tie (two covers
whose exact costs differ by 2 near 2^104 while their doubles are equal) and four
fuzzed witnesses (`mutant_witnesses.json`). `n210_duals.json` holds the `N = 210`
duals as exact hex doubles.

Mutation checks, run by hand during the study: the exact search's eight mutants
are killed except one equivalent (a dead end not detected backtracks at once);
the numba exact kernel's six are killed except one equivalent (zero elimination
skipped); the pricing window's six are killed except "split check removed"
(equivalent: membership is decided exactly) and "no rounding margin" (not shown
observable in 8,900 targeted cases; the proof needs the margin).

The matcher (`test_matching.py`) is checked against brute force over every
matching for up to 11 markets with an exact certificate, the sorted pairing
for the scalar cost, and networkx where it is installed; the fast certificate
check against the direct rational check on solved and tampered certificates.
Its targets `supergeo-matching` (15 mutants) and `supergeo-matching-check`
(10) are killed in full.

## Running a script and the tests in one tree

pytest imports this package as `studies.supergeo` and `python -m` as
`benchmarks.studies.supergeo`, and numba's disk cache records the module name.
Point `NUMBA_CACHE_DIR` somewhere else for script runs.

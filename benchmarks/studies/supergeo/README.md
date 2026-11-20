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

## Running a script and the tests in one tree

pytest imports this package as `studies.supergeo` and `python -m` as
`benchmarks.studies.supergeo`, and numba's disk cache records the module name.
Point `NUMBA_CACHE_DIR` somewhere else for script runs.

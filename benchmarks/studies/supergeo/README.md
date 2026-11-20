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

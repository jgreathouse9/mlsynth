# An exact MAREX design without branch-and-bound

MAREX (Abadie and Zhao 2026) chooses which markets to treat by solving a
mixed-integer quadratic program, which `mlsynth` hands to SCIP through cvxpy.
This study shows the program does not need branch-and-bound, measures a
search that replaces it, and times the two against each other on a quiet
machine. Nothing in MAREX itself is changed here.

## The decomposition

The standard design chooses a treated set `S` of `m` markets, treated weights
`w` on the simplex over `S` and control weights `v` on the simplex over the
rest, to minimise

    ||A - B w||^2 + ||A - B v||^2,

where `A` is the cluster's mean over the fit window and the columns of `B`
are its markets. Once `S` is named the two terms share nothing, so the
objective is

    f(S) = g(S) + h(N \ S),

two simplex-constrained least-squares fits, each solved exactly by
`mlsynth.utils.solvers.active_set.solve_simplex_qp`. The design is a search
over treated sets with an exact oracle. Plain enumeration reproduces SCIP's
optimum, but it pays for two fits per candidate and loses to SCIP by J = 20,
so the search needs a bound.

## The search

Both terms are non-negative, so `f(S) >= g(S) >= lb(S)` for any lower bound
on the treated fit. `exact_design` in `search.py`:

1. seeds an incumbent by forward selection on the treated side;
2. prices every candidate by `lb` in one batched pass and sorts;
3. walks up the order, solving `g` exactly only while the bound is under the
   best total found, and `h` only while `g` is;
4. stops the moment the bound reaches the incumbent, since every remaining
   candidate is then dominated.

## The bound

Because the weights sum to one, the target can be subtracted from every
column without changing the fit: `||A - B w|| = ||(B - A 1') w||`. On the
centred columns the target is zero, and dropping only the non-negativity of
`w` leaves a closed form,

    lb(S) = min { w' G w : 1'w = 1 } = 1 / (1' G_SS^-1 1),   G = B~' B~,

which equals `g(S)` whenever its minimiser is already non-negative. SYNDES's
exact backend bounds its own objective the same way
(`mlsynth/utils/syndes_helpers/gram.py`).

The first prototype used a different bound, `||A||^2 - c' G^-1 c` on the raw
columns with both constraints dropped and a `1e-12` ridge to survive a
singular block. The ridge raises that quantity, the direction that can make a
bound invalid, and with `||A||^2` near 7,200 against residuals of 1 to 7 the
subtraction cancels most of the precision. `check_bounds.py` compares both
bounds with the exact treated fit on every treated set of nine panels (55,632
sets):

| | violations | equal to the exact fit | median gap to it |
|---|---|---|---|
| centred bound | 0 | 38-69% of candidates | about 1e-14 |
| prototype bound | 0 | 0 to 25 candidates per panel | 0.22 to 0.98 |

So the prototype never pruned wrongly on these panels, and its answers were
right; the centred bound removes the risk and settles most candidates
outright. Two guards cover floating point: a candidate is discarded only when
its bound exceeds the incumbent by `1e-6` relative, and a block whose solve
does not reproduce `G x = 1` to `1e-8` gets the bound zero, so it is never
discarded.

## Correctness

- Plain enumeration and the search agree at every size where enumeration is
  run (J up to 20).
- Every instance in the timing run is checked against SCIP's objective, which
  SCIP certifies with a zero gap.
- The weakly targeted design (`weakly_targeted.py`) adds `beta ||B w - B v||^2`,
  which couples the two sides, so the per-candidate problem becomes one
  quadratic program over both simplices; the bound still holds because the
  coupling term is non-negative. The search matches SCIP at every `beta`
  tried, solving 1, 1, 55 and 127 of 220 candidates at `beta` = 1e-6, 0.01, 1
  and 5. Those times are not from the clean run.

## Clean timing

`timing.py` times the search against SCIP's MIQP on six sizes (J = 12 to 40)
and three panels each, alone on the machine:

- single-threaded on both sides: SCIP's LP solver is single-threaded, and the
  BLAS threads behind NumPy are pinned to one before NumPy is imported;
- the search gets a warm-up run and then the median of seven; SCIP is run
  three times where a solve takes seconds and once where it takes minutes,
  and its node count has to agree across repeats;
- SCIP is timed two ways, the wall time of `prob.solve` on a freshly built
  cvxpy problem, which is what MAREX pays, and SCIP's own solving time,
  which leaves out cvxpy's compilation;
- the one-minute load average is recorded before and after every case. One
  busy single-threaded process holds it near 1.0, so values at or below that
  mean nothing else was running;
- every instance's objective is checked against SCIP's.

The table below is rendered from `results/timing.csv` by `report.py`.

## Relation to SYNDES's exact backend

SYNDES already ships this kind of search as its default
(`backend='exact'`, `mlsynth/utils/syndes_helpers/exact.py`), for a different
objective: SYNDES matches the synthetic treated unit to the synthetic control
directly, a coupled problem with no target, where MAREX matches each to the
cluster mean. So MAREX cannot call it, but most of the machinery around the
objective carries over:

- `syndes_helpers/enumeration.py` builds the admissible treated sets under
  forced units, forbidden units and stratum quotas, counts them exactly
  without generating them, and imports nothing objective-specific;
- the streaming in chunks, the top-K pool and the fallback past a candidate
  limit;
- the conditioning checks around the closed-form bound.

## Limits

- `all_subsets` materialises every candidate. That is 26 MB at J = 40 and
  `m` = 5 but about 3 GB at J = 100; SYNDES's streaming is the fix.
- Configurations whose per-candidate problem is not a plain simplex fit need
  their own leaf: the weakly targeted design (above), a cost budget, which is
  a constraint on the treated weights, and `max_control_weight`, a capped
  simplex. The bound holds for all three.
- Past some number of candidates enumeration stops being affordable, and a
  fallback is needed, as in SYNDES.

## Files

| file | what it does |
|---|---|
| `panels.py` | the panels, the fit matrices, and MAREX's MIQP for SCIP |
| `search.py` | the oracle, both bounds, the bounded search and plain enumeration |
| `check_bounds.py` | validity and tightness of both bounds over every treated set |
| `weakly_targeted.py` | the coupled design against SCIP |
| `timing.py` | the clean timing run |

    cd benchmarks/studies/marex_exact_search
    python check_bounds.py results/check_bounds.csv
    python weakly_targeted.py results/weakly_targeted.csv
    python timing.py results/timing.csv

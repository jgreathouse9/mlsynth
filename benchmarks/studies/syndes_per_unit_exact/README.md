# SYNDES per-unit design without SCIP

`mode="per_unit"` is solved only by SCIP today, at a 60 s time limit and a 5%
gap limit. This study asks whether a search over treated sets can replace it, as
it did for MAREX (`benchmarks/studies/marex_exact_backend/`) and for SYNDES's
own two-way design. It is the study #806 asks for before anything is built.

This is a study, not a benchmark case. Nothing here is pinned, and the timings
are machine-dependent.

## The reduction

Doudchenko et al. (2021, Appendix A.3) note that the per-unit problem is `K`
one-way problems, one per treated unit, each with every control as a donor. In
`mlsynth`'s formulation a treated unit's weight on another treated unit is
forced to zero (`q_ij <= 1 - D_j`), and the penalty falls on the weights, so for
a treated set `S`

    f(S) = (1/K) sum_{i in S} phi_i(S^c),
    phi_i(P) = min over v on the simplex over P of (1/T)||y_i - Y_P v||^2 + lam ||v||^2.

`phi_i` can only rise as its pool shrinks. Three lower bounds follow, each
tighter than the last.

- Root. `lb(S) = (1/K) sum_{i in S} phi_i(N \ {i})`. `N` fits bound every
  set, and the bound is a sum, so the sets can be visited in its order (#806's
  proposal).
- Pair. `lb2(S) = (1/K) sum_{i in S} max_{j in S \ {i}} phi_i(N \ {i, j})`.
  Only a `j` in the support of `i`'s root fit can raise a term, so the pair fits
  are few and every set containing the pair reuses them.
- Branch and bound on units. A node fixes a set `F` of treated units and a
  set `E` of controls. Every completion lacks `F` from every pool, so

      f(S) >= (1/K) [ sum_{i in F} phi_i(N \ F)
                      + the K - |F| smallest phi_u(N \ (F + u)), u undecided ].

  The bound tightens as `F` grows and equals `f(S)` at a leaf.

The first two drive one search, `search` in `search.py`. It visits sets in
nondecreasing root bound (a heap over a tree in which no child sums below its
parent), prunes a set on its pair bound, prices the rest one unit at a time, and
stops pricing a set once its running total reaches the incumbent. It stops when
the next root bound reaches the incumbent, which proves the incumbent optimal.
The third is `branch_and_bound`, depth first, from the incumbent of the `K`
units with the smallest root fits.

Two reuses keep the fits cheap, and both are exact. A root fit stays optimal on
a smaller pool whenever the removed donors carried no weight in it. A fit with
only the root's support removed stays optimal for any larger removal that misses
its own support.

Every fit is `solve_simplex_qp` on the design stacked with `sqrt(lam) I`, its
`converged` flag read, and its Frank-Wolfe gap recorded.

## Panels

`run.py` builds them.

| group | panels | sizes |
| --- | --- | --- |
| `bls` | `benchmarks/cases/syndes_bls.py`'s draws, 10 states by 7 months, in that case's order | first 30 draws at `K = 3` and `K = 7` |
| `small` | the 20-unit `factor` and `noise` panels | `K` in {3, 5, 10}, seeds 0-2 |
| `factor` | rank-4 factors plus noise 0.3, `T = 30` (#806's panel) | `N` in {20, 40, 60, 100}, `K` in {3, 5, 10}, seeds 0-2 |
| `noise` | a common shock plus independent noise, `T = 30`: no unit is a better donor than another | `N` in {20, 40, 60}, `K` in {3, 5, 10}, seeds 0-2 |
| `real` | GeoLift (40 markets, 90 days), Walmart (45 stores, weeks 1-100), BLS county unemployment (40, 70 and 100 counties, 36 months) | `K` in {3, 5, 10} |

`noise` is there because MAREX's study found its searches slowest where no unit
stands out (`benchmarks/studies/marex_bnb/`): every root bound is close to every
other, and an ordered search visits many sets before the bound passes the
incumbent. GeoLift and the county panels come from `marex_bnb/panels.py`.

`lam` is `estimate_lambda(Y)`, the library's default.

## SCIP

`scip` builds the library's per-unit MIP with `build_syndes_problem_components`
and solves it through `miqp_accel.solve_warm_cut`, the path #811 routes every
SCIP solve through, so SCIP's own status is read. Each solve runs in a fresh
process with a wall clock of 600 s plus twice its time limit. In the first
runs SCIP corrupted the heap (`malloc(): invalid size`) at `N = 40`, which
killed the run; on one 40-unit panel it ran past its time limit without end;
and on Walmart it returned `solver_error` at its limit, which cvxpy cannot
unpack. All three are recorded as outcomes: `crashed`, `hung` and `error`. The time is from building the problem to the answer, which
includes cvxpy's compile: 16 s at `N = 60` and 70 s at `N = 100` before SCIP
starts.

- `default`: the library's limits, 60 s and a 5% gap.
- `proof`: no gap limit, 300 s. Used on `bls` and `small`.

The returned design is priced exactly, from fresh fits, and compared with the
search's optimum as `scip_excess = (f(SCIP's design) - f*) / f*`.

## Running it

From the repository root:

```
python -m pytest benchmarks/studies/syndes_per_unit_exact/test_search.py -q
python -m benchmarks.studies.syndes_per_unit_exact.run bls scip=proof
python -m benchmarks.studies.syndes_per_unit_exact.run small scip=proof
python -m benchmarks.studies.syndes_per_unit_exact.run factor scip=default
python -m benchmarks.studies.syndes_per_unit_exact.run noise scip=default
python -m benchmarks.studies.syndes_per_unit_exact.run real scip=default
```

Each writes `results/<group>_scip-<mode>.csv`, one row per instance, and the
logs here are their printed lines.

`test_search.py` checks the searches against brute force on small panels: the
ordered enumeration visits every set once in order, the reused prices equal
fresh ones, the three bounds are ordered below the price, all three searches
find the brute-force optimum, and the search matches SCIP's proven optimum.

## How it was measured

One machine with four cores visible: Python 3.13.16, NumPy 2.5.3, cvxpy 1.9.3,
SCIP 10.0 through PySCIPOpt 6.3.0. BLAS threads are pinned to one. The four groups ran
side by side, one process each, so a time here can be off by tens of percent,
and SCIP's time limit binds a little sooner than on an idle machine.

## Results

156 runs on 138 distinct panels (the 18 of `small` are also in `factor` and
`noise`). The three searches agree wherever they close, to `1e-9`, and every
fit's Frank-Wolfe gap is at most `5.6e-13` of its value.

### The bound and the three searches

The root gap is `(f* - lb_root) / f*`, the best root bound against the optimum.
Times are the slowest of three seeds; the root-only search was capped at 120 s.

| panel | N | K | root gap (median) | root only | + pair | branch and bound (nodes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor | 20 | 3 / 5 / 10 | 0.04 / 0.08 / 0.23 | 0.03 / 0.06 / 4.7 s | 0.02 / 0.08 / 1.0 s | 0.06 / 0.19 / 1.3 s (603) |
| factor | 40 | 3 / 5 / 10 | 0.01 / 0.03 / 0.08 | 0.13 / 0.15 / 7.3 s | 0.14 / 0.19 / 3.1 s | 0.22 / 0.42 / 5.2 s (617) |
| factor | 60 | 3 / 5 / 10 | 0.01 / 0.02 / 0.05 | 0.32 / 0.39 / 28 s | 0.30 / 0.41 / 10 s | 0.49 / 0.87 / 5.6 s (191) |
| factor | 100 | 3 / 5 / 10 | 0.01 / 0.01 / 0.03 | 0.59 / 0.87 / 3.5 s | 0.74 / 0.90 / 3.5 s | 0.86 / 1.4 / 3.7 s (43) |
| noise | 20 | 3 / 5 / 10 | 0.05 / 0.10 / 0.23 | 0.01 / 0.19 / > 120 s | 0.01 / 0.09 / 114 s | 0.02 / 0.10 / 26 s (31,083) |
| noise | 40 | 3 / 5 / 10 | 0.02 / 0.04 / 0.07 | 0.06 / 0.48 / > 120 s | 0.11 / 0.34 / 60 s | 0.22 / 0.60 / 11 s (1,025) |
| noise | 60 | 3 / 5 / 10 | 0.01 / 0.02 / 0.05 | 0.24 / 0.34 / 35 s | 0.28 / 0.42 / 6.7 s | 0.57 / 1.4 / 8.1 s (159) |
| GeoLift | 40 | 3 / 5 / 10 | 0.02 / 0.05 / 0.12 | 0.07 / 0.35 / > 120 s | 0.09 / 0.37 / 64 s | 0.26 / 0.86 / 28 s (1,925) |
| Walmart | 45 | 3 / 5 / 10 | 0.03 / 0.04 / 0.09 | 0.12 / 0.17 / 15 s | 0.19 / 0.17 / 5.5 s | 0.26 / 0.43 / 4.9 s (173) |
| county | 40 / 70 / 100 | 10 | 0.15 / 0.07 / 0.04 | 1.3 / 10 / 9.3 s | 0.81 / 4.9 / 8.5 s | 0.96 / 6.3 / 9.6 s (≤ 107) |
| BLS draws | 10 | 3 / 7 | 0.12 / 0.45 | ≤ 0.13 s | ≤ 0.14 s | ≤ 0.20 s (median 10 / 140) |

- The bound is tight where the problem is easy, and the problem is easy at
  scale. The root gap shrinks as `N` grows at fixed `K`: removing `K - 1`
  donors from a pool of `N - 1` matters less in a large pool. At `N = 100,
  K = 10` it is 3%, and the branch and bound visits 43 nodes.
- The hard regime is a large share treated on a panel without structure. At
  `N = 20, K = 10`, half the units leave every pool and the root gap is 23%.
  There the root-ordered search of #806's proposal takes 100 s or more on each
  noise panel, and does not close within its 120 s in 3 of the 6 runs of them
  (they appear in both `noise` and `small`). It also does not close on noise
  `N = 40, K = 10` and on GeoLift at `K = 10`: 5 of 156 runs in all.
- The branch and bound closes everything, in at most 28 s (GeoLift,
  `K = 10`) and 26 s (noise, `N = 20, K = 10`). Fixing a unit in the treated set
  removes it from every pool, so its bound tightens as it descends, which the
  sum of root fits cannot do. The pair bound closes everything too, but takes
  up to 119 s.
- The time is fits. On the hardest instances the branch and bound spends
  0.12 ms per fit at `T = 30, N = 20` and 0.9 to 1.2 ms at `N = 40` (GeoLift has
  `T = 90`): 204,000 fits on noise `N = 20`, 23,000 on GeoLift. A node's fits
  are its parent's with a few donors removed, so warm-starting them from the
  parent's weights is the obvious next saving.

### SCIP at the library's defaults (60 s, 5% gap)

`factor`, `noise` and `real`, 78 instances.

| outcome | instances |
| --- | ---: |
| a design | 57 |
| `solver_error` at the limit, no design (all of `N = 100` and Walmart) | 15 |
| hung past its time limit, killed at 720 s | 4 |
| crashed (heap corruption) | 2 |

Of the 57 designs, 14 are optimal. 31 are more than 10% worse than the optimum,
and the median excess among the non-optimal ones is 56%.

| panel | N | designs returned | median excess | worst |
| --- | ---: | ---: | ---: | ---: |
| factor | 20 | 9 | 0 | 0 |
| factor | 40 | 7 | 97% | 1,131% |
| factor | 60 | 9 | 203% | 719% |
| noise | 20 | 9 | 0.8% | 5.0% |
| noise | 40 | 6 | 35% | 84% |
| noise | 60 | 9 | 58% | 82% |
| GeoLift, county | 40 | 6 | 49% | 58% |
| county | 70 | 2 | 929% | 1,166% |

At `N = 20` SCIP's incumbent is usually good. From `N = 40` the 60 s limit binds
on nearly every instance, and what it returns is SCIP's first incumbent, often
far from optimal. #811's `solve_outcome` now reports this as `timelimit` with a
gap, where cvxpy had reported `optimal_inaccurate`.

SCIP's failures are its own. Building the model is 1.4 s at `N = 60` and 5.7 s
at `N = 100`, and cvxpy's compile 14 s and 64 s. On the factor panel with
`N = 40, K = 5`, seed 1, SCIP alone, with nothing else running, ran past its
60 s limit until killed at 400 s, twice; at 120 s the Python stack was inside
SCIP's `optimize()` (cvxpy's `scip_conif.py:339`). Across the default runs,
SCIP's child processes printed glibc heap-corruption errors seven times
(`malloc(): invalid size`, `free(): invalid pointer`, `munmap_chunk(): invalid
pointer`), visible in the logs.

### SCIP with no gap limit (`small` and `bls`)

On the 20-unit panels, given 300 s, SCIP proves 15 of 18 optima, all equal to
the search's, in 10 to 172 s (median 69 s). The search proves them in 0.01 to
1.8 s. It does not finish the three noise panels at `K = 10`, which the branch
and bound closes in 11 to 26 s.

On the 60 BLS draws SCIP reports `optimal` on all 60, in a median of 3.4 s. On
2 its design is not optimal: K = 7, draw 6, is 6.9% worse, and K = 7, draw 28,
0.04% worse. Brute force over all 120 sets agrees with the search. Rescaling
the panel to unit variance makes SCIP return the true optimum on both: its
"optimal" depends on the units of the outcome, whose objective here is near
`1e-4` (#814).

## What to build

The branch and bound on units, as `per_unit`'s exact backend and its default:

- It is the only search that closed every instance here, in under 30 s, and its
  bound tightens with depth where the separable bound of #806 cannot.
- Price fits with `solve_simplex_qp`, read `converged`, and carry the
  Frank-Wolfe gap, as here. Warm-start each node's fits from its parent's.
- The incumbent is certified when the search closes: every pruned node's bound
  was at or above it. Report the search's own bound and gap in the typed
  outcome #811 added.
- `top_K` continues the same search with the found sets excluded. `K = None`
  runs it per size, dividing by `|S|` (Section 3).
- Donor-side restrictions shrink each treated unit's pool, so fits on a
  superset still bound; treated-side restrictions prune nodes through
  `SubsetFilter`.
- Keep SCIP as the cross-check in `syndes_exact_vs_mip` and `syndes_bls` until
  #808 retires it, with #814's rescaling, since it is what makes SCIP's own
  answers comparable.

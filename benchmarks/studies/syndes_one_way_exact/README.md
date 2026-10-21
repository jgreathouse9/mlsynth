# SYNDES one-way design without SCIP

`mode="one_way_global"` is solved only by SCIP today. This study asks whether a
search over treated sets can replace it, as it did for the two-way and per-unit
designs (`benchmarks/studies/syndes_per_unit_exact/`). It is the study #807 asks
for before anything is built.

This is a study, not a benchmark case. Nothing here is pinned, and the timings
are machine-dependent.

## The reduction and two bounds

One-way is two-way with the treated weights fixed at `1/K` (Doudchenko et al.
2021, Section 2 and A.2). With the treated set `S` named, `F = S^c` and
`M = G + lam I` the two-way Gram matrix (`gram.py`, centred since #812),

    f(S) = min { u'Mu : u_S = 1/K, u_F <= 0, 1'u = 0 }
         = lam/K + min over c on the simplex over F of
           (1/T)||ybar_S - Y_F c||^2 + lam ||c||^2,

one simplex fit per set. Two relaxations bound it from below.

- Two-way. Drop `u_S = 1/K`; keep `sigma'u = 2` and `1'u = 0`. That is
  `gram.py`'s closed form, `lb(S) = 4 alpha / sigma'R sigma`.
- Ridge. Keep `u_S = 1/K` and `1'u = 0`; drop only `u_F <= 0`. With
  `C = [E_S, 1]`, `d = [1/K; 0]` and `H = M^{-1}`, its value is
  `d'(C'HC)^{-1}d`, one `(K + 1) x (K + 1)` solve per set, batched.

Ridge is at least two-way, since its constraints imply the two-way ones.
`search` in `search.py` enumerates every treated set, bounds each both ways,
prices them exactly in order of the ridge bound, and stops when the next bound
reaches the incumbent, which proves it optimal. Exact pricing runs on the panel
centred within each period and divided by its root mean square, as the per-unit
backend's fits do (#817).

Above an enumeration limit, `fallback` takes the two-way backend's swap-search
candidates (`maxcut_candidates`, 32 restarts), prices each as a one-way design,
and reports the best against the Rayleigh bound over every size-`K` set, which
bounds one-way too: `one-way >= two-way >= Rayleigh`.

## Panels and SCIP

The per-unit study's panels and groups, from its `run.py`: the `syndes_bls`
draws, factor and structureless noise panels at `N` in {20, 40, 60, 100} and `K`
in {3, 5, 10}, and GeoLift, Walmart and BLS county panels. `lam` is
`estimate_lambda(Y)`.

SCIP runs as in the per-unit study: the library's one-way MIP through
`miqp_accel.solve_warm_cut`, in a child process with a wall clock, timed from
building the model. `default` is the library's limits (60 s, a 5% gap); `proof`
has no gap limit and 300 s, on `bls` and `small`. SCIP's design is priced
exactly and compared with the search's optimum.

```
python -m pytest benchmarks/studies/syndes_one_way_exact/test_search.py -q
python -m benchmarks.studies.syndes_one_way_exact.run bls scip=proof
python -m benchmarks.studies.syndes_one_way_exact.run small scip=proof
python -m benchmarks.studies.syndes_one_way_exact.run factor scip=default
python -m benchmarks.studies.syndes_one_way_exact.run noise scip=default
python -m benchmarks.studies.syndes_one_way_exact.run real scip=default
python -m benchmarks.studies.syndes_one_way_exact.fallback <group>
```

The limit is 2e7 treated sets. One machine with four cores visible: Python
3.13.16, NumPy 2.5.3, cvxpy 1.9.3, SCIP 10.0 through PySCIPOpt 6.3.0, BLAS
pinned to one thread. Four groups ran side by side.

## Results

156 runs on 138 panels: 132 within the enumeration limit, 24 beyond it.

### The bounds and the search

| | median | worst |
| --- | ---: | ---: |
| two-way gap at the optimum, `(f* - min lb) / f*` | 0.19% | 13.3% (a BLS draw) |
| ridge gap at the optimum | 0.00% | 0.20% |
| sets priced exactly | 1 | 2 (once) |
| bound time per set | 10 µs | 55 µs |

- The ridge bound equals the optimum to rounding on 126 of 132 runs, and is
  within 0.005% to 0.2% of it on the other 6. The sign conditions on the control
  weights are slack at, or near, the optimum: with the treated weights fixed at
  `1/K`, the best equality-constrained control is already nonnegative or
  nearly so. So the search prices one set and stops, on all but one run.
- The search's cost is the enumeration. The largest within the limit, 12.1
  million sets (70 counties, `K = 5`), took 124 s; 5.5 million (60 units,
  `K = 5`) about 43 s. Every fit's Frank-Wolfe gap is at most `1.4e-13` of its
  value.
- The two-way bound is weaker on the BLS draws (10 states, 7 months), where
  fixing the treated weights matters more, and within 0.25% elsewhere.

### SCIP at the library's defaults (60 s, 5% gap)

On the 54 enumerable runs of `factor`, `noise` and `real`, SCIP returned a design
every time: 35 optimal, the median non-optimal one 0.3% worse, 4 more than 1%
worse, and the worst 10.1% (Walmart, `K = 5`). It stopped at its time limit on
44 of 78 runs and at its gap limit on 34. It never crashed, hung or errored,
unlike the per-unit MIP: the one-way program has `N` binaries and `N` control
weights, the per-unit one `N^2` weights.

With no gap limit, SCIP proved the optimum on all 18 `small` panels and all 60
BLS draws, each equal to the search's, in a median of 10 s and 0.4 s. The units
defect of #814 did not show on the one-way BLS designs.

### Beyond the enumeration limit

24 runs: factor and noise at `K = 10` from `N = 40`, factor at `N = 100,
K = 5`, and the real panels at `K = 10` and county `N = 100, K = 5`.

- The fallback takes under 0.75 s. Its design is within 0.3% to 2.7% of the
  Rayleigh bound (median 0.8%), so within that of the optimum.
- Against SCIP's 60 s answer: SCIP is more than 1% worse on 5 runs, all real
  panels (1.1% to 24%, Walmart `K = 10` the worst), and better on 5, by at most
  0.6%.
- Where the optimum is known, the fallback found it on 107 of 132 runs. Its
  worst miss is 3.3%, on a BLS draw; elsewhere at most 0.42%.

## What to build

An exact one-way backend, as the default for `one_way_global`:

- Enumerate the admissible treated sets through the two-way backend's
  `SearchSpace` and `SubsetFilter`, bound each by the ridge relaxation in
  batches, price in bound order with `solve_simplex_qp` on the centred,
  unit-scale panel, and stop when the next bound reaches the incumbent. That is
  a certified optimum, at about one fit per instance.
- Donor-side restrictions shrink the control pool, which raises every set's
  value, so the ridge bound on the unrestricted pool still bounds; treated-side
  restrictions go through `SubsetFilter`.
- Above a candidate limit, return the swap-search fallback with
  `certified=False` and the Rayleigh bound, as the two-way backend does today.
  It beat or matched SCIP's 60 s answer within 0.6% on every run here. A
  certified search above the limit needs a node bound, the subject of #803; the
  ridge relaxation, with the treated weights of fixed units pinned, is a
  candidate for it.

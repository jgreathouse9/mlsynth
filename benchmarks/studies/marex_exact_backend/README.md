# MAREX's two design backends, timed as shipped

`backend="exact"` solves MAREX's design by searching treated sets
(`mlsynth/utils/marex_helpers/exact.py`); `backend="mip"` hands the
mixed-integer program to SCIP. This study times both on the code that ships,
one core each, and checks on every case that they return the same design.

This is a study, not a benchmark case. Timings are machine-dependent and
nothing here is pinned; the agreement between the backends is pinned in
`mlsynth/tests/test_marex_exact_backend.py`.

```
python benchmarks/studies/marex_exact_backend/timing.py \
    benchmarks/studies/marex_exact_backend/results/timing.csv
python benchmarks/studies/marex_exact_backend/walmart_fit.py
PYTHONHASHSEED=0 python benchmarks/studies/marex_exact_backend/walmart_scip_repeat.py
python benchmarks/studies/marex_exact_backend/walmart_layout.py
```

Run them from the repository root; they read `basedata/walmart_weekly_sales.csv`.

## What is here

| file | what it does |
| --- | --- |
| `timing.py` | the design step of each backend, on the Walmart design and 18 simulated panels |
| `walmart_fit.py` | the Walmart example of `docs/marex.rst`, the whole fit, with each backend |
| `walmart_scip_repeat.py` | the Walmart MIQP in a fresh process, under a fixed hash seed |
| `walmart_layout.py` | the Walmart MIQP on the same values in two memory layouts |
| `results/timing.csv`, `results/timing.log` | the run behind the tables below |
| `results/walmart.log` | the output of the three Walmart scripts |

## How it was measured

One machine with four cores visible: Python 3.13.16, NumPy 2.5.3, cvxpy 1.9.3,
SCIP 10.0 through PySCIPOpt 6.2.1. BLAS threads are pinned to one before NumPy
loads and SCIP's LP solver is single-threaded, so each backend runs on one core.
Nothing else heavy ran: the one-minute load average stayed between 0.69 and
1.12, and the run itself keeps one core busy.

The search is the median of seven runs after a warm-up. SCIP is timed three
times per panel up to 20 markets and once above, as `solve_design` from end to
end. That includes cvxpy building and compiling the problem, which a MAREX fit
also pays.

The simulated panels are those of the prototype study of this search: two
random-walk factors, five markets loaded at the cluster's mean loading with
almost no noise and the rest loaded differently and noisy, 24 periods of which
18 come before treatment, one cluster, and `m_eq` treated markets. Three seeds
per size.

## Agreement

All 19 cases return SCIP's treated set. The search's objective is at or below
SCIP's in every case, lower by between 5e-9 and 4e-7 relative: SCIP stops
within its tolerance of the optimum, and the search's fits are exact.

## Times

The design step only; everything after it is shared code.

| markets | treated | candidates | search, median | SCIP, median | speedup, geometric mean (range) |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 12 | 3 | 220 | 17 ms | 0.69 s | 48 (27 to 103) |
| 16 | 4 | 1,820 | 15 ms | 2.1 s | 96 (40 to 216) |
| 20 | 5 | 15,504 | 54 ms | 4.1 s | 121 (61 to 217) |
| 25 | 5 | 53,130 | 52 ms | 27 s | 482 (415 to 548) |
| 30 | 5 | 142,506 | 119 ms | 45 s | 394 (314 to 586) |
| 40 | 5 | 658,008 | 503 ms | 162 s | 345 (265 to 499) |
| Walmart, 45 | 2 | 990 | 2.5 ms | 46 s | 18,700 |

At 40 markets the whole search costs 0.76 microseconds per candidate. At 12
markets, with 220 candidates, it spends 79 microseconds per candidate, so there
the time goes to costs that do not grow with the candidate count: the exact fits
and the setup. From 25 markets on, the search's time grows about as fast as
the candidate count, 2.3-fold for 2.7 times the candidates from 25 to 30 and
4.2-fold for 4.6 times from 30 to 40. SCIP's follows its proof of optimality.

The Walmart row's SCIP time is the faster of two that the same data produced;
the last section has the other and the reason.

## The Walmart fit, end to end

`walmart_fit.py` times the example of `docs/marex.rst`, inference included. The
search takes 0.029 s (median of five), the MIQP 78.9 s (one run). Both treat
stores 1 and 15 and report p = 0.937; the objectives are 0.17765442 and
0.17765453.

## SCIP's time varies with the problem's last bits

The Walmart MIQP took 46 s in `timing.py` and 79 s in `walmart_fit.py`, on the
same values. The difference is the memory layout of the panel. `timing.py`
pivots the data frame, which hands `solve_design` a C-ordered array; the
estimator's panel is a transpose and arrives F-ordered. The values are bitwise
equal, but the standardised predictors and the target computed from them are
not: reductions over the two layouts sum in different orders, and the results
differ by up to 1.1e-15. `walmart_layout.py` passes the same values as plain
arrays in both orders. SCIP takes 48 s on the C-ordered array and 83 s on the
F-ordered one, and stops at different incumbents, 0.17765447075 and
0.17765452977; the search's optimum, 0.17765442453, is below both.

The hash seed plays no part. In a fresh process per run, `walmart_scip_repeat.py`
took 80.8, 80.7 and 80.5 s under seeds 0, 0 and 1 and stopped at the same
incumbent each time.

So a SCIP time on one instance is one draw from a spread that perturbations at
the level of rounding error open up, here by a factor of 1.7. The panel times
above carry that spread too, and the geometric mean over three panels per size
is the steadier figure. On the same two arrays the search returns the same two
stores, with objectives 2e-16 apart.

# MAREX's design by a certified branch and bound

`backend="exact"` finds MAREX's design by pricing every treated set in a
cluster (`mlsynth/utils/marex_helpers/exact.py`). The number of sets grows as
`C(n, m)`, and the backend refuses a cluster with more than twenty million.
This study asks whether a branch and bound can bound whole groups of sets
instead, with bounds that floating point cannot break, and where that beats
pricing them one at a time.

The answer, on GeoLift (40 markets) and Walmart (45 stores): the bound that
works is persp+N, a semidefinite relaxation with one block per side of the
design (`persp_n.py`). The search that uses it bounds a group of sets only when
pricing the group would cost more (`hybrid.py`). It returns the library's
optimum, certified, and wins once a cluster holds tens of millions of sets:

| panel | treated | sets | library's search | persp+N search |
| --- | ---: | ---: | ---: | ---: |
| GeoLift | 6 | 3,838,380 | 3.9 s | 8.7 s |
| GeoLift | 8 | 76,904,685 | 105.1 s | 39.9 s |
| GeoLift | 10 | 847,660,528 | 1,866.7 s | 33.4 s |
| Walmart | 6 | 8,145,060 | 7.5 s | 13.6 s |
| Walmart | 8 | 215,553,195 | 305.2 s | 165.0 s |

The library's search was run with its limit raised (`exact_reference.py`).

This is a study, not a benchmark case. Nothing here is pinned, the library is
unchanged by it, and the timings are machine-dependent. Two notes record the
findings in full:

- `results/persp_n.md`: what a tight node bound has to contain, with the
  twenty-seven papers read to find it; the bound, its certificate and its
  quality; the searches above; the penalized design; open questions; and the
  references.
- `results/rca_enumeration_vs_bnb.md`: why the earlier certified search lost
  to enumeration at every treated count, as a root-cause ladder.

## Running it

From the repository root. The scripts read `basedata/geolift_market_data.csv`,
`basedata/spsydid_bls_county_subset.csv` and the Walmart data of
`benchmarks/cases/marex_walmart.py`, and cache
each panel's exact objective on every treated set (`results/leaves_*.npz`,
ignored by git).

```
python -m pytest benchmarks/studies/marex_bnb/test_persp_n.py benchmarks/studies/marex_bnb/test_hybrid.py -q
python -m benchmarks.studies.marex_bnb.persp_n_geolift both
python -m benchmarks.studies.marex_bnb.run_hybrid 8 10 bounder=persp_n
python -m benchmarks.studies.marex_bnb.run_hybrid 6 8 bounder=persp_n panel=walmart
python -m benchmarks.studies.marex_bnb.exact_reference geolift 10
python -m benchmarks.studies.marex_bnb.exact_reference walmart 6 8
```

`lorads_bridge.py` needs a LoRADS binary at `$LORADS_BIN`, built with the
three-line patch its docstring describes. Four scripts run on import and are
meant only for `python -m`: `certified_dnn_geolift`, `check_certified_bnb`,
`check_certified_dnn` and `penalized_relaxation`.

## What is here, in the order it was built

| stage | files | results |
| --- | --- | --- |
| How tight a bound must be | `ceiling.py`, `monotone.py` | `ceiling.*`, `monotone_*.json` |
| Prototype trees | `bnb12.py`, `bnb_geolift.py`, `bnb_binary.py` | `bnb12.log`, `bnb_geolift.*`, `bnb_binary_*` |
| Semidefinite lifts | `sdp_bound.py`, `sdp_indicator.py`, `perspective.py`, `compare_lifts.py`, `root_gap_by_m.py`, `treated_lift_geolift.py` | `compare_lifts.log`, `root_gap_by_m.log`, `treated_lift_geolift.log` |
| Exact certificates | `certify.py`, `certified_dnn.py`, `check_certified_dnn.py`, `certified_dnn_geolift.py` | `certified_dnn12.log`, `certified_dnn_geolift.log`, `certified_indicator_geolift.log` |
| Node solvers | `admm.py`, `check_admm.py`, `bundle.py`, `lorads_bridge.py` | `node_solvers.log` |
| Certified searches | `certified_bnb.py`, `check_certified_bnb.py`, `certified_bfs.py`, `run_certified_bfs.py` | `certified_bnb12.log`, `certified_bfs_geolift_m*.log` |
| Root-cause analysis and the cost rule | `hybrid.py`, `run_hybrid.py`, `test_hybrid.py` | `rca_enumeration_vs_bnb.md`, `hybrid_geolift*.log` |
| Bounds inherited from a parent | `inherit.py`, `inherit_probe.py`, `inherit_fixing.py` | `inherit_probe_geolift_m6.log`, `inherit_fixing_geolift.log` |
| persp+N | `persp_n.py`, `persp_n_geolift.py`, `test_persp_n.py`, `exact_reference.py` | `persp_n.md`, `persp_n_geolift_*.log`, `hybrid_persp_n_*.log`, `exact_reference_*.log` |
| The penalized design | `penalized_relaxation.py` | `penalized_relaxation_geolift.log` |

`panels.py` builds every panel: GeoLift, county panels from the BLS subset,
synthetic factor panels, and Walmart as MAREX's exact backend sees it
(`results/walmart_panel.npz`).

## How it was measured

One machine with four cores visible: Python 3.13.16, NumPy 2.5.3, cvxpy 1.9.3,
Clarabel 0.11.1. BLAS threads are pinned to one. Other jobs of this study
sometimes ran beside a measurement, so a time here can be off by a few tens of
percent; the comparisons above are between runs a factor of two or more apart.

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

## Phases

### Phase 0: measure PANGEO as it is (no code change)

On the 50-state panel, GeoLift, and simulated panels at 20 to 80 markets, Q = 2:
time and status of `fast=False`; the `fast=True` design's cost; and the proven
gap of the fast design, from a study-side port of the certificate to PANGEO's
score. Decides what phases 2 to 4 can gain: if the fast design sits within
noise of the optimum on real panels, phase 4 is a certificate, not a better
design.

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
  12%, so spend balance is a constraint for small designs, not an objective.
- Also from Trimmed Match Design: the maximum supergeo size (Q) and the number of
  pairs chosen by the cross-validated RMSE, rerandomization with recorded balance
  checks (TMD section 4.3), and optional geo exclusion as singleton columns with a
  bound on their number. Excluding geos changes the estimand to the included
  geos' spend-weighted iROAS and can bias it when their iROAS differs (Chen et al.
  2023, err_3); the result says which estimand it reports.
- Acceptance: the closed form against simulation on the same windows; a placebo
  study on held-out windows showing whether the cross-validated choices beat the
  in-sample ones in RMSE.

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
  `Report(confidence=...)` returning the estimate and the interval. Its design
  package is described as preliminary and demonstrated in a notebook; its entry
  points are to be read from the package source before the design benchmark is
  written.
- Trimming and supergeos: Chen et al. (2023, section 4) show trimming poorly
  matched pairs can bias the estimate under heterogeneous iROAS, and that the
  supergeo design does not introduce that bias; the estimator reports how many
  pairs it trimmed, and phase 5 measures the bias on simulated heterogeneous
  effects.

### Phase 7: benchmarks and docs (separate workstream)

- `benchmarks/cases/trimmed_match.py`, cross-validation against the reference
  implementation: the estimate, the trimming and the interval on fixed inputs
  (the README's toy example, simulated pairs, pairs from a public panel) match
  google/trimmed_match, following the benchmarks' existing pattern for pinned
  references. A pinned installer (`benchmarks/reference/install_trimmed_match.sh`)
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
  simulation, and the power analysis against the reference design package's
  RMSE on the same pairs and windows, once its entry points are confirmed.
- `benchmarks/cases/pangeo_certificate.py`: certified optima on fixed panels.
- `docs/replications/` pages for each, and the PANGEO docs page's Verification
  pointer, solver section and power-analysis section.
- The supergeo paper's results (Chen et al. 2023, Tables 1 to 3) inject synthetic
  effects into two private advertising datasets of the 210 US DMAs, so neither
  path A nor path B applies; the Trimmed Match case is cross-validation.

## Order and dependencies

Phase 0, then 1 (independent, small), then 2 and 3 in sequence, then 4. Phase 5's
closed form and backtest can start after phase 1 and run beside 2 to 4; its
Trimmed Match simulations follow phase 6. The Trimmed Match benchmark (phase 7)
lands with phase 6. The rest of phase 7 after each phase lands.

## Risks

- Q > 2 grows the groups to 2Q markets: pricing enumerates more groups per scan
  (at 50 markets, 15.9 million groups of 6) and each score enumerates more
  splits; phase 2 measures where that stops being affordable.
- The certificate past 62 markets costs minutes (study: 102 s at 64, 566 s at 80);
  at a few hundred markets a proof is out of reach (study: 85,547 survivors at
  210), and the deliverable there is a design with a proven gap.
- The objective overfits whatever the solver: phase 5 is what turns an optimal
  design into a better experiment.

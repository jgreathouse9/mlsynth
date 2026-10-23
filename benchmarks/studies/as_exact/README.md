# The Abadie-Shang exact synthetic control without cvxpy, against the authors' code

Abadie and Shang (2026), "Synthetic Control: An Exact Geometric Formulation"
(arXiv:2610.26576), replace the bilevel choice of predictor weights with a search over
the faces of the donor hull. This study reimplements the exterior case (the treated
unit outside the donor hull) with NumPy and SciPy only, and asks two questions on
Proposition 99: does it return the authors' answers, and how fast is it against their
current code.

This is a study, not a benchmark case, and it is not wired into the library. The
seconds are machine-dependent: one machine with four cores visible, Python 3.13.16,
NumPy 2.5.3, SciPy 1.18.1, cvxpy 1.9.3, Clarabel 0.11.1, BLAS held to one thread,
nothing else running.

    python -m benchmarks.studies.as_exact.capture    # the oracle bundle, from the authors' code
    python -m benchmarks.studies.as_exact.run        # Proposition 99 timings -> results/raw.json
    python -m benchmarks.studies.as_exact.scaling    # random designs, K = 7 to 9 -> results/scaling.json
    pytest benchmarks/studies/as_exact               # offline, against the bundle
    MLSYNTH_AS_EXACT_LIVE=1 pytest benchmarks/studies/as_exact -k live   # reruns their code

## The reference

The authors' two repositories (MIT licence), at their heads when the study ran:

| repository | commit | version |
| --- | --- | --- |
| [synth-AS-replication](https://github.com/lychemon/synth-AS-replication) | `791a3865` (2026-10-19) | 1.0.0 |
| [synthx_package](https://github.com/lychemon/synthx_package) | `a10a28c5` (2026-10-20) | 0.9.1 |

`reference.py` clones both into a cache directory (`MLSYNTH_AS_EXACT_CACHE`) and
checks out these commits; `--latest` on `run.py` uses the default-branch heads
instead. Nothing of theirs is vendored. Three implementations are timed:

- `replication`: `visible_facet_solver.solve_exterior_hullfree` and `solve_exterior_pd`
  for the donor weights, `predictor_weights.diag_centers`,
  `compute_pd_logdet_and_ranges` and `analytic_center` for V. The quadratic programs
  go through cvxpy and Clarabel, the linear programs through SciPy's HiGHS.
- `synthx.estimate`: their general-use entry point, the Python core of the Stata
  command. Same solvers, plus input and geometry checks.
- `study (no cvxpy)`: `exact.solve_diag`, `exact.solve_pd` and `rationalize.py`.

All three read the same arrays, built by the authors' `build_california` from their
repository's `data/smoking_data.csv`.

## What changes

| step | authors' code | this study |
| --- | --- | --- |
| least squares on a face | cvxpy with Clarabel, about 4.6 ms a call | Lawson and Hanson's reduction to one compiled NNLS call (`lsq.py`), about 0.2 ms |
| facets visible from X1 | per ridge, a candidate hyperplane for each donor, each checked against every donor | gift wrapping: one QR factorisation and one J x K product per ridge (`hull.py`) |
| allowed sign patterns of a face | extreme rays with every donor as a wall, or one LP per pattern | facet: the normal's signs; ridge: sorted sign changes of a two-generator cone; smaller faces: only the link vertices as walls, in one batched SVD; edges: at most K + 1 residual intervals (`patterns.py`) |
| pruning a face | the best unconstrained fit on the face | the same bound, then one solve with the signs every piece of the face shares (Balas, Disjunctive Programming, 2018, Theorem 1.2) before any pattern is listed (`exact.py`) |
| predictor weights V | cvxpy `log_det`, analytic centre and semidefinite range programs | Newton's method with the equalities eliminated and the barrier method (Boyd and Vandenberghe, sec. 10.2 and ch. 11), started at the paper's own Lemma B.1 construction (`rationalize.py`) |

The faces containing a given face are reached by walking ridges that contain it; in
a simplicial polytope they form a connected subgraph of the facet-ridge graph (the
polar is simple; Achatz and Kleinschmidt in Kalai and Ziegler, eds., Polytopes:
Combinatorics and Computation, 2000, p. 155). The link vertices they expose are the
only walls of the face's normal cone, which is what makes the ray enumeration small.

## Correctness

The oracle is the authors' code on their inputs, not the printed tables. `capture.py`
runs the replication at the pinned commit on Proposition 99, its leakage-free variant
(Section 6) and West Germany, and on 24 seeded random exterior instances (K = 3, 4, 5
predictors; J = 10, 15, 20 donors). It writes inputs and outputs at full precision to
`results/reference.json`, with commits, versions and data checksums in
`results/provenance.json`. `test_exact.py` reads that bundle and runs offline; its
live test reruns their code on fresh instances.

- Donor weights, both classes, all 27 cases: losses agree to 1e-9 (relative) and
  weights to 1e-7. On Proposition 99 the largest weight difference is 8.4e-13 in the
  diagonal class and 3.2e-11 in the positive definite class.
- Predictor weights, both applications: the positive definite log-det maximiser and
  its identified ranges agree to 1.2e-8 at worst. The replication solves the diagonal
  programs and both analytic centres at Clarabel's default tolerances, and those agree
  to 2.3e-5 at worst; on the diagonal log-det problem the objective here is the higher
  of the two in both applications. On Proposition 99 the paper's analytic answer is
  one third on each of the three retained predictors (Section 4.5.2, Table 2); this
  study returns it to 3e-9 and the replication to 9e-6.
- Properties over seeded random designs: the diagonal weights do not change when the
  predictors are reordered or rescaled, the positive definite weights do not change
  under an invertible linear map of the predictors, the positive definite loss never
  exceeds the diagonal loss, and no randomly drawn positive diagonal V gives a lower
  loss than the diagonal optimum.

Two defects came out only because the oracle was the authors' code on their inputs.
The first port of the diagonal V recovery projected its starting point onto equalities
made nearly singular by residuals of 6e-13 on the exactly matched predictors, and
returned an infeasible V; the start is now a mean of feasible points with no
projection. And building Proposition 99 from mlsynth's
`basedata/augmented_cali_long.csv` reproduced the weights but not V, because that
file's `loginc` column holds the cigarette price (California's 1980-1988 mean is
89.42) and its `p_cig` column holds log income (10.08). `docs/replications/vanillasc.rst`
records the relabelling; no other page or case does. Where both columns are averaged
over the same window, as in the Prop 99 benchmark cases, the swap permutes two
predictors, so a fit that treats predictors alike is unchanged and only its labels
are wrong. SparseSC is not such a fit: it fixes the first listed predictor's weight
at one, and its documented example lists `p_cig` first, so on this file it anchors
on log income.

## Speed on Proposition 99

Median wall clock of 10 calls after one warm-up, in seconds; `results/raw.json` holds
every call and the interquartile ranges.

| panel | task | replication | synthx.estimate | this study | speed-up |
| --- | --- | ---: | ---: | ---: | ---: |
| Proposition 99 | donor weights, diagonal (Table 1) | 1.771 | 1.807 | 0.129 | 14x |
| Proposition 99 | donor weights, positive definite (Table D.1) | 0.660 | 0.692 | 0.094 | 7.0x |
| Proposition 99 | V, diagonal (Table 2) | 0.108 | | 0.075 | 1.4x |
| Proposition 99 | V, positive definite with ranges (Table 4) | 0.423 | | 0.268 | 1.6x |
| leakage-free | donor weights, diagonal | 2.182 | 2.120 | 0.135 | 16x |
| leakage-free | donor weights, positive definite | 0.698 | 0.732 | 0.082 | 8.6x |

The speed-up is against the replication. `synthx.estimate` runs within 5 per cent of
it, since it calls the same solvers. Where the time goes, from `breakdown.py`
(`results/breakdown.json`):

- Donor weights, diagonal class. The replication spends 53 per cent of its time in
  186 cvxpy quadratic programs at about 4.6 ms each, 25 per cent enumerating the
  visible facets, and 13 per cent in 170 SciPy LPs that test sign patterns face by
  face. This study spends 36 per cent gift-wrapping the 132 visible facets and 55 per
  cent in 343 least-squares solves at about 0.2 ms each. On Proposition 99 no face it
  reaches needs an LP to list its sign patterns, and 17 faces are ruled out by the
  common-sign relaxation before any are listed.
- V, diagonal class. Both codes spend most of their time in the same 40 LPs that find
  the support of V and its implicit equalities; only the two convex programs after
  them change, from 31 ms in cvxpy to about 1 ms of Newton steps.
- V, positive definite class. The replication's 17 semidefinite programs, at about
  23 ms each, become 164 Newton runs of the barrier method at about 1.6 ms each.

## Limits

- Only the exterior case is implemented; `synthx` also has the interior case (a walk
  over Delaunay cells).
- The search assumes the paper's general position: simplicial facets, no ties. A donor
  hull of lower dimension, as when predictors are shares of a whole, is refused in both
  classes; `synthx` refuses it in the diagonal class and reduces to the donors' span in
  the positive definite class.
- Proposition 99 has K = 7 predictors and 38 donors. `scaling.py` times both codes on
  seeded random designs of that size, and the study alone at K = 8 and 9. The search
  cost grows quickly with K: from K = 8, some faces have normal cones with too many
  walls for the batched ray enumeration, and those fall back to one LP per sign pattern.
- The timings are single-threaded wall clock on one machine; read the ratios, not the
  absolute seconds.

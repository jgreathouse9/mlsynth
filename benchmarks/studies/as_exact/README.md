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
| least squares on a face | cvxpy with Clarabel, about 5 ms a call | Lawson and Hanson's reduction to a least-distance problem through one thin SVD, solved by one compiled NNLS call (`lsq.py`), 0.12 to 0.2 ms |
| facets visible from X1 | per ridge, a candidate hyperplane for each donor, each checked against every donor | gift wrapping: one QR factorisation and one J x K product per ridge (`hull.py`) |
| allowed sign patterns of a face | extreme rays with every donor as a wall, or one LP per pattern | facet: the normal's signs; ridge: sorted sign changes of a two-generator cone; smaller faces: none listed, the face searched by branch and bound over sign sub-cubes, each test one NNLS with the link vertices as walls whose Farkas certificate picks the split; edges: at most K + 1 residual intervals (`patterns.py`) |
| pruning a face | the best unconstrained fit on the face | the same bound, then one solve with the signs every piece of the face shares (Balas, Disjunctive Programming, 2018, Theorem 1.2) before the face is searched (`exact.py`) |
| predictor weights V | one LP per predictor and per inequality for the diagonal class's support and implicit equalities; cvxpy `log_det`, analytic centre and semidefinite range programs | one LP for the support and implicit equalities together (Schrijver, Theory of Linear and Integer Programming, sec. 8.1); Newton's method with the equalities eliminated and the barrier method (Boyd and Vandenberghe, sec. 10.2 and ch. 11), started at the paper's own Lemma B.1 construction, with a line search that asks for the objective alone (`rationalize.py`) |

The faces containing a given face are reached by walking ridges that contain it; in
a simplicial polytope they form a connected subgraph of the facet-ridge graph (the
polar is simple; Achatz and Kleinschmidt in Kalai and Ziegler, eds., Polytopes:
Combinatorics and Computation, 2000, p. 155). The link vertices they expose are the
only walls of the face's normal cone, which keeps each pattern test small.

Below the ridges, a face's patterns are not listed. `patterns.face_min` searches the
face by branch and bound over sign sub-cubes. A node fixes some signs and stands for
every sign vector that completes them. Its relaxation keeps only the fixed rows
s_k r_k(w) >= 0, so one least-squares solve bounds every piece below it, and a node
whose bound cannot beat the incumbent is dropped without a cone test. Otherwise the
relaxation's minimiser w is tested: when some c in the cone has the node's signs and
c_k r_k(w) >= 0 on the free coordinates, w is attainable and settles the node. When
none has, the test's Farkas certificate names the free coordinates whose signs at w
the cone cannot meet together, so every attainable piece below the node flips one of
them. The node splits into one child per named coordinate, the j-th keeping the
first j - 1 at w's signs and flipping the j-th. Every child excludes w, so its bound
rises, and a node whose certificate uses fixed signs alone ends. Each test is a
least-distance problem solved by one NNLS, whose verdict carries its evidence: a
point that satisfies the rows, or a Farkas certificate read off the same NNLS
(Dantzig and Thapa, Linear Programming 2, 2003, sec. 6.7, a Phase I by least squares;
Schrijver, Theory of Linear and Integer Programming, 1986, sec. 7.3). HiGHS decides a
test only when neither checks, and a sub-cube it calls empty is split on every signed
coordinate.

The search takes from the SYNDES and MAREX branch and bounds
(`mlsynth/utils/syndes_helpers/branch.py`, `mlsynth/utils/marex_helpers/branch.py`)
what applies: best first, a child queued under its parent's bound and solved when it
comes off the heap, and a node dropped only on a least-squares fit that cannot beat
the incumbent, a region found empty, or a certificate that checks, with HiGHS
deciding where none does. SYNDES's reliability branching (Achterberg, Koch and
Martin, 2005) has nothing to score here. Split on one coordinate, the side that keeps
w's sign still holds w and so keeps its parent's bound, and the smaller of the two
rises is always zero. The certificate closes that side instead: with every coordinate
it names fixed at w's signs the cone is empty, so the children above are the whole
split. On the 16 seeded designs of `scaling.py` the search ran on 1,549 faces below
the ridges and ended after one test on 1,074 of them, before any test on 13 (dropped
on their bound), and after at most eight on the rest.
`test_patterns.py` checks the search against one least-squares solve per sign vector
that one HiGHS LP admits, on faces of every size.

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

A third defect was in the least-squares layer, and the authors' code was not the one
that exposed it: a prototype that tests sign patterns with `lsq._ldp` disagreed with
HiGHS on one pattern. Lawson and Hanson's least-distance step decided feasibility by
an absolute floor of 1e-12 on the NNLS residual, and an infeasible system with
margin delta leaves a rounding residual of about eps / delta, so below a margin of
about 1e-4 it returned the quotient of two rounding numbers as a solution. The step
now decides by the identity r_{n+1} = -||r||^2 that a genuine residual satisfies,
every returned point is checked against its constraints, and a face whose validation
block is rank-deficient or has condition number above 1e4 is refused instead of
solved through a 1e-12 ridge. The defect fired once on the committed designs, on the
K = 9 design seed 9000, on a piece that was in fact empty, so no answer moved; old
and new code give identical answers on every other design. `test_lsq.py` holds the
diagnosis as tests, and six mutants in `tools/mutation/targets.toml` (target
`as-exact-lsq`) reintroduce its causes; all six are killed. The checks cost about
14 per cent on the Proposition 99 diagonal search.

A fourth came from SciPy. On a feasible system whose feasible set is one point,
scipy 1.18.1's `nnls` returns a success status and a residual norm of 0 for an answer
that is no minimiser (its true residual is 1.16, worse than the zero vector's). The
checks above kept a wrong point from coming back, but `_ldp` called the system
infeasible. Every NNLS answer is now checked against the optimality conditions and
solved again by bounded-variable least squares when it fails them, and an empty piece
is confirmed by a verdict that carries evidence, or by HiGHS. The system is kept in
`results/nnls_incident.json`; on 56,073 solves from the applications and the seeded
designs the largest violation of the conditions is 6.1e-13 of scale, so the check
never fires there.

## Speed on Proposition 99

Median wall clock of 10 calls after one warm-up, in seconds; `results/raw.json` holds
every call and the interquartile ranges.

| panel | task | replication | synthx.estimate | this study | speed-up |
| --- | --- | ---: | ---: | ---: | ---: |
| Proposition 99 | donor weights, diagonal (Table 1) | 1.635 | 1.652 | 0.065 | 25x |
| Proposition 99 | donor weights, positive definite (Table D.1) | 0.660 | 0.629 | 0.049 | 13x |
| Proposition 99 | V, diagonal (Table 2) | 0.088 | | 0.003 | 25x |
| Proposition 99 | V, positive definite with ranges (Table 4) | 0.418 | | 0.115 | 3.6x |
| leakage-free | donor weights, diagonal | 2.010 | 2.023 | 0.105 | 19x |
| leakage-free | donor weights, positive definite | 0.669 | 0.568 | 0.050 | 13x |

Repeated runs move these ratios by 10 to 20 per cent. Taken together, Tables 1 and 2
come back 25x faster, Tables D.1 and 4 6.6x, and all four 12x.

The speed-up is against the replication. `synthx.estimate` runs within 15 per cent of
it, either way, since it calls the same solvers. Where the time goes, from
`breakdown.py` (`results/breakdown.json`):

- Donor weights, diagonal class. The replication spends 54 per cent of its time in
  186 cvxpy quadratic programs at about 5 ms each, 23 per cent enumerating the
  visible facets, and 15 per cent in 170 SciPy LPs that test sign patterns face by
  face. This study spends 46 per cent gift-wrapping the 132 visible facets and 43 per
  cent in 343 least-squares solves at 0.12 to 0.2 ms each. Proposition 99 reaches one
  face below the ridges, which the branch and bound settles with one least-distance
  test, and 17 faces are ruled out by the common-sign relaxation before they are
  searched.
- V, diagonal class. The replication finds the support of V and its implicit
  equalities with 40 LPs, about 1.6 ms each, mostly in SciPy's HiGHS wrapper; here one
  LP finds both, and 9 Newton runs follow, about 5 ms in all.
- V, positive definite class. The replication's 17 semidefinite programs, at about
  22 ms each, become 164 Newton runs of the barrier method at about 0.5 ms each; a
  trial step of the line search costs one Cholesky factor.

## Limits

- Only the exterior case is implemented; `synthx` also has the interior case (a walk
  over Delaunay cells).
- The search assumes the paper's general position: simplicial facets, no ties. A donor
  hull of lower dimension, as when predictors are shares of a whole, is refused in both
  classes; `synthx` refuses it in the diagonal class and reduces to the donors' span in
  the positive definite class.
- Each face's least-squares fit needs a validation block of full column rank with
  condition number below 1e4. A face with more free weights than validation periods,
  or with donors whose validation paths are nearly collinear, raises ValueError; cvxpy
  in the authors' code solves such a face. Proposition 99, West Germany and the seeded
  designs stay below 600.
- Proposition 99 has K = 7 predictors and 38 donors. On eight seeded random designs
  of that size (`scaling.py`, `results/scaling.json`) the diagonal search here took
  0.33 to 1.7 s against the replication's 8.8 to 118 s, 26x to 77x faster (median
  66x); on four designs at K = 8 it took 1.4 to 4.2 s against 38 to 548 s, 27x to
  228x faster (median 97x). The weights agree to 1.1e-10 in every design. The search
  cost still grows quickly with K, now through the least-distance tests that list the
  sign patterns of faces below the ridges: at K = 9 four designs took 3.3 to 20 s on
  their own, with 3,354 to 108,897 tests, and HiGHS decided none of them. Before the
  branching the slowest of these designs took 219 s, nearly all of it in one LP per
  sign pattern.
- The timings are single-threaded wall clock on one machine; read the ratios, not the
  absolute seconds.

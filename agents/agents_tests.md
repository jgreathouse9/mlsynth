# agents_tests.md

# Testing Philosophy for `mlsynth`

The `mlsynth` testing framework is designed to validate:

* econometric correctness
* numerical stability
* optimization feasibility
* API contracts
* orchestration behavior
* exception semantics

Tests should validate *behavior and invariants*, not implementation details.

The guiding principle is:

> `mlsynth` tests validate econometric behavior and public API guarantees rather than internal implementation details.

---

# Test-Driven Development (test-first is the default)

**Write the tests before the code.** Whenever you add a new feature, helper,
function, estimator branch, config option, or inference mode, its tests come
*first*: write them, run them, and watch them fail for the right reason (red),
then implement until they pass (green). Tests are part of the unit of work, not
a follow-up chore — this is what keeps the contract pinned before the
implementation can drift, and turns every case into a permanent regression
guard.

Every new unit of behavior ships with **at least** these levels:

- **Smoke** — it runs end-to-end on a minimal valid input and returns the
  expected type / a finite result. Proves the happy path is wired.
- **Unit** — its core invariants hold: feasibility, normalization
  (`weights.sum() == 1`), dimensional correctness, the specific contract the
  function promises. Assert *invariants*, not brittle floats (see below).
- **Edge** — boundary and degenerate inputs: empty donor pool, single donor,
  no pre-periods, treatment at the first period, near-singular / collinear
  matrices, target outside the donor hull, `J > T0`. Econometric code fails at
  the edges; cover them deliberately.
- **Failure** — invalid inputs raise the correct *translated* exception
  (`MlsynthConfigError` / `MlsynthDataError` / `MlsynthEstimationError`), and a
  test asserts the failure is **reported** (right type, informative message),
  never silently swallowed or leaked as a raw solver / NumPy / CVXPY error.

A change is not "done" until it is red→green across these levels **and** the new
code is fully covered. Genuinely unreachable / defensive branches are excluded
with `# pragma: no cover` plus a one-line reason — never with an untested gap.
Measure with the per-estimator coverage command in `CLAUDE.md`.

The layered architecture below says *where* each level lives; this section says
the levels are *non-optional* and come *first*.

---

# Cross-Implementation Differential TDD (matching a reference)

Most of `mlsynth` is validated against a **reference implementation** — the
paper's R package, a reference repo, an authoritative competitor (the
replication contract in `CLAUDE.md` and `agents_benchmarking.md`: Path A / Path
B / cross-validation). When a port *disagrees* with its reference — a number is
off, a band is too wide, a weight vector drifts — the temptation is to stare at
the Python and guess. Don't. **Use TDD across the two implementations**: treat
the reference as a chain of testable units and assert mlsynth against it at
every joint, the same way you'd red→green a single function — except the
"expected" value is dumped from the other implementation.

This is what turns "the output is wrong somewhere" into "the disagreement is
`Calculate.PValue`, and it's `1 - mean(<)` vs `mean(>=)` at the discrete
threshold." It works because a faithful port has a faithful *seam structure*:
the two implementations compute the same intermediates in the same order, so
they can be compared intermediate-by-intermediate, not just end-to-end.

## The recipe

0. **Pre-flight: confirm the reference implements the *same version* of the
   spec.** Before mapping a single seam, check that the reference targets the
   paper (and the *edition* of the paper) you target. A method can evolve between
   a working paper and its published version, and two faithful implementations of
   "the same" estimator can then optimize genuinely different objectives. If you
   skip this, the seam will "disagree" and you will be tempted to *break* a
   correct port to match a reference that is solving a different problem. Verify
   first; only a same-spec reference earns a bit-for-bit comparison.
1. **Decompose both sides into matching units.** Find the function boundaries
   that correspond across the implementations (`Spline.Trend` ↔
   `_build_detrend_matrix`, `MASS::ginv` ↔ `_ridge_ginv`, `CV.lambda` ↔
   `_cv_lambda`, `HAC.Meat` ↔ `_hac_meat`, `Calculate.PValue` ↔
   `_conformal_pvalue`). The seams are where you'll compare.
2. **Dump the seam, not just the endpoint.** Have the reference write its
   intermediates out (`write.csv` of the moment matrix, the weight vector at one
   grid point, the per-step p-values) and load them in Python. The breakthroughs
   come from pinning the *intermediate*, not the final answer — the final answer
   only tells you *that* it's wrong, a dumped seam tells you *where*.
3. **Share the input.** Feed both sides the *identical* input at the seam you're
   testing — same matrix, same grid, same λ. Differential testing only assigns
   blame if the input is held fixed; otherwise a mismatch could be either side's
   fault. (Feeding mlsynth the reference's exact grid is what proved a residual
   gap was the *grid*, not the refit — the p-values matched to 1e-17 on the
   shared grid.)
4. **Bisect the disagreement.** Confirm agreement upstream, unit by unit; the
   fault lies downstream of the last seam that still agrees. Walk inward until it
   collapses onto one boundary. "Everything up to the HAC matches, so it's the
   HAC or below" is the move that makes a six-cause bug tractable.
5. **Reason about each disagreement — it is usually meaningful.** A seam
   mismatch is a finding, not noise: a different solver root, a truncation
   tolerance, a floating-point reassociation. Name *why* before you "fix" it, so
   the fix is faithful rather than a fudge that happens to match on this dataset.
6. **Pin the resolved boundary.** Distill each differential probe into a
   permanent unit test that pins the *reference form* (e.g. the p-value is
   `1 - mean(|r| < s)`, including the boundary case), and add a **durable
   benchmark** (`benchmarks/cases/<name>.py`) that re-runs the reference live and
   cross-checks the end-to-end output. The scratch probes are scaffolding; the
   unit test + benchmark are the standing guard.

## Lessons (learned the hard way)

- **Audit your own harness first.** The comparison scaffold can be the bug. A
  driver that did `T <- length(y)` silently shadowed R's built-in `T` (=`TRUE`),
  corrupting `bs(intercept=T)` and inventing a phantom disagreement. If the
  reference suddenly disagrees with *itself* across two runs, suspect the
  harness, not the library.
- **Algebraically equal ≠ bit-for-bit.** `1 - mean(x < s)` and `mean(x >= s)`
  differ by one ULP at a discrete threshold; the inversion's `>= valid_p` test
  then includes/excludes a boundary point and the interval width jumps ~13%.
  Match the reference's *exact* expression, not a tidier equivalent.
- **Estimators have method-dependent roots.** statsmodels' default state-space
  AR(1) ML and R's `arima` (CSS-ML) converge to *different* coefficients on a
  near-unit-root series; `innovations_mle` reproduces R's. When a plug-in
  (bandwidth, tolerance, penalty) is fed by a sub-estimate, the sub-estimate's
  *method* is part of the contract.
- **Near-singular truncation is λ-dependent.** A pseudo-inverse cut on a raw
  rank test deviates from `MASS::ginv`'s cut on the *penalised* eigenvalue once a
  ridge floor lifts the borderline direction — match the tolerance the reference
  actually applies.
- **Mind the achievable discrete level.** A short pre-period caps the conformal
  level (the finest is `2/(T0+1)`); the reference errors or coarsens there, and
  the comparison must use the same achievable α rather than a nominal 0.05.
- **Isolate the disagreement to a *scalar*, then find the line that computes
  it.** When end-to-end numbers are close-but-off, do not sweep hyper-parameters
  hoping to land the match — that is guessing, and it can "fix" the wrong thing.
  Instead collapse the gap onto one constant (force a candidate value and watch
  the whole answer snap into place: if forcing `sdx = 0.07030654` makes every
  downstream number match to <1%, the *entire* discrepancy is that one scalar),
  then go read the reference's source for where that scalar is born. The PPSCM
  covariate `sdx` was a raw pooled-outcome sd in mlsynth vs
  `sd(X[[1]][is.finite(trt)])` in augsynth (multi_synth_qp.R:98) — the sd over
  the treated rows of the *first cohort's residual* block. No amount of formula
  guessing found it; reading the one line did, in seconds, once the scalar was
  isolated.
- **Dump the reference's intermediate, do not re-derive it.** You have R and
  Python side by side — use R to emit the exact object the reference feeds into
  the next step, load it in Python, and diff element-by-element. `trace(pkg:::fn,
  exit = quote({ ... }), print = FALSE)` injects a dumper at function exit with
  every local in scope; `saveRDS(list(X = as.matrix(X), trt = trt, sdx = sdx),
  path)` round-trips a clean object (prefer RDS over CSV — a `dgeMatrix`/`Matrix`
  or a *list* of matrices silently mangles through `write.csv`, and
  `as.matrix(a_list)` yields a 30×1 array-of-lists that looks like data but
  isn't). Diffing augsynth's scaled covariate matrix against ours showed the
  ratio was a *constant* (`aug / ours ≡ 0.07031`), which proved our z-scoring
  was already bit-exact and localised the whole gap to that one multiplier.
- **`X` may be reassigned before the line you care about.** The value a variable
  holds at function *exit* (what a naive trace dumps) is not necessarily its
  value at the interior line that computed your quantity — `multisynth_qp`
  rebinds `X` from a per-cohort list to donor sub-blocks between entry and exit.
  Read the source to confirm *which* binding feeds the scalar; when in doubt,
  trace the tracer's own value against the reference's reported number (the
  reported `sdx` matched `sd(X[[1]][...])`, not `sd(X)`).

## Worked example — the SPSC conformal bands

`mlsynth`'s SPSC conformal intervals ran ~13% too wide and were flat for a
time-varying ATT, vs `qkrcks0218/SPSC`. Walking the seams pinned **six** stacked
causes — none visible from `conformal.py` alone, because mlsynth agreed with
*itself* perfectly:

| Reference unit            | mlsynth unit                       | Verdict                                            |
| ------------------------- | ---------------------------------- | -------------------------------------------------- |
| `Spline.Trend`            | `_build_detrend_matrix`            | match (1e-16)                                      |
| `MASS::ginv`              | `_ridge_ginv`                      | fixed — cut on `s² + 10^λ`, not a raw rank test    |
| `CV.lambda`               | `_cv_lambda`                       | match (same λ)                                     |
| `arima` AR(1) → bandwidth | `_ar1_params`                      | fixed — `innovations_mle` matches R's root         |
| `HAC.Meat` → `Scale`      | `_scaled_detrend_basis`            | refit must run on the *rescaled* basis             |
| `Calculate.PValue`        | `_conformal_pvalue`                | fixed — `1 - mean(<)`, the ULP that set band width |
| narrow-grid refinement    | `interval_for`                     | per-period SE grid + `10*unit` edge extension      |

The end state is pinned by unit tests (the p-value form incl. the boundary ULP)
and the `spsc_prop99` durable benchmark, which now cross-checks the conformal
LB/UB against live R — so the six-cause bug can never silently come back.

## Worked example — CTSC vs `pgsc` (the pre-flight catch)

`mlsynth`'s `CTSC` is Powell's generalized synthetic control; Philip Barrett's
`pgsc` R package is "the" reference implementation. Mapping the very first seam —
the objective — showed an immediate "disagreement": `pgsc` optimizes a *single
shared* coefficient `b` across all units, while `CTSC` fits *per-unit* slopes
`b_i` and averages them (`α^AE = Σ π_i α_i`). The tempting conclusion was a
`CTSC` bug. It was the opposite. **The reference implements a different edition
of the paper.** Powell's 2017 working paper (which `pgsc` and its vignette cite)
used a shared `α₀`; the published 2022 JBES version's baseline is the per-unit
`α_i` with an average-effect summary — exactly what `CTSC` implements (its
docstring cites Powell 2022). The shared-`b` form survives only as the published
paper's §3.3.3 "homogeneous effects" *simplification*. Had we "fixed" `CTSC` to
match `pgsc` bit-for-bit, we'd have regressed it *away* from the paper it
correctly follows.

Once the version skew was understood, the comparison was reframed as
corroboration on the vignette's homogeneous DGP (true `b = (1,2)`), where both
estimators are consistent for the same truth:

| Estimator | `CTSC` (published, per-unit AE) | `pgsc` (2017, shared `b`) | truth  |
| --------- | ------------------------------- | ------------------------- | ------ |
| one-step  | `[0.886, 1.782]`                | `[0.874, 1.897]`          | `(1,2)`|
| two-step  | `[0.997, 1.992]`                | `[0.987, 1.993]` (aggte)  | `(1,2)`|

The two-step estimates agree to ~1%, both recovering the truth, with the residual
gap explained by per-unit-average vs jointly-shared — not a bug in either. The
lesson: the pre-flight version check (recipe step 0) is not optional; here it was
the entire finding.

## Worked example — PPSCM auxiliary covariates vs `augsynth` (trace the seam)

`PPSCM`'s new covariate mode (augsynth::multisynth Sec 5.2) fit *directionally*
right but landed ~15% off the live reference (`nu` 0.2415 vs 0.2244, ATT −0.011
vs −0.019). The seam walk, and the order that made it tractable:

1. **Prove the untouched path is bit-exact first.** With no covariates, PPSCM
   already matched augsynth to the digit (`nu` 0.2733, both L2s identical). That
   localised the entire bug to the *new* covariate term — the outcome machinery
   was not suspect.
2. **Dump the reference intermediate, diff element-wise.** Traced
   `multisynth_qp` to `saveRDS` the scaled covariate matrix `Z_scale`. `aug /
   ours` was a *constant* `0.07031` across all 47×2 entries → our control
   z-scoring was already bit-identical; the whole gap was one scalar multiplier.
3. **Confirm the scalar is the *entire* gap.** Monkeypatched our scaler to force
   `sdx = 0.07030654`; every downstream number snapped to the reference (<1%).
   Now it was a one-line hunt, not a model debug.
4. **Read the source for where the scalar is born.** `sdx <-
   sd(X[[1]][is.finite(trt)])` (multi_synth_qp.R:98) — sd over the treated rows
   of the *first cohort's residual* block, not the raw pooled outcome sd we used.
   Replicating that on mlsynth's `res[first_cohort]` reproduced `0.07030654`
   bit-for-bit.

Pinned by `test_ppscm_covariates.py` (differential vs a captured live run) and
the `ppscm_paglayan_covs` durable benchmark. The lesson compounds the recipe:
isolate to a scalar → dump-and-diff to prove where it is → read the exact source
line → replicate that expression, not a plausible equivalent.

---

# Core Testing Philosophy

The architecture of `mlsynth` is intentionally layered:

| Layer               | Responsibility         |
| ------------------- | ---------------------- |
| Helpers / utilities | Numerical computation  |
| Estimators          | Pipeline orchestration |
| Results objects     | Stable API contracts   |
| Plotters            | Visualization          |

Tests should respect these boundaries.

---

# Layered Testing Architecture

## Layer 1 — Numerical / Utility Helper Tests

These tests validate low-level numerical and optimization behavior.

### Scope

Examples include:

* optimization helpers
* SCM solvers
* penalty functions
* inference kernels
* branch-and-bound routines
* matrix transforms
* conformal inference methods

### Primary Goals

Validate:

* convex feasibility
* numerical finiteness
* dimensional correctness
* optimization convergence
* stability near boundaries
* singularity handling
* degeneracies

### Preferred Assertions

Good:

```python
assert np.isfinite(value)
assert prob.status == cp.OPTIMAL
assert np.allclose(...)
```

Preferred:

```python
assert np.isclose(weights.sum(), 1)
```

Avoid brittle floating-point equality:

```python
assert weights[3] == 0.42184721
```

Optimization solutions may differ across:

* solvers
* platforms
* tolerances
* implementations

Tests should validate invariants instead.

---

## Layer 2 — Data Utility Tests

These tests validate panel-data integrity and identification assumptions.

### Scope

Examples include:

* `balance`
* `dataprep`
* `logictreat`
* proxy preparation utilities
* cohort construction
* donor pool generation

### Primary Goals

Validate:

* strongly balanced panels
* treatment timing logic
* sustained treatment assumptions
* donor availability
* proper reshaping
* identification validity
* malformed input detection

### Important Principle

Data utility tests enforce econometric assumptions *before estimation begins*.

Examples include:

* no donor units
* no pre-treatment periods
* unsustained treatment
* duplicate observations
* invalid treatment matrices

---

## Layer 3 — Estimator Integration Tests

Estimators in `mlsynth` are orchestration layers.

They coordinate:

* data preparation
* optimization
* inference
* plotting
* result assembly

They should NOT contain heavy numerical logic.

### Scope

Examples include:

* `SCDI.fit()`
* `LEXSCM.fit()`
* `SDID.fit()`
* `PROXIMAL.fit()`

### Primary Goals

Validate:

* successful end-to-end execution
* helper coordination
* branching behavior
* config parsing
* result assembly
* exception translation

Estimator tests should validate:

```python
results = estimator.fit()

assert results is not None
assert results.summary is not None
```

Estimator tests should NOT:

* duplicate helper algebra tests
* re-test optimization primitives
* verify internal matrix calculations

Those belong in helper tests.

---

## Layer 4 — Public API Contract Tests

These tests validate the external user-facing API.

### Scope

Examples include:

* package imports
* estimator constructors
* results object structure
* plotting interfaces
* serialization behavior
* reproducibility

### Examples

```python
from mlsynth import SCDI
```

Validate:

* imports work
* public interfaces remain stable
* outputs expose expected fields
* metadata dimensions align

### Result-contract conformance (`test_result_contract.py`)

The two-family result contract (see `agents_results.md`) is machine-checked by
a shared, parametrized harness. Every migrated estimator is added to the
`OBSERVATIONAL` (or design) list; the harness asserts it returns an
`EffectResult`/`DesignResult`, populates the standardized sub-models, and that
the flat accessors resolve. Lessons from wiring up the first batch:

* **The `fitted` fixture is module-scoped and cascades.** A single estimator
  whose `fit()` *errors* during collection fails **every** conformance test,
  not just its own param. Before adding an estimator to the list, run its fit
  in isolation and confirm the config is accepted — a missing required field
  (e.g. a spatial matrix) surfaces as a wall of unrelated red.
* **Estimators needing non-standard inputs can't join the single-`df` loop.**
  The harness feeds one canonical long panel. Estimators that require a
  two-level panel (MLSC), a spatial weight matrix (SpSyDiD), or at least one
  predictor (SparseSC) cannot be parametrized into it — pin a **dedicated
  in-file** `test_two_family_result_contract` in the estimator's own test file
  instead, asserting the same surface against a fixture that supplies the
  special input.
* **Pass cheap, deterministic config via the param's `extra` dict** so the
  conformance fit stays fast and reproducible: a fixed penalty
  (`{"lambda_": 0.5}`), few EM iterations (`{"d": 2, "n_em_iter": 2}`), or
  explicit lags (`{"outcome_lag_periods": [1, 2]}`).

### Frozen-result tests: `ValidationError`, and the accessor trap

Migrated result objects are frozen **pydantic** models, so mutation raises
`pydantic.ValidationError` — update any test that expected
`dataclasses.FrozenInstanceError`. **Trap:** the flat fields (`att`,
`counterfactual`, `gap`, `pre_rmse`) are now inherited **read-only properties**
with no setter, so assigning to them raises `AttributeError`, *not*
`ValidationError`. To assert immutability of the model itself, mutate a real
**field** (e.g. `res.aite = ...`), not an accessor.

---

# Preferred Testing Patterns

## Prefer Parametrization

Use `pytest.mark.parametrize` extensively.

Examples:

* constraint families
* optimization variants
* inference modes
* penalty types
* solver settings

This mirrors the configuration-driven architecture of `mlsynth`.

---

## Prefer Minimal Synthetic Fixtures

Use:

* tiny balanced panels
* deterministic seeds
* interpretable toy examples

Benefits:

* fast CI
* reproducibility
* readable failures
* easier debugging

Preferred examples:

* 3–5 units
* 5–10 periods
* small synthetic matrices

---

## Prefer Deterministic Tests

Always use fixed random seeds where stochasticity is involved.

Example:

```python
np.random.seed(0)
```

Tests should produce reproducible results across environments.

---

## Prefer Invariant-Based Assertions

Validate:

* feasibility
* dimensional consistency
* normalization
* monotonicity
* finiteness

Avoid:

* exact floating-point equality
* implementation-specific internal states

---

## Prefer Numerical Robustness Tests

Econometric software frequently fails at edge cases.

Tests should explicitly cover:

* near-singular matrices
* simplex boundaries
* empty donor pools
* treatment at first period
* no pre-periods
* degenerate optimization problems
* collinearity

---

# Exception Philosophy

`mlsynth` uses structured exception translation.

Public-facing APIs should expose stable exception types.

## Exception Layers

| Layer                | Exception Type             |
| -------------------- | -------------------------- |
| Config parsing       | `MlsynthConfigError`       |
| Data utilities       | `MlsynthDataError`         |
| Estimation pipelines | `MlsynthEstimationError`   |
| Plotting             | `MlsynthPlottingError`     |
| Internal helpers     | native/internal exceptions |

## Important Principle

Public estimators should never leak:

* raw solver errors
* NumPy internals
* CVXPY internals
* plotting backend errors

Estimator tests should therefore validate translated exceptions:

```python
with pytest.raises(MlsynthEstimationError):
    estimator.fit()
```

rather than low-level internal exceptions.

---

# Results Object Contracts

Results objects define stable API contracts.

Tests should validate:

* required fields exist
* metadata is internally coherent
* dimensions align
* labels match matrix structure
* time indices are consistent

Examples:

```python
assert results.time.n_pre > 0
assert len(results.units.treated_labels) == config.m
```

Result object integrity is more important than exact numerical replication.

---

# Plotting Tests

Plotting tests should validate execution behavior, not visual appearance.

Validate:

* plotting executes without crashing
* correct object types are returned
* plotting exceptions are translated properly

Avoid:

* pixel-perfect snapshot testing
* backend-specific rendering checks

Preferred:

```python
estimator.fit(display_graph=True)
```

with assertion that no exception is raised.

---

# Econometric Testing Philosophy

`mlsynth` is not a generic machine learning library.

Tests should encode:

* causal identification assumptions
* panel structure assumptions
* donor feasibility requirements
* treatment timing logic
* synthetic control geometry constraints

This is a core design principle of the library.

---

# Summary Principle

The central testing philosophy of `mlsynth` is:

> Validate econometric behavior, optimization feasibility, numerical stability, and public API contracts — not implementation details.

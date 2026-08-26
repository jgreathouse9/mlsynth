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

# Vocabulary and Framework

The terminology below follows Jorgensen, *Software Testing: A Craftsman's
Approach*, 4th ed. (CRC Press, 2013), which is compatible with the ISTQB
glossary. Using it precisely is what lets the rest of this document make claims
that can be checked instead of asserted.

## Error, fault, failure, incident (§1.1)

* **Error** — a mistake a person makes. Errors propagate: a specification error
  is amplified in design and again in code.
* **Fault** — the representation of an error in some artifact. Source code,
  a docstring, and a config validator can each carry one. `defect` and `bug`
  are synonyms.
* **Failure** — what happens when the code corresponding to a fault executes.
* **Incident** — the symptom that alerts someone to a failure.

A fault of **commission** puts something incorrect into the artifact. A fault of
**omission** leaves out something that should be there. Omission is the harder
of the two, and the reason is definitional: a failure requires code to execute,
so the failure concept attaches to faults of commission. Absent code cannot
execute, so no amount of running the program reveals it.

## Specified, programmed, tested (§1.3)

Three sets of behaviors, and every testing claim in this repo is a claim about
their overlap:

| Set | In `mlsynth` |
| --- | --- |
| `S` — specified | the estimator as defined by its source paper, plus this repo's own contracts (result families, exception translation, config validation) |
| `P` — programmed | what `mlsynth` actually computes |
| `T` — tested | what the suite exercises |

The regions name the problems. `S \ P` is a fault of omission: the paper
specifies behavior the library does not implement. `P \ S` is a fault of
commission: the library does something the paper never asked for. `S ∩ P \ T`
is untested correct behavior. Testing is the determination of how much of
`S ∩ P` is in `T`, and correctness is meaningful only relative to a chosen `S`
— which for this library means relative to a named paper or reference
implementation, never in the abstract.

## Specification-based and code-based (§1.4, Figure 1.7)

Two ways to identify test cases, with different reach:

* **Specification-based** (black box) derives cases from `S` alone. It
  establishes confidence. It cannot find behavior in `P \ S`, because nothing
  in the specification points at it.
* **Code-based** (white box) derives cases from `P`. It seeks faults and
  supports coverage measurement. It cannot find behavior in `S \ P`, because
  there is no code to look at.

Neither is sufficient alone, and the deficiency of each is exactly the other's
territory. Specification-based methods additionally suffer gaps and
redundancies, both of which become visible once their cases are run under a
code-based coverage metric. That pairing — generate from the specification,
measure against the code — is the whole argument for using more than one
instrument, and the section on instruments below is an application of it.

## Consequence: what each instrument can and cannot reach

This frame settles by definition what was earlier observed by accident.

Mutation runs derive from `P` and produce mutants of `P`, so every mutant they
construct is a fault of commission. A fault of omission has no code to mutate.
The `pcr_weights` guard is the worked case: three mutants on the offending line
were all killed, giving that line a perfect mutation score while the defect
stood, because the missing cutoff was in `S \ P` and mutation never leaves `P`.

Property tests derive from `S` — an invariant is a statement of specified
behavior — and generate inputs, so they expand `T` inside `S`. That is the
region a fault of omission lives in, which is why they reach what mutation
cannot. They in turn cannot judge whether an assertion is strong enough, since
that is a question about sensitivity within `P`.

---

# The Oracle Problem

Jorgensen puts the hard part of a test case in its expected output (§1.2): for
software computing something nobody knows the answer to, the academic response
is to postulate an **oracle** that knows all answers, and the industrial
response is **reference testing**, running the system in the presence of expert
users who judge whether the outputs are acceptable.

This is the central problem of testing `mlsynth`, and naming it explains the
shape of the whole repository. For most estimators there is no oracle. Given a
panel, nobody knows the true counterfactual — that is the entire reason the
estimator exists. So a Layer 4 test on real data has no expected output to
assert, and no amount of test-design technique manufactures one.

`mlsynth` answers this the industrial way, and `benchmarks/` is that answer.
The replication contract is reference testing with the source paper and the
authors' implementation as the expert:

* **Path A** — the paper's empirical result on the authors' data. The expert is
  the published number.
* **Path B** — the paper's Monte Carlo. The DGP supplies a known truth, so this
  is the one route where a real oracle exists, and the estimand is computable in
  closed form.
* **Cross-validation** — an authoritative reference implementation, run live and
  compared cell by cell. The expert is the other program.

Three consequences follow, and they are binding.

1. A benchmark case is not documentation or a nice-to-have. It is the source of
   expected outputs for an entire estimator, so an estimator without one has a
   testable `S` only for its invariants, never for its values.
2. Where a real oracle exists, prefer it. Path B and constructed fixtures with
   analytic answers (a factor model with known loadings, a survival design whose
   estimand is `exp(-rate * t)`) admit exact assertions that reference testing
   cannot.
3. Where no oracle exists at any price, say so and stop. A design whose
   behaviour depends on a quantity the paper never reports is not reproducible
   in principle, and the honest output is a recorded finding, not a widened
   tolerance.

Metamorphic relations are the third way out, and they need no oracle at all:
they assert how an output must *respond* to a transformation of the input,
which is checkable without knowing any correct answer. See the instrument
section below.

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

## What the four levels are, in the classical vocabulary

The levels are not an invention; each is a named specification-based method
(Jorgensen §§5–6), and naming them shows what the set does and does not cover.

| Level | Classical method | Assumption |
| --- | --- | --- |
| Smoke | weak normal equivalence class testing — one representative valid input | single fault |
| Unit | invariant and special-value testing over the valid domain | single fault |
| Edge | boundary value analysis — `min`, `min+`, `nom`, `max-`, `max` on each input | single fault |
| Failure | robustness testing — inputs outside the valid domain, plus exception handling | single fault |

Every one of them varies one thing at a time. That is the **single-fault
assumption**: failures arise from one variable being extreme, so holding the
others nominal is enough. It is what makes the levels affordable — `4n + 1`
cases for `n` inputs instead of `5^n`.

The assumption is not always warranted here, and where it fails the levels have
a blind spot the coverage number will not show. `mlsynth`'s inputs are
**physical** quantities (donor count, pre-period length, panel dimensions) and
they are **dependent** — `N` versus `T0` governs overfitting, `J > T0` changes
which solver branch is feasible, a short pre-period interacts with a
rank-deficient donor matrix. Under the classical selection guidance (§10.5,
Table 10.13) dependent variables indicate decision-table testing, and a
warranted multiple-fault assumption indicates worst-case testing — the cross
product of boundaries, not one boundary at a time.

Two standing gaps follow, and both are open:

* **No worst-case testing anywhere.** Combinations such as "single donor *and*
  no pre-periods *and* collinear" are untested at every estimator. Where an
  estimator's failure modes interact, add the cross product deliberately for the
  two or three inputs that interact, not for all of them.
* **Config validators are a decision table nobody has written down.** Each
  `*Config` enforces a set of conditions with dependencies between them, which
  is exactly the shape a decision table represents. Writing the table first
  makes the impossible rules explicit and shows which rule combinations no test
  reaches.

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
- **When the reference has an *option*, run the reference under *both* settings
  before you conclude anything.** A close-but-off band is ambiguous: our impl
  might have a bug, or it might be faithfully implementing a *different* option
  than the one the reference run used. Generate the reference under each setting
  of the suspect flag and see which our output matches. mlsynth's SCPI band was
  ~15% off the `scpi` tutorial — but running `scpi` under both
  `cointegrated_data=False` and `True` showed we matched `False` to
  Monte-Carlo error and the tutorial had used `True`. That reframed the task
  from "debug a discrepancy" to "implement a missing option", and told us
  exactly which component moved (out-of-sample `e`, the levels-vs-differences
  seam) — no bug hunt required.
- **Read the reference's source to find the seam — it need not be R.** The
  differencing that `cointegrated_data=True` applies is not in the paper; it is
  in `scpi_pkg`'s `funs.py` (`u_des_prep` / `e_des_prep`): difference the donor
  design `B -> ΔB`, drop the first (now-NaN) pre-period via `complete_cases`,
  and bridge the predictand with `ΔP[0] = P[0] - B[T0-1]`. Porting *that*
  expression — not a plausible equivalent — reproduced both bands. A pip
  package is as readable as an R kernel; open it.

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

## Worked example — SCPI cointegration vs `scpi_pkg` (which spec are we?)

mlsynth's `VanillaSC(inference="scpi")` reproduced the `scpi` prediction bands on
one panel but sat ~15% wide on the Mendez German-reunification tutorial. The
temptation is to debug our sampler. The disciplined path was shorter:

1. **Split point from interval first.** The simplex weights matched `scpi` to
   4 dp (Austria 0.2911, USA 0.2728, …). So the synthetic control was identical;
   only the *bands* differed. That localised the gap to the uncertainty model,
   not the fit.
2. **Run the reference under both settings of the suspect option.** The tutorial
   set `cointegrated_data=True`. Regenerating `scpi`'s band under *both* `True`
   and `False` showed mlsynth matched `False` to Monte-Carlo error (mean width
   diff 0.10) and was far from `True` (0.87). Verdict: not a bug — a missing
   option. mlsynth only implemented the levels model.
3. **Split the band into its components to see what moves.** Comparing the
   in-sample (`w`) and out-of-sample (`e`) pieces separately showed cointegration
   shifts the in-sample term a little (~0.1) and the out-of-sample term a lot in
   the far-post years — the levels-vs-differences extrapolation.
4. **Read the reference's source for the exact transform.** `scpi_pkg`'s
   `funs.py` (`u_des_prep`/`e_des_prep`): difference the donor design `B -> ΔB`,
   drop the first (NaN) pre-period via `complete_cases`, predictand bridge
   `ΔP[0] = P[0] - B[T0-1]`. Ported verbatim; a standalone prototype hit the
   `scpi` `True` band (mean diff 0.04) before touching the library.

Pinned by `test_scpi_cointegration.py` (matches both specs; the levels default is
a byte-identical regression guard) and the `scpi_germany_pi` durable benchmark.
Lesson: a close-but-off *interval* against a configurable reference is usually a
spec question ("which option are we?"), not a numerics bug — answer it by running
the reference under each option before you open a debugger.

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

# Root-Cause Analysis: the five whys as a test ladder

Every failure in this library gets diagnosed the same way, and the diagnosis is
written down as tests. This section is the procedure.

The vocabulary is already in place. An *incident* is the symptom that alerts
someone; a *failure* is the execution that produced it; a *fault* is the thing in
the artifact; an *error* is the mistake a person made. Root-cause analysis is the
walk from incident back to error, and the rule here is that each step of that
walk leaves a test behind. A fix with no ladder under it is a fix to a symptom.

## Why a ladder and not a fix

The reason to formalise this is that the top of the ladder lies. The quantity a
reader checks first -- the headline estimate -- is usually the quantity least
sensitive to the defect, because estimators are built to be stable in exactly
that number. A fault can move every donor weight and leave the ATT alone. So
"the number looks right" is not evidence, and a suite that asserts only the
number is not a suite. The worked example below is that situation exactly.

## The ladder

Each rung is a question, and each question has an instrument that answers it.
The rungs are ordered from symptom to cause, so a failure at rung *k* means rungs
below *k* have not been ruled out yet.

| Rung | The why | What it asks | Instrument |
| --- | --- | --- | --- |
| 0 | Why did the analyst notice? | Which reported quantity looks wrong? | smoke / Layer 4 contract |
| 1 | Why is that quantity wrong? | Which *other* outputs moved with it? | example tests on the result object |
| 2 | Why did those outputs move? | Which term, step or branch produced them? | Layer 1 unit tests on the component |
| 3 | Why was that step allowed to do it? | Which invariant does the behaviour violate? | `hypothesis` property tests |
| 4 | Why did the input reach it? | Which contract was never enforced? | edge and failure tests, config validators |
| 5 | Why did nobody notice? | Would the suite have caught this? | mutation, semantic or `cosmic-ray` |

Rung 5 is not decoration. It is the RCA step that asks whether the corrective
action worked, and it is the one people skip. A fix whose mutant survives has not
been verified; it has been asserted. Add the mutant to
`tools/mutation/targets.toml` with a `models` line naming the defect it stands
for, and confirm it is killed.

Rungs branch. One why can have several answers, and each answer is its own
descent -- the malformed objective below has two independent faults that both
reach rung 3. Follow every branch; a ladder with one rung per level is usually a
ladder that stopped at the first plausible story. The next section makes that
branching systematic instead of opportunistic.

## Breadth: the dependency chain

The rungs above are the depth axis. Used alone they find one cause and stop,
which is the failure mode the anti-pattern list names and the one hardest to
notice, because a single cause always reads as a complete explanation.

The breadth axis is the fix. Before descending, write down what the reported
number is a function of, in order, each stage depending only on the ones to its
right:

```
reported number  <-  aggregation  <-  effect series  <-  counterfactual
                 <-  weights  <-  objective + constraints  <-  code / data
```

The chain is what makes breadth finite. Each link has a small, listable set of
ways to be wrong, so at each depth the question stops being "what went wrong"
and becomes "which of these five things went wrong", which is answerable by
test. The standing enumeration for the estimation link is in
`.claude/commands/rca.md`; it runs intercept, simplex constraint, non-negativity
versus strict positivity, objective form, solver status, cheapest first.

Three rules govern the descent.

1. **Clear a link before leaving it.** Record the result for every candidate,
   including the ones that pass. A cleared link is a finding: it is what
   licenses ruling out the whole stage and moving down.
2. **Clearing requires power.** A design where two candidates coincide
   numerically cannot separate them. A single post period makes a total and an
   average the same number, so a units check passes there for a reason that has
   nothing to do with the code being right. Ask what the check would have to see
   in order to fail before believing it passed.
3. **Count the causes, and check the count against the magnitude.** Independent
   faults on different links contribute multiplicatively. Decompose the observed
   error into one factor per link and confirm the factors reproduce it. A ladder
   that reports one cause for a number wrong by a large factor, and cannot say
   where the rest of the factor went, is unfinished.

The number of causes is itself a measured quantity, not something read off the
source. Two suspicious things on one line may be one fault with two symptoms;
the way to find out is to correct each independently and see what moves.

## Confirming you have reached the bottom

Two questions, both of which must answer *no*:

* Would the failure still have occurred if this cause were absent?
* Will the failure recur if this cause is corrected and nothing else changes?

If either answers *yes*, the cause is a contributing factor and the ladder
continues. In practice the bottom rung is almost always a fault of omission --
an invariant nobody wrote down -- which is why rung 3 and rung 5 are where the
real causes live. Faults of commission are found by running the code; absent
assertions are not, and no amount of running finds them.

## The ways this goes wrong

The failure modes of the procedure itself, in the order they are usually hit:

* **Stopping at the incident.** Writing the fix from the reported number alone.
  The number is the thing least likely to move.
* **Taking the docstring's word for it.** A comment saying what the code does is
  hearsay from someone who was not present at the failure. Read the executed
  path -- the objective actually built, the solver actually called.
* **Skipping the obvious candidates.** Constraints, signs and solver status take
  a minute each to check. Skipping them turns the rest of the diagnosis from an
  elimination into a guess, because nothing licenses ruling the stage out.
* **Descending past a link that was never cleared.** Moving to the objective
  before establishing that the constraints are the ones the method specifies
  leaves an untested stage above the one being blamed.
* **Clearing a link on a design with no power.** A check that could not have
  failed under this specification has not passed.
* **Reporting one cause without accounting for the magnitude.** If the number is
  wrong by a factor of sixty and the cause found explains five, the ladder is
  unfinished.
* **Diagnosing only what surfaced.** A defect found on one estimator usually
  lives in a shared helper; check who else calls it before scoping the fix.
* **Declaring victory without data.** The corrective action needs evidence, and
  for a test suite the evidence is a killed mutant, not a passing run.

## Worked example: a malformed synthetic control on Basque

The subject is an SCM variant that minimises `||X_t - X_d w||_1 + lam ||w||_2^3`
over the simplex, run outcome-only on Abadie & Gardeazabal's Basque panel
(treated Basque Country, 16 donors, pre-period 1955-1969).

Rung 0 passes. The correct outcome-only SCM gives an ATT of -0.8946 over 1970
onward; the malformed one gives -0.8915, a difference of 0.3 percent. Every
headline claim in the paper survives. On this evidence the estimator ships.

Rung 1 fails. The correct fit puts weight on three donors -- Madrid 0.483,
Baleares 0.311, Rioja 0.206 -- and the malformed one on five, moving 0.18 onto
Cataluna and 0.07 onto Navarra while cutting Rioja to 0.081. For a method whose
output *is* the weights, that is the entire result.

Rung 2 finds two faults, not one:

* The penalty `||w||_2^3` is minimised on the simplex at the uniform vector, so
  it rewards spreading weight out. Sweeping `lam` confirms the direction: the
  effective support `1 / sum(w^2)` runs 3.29, 3.65, 5.47, 15.98, 16.00 as `lam`
  goes 1e-2, 1, 1e2, 1e4, 1e6. At the top the synthetic control is the simple
  average of all 16 donors. It is an anti-sparsity term wearing a regulariser's
  name.
* The fit loss is `norm1`, not `sum_squares`. That is a different estimand, and
  it separates from the correct one on a pre-period carrying a single outlier.

Rung 3 is where the ladder pays. Both faults violate an invariant nobody wrote
down: the fit loss is homogeneous of degree one in the outcome while the penalty
is homogeneous of degree zero, since `w` lies on the simplex whatever the units.
Their ratio therefore scales as `1/c` when the outcome is multiplied by `c`, and
the estimator is not equivariant to a change of units. Measured on the same
panel, the number of selected donors goes 16, 9, 8, 5, 5, 4, 4 as the outcome is
scaled by 1e-3, 1e-2, 0.1, 1, 10, 1e3, 1e5. GDP in millions and GDP in thousands
give different synthetic controls. The same asymmetry makes the answer depend on
the length of the pre-period, because the `norm1` sum grows with `T_pre` and the
penalty does not: effective support runs 3.15, 3.43, 3.65 at `T_pre` of 5, 10,
15.

Rung 4: `lam` is an absolute constant with nothing normalising it against the
scale of the fit loss, and the docstring's instruction to "use a conic-capable
solver" is a comment, not an argument -- `problem.solve()` takes whatever the
default is. Neither is checked anywhere.

Rung 5, the root: the suite asserted the ATT, and the ATT is the one quantity
this fault does not move. The invariant that separates the correct estimator from
this one -- scale equivariance -- was never asserted, so no execution of the code
could reveal it. Both confirmation questions answer *no*: with an equivariance
property in place the fault fails at authoring time, and with equivariance
asserted for every estimator whose output is interpreted, the class of fault
stops recurring.

The corrective action is therefore not "fix the penalty". It is: assert scale
equivariance as a property test wherever weights are interpreted, and add a
mutant that reintroduces a scale-dependent regulariser to confirm the assertion
notices.

Rung 5 repays the effort twice over, and the second time is the instructive one.
A mutant that puts an unnormalised ridge on the simplex Gram dies to the
equivariance property, which is the result the ladder predicts. A mutant that
normalises the same ridge by the Gram trace survives it -- and has to, because
both sides of a scale comparison carry the normalised term, so the comparison is
satisfied while the fit is wrong. That is the argument in "Two of them are
complements", arrived at from the other direction: a property test cannot reach a
fault that respects the property. Only an absolute pin can, which on Basque means
the replication cases that fix the weights themselves.

Magnitude decides which rung sees a fault, so measure it instead of assuming.
Sweeping the normalised ridge on that panel moves the effective support 2.68,
2.68, 3.78, 5.20, 10.36, 15.74 at relative sizes of 0, `1e-6`, `1e-4`, `1e-2`,
`1e-1`, `1`. At `1e-6` nothing observable changes and the mutant is equivalent --
record it as such and retire it. A floor set at half the donor pool bites only on
the last two. Everything between changes which units the fit rests on while
clearing every property in the file. Knowing which rung a class of fault is
visible from is the point of building the ladder instead of guessing at a fix.

## Worked example: a hand-rolled synthetic control on Proposition 99

Basque shows the depth axis on a fault that hides. This one shows the breadth
axis on a fault that does not hide at all, and is still misdiagnosed by a ladder
that only descends.

The subject is a short outcome-only SCM: weights on the simplex, no covariates,
run on the Proposition 99 panel (`basedata/P99data.csv`, California treated, 38
donor states, 1970-2000, intervention 1989).

Rung 0 fails for once. The reported treatment effect is `-1176.00` against a
literature band of roughly `-14` to `-22` packs. The ratio is `60.28`, and that
number is the instrument for the rest of the ladder.

The chain: reported number <- aggregation <- weights <- objective <- code.

**Aggregation.** One candidate: the reported scalar is not the estimand its name
claims. It fails. The field is called `average_treatment_effect` and is assigned
`np.sum(post_effects)` over 12 post periods. Correcting only this gives `-98.00`.
Factor: `12.0000`.

**Estimation.** Five candidates, cheapest first, four of them cleared:

| # | Candidate | Result |
| --- | --- | --- |
| 1 | intercept fitted where the formulation forbids one | cleared -- the program holds 38 donor weights and no constant |
| 2 | simplex constraint missing or misstated | cleared -- `cp.sum(weights) == 1`, and `sum(w) = 1.0` at the solution |
| 3 | non-negativity strengthened to strict positivity | cleared -- `weights >= 0`, and `w = 0` is attained for 37 donors |
| 4 | objective is not the one the method specifies | FAILS |
| 5 | solver returned without converging | cleared -- `problem.status` is `optimal` |

Candidates 1, 2, 3 and 5 are the ones people skip because they are obvious. They
are also the ones that take a minute each, and skipping them is what makes the
remaining diagnosis a guess instead of an elimination. Clearing them is what
licenses the claim that the objective is the only estimation-stage fault.

**The objective, and how many faults it carries.** The line is
`cp.Minimize(cp.sum(residuals))`. Reading it suggests two independent defects --
the criterion is signed, and it is linear in `w`, so on a simplex the optimum is
a vertex. Correcting each independently settles it:

| objective | ATT | pre-RMSE | donors carrying weight |
| --- | --- | --- | --- |
| `cp.sum(r)` as written | `-98.00` | `133.52` | 1 (New Hampshire 1.00) |
| `cp.sum(cp.abs(r))` | `-19.81` | `1.76` | 6 |
| `cp.sum_squares(r)` | `-19.51` | `1.66` | 6 (Utah .39, Montana .23, Nevada .20) |

Taking absolute values recovers essentially the whole answer, so the vertex
collapse is a symptom of signedness and not a second fault. One cause, two
symptoms. The remaining `L1` versus `L2` gap is `0.30` packs and is an estimand
choice, not a defect. The reading was wrong about the count and the test was
right, which is the third rule above in action.

What signedness does is invert the criterion relative to fit. On this panel the
sum of residuals scores its own optimum at `-2496.80`, where the pre-period RMSE
is `133.52`, and scores the least-squares weights at `-1.95`, where the RMSE is
`1.66`. It strictly prefers the fit that is eighty times worse, because negative
residuals are rewarded without bound inside the simplex. The solution is a fixed
selection rule -- take the donor with the largest pre-period mean, here New
Hampshire at 247.62 packs against California's 116.21 -- that never consults the
shape of the treated series. Factor: `98.00 / 19.51 = 5.0231`.

**The count checks out.** `12.0000 x 5.0231 = 60.28`, which is the ratio rung 0
measured. Two causes, on two links, accounting for the whole discrepancy with
nothing left over. Either one alone leaves a wrong number that still looks like
an answer: fix the aggregation and the estimate is `-98.00`; fix the objective
and it is `-234.16`. Both are the kind of number a reader argues with instead of
rejecting.

**Rung 5.** Neither fault is reachable from the ATT alone, and neither needs a
clever instrument. A units assertion -- the reported scalar equals
`post_effects.mean()` -- kills the first. A pre-period fit assertion -- RMSE
within a small multiple of the donor pool's own dispersion -- kills the second,
and would also have caught the collapse to one donor. Both are absent, so both
faults are faults of omission, and the mutants that reintroduce them
(`sum_squares` to `sum`, `mean` to `sum`) survive any suite that checks only that
the ATT is finite and negative.

# Four Instruments, Four Questions

Testing `mlsynth` uses four instruments, and they are not interchangeable. Each
answers a different question, and whether one can catch a given defect follows
from which question it asks.

| Instrument | Question it answers | Varies | Holds fixed |
| --- | --- | --- | --- |
| `coverage` | Did the suite execute this line? | — | — |
| `pytest` example tests | Does *this* input give the expected output? | — | inputs and code |
| `hypothesis` property tests | Does the invariant hold across the *input domain*? | inputs | code |
| mutation runs | Are the *assertions* strong enough to notice? | code | inputs |

Status: `pytest`, `pytest-cov`, `pytest-xdist`, `coverage` and `hypothesis` are
wired up and run in CI. Mutation runs are wired up as the semantic catalogue in
`tools/mutation`, on a weekly out-of-band workflow; `cosmic-ray` itself cannot
currently be installed (see below).

## Two of them are complements, not substitutes

This is the Figure 1.7 argument applied, and it is settled above by definition,
not by anecdote: property tests generate inside `S`, mutation runs operate inside
`P`, and each is blind exactly where the other looks. `pcr_weights` is the
worked case in that section.

The composition rule follows:

> Hypothesis supplies the inputs; mutation runs audit whether the properties you
> asserted are strong enough to notice the code being wrong.

A mutant surviving a property-backed suite says something specific: no property
separates correct behaviour from this corruption. That is a specification gap,
not a missing fixture.

## Reading a mutation score (§21.1)

The definitions are precise, and the third one is why the score is a diagnostic
and not a target:

* A **mutant** `P'` is `P` with one small source change.
* Given a suite `T` where every test passes on `P`, `P'` is **killed** if at
  least one test fails on `P'`, and is a **live mutant** otherwise.
* A live mutant means one of two things: `P'` is logically equivalent to `P`, or
  `T` is too weak to separate them. Deciding which is **formally undecidable**.
* The **mutation score** is `x / y`, killed over total, and its denominator
  therefore contains an unknown number of equivalent mutants.

So a score below 1 is not a defect count and 1 is not a goal — an equivalent
mutant can never be killed by any suite. Read survivors individually and record
the accepted ones with a reason, the way `# pragma: no cover` records
unreachable branches.

## The two mutation instruments

Mutation runs come in two forms here, and the split follows from the frame
above rather than from tooling convenience.

A **syntactic sweep** (`cosmic-ray`) applies general operators over the syntax
tree — replace a binary operator, flip a comparison, delete a statement —
exhaustively and without imagination. It asks whether any assertion in a module
is weak.

A **semantic catalogue** (`tools/mutation/targets.toml`) applies a short list
of specific defects a reviewer thought plausible, each at the one site where it
would be meaningful. It asks whether *this* defect would be noticed.

Neither subsumes the other. Operators perturb the program syntactically, so
every mutant they build is a fault of commission at a local site. Several of
the catalogued mutants model something else: "report only the first clash" is a
statement insertion at one place, "exclude only this cohort's treated units" is
a name swap that would be noise applied anywhere else. A syntactic sweep cannot
generate those without drowning the meaningful site. Equally, the catalogue is
only as good as the defects someone imagined, which is exactly what a sweep
does not depend on.

The practical rule: generic operator swaps do not go in the catalogue. They
duplicate the sweep partially and worse.

`cosmic-ray` is currently blocked upstream — its `yattag` dependency ships a
legacy `setup.py` that modern setuptools rejects — so only the catalogue runs
today. `module-path`, `test-command` and `timeout` in `targets.toml` are
cosmic-ray's own configuration keys, and `emit_cosmic_ray_config.py` renders a
valid session config from them, so the blocker costs a `pip install` and not a
redesign.

A harness that scores mutants has failure modes of its own, and all four of
these were hit while building this one: a mutant that never applied must not be
reported as a survivor; the unmutated suite must be checked first, or every
mutant "fails" a suite that was already red; cached bytecode must be purged,
because a mutant the same length as the code it replaces matches the `.pyc`
the baseline just wrote; and the module must be restored byte for byte, since a
text-mode round trip rewrites the line endings of a CRLF file. `tools/mutation/README.md`
records each with its test.

## Reading a coverage number (§10.3)

Coverage is a ratio of a specification-based method against a code-based metric,
and it comes with two companions the badge does not show. For a method `M`
generating `m` cases, a code-based metric `S` identifying `s` elements, of which
the cases traverse `n`:

* **coverage** `C(M, S) = n / s` — below 1 means gaps
* **redundancy** `R(M, S) = m / s`
* **net redundancy** `NR(M, S) = m / n` — the useful one: cases per element
  actually reached

A high line-coverage number with high net redundancy is a suite testing the same
few paths repeatedly. `pcr_weights` sat at full line coverage the whole time it
carried the defect, because coverage counts execution, not input variety.

## Which instrument for which test

Choose by what the test claims.

* A **number** — a paper's reported value, a reference implementation's cell, a
  pinned regression — is a `pytest` example. Generated inputs carry no known
  truth, so the replication contract stays example-based.
* A **property** — symmetry, positive semi-definiteness, feasibility,
  normalization, monotonicity, equivariance, or a differential equality between
  two implementations — is a `hypothesis` test. Asserting it at one fixture
  tests an example; asserting it over the domain tests the claim.
* A **named edge case** stays a `pytest` example even when a property test
  covers the same ground. Named edges are documentation, and `@example` pins
  one inside a property test when both are wanted.

Layer 1 helpers are the first target for property tests: pure functions, no
solver, no DGP, microseconds per call. Layer 4 `fit()` contracts are the last —
there is no known truth to assert a generated panel against.

## Metamorphic properties, for statistical code

Exact properties are scarce in econometric kernels. Metamorphic relations — how
an output must respond to a transformation of the input — are abundant, and are
where property testing earns its keep here:

* scale: `Y -> cY` scales the ATT by `c` and leaves the weights unchanged
* location: adding a constant to every outcome leaves the weights unchanged
* permutation: relabelling donors permutes the weight vector identically
* duplication: duplicating a donor leaves the fitted counterfactual unchanged
* monotonicity: pre-period fit is non-decreasing in a regularisation penalty
* differential: two implementations of one program agree to solver tolerance

## Standing constraints (decide once, not per test)

* **Determinism.** Fixed seeds are already required above; property tests
  inherit it. Any run feeding a mutation score must set `derandomize=True`,
  because a flakily-killed mutant corrupts the score.
* **Runtime multiplies.** A mutation run executes the suite once per mutant and
  a property test runs many examples per call, so composing them naively costs
  a multiple of the suite per mutant. Mutation targets the fast deterministic
  layers only, never `fit()`.
* **Solver noise.** `cvxpy` / SCS results move at tolerance. Property tolerances
  come from measured solver spread, not a guessed constant.
* **Equivalent mutants are real.** A surviving mutant is a question, not a
  defect. Do not chase a perfect score. Record accepted survivors and the reason
  they are accepted, the way `# pragma: no cover` records unreachable branches.

---

# The Unix Rules, Applied to Tests

The doctrine — which of Raymond's seventeen rules `mlsynth` adopts, adapts, or
refuses, and why — is in `agents_unix.md`. This section is the testing half, and
it is here because this file is where testing practice is decided.

Five of the rules say something about tests that the sections above do not
already say. The other twelve either govern production code or are already
argued here under another name: Diversity is the four-instrument section,
Generation is `hypothesis` and `parametrize`, Economy is the minimal-fixture
rule, and Extensibility is the result-contract conformance test.

## Separation: a hard-to-test function is a design report

Raymond gives the testing consequence of the Rule of Separation directly (§1.6):
separating policy from mechanism "make[s] it much easier to write good tests for
the mechanism (policy, because it ages so quickly, often does not justify the
investment)."

Read that backwards and it becomes a diagnostic. When a function resists testing,
the usual cause is not a missing fixture — it is policy fused into the mechanism,
and the fix belongs in the source, not in the test.

The plotting layer is the worked case, and `agents_utils.md`'s standing rule
already states the fix: a plotter builds and returns a `Figure`; displaying and
saving are the caller's. The difference this makes to a test is total.

```python
# fused: the only assertion available is "it did not raise"
def test_plot_runs():
    plot_bvss(results)          # calls plt.show(), returns None

# separated: the mechanism's output is in hand
def test_plot_draws_observed_and_counterfactual():
    fig = plot_bvss(results)
    ax, = fig.axes
    drawn = {line.get_label(): line.get_ydata() for line in ax.get_lines()}
    assert "Observed" in drawn and "Counterfactual" in drawn
    npt.assert_allclose(drawn["Counterfactual"], results.time_series.counterfactual)
    assert ax.get_xlabel() and ax.get_ylabel()
```

The first test passes on a figure that plots the counterfactual twice, labels
neither axis, and draws the treatment line in the wrong period. It is the
coverage number in miniature: the line executed, and nothing was checked.

The same reading applies past plotting. A helper that can only be tested through
`fit()` has a seam missing; a helper that reads a config field it does not need
has taken on policy. Both are reported by the difficulty of writing the test.

## Silence: a test that prints is a test that is not asserting

The Rule of Silence (§1.6) applies to the suite twice over.

Diagnostic output in a test is an assertion someone declined to write. If the
value matters enough to print, it matters enough to assert on, and if the reader
needs it to interpret a failure it belongs in the assertion message, where
`pytest` will show it only when it is relevant:

```python
assert abs(att - expected) < tol, f"ATT {att:.4f} vs paper {expected:.4f} (tol {tol})"
```

A passing suite should also be quiet. Prints from library code (`agents_unix.md`
counts them) surface once in a script and once per call across the 426 test
modules here, which is how real output gets lost. A warning a test provokes on
purpose is asserted on with `pytest.warns`, not allowed to scroll past.

## Postel, inverted: generous fixtures, strict assertions

`agents_unix.md` records that `mlsynth` refuses the first half of Postel's
Prescription in production code — a liberal validator turns a malformed panel
into an invalid estimate. Tests invert it, and each half does work in its own
direction.

Be generous in what the fixtures generate. Henry Spencer's note in §1.6 is the
argument: "input generated by other programs is notorious for stress-testing
software... accepting empty lists/strings/etc., even in places where a human
would seldom or never supply an empty string, avoids having to special-case such
situations when generating the input mechanically." That is the case for the edge
level and for `hypothesis` in one sentence — the generator supplies the single
donor, the collinear block, the zero-length post window that no human fixture
author would have thought to write down.

Be strict in what the assertions accept. A tolerance widened until the test
passes is a live mutant with the paperwork pre-filed: the mutation section above
defines a survivor as a corruption no assertion separates from correct behaviour,
and a loosened bound manufactures them. Tolerances come from measured solver
spread, and a tolerance that has to grow is a finding to record, not a constant
to edit.

## Representation: fold the case list into data

Pike's rule 5, by way of §1.6 — data is more tractable than logic. A table of
cases can be read, diffed, and extended; six near-identical test functions can
only be compared by eye.

```python
@pytest.mark.parametrize("n_donors,n_pre,expect", [
    (1,  10, "single donor: weight is exactly 1.0"),
    (0,  10, MlsynthDataError),
    (5,   0, MlsynthDataError),
    (5,   1, "one pre-period: fit is exact and meaningless"),
])
```

The existing parametrization rule below says to do this; the Unix framing says
why it is more than a convenience. The same move at the next level up is the
benchmark case that returns a record instead of printing a table — the record is
data, so it can be compared against a stored one, aggregated across cases, and
re-reported by a different runner.

## Transparency: the failure names the invariant

§1.6 asks that a program "be able to both demonstrate its own correctness and
communicate to future developers the original developer's mental model of the
problem it solves." A test suite is the artifact where that is literally
achievable, and it is achieved through names.

- Name a test for the invariant, not the function: `test_weights_sum_to_one`,
  not `test_pcr_weights_2`.
- One behavior per test. When a failure message cannot say which claim broke
  without the reader opening the file, the test is asserting several things.
- The five-why ladder above is the same rule applied after a failure: each rung
  is a test that names one link in the causal chain, so the ladder reads as the
  explanation.

Compactness (§4.2) sets the ceiling: a reader who has to hold more than a handful
of facts to know what a test claims will not check whether the claim is right.

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

Stated in the vocabulary of the framework section: determine how much of
`S ∩ P` lies in `T`, using specification-based methods to generate cases and
code-based metrics to measure the gaps and redundancies they leave — and, for
the estimators, remember that `S` is a named paper and the expected outputs come
from reference testing against it, since no oracle exists.

---

# Reference

Jorgensen, P. C. (2013). *Software Testing: A Craftsman's Approach*, 4th ed.
CRC Press. Section numbers cited above refer to this edition: §1.1 the
error/fault/failure progression and the omission–commission distinction, §1.2
the oracle problem and reference testing, §1.3 the specified/programmed/tested
sets, §1.4 and Figure 1.7 specification-based versus code-based reach, §§5–6
boundary value and equivalence class methods with the single- and multiple-fault
assumptions, §10.3 the coverage and redundancy metrics, §10.5 and Table 10.13
method selection from variable attributes, §21.1 the formalization of program
mutation.

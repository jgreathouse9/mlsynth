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

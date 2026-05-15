# Module / Helper Design Philosophy

The guiding principle for MLSynth modules and helper systems is:

> Every function should do one thing well.

Modules should not contain giant monolithic estimation pipelines where all logic is embedded into a single `.fit()` function. Instead, estimation should be decomposed into:

1. High-level module orchestration
2. Module-specific helper systems
3. Small reusable task-level helper functions
4. Structured containers for inputs/results/problems

This creates:

- Easier testing
- Easier debugging
- Easier extension
- Better readability
- Better future agent compatibility
- Cleaner optimization pipelines
- Stronger typing and validation

---

# Core Architectural Philosophy

## 1. Modules Orchestrate

Main estimator modules should primarily:

- Validate configuration
- Coordinate pipeline stages
- Call helper functions
- Handle exceptions
- Assemble outputs
- Trigger plotting or inference

The module itself should NOT contain dense mathematical logic whenever possible.

Good example:

```python
# Step 1: Balance
balance(...)

# Step 2: Prepare matrices
prepared_data = dataprep(...)

# Step 3: Estimate
results = fast_DID_selector(...)

# Step 4: Organize outputs
gathered = DID_org(...)

# Step 5: Plot
plot_estimates(...)
````

The module acts as a pipeline controller.

---

# 2. Every Task Gets Its Own Helper

Each logical task should become a helper function used in the fit method.

NOT:

```python
def fit():
    ...
    compute_r2()
    update_mean()
    select_best()
    compute_att()
    build_ci()
    compute_rmse()
    ...
```

INSTEAD:

```python
def _r2_batch(...):
    ...

def _select_best_donor(...):
    ...

def _update_synthetic_control(...):
    ...

def _did_from_mean(...):
    ...

def _compute_fdid_result(...):
    ...
```

See the fdid modules or the lexscm/scdi modules for examples. Each helper should:

* Have one clear responsibility
* Be independently testable
* Be reusable
* Be composable
* Have minimal side effects

---

# 3. Helpers Should Build Larger Tasks Incrementally

Helpers should compose upward.

Small helpers support medium helpers.

Medium helpers support estimation pipelines.

Example hierarchy:

```text
_r2_batch
    ↓
_select_best_donor
    ↓
_forward_selection_loop
    ↓
fast_DID_selector
    ↓
FDID.fit()
```

This layered decomposition is preferred across ALL future modules, or when you're asked to refactor existing ones.

---

# 4. Future Modules Should Use Module-Specific Helper Packages

Older helpers were shared globally across modules.

This should NOT continue for future modules.

Preferred future structure:

```text
mlsynth/
    fdid/
        helpers/
            selection.py
            estimation.py
            inference.py
            plotting.py
            validation.py
            structure.py

    scdi/
        helpers/
            optimization.py
            constraints.py
            inference.py
            plotting.py
            structure.py
```

Each module should own its own helper ecosystem, UNLESS the helper is so general (i.e., the dataprep function in the datautils.py module) that it is readily conceivable that more than one main module would need it. 

Benefits:

* Avoids giant shared utility files
* Reduces hidden coupling
* Makes maintenance easier
* Prevents helper pollution
* Improves discoverability

---

# 5. Structure Submodules Are Required Going Forward

Every future module helper system should include:

```text
helpers/
    structure.py
```

This file contains structured containers/dataclasses.

These dataclasses separate:

* Inputs
* Optimization problems
* Intermediate states
* Estimation outputs
* Inference outputs

from the procedural estimation code.

---

# Structure Philosophy

Structure files should contain immutable dataclasses representing:

## Inputs

Example:

```python
@dataclass(frozen=True)
class SCDIInputs:
    ...
```

Purpose:

* Hold preprocessed matrices
* Store aligned indices
* Preserve metadata
* Standardize estimator inputs

---

## Optimization Problems

Example:

```python
@dataclass(frozen=True)
class SCDIProblemComponents:
    ...
```

Purpose:

* Hold symbolic CVXPY objects
* Separate construction from solving
* Keep optimization pipelines modular
* Enable constraint extension

Optimization containers should generally include:

* Objective
* Constraints
* Variables
* Solver metadata

---

## Design Solutions

Example:

```python
@dataclass(frozen=True)
class SCDIDesign:
    ...
```

Purpose:

* Store optimization outputs
* Preserve selected units
* Store assignment vectors
* Store weights and latent variables
* Preserve raw solver outputs

---

## Inference Outputs

Example:

```python
@dataclass(frozen=True)
class SCDIInference:
    ...
```

Purpose:

* Store p-values
* Hypothesis test results
* Null distributions
* Confidence intervals
* Inference metadata

---

## Final Results Containers

Example:

```python
@dataclass(frozen=True)
class SCDIResults:
    ...
```

Purpose:

* Bundle all outputs together
* Provide clean user-facing API
* Expose convenient properties
* Standardize `.fit()` returns

---

# Dataclass Design Rules

Preferred conventions:

## Use Frozen Dataclasses

```python
@dataclass(frozen=True)
```

Reason:

* Prevent accidental mutation
* Safer optimization pipelines
* Easier debugging
* Functional-style reliability

---

## Include Rich Docstrings

Every structure should include:

* Purpose
* Parameters
* Notes
* Shapes
* Usage semantics

These are part of the public API.

---

## Include Convenience Properties

Example:

```python
@property
def mode(self) -> str:
    return self.design.mode
```

This improves usability without duplicating storage.

---

# Helper Design Rules

## Prefer Pure Functions

Helpers should ideally:

* Depend only on inputs
* Avoid mutating external state
* Return explicit outputs

Avoid hidden state.

---

## Avoid Giant Utility Files

Bad:

```text
utils/helpers.py
```

Preferred:

```text
helpers/
    selection.py
    optimization.py
    inference.py
    diagnostics.py
```

Organize by responsibility.

---

## Helpers Should Be Independently Testable

Every helper should be testable without invoking the full estimator.

If a helper is difficult to test independently, it is likely too large.

---

## Private Helpers Are Encouraged

Internal helpers should use:

```python
def _helper(...):
```

This clearly separates internal implementation from public API.

---

# Exception Handling Philosophy

Modules orchestrate exception translation.

Helpers should raise natural low-level errors.

Main modules should translate them into MLSynth-specific exceptions.

Example:

```python
try:
    prepared_data = dataprep(...)
except Exception as e:
    raise MlsynthDataError(...) from e
```

This preserves:

* Clean stack traces
* User-facing clarity
* Internal debugging detail

---

# Pipeline Design Philosophy

Preferred estimator flow:

```text
Validation
    ↓
Balancing
    ↓
Preprocessing
    ↓
Optimization / Estimation
    ↓
Post-processing
    ↓
Inference
    ↓
Packaging Results
    ↓
Plotting
```

Each stage should have dedicated helpers.

---

# Testing Implications

This architecture exists partly to improve testing.

Because helpers are decomposed:

* Unit tests become simple
* Numerical edge cases become isolated
* Optimization debugging becomes easier
* Failures localize quickly

Every helper should ideally support:

* Shape tests
* Numerical correctness tests
* Error-path tests
* Boundary-condition tests

---

# Desired Future Standard

All future MLSynth modules should aim for:

## Clean orchestration modules

Minimal heavy logic in `.fit()`.

---

## Dedicated helper packages

No giant shared helper files.

---

## Structured dataclass containers

Separate symbolic/state/data layers.

---

## Composable pipelines

Helpers feeding helpers feeding estimators.

---

## Independent testability

Every task isolated and verifiable.

---

# Summary

The MLSynth architectural philosophy is:

> Decompose estimation into small, composable, testable tasks.

Modules orchestrate.

Helpers perform focused work.

Structure files organize state.

Dataclasses standardize outputs.

Everything should remain:

* modular
* testable
* extensible
* typed
* debuggable
* composable
* numerically transparent

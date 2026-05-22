# Estimator Architecture Guide

This document defines the architectural conventions used throughout
the `mlsynth` estimator ecosystem.

Agents should prioritize consistency with existing estimator structure
over introducing novel abstractions.

The repository is designed around:
- typed configurations
- structured orchestration
- decomposed statistical pipelines
- standardized outputs
- explicit exception semantics
- reusable helper modules

New estimators should extend these conventions rather than reinvent them.

---

# Core Philosophy

Estimators in `mlsynth` are orchestration layers.

Estimator classes are responsible for:
- parsing validated configuration objects
- coordinating statistical submodules
- enforcing repository APIs
- assembling standardized outputs
- exposing stable user interfaces

Heavy computational logic should generally NOT live directly inside
estimator classes.

Instead, estimators coordinate specialized helper modules.

---

# Canonical Estimator Workflow

Most estimators follow this high-level pipeline:

1. Validate configuration
2. Balance / preprocess panel data
3. Prepare structured inputs
4. Execute optimization or estimation
5. Run optional inference procedures
6. Assemble typed result objects
7. Optionally render diagnostics or plots
8. Return structured results

This orchestration pattern should remain recognizable across estimators.

---

# Config Architecture

Every estimator must expose a dedicated typed Pydantic config object.

Config models:
- inherit from `BaseEstimatorConfig` or `BaseMAREXConfig`
- use explicit typing throughout
- document all fields with `Field(...)`
- forbid undeclared fields (`extra="forbid"`)
- centralize validation logic
- raise repository-specific exceptions

Avoid:
- free-form kwargs as primary APIs
- hidden parameter mutation
- untyped dictionaries internally
- ad hoc runtime validation

Estimators may accept:
- a typed config object
- or a compatible dictionary

Dictionary inputs should immediately be converted into typed config models.

---

# Validation Philosophy

Validation is considered part of the statistical API.

Validators should enforce:
- panel consistency
- dimensional correctness
- admissible parameter ranges
- identification assumptions
- inferential feasibility
- optimization feasibility
- budget constraints
- temporal consistency

Validation errors should:
- fail early
- be explicit
- include actionable messages

Auto-correction is acceptable only when:
- deterministic
- statistically safe
- accompanied by warnings

---

# Estimator Decomposition

Complex estimators should decompose functionality into reusable helper modules.

Preferred helper categories include:

- setup
- optimization
- inference
- plotting
- structures
- evaluation
- search
- diagnostics
- power analysis

Typical structure:

utils/<estimator>_helpers/
    setup.py
    optimization.py
    inference.py
    plotter.py
    structures.py

Larger pipelines may additionally include:
    evaluation.py
    search.py
    power_helpers.py
    diagnostics.py

Avoid monolithic estimator files containing:
- optimization routines
- inference engines
- plotting systems
- large procedural pipelines

---

# Pipeline Composition

Estimators should expose explicit staged computation.

Examples of acceptable stages:
- data preparation
- matrix construction
- candidate generation
- optimization
- evaluation
- ranking
- inference
- post-processing
- structured assembly

Intermediate artifacts should remain interpretable and reusable.

---

# Results Architecture

Estimators should return typed structured result objects whenever possible.

Preferred result architecture:
- `BaseEstimatorResults`
- estimator-specific result structures
- typed metadata containers
- structured diagnostics
- inference result objects

Avoid:
- opaque nested dictionaries
- unnamed tuple returns
- inconsistent field naming

Results should expose:
- primary estimates
- diagnostics
- metadata
- inference outputs
- execution summaries

---

# Metadata Philosophy

Metadata is considered a first-class component of the API.

Estimators should expose structured metadata describing:
- time dimensions
- unit structure
- candidate selection
- optimization diagnostics
- inference settings
- execution details

Metadata should be:
- typed
- interpretable
- reusable downstream

---

# Exception Semantics

Use repository-specific exceptions consistently.

Available exception categories:
- `MlsynthConfigError`
- `MlsynthDataError`
- `MlsynthEstimationError`
- `MlsynthPlottingError`

Avoid generic public-facing exceptions.

Unexpected internal exceptions should generally be wrapped inside:
`MlsynthEstimationError`.

---

# Documentation Standards

All estimators should include:
- publication-quality docstrings
- mathematical overview
- assumptions
- parameter documentation
- return object documentation
- references
- notes on inference and diagnostics

Estimator documentation should prioritize:
- statistical clarity
- reproducibility
- API discoverability

---

# API Stability

Public estimator APIs should remain:
- typed
- explicit
- stable
- composable

Preferred interface:

```python
estimator = ESTIMATOR(config)
results = estimator.fit()
```


Avoid introducing incompatible API patterns unless explicitly instructed.

Preferred Design Principles

Prefer:

explicit over implicit behavior
typed structures over free dictionaries
reusable helpers over duplicated logic
standardized outputs over custom formats
composable pipelines over monolithic procedures
continuation of repository conventions over novel architecture

Agents should study existing estimators before implementing new abstractions (the best examples are the FDID, SCDI, LEXSCM, SparseSC, SBC, and SPCD modules, as templates).


**Example sections must include a self-contained one-draw Monte Carlo.**
Every estimator docs page's *Example* block has to be copy-paste runnable
from a fresh interpreter — no external data files, no missing imports.
Simulate one panel from the estimator's intended DGP, fit the estimator,
print the headline output. Users should be able to read the page top-to-
bottom, paste the block, and see the estimator work in seconds. Alternatively, if we do indeed have the emprirical data, ask which dataset should be used first/if we have data fro the docs example.

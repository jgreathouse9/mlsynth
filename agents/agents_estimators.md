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


New estimators should extend these conventions rather than reinvent them.

---

# Definition of Done (Replication Contract)

**This is a hard gate. An estimator or module is NEVER considered "done"
— not when it runs, not when tests pass, not when the numbers look
plausible — until its results are validated against the source paper by
ONE of the following two paths.**

## Path A — Empirical replication (preferred)

The estimator reproduces the paper's published **empirical** result(s) on
the **same dataset**, to reasonable machine precision (small differences
from solver tolerances, BLAS, or float order are acceptable; a different
sign, a different donor selection, or a materially different magnitude is
NOT).

To claim Path A you must:
- obtain the **exact data** the paper used, and
- where it exists, obtain the authors' **reference code** and replicate
  against it (not against the paper's prose or display equations — see
  Lessons below), and
- reduce the claim to a **concrete target number** (e.g. "ATT = −952.2,
  weights Portugal 0.44 / Japan 0.37 / Italy 0.16") and show the
  implementation hits it, and
- capture the reproduction as a **runnable empirical example in the
  docs**.

## Path B — Monte Carlo replication (when data are inaccessible)

If the paper's data are **proprietary or otherwise unavailable**, the
estimator must **largely reproduce the paper's own Monte Carlo result**,
and that reproduction must live **in the docs** as a self-contained,
runnable example.

"Largely reproduce" means the headline simulation finding holds — e.g.
the estimator is approximately unbiased across draws when the paper
claims unbiasedness, or it attains the paper's reported MSE-ratio /
size / power pattern — using the paper's own data-generating process and
parameter settings.

## What does NOT count as done

- "It runs without error."
- "The output looks reasonable / has the expected shape."
- "Tests pass." (Unit tests check code behavior, not statistical
  correctness against the paper.)
- Matching the paper's *prose* or *display equations* without checking
  against the authors' reference code.

## Lessons (why this gate exists)

A real episode: an SBC implementation **ran, returned plausible donor
weights, and produced a confident ATT — with the wrong sign.** Three
independent defects were invisible without replication against the
authors' exact code and data:

1. The paper's printed Step-2 equation **dropped a term** that the
   authors' reference code kept; the implementation faithfully copied the
   *paper* and was therefore wrong.
2. A detrending convention lived only in the **code**, not the text.
3. The bundled **dataset had scrambled column labels** — invisible to any
   amount of code review, caught only by a value-by-value diff against
   the source data.

The takeaway: "it runs and the number looks right" is the most dangerous
failure mode, because it survives every check short of replication
against ground truth. Replicate against the **exact code and exact
data**, not the write-up.

---

# Core Philosophy


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
8. Return structured results, including visuals. All estimators need some form of visual.

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

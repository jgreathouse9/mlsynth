# AGENTS.md

## Repository Philosophy

mlsynth is a strongly-typed causal inference and synthetic control library.

Architectural consistency, explicit validation, and standardized estimator interfaces
are core design principles.

Agents should extend existing repository conventions rather than invent new patterns.

---

## Core Design Rules

### 1. Every estimator must define a dedicated Pydantic config object

All estimators expose explicit typed configuration models.

Configuration models:
- inherit from BaseEstimatorConfig or BaseMAREXConfig when appropriate
- use `Field(...)` with meaningful descriptions
- forbid undeclared fields (`extra = "forbid"`)
- centralize statistical and structural validation logic
- raise repository-specific exceptions (`MlsynthConfigError`, `MlsynthDataError`)

Avoid:
- free-form kwargs
- hidden defaults
- runtime parameter mutation outside validators

---

### 2. Validation is part of the statistical API

Validators should enforce:
- panel consistency
- identification assumptions
- dimensional correctness
- admissible parameter ranges
- feasibility constraints
- inferential validity

Fail early with explicit, actionable error messages.

Auto-correction is acceptable only when:
- behavior is deterministic
- user intent is unambiguous
- a warning is emitted

---

### 3. Estimator outputs must use standardized result models

Whenever possible, estimators should return structured result objects built from:
- BaseEstimatorResults
- EffectsResults
- FitDiagnosticsResults
- TimeSeriesResults
- WeightsResults
- InferenceResults
- MethodDetailsResults

Avoid ad hoc dictionaries unless absolutely necessary.

---

### 4. Match existing repository style before implementing

Before creating or refactoring a module:

1. Identify the closest existing estimator(s)
2. Reuse existing patterns and abstractions
3. Match naming conventions
4. Match typing conventions
5. Match documentation style
6. Match validator structure
7. Reuse existing utilities whenever possible

Prefer continuation over reinvention.

---

### 5. Documentation is mandatory

New estimators should include:
- mathematical overview
- parameter documentation
- assumptions
- references
- usage examples
- notes on inference and diagnostics

Docstrings should be clear, technical, and publication-quality.

---

### 6. Preserve stable APIs during refactors

Refactors should:
- preserve public interfaces unless explicitly instructed otherwise
- improve consistency
- reduce duplication
- strengthen typing
- improve diagnostics
- improve validation quality

Avoid unnecessary rewrites of stable logic.

---

## Canonical Reference Implementations

The following modules should be treated as architectural references:
- MAREXConfig
- LEXSCMConfig
- RESCMConfig
- BaseEstimatorResults hierarchy

Study these implementations before introducing new architecture.

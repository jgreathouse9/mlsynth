# agents_results.md

Agent-facing contract for **what an mlsynth estimator returns** and the
**Definition of Done (DoD)** for migrating an estimator onto it. Read this
together with `agents_estimators.md` (how an estimator is built),
`agents_docs.md` (docs conventions), and `agents_tests.md` (testing layers).

> **Status.** The contract is validated end-to-end on the three reference
> estimators **VanillaSC, FDID, TSSC** (see `mlsynth/tests/test_result_contract.py`).
> Roll it out one estimator (or one family) at a time, applying the DoD below
> to each.

---

## 1. The two-family result contract

mlsynth has exactly **two** output families, because panel causal inference
has exactly two modes — *measure an effect*, or *design to measure one* — and
the design mode resolves to the measurement mode. Every `fit()` returns one of
two types, both subclassing `MlsynthResult` (in `config_models.py`, re-exported
from `mlsynth/results.py`):

| Family | Type | `fit()` returns | Meaning |
|---|---|---|---|
| **Observational** | `EffectResult` (alias of `BaseEstimatorResults`) | `EffectResult` or a subclass | An *observational report*: ATT, counterfactual, weights, inference. |
| **Experimental / design** | `DesignResult` | `DesignResult` | A *research design*: assignment, selected units, design weights, power. Its `report` is an `EffectResult` once outcomes exist. |

Invariant: `isinstance(result, MlsynthResult)` always holds, and every
`DesignResult.report` (when set) is an `EffectResult`. The conformance test
pins this.

### How a result class should look

- **The top-level container is a Pydantic model** subclassing the correct
  base:
  - observational → `class FooResults(BaseEstimatorResults):`
  - design → `class FooResults(DesignResult):`
  - **Freeze where practical**: set
    `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)` on
    the leaf result class when the pipeline builds it in one shot. The shared
    `BaseEstimatorResults` base is intentionally left mutable so it can serve
    as a building block (e.g. an estimator that returns it directly, or holds
    one as a `summary`); immutability is a leaf-class property, not a
    hard requirement of the base.
- **Estimator-specific outputs stay typed** as fields on the subclass (factor
  matrices, QTE curves, selection trees, …). Do **not** dump them into
  `additional_outputs: dict` — that loses typing.
- **Nested helper containers may remain frozen dataclasses** (e.g.
  `FooInputs`, `FooMethodFit`); the Pydantic container holds them via
  `arbitrary_types_allowed`. No need to convert the whole tree.
- **Populate the standardized sub-models** so the common surface is uniform:
  `effects`, `time_series`, `weights`, `inference`, `fit_diagnostics`,
  `method_details`. `WeightsResults` is the single weights container and
  exposes `donor_weights` / `time_weights` / `unit_weights` — populate
  whichever faces apply (donor mapping, SDID-style time weights, MCNNM-style
  unit-weight arrays). **Prefer pipeline-native population** (the pipeline builds
  these directly, so the object is *born standardized*); a `model_validator`
  that back-fills from the primary fit is acceptable when a pipeline rewrite is
  out of scope — if used, mutate via `object.__setattr__` (the model is frozen).
- **Flat convenience accessors** (`att`, `att_ci`, `counterfactual`, `gap`,
  `donor_weights`, `pre_rmse`) are inherited from `BaseEstimatorResults`;
  override them only in dispatchers (see below).
- **Dispatcher estimators** (multiple variants, e.g. TSSC/FDID/PDA/PROXIMAL/
  CLUSTERSC) keep variants in `sub_method_results` (or a typed `variants`
  field) and delegate the flat accessors / standard sub-models to the selected
  variant. "Which variant" is an intra-`EffectResult` concern, **not** a third
  family.
- **CI / inference** always lands in `InferenceResults`
  (`ci_lower`/`ci_upper`/`p_value`/`standard_error`/`method`); `att_ci` reads
  from it. Do not invent a bespoke per-estimator CI field as the public surface.
  The `inference` slot is **reserved** for this — if the estimator already ships
  a rich `inference`-named object (jackknife/SCPI/placebo/posterior bands),
  rename it to `inference_<kind>` and mirror it into `.details` (see §4.1).
- **Per-period prediction bands** land on the canonical `TimeSeriesResults`
  fields, not in a per-estimator object: `counterfactual_lower` /
  `counterfactual_upper` (pointwise) and `counterfactual_lower_simultaneous` /
  `counterfactual_upper_simultaneous`, aligned to `time_periods` (NaN where the
  method has no band), tagged with `prediction_interval_level` and
  `prediction_interval_kind`. This is the *per-period band on the counterfactual*;
  the scalar ATT interval stays on `InferenceResults`. Populate it through
  `build_effect_submodels(..., prediction_interval=...)` (which also auto-derives
  it from an `inference.details` mapping carrying `counterfactual_lower/upper`),
  or, for a scpi fit, from `ScpiPIInference.to_prediction_interval_spec()`.
  `compare_estimators` / `compare_counterfactuals` read this one field, so any
  estimator that populates it lines up in a cross-method overlay for free.

### Forbidden

- ad-hoc `dict` or unnamed `tuple` as the public return,
- inconsistent field naming for the same quantity (`counterfactual` vs
  `counterfactual_full`; raw `weights` vector where a `donor_weights` mapping
  is meant) — for matrix-completion estimators the 1-D path is `counterfactual`
  and the `(N, T)` matrix is `counterfactual_matrix` (see §4.2),
- mutating a result after construction outside a validator.

---

## 2. Definition of Done — per-estimator migration

A migration is **not** done when tests pass. It is done when **all** of the
following hold. Use this as a literal checklist in the PR description.

### A. Code

- [ ] `fit()` return annotation is `EffectResult`/subclass (observational) or
      `DesignResult` (design); no `dict`/`tuple` in the public return.
- [ ] Result container is a Pydantic model subclassing the correct base with
      `arbitrary_types_allowed=True`; **frozen where practical** (freeze the
      leaf class when the pipeline builds it in one shot — the shared base may
      stay mutable).
- [ ] Standardized sub-models populated (`effects`, `time_series`, `weights`,
      `inference`, `fit_diagnostics`, `method_details`), pipeline-native where
      feasible.
- [ ] Estimator-specific outputs retained as **typed** fields (no regression to
      `additional_outputs` dicts).
- [ ] **Back-compat preserved**: every previously public attribute/method still
      resolves (add `@property` shims if internals moved). If a name *must*
      change, keep the old one as a deprecated alias.
- [ ] Design estimators: `report` is populated with the `EffectResult` (fold
      any `post_fit`/`summary`/`SyntheticControlPostFit` into it).

### B. Tests

- [ ] Estimator added to the appropriate list in
      `mlsynth/tests/test_result_contract.py` (`OBSERVATIONAL` or the design
      list) and its conformance tests pass.
- [ ] Existing estimator test file updated for any intentional surface change
      (e.g. frozen-exception type: pydantic `ValidationError`, not
      `FrozenInstanceError`).
- [ ] Full suite green: `pytest mlsynth/tests/`.

### C. Docs (`docs/<name>.rst`)

- [ ] **Return-object documentation** reflects the standardized surface (what
      `res.att` / `res.effects` / `res.weights` / `res.inference` expose).
- [ ] The runnable **Example** uses the canonical accessors (`res.att`,
      `res.att_ci`, `res.counterfactual`, …), not removed/internal paths.
- [ ] `.. autoclass::` / `:class:` references point to the **actual** module
      (configs now live in `utils/<name>_helpers/config.py`).

### D. Replication / benchmarks

- [ ] Replication **numbers are unchanged** (return-type refactors must not move
      estimates); `benchmarks/cases/<name>.py` still passes
      (`python benchmarks/run_benchmarks.py`).
- [ ] Any code snippet in `docs/replications/<name>.rst` that reads the result
      uses the canonical accessors.

### E. Agent instructions & indices

- [ ] If the public surface changed, regenerate `llms.txt`
      (`python tools/gen_llms_txt.py`).
- [ ] If a new pattern was introduced, update **this file** and the
      "Results Architecture" pointer in `agents_estimators.md`.

### F. Migration notes

- [ ] A `CHANGELOG` entry records the migration: what the estimator now
      returns, any renamed/aliased public attributes, and the back-compat
      guarantee. (This is the user-facing record; B/C/D are the proof.)

### G. Done means done

- [ ] Working tree committed with a clear message on the assigned branch
      (author/committer `noreply@anthropic.com`); no stray debug code.

---

## 3. Rollout order (suggested)

1. **Reference set (done):** VanillaSC, FDID, TSSC — the contract + conformance
   test live here.
2. **Single-method observational** estimators (flat `att` already present):
   cheapest; mostly add the conformance entry + verify sub-models.
3. **Dispatchers** (PDA, PROXIMAL, CLUSTERSC, MUSC, SCMO, SIV): apply the
   variant-delegation pattern.
4. **Design family** (MAREX, PANGEO, SYNDES, SPCD, LEXSCM): migrate onto
   `DesignResult`, folding `post_fit`/`summary` into `report`.

Each estimator carries its own DoD checklist; do not batch-migrate without
running the per-estimator docs/replication/conformance steps.

---

## 4. Conventions learned from the migration

These are the concrete, recurring decisions distilled from migrating the first
batch of estimators (SPOTSYNTH → RMSI → SNN → MSQRT → MCNNM → MLSC → SparseSC →
TASC → SpSyDiD). The principles above say *what* the contract is; this section
says *what to actually do* when you hit the patterns that keep coming up.

### 4.1 The `inference` slot is reserved — rename pre-existing rich objects

The `inference` slot holds **only** the standardized `InferenceResults`
(`ci_lower`/`ci_upper`/`p_value`/`standard_error`/`method`). Estimators
routinely arrive with a field *already named* `inference` (or a similarly
generic name) holding something that is **not** a standardized CI object —
a jackknife container, an SCPI interval bundle, conformal/placebo output,
posterior bands, or (MLSC) the fitted paths that were never inference at all.

**Do not overload the slot.** Rename the raw rich object to
`inference_<kind>` and populate the standardized slot alongside it, mirroring
the raw object into `InferenceResults.details`:

| raw object                         | renamed field        |
| ---------------------------------- | -------------------- |
| jackknife SEs / draws              | `inference_jackknife`|
| SCPI / prediction intervals        | `inference_intervals`|
| placebo / conformal / posterior    | `inference_detail`   |
| fitted counterfactual paths (MLSC) | `paths`              |

```python
InferenceResults(method="snn_jackknife", standard_error=se,
                 ci_lower=lo, ci_upper=hi, details=raw_jackknife_obj)
```

This rename is a **breaking surface change** → document it in the docstring and
CHANGELOG (the old name keeps resolving only if you add an alias).

### 4.2 Matrix-completion convention: 1-D path vs full matrix

For estimators that impute a full `(N, T)` matrix (RMSI, SNN, MSQRT, MCNNM):

- `res.counterfactual` / `time_series.counterfactual_outcome` is the **1-D
  treated path** (what plots and `att` consume),
- the full imputed matrix → **`counterfactual_matrix`** (typed field),
- per-cell treatment effects → **`effects_matrix`**.

Never expose the `(N, T)` matrix *as* `counterfactual`; never invent
`counterfactual_full`/`cf_matrix`/etc.

### 4.3 Pin headline numbers with `build_effect_submodels` overrides

When the standardized observed/counterfactual path differs from how the
estimator computed its own headline numbers — staggered adoption (MCNNM),
multiple treated units (MSQRT), or a decomposition reconstruction (SpSyDiD) —
`build_effect_submodels` will **recompute** `att`/`pre_rmse` from the aggregate
path and silently drift from the replication number. Pass the estimator's exact
values through so the public accessors stay byte-identical:

```python
build_effect_submodels(..., effects_overrides={"att": tau},
                        fit_overrides={"rmse_pre": pre_rmse})
```

This protects replication validity (DoD §D: "numbers are unchanged").

### 4.4 Estimators with no donor weights / no statistical inference

State-space, decomposition, and two-level estimators (TASC, SpSyDiD, MLSC) have
no per-donor weights and sometimes no CI. Conformance still requires a non-None
`weights` slot: use a `WeightsResults` with empty `donor_weights` and a method
note (record *what the weighting is* — EM smoother, spatial decomposition).
`inference=None` is acceptable when the estimator genuinely produces no CI.

### 4.5 `model_rebuild()` for forward-referenced sub-models

With `from __future__ import annotations`, a frozen pydantic result that
references a sub-dataclass **defined later in the same module** (e.g. SNN's /
MCNNM's inference container declared after the `Results` class) will fail to
resolve the annotation. Call `Results.model_rebuild()` at module end.

### 4.6 Plotting routes through `result.plot()`

Standardize on `result.plot()`. The estimator attaches a resolved `PlotConfig`
before calling it:

```python
pc = self.config.resolved_plot()          # BaseEstimatorConfig provides this
results = run_foo(inputs)
object.__setattr__(results, "plot_config", pc)   # result is frozen
if self.display_graphs:
    results.plot()
```

Gotcha: configs that are plain `BaseModel` (not `BaseEstimatorConfig`, e.g.
MLSC) have **no** `resolved_plot()` — build the `PlotConfig` by hand from the
legacy color/label fields. Keep any legacy `plot_<name>` helper functional
(repoint it to renamed fields or let it read the accessors) so its existing
coverage test stays green.

### 4.7 Reserved sub-model names collide — rename the estimator's raw field

`BaseEstimatorResults` declares `effects` / `time_series` / `weights` /
`inference` / `fit_diagnostics` / `method_details` as **fields** (holding the
standardized sub-models). An unmigrated dataclass routinely already has a field
or property with one of those names holding something else — a raw weight array
(`weights`), a per-cell effect matrix (`effects`), a rich diagnostics dict
(`fit_diagnostics`), or a bespoke inference object (`inference`). You **cannot**
keep the old meaning under the reserved name; rename the raw field and let the
reserved name be the standardized sub-model. The renames that recurred across
the FSCM/MASC/NSC/FMA/SDID/SeqSDID/SSC/BVSS batch:

| raw field (old)   | renamed to        | reserved name now holds |
| ----------------- | ----------------- | ----------------------- |
| `weights` (array) | `weights_vector`  | `WeightsResults`        |
| `effects` (matrix)| `effects_matrix`  | `EffectsResults`        |
| `fit_diagnostics` (dict) | `diagnostics` | `FitDiagnosticsResults` |
| `inference` (rich obj)   | `inference_detail` | `InferenceResults` (§4.1) |

This is a **breaking surface change** → update the estimator's tests/plotter
and docs (note the rename), and add a one-line note in the docs return-object
section. The flat accessors (`att`/`counterfactual`/`gap`/`donor_weights`/
`pre_rmse`) must keep resolving via the sub-models afterwards.

### 4.8 Event-study / multi-unit estimators: lay `time_series` over event-time

Staggered / multi-unit estimators with no single calendar-time treated path
(SequentialSDID, SSC) still satisfy the flat contract by laying the
standardized `time_series` over **event-time** rather than calendar time:
`time_periods` = the event-time horizons, `estimated_gap` = the event-study
effect curve (`tau_hat_k` / `ATT_e`), `counterfactual_outcome` = the no-effect
baseline, and `att` = the appropriate scalar summary (mean pooled effect /
overall ATT). SDID instead aggregates a treated-unit-weighted calendar path
across cohorts (a single cohort reduces to its own path). The conformance
harness only requires `counterfactual.shape == gap.shape`, `ndim == 1`, and a
populated `time_series`; an honest event-time layout satisfies it.

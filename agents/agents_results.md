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

### Forbidden

- ad-hoc `dict` or unnamed `tuple` as the public return,
- inconsistent field naming for the same quantity (`counterfactual` vs
  `counterfactual_full`; raw `weights` vector where a `donor_weights` mapping
  is meant),
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

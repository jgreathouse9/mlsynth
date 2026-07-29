## Related Links

- Source paper (DOI) and, if one exists, the authors' implementation
- `docs/<name>.rst` — the estimator page added here
- `docs/replications/<name>.rst` and `benchmarks/cases/<name>.py`
- Any paper-review or replication-spike PR that recommended building this

## What

- Added `<NAME>` implementing <Author, Year>, <Title>, <Journal Vol:pages>.
- `estimators/<name>.py` (thin), `utils/<name>_helpers/{config,setup,pipeline,structures,plotter}.py`
- `<NAME>Config`, `<NAME>Results`, export in `mlsynth/__init__.py`, entry in `docs/choose.rst`

## Why

What can a user do now that they could not before? Which regime does this cover
that the existing estimators do not — and which existing estimator would someone
otherwise have misapplied?

## How

- Ported from the authors' <language> code / built from the paper alone
- Departures from the reference, and why each is deliberate
- Anything in the method that turned out to be numerically fragile

## Contract compliance

The invariants in `CLAUDE.md`. Tick what holds; if one does not, say why here
rather than leaving it unticked and unexplained.

- [ ] Dedicated Pydantic config inheriting `BaseEstimatorConfig`, `extra="forbid"`, every field carrying a `Field(...)` description
- [ ] Validators fail early with `MlsynthConfigError` / `MlsynthDataError`; no free-form kwargs
- [ ] Ingestion goes through `datautils.dataprep` (or a `<name>_helpers/setup.py` wrapping it) — no hand-pivoted pandas in the estimator
- [ ] `fit()` returns an `EffectResult` or `DesignResult`, populating the standardized sub-models, with estimator-specific outputs as typed fields — no ad-hoc dicts
- [ ] `tests/test_result_contract.py` passes
- [ ] One estimator, one package; a dispatcher adds a method subpackage rather than a top-level estimator
- [ ] Layout follows the nearest existing estimator rather than inventing a pattern

## Verification

Which path from `docs/replications.rst`, and what agreement?

- Path: A (paper's empirical result on the authors' data) / B (paper's Monte Carlo) / cross-validation against a reference implementation
- Quantities compared, and to what tolerance
- Pinned in `benchmarks/cases/<name>.py` so a regression fails a check
- What does not match, and why it is recorded rather than closed

## Designs

- Counterfactual and gap paths, ideally overlaid on the reference implementation's
- Output of the estimator's own plotter

## Test Steps

- [ ] `pip install -e .`
- [ ] `pytest mlsynth/tests/test_<name>.py -q` — smoke, unit invariants, edge cases (no donors, single donor, no pre-periods, collinear, treatment at `t=0`), and failure tests asserting the correct translated `Mlsynth*Error`
- [ ] `pytest mlsynth/tests/` — no regressions elsewhere
- [ ] `coverage run --timid -m pytest mlsynth/tests/test_<name>.py && coverage report` — new code fully covered; every `# pragma: no cover` states its reason
- [ ] `python benchmarks/run_benchmarks.py --all`
- [ ] Docs build; section underlines at least as long as their titles

## Scope

- [ ] This branch contains only this estimator — no shared-helper refactors, no unrelated doc edits, no new benchmark cases for other estimators

## Other Notes

What is deferred, and to what?

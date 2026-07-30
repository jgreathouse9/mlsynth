## Related Links

What links will make reviewing these code changes as straightforward as possible?

- The issue this resolves, if there is one
- Docs page, replication page, benchmark case: `docs/<name>.rst`, `docs/replications/<name>.rst`, `benchmarks/cases/<name>.py`
- Source paper (DOI) and the reference implementation it was checked against
- The PR this one follows from or depends on

## What

What changes did you make at a high level?

- Added...
- Updated...
- Refactored...
- Moved...

## Why

Why are these changes helpful or necessary?

- New estimator from <Paper, Year>...
- Refactoring in preparation for...
- Correcting a result that was reported wrongly...
- Fast-follow to previous PR...

## How

How did you go about making these changes?

- Ported from the authors' <language> implementation, matching X to make Y exact
- Tried another approach but it didn't work because X, Y, Z
- I followed this resource by <Author Name>: <Resource Link>
- Used these libraries/APIs: <Name>, <Name>, <Name>

## Verification

How do we know the numbers are right? Every estimator is validated by one of the
paths in `docs/replications.rst` — the paper's empirical result on the authors'
data, the paper's simulation table, or a match against an authoritative reference
implementation. For a change that touches no numerical code, one line saying so
is the right answer.

- Path: A (paper's empirical result) / B (paper's Monte Carlo) / cross-validation / not applicable
- Quantities compared, and to what tolerance
- Where it is pinned durably, so a regression fails a check rather than going unnoticed
- Anything that does not match, and why it is recorded rather than fixed

## Scope checks

Uncomment the block that fits this change and delete the rest. If none fits —
docs, tooling, CI, a shared-helper refactor — delete all four and say so in one
line.

<!-- NEW ESTIMATOR
- [ ] Dedicated Pydantic config inheriting `BaseEstimatorConfig`, `extra="forbid"`, every field with a `Field(...)` description, validators raising `MlsynthConfigError` / `MlsynthDataError`
- [ ] Ingestion through `datautils.dataprep` — no hand-pivoted pandas in the estimator
- [ ] `fit()` returns an `EffectResult` or `DesignResult` populating the standardized sub-models; `tests/test_result_contract.py` passes
- [ ] One estimator, one package; exported in `mlsynth/__init__.py`; entry in `docs/choose.rst`
- [ ] Tests at all four levels: smoke, unit invariants, edge cases (no donors, single donor, no pre-periods, collinear, treatment at `t=0`), and failure tests asserting the correct translated error
- [ ] Branch carries only this estimator
-->

<!-- FEATURE ON AN EXISTING ESTIMATOR
- [ ] The default preserves existing behaviour exactly, including the units effects are reported in — or, if not, that is stated and justified
- [ ] Existing pinned benchmark values unchanged; any that moved are explained below
- [ ] A test pins that the default is unchanged
- [ ] New config field has a `Field(...)` description saying when to reach for it
-->

<!-- BUG FIX
- [ ] A test reproduces the defect and fails without the fix
- [ ] It asserts the correct behaviour, not merely the absence of a crash
- [ ] No docs page, docstring, or comment still states the old, wrong behaviour
- [ ] Any pinned value that moved is justified as the right number — not re-fitted to the new output
-->

<!-- BENCHMARK / VALIDATION
- [ ] Replication path stated, and the definition of done in `agents/agents_benchmarking.md` for that path and input scenario is met
- [ ] Where expected values come from, and whether the generator is committed
- [ ] Each tolerance justified: tight enough to catch a regression, loose enough to survive a different BLAS
- [ ] Determinism checked — the case gives identical values across repeat runs
- [ ] Pinned quantities are the ones that actually move, not summaries that average movement away
- [ ] Branch is benchmark authoring only
-->

## Designs

What visual context do reviewers need?

- Counterfactual and gap paths, ideally overlaid on the reference implementation's
- Placebo band or inference figure
- Output of the estimator's own plotter

## Test Steps

What are all the steps to testing your code changes?

- [ ] `pip install -e .`
- [ ] `pytest mlsynth/tests/test_<name>.py -q`
- [ ] `pytest mlsynth/tests/` — full suite, no regressions elsewhere
- [ ] `coverage run --timid -m pytest mlsynth/tests/test_<name>.py && coverage report`
      — new code fully covered, with any `# pragma: no cover` carrying a stated reason
- [ ] `python benchmarks/run_benchmarks.py --all` (or the single case)
- [ ] Docs build, and section underlines are at least as long as their titles

## Other Notes

What, if anything, hasn't been addressed in these code changes but should be in future changes?

- ABC wasn't working as expected...
- XYZ needs more research...
- A fast-follow PR is already planned for addressing 1, 2, 3...

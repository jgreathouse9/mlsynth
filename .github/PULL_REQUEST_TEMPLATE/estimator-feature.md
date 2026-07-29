## Related Links

- `docs/<name>.rst` — the page documenting the new behaviour
- `benchmarks/cases/<name>.py` if its rows move
- The paper section or reference-implementation option this mirrors
- The PR that added the estimator, if recent

## What

- Added `<option>` to `<NAME>Config` / new branch in `<NAME>.fit()` / ...

## Why

Why does the estimator need this? If it mirrors something the reference
implementation or the paper does, say which — and what could not be reproduced
without it.

## How

- Where the new behaviour sits in the pipeline, and why there
- What it deliberately does not touch

## Backward compatibility

The part most easily got wrong.

- [ ] The default preserves existing behaviour exactly — or, if it does not, that is stated here and justified
- [ ] No change to what existing configurations return, including the units effects are reported in
- [ ] Existing pinned benchmark values unchanged; if any moved, each is explained below rather than re-fitted to the new output
- [ ] New config field has a `Field(...)` description saying when to reach for it

## Verification

- What establishes the new path is correct — reference values, an analytic invariant, or an existing benchmark row
- If it changes what a benchmark measures, the before and after, and why the after is the honest number

## Designs

- Before and after, where the change is visible in a fitted path or diagnostic

## Test Steps

- [ ] `pytest mlsynth/tests/test_<name>.py -q`
- [ ] `pytest mlsynth/tests/` — full suite
- [ ] `coverage run --timid -m pytest mlsynth/tests/test_<name>.py && coverage report` — the new branch fully covered
- [ ] Tests exist for: the new behaviour's invariants, its edge and degenerate inputs, and that invalid configuration raises the correct translated error
- [ ] A test pins that the default is unchanged
- [ ] `python benchmarks/run_benchmarks.py --all` (or the single case)

## Other Notes

## Related Links

- `benchmarks/cases/<name>.py` — the case added or changed
- `benchmarks/R/<script>.R` or the generator, if a reference is needed
- `docs/replications/<name>.rst` and the benchmarks index page
- Source paper (DOI) and the reference implementation or archive the values come from

## What

- Added `benchmarks/cases/<name>.py` pinning <n> rows / added rows to an existing case
- Reference generator, gold data, docs page

## Why

What was previously unvalidated, or validated only by a throwaway script? If
this replaces a one-off measurement, say so — durability is the point.

## Replication path

- Path: A (paper's empirical result on the authors' data) / B (paper's Monte Carlo table) / cross-validation against a reference implementation
- Input scenario: paper only / code excerpt / full replication package
- The definition of done for that combination is in `agents/agents_benchmarking.md` — confirm which applies

## How

- Where the expected values come from. If generated rather than published, say what generated them and whether it is committed
- Whether gold data is vendored, and if not, how the case behaves without it (`nan` rows versus a skip)
- Anything about the source that had to be matched exactly to make the comparison valid — units, windows, donor pool, sample restrictions

## Tolerances

The row that fails on a different machine is the row someone deletes.

- [ ] Each tolerance is justified: tight enough to catch a real regression, loose enough to survive a different BLAS or platform
- [ ] Determinism checked — the case gives the same values across repeat runs
- [ ] Any bounded coverage (top-N, sampling, no-retry) is logged rather than left to look like full coverage

Rows and why each tolerance is what it is:

## Verification

- [ ] Every row passes: `python benchmarks/run_benchmarks.py --all` (or the single case)
- [ ] Quantities pinned are the ones that actually move, not just the summary statistics that average movement away
- [ ] Anything that does not match is recorded with its cause, not omitted

## Designs

- Reference against mlsynth, where a plot makes the agreement or the gap legible

## Test Steps

- [ ] `python benchmarks/run_benchmarks.py --all`
- [ ] Re-run the case a second time and confirm identical values
- [ ] With reference/gold data absent, the case still imports and reports rather than erroring
- [ ] Docs build; the benchmarks index lists the new case

## Scope

- [ ] This branch is benchmark authoring only — no estimator changes, no refactors bundled in

## Other Notes

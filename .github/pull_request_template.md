<!--
Scope-specific templates live in .github/PULL_REQUEST_TEMPLATE/ . GitHub does not
offer a picker for pull request templates the way it does for issues, so pick one
by appending ?template=<file> to the compare URL, replacing YOUR-BRANCH:

  new estimator          ...compare/main...YOUR-BRANCH?quick_pull=1&template=new-estimator.md
  feature on an existing ...compare/main...YOUR-BRANCH?quick_pull=1&template=estimator-feature.md
  bug fix                ...compare/main...YOUR-BRANCH?quick_pull=1&template=bugfix.md
  benchmark / validation ...compare/main...YOUR-BRANCH?quick_pull=1&template=benchmark.md

Or open the PR, then copy the template you want over this body.

This default is the right one for anything else: docs-wide edits, refactors of
shared helpers, tooling, CI. Delete this comment before submitting.
-->

## Related Links

What links will make reviewing these code changes as straightforward as possible?

- Docs page for the estimator: `docs/<name>.rst`
- Replication page and benchmark case: `docs/replications/<name>.rst`, `benchmarks/cases/<name>.py`
- Source paper (DOI or link) and the reference implementation it was checked against
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
data, the paper's simulation table, or a match against an authoritative
reference implementation. State which, and what agreement was reached.

- Path: A (paper's empirical result) / B (paper's Monte Carlo) / cross-validation
- Quantities compared, and to what tolerance
- Where it is pinned durably, so a regression fails a check rather than going unnoticed
- Anything that does not match, and why it is recorded rather than fixed

## Designs

What visual context do reviewers need?

- Counterfactual and gap paths against the reference implementation
- Placebo band or inference figure
- Plot produced by the estimator's own plotter

## Test Steps

What are all the steps to testing your code changes?

- [ ] `pip install -e .`
- [ ] `pytest mlsynth/tests/test_<name>.py -q`
- [ ] `pytest mlsynth/tests/` — full suite, no regressions elsewhere
- [ ] `coverage run --timid -m pytest mlsynth/tests/test_<name>.py && coverage report`
      — new code fully covered, with any `# pragma: no cover` carrying a stated reason
- [ ] `python benchmarks/run_benchmarks.py --all` (or the single case) — validation still passes
- [ ] Docs build, and section underlines are at least as long as their titles

## Other Notes

What, if anything, hasn't been addressed in these code changes but should be in future changes?

- ABC wasn't working as expected...
- XYZ needs more research...
- A fast-follow PR is already planned for addressing 1, 2, 3...

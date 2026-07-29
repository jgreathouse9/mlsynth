## Related Links

- The PR or commit that introduced the defect, if it can be identified
- The docs page or benchmark case whose stated numbers were affected
- Reference implementation output, if that is what revealed it

## The defect

Be concrete. A reviewer needs to judge whether the fix addresses the cause or
the symptom.

- What was wrong:
- How it manifested — wrong number, wrong error, silent wrong result, crash:
- Cause:
- Since when, and what else it reached:

## What

- Fixed...

## Why

What breaks if this stays unfixed, and for whom? If it produced a wrong number
rather than an exception, say what a user would have concluded from it.

## How

- Why the fix is at this seam and not further up or down
- Anything tried first that did not work, and why

## Regression test

- [ ] A test reproduces the defect and fails without the fix
- [ ] It asserts the correct behaviour, not merely the absence of a crash
- [ ] If the bug was a swallowed error, a test asserts the failure is now reported

Which test, and what it pins:

## Claims to correct

Wrong numbers often reach the docs before they are caught.

- [ ] No docs page, docstring, or benchmark comment states the old, wrong behaviour
- [ ] If a published or documented figure changes, it is corrected in this PR and the earlier claim recorded rather than quietly replaced

## Verification

- [ ] Pinned benchmark values re-checked. Any that moved are listed below with why the new value is right and the old one wrong — not adjusted to make the suite pass

## Test Steps

- [ ] `pytest mlsynth/tests/test_<name>.py -q` — the new regression test fails on `main`, passes here
- [ ] `pytest mlsynth/tests/` — full suite
- [ ] `python benchmarks/run_benchmarks.py --all` (or the affected case)

## Other Notes

Anything adjacent that looks similarly fragile but is out of scope here.

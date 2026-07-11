# Reference corpus

This directory holds the reference material behind the cross-validation
benchmarks. Two patterns coexist:

- Reference *code* — `clone_*.py` import an authors' implementation (cloned at a
  pinned commit, or an installable package) so a case can run it live.
- Captured reference *bundles* — `<case>/` directories that record the exact
  reference run and its output, so every pinned number in a case's `EXPECTED`
  traces to an inspectable, regenerable artifact rather than a bare constant.

## Anatomy of a captured bundle

```
benchmarks/reference/<case>/
  manifest.json     # hand-written: case metadata + input-data dependencies
  reference.R       # hand-written: the exact reference script (or .py/.sh)
  reference.out     # generated: verbatim stdout of the reference run
  reference.json    # generated: parsed {values, weights} the case pins against
  provenance.json   # generated: tool versions, OS, git SHA, data sha256, timestamp
```

The reference script prints its numbers in a stable, parseable block:

```
== REFERENCE VALUES ==
<key>\t<value>
weight\t<label>\t<value>
== SESSION INFO ==
<sessionInfo()/version dump for provenance>
```

## How a case pins to a bundle

A case reads its reference number from the bundle, so the constant and the
captured run cannot drift:

```python
from benchmarks.reference import reference_value

EXPECTED = {
    "synth_pre_ssr": (reference_value("synth_prop99", "synth_pre_ssr"), 0.1),
}
```

`benchmarks/cases/synth_prop99.py` is the worked exemplar.

## Side-by-side comparison tables

A case may also define `comparison() -> list[dict]` returning rows
`{quantity, mlsynth, reference}` that pair the library's own number against the
captured reference value, quantity by quantity (donor weights, the fit loss, the
ATT, ...).

Compare uncertainty, not just point estimates. When both the estimator and the
reference report the *same* inferential quantity — standard errors, confidence
or prediction interval bounds, a p-value — pair those too, so the dashboard
proves the uncertainty is reproduced, not only the point estimate (e.g.
`spsc_prop99` pairs the sandwich-GMM SEs and the conformal prediction-interval
bounds; `spsc_panic` the ATT and its SE). Only compare a statistic both sides
compute the *same way*: where the two implementations use different variance
estimators (e.g. a resampling placebo vs an analytic placebo), a raw SE diff
would mislabel a method difference as an error — record it as context or skip it.

Export them to committed, inspectable tables:

```bash
python benchmarks/reference/export_comparison.py synth_prop99   # one case
python benchmarks/reference/export_comparison.py --all          # every case
```

Both `comparison()`'s `mlsynth_call` (estimator + config) and the export carry
provenance: each `comparison.csv` opens with a metadata header — when it was
written, the mlsynth version, the exact call, and the reference implementation
and version.

This writes `benchmarks/reference/<case>/comparison.csv` (GitHub renders it as a
table) reading mlsynth next to the reference, with the absolute difference. The
public, web-native rollup a reviewer opens is the Validation dashboard
(`docs/validation.rst`), generated from these CSVs by `build_validation.py`:

| quantity | mlsynth | reference | abs_diff |
|----------|--------:|----------:|---------:|
| weight[Utah] | 0.393908 | 0.396212 | 0.002304 |
| pre_period_SSR | 52.129571 | 52.136289 | 0.006718 |
| ATT | -19.51363 | -19.493546 | 0.020084 |

## Generating / refreshing

```bash
python benchmarks/reference/generate.py synth_prop99   # one case
python benchmarks/reference/generate.py --all          # every bundle
```

The generator runs `manifest.json`'s `command`, captures stdout to
`reference.out`, parses `reference.json`, and writes `provenance.json` (versions,
git SHA, and a sha256 of every input data file). It skips cleanly when the
reference toolchain is absent. `mlsynth/tests/test_benchmark_reference.py` checks
that the committed artifacts stay self-consistent and that the recorded data
checksums still match the shipped data — a changed input surfaces as a stale
reference rather than a silent drift.

## The validation report

```bash
python benchmarks/run_benchmarks.py --all --report
```

writes `benchmarks/REPORT.md` (a human-readable dashboard) and
`benchmarks/report.json` (machine-readable), each row tying a case to its
status, reference implementation, and bundle.

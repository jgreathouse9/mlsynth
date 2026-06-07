# mlsynth benchmarks

Durable, re-runnable validation of mlsynth estimators against the source
paper's own numbers and against authoritative **reference implementations**
(usually R). This institutionalizes the cross-checks that otherwise get done
once by hand and thrown away.

Each estimator's replication follows one of three "paths" (same vocabulary as
`docs/replications.rst`):

* **Path A** — reproduce the paper's empirical result on the authors' data.
* **Path B** — reproduce the paper's Monte Carlo / simulation table.
* **Cross-validation** — match an authoritative reference implementation
  (e.g. R `Synth`, `synthdid`, `did`) cell by cell.

The **definitions of done** for each path — broken out by what the source gives
you (paper only / code excerpt / full repo) — live in
[`agents/agents_benchmarking.md`](../agents/agents_benchmarking.md). This README
is the mechanics ("what to run"); that doc is the process ("when is it done").

## Layout

    benchmarks/
      run_benchmarks.py     # driver: runs registered cases, prints pass/fail vs tolerance
      registry.py           # the list of benchmark cases (name -> callable -> expected)
      compare.py            # tolerance-based comparison + reporting
      cases/                # one module per benchmark (pure-Python where possible)
        fdid_table5.py      # Path B: Li (2024) Table 5 PMSE grid (no R needed)
        fdid_hongkong.py    # Path A: Li (2024) Hong Kong GDP empirical (no R needed)
      R/                    # reference-implementation cross-checks
        requirements.R      # install the reference R packages
        synth_crosscheck.R  # run R's Synth on a dumped panel for cell-by-cell comparison

## Quick start

```bash
# Pure-Python benchmarks (no R required)
python benchmarks/run_benchmarks.py --all

# A single case
python benchmarks/run_benchmarks.py --case fdid_table5

# Reference cross-checks (need R + packages)
Rscript benchmarks/R/requirements.R          # one-time
python benchmarks/run_benchmarks.py --with-reference
```

A case **passes** when every reported number is within its declared tolerance
of the expected (paper / reference) value. Tolerances are intentionally loose
enough to absorb Monte-Carlo noise (smaller M than the paper) but tight enough
to catch real regressions.

## Adding a benchmark

1. Add `benchmarks/cases/<name>.py` exposing `run() -> dict[str, float]` and
   `EXPECTED: dict[str, tuple[float, float]]` (value, abs-tolerance).
2. Register it in `registry.py`.
3. If it needs a reference run, drop the R script under `R/` and have the case
   read the reference output (or skip gracefully when R is absent).

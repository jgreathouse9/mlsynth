# Pre-build spikes

A spike is the evidence behind a build / park / pass decision on a candidate
paper, produced by `/paper-review` and `/replicate` before any estimator exists.

This is not `benchmarks/cases/`. A case validates an mlsynth estimator against
its source paper or a reference implementation, is registered in
`registry.py`, and runs in `run_benchmarks.py`. A spike validates nothing in
mlsynth — the method is not in the library and may never be. Spikes are not
registered and are not run by the driver.

Each spike directory holds the port, one re-runnable script, and a `REPORT.md`
with the numbers and the recommendation. A spike whose verdict turns into a
build gets deleted, and the durable validation moves to `benchmarks/cases/`.

| Spike | Paper | Verdict |
| --- | --- | --- |
| `degeest_wang_fwdsel/` | De Geest and Wang (2025), "Designing Synthetic Control Experiments with Forward Selection" (CODE '25) | pass as an estimator; harvest the search as a backend |

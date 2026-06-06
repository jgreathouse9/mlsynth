---
description: Demonstrate-first replication of a paper's method before committing to a build
argument-hint: "<paper> [--data <path>] [--reference <repo|R-pkg>]"
---

# Replicate (demonstrate-first)

Reproduce a method's result on real/standard data *before* wiring it into
mlsynth, so a build is only ever started on something validated.

## Loop

1. **Ingest via `dataprep`** — never hand-pivot. Use a canonical dataset
   (`basedata/`) or the paper's data.
2. **Port the algorithm faithfully**, citing the reference (paper section /
   reference-repo function). Keep a readable version as the oracle.
3. **Validate against ground truth**:
   - reference implementation (R/other) — run it, compare cell by cell;
   - or the paper's reported numbers (Path A/B).
   Quantify the match ("ρ to 3 digits"; "PMSE 0.084 vs 0.082").
4. **Surface the honest finding** — including when the paper's advantage is a
   weak-baseline artifact or its key quantity is weakly identified.
5. **Decide**: build (→ `/new-estimator`), park, or pass — with the evidence.

## Output
A short report: what matched, to what tolerance, the caveats, and the
build/park/pass recommendation. Stage durable validation under
`benchmarks/` rather than leaving it in scratch.

# MASC reference (parked — not yet a live bundle)

These are the vendored R sources for Kellogg, Mogstad, Pouliot & Torgovitsky's
MASC estimator (matching + synthetic control), fetched for a planned live
conversion of the `masc_basque` benchmark. They are parked here for a future
session: the case currently remains a Path-A paper-pinned benchmark.

Files
- `masc_estimator.R`, `masc_crossvalidation.R` — the authors' MASC estimator and
  its leave-one-out cross-validation of the match/SC mixing parameter.
- `masc_AG_replication.R` — the authors' Abadie-Gardeazabal (Basque) replication
  driver.
- `LICENSE.masc` — the upstream license.

Why parked
The MASC synthetic-control step is written against the Gurobi commercial solver
(`nogurobi = FALSE`, with `BarConvTol`/`BarIterLimit` Gurobi parameters); a
`nogurobi` QP fallback exists but the full leave-one-out CV is memory-heavy and
exceeded this environment's limits on the first attempt. Finishing the
conversion needs the `nogurobi` path with a bounded CV (or Gurobi), then the
usual bundle: a `reference.R` emitting the `== REFERENCE VALUES ==` block, a
`manifest.json`, and `generate.py masc_basque`, with `masc_basque.py`'s
`comparison()` / `EXPECTED` rewired to `reference_value`.

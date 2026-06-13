# Vendored reference: Synthetic Interventions (Agarwal, Shah & Shen 2026)

`estimation.py` and `inference.py` are **verbatim** from the authors' replication
package for

> Agarwal, Shah & Shen (2026). "Synthetic Interventions: Extending Synthetic
> Controls to Multiple Treatments." *Operations Research* 74(2):840-859.
> INFORMS supplemental code `opre.2025.1590.cd` (`synth-iv-OR25/`).

They are imported, not modified, so the benchmark cross-validates mlsynth's public
`SI` API against the authors' own code (the SI-PCR + bias-corrected estimator:
`HSVT` -> `qr_column_pivoting_selection` -> `OLS`, with `variance_estimation` +
`predictionInterval`). Only the functions exercised by the Section 6 case study
are used; the simulation/DGP helpers are omitted.

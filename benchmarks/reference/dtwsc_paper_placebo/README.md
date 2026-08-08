# Reference inputs for the DTWSC paper-placebo replication

Cao, J. & Chadefaux, T. (2025). *Dynamic Synthetic Controls: Accounting for
Varying Speeds in Comparative Case Studies*. Political Analysis 33:18-31.
[10.1017/pan.2024.14](https://doi.org/10.1017/pan.2024.14)

These two files are what the authors' own placebo design needs as input. They
support a Path A replication: matching the numbers printed in the paper, rather
than cross-validating against a reference implementation.

## `gold_gridopt.csv` (8453 rows)

The hyperparameters the authors used for every individual placebo run, on all
three panels — 1789 runs for Basque, 5008 for California, 1656 for German
reunification.

The authors do not fix a warping configuration. Each run gets its own
`filter.width`, `k` and DTW step pattern, chosen by a grid search over
9 x 6 x 7 = 378 combinations. The selection rule is not published, but it does
not need to be: their replication archive ships the resulting choices, which is
what this file records.

Provenance: `Figure_5_{1,2,3}_gridOpt.Rds` in the archive, which store the
choice as an integer `grid.id` into an `expand.grid`. The generator decodes that
index into explicit `filter.width` / `k` / `step.pattern` columns so the table
can be read without reconstructing R's column-major ordering.

Six of the seven step patterns appear among the selected optima —
`symmetricP1`, `symmetricP2`, `asymmetricP1`, `asymmetricP2`, `typeIc` and
`mori2006`. `typeId` is in the sweep but never chosen on any panel.

## `gold_basque_panel.csv` (18 units, 774 rows)

`Synth::basque` as the authors load it, plus the `invest_ratio` predictor they
derive from it.

This is committed rather than read from `basedata/basque_data.csv` because the
two are not the same panel. Our copy has Spain removed and the unit ids
renumbered, so its `id 1` is Andalucia where the authors' `id 1` is Spain. Their
design uses Spain both as a donor and as a placebo target, so the two are not
interchangeable and substituting ours would silently change the design.

## Regenerating

`gen_paper_placebo_gold.R` produces both. The Basque panel needs only R with
`Synth` and `dplyr`. The grid table additionally needs the authors' replication
archive, which is not vendored here — download it from Dataverse
([10.7910/DVN/DIUPUA](https://doi.org/10.7910/DVN/DIUPUA)) and pass the path to
its `data/` directory:

```sh
Rscript benchmarks/reference/dtwsc_paper_placebo/gen_paper_placebo_gold.R \
    /path/to/dataverse_files/data
```

Without that argument the script writes the panel and skips the grid table.

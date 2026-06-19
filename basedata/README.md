# `basedata/` — bundled datasets

Reference datasets for mlsynth's paper replications, benchmark suite, and doc
examples. Each estimator is validated against a published result, and several of
those papers use the *same underlying study* (Proposition 99, German
reunification, the Basque Country) processed differently. So a few files here
look like duplicates — they are **deliberately** kept distinct because each is
the canonical input a specific replication matches against, and the "subsets"
differ in their covariate processing (column names, scales, unit pool). They are
not interchangeable; consolidating them would break the replication contract.

This manifest maps each family to its files and primary consumers so the
redundancy is navigable rather than confusing.

Note on packaging: these files are **not shipped in the PyPI wheel** (they live
at the repo root, not inside the `mlsynth` package). Load them from a checkout,
or via the raw GitHub URL the doc galleries use
(`https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/...`).
The larger tables are stored as Parquet (needs `pyarrow`); read with
`pd.read_parquet`.

## Proposition 99 (California tobacco control) — Abadie, Diamond & Hainmueller (2010)

The same 39-state × 31-year (1970–2000) cigarette-sales panel, in several column
slices, plus a larger raw-state pool:

| File | What it is | Used by |
|---|---|---|
| `augmented_cali_long.csv` | the ADH MLAB superset (343 cols: outcomes + all predictors) | VanillaSC / SparseSC / pensynth / CWZ Prop 99 replications |
| `P99data.csv` | 7-col slice: `cigsale` + `lnincome`/`beer`/`age15to24`/`retprice` | FSCM / RMSI / SpotSynth Prop 99 docs |
| `smoking_data.csv` | 4-col slice: `cigsale` + the `Proposition 99` treatment flag | many estimators' Prop 99 smoke/benchmark cases |
| `california_panel.csv` | `cigsale` + `retprice` + `state_id` | SpillSynth doc example |
| `california_W_matrix.csv`, `california_w_vector.csv` | adjacency / weight vectors | SpillSynth spatial example |
| `prop99_packsales.csv`, `prop99_with_dc.csv` | a *larger* state pool, `cigsale` only (`with_dc` adds DC) | SI / SpillSynth / TASC cases |

## German reunification — Abadie, Diamond & Hainmueller (2015)

The same 17-country × 44-year (1960–2003) GDP panel, in three covariate depths:

| File | What it is | Used by |
|---|---|---|
| `germany_augmented.csv` | the superset (106 cols) | SCMO multi-outcome replication |
| `repgermany.dta` | the standard ADH covariates (`gdp`, `infrate`, `trade`, `schooling`, `invest*`) | SpillSynth / IncSCM / west-Germany cases |
| `german_reunification.csv` | `gdp` + the `Reunification` flag | ClusterSC / SpotSynth / several west-Germany cases |

## Basque Country — Abadie & Gardeazabal (2003)

Two near-identical 17-column regional panels that differ by a region/year block:

| File | What it is | Used by |
|---|---|---|
| `basque_jasa.csv` | the JASA replication panel (774 rows) | MASC / CWZ Basque replications |
| `basque_data.csv` | the variant used by the spatial/FDID examples (731 rows) | FDID / SpotSynth Basque cases |

## Carbon tax — sample vs full

| File | What it is | Used by |
|---|---|---|
| `carbontax_data.dta` | the analysis sample (per-capita CO2 / GDP / fuel) | CWZ Monte Carlo / t-test, VanillaSC t-test |
| `carbontax_fullsample_data.dta.txt` | the larger full sample | ORTHSC carbon-tax replication |

## Other datasets

The remaining files back a single estimator's replication each — e.g.
`HongKong.csv` / `hong_kong_handover.csv` (HSC / handover), `HubeiSCM`-style PPI
(`china_ppi_long.csv`), `dube_minwage.parquet` (Distributional SC),
`brexit_long.parquet` (PDA Brexit), `state_unemployment.parquet` (SpSyDiD),
`seattledmi.parquet` (MicroSynth), `kansas_*` (ASCM), `markets/` (the DMA
contiguity map + metadata for the SYNDES / GEOLIFT / MAREX / LEXSCM geographic
designs). See each estimator's `docs/replications/<name>.rst` and the
`benchmarks/cases/<name>.py` that consumes it.

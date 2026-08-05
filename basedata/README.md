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
| `prop99_mediation.csv` | 51-state balanced panel (all 50 states + DC), `cigsale` + `price` (tax-inclusive average cost per pack), from the CDC / Orzechowski-Walker Tax Burden on Tobacco file; carries the seven high-tax states the curated pool drops, so the mediator-matched (cross-world) pool can span California's post-treatment price | MEDSC Prop 99 mediation replication |

## Virginia HPV vaccine mandate — Feldman & Semprini (2026)

| File | What it is | Used by |
|---|---|---|
| `hpv_cervical_ddd.csv` | 39-state × 17-year (2003–2019) panel of age-adjusted cervical-cancer incidence (`cervix_adj`) by 5-year `age` band (20–24 + 30–49), from public NPCR/SEER via the authors' repo `jsemprini/Virginia_HPVmandate_causal`; the `age` dimension is the subgroup for the synthetic triple difference (20–24 exposed, older bands control) | SDID synthetic-triple-difference (SC-DDD) replication |

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

## MAREX go-dark experiment (simulated) — Abadie & Zhao (2026)

| File | What it is | Used by |
|---|---|---|
| `apple_godark_dmas.csv` | one draw of the Abadie–Zhao baseline DGP (Sec. 5) framed as an Apple paid-media go-dark test: 20 DMAs × 50 weeks of `sales` (paid media on everywhere through week 39, off in the designed `went_dark` markets for weeks 40–49), the seven observed DGP covariates as per-DMA market traits (`median_income`, `population`, `iphone_share`, `retail_density`, `median_age`, `broadband_pct`, `ad_spend_index`), and the simulation's known per-week effect `tau_true` | the JOSS paper's MAREX experimental-design illustration |

Regenerate with `python tools/gen_apple_godark.py` (it runs the MAREX selection
solve once, offline, and pre-commits the chosen dark markets so the paper can
load the panel and recover the same design in a single `fit`). The selection
matches on the pre-launch sales and the seven covariates with `standardize=True`.

## Walmart Supercenter entry and county employment — Wiltshire (2021, 2025)

| File | What it is | Used by |
|---|---|---|
| `allsynth_walmart.parquet` | the `allsynth_walmart` panel shipped with Wiltshire's `allsynth` Stata package (`sysuse allsynth_walmart`), trimmed to the columns the staggered design needs: 605 counties x 1990–2005, aggregate (`emps_n10`) and retail (`emps_n44`) county employment, the ever-treated flag `supercenter` and entry year `super_year` (566 counties adopting 1995–2000 in six cohorts; 39 never-treated donors where entry was blocked), the commuting zone `czone` and 1990 population `pop90` used by the article's donor restriction and averaging weights | `examples/rolldid_allsynth_walmart.py` (ROLLDID on the article's Example 12) |

## Pennsylvania electricity generation mix — Boussim (2026)

| File | What it is | Used by |
|---|---|---|
| `pa_aeps_generation.csv` | annual net generation in MWh by state and category, 1990–2023, for Pennsylvania plus the 42 donors surviving the paper's screens: three categories (`gas` = EIA Natural Gas + Other Gases; `fossil` = Coal + Petroleum; `renewables` = conventional hydro, wind, solar, geothermal, wood, other biomass, other, pumped storage), nuclear excluded | COMPSC Pennsylvania AEPS replication |

Built from the EIA state historical table `annual_generation_state.xls` (Total
Electric Power Industry). Raw megawatt hours are stored rather than shares, so
the file can be audited directly against the EIA source; COMPSC closes each row
to the simplex on ingestion. The category assignment is not stated outright in
the paper — it was recovered by matching its Table 1 balance row; see
`docs/replications/compsc.rst`.

## Okano & Kurisu (2026) functional synthetic control

| File | Contents | Used by |
|---|---|---|
| `okano_fsc_fertility.csv` | age-specific fertility rates by country, year (1956–1975) and age (12–55): East Germany plus 20 controls, treatment 1972 | `fsc_okano` section 6.1 |
| `okano_fsc_mortality.csv` | age-at-death quantile functions by country, year (1970–1999) and quantile level (100 points on [0.01, 0.99]): Russia plus 17 Western European controls, treatment 1991 | `fsc_okano` section 6.2 |
| `okano_fsc_service.csv` | service-trade covariance matrices by country and quarter (2009Q1–2018Q2), stored as the 45 lower-triangle entries of the 9×9 matrix over service categories SC–SL: the UK plus 22 controls, treatment 2016Q2 | `fsc_okano` section 6.3 |

Converted from the authors' `asfr.RData` / `aad.RData` / `service.RData`
(<https://github.com/RyoOkano21/FSC>) with the `rdata` package; no R is needed to
rebuild them. One row per (unit, time, argument), so each argument slice pivots
through `dataprep` independently and the slices stack into the `(N, T, M)` cube
the method wants. Underlying sources are the Human Fertility Database, the Human
Mortality Database, and UN Trade and Development. Note that the service file
stores the plain half-vectorisation the authors use, with no √2 on the
off-diagonals — see `benchmarks/cases/fsc_okano.py` for why that matters.

## Election Day Registration and voter turnout — Xu (2017)

| File | Contents | Used by |
|---|---|---|
| `xu_edr_turnout.parquet` | state-level turnout in US presidential elections, 1920–2012 (24 quadrennial periods × 47 states), with the `policy_edr` treatment and the `policy_mail_in` / `policy_motor` covariates | GSYNTH Table 2 replication and benchmark |

The panel behind Xu (2017) Table 2, taken from `turnout.rda` in
<https://github.com/xuyiqing/fect> and written to Parquet unchanged — no R is
needed to rebuild it. Nine states adopt EDR (three in 1976, three in 1996, two in
2008, one in 2012) and thirty-eight never do, which is the 9 / 38 / 1,128 split
the table's header row reports. Adoption is absorbing, so the never-treated
thirty-eight are the donor pool the estimator's factor space comes from.

## Age verification laws and search behavior — Lang et al. (2026)

| File | Contents | Used by |
|---|---|---|
| `lang_av_laws.parquet` | weekly Google Trends search interest by state, 2022-01-01 to 2024-10-31 (149 weeks × 46 states), for four search terms tagged by `outcome`: `pornhub`, `xvideos`, `vpn`, `porn`, with the `post_treat` adoption indicator | GSYNTH age-verification cross-validation benchmark |

Sliced from `data/{pornhub,xvideos,vpn,porn}.csv` in
<https://github.com/davidnathanlang/internet_regulation_synth_project> at commit
`38ab54b`, restricted to the paper's analysis window (`time == "2022-01-01
2024-10-31"`) and dropping the five states the authors' own
`03_preregistered_hypotheses.R` drops (ND, MO, AZ, OH, GA). Each of the four
frames is 46 × 149 and balanced; 14 states adopt across staggered dates and 32
never do, and adoption is absorbing.

The slice is byte-identical across the two committed vintages of the upstream
data that contain this window, so it does not depend on which one is checked
out. `benchmarks/reference/gsynth_av_laws/reference.R` reads this same file
through `nanoparquet`, so the R and Python sides of the comparison cannot run on
different inputs.

## EU emissions trading system and air pollution — Basaglia, Grunau & Drupp (2024)

| File | Contents | Used by |
|---|---|---|
| `euets_cobenefits.parquet` | annual `log(emissions)` of three air pollutants (SO2, PM2.5, NOx, tagged by `pollutant`) for EU-25 countries split into ETS-regulated and unregulated sectors, 1990–2021, with the `treat_post` indicator (regulated sectors from 2005) and the `log_gdp` / `log_gdp_2` controls | SDID EU ETS co-benefits replication and benchmark |

Concatenated from `Stata_SDID/data_in/{so2,pm25,nox}_gscm_data.csv` in
<https://github.com/ccs282/EU_ETS_Co_Benefits> and written to Parquet unchanged
— no Stata or R is needed to rebuild it. Each pollutant is 50 units (25 countries
× regulated/unregulated) by 32 years.

The frame is unbalanced as shipped: Estonia, Latvia, Lithuania and Slovenia enter
in 1995, Slovakia in 1992, Hungary in 1991, and the United Kingdom leaves after
2019. The gaps are all leading or trailing, so nothing can be interpolated. The
authors' generalized synthetic control runs on the panel as it stands; their SDID
do-file drops the six late-entering countries and caps at 2019 to reach a
balanced 38 × 30 sample, which is what `benchmarks/cases/sdid_euets.py`
reconstructs.

## Tokyo 2020 Olympics and COVID-19 — Yoneoka et al. (2022)

| File | Contents | Used by |
|---|---|---|
| `yoneoka_olympics_covid.parquet` | daily COVID-19 confirmed cases per million (7-day moving average) by country and a 50-day integer time index (`date2` 25–74, ending 2021-08-13): Japan plus the 42 donor countries the paper lists, with the 30 predictor columns the authors' specification uses | VanillaSC Tokyo Olympics replication and benchmark |

Sliced from `Synthetic_Olympic/data/df.csv` in
<https://github.com/kingqwert/R> at commit `bde42e2`, restricted to the paper's
analysis window (`date2 < 75`) and written to Parquet unchanged — no R is needed
to rebuild it. The frame is 43 × 50 and balanced on the outcome; three predictor
columns carry a few missing cells, which the authors' `mean(..., na.rm = TRUE)`
predictors absorb.

Japan is treated from the opening ceremony. The authors' `tidysynth` call sets
`i_time = 53`, which in that package is the last pre-treatment period, so the
treatment indicator built from this file is `date2 > 53`.

`benchmarks/reference/vanillasc_olympics/reference.R` reads this same file
through `nanoparquet`, so the R and Python sides of the comparison cannot run on
different inputs. The authors' own committed outputs — weights, balance table
and placebo p-values at all three intervention timings — are vendored beside it
under `authors/`.

## Other datasets

The remaining files back a single estimator's replication each — e.g.
`HongKong.csv` / `hong_kong_handover.csv` (HSC / handover), `HubeiSCM`-style PPI
(`china_ppi_long.csv`), `dube_minwage.parquet` (Distributional SC),
`brexit_long.parquet` (PDA Brexit), `state_unemployment.parquet` (SpSyDiD),
`seattledmi.parquet` (MicroSynth), `kansas_*` (ASCM), `markets/` (the DMA
contiguity map + metadata for the SYNDES / MAREX / LEXSCM geographic
designs). See each estimator's `docs/replications/<name>.rst` and the
`benchmarks/cases/<name>.py` that consumes it.

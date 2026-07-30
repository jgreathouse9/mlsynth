# song_ml_ascm_china.parquet

Weather-normalized ("deweathered") and raw air-pollutant concentrations for 197
Chinese city groups, weekly, 2014-01-05 to 2021-12-19 — the input panel for the
ML-ASCM analysis of China's clean winter heating policies.

## Provenance

Song, C., Liu, B., Cheng, K., Cole, M. A., Dai, Q., Elliott, R. J. R. & Shi, Z.
(2023). *Attribution of Air Quality Benefits to Clean Winter Heating Policies in
China: Combining Machine Learning with Causal Inference.* Environmental Science &
Technology 57(46):17707–17717. <https://doi.org/10.1021/acs.est.2c06800>
(open access, CC-BY 4.0).

Taken from `RegionAverageAll.xlsx` in the authors' repository named in that
paper's Data Availability Statement, <https://github.com/songnku/ML-ASCM>.
Converted from `.xlsx` to zstd parquet without modification: every float column
round-trips bit-exactly and no value is rounded, reordered or renamed.

Note on licence: the article is CC-BY 4.0 and this repository is the reuse
location its Data Availability Statement designates, but the GitHub repository
itself carries no explicit licence file. Vendored here with attribution on that
basis.

## Why the whole panel is vendored

The authors' four analysis scripts between them reference 114 of the 197 unit
identifiers, and a fifth (`citylevel_main_result.R`, the city-level arm) could not
be retrieved at all. Subsetting would have meant guessing which arms matter and
would leave a future one unable to run. The full panel is 9 MB, which is
proportionate, and the only fetch route that works from a sandbox is
`raw.githubusercontent.com` — so refetching later is fragile enough to be worth
avoiding.

## Shape

- 81,952 rows x 18 columns
- `date` — weekly, 416 distinct values
- `ID` — 197 values, mixing individual cities with pre-aggregated groups. Note
  that `"2+26 cities"`, `"Northern"`, `"China"` and similar are themselves rows:
  several of the paper's arms treat a population-weighted regional average as the
  treated unit. A reader who assumes every `ID` is a city will misread the
  estimand.
- eight pollutants, each in two forms: raw (`PM2.5`, `SO2`, `NO2`, `CO`, `O3`,
  `O3_8h`, `Ox`, `PM10`) and weather-normalized with a `wn` suffix
  (`PM2.5wn`, ...). The paper's causal analysis uses the `wn` series; the raw
  series are what a synthetic control on unadjusted concentrations would see, and
  the paper shows the two give materially different conclusions.

The weather normalization is a random-forest step (`rmweather`) applied upstream
by the authors. It is not reimplemented in mlsynth and does not need to be — the
deweathered series ship here, so the synthetic-control half is reproducible on its
own.

## Used by

- `benchmarks/cases/song_ml_ascm.py`
- `docs/replications/song_ml_ascm.rst`

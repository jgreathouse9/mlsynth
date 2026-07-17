# `fdi_oecd_brexit.csv`

The OECD foreign-direct-investment panel behind the Brexit application in
Wang (2024), *"Counterfactual and Synthetic Control Method: Causal Inference
with Instrumented Principal Component Analysis"* — the empirical study the
`CSCIPCA` estimator replicates (Path A).

- 30 units (the UK plus 29 OECD controls), 1995–2022 (840 rows).
- Treated unit: United Kingdom (`country_code == "GBR"`), from 2017 onward
  (post-Brexit-referendum), encoded in `treated`.
- Columns:
  - `country`, `country_code` (ISO3), `year`
  - `fdi` — foreign-direct-investment net inflows as a share of GDP (%); the
    outcome
  - `treated` — treatment indicator (UK from 2017)
  - nine covariates that instrument the factor loadings: `log_gdp`,
    `log_gdp_percap`, `import_to_gdp`, `export_to_gdp`,
    `inflation_gdp_deflator`, `gross_capital_forma_gdp`, `unemployment`,
    `employment_15`, `log_population`.

This is a validated benchmark dataset: `CSCIPCA` reproduces the paper's
reported ATT path on it (2017 −7.8%, 2018 −12.9%, 2019 −18.3%). See
`docs/replications/cscipca.rst` and `benchmarks/cases/cscipca_brexit.py`.

## Provenance

Sourced from the author's replication package
(<https://github.com/CongWang141/JMP>, `data/country_fdi.csv`), a World
Development Indicators (WDI) export, and processed exactly as the paper's
`test7_empirical_study` notebook does:

- restrict to the 38 OECD member states (ISO3 list in the notebook);
- convert `..` to missing and cast to float; take logs of GDP, GDP per capita,
  and population;
- drop any country with a missing FDI observation;
- drop "extreme" countries whose FDI-to-GDP ratio exceeds ±25% in any year
  (the notebook's outlier screen — this removes financial-hub outliers such as
  Luxembourg, Ireland, and the Netherlands whose FDI series are an order of
  magnitude larger).

The remaining 30 countries have complete covariate coverage over 1995–2022.
The raw `country_fdi.csv` (a 1.2 MB WDI export) is not vendored; this file is
the processed analysis panel.

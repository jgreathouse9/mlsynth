# `iran_sanctions_oil.csv`

The country panel behind the Iran application in Spruk (2026), *"Synthetic
Control under Out-of-Span Common Shocks: Diagnosis, Exposure Balance and
Correction"*, arXiv:2610.05535. It is the panel the EASC replication study
(`benchmarks/studies/easc_replicate/`) ingests through `dataprep`.

- 22 units, 1996–2024 (638 rows): Iran plus the 21 donors of the paper's
  exposure-feasible pool. The 19 normalisers are KAZ, NGA, ECU, MEX, COL, JOR,
  ROU, BGR, VNM, ZAF, PHL, THA, KEN, BRA, CHL, PER, CRI, DOM, PAN; the
  exposure-feasible pool adds AZE and AGO.
- Treated unit: Iran (`iso == "IRN"`) from 2012, encoded in `sanctions`.
- Columns:
  - `iso` (ISO3), `year`
  - `gdppc` — GDP per capita, PPP, constant 2021 international dollars (WDI
    `NY.GDP.PCAP.PP.KD`)
  - `lgdppc` — its natural log; the outcome
  - `oilrent` — oil rents, percent of GDP (WDI `NY.GDP.PETR.RT.ZS`),
    1996–2011 only. The Dominican Republic and Panama report no values; the
    paper sets their exposure to zero.
  - `oil_share` — V-Dem v16 petroleum, coal and gas income per capita relative
    to GDP per capita (Haber and Menaldo 2011), in percent, available
    1996–2006; the paper's alternative exposure index.
  - `brent` — Brent crude, US dollars per barrel, annual average (FRED
    `POILBREUSDA`); the observed common shock, identical across units.
  - `sanctions` — treatment indicator.

The paper's exposure index is each unit's 1996–2011 mean of `oilrent` (Iran
22.87); the observed shock is `log(brent)`.

## Provenance

Extracted from the author's replication package (submitted to the *Journal of
Applied Econometrics*), whose README states that the WDI, FRED and V-Dem terms
of use permit redistribution of these extracts for replication:

- `lgdppc`, `gdppc`, `oil_share` from `data/panel_2012.csv` (built by the
  package's `build_panel.py`; sha256 `afc35ee8…c1451341`);
- `oilrent` from `data/oilrents.csv` (sha256 `980004d2…10c20b84e`);
- `brent` from `data/brent.csv` (sha256 `e5a57769…b08822d`).

Values are copied unchanged; `lgdppc`, `oilrent`, `oil_share` and `brent`
round-trip bit for bit. The package's README records that the WDI and oil-rent
extracts were transcribed from API responses on 30 September and 2 October
2026, and ships `get_data.py` to re-download them; that comparison has not been
run here because the World Bank and FRED hosts were not reachable from the
environment that built this file. Annual growth of `lgdppc` correlates 0.70 to
0.93 with V-Dem's independent GDP series (`e_gdppc`) over 1997–2019 for 21 of
the 22 units (Vietnam: 0.28), and no series shows repeated growth rates.

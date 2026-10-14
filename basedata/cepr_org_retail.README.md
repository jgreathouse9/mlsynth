# `cepr_org_retail.parquet`

Retail-trade records from the CEPR Uniform Extracts of the Current Population
Survey Outgoing Rotation Group ([Center for Economic and Policy Research](https://ceprdata.org/cps-uniform-data-extracts/cps-outgoing-rotation-group/),
version of 2020-02-05). Individual records: one row per person-interview in the
quarter-sample that answers the earnings questions (`minsamp` 4 or 8).

604,271 rows, 51 state identifiers, 1979 through 1991.

This is the slice the California minimum-wage example needs and nothing else.
Retail trade is the sector Card (1992) studies; 1979 is the first year CEPR's
earnings variables are usable, and 1991 the last year before the federal
minimum wage reached California's, which is where the comparison stops being
informative. The full extract across all 49 industries and 1979-1993 is
roughly 94 MB, and `benchmarks/data/build_cepr_org.py` rebuilds any part of it
from the CEPR releases.

```python
import numpy as np, pandas as pd
d = pd.read_parquet("basedata/cepr_org_retail.parquet")
d = d[d.wage3.notna() & (d.orgwgt > 0)]
state_year = (d.groupby(["state", "year"])
                .apply(lambda x: np.average(x.wage3, weights=x.orgwgt),
                       include_groups=False))
```

Records, not a panel: aggregate them to whatever unit a design needs. Rows are
sorted by `year`, then `state`.

## Columns

Geography and time
  `year`, `month`, `minsamp` (month in sample: 4 or 8), `state` (name),
  `metro` (1 inside a metropolitan area). There is no sub-state geography:
  every metro-area code in the source (`cmsacode05/14`, `smsastat05/14`,
  `fipscounty`, `la`, `nyc`, `cbsasz`) is empty for these years, so individual
  metros such as Dallas-Fort Worth cannot be identified from this extract.

Demographics
  `age`, `female`, `wbho` (race), `educ`, `married`.

Employment
  `empl`, `selfemp`, `pubsect`, `unmem` (union member), `uncov` (union
  coverage), `uhours` (usual hours, main job), `hourslw` (hours last week),
  `paidhre` (paid by the hour), `ind70` / `ind80` (3-digit census industry).

Earnings
  `weekpay`, `earnwke`, `wage1` (hourly workers), `wage2` (non-hourly workers,
  computed), `wage3` and `wage4` (CEPR's combined hourly wage, nominal),
  `rw` (real wage, 2019 dollars), `tc` (weekly earnings topcoded),
  `blsimph` / `blsimpw` (BLS allocation flags).

Weights
  `orgwgt` (earnings weight -- use this for ORG wage means), `fnlwgt`.

The wage measures are not interchangeable. `wage1` covers only workers paid by
the hour; `wage2` imputes an hourly rate for salaried workers by dividing weekly
earnings by usual hours, which is noisier (within state-year coefficient of
variation 0.60 for `wage3` against 0.46 for `wage1` over 1984-1987). `wage3` and
`wage4` combine them under slightly different topcode and allocation rules.
Which one a study uses changes its answer, so it is part of the specification.

## Companion file

`vz_minimum_wage.csv` -- effective state minimum wages 1979-1991, built from the
raw spreadsheets of
[equitablegrowth/VZ_historicalminwage](https://github.com/equitablegrowth/VZ_historicalminwage)
(Vaghul and Zipperer 2016). 663 state-year rows:

- `state_mw` -- state statutory minimum in force on 1 July of that year
- `fed_mw` -- federal minimum in force on the same date
- `eff_mw` -- the larger of the two, the effective minimum
- `first_effective_hike` -- first year after 1987 in which `eff_mw` rose above
  its 1987 value, or blank
- `first_statute_hike` -- the same for `state_mw` alone

The two hike columns disagree for four states: a statute can rise to a level at
or below the federal floor, which leaves the effective minimum unchanged
(Nevada in 1988 is the clearest case). Which column a design screens on changes
the usable control set by nine states.

California's own series is the one these files were assembled for: $3.35 through
1987, $4.25 effective July 1988, with the federal minimum reaching $4.25 in
April 1991.

## Provenance

Built by `benchmarks/data/build_cepr_org.py`, which keeps 33 of the 162 source
variables (the occupation recodes `occ70` / `occ80` are dropped: their Stata
value labels collide on reread) and then selects one industry and a year range.
`benchmarks/data/build_vz_minimum_wage.py` rebuilds the minimum-wage file and
reproduces the committed CSV exactly. Nothing is rescaled, filtered or imputed
beyond that selection -- wages are nominal as CEPR supplies them.

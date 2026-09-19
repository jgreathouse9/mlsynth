# Runs behind the tables

| file | produced by | settings |
| --- | --- | --- |
| `calibration.json` | `calibration.py` | 40 locations, pre 1..75, placebo post 76..90, alpha 0.10; MVBBSC at 1000 warmup / 1000 samples / 2 chains, max r-hat 1.003 |
| `market_selection.json` | `market_selection.py` | GeoLift walkthrough config, MVBBSC at 600 / 600 / 2 |
| `criterion.json` | `criterion.py` | three origins, 150 sampled designs each, sdid engine, 6 backtests |
| `bayes_sc_geolift.png` | `plot.py` | test panel, chicago + portland, treatment at period 91 |

## Port check

`calibration.json` holds the 1000 / 1000 / 2 run. Re-running the same arm
through the committed `engines.py` at its default 600 / 600 / 2 gives
`mvbbsc_iid` 67.5% coverage at width 301 and `mvbbsc_ar` 85.0% at width 416,
against 65.0% / 299 and 87.5% / 418. Widths agree to within 1%; coverage differs
by one location out of 40 in each arm, which is the sampling budget.

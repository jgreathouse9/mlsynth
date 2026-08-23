# Calibrating a cumulative conformal band

The narrative version of this study, with all the tables, is
`docs/cumulative_calibration.rst`. This file is the operating manual.

## What is here

| file | what it measures |
| --- | --- |
| `panel.py` | the DGP: real factor paths, loadings, unit means and residuals from a configured panel, with a circular shift common to all units |
| `window_curve.py` | coverage against the number of calibration windows, with exchangeability and normality granted |
| `design_cost.py` | the mixed-integer design solve against a simplex QP replicate, and the rank arithmetic for an in-space placebo |
| `burstiness.py` | whether the pointwise band's violations cluster in time |
| `loo_ensemble.py` | four cumulative bands per draw, arranged so each comparison isolates the ensemble or the window count |
| `external_libraries.py` | whether MAPIE can carry the construction (it carries the ensemble, not the estimand) |
| `results/` | the 100-draw runs behind the tables in the docs page |

The self-contained part of the study is the durable benchmark case
`benchmarks/cases/conformal_window_count.py`, run by
`python benchmarks/run_benchmarks.py --case conformal_window_count`.

## Which panel the arms use

The panel these arms were measured on is proprietary and is not shipped. With
nothing configured they build a stand-in instead — `synthetic_geo_panel.py` —
so every arm runs anywhere, out of the box:

```bash
python burstiness.py 100 0 results/burstiness.jsonl     # no configuration needed
```

The stand-in is drawn to behave like the panel it replaces, not to copy it. One
dominant factor carrying trend and seasonality, mildly persistent idiosyncratic
errors once that factor is removed, market sizes spread over orders of magnitude,
and a small correlation shared inside a region. Every constant in it is a design
parameter with a round value; markets and weeks are integers, log-levels are
centred at zero, and no observation, label, date or magnitude from any real panel
appears in it. `benchmarks/tests/test_synthetic_geo_panel.py` holds it to those
behaviours.

To run against a real panel instead, point the scripts at one:

| variable | meaning | default |
| --- | --- | --- |
| `MLSYNTH_CAL_PANEL` | path to a CSV | the stand-in |
| `MLSYNTH_CAL_TIME` | column holding the period | `start_date` |
| `MLSYNTH_CAL_UNIT` | column holding the unit | `dma` |
| `MLSYNTH_CAL_VALUE` | column holding the outcome | `total` |

A configured path is never replaced by the stand-in: if it cannot be read, the
error stands. Each result row carries a `panel` field naming its source, so a
results file cannot be read as real-panel numbers that a substitution produced.

A real panel needs at least `T0 + H = 117` periods and enough units to draw
`J = 20` from. Outcomes are logged, so they must be positive.

## Running

```bash
cd benchmarks/studies/cumulative_calibration
export MLSYNTH_CAL_PANEL=/path/to/panel.csv

python window_curve.py                     # no panel needed
python design_cost.py
python burstiness.py 100 0 burst.jsonl
python loo_ensemble.py 100 0 loo.jsonl
python loo_ensemble.py --summarise loo.jsonl
python external_libraries.py               # needs MAPIE, which mlsynth does not depend on
```

The two arms that solve a MAREX design need the `[design]` extra (SCIP) and take
roughly 20 seconds a draw, so the 100-draw runs are best split across processes:

```bash
for i in 0 1 2 3; do
  python loo_ensemble.py 25 $((i * 25)) loo_$i.jsonl &
done
wait
python loo_ensemble.py --summarise loo_*.jsonl
```

`window_curve.py` and the benchmark case need no panel and no solver.

## Headline findings

1. Coverage of a cumulative band is bounded by the number of calibration windows,
   not the number of periods, and reaches nominal only when that count is in the
   dozens. This survives full exchangeability, so it is not a dependence problem.
2. The pointwise band's violations do cluster, mildly: a miss triples the odds of
   the next period missing, but the runs test does not reject and the worst burst
   is two periods out of thirteen.
3. An EnbPI-style ensemble improves the band, beating what ships four discordant
   pairs to nil while being six percent narrower. The improvement comes from
   aggregating the members, not from the extra windows, which lose because the
   treated set was chosen having seen the fitting periods.
4. A conformal replicate costs a simplex QP, not a mixed-integer solve: 33
   milliseconds against 12.4 seconds. What limits an in-space placebo is the rank
   requirement of 19 usable donors, not compute.

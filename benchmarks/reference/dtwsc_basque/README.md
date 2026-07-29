# DTWSC reference — Cao & Chadefaux (2025) on the Basque panel

Cross-validation material for a future `DTWSC` estimator (Dynamic Synthetic
Control; DTW speed-warping). There is no `benchmarks/cases/` entry yet — the
estimator does not exist. This directory holds the reference generator so the
case can be written against a fixed, regenerable dump when the build lands.

## Producing the dump

```bash
bash benchmarks/R/install_dtwsc.sh
DSC_REPO=benchmarks/reference/.cache/dsc_repo \
  Rscript benchmarks/reference/dtwsc_basque/reference.R
```

Reference package pinned at `conflictlab/dsc` `b1cd241518329ac2bc8cfe21a871a798ac14d74f`
(MIT). Not vendored.

## What the dump contains

| file | contents |
|---|---|
| `gold_preproc.csv` | the rescaled + Savitzky–Golay-filtered panel that feeds TFDTW |
| `gold_tfdtw.csv` | per donor: `cutoff`, `weight.a`, `avg.weight`, warped series |
| `gold_fit.csv` | `time`, treated `value`, DSC counterfactual, standard-SC counterfactual |

All floats at `%.17g`. This matters: at R's default 15 significant digits,
`weight.a` appears to disagree with a correct port at 3e-12, which is the dump
precision rather than a real difference.

## Numbers to match

Basque, 1955–1997, treatment 1970, Spain (the national aggregate) dropped so 16
donors, full 14-predictor Abadie spec, `k = 4`, `filter.width = 5`, `buffer = 0`,
`n.burn = 3`, `symmetricP1` / `asymmetricP2`.

| method | pre-RMSE | ATT |
|---|---|---|
| standard SC | 0.0886 | −0.6027 |
| DSC (R reference) | 0.0705 | −0.5579 |

The ATT is averaged over 1971–1996: three donors' warped series end one period
short of 1997, so the reference's own counterfactual is `NA` there. Any port
must reproduce that `NA`, not paper over it.

## State of the Python port

A demonstrate-first port of the warping engine reproduces, against this dump:

| quantity | agreement |
|---|---|
| `cutoff` | 16/16 donors exact |
| `weight.a` (first-phase warp) | 16/16 exact, worst 2.2e-16 (1 ULP) |
| second-phase window search (`j_opt`, `margin_opt`, candidate count) | 28/28 windows exact on the worst donor |
| outlier-filter decisions | 13848/13888 cells (99.71%) |
| pre-RMSE, R's `Synth` on the ported warped donors | 0.0705 — matches to 4 dp |
| ATT | −0.5592 vs −0.5579 (0.23% of the ATT, 2.9% of the DSC-vs-SC gap) |

The residual is 40 cells where `RemoveOutliers` disagrees, split near-evenly in
both directions (21 R-only, 19 Python-only). Those are floating-point ties: the
weights are small-denominator rationals (2/3, 1, 4/3), so they land within a few
ULP of the `Q1 ± 3·IQR` bound routinely, and the decision then turns on
summation order rather than on the data. One contributor is that R's
`quantile(type = 7)` evaluates `(1-h)·x[lo] + h·x[hi]` while numpy's `quantile`
uses a different lerp branch — a 2 ULP difference on a bound that sits 3 ULP from
the data. A symmetric tolerance does not close it, because the disagreements run
both ways.

Treat this as the known floor for the cross-check tolerance, and pin the ATT with
a tolerance no tighter than ~0.005.

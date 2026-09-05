# GP-ITS — Cho (2026), D.C. Heller replication

A demonstrate-first replication of Gaussian-process interrupted time series
([arXiv:2608.20610v1](https://arxiv.org/abs/2608.20610)), the donor-free
counterfactual estimator for universal treatments — an intervention that reaches
every unit at once, so difference-in-differences and synthetic control have no
comparison group and the counterfactual has to come from the unit's own history.
No estimator has been added to `mlsynth` yet; this bundle establishes that the
paper reproduces before one is built. The roadmap entry is item 23 of
`agents/future_integrations.md`.

Result: the headline reproduces exactly, and the NumPy port agrees with the
author's R to roughly 1e-11 on every quantity. Recommendation is build.

## What is here

```
gpits_port.py            NumPy port of gpss::gp_its, traced function by function
run_heller.py            Path A: the D.C. series, port vs the paper
reference.R              gpss::gp_its on the same series -> reference.json
placebo_ref.R            gpss placebo checks -> placebo_reference.csv
run_panel.py             all 50 jurisdictions -> panel_results.json (Figure 4A)
run_coverage.py          Section 5 coverage, GP vs segmented -> coverage_results.json
verify.py                re-derives every number below and checks provenance
dc_series.csv            the D.C. monthly series (derived; see Provenance)
```

```bash
python benchmarks/reference/gpits_heller/verify.py            # all checks
python benchmarks/reference/gpits_heller/verify.py --tamper   # proves they bite
```

`verify.py` needs only `dc_series.csv`, which is committed. `run_panel.py` and
`run_heller.py` need the authors' NICS and census files; `reference.R` and
`placebo_ref.R` need R with `gpss`. See Provenance for both.

## Path A — the headline

D.C.'s cumulative four-month effect on handgun background checks after
*District of Columbia v. Heller*, per 100,000 population:

| | cumulative | 95% interval |
|---|---|---|
| paper, Section 6 | 15.1 | [13.0, 17.3] |
| this port | 15.1323 | [12.9687, 17.2960] |

Exact at the precision the paper reports. The port also gives roughly 88
additional background checks against the paper's "roughly 90" — the paper does
not say which population vintage it divided by, and the 2008 D.C. estimate of
580,236 gives 87.8.

The monthly effects show the shape the paper describes: near zero in July and
August 2008, then 6.83 and 8.10 in September and October.

## Cross-validation against the author's R

`gpss::gp_its` (GPL-3, CRAN) run live in R 4.3.3 on the same series, same
configuration as `gpits/code/02_heller_main.R`: Gaussian + periodic + linear
kernel, period 12, month as a categorical covariate, prediction intervals.

| quantity | max abs diff | max rel diff |
|---|---|---|
| `b` (length-scale) | 5.6e-12 | 2.0e-12 |
| `s2` (noise variance) | 4.7e-11 | 5.5e-11 |
| counterfactual | 1.0e-11 | 9.6e-11 |
| `y0_se` | 1.1e-11 | 2.3e-11 |
| `tau_t` | 1.3e-11 | 1.6e-12 |
| `tau_cum` | 7.3e-11 | 4.8e-12 |
| `tau_cum_se` | 1.9e-11 | 1.8e-11 |
| placebo `tau` | 1.0e-11 | 2.1e-11 |

Two independent implementations — Rcpp/Armadillo against NumPy — at
floating-point agreement. The residual is the Brent optimizer's convergence
path, not the arithmetic: `getb_maxvar` and `gp_optimize` both call R's
`optimize()`, and the port matches its default tolerance
(`.Machine$double.eps^0.25`) and the `tol = 0.1` that `gpss` passes for `s2`.

The port preserves several conventions of the reference that a from-the-paper
implementation would get wrong. The periodic and linear components run over
every column of the design, the one-hot month dummies included, not the time
column alone. One-hot columns are multiplied by `sqrt(0.5)` and are never
centred or scaled. The period is rescaled by the standard deviation of the first
continuous column. The marginal likelihood uses `sum(log(diag(L)))`, half the
log-determinant. Each is flagged at its site in `gpits_port.py`.

## Figure 4A — the nationwide null

Fitting an independent GP to each of the 50 jurisdictions, on the standardised
scale the paper's Figure 4A uses (`effect_cum / pre_treatment_sd`, from
`02_heller_main.R`):

| | standardised cumulative |
|---|---|
| D.C. | 36.99 |
| next highest (Nevada) | 8.73 |
| other 49, median | 0.91 |
| other 49, interquartile range | [−0.42, 1.80] |

D.C. is rank 1 of 50 and the claim reproduces. Read the scale carefully, though.
On the raw per-100k scale D.C. ranks 25th of 50, and six other jurisdictions are
significant at 95% — Alaska at 200.2 and Missouri at −193.1 are both larger in
magnitude than D.C.'s 15.1. The separation comes from the standardisation, and
D.C.'s pre-treatment standard deviation is 0.41 against a median of 19.8 across
the others, a factor of about 48. The paper states this itself: the near-total
ban holds D.C.'s pre-treatment series near zero, "its suppressed variance
inflates the standardised scale, so we report the magnitude on the count and
per-capita scales instead." The figure and the reported magnitude are on
different scales by design, and the visual claim rests on the one where D.C.'s
denominator is smallest.

The four placebo periods all cover zero, matching the paper.

## Section 5 — the coverage claim

The paper's central claim is about interval calibration, not point accuracy, so
this is the part that decides whether the method earns its place. DGPs
transcribed from `05_simulations.R`; 200 replications against the paper's 500;
C-ARIMA and CausalImpact omitted because their R packages need CRAN, which this
environment cannot reach. Segmented regression is the ITS workhorse and the arm
the argument is really against.

Coverage of the nominal 95% interval, and the mean interval half-width:

| scenario | n_pre | GP coverage | GP half-width | segmented coverage | segmented half-width |
|---|---|---|---|---|---|
| kernel_smooth | 12 | 0.986 | 5.88 | 0.730 | 1.20 |
| kernel_smooth | 60 | 1.000 | 1.91 | 0.665 | 0.88 |
| kernel_smooth | 120 | 1.000 | 1.44 | 0.768 | 0.85 |
| nonlinear_trend | 12 | 1.000 | 4.26 | 0.201 | 1.17 |
| nonlinear_trend | 96 | 0.998 | 1.10 | 0.804 | 1.10 |
| trend_seasonal | 120 | 0.998 | 1.12 | 0.827 | 0.49 |

Across all 15 cells the GP runs 0.986 to 1.000 and segmented regression 0.201 to
0.844. The full grid is in `coverage_results.json`.

The qualitative claim holds and is not a weak-baseline artifact: segmented
regression's interval reflects residual variance around its own fitted form and
does not widen off support, so it under-covers everywhere, badly at short
pre-periods.

The quantity the paper's coverage panels do not show is width. "At or above
nominal" is accurate, and at moderate pre-lengths the GP is at 1.000 with
intervals a median of 2.3 times wider than segmented regression's, up to 4.9
times at the shortest pre-periods. That is the worst-case bound doing what
Section 4 says it does, and the paper is explicit
that this is the intent — but a practitioner should read the intervals as
conservative, not calibrated, and an estimator page should say so. It also means
the method buys its coverage in power: an effect small relative to the band will
not be detected. In the Heller application the effect is roughly 20 times the
pre-period standard deviation, which is why it survives.

## Provenance

Reference code: `soonhong-cho/gpits` (MIT) for the pipeline and data,
`doeun-kim/gpss` (GPL-3) for the estimator. `gpss` was installed from source
with `R/gp_rdd.R` removed and `ggplot2` / `rlang` dropped from `Imports` —
those are used only by the regression-discontinuity plot and CRAN is unreachable
here. No file under `R/gp_functions.R`, `R/gp_its.R`, `R/helper_functions.R` or
`src/kernels.cpp` was touched, so the estimator's numerics are the author's.

`gpits_port.py` is an independent NumPy implementation written from the paper's
equations and the reference's structure. `gpss` is GPL-3 and `mlsynth` is MIT,
so no `gpss` source is copied or vendored here; the R scripts invoke the
installed package as an external reference, which is the pattern this directory
already uses for Synth and mscmt.

`dc_series.csv` is committed because `verify.py` needs it and it is small and
derived, not redistributed source: FBI NICS monthly background checks for the
District of Columbia, divided by Census population and scaled per 100,000, over
2002-07 to 2008-10, built by `run_heller.py` from
`gpits/data/nics/NICS_state_month_11.1998_1.2024.csv` and
`gpits/data/census_pop/pops_by_state.RDS`. The full 50-jurisdiction panel is not
committed; clone the MIT repo to regenerate `panel_results.json`.

## Artifact provenance

`coverage_results.json` carries a `meta` block recording the replication count,
grid, seed and a hash of the code that produced it, and `verify.py` refuses any
artifact whose `meta` disagrees with `manifest.json`. That exists because a
10-replication smoke run was briefly staged here in place of the 200-replication
run: both write the same filename, the gate that was supposed to wait for the
real run tested only that the file existed, and a file from the smoke run
already did. `verify.py --tamper` reproduces the defect and shows the check
catching it.

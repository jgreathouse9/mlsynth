# Ferman, Pinto & Possebom (2020) on SpecificationStudy: demonstrate-first spike

Replication spike for Bruno Ferman, Cristine Pinto and Vitor Possebom, "Cherry
Picking with Synthetic Controls", *Journal of Policy Analysis and Management*
39(2): 510-532, built on top of PR #958 (`SpecificationStudy` for VanillaSC).
Reference code: the authors' replication package, openICPSR project 117261,
version 2 (https://doi.org/10.3886/E117261V2), file `replications_emp.zip`.
This is the evidence behind the build decision. The recommendation below is
now built on the same branch: `SpecificationStudy(placebo=True)`, the
inference methods on its result and the `identification` table, documented in
`docs/sensitivity.rst` and checked against this spike's reference traces by
`mlsynth/tests/test_sensitivity_inference.py`. The code in this directory is
the spike as it was run.

FPP ask how much a synthetic-control result depends on which pre-treatment
outcome lags the analyst uses as predictors. They refit seven predictor sets,
each with and without covariates (14 specifications), run the in-space placebo
test on each, and recommend a test that combines the specifications: the mean
post/pre statistic across specifications, ranked by permutation, reported with
the average effect across specifications and a Firpo-Possebom (2018)
confidence band around it.

## Findings

### West Germany (main text: Table 3, Figures 1 and 2)

1. The authors' code still reproduces their main-text results. Run live under
   Synth 1.1-5, `replication_germany.R` gives 13 of Table 3's 14 p-values
   exactly. The fourteenth (the paper's 7b) has the same rank, first of the
   units with a statistic; the script skips Austria for it ("the optimization
   routine cannot find a solution") and divides by 16, giving 0.0625 where the
   table prints 0.059 (1/17). The joint test is 0.059, the Dube-Zipperer and
   Donohue-Aneja-Weber rules pick the paper's 1a (p = 0.059) and 2b
   (p = 0.118), and the Figure 2 band ends 2003 at -5,741 to -2,777 around an
   average of -4,429, matching the authors' own figure output to the
   resolution of the plot.
2. The inference port is exact. Recomputed here from the R run's own donor
   weights, every unit's statistic agrees with R's to 5e-13, every p-value and
   R-tilde agrees, and the joint test, both choice rules and the band are the
   R values. The predictor matrices agree with R's `dataprep` to 7e-6.
3. The specification-search evidence in Table 3 is partly solver evidence.
   Synth solves the inner problem (donor weights given V) with kernlab's
   interior-point `ipop`, which stops short of the minimum: for the V each R
   fit chose, mlsynth's exact active-set QP reaches a predictor loss at or
   below Synth's in all 237 fits, by orders of magnitude in some
   (West Germany, the paper's 3a: 3e-17 against 6e-6). Synth's weights are
   interior points: 168 of the 237 give every donor a weight above 1e-6.
   Holding each fit's V fixed and solving the inner problem exactly changes 4
   of the 14 p-values, among them the paper's two largest: 6a (the pre-period
   mean alone) goes from 0.588 to 0.059 and 7a (three lags) from 0.353 to
   0.059. The other two are 1a (0.059 to 0.118) and 3a (0.118 to 0.176). The
   designs with one, three or five predictors are the extreme case: the
   predictors of many units can be matched exactly (15 of 17 units with the
   mean alone, 8 of 17 with three lags, 3 or 4 of 17 with five lags), so the
   specification does not pin the weights down and the p-value records where
   the solver stopped inside that set.
4. Searching V with mlsynth's engine on the same predictor matrices gives the
   same answer for 10 of 14 entries. The four that differ are the same two
   (6a and 7a, now 0.059) and two covariate designs whose p-values rise (2b:
   0.118 to 0.353; 3b: 0.059 to 0.176). The joint test stays 0.059 and the
   band's conclusion stands (2003: -5,295 to -2,742 around -4,038), but the
   Dube-Zipperer rule now picks 5a and the Donohue rule 1a.
5. Neither optimizer dominates on the outer objective. mlsynth's search reaches
   a validation MSE at or below Synth's in 205 of 237 fits (86 percent). In 18
   of the other 32, Synth's weights are not optimal for Synth's own V: `ipop`
   stopped with a larger predictor loss than the exact minimum, which lets the
   outer search trade predictor fit for outcome fit the bilevel program does
   not allow. In the remaining 14, Synth's local search found a V that
   mlsynth's differential evolution missed.

The paper's conclusions survive replication: specifications disagree, and the
combined test and band give a clear answer where single specifications do not.
Which specifications disagree, in this application, depends on the optimizer
as much as on the predictor set.

### California (Appendix F, Table F2)

6. `replication_california.R` reproduces 13 of Table F2's 14 p-values
   exactly. The fourteenth, 6b, is one of the three designs the authors ran
   with LowRankQP; under Synth's default `ipop` California's own fit fails, so
   it has no statistic here. R-tilde agrees at three decimals for 1a, 1b, 6a,
   7a and 7b and within 0.010 for the other six printed values. The combined
   test over specifications 1 to 5 with R-tilde at or above 0.975 (1a, 1b, 2b,
   4a, 4b, 5a, 5b) gives 0.077, the paper's value.
7. The Dube-Zipperer rule computed from Table F2's fits picks 2b (p = 0.077),
   not the paper's 4b (p = 0.026). The authors computed it in a separate
   script whose placebo fits drop California from every donor pool
   (`setdiff(1:39, c(3, j))`) and use LowRankQP for two designs, so the rule
   rests on different fits from the table's; that script was not rerun.
8. The inner problem moves the California results the same way. Given each
   fit's V, mlsynth's exact QP reaches a predictor loss at or below Synth's in
   every fit. Six of the thirteen p-values move by one rank (1/39), in both
   directions, and the pre-period mean alone (6a), printed with R-tilde
   0.525, fits at 0.979 once the tie-break chooses among the weight vectors
   that match the mean exactly; three lags alone (7a) go from 0.909 to 0.975.
   Appendix D's evidence that specifications 6 and 7 fit poorly is, in this
   application, partly evidence about where `ipop` stops.
9. Through the PR's runner (`PlaceboSpecificationStudy`: 14 designs x 39
   units, 546 VanillaSC fits with the `mscmt` backend, no failures), 9 of the
   14 p-values are the paper's and the combined test is again 0.077. ADH
   (2010)'s own specification (7b) moves from 0.026 to 0.051, both odd-year
   designs (4a, 4b) rise to 0.077, and the Dube-Zipperer rule picks 3a. The
   pre-period mean alone has three answers: R-tilde 0.525 under Synth, 0.979
   from mlsynth's lower-level QP with its outcome tie-break, and 0.393 from
   VanillaSC's single-predictor path. All three match the predictor,
   California's 1970-1988 mean of 116.21 packs, to within 1.1e-6 packs; the
   design does not choose between them.

Across both applications the combined test is the same under both solvers
(Germany 0.059, California 0.077), and the effects keep their sign and size
(California: -18.7 to -20.6 packs per capita over 1989-2000 in every design
with R-tilde at or above 0.975, under either solver). The individual cells
FPP use to show that specifications disagree are the part that moves.

## What extending the PR needs

The spike runs FPP's procedure on #958 without changing the PR's code. A build
on top of it needs five things, in this order of necessity.

1. Placebo runs inside the study. FPP's inference needs every unit's statistic
   for every design, with the treated unit kept in the placebo units' pools
   (ADH 2010; FPP's code does the same). `PlaceboSpecificationStudy` gets this
   by running `SpecificationStudy` once per unit with that unit treated, so
   every placebo fit passes through the PR's validation, fixed backend and
   failure log. VanillaSC's own `inference="placebo"` cannot stand in: it
   leaves the treated unit out of the placebo pools and keeps only the
   treated unit's ratio, its rank and the p-value
   (`utils/vanillasc_helpers/pipeline.py`, the in-space branch), so it cannot
   feed a combined test.
2. The combined inference on the result: the joint test (the mean statistic
   across a chosen set of designs, ranked by permutation), the
   Firpo-Possebom band around the average effect, and the Dube-Zipperer and
   Donohue-Aneja-Weber choice rules. `fpp.py` has all four, checked against R
   to 5e-13, and they need nothing beyond the per-unit gaps.
3. An identification flag per design. Finding 3 is the reason: where the
   predictors can be matched exactly by many weight vectors, the weights and
   the p-value come from the solver's tie-break, not from the design. The
   study should report, per design and unit, whether the minimum predictor
   loss is zero to tolerance, which makes the minimizing weights a set and
   not a point (the usual case with few predictors and many donors), and that
   mlsynth's tie-break chose within that set by pre-period outcome fit.
4. A one-line fix to the PR. `SpecificationStudy.fit()` recorded
   `importlib.metadata.version("mlsynth")`, which raised
   `PackageNotFoundError` from a checkout that was never pip-installed, after
   every design had been fit. #958 fixed this at `e26902be` (the version is
   now `None` there).
5. Two-step cross-validation (ADH 2015: choose V on a training problem, reuse
   it on predictors re-measured over a later window). VanillaSC has no way to
   take a V from elsewhere, so the Germany refit calls the bilevel engine and
   mlsynth's lower-level QP directly. That is a VanillaSC option, separate
   from the study, and the only one of the five the California application
   does not need.

Out of scope: the Monte Carlo tables (Path B: Tables 2, B1, C1-C2, D1-D3,
E1-E2, G1-G3) and the two applications whose data are not in the package.

## Recommendation

Build, after #958 merges and on its own branch: items 1 to 3 as a placebo mode
on the study and the four inference functions on its result (item 4 is fixed
in #958), with this
spike's Germany trace as the durable check, a `benchmarks/cases/` case that
rebuilds Table 3's fourteen p-values, the joint test, both choice rules and
the Figure 2 band from the captured R weights (exact to 5e-13 here). Item 5
belongs to VanillaSC and can wait for a user who needs ADH (2015)'s
cross-validation.

The build is small because the inference is already ported and checked. Its
open design question is item 3. In both applications the per-design cells
move with the solver, and in California mlsynth's own two code paths give
different weights for the same unidentified design, so a study that reports
p-values across designs without saying which designs pin their weights down
reports solver variation as specification variation.

Cost is the other constraint. The placebo layer multiplies every design's fit
by the number of units: California's 546 fits took 3.4 hours of
single-process time here (four processes sharing four cores) with the
`mscmt` backend. A design with 19 to 23 predictors takes 20 to 40 seconds a
fit, one with seven under a second.
The build should fit units in parallel and reuse fits across designs that
share a predictor set.

## What is here

| file | what it does |
| --- | --- |
| `fpp.py` | specification registries (both applications, in the authors' exact predictor windows), a port of Synth's predictor construction and its `custom.v` step on mlsynth's lower-level QP, and FPP's inference: ratio statistic, permutation p-value, joint test, R-tilde, Dube-Zipperer and Donohue-Aneja-Weber choice rules, Firpo-Possebom band |
| `placebo_study.py` | `PlaceboSpecificationStudy`: PR #958's `SpecificationStudy`, run once per unit with that unit treated, so every placebo fit goes through the PR's runner unchanged |
| `replicate.py` | the `paper`, `port` and `inner` layers against the reference traces |
| `refit.py` | the `refit` layer: mlsynth's own predictor-weight search on the same specifications |
| `report.py` | tables and figures under `results/` |
| `R/prepare_scripts.py` | writes the runnable copies of the authors' scripts: Germany split by specification, California with the R 4.3 edits |
| `R/run_fpp_reference.R` | runs one of those scripts and traces every `dataprep()` and `synth()` return |
| `R/export_trace.R` | re-emits those traces as JSON |
| `benchmarks/reference/fpp_cherry_picking/*_trace.json` | the captured reference: every synth() call's V, W and losses |
| `results/` | layer-by-layer numbers, tables, figures |

## Reproducing

The R reference needs the authors' package (not vendored here; download it
from the DOI above) and R with Synth 1.1-5. `agents/agents_r_environment.md`
covers installing R in this sandbox; Synth 1.1-5 is `cran/Synth` at
`c946535`. Their scripts were written for R 3.x; the edits that let them run
on R 4.3 are listed under "Reference environment" below.

The traces under `benchmarks/reference/fpp_cherry_picking/` came from the copies `R/prepare_scripts.py`
writes (Germany one specification block per script, California with the edits
below), each run from a working directory holding the script's data file
(`repgermany.dta`, `smoking_dataset.csv`):

```bash
python R/prepare_scripts.py <unzipped replications_emp> <scripts>
Rscript --vanilla R/run_fpp_reference.R <scripts>/spec_1a.R <workdir> germany_1a.rds   # ... spec_7b.R
Rscript --vanilla R/export_trace.R ../../reference/fpp_cherry_picking/germany_trace.json 7 germany_1a.rds ... germany_7b.rds
Rscript --vanilla R/run_fpp_reference.R <scripts>/replication_california_seq.R <workdir> california.rds
Rscript --vanilla R/export_trace.R ../../reference/fpp_cherry_picking/california_trace.json 3 california.rds
```

The Python layers then run from the captured traces alone:

```bash
cd benchmarks/studies/fpp_cherry_picking
PYTHONPATH=../../.. python replicate.py germany california
PYTHONPATH=../../.. python refit.py germany 4
PYTHONPATH=../../.. python refit.py california 4
PYTHONPATH=../../.. python report.py germany
PYTHONPATH=../../.. python report.py california
```

Python needs only `requirements.txt`.

## Reference environment

R 4.3.3 (Ubuntu noble), Synth 1.1-5 (`cran/Synth` at `c946535`, the CRAN
release from January 2014 until 2022, so the version the authors ran in
2018-2019), kernlab 0.9-32 and optimx 2023.10.21 from apt. Pinning optimx
2018-7.10 from `cran/optimx` was refused by this session's sandbox; Synth
reaches Nelder-Mead and BFGS through optimx, which delegates both to
`stats::optim`.

The scripts ran unmodified except for these edits, none of which touches an
estimate:

1. California: `read.csv(..., stringsAsFactors = TRUE)`. The script builds the
   state names from `levels(smoking$state)`, which is `NULL` under R 4's
   default, so the state column is deleted and `dataprep` stops.
2. California: `library("statar")` removed (loaded, never used, not packaged
   for apt) and the `doParallel` cluster replaced by `registerDoSEQ()`, so the
   in-process trace sees every fit.
3. California: three specifications (code `b1`, `c2`, `g1`) call
   `quadopt = "LowRankQP"`; LowRankQP was archived from CRAN and is not
   installable here, so they ran with Synth's default `ipop`.
4. California: each `synth()` call wrapped in `tryCatch`, so a fit that fails
   is recorded as missing instead of stopping the script (see the findings).
5. Germany: the script was run one specification at a time (the preamble plus
   one specification block, verbatim), so a failing fit costs one block.

## Reading the authors' code

These hold whatever the estimates turn out to be.

- The code and the paper label covariates the opposite way. In the code a
  specification ending in `a` carries the covariates; the paper's tables say
  `a` does not. Table 3 and Table F2 are the code's results with `a` and `b`
  swapped, and the "original" ADH (2015) specification the paper calls 6b is
  the code's `6a` (it is the black line of the authors' own Figure 1 output).
  `fpp.GERMANY_PAPER_LABEL` and `fpp.CALIFORNIA_PAPER_LABEL` hold the maps.
- Figure 2's band is computed by `cs_germany.R`, which assigns
  `weights.2.a <- spec.1.a[47:62, ]`, `weights.2.b <- spec.1.b[...]`,
  `weights.3.a <- spec.1.a[...]` and `weights.3.b <- spec.1.b[...]`. The
  "average across specifications 1 through 5" there is
  `(1a + 1b + 1a + 1b + 1a + 1b + 4a + 4b + 5a + 5b) / 10`: specification 1
  three times, 2 and 3 not at all. The band test uses the same list.
- Table 3's p-values and Figure 2 use different periods and statistics.
  Table 3 ranks the ratio of root mean squared gaps over 1960-1989 and
  1990-2003; the band ranks the ratio of mean squared gaps over 1960-1990 and
  1991-2003.
- The joint test in `replication_germany.R` averages the statistic over all
  14 specifications (`rowMeans(rmspe_matrix)`); the paper describes a test on
  specifications 1 through 5.
- `replication_california.R` passes `method = "BFGS"` to `synth()`. Synth
  1.1-5 accepts `...` and never forwards it, so Table F2 used Synth's default
  Nelder-Mead and BFGS. The band script `cs-uniform_california.R` passes
  `optimxmethod = "BFGS"`, so Figure F4 rests on weights estimated differently
  from Table F2's.
- "Odd" and "even" pre-treatment values follow the calendar year in the code
  (California: 1971, 1973, ... is specification 4; 1970, 1972, ... is 5), not
  the period index of the paper's formula, under which the two swap when the
  pre-period starts in an even year. Table F2's p-values follow the code.
- California lag set `f` (specification 4, odd years from 1971) lists 1971,
  1973, ..., 1985, 1986, 1987: 1986 is extra.
- The Bartel et al. (2018) and Smith (2015) applications are not in the
  package, so Table 4 and Table F1 cannot be checked.

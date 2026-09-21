# Single-treated-unit simulations (Pang, Liu & Xu 2022)

The R side of `benchmarks/cases/pang_liu_xu_sims.py`: the authors' own generator,
the panels it drew, and what `gsynth 1.0` and `pblasso 1.0.8` returned on them.

## What is here

`benchmarks/R/pang_liu_xu_sims.R` writes all of it. It reproduces
`code/simulateCalib.R` and `effSummary` from `code/summary_function.R` verbatim,
then sets the two designs the way their drivers do:

| file | what it holds |
|---|---|
| `ar1_moments.csv` | second moments of `arima.sim` itself, 20000 draws at each of the three path lengths |
| `dgp_moments.csv` | six moments of the generator at four panel sizes, averaged over 4000 draws |
| `effect_column.csv` | the largest effect `simulateCalib.R` writes, and the treated-unit count, over 500 panels per configuration |
| `seam_panels_r8.csv`, `seam_panels_X.csv` | four panels per design, long |
| `seam_reference.csv` | `gsynth` and `pblasso` on each of those panels |

The Python case pins its own transcription of the generator against
`ar1_moments.csv` and `dgp_moments.csv`, then puts mlsynth's `GSYNTH` and
`DMLFM` on the vendored panels and compares against `seam_reference.csv`.

Run any one part on its own with `--only moments|seam|effect|arsim`.

The published cells these designs produced are not here. They live next door in
`benchmarks/reference/dmlfm_germany/montecarlo_single_treated.csv`, which is the
authors' saved `tempdata/sim_single_{X,r3,r8}.RData`.

## The designs

Both come from the replication package (Harvard Dataverse doi:10.7910/DVN/B6SWA1),
both have one treated unit and ten post-treatment periods, both use `force = 2`
(time effects only), `error.sd = 5` and AR(1) coefficient 0.6, and both hold a
single draw of the time effect fixed within a case.

`r8` is `9_sim_single_r8.R`, Table A6: eight factors with loading standard
deviation 2, no covariates, and `gsynth` is handed r = 8, the true rank.

`X` is `10_sim_single_X.R`, Table A7: three factors with loading standard
deviation 4, plus six time-invariant covariates with `beta = (4,3,2,1,0,0)` whose
coefficients move with independent AR(1) series scaled by the same betas. Here
`gsynth` is handed `r + 4 = 7`, which is not the true factor count of 3 — the
covariate terms carry rank of their own, four from the AR(1) coefficients and one
from the time-constant part — so this cell is over-specified, not "r known".

`sim_single_r3` (Table A5) has no driver in the supplied material and is not
covered.

## The treatment effect is zero

`simulateCalib.R` builds `eff` as a zero matrix and the block that would have
filled it is commented out, so `true.eff` is identically zero in both drivers.
These are placebo designs: the tables' bias column is the mean estimate, and
their coverage column is coverage of zero. `effect_column.csv` measures this on
the authors' code instead of taking it from a reading of the source: over 2000
panels the largest absolute entry of `eff` is 0, and every panel has exactly one
treated unit.

## Two details a port has to match

The AR(1) series come from `arima.sim`, whose marginal variance is
1/(1 - 0.6²) = 1.5625. A transcription that starts the recursion at zero
instead of drawing the stationary first value understates the factor term. The
panel moments cannot catch that on their own: the shortfall is 1.9% of E[x²] at
length 30, and by the time the loadings, the covariates and an error standard
deviation of 5 have diluted it, it moves the cross-sectional moment by about 1%,
which is inside the noise of any affordable number of panel draws. Measured on
the series alone, in `ar1_moments.csv`, the separation is fortyfold: 0.04%
against 1.9%.

The time effect is drawn once per case and held fixed across replications, and
its deterministic part climbs by four a period. `dgp_moments.csv` therefore
redraws it, which the drivers do not: held fixed, a single draw decides most of
the within-unit variance, and the moment would report that draw and not the
generator. The within-time moment, which the time effect cannot touch, is the
one that pins the cross-sectional half on its own.

## Chain length

`seam_reference.csv` runs `pblasso` at the drivers' `niter = 10000 / burn = 2000`
and again at the `1500 / 375` the Python case can afford. The short chain costs
nothing here. The posterior mean moves 0.16 on average and 0.48 at worst against
a sampling standard deviation near 3, and the credible interval does not narrow:
the width ratio is 1.013 on average, between 0.965 and 1.042, with the short
chain the wider of the two on six of the eight panels.

The expectation going in was the opposite. One panel run at three chain lengths
gave widths of 9.79, 10.19 and 10.50 at 1000, 2000 and 5000 draws, which reads
as monotone. Eight panels put that sequence inside a single chain's own scatter.

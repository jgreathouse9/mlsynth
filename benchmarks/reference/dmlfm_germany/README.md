# DM-LFM reference bundle (Pang, Liu & Xu 2022)

Oracle material for a future port of the dynamic multilevel latent factor model.
mlsynth has no DM-LFM estimator, so there is no `benchmarks/cases` entry yet;
this directory holds what a port validates against.

## What is here

`reference.R` reproduces the DM-LFM half of the authors' `2_ex_adh2015.R` at five
seeds and writes the gold CSVs. It needs `pblasso` 1.0.8, which ships inside the
replication package as `packages/pblasso_1.0.8.tar.gz` and installs with
`R CMD INSTALL`. The public `github.com/liulch/bpCausal` repository is a later
rename with a different argument list, and searching every commit reachable from
all refs finds none of the arguments the paper's scripts pass, so it is not a
substitute.

Two of the captured objects are not CSVs. `german_design.json` holds every
matrix the sampler consumes on this panel -- `y`, `X`, `A`, the treated
blocks, the unit and time codes, the group-break vectors and the time-sort
permutation -- and `covariate_scaling.json` holds the covariate names and the
standard deviations the reference scales them by. They sit here, with the rest
of the captured reference, because the benchmark case reads them; they lived
under `mlsynth/tests/fixtures/dmlfm/` until a dependency-map refresh made a
benchmark case claim a path under the test tree, which the map forbids by
construction. `test_dmlfm_against_pblasso.py` reads `covariate_scaling.json`
from here too, and the two step fixtures it alone uses stay where they were.

## The specification

Six time-invariant covariates, each the unit's mean over all 44 years
(`pgdp`, `trade`, `inflation`, `industry`, `schooling`, `invest`, the last
pooling `invest60/70/80`); `Xname = Aname` so each enters with both a constant
coefficient and a time-varying one; `Zname = NULL`; `re = "time"`; `r = 10`;
AR(1) on the time-varying terms; `niter = 25000`, `burn = 5000`; flat priors on
the coefficients (`xlasso = zlasso = alasso = 0`) with shrinkage only on the
factor loadings (`flasso = 1`).

Two details a port has to match and neither is stated in the paper. The
covariate means run over the whole sample, post-treatment years included, so the
treated unit's covariates use post-treatment outcomes; `dataprep`'s
`covariate_aggregation="pre_mean"` will not reproduce them. And the counterfactual
draw adds `rnorm(sd = sqrt(sigma2))` on top of the mean function
(`blasso_core.R:538`), making it a posterior predictive draw; omitting the noise
narrows the credible intervals.

## Targets

| quantity | value |
|---|---|
| ATT 1990–2003 | −1597.9, sd 38.7 across ten seeds |
| range across seeds | [−1639.4, −1509.2] |
| pre-treatment gap | max abs 117.0 |
| gap 1990 / 1993 / 2003 | +457 / −173 / −4117 |

The seed-to-seed spread is wide relative to the effect, so compare a port on
means across several seeds, not on one run. The first five seeds gave sd 18.0
and the next five gave 54.7; treating the former as the tolerance would reject
a correct port. The current port agrees at Welch p = 0.16 on means and F-test
p = 0.42 on variances, over eight runs against these ten.

Factor loadings cannot be compared directly. The sampler permutes factor labels
each iteration (`permute`, `blasso.cpp:454`), so only the sorted spectrum of
`|omega_gamma|` is invariant. Across seeds it reads 3247, 901, 524, 395, 305,
202, 163, 152, 111, 89, with rank-wise coefficients of variation between 0.12
and 0.25 — consistent with the paper's report of four to six active factors.

## Seams verified against the reference

The port is pinned to pblasso at two levels, both in
`mlsynth/tests/test_dmlfm_against_pblasso.py`.

Design: every object the sampler consumes -- `y`, `X`, `A`, the treated blocks,
the unit and time codes, the group-break vectors and the time-sort permutation
-- matches element-wise to machine precision on this panel.

Draws: each conditional draw factors into a deterministic mean and covariance
plus a normal, and those halves match at 1e-10 -- `genXY`, `sampleN` including
its flat prior on the first coefficient, both `genTildeZ` forms, `genTildeTau`,
`getREfit`, `getFactorFit`, `samplePhi`, `iterGenAlpha`'s per-unit posterior,
and `iterGenXi`'s per-period posterior with its first-period case.

## Monte Carlo

`montecarlo_single_treated.csv` is the authors' own saved output from
`tempdata/sim_single_{X,r3,r8}.RData`: bias, standard deviation, RMSE, coverage
and runtime for `synth`, `gsynth` and the Bayesian DM-LFM across three designs
and six cases.

Both factor-model methods dominate `synth` everywhere (RMSE 1.8–3.7 against
3.4–6.2). Against `gsynth` the picture is narrower than the abstract suggests:
DM-LFM has lower RMSE in 6 of 18 cells and higher in 12, and its coverage is
closer to nominal in 7 of 18 while `gsynth` is closer in 10. The cells DM-LFM
wins are concentrated in the eight-factor design, matching the paper's own text
that the advantage appears "when the number of factors is large and each of them
produces relatively weak signals." Runtime is 11–80 seconds against 0.3–1.8.

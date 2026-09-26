# Hsiao and Zhou (2019): parametric, semiparametric and nonparametric counterfactuals

Reference: Cheng Hsiao and Qiankun Zhou, "Panel parametric, semiparametric, and
nonparametric construction of counterfactuals", *Journal of Applied
Econometrics* 34(4):463-481. Sections 2-6, Tables 6-8.

This is a replication spike, not a durable benchmark case. mlsynth has three of
the paper's seven methods; it has neither the CCE second stage nor the two
covariate variants, so there is no estimator here to pin. The study exists to
record what reproduces and what does not, so a later build starts from measured
ground.

## What is here

| file | what it does |
| --- | --- |
| `methods.py` | DGP6 and DGP7 (Equations 32-33) and the seven counterfactual constructions E1-E7 |
| `experiment.py` | one cell, printed beside the paper's row: `python -m benchmarks.studies.hsiao_zhou_counterfactuals.experiment --dgp dgp6 --T 40` |
| `plot_empirics.py` | observed against the fitted counterfactual, one panel per method (Table 9) |
| `standin.py` | drawn substitutes for the two smoking panels, and the resolver that picks between them and the paper's |
| `results/` | the runs and the figures behind the tables below |

DGP1-DGP5 are not ported. All five carry covariates, and the paper's step 1
estimates their coefficient by "Pesaran's (2006) CCE or Bai's (2009) method"
without saying which produced which cell. Section 6 then says that several DGPs
fail Pesaran's rank condition and that only Bai's slope is reported, so the
label "CCE" in Tables 1-7 names the second stage, not the first. That leaves the
covariate cells ambiguous in a way DGP6 and DGP7 are not: with no covariates
there is no coefficient to estimate and the ambiguity cannot arise.

## The seven methods, and which three collapse

E1 PCA is Bai (2009) factors from the control panel with Xu (2017)'s Steps 2-3.
E2 CCE regresses the treated unit on an intercept and every control over the
pre-period. E3 CPDA, E4 PDA and E5 PDAX select a subset by LASSO and refit. E6
MA and E7 MB are the mean- and mean-and-scale-corrected averages of E1-E5.

On a panel with no covariates, E3, E4 and E5 are the same estimator. CPDA's step
1 leaves `v~_t = y~_t` when there is no `X` to residualise, and PDAX's pool
`(y~_t, X_t)` is `y~_t`. `experiment.py` calls all three separately so that any
difference a separate LASSO path introduced would show, and they agree bit for
bit.

The paper reports them differing, and by different amounts on the two DGPs: 1 to
4 percent apart on DGP6, 16 to 35 percent apart on DGP7. The explanation is in
Section 6, which prescribes the stepwise method when `y_it` is stationary and the
random-split method (M2, `G = 2`) when it is not. A random split is seed
dependent and each method draws its own, so the three separate on the
nonstationary DGP and nearly coincide on the stationary one. That matches the
sizes observed.

## The row labelled MSE is a root mean squared error

Tables 1-7 carry MAB, MSE and MAP rows. Table 8 reprints the MSE rows under the
heading RMSE, and the values are identical: DGP1 at `N = 30`, `T = 40` reads
2.554 in both. Two measurements settle which label is right.

The ratio MSE/MAB is 1.354 at DGP1 and 1.344 at DGP5, whose scale is ten times
larger. A squared criterion's ratio to an absolute one grows with scale; a root
criterion's does not. And for a mean-zero error `E|e| = sqrt(2/pi) sigma`, so
RMSE/MAB is about 1.253 plus whatever bias contributes, which 1.35 is consistent
with and 2.0 is not.

Computing both on our own draws confirms it: the RMSE column lands beside the
paper's MSE column while the true MSE is roughly its square.

## What reproduces

100 replications, `r = 2`, 30 controls, ten post-treatment periods, minimum-norm
CCE. Mine / paper, MAB:

| method | DGP6 T0=30 | DGP6 T0=50 | DGP7 T0=30 | DGP7 T0=50 |
| --- | --- | --- | --- | --- |
| PCA | 1.850 / 1.717 | 1.961 / 1.712 | 1.933 / 1.873 | 2.032 / 1.800 |
| CCE | 4.270 / 3.728 | 0.769 / 0.693 | 6.682 / 4.906 | 0.760 / 1.186 |
| CPDA | 1.524 / 1.193 | 0.873 / 0.792 | 1.598 / 1.531 | 1.022 / 1.154 |
| PDA | 1.524 / 1.181 | 0.873 / 0.739 | 1.598 / 1.315 | 1.022 / 0.857 |
| PDAX | 1.524 / 1.234 | 0.873 / 0.792 | 1.598 / 1.347 | 1.022 / 0.884 |
| MA | 1.568 / 1.368 | 0.839 / 0.735 | 2.051 / 1.656 | 0.942 / 0.928 |
| MB | 1.804 / 1.461 | 0.837 / 0.726 | 2.173 / 1.726 | 0.956 / 0.950 |

Three structural findings reproduce, and they are the paper's substance.

The CCE reversal. When the donor pool is as wide as the pre-period is long the
unrestricted regression is rank deficient and CCE is the worst method by a wide
margin; when the pre-period is longer it is the best. Measured at 4.270 against
the paper's 3.728 in the first regime and 0.769 against 0.693 in the second,
with the ordering against every other method preserved in both. This is the
mechanism behind the paper's finding (v).

The ordering within each cell. PCA worse than the selection methods on the
stationary DGP; the averages between the best and the worst single method,
never the best. The paper's claim that model averaging is robust and never
optimal holds on our draws too.

The scale. A method that recovers the factor part exactly and none of the error
has MAB `E|u_1t|`, which measures 1.736 on these panels against the paper's PCA
column of 1.717, a ratio of 1.011. PCA is that method, so the anchor is
independent of any estimation choice and it pins the DGP's constants.

## What does not, and the two discrepancies it separates into

Levels sit 8 to 29 percent above the paper. The useful measurement is not the
level but the ratio between two methods inside one cell, which a common scale
factor cannot change. Taking PCA over PDA, since PCA is the only method with no
tuning parameter once `r` is fixed:

| cell | paper PCA/PDA | mine | ratio | regime |
| --- | --- | --- | --- | --- |
| DGP6 `T0=30` | 1.454 | 1.214 | 0.835 | 30 donors, 30 pre-periods |
| DGP7 `T0=30` | 1.424 | 1.210 | 0.849 | 30 donors, 30 pre-periods |
| DGP6 `T0=50` | 2.317 | 2.246 | 0.970 | 30 donors, 50 pre-periods |
| DGP7 `T0=50` | 2.100 | 1.988 | 0.947 | 30 donors, 50 pre-periods |

The shape reproduces to 3 to 5 percent wherever the pre-period is longer than
the donor pool, and is 15 to 17 percent off where the two are equal. So there
are two discrepancies, not one, and they separate by regime:

1. A level factor present in every cell and near uniform across methods.
   Excluding CCE, the median of mine over the paper is 1.143 at DGP6 `T0=50`
   (range 1.102 to 1.181) and 1.072 at DGP7 `T0=50` (range 0.886 to 1.193).
   Seven methods moving together by one factor is what a difference in the
   error scale looks like.
2. A shape factor of about 1.20, confined to the cells where the donor pool is
   as wide as the pre-period is long. Every method-side candidate for it was
   tested and cleared.

Neither is resolved to a mechanism. The level factor cannot be settled without
the authors' code, because the anchor that would settle it is confounded (see
Link B). The shape factor is confined to a regime whose selection step the
paper specifies only loosely.

## The ladder

Chain:

```
MAB  <-  aggregation  <-  per-period error  <-  counterfactual
     <-  method choice (r, penalty, rank handling)  <-  panel draw  <-  DGP constants
```

### Link B, the panel: one fault, found and fixed

The first implementation built Equation 33's neighbour terms with `np.roll`,
which wraps unit 1's `v_{i-1}` onto unit `N`. That hands the treated unit two
error components shared with the donor pool where the equation gives it one, and
every method that regresses on controls can predict the extra one. PDA's MAB
read 0.924 before the fix and 1.524 after, a factor of 1.65, and 0.924 is below
anything the design allows.

Five candidates at this link, cleared against the `E|u_1t|` anchor (paper 1.717):

| candidate | anchor | vs paper |
| --- | --- | --- |
| `sigma2_i ~ 0.5(chi2(1) + 1)`, own coefficient 2, no wrap | 1.736 | 1.011 |
| `sigma2_i ~ 0.5 chi2(1) + 1` | 2.167 | 1.262 |
| `sigma2_i = 1` | 1.758 | 1.024 |
| `sigma2_i ~ chi2(1)` | 1.531 | 0.892 |
| own coefficient 1 instead of `1 + b^2` | 1.091 | 0.635 |
| `np.roll` wrap | 1.917 | 1.117 |

The clearing has power against four of the five alternatives, which move the
anchor by 12 to 37 percent. It has none against `sigma2_i = 1`: homogeneous and
heterogeneous variances give anchors 1.3 percent apart, because `E|u|` depends
on `E[sigma]` and the two laws have nearly the same one. That candidate is not
separated here, and at 1.3 percent it cannot account for the residual gap
either way.

The anchor has a deeper limitation, and it is the reason the level factor stays
open. It reads the paper's PCA column against `E|u_1t|` computed on our own
draws, and those two agree to 1.1 percent -- but that agreement admits two
readings. Either the error scale is right and the paper's PCA sits at the
perfect-factor-recovery bound, or the error scale is about 14 percent too large
and the paper's PCA sits about 13 percent above a correspondingly smaller bound.
Our own PCA is 1.813 against a bound of 1.736, 4 percent above it, so a PCA
carrying estimation noise is the more ordinary of the two readings and it is the
one that implies our scale is too large. A single number cannot separate them.
Separating them needs a second anchor that depends on the error scale
differently from `E|u|` -- the paper reports none, and the ratio measurements
above are scale free by construction, so they cannot do it either.

### Link C, the method: CCE's rank handling, and a rejected hypothesis

The obvious candidate was that R's `lm()` drops aliased columns where
`numpy.linalg.lstsq` returns the minimum-norm solution, so the two disagree
whenever the donor pool is at least as wide as the pre-period. Measured over 200
draws:

| cell | paper MAB / RMSE | minimum norm | drop aliased |
| --- | --- | --- | --- |
| DGP6 `T0=30` | 3.728 / 9.187 | 4.601 / 10.998 | 32.433 / 344.981 |
| DGP6 `T0=50` | 0.693 / 0.895 | 0.713 / 0.912 | 0.713 / 0.912 |
| DGP7 `T0=30` | 4.906 / 7.932 | 4.678 / 7.987 | 13.342 / 75.454 |
| DGP7 `T0=50` | 1.186 / 1.540 | 0.750 / 0.970 | 0.750 / 0.970 |

The hypothesis is rejected, and by a wide margin: dropping aliased columns
leaves an interpolating fit whose extrapolation is 30 times worse than anything
the paper reports. Minimum norm is the match, and on DGP7 at `T0 = 30` it lands
within 0.7 percent of the published RMSE. The two agree exactly once the
pre-period is longer than the pool, which is the design where neither is
rank deficient, so that pair of cells had no power to separate them and is not
evidence for either.

CCE is also not estimable at 100 replications. Its Monte Carlo standard error
reached 2.201 on an RMSE of 25.166 in one run of the same cell that measured
7.987 over 200 draws; `max|e|` is 93. Any CCE cell quoted from a short run,
including the paper's at `R = 1000`, carries more uncertainty than its three
decimals suggest.

### Link C, the method: four more candidates, all cleared

120 replications per cell. MAB for PDA under each candidate, against the
paper's PDA column:

| candidate | DGP6 `T0=30` | DGP6 `T0=50` | DGP7 `T0=30` | DGP7 `T0=50` |
| --- | --- | --- | --- | --- |
| paper | 1.181 | 0.739 | 1.315 | 0.857 |
| cross-validated penalty | 1.636 | 0.896 | 1.721 | 1.066 |
| `alpha = 0.02` | 1.773 | 0.962 | 2.007 | 1.025 |
| `alpha = 0.05` | 1.698 | 1.042 | 1.812 | 1.093 |
| `alpha = 0.1` | 1.667 | 1.128 | 1.785 | 1.187 |
| `alpha = 0.2` | 1.633 | 1.202 | 1.741 | 1.252 |
| `alpha = 0.4` | 1.524 | 1.244 | 1.686 | 1.333 |
| `alpha = 0.8` | 1.500 | 1.364 | 1.598 | 1.465 |
| M2 random split, `G = 2` | 1.596 | 1.295 | 1.704 | 1.363 |
| best available | 1.500 | 0.896 | 1.598 | 1.025 |

The penalty candidate is rejected, and this was the hypothesis the spike started
from. The paper's value lies below the entire penalty path in all four cells, by 27,
21, 22 and 20 percent against the best setting available. The grid has power and its
optimum moves where it should: the sparsest penalty wins where the pre-period is
as short as the pool is wide, the densest wins where the pre-period is longer,
and cross-validation beats every fixed setting in the second regime. So the path
is behaving, and no point on it reaches the paper.

The random-split selection is rejected as an explanation for the level: 1.596
against 1.636 where the design is at the boundary, and actively worse (1.295
against 0.896) where it is well posed, which is consistent with the paper
prescribing it only for the wide-pool case. It does explain a different thing --
why the paper's CPDA, PDA and PDAX differ at all on a covariate-free DGP, since
each draws its own split.

The factor count is cleared. Cross-validation selects `r = 5` on every draw,
which is the search cap, and PCA's MAB moves from 1.813 at `r = 2` to 1.847,
about 2 percent. PCA's gap to the paper is 5.6 percent, so `r` cannot carry it.

The aggregation form is rejected. Reading MAB as the absolute value of the mean
error instead of the mean absolute error gives 0.577, 0.260, 0.920 and 0.483
against the paper's 1.181, 0.739, 1.315 and 0.857 -- too small by half, and in
the wrong direction.

### Link D, the replication count: cleared, with the number

100 replications give a Monte Carlo standard error of 0.045 on PDA's 1.524. The
gap to the paper's 1.181 is 0.343, or 7.6 standard errors. Sampling noise does
not account for it for any method except CCE.

### Where the ladder stops

It does not bottom out, and that is the result. Every candidate at Link A, Link
C and Link D is cleared or rejected, Link B's one fault was found and fixed, and
the two discrepancies that remain are separated and bounded but not attributed:

| cause | size | status |
| --- | --- | --- |
| 1. level, all seven methods together | 1.07 to 1.14 | open; the anchor that would settle it is confounded two ways |
| 2. shape, confined to pool width equal to pre-period length | about 1.20 | open; every method-side candidate tested and cleared |
| 3. `np.roll` wrap in Equation 33 (ours) | 1.65 | fixed |

Cause 3 confirms both bottom questions: with it absent the failure did not occur
(PDA moved 0.924 to 1.524), and with it corrected nothing reintroduced it.
Causes 1 and 2 answer the first question yes, so the ladder continues, and it
continues past what this repository can measure. Settling cause 1 needs a second
anchor whose dependence on the error scale differs from `E|u_1t|`'s, and the
paper reports none. Settling cause 2 needs the selection step the paper
specifies as "the stepwise method" or "the random split method" without fixing a
seed, a stopping rule for the stepwise variant, or which of the two runs at
`N = 30`.

Neither is reachable without the authors' code, which the paper does not ship.

## Section 7, the empirics

Run by `run_empirics.py`, which works out of the box.

The turnout arm is a replication whatever else happens: its panel is already
here as `basedata/xu_edr_turnout.parquet`, which is the paper's `turnout.csv`
row for row. The two smoking panels are not here, and they carry the covariates
the paper substitutes for Abadie's (poverty rate and educational attainment,
since price, beer and per-capita GDP are themselves treated). With nothing
configured those two are drawn by `standin.py` instead, and the run says so and
prints no published comparison, because a drawn panel has nothing to compare
against. `MLSYNTH_HZ_DATA` pointed at the paper's directory gives the real
replication, which is what the numbers below are from.

The stand-in is drawn to behave like the panel it replaces, not to copy it.
One of its three covariates is near-constant and carries a large coefficient,
which is what `lnincome` is, and that is not decoration: it is the feature that
made Bai's iteration stall, so a stand-in without it would let the regression
test below pass for a reason unrelated to the defect.
`benchmarks/tests/test_hsiao_zhou_standin.py` holds it to producing a panel on
which the old scheme still loses.

The treated series check out first, which is what licenses the rest. California
is unit 1 in `smoking.csv`, whose 1989 and 2000 cigarette sales are 82.4 and
41.6, matching Table 9's Actual column to 0.05; the health-expenditure actuals
match Table 10 to 0.0007; all six turnout actuals match Tables 11-16 to 0.10.

### Which beta the empirical columns use

This is the ambiguity the simulation tables leave open, and Section 7 settles
it. CCE's MAB on Table 9 is 8.616 against the published 9.120 when beta comes
from Pesaran's Equation 16, and 16.31 when it comes from Bai's method -- a ratio
of 0.94 against 1.79. The empirical columns use Pesaran's CCE estimator. It
stays open for Tables 1-7, where Section 6 says only Bai's slope is reported.

### Table 9, cigarette consumption

MAB, mine over the paper, 38 controls, `T0 = 19`:

| method | mine | paper | ratio |
| --- | --- | --- | --- |
| SCM | 19.514 | 18.500 | 1.05 |
| PDA | 14.004 | 14.300 | 0.98 |
| PDAX | 15.231 | 16.200 | 0.94 |
| CCE | 8.616 | 9.120 | 0.94 |
| PCA | 6.987 | 7.460 | 0.94 |
| MA | 9.904 | 8.330 | 1.19 |
| MB | 10.577 | 7.270 | 1.45 |
| CPDA | 4.934 | 9.560 | 0.52 |

The SCM column is `mlsynth.VanillaSC` outcome-only, and it reproduces to 5
percent, with ten of the twelve annual counterfactuals inside 2.0 of the
published path. Six of the eight columns land between 0.86 and 1.19.

### How much of that agreement is the LASSO's tuning

The three selected columns sit on 19 pre-period observations against 38
donors, and the penalty is chosen by cross-validation on those 19 points. The
fold count and the seed are both free, and neither is pinned by the paper.
Sweeping 32 settings that are all defensible:

| column | min | median | max | paper | spread over the published value |
| --- | --- | --- | --- | --- | --- |
| PDA | 8.17 | 14.00 | 17.81 | 14.30 | 0.67 |
| PDAX | 15.23 | 15.40 | 16.74 | 16.20 | 0.09 |

PDA's ratio of 0.98 is the median of that sweep, so the setting this study
ships is a fair one, and only 38 percent of the settings land within 10 percent
of the published value. The agreement is therefore not evidence that the
construction is right; it is one draw from a spread two thirds as wide as the
quantity being reproduced. The same caution applies to CPDA below, and not to
SCM, CCE or PCA, which select nothing.

PDAX is steady by comparison because its selection is standardized. Its pool is
the one design here whose columns are different measurements -- control outcomes
beside a log income -- and the LASSO's penalty has no scale-free meaning across
them. Without standardizing, the covariates cannot pay the penalty at any
weight, and PDAX returned PDA's path to three decimals: a column that was not
running the method it was named for. Table 10 was degenerate in the same way,
at a pool spread of only 2 to 1, which is why the criterion here is whether the
columns share a measurement and not how far their numbers spread.

PCA reproduces in magnitude and not in sign, and separating those two took an
outside implementation. The first version of this study shipped a defective
Bai (2009) step and read 18.44 against the paper's 7.46; with that corrected it
reads 6.99, a ratio of 0.94. See the Bai step below. What survives is the sign:
the mean signed effect is -6.70 here against Table 9's +7.51, so this
replication says Proposition 99 lowered consumption and the published PCA
column says it raised it.

The published PCA column is also the only one of the paper's eight whose sign
differs from its own neighbours. Its CCE reads -9.07 and its PDA -14.30, both
negative, both of which this replication matches at -8.62 and -14.00. So the
disagreement is not between two implementations of a method, it is between one
column of Table 9 and the rest of Table 9. The paper's footnote 8 records the
result as counterintuitive and keeps it deliberately.

### CPDA, the column the paper's own description does not pin

CPDA sits at 0.52 and it is the one column where the gap is not a defect here.
Equations 11-15 fix everything except which donors enter, and the paper says
only that the subset "can be chosen using a model selection criterion as in
Hsiao, Ching, and Wan (2012), or the LASSO method (Tibshirani, 1996), as
suggested by Li and Bell (2017)". CCE is the same construction with every donor
kept, and it reproduces at 0.94 on the same beta and the same residuals, so the
selection is the only thing between them.

Six readings of that sentence, on the Table 9 panel:

| selector | donors kept | nested LOO error | MAB | ratio to 9.56 |
| --- | --- | --- | --- | --- |
| LASSO, CV penalty (what runs here) | 9 | 2.167 | 4.93 | 0.52 |
| LARS path, AICc, standardized | 6 | 2.227 | 3.79 | 0.40 |
| LARS path, AICc | 5 | 2.559 | 5.04 | 0.53 |
| LARS path, BIC, standardized | 15 | 2.653 | 13.78 | 1.44 |
| LASSO, CV penalty, standardized | 14 | 2.932 | 10.92 | 1.14 |
| LARS path, BIC | 15 | 3.849 | 14.04 | 1.47 |

The estimates span a factor of 3.7 and the published 9.56 lies inside them, so
the paper's value is reachable and is not singled out by anything measurable
from the pre-period. The middle column is the test: it is leave-one-pre-period-
out error with the selection repeated inside every fold, so no held-out period
informs its own prediction. The lowest such error belongs to the selector
giving 4.93, and the two landing nearest the published value score worst on it.
Choosing between them by how close the answer comes to 9.56 would be fitting to
the post-period, which is the one thing a counterfactual may not do.

A leaky version of that check ranked them the other way round. Selecting once
on the full pre-period and then refitting inside each fold made the
standardized LASSO look both best on error and closest to the paper, which is
the reading this study nearly adopted. The leak matters because the selection
is what the fold is meant to test, and holding it fixed tests nothing.

So CPDA is recorded at 0.52 and not tuned toward 9.56. What the spread says
about a future CPDA estimator is in `docs/` terms a design constraint: the
selector has to be an explicit choice with a documented default, and the result
has to carry its sensitivity, because at `T0 = 19` against 38 donors the
selector decides the answer more than the construction does.

### The Bai (2009) step, and the two references that settled it

Both `beta_bai` and `beta_cce` were hand-rolled here and neither was validated
against anything. One was wrong.

Bai's estimator is the argmin of `||Y - X b - F L'||^2`, so the objective
decides. On the 38-state control panel, concentrating out the factors:

| beta from | lnincome | objective at r=2 |
| --- | --- | --- |
| PCA2 from a zero start (first shipped here) | -0.53 | 42,007 |
| PCA1 from a zero start | -6.71 | 41,585 |
| PCA2 from pooled OLS | 19.74 | 37,610 |
| PCA1 from pooled OLS (what runs now) | 61.68 | 35,640 |
| xtife 0.1.5 | 73.34 | 36,212 |

Bai (2009) gives two iteration schemes and Hsiao, Shi and Zhou (2022) write
both out: PCA1, their Equation 54, projects the estimated factors out of the
regressors as well as the outcome; PCA2, their Equation 56, subtracts the
common component and regresses on raw `X`. This study shipped PCA2 from a zero
start, which is the worst cell in the table.

Their Table 1 measures the scheme half independently at 1000 replications on
Bai's own DGP1, with a pooled-OLS start and 100 iterations: PCA2's bias holds
near 0.13 whatever `N` and `T` are, and its empirical size reaches 100 percent
against a 5 percent nominal. On this panel the start matters more than the
scheme, but both matter, and `beta_bai` now runs PCA1 from several starts and
keeps the lowest objective.

The defect reached the PCA column of Tables 9 and 10 and the MA and MB averages
that include it. It did not reach Tables 11-16: the turnout covariates are 0/1
dummies, and on those panels both schemes converge to the same point from any
start, identical to four decimals with identical objectives. Nor did it reach
the simulations, whose panels carry no covariates, so `pca_counterfactual`
takes an SVD directly and never calls `beta_bai`.

`beta_cce` was clean. An independent implementation of Pesaran's Equation 16 --
the `Estimation_RCCE` routine in the replication package for "Panel Sample
Selection Model with Interactive Effects" -- reproduces it to 0.00e+00. That
package demeans the cross-sectional averages before forming the projector where
Hsiao and Zhou's printed Equation 16 does not, and the choice is not cosmetic:
demeaning moves Table 9's CCE column from 8.62 to 15.24 against a published
9.12, so the printed convention is the one that reproduces.

Neither error had an internal signature. The fit looked plausible, the
counterfactual looked plausible, and the objective is not something a single
implementation can check itself against. `benchmarks/tests/test_hsiao_zhou_study.py`
now asserts the property, not any coefficient: the returned beta is not
improvable from a pooled-OLS restart, the answer does not depend on the start,
and it recovers the truth on Bai's DGP1.

### Table 10, personal healthcare expenditure

Every method lands within a factor of 0.47 to 1.31, and the substantive claim
reproduces: the paper reports hardly any effect, with a 2000 actual of 9.755
against a model-average counterfactual of 9.735, and ours is 9.756 against
9.733. A 38-donor pool against a nine-period pre-period is 4.2 times as wide as
the sample.

The PDAX column is the one that moved, from 0.080 to 0.029, when its selection
was standardized. Before that it returned PDA's 0.080 exactly, which read as a
1.31 agreement and was the degenerate case described under Table 9: the
covariate never entered, so the column was PDA under another name. The
corrected figure is further from the published 0.061, and it is the one
produced by the method the column is named for.

Here the paper's own numbers are the interesting part. Its PDA and PDAX columns
are identical in eleven of twelve rows and both MABs read 0.061 -- which is
what a PDAX that never selects its covariate looks like, and is exactly the
signature this study had before the fix. The 1989 row differs (8.794 against
8.974) in a way that reads as a transposition. That is an inference from two
published columns and not a claim about the authors' code, but it is the
simplest account of two constructions with different predictor sets returning
the same eleven numbers. If it is right, the published 0.061 is a PDA value and
no PDAX implementation should be expected to reproduce it.

On a nine-period pre-period the column is in any case barely identified: swept
over the fold counts and seeds that fit nine observations, the unstandardized
selection ranges 0.036 to 0.080 while the standardized one returns 0.029 at
every setting.

### Tables 11-16, turnout

Mean effect per state, model-averaged:

| state | MA mine | paper | MB mine | paper |
| --- | --- | --- | --- | --- |
| ME | 4.11 | 3.22 | 3.84 | 2.85 |
| MN | 3.23 | 4.49 | 3.23 | 4.52 |
| WI | 5.82 | 8.39 | 5.97 | 8.59 |
| WY | 7.23 | 7.20 | 7.95 | 8.22 |
| ID | -4.20 | -0.42 | -3.96 | 0.64 |
| NH | 6.44 | 8.04 | 6.70 | 8.28 |

Per-state levels differ by up to 3.8 points and Idaho is the one sign
disagreement, near zero in the paper and clearly negative here; it is the
weakest state in both. The headline reproduces on both waves. The paper's claim
is that model averaging puts the first wave below Xu (2017)'s 7.2 percent and
the second above his 2.17, compressing the gap he reports between them. Ours
gives 4.39 and 3.16 against Xu's 7.2 and 2.17, so both inequalities hold and
the compression is larger here than in the paper: Xu's spread of 5.03 becomes
about 2 in the paper's own columns and 1.23 in ours.

## What this means for a build

Three conclusions carry over, and they sharpen the recommendation from the
paper review.

CPDA, PDA and PDAX are one estimator on any panel with no covariates. So the
covariate channel is not an enhancement to those methods, it is the entire
difference between them. `PDAConfig` has no `covariates` field today, which
makes it the prerequisite for all three, not one option among them.

CCE is the one genuinely distinct construction, and it is either the best method
or the worst by a wide margin depending on a condition that is known before
fitting: whether the pre-period is longer than the donor pool. Measured here at
0.769 against PDA's 0.873 when it is, and 4.270 against 1.524 when it is not.
That is a diagnosable regime, not a tuning risk, and an estimator offering CCE
should refuse or warn on the wrong side of it.

Do not pin any cell where the pool is as wide as the pre-period. Those are the
cells whose shape does not reproduce, whose selection step the paper leaves
underspecified, and where CCE's own Monte Carlo error is large enough that three
decimals overstate what the table knows.

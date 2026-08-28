# Hsiao & Zhou (2024) reference bundle — German reunification

Path-A referent for the empirical section of Hsiao, C. and Zhou, Q. (2024),
"Panel treatment effects measurement: Factor or linear projection modelling?",
Journal of Applied Econometrics 39(7):1332–1358,
[10.1002/jae.3081](https://doi.org/10.1002/jae.3081). Replication package:
[10.15456/jae.2024145.0725725591](https://doi.org/10.15456/jae.2024145.0725725591).

## Why the referent is digitised

The replication package contains data only — `readme.pdf`, `readme.txt`,
`repgermany.dta`, `repgermany.xlsx` — and no code. Its `repgermany.dta` is
byte-identical (md5 `83bb9922938ffa27846f4a7577f625dc`) to the copy already in
`basedata/`, so the input is settled and the estimator is not.

Section 7 reports its result as Figures 1 and 2 and prints no number: no ATT,
no interval endpoint, no table. Both figures are vector graphics, so the plotted
series are recoverable exactly. `digitise_figures.py` pulls the polylines out of
the PDF and calibrates them.

Figure 1 carries three 44-point paths over 1960–2003 (black observed, blue LP,
red FB); Figure 2 carries six 13-point paths over 1991–2003 (LP and FB effects
with their 95% bounds). Figure 1 is self-calibrating: its black path is West
Germany's observed log real GDP per capita, which is in `repgermany.dta`, so
regressing PDF y-units on the known series both fixes the axis map and confirms
the series identification. That regression returns max |residual| = 1e-5 log
points and R² = 1.00000000. Figure 2 is calibrated off its labelled ticks and
then checked against Figure 1: the effects it plots equal the observed series
minus Figure 1's counterfactuals to within 0.0004 log points, on both methods.
Two independent calibrations agreeing to 4e-4 is what makes these CSVs usable
as a referent.

## The specification

Log real GDP per capita, 17 OECD countries, 1960–2003, West Germany treated.
Pre-period 1960–1990 (T = 31), post-period 1991–2003 (m = 13), so N = 17 and
T > N. Note 13 sets the factor count by Bai & Ng (2002) with a maximum of 5 and
reports the criterion returning 5; IC_p2 on this panel does return 5. Note 14
records that covariates are dropped for missingness, leaving a pure factor
specification on the outcome. Note 15 gives both interval formulas.

## Targets

| quantity | value |
|---|---|
| LP effect, 1991 / 2003 | +0.030 / −0.108 |
| LP mean effect 1991–2003 | −0.0565 |
| LP 95% half-width, mean | 0.0348 |
| LP interval excludes zero | 11 of 13 years |
| FB effect, 1991 / 2003 | −0.054 / +0.344 |
| FB mean effect 1991–2003 | −0.2347 |
| FB 95% half-width, mean | 1.2997 |
| FB interval excludes zero | 0 of 13 years |

## The FB path is not reproducible

The LP column replicates. A least-squares projection of the treated series on
the 16 controls with an intercept lands 0.030 log points from the plotted blue
path, and `PDA(method="fs")` lands 0.0086 from it.

The red path does not. Its pre-period fit misses the observed series by RMSE
0.6230 log points — an in-sample miss of about 86% for a five-factor model on a
17-series panel. A principal-component fit of that panel does not behave that
way: taking the top r components gives a pre-period RMSE of 0.0421 at r = 1,
falling to 0.0115 at r = 5, and the counterfactual is invariant to which of the
paper's two normalisations (eqs. 17–18 or 19–20) is used, so the branch choice
cannot explain the gap. A search over all 6188 five-subsets of the panel's
principal components gets no closer than RMSE 0.5633 to the plotted path,
against 0.5700 for the specified top five — the search buys essentially nothing,
so this is not a component-selection bug. Nor is it a wrong treated unit: the
path matches no country's observed series and no unit's FB counterfactual, the
closest being USA at RMSE 0.558.

Two correct implementations agree with each other and disagree with the figure.
A faithful port of eqs. (30)–(31) and `FMA` (Li & Sonnier 2023, a different
published factor method) produce post-period counterfactuals 0.0182 apart, and
sit 0.4182 and 0.4105 from the plotted red path. Their pre-period fits are
0.0115 and 0.0107 against the observed series, so the figure's in-sample miss
is 54 times either.

The paper's own note 15 is the reason this matters for the reported inference.
SE²_t,FB opens with σ̂²₁, the mean squared in-sample residual, so the plotted
interval half-width of 1.2997 is close to what 1.96 × 0.6230 = 1.2210 implies:
the formula is applied correctly to a counterfactual that misses badly. That
is the whole of Section 7's headline — that the FB intervals cover zero while
the LP intervals do not. With σ̂²₁ from a correct factor fit the FB half-width
is 0.047, not 1.30.

The authors published no code, so the discrepancy cannot be traced further.

# Augmented DID, demonstrated before it is built

Reference: Kathleen T. Li and Christophe Van den Bulte (2022), "Augmented
Difference-in-Differences", *Marketing Science*, DOI
[10.1287/mksc.2022.1406](https://doi.org/10.1287/mksc.2022.1406). Equation 2.4 is
the estimator, Table 1 its identifying assumption, Proposition 3.1 the limit
distribution, Appendix A.1 the variance estimator.

This is a replication spike. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.adid_replicate.run --both
```

## The method

ADID fits

    y_1t = delta_1 + delta_2 * ybar_co,t + e_1t,    t = 1, ..., T_1

by least squares on the pre-period and predicts the counterfactual as
`delta_1 + delta_2 * ybar_co,t`. DID is the same construction with the slope held
at one. That single freed parameter is the whole method: where DID assumes the
treated unit would have run parallel to the control average, ADID assumes it
would have run parallel to a slope-adjusted control average.

The inference is closed form, which is what it offers over SC, MSC and HCW.
Proposition 3.1 gives `sqrt(T_2)(ATT_hat - ATT) / sqrt(Sigma_hat) -> N(0,1)` with
`Sigma_hat = Sigma_1 + Sigma_2`, the first term carrying the pre-period
estimation error and the second the post-period idiosyncratic error. For short
panels the paper recommends `t_{T_1 - 2}` in place of the normal.

## What matched

The port ingests through `dataprep` and is compared against the authors' own
script, run live under Octave. Both splits their script offers, every scalar it
prints and both fitted paths:

| quantity | pre-period 83 | pre-period 90 |
| --- | --- | --- |
| `delta1`, `delta2` | 3.5e-12, 8.0e-14 | 2.4e-12, 4.8e-14 |
| `adid_att`, `adid_att_pct` | 3.9e-12, 1.4e-14 | 2.6e-12, 2.5e-13 |
| `adid_sigma2`, `adid_omega1`, `adid_omega` | 0.0, 4.7e-10, 0.0 | 5.8e-11, 1.9e-09 , 3.5e-10 |
| `adid_std_stat` | 1.4e-13 | 1.2e-13 |
| `did_intercept`, `did_att`, `did_att_pct`, `did_r2_pre` | 2.3e-13, 1.4e-12, 1.3e-13, 2.3e-13 | 4.5e-13, 4.5e-13, 4.1e-13, 4.5e-13 |
| ADID counterfactual, 110 periods | 7.3e-12 absolute, 1.0e-15 relative | 4.5e-12, 6.2e-16 |
| DID counterfactual, 110 periods | 9.1e-13 absolute, 2.8e-16 relative | 4.5e-13, 1.3e-16 |

Absolute differences, largest over the series for the paths. `adid_omega1` is
about 2.4e+06 in magnitude, so its 4.7e-10 is 2e-16 relative. The agreement is at
machine precision throughout.

## What did not match, and why it cannot

Their published Table 7 reports ADID ATTs of 946 for Boston (65 percent) and 705
for Columbus (72 percent). The shipped data returns 1067 (21.0 percent) at
`t1 = 83` and 889 (15.3 percent) at `t1 = 90`.

That is expected and is the authors' own doing. Their script's header says the
treated series is generated -- "We generate data using a factor model to fit each
control unit's data first" -- because the transactions belong to an eyewear
company that shared them under confidence. The data agrees: 69.7 percent of
control cells are exact multiples of 23.75, a quarter of the $95 price point the
paper names, against 0 percent of treated cells. The controls are real
aggregates; the treated column is simulated.

So Path A is not available for this application. The estimator is validated; the
published number is not reachable from anything shipped.

Nor can the file be assigned to a city. The script offers Boston at `t1 = 83` and
Columbus at `t1 = 90`, and ships with Boston uncommented while its `saveas` calls
name Columbus. Neither split leaves a level break that identifies it -- the
treated series trends upward throughout, and the ten-week means either side read
4061 against 4756 at week 83 and 4611 against 6605 at week 90, both consistent
with the trend alone. The spike reports both splits and claims neither.

## What the spike found on the way

mlsynth already computes ADID. `mlsynth/utils/pangeo_helpers/effects.py::_adid`
implements Equation 2.4 with Appendix A.1's variance, and reproduces the authors'
ATT to 6.6e-12 on their own data. It is reachable only through PANGEO's
geo-design pair aggregates: `FDIDResults` carries `.fdid` and `.did` and no
`.adid`, and no estimator exposes it for a panel.

The two implementations differ in one place, and it is a real choice, not a
discrepancy. Appendix A.1 gives `Sigma_2` in two branches, a HAC form truncated
at `l_1 = O(T_1^{1/4})` for serially correlated errors and `T_1^{-1} sum e^2` when
they are uncorrelated. The authors' script ships the second; `pangeo`'s
`_lr_variance` uses Newey-West. On this panel:

| variance of the residual | value | ADID standard error |
| --- | --- | --- |
| `mean(e^2)`, their script and the uncorrelated branch | 283,391 | 199.35 |
| Newey-West, `pangeo` and the general branch | 415,929 | 241.51 |

A 47 percent inflation, so the interval's width turns on which branch is the
default. Both are the paper's.

## Whether the method earns its place here

On this panel it does, and visibly. The fitted slope is `delta_2 = 3.51`, nowhere
near the 1 that DID imposes, and the two estimates are a factor of 3.2 apart:

| | ATT | pre-period R-squared |
| --- | --- | --- |
| DID | 3402 | 0.407 |
| ADID | 1067 | 0.833 |

This is the paper's DGP2 regime, where the treated unit's trend sits outside the
range of the controls' and DID's restriction is the binding error.

## Recommendation

Build, as a third fit on `FDIDResults` beside `.fdid` and `.did`, not as a new
top-level estimator. The ingredients are already FDID's: both methods build a
counterfactual from a donor average over a pre/post split, and ADID differs only
in freeing the slope. The engine exists and is verified; the work is exposure,
the choice of variance branch, and moving `_adid` out of `pangeo_helpers` into a
shared helper both callers use -- which, per `CLAUDE.md`, lands on its own branch
first.

Three things the build should not do.

It should not claim ADID on forward-selected donors. The 2x2 of {all donors,
forward-selected} by {slope one, slope free} has Li's two papers on the diagonal
and nothing published on the off-diagonal, so if that combination is exposed at
all it is mlsynth's own and its inference is unproven, not inherited from
Proposition 3.1.

It should not pin Table 7. Nothing shipped reaches it.

It should not leave `delta_2` unreported. The whole method is that one number, it
is unconstrained, and at 3.51 on this panel it is doing all the work; a reader
needs to see how far it sits from DID's 1.

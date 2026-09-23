# LPCA Path-A target — Feng, Section 6.1 (2012 Kansas tax cut)

A pre-build replication spike for local principal component analysis
([arXiv:2311.07243v1](https://arxiv.org/abs/2311.07243)), the estimator that
imputes a treated unit's counterfactual under a factor structure that may be
nonlinear. No estimator has been added to `mlsynth`; this bundle establishes
that the paper's empirical application reproduces before one is built, and
records what the reproduction turned up about the paper's comparison arm.

The panel is already in the repository. `basedata/kansas_taxcut.csv` is the
augsynth `kansas` dataset — 50 states, 105 quarters (1990Q1–2016Q1), log GDP
per capita, Kansas treated from 2012Q2 — which is the same panel Feng ships as
`kansas.rda`.

## What is here

The spike itself:

```
lpca_oracle.py     readable port of Algorithm 1 (K-NN matching + local PCA)
run_kansas.py      ingestion via dataprep, both arms, the tuning sweep
dump_results.py    regenerates results.json (the Kansas numbers)
simulation.py      the three Section 5 DGPs and the global-PCA baseline
run_simulation.py  regenerates simulation_results.json (Table 1)
results.json       generated
simulation_results.json  generated
```

```bash
python benchmarks/reference/lpca_kansas/dump_results.py
python benchmarks/reference/lpca_kansas/run_simulation.py 500   # ~45 min on 4 cores
```

And the captured reference run that `benchmarks/cases/lpca_kansas.py` pins
against, added once the estimator existed:

```
manifest.json      case metadata + the data generate.py checksums
reference.R        Feng's own knn.index / lpca / sel.r, run on this panel
reference.out      its verbatim stdout                    (generated)
reference.json     the parsed values the case pins against (generated)
provenance.json    versions, OS, git SHA, data checksums   (generated)
```

```bash
bash benchmarks/R/install_feng_lpca.sh
python benchmarks/reference/generate.py lpca_kansas
python benchmarks/run_benchmarks.py --case lpca_kansas
```

The two coexist by design. The oracle above is a readable port written to find
out whether the method reproduced at all; `reference.R` is the author's code,
and the benchmark case pins mlsynth against it -- the ATT to 4.0e-11 and the
sixteen-quarter counterfactual path to 6.4e-11, which is the reference's own
ten-decimal print precision.

## The estimator

Two steps (paper Section 3). Split the time index in two. On the first block,
find each unit's `K` nearest neighbours under the pseudo-max distance
`rho(x_i, x_j) = max_{l != i,j} |(1/p)(x_i - x_j)' x_l|`. On the second block,
take the `K`-neighbour submatrix, truncate its SVD at a data-chosen rank, and
read off the reconstructed row. The treated unit's post-treatment cells are set
to zero before any of this; Theorem 6.1 is the statement that doing so
perturbs the estimate by a vanishing amount when the post-period is short.

The reference is the author's own code,
[`Feng_LPCA_app_suppfuns.R`](https://github.com/yingjieum/Replication_NonlinearFactorModel_2023/blob/main/Application/Feng_LPCA_app_suppfuns.R)
(`knn.index` / `findknn` / `lpca` / `sel.r`), and every tuning constant comes
from `Feng_LPCA_app.R`: `K = round(n^(2/3))` = 14, the first 40 differenced
quarters for matching, at most 3 local components. The rank rule keeps
components while consecutive singular values are separated by more than
`log log K`, floored at one.

## What reproduced

Preprocessing follows the script: mask the treated cells, first-difference log
GDP per capita in percent, centre each quarter across states, zero the treated
row's 16 post-treatment cells.

| Quantity | This port | Paper v1 |
|---|---|---|
| Observed minus LPCA counterfactual, mean post growth | −0.5306 pp | −0.53 pp |
| Post quarters with observed below the LPCA path | 9 of 16 | 9 of 16 |
| Observed minus SC counterfactual, mean post growth | +0.1948 pp | +0.19 pp |

All three land on the paper. The third one needs an explanation, because the
straightforward reading of the current replication script gives −0.3340 pp.

## Path B: Table 1 reproduces

Section 5 is the check the build should rest on, and it is untouched by
everything below: Table 1 is identical in the November 2023 and July 2024
versions. Three DGPs at `n = p = 1000`, half the columns for matching, the
`K` grid 49/99/149, and a global-PCA baseline whose factor count comes from an
eigenvalue ratio on doubly demeaned data.

500 replications against the paper's 2 000. Across the 48 cells the median
disagreement is 0.83 Monte Carlo standard errors, 43 land within 2 and 47
within 3. The maximum-absolute-error row:

| | K=49 | K=99 | K=149 | GPCA |
|---|---|---|---|---|
| Model 1 | 0.683 / 0.680 | 0.963 / 0.937 | 2.208 / 2.203 | 1.141 / 1.148 |
| Model 2 | 0.602 / 0.599 | 0.633 / 0.636 | 0.703 / 0.706 | 0.866 / 0.870 |
| Model 3 | 0.475 / 0.475 | 0.461 / 0.461 | 0.461 / 0.461 | 0.469 / 0.470 |

Cells are `ported / paper`; `simulation_results.json` carries every cell with
its standard error.

Each of the paper's qualitative claims holds. Local PCA beats global PCA on
Models 1 and 2 at the two smaller neighbourhoods and ties it on Model 3, which
is what "smaller or at least comparable" describes — including the detail that
global PCA edges local PCA by 0.006 at `K` = 49 on Model 3, an ordering the
port reproduces. The advantage widens with the severity of the nonlinearity:
Model 1 at `K` = 49 is 40 percent below the baseline, Model 2 is 30 percent
below, Model 3 is level. And Model 1 at `K` = 149 blows up to 2.208 against a
baseline of 1.141, the paper's own warning that too large a neighbourhood
destroys the local approximation where the surface bends hardest.

One residual has no explanation. Model 1's `q_alpha = .9` prediction error
comes in low — 0.066 against 0.076 at `K` = 49 (3.9 standard errors) and 0.061
against 0.068 at `K` = 99 — and those two cells share their draws and
evaluation units, so it is one effect and not two. The model's surface is
symmetric in the latent variable, and the paper's `q_alpha = .1` and
`q_alpha = .9` rows are near-identical as that symmetry implies, while this
port's are not. Nothing else in the table shows the asymmetry, no claim turns
on the cell, and local PCA still beats the baseline there by a wide margin
(0.066 against 0.107). Recorded as open.

## The comparison arm changed between paper versions

The paper reports that the synthetic control predicts post-treatment growth
0.19 points below observed Kansas, against local PCA's 0.53 points above —
opposite signs, which is what makes the sentence "By contrast" work and what
motivates the remark that the SC answer "is not plausible given the poor fit of
SC in the pre-treatment period."

That number comes from a defect the author has since fixed. The replication
repository has two uploads of the application script, and they differ in one
token on one line:

```r
# 2023-11-12, the version the v1 paper reports
sc.fit <- sc.pred(x, ind=id, post=(T0:p));            sc.fit <- sc.fit[(p1+1):p,]
# 2024-08-01, current
sc.fit <- sc.pred(x, ind=id, post=(T0:p)) + col.mean; sc.fit <- sc.fit[(p1+1):p,]
```

The LPCA line carries `+ col.mean[(p1+1):p]` in both versions, and the
supporting functions are identical apart from a date in a comment. So in v1 the
synthetic control path stayed in the quarter-centred space while the observed
Kansas series it was subtracted from did not. The quantity being reported was
the treatment effect minus the average of the quarter means it was never put
back on, and over the 16 post-treatment quarters that average is +0.5288 pp.
Adding it back moves +0.1948 to −0.3340.

Both conventions are in `results.json` (`sc.gap_demeaned`, `sc.gap_recentred`).
The v1 value reproduces to 0.005 pp, which is how the defect was identified.

Corrected, the two estimators agree on sign. Local PCA puts the counterfactual
0.53 points above observed Kansas and the synthetic control puts it 0.33 points
above: both say the tax cut cost growth, and they differ by 0.20 points of
magnitude. The published contrast between them does not survive the fix.

The current version of the paper confirms this. Feng (2024), dated 31 July
2024, Section 6.1: "the average growth rate over the entire post-treatment
period for the counterfactual Kansas ... is 0.53 percentage points higher than
that of the observed Kansas. By contrast, SC yields a smaller effect, with an
average growth rate 0.33 percentage points higher than that of the observed
Kansas." Both arms now match this port to the printed precision — −0.5306 and
−0.3340 — and the sign of the SC arm has flipped from the v1 text by exactly
the omitted column mean. The surrounding argument changed with it: where v1
called the SC answer implausible "given the poor fit of SC in the pre-treatment
period", the 2024 text says instead that SC "may be more vulnerable to large
temporary shocks in a few pre-treatment periods, as the weights used for SC
prediction are constructed to make the pre-treatment fit as close as possible
to the observed sequence." The fit claim is gone, which is consistent with what
the measurement below shows.

## Where the paper's number could have come from, and where it could not

The port was checked before the provenance was, and two instruments ruled out
the obvious explanations.

The simplex solver is right. On the same panel in levels it reproduces
`benchmarks/cases/ascm_kansas.py`'s classic-SCM rung — itself cross-validated
against live augsynth — to 1.3e-07 (−0.029435).

Re-specification cannot get there either. Nine variants of the fitting window
and outcome space (growth versus levels, the full pre-period versus the PCA
block, windows shifted by one quarter) all land between −0.23 and −0.55.
Constraining the weights to hit the published number requires a pre-treatment
fit 156 percent worse in sum of squares than the optimum, and the optimum here
is unique: holding the pre-period fit at its best value, the achievable
post-treatment mean is the single point +0.5026.

## Two findings that bear on a build

The pre-treatment fit argument runs the other way. The paper discounts the
synthetic control for fitting Kansas poorly before treatment. On the window
where both arms predict — the 48 differenced quarters from the end of the
matching block to treatment — local PCA's RMSE is 0.866 pp and the synthetic
control's is 0.624 pp. Both are in-sample on that window, and the synthetic
control is fitted directly against Kansas while local PCA reconstructs it from
a rank-2 approximation of 14 states, so the gap is unsurprising. It does not
support preferring local PCA on fit.

The headline sits at the edge of its tuning grid. The sign is stable —
every setting tried returns a negative gap — but the magnitude moves by about
40 percent, and the paper's settings are at the extreme:

| `K` | 7 | 10 | 14 (paper) | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| gap (pp) | −0.397 | −0.369 | −0.531 | −0.397 | −0.464 | −0.454 |

| matching quarters | 20 | 30 | 40 (paper) | 50 | 60 | 70 |
|---|---|---|---|---|---|---|
| gap (pp) | −0.429 | −0.379 | −0.531 | −0.529 | −0.526 | −0.451 |

| component cap | 2 | 3 (paper) | 4 | 5 |
|---|---|---|---|---|
| gap (pp) | −0.522 | −0.531 | −0.428 | −0.413 |
| rank chosen | 1 | 2 | 3 | 4 |

The split point is the tame one: anything from 40 to 60 quarters gives the same
answer. `K` and the component cap are not, and the paper leaves both to a
heuristic it declines to justify — Remark 4.3 defers rank selection to future
research. An estimator should surface the chosen rank and neighbourhood as
diagnostics.

## Provenance

Reference code: `yingjieum/Replication_NonlinearFactorModel_2023`, commit
`ca34fba` (current application script uploaded 2024-08-01 as `59fa89f`; the
v1-era script recovered from `0608950`, 2023-11-12). Panel:
`basedata/kansas_taxcut.csv`, 5250 rows.

Two versions of the paper are in play, and the table above compares against the
first. arXiv:2311.07243v1, 14 November 2023, reports the SC arm as 0.19 points
below observed Kansas. The current working paper, dated 31 July 2024 and linked
from the replication repository's README, reports it as 0.33 points above.
Table 1 of Section 5 is identical in both. Cite the 2024 version for anything
built on this; cite v1 only when the point is the defect itself.

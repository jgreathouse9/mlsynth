# MOSC spike -- Wang, Schein, Shou & Blei, "A Many-outcomes Perspective on the Synthetic Control Method"

Demonstrate-first spike for [issue #535](https://github.com/jgreathouse9/mlsynth/issues/535).
No estimator is added. This reproduces the paper's method from the authors' own
code, runs the comparison the paper reports, and answers the two questions the
spike was opened to settle.

Sources:

* Paper: unpublished JMLR submission, supplied as PDF. No arXiv posting.
* Code and data: <https://github.com/Joshuashou/Synthetic-Control-Paper-Model>,
  by the paper's third author. README there names the paper. No licence file.
  Recorded in `agents/future_integrations.md` §17 while assessing Shen (2026).

## Verdict

Build, with one scope change and two caveats.

The paper's substantive claim reproduces: a Poisson-likelihood factor model
predicts the counterfactual better than the Gaussian alternatives on count
panels, and it does so for the reason the paper gives. The margin over robust
synthetic control is thinner than the paper's figure suggests, and the model
criticism the paper leans on does not work as specified.

## What was run

`extract_panels.py` recovers the authors' per-team panels. Upstream ships them as
`torch.save` tensors while its own scripts read `train_pivot.csv` / `test_pivot.csv`,
which are not in the repository, so they have to be reassembled first. Reading them
needs no torch. Two teams are vendored: Indianapolis (301 periods x 84 counties,
198 pre) and Baltimore (296 x 17, 235 pre).

`mosc_port.py` ports Algorithm 1. `dgp.py` reproduces the semi-synthetic
data-generating process. `run_bakeoff.py` runs 48 cells: 2 teams x pre-period
{25, 100} x factor-model departure rho {0, 0.5, 1} x 2 effect sizes x 2 data
seeds, fitting five counterfactuals per cell.

The robust-synthetic-control baseline runs through `mlsynth.CLUSTERSC` -- the
estimator a user of this library would reach for -- configured to the PCR family
with clustering off, an OLS objective and the rank pinned. That reproduces the
`pcr_weights` kernel to 4.4e-08 on the same panel, and reproduces what upstream
gets from `tslib`, while exercising `dataprep` ingestion and the result contract.

## Question 1: does the advantage survive dropping the lagged outcome?

Yes, and the answer runs opposite to the concern that motivated asking.

Upstream's design matrix carries `Y_{i,-1}`, the last pre-intervention outcome,
alongside the loadings and the treatment dummy. It is hardcoded `True` in the
script that produced Figure 8 and set `True` in the ATT notebook; it appears
nowhere in equations 40-41; and the rSC baseline gets no equivalent term. The
worry was that it, not the likelihood, carried the published result.

Mean relative error over 30 post-periods, all 48 cells (lower is better):

| model | without the lag | with the lag (as published) |
|---|---|---|
| GAP  | 0.0324 | 0.0343 |
| PPCA | 0.0462 | 0.0357 |
| rSC  | 0.0376 | 0.0376 |

The lag improves PPCA by 0.0105 and costs GAP 0.0019. It was propping up the
Gaussian arm, not the Poisson one, so the undocumented regressor *understates*
the paper's own claim: GAP's margin over PPCA is 0.0014 in the published
specification and 0.0138 without it, a tenfold difference in the paper's favour.

The regressor still has to go in any build -- it is undisclosed and it changes
the comparison -- but removing it strengthens the result.

## Question 2: does the advantage hold at a short pre-period?

Yes on the averages, at both pre-period lengths:

| pre-periods | GAP | PPCA | rSC |
|---|---|---|---|
| 25  | 0.0388 | 0.0571 | 0.0437 |
| 100 | 0.0261 | 0.0352 | 0.0314 |

This is the question that matters for this library, because the paper's
identification is asymptotic in the pre-period and panels here typically carry
12-30. The advantage does not need the long pre-period.

Two things cut against reading the averages as decisive.

First, the head-to-head is much closer than the means. Counting cells:

| | cells won |
|---|---|
| GAP best of the three | 28/48 |
| rSC best of the three | 14/48 |
| PPCA best of the three | 6/48 |
| GAP beats rSC | 29/48 |
| GAP beats PPCA | 37/48 |

GAP beats PPCA reliably. Against rSC it wins 29 of 48, which the averages dress
up as a larger effect than the per-cell record supports.

Second, where GAP wins is not where the paper says. Splitting by departure from
the factor model:

| rho | GAP | rSC | GAP beats rSC |
|---|---|---|---|
| 0.0 | 0.0385 | 0.0354 | 9/16 |
| 0.5 | 0.0227 | 0.0376 | 12/16 |
| 1.0 | 0.0361 | 0.0396 | 8/16 |

rSC is ahead when the factor model is exactly right. GAP's advantage is
robustness to departures from it, not fidelity to it.

The paper's stated pattern -- "when there are more [pre-intervention time steps],
the performance gap between GaP and the Gaussian models becomes much more
pronounced" -- does not reproduce. The GAP-over-PPCA margin is 0.0183 at 25
pre-periods and 0.0092 at 100: it narrows as the pre-period grows.

## The model criticism does not work as specified

Section 3.4 makes model criticism the step that licenses `Z` to stand in for the
unobserved confounder, so the holdout predictive check is not an accessory.

Section 4.3.1 states its property: "If the model is well-specified `p_pop` should
be uniformly distributed. Therefore, a policy that rejects a model unless it has
`p_pop` in `[alpha/2, 1 - alpha/2]` will have a false rejection rate of `alpha`."

On data drawn from the gamma-Poisson model and checked against that same model,
`validate_predictive_check.py` sweeps the size of the held-out set:

| held-out cells | p_pop | verdict |
|---|---|---|
| 13  | 0.075 | accept |
| 34  | 0.025 | accept |
| 101 | 0.117 | accept |
| 181 | 0.000 | REJECT |
| 420 | 0.000 | REJECT |

False rejection rate on a correctly specified model: 0.40, against the stated
0.05. The cause is the aggregation, not the models. Equation 36 sums the
discrepancy over held-out cells and equation 35 compares that sum for a replicate
against the real data at the same posterior draw. The replicate is drawn at the
fitted rate and so matches it exactly, while the real data carries the rate's
estimation error. Per cell that is a small systematic gap in the replicate's
favour; summed over `n` cells the gap grows like `n` and its spread like
`sqrt(n)`, so the comparison becomes deterministic and `p_pop` collapses onto 0
or 1.

The paper holds out 10% of a 198x84 matrix, about 1,660 cells, which is far
inside the degenerate regime. Its own reported results carry the signature:
PPCA returning "p_pop-values of 1.0" almost everywhere, and GAP, where it passes,
"rarely doing so for all masks."

The comparison the check was reached for survives when scored without the
calibration claim. Held-out predictive log density per cell, Indianapolis
pre-period, 10% held out, K=10 (higher is better):

| series | GAP | PPCA |
|---|---|---|
| cumulative (as shipped) | -3.711 | -6582.488 |
| daily (first difference) | -2.537 | -14.107 |

A build should report this score and drop `p_pop`.

## The panels are cumulative counts

The paper argues for a Poisson likelihood because the outcomes are counts. The
panels the authors ship are cumulative case counts -- each series starts at 1 and
climbs to about 70,000, non-decreasing at 98.5% of steps. `check_outcome_scale.py`
measures what that does to the two assumptions the argument needs, on held-out
cells against a rank-10 fit:

| team | series | Pearson dispersion | residual lag-1 |
|---|---|---|---|
| Indianapolis | cumulative | 13.0 | 0.448 |
| Indianapolis | daily | 1.7 | 0.173 |
| Baltimore | cumulative | 193.8 | 0.204 |
| Baltimore | daily | 2.6 | 0.067 |

Poisson requires a dispersion of 1. Equations 12 and 19 require the factors to
render a unit's outcomes conditionally independent, so the residual
autocorrelation is what has to be near zero.

Differencing moves dispersion from 13-194 down to 1.7-2.6 and residual
autocorrelation from 0.20-0.45 down to 0.07-0.17. The method is being run on the
scale that fits its assumptions worst, and the fix strengthens exactly the
assumptions identification rests on. This is the same cumulative-versus-daily
ambiguity §17 hit on this data for a different paper.

## Where the cost is

Measured on the 100-period Indianapolis panel at 150 posterior samples:

| step | seconds |
|---|---|
| GAP Gibbs sampler, 300 sweeps | 7.0 |
| downstream CV-ridge, per draw | 0.20 |
| ...x150 draws | 29.5 |

The Bayesian inference is 7 seconds; the downstream regression is 30. Upstream
refits a 5-fold cross-validated Ridge over a 4-point alpha grid for every
posterior draw, and page 23 says why -- label switching makes averaging the draws
ill-defined. The per-draw refit is intrinsic to Algorithm 1 and costs four times
the inference it wraps. That is what a build should attack.

The sampler cost is already solved. Upstream runs NUTS; the conjugate multinomial
augmentation its own dead `gibbs_sample` sketches gets the same posterior in
7 seconds of pure NumPy, with no torch and no `[bayes]` extra.

## Defects in the upstream code

Found by running it. Each is corrected in the port and the correction is
documented at the call site.

* `GAP.gibbs_sample` cannot run: `np.random.default_rn()` is a typo for
  `default_rng` and also shadows the `numpy.random as rn` alias used above it;
  both `einsum` subscript strings contract axes whose lengths disagree with the
  shapes the same function builds; and `data[mask] = 0` zeroes the observed cells
  and not the held-out ones. Never called -- the paper uses NUTS.
* `create_mask` tests `if mask_type == ["plaid", "random"]`, a string against a
  list, so those two mask types hold nothing out. `speckled` holds out 1% where
  the paper says 10%. The semi-synthetic runs pass `mask=None`, so the predictive
  check never runs there at all.
* `deconfound_and_plot.py.__main__` calls `load_team_data(..., data_supdir=)`
  against a `dat_supdir=` signature, and reads `train_pivot.csv` / `test_pivot.csv`
  which the repository does not ship.
* Paths are hardcoded to `/net/projects/schein-lab/`; there is no requirements
  file; the rSC baseline needs an unpinned external `tslib`.

## Paper against code

* Equation 49 prints as `delta_t = [1 + log(t)] ^ (t*alpha/1000)` -- the fraction
  sits in the exponent, the only reading consistent with the paper's own sentence
  that `alpha = 0` gives `delta_t = 1`. Upstream writes
  `(1 + log1p(t)) ** (t / alpha)`, so `alpha_code = 1000 / alpha_paper`. Same
  curve, reciprocal parameterisation.
* Upstream's grid {250, 500, 100000} is `alpha_paper` {4, 2, 0.01}, against the
  paper's stated {1/10, 4}. Its smallest arm multiplies the treated path by
  between 1.0000 and 1.0004 over thirty periods -- no effect at all.
* Upstream's plotting cell pairs those values with labels as
  `zip([250, 500, 100000], ['Small', 'Medium', 'Large'])`, which is inverted
  against the effects they produce: it calls the largest effect Small and the null
  one Large. That cell writes into the paper's own figure directory.
* Equation 46 asks for a Poisson factorisation; upstream calls `sklearn` NMF at
  its Frobenius (Gaussian) default.
* Equation 50 mixes against the previous realised outcome, a recursion; upstream
  mixes against a Poisson panel drawn once before the loop, so its autoregressive
  arm never compounds.
* Ground truth is the noiseless rate, which is what upstream scores against; the
  paper's MRE definition writes an outcome.
* The ATT notebook selects `max(best_latent_dim_dict[team])` -- the largest rank
  that passes the check -- a rule stated nowhere in the paper.
* `A[-1] = 1`: one treated unit per dataset, a single aggregated `Stadium_County`
  column, against Section 4.1's plural targets.

## If it is built

* New top-level estimator, not a mode on an existing one -- no weight solve, no
  donor selection, no balancing. Name it for the mechanism (`MOSC`), since
  "nonlinear synthetic control" collides with `NSC` (Tian 2023).
* Gibbs for the gamma-Poisson arm, so no `[bayes]` extra.
* Drop the lagged outcome. Drop `p_pop` and report held-out predictive log
  density instead.
* Difference the outcome, or expose the scale and default to differencing, with
  the dispersion and residual-autocorrelation diagnostics on the result.
* Attack the per-draw regression refit before anything else.
* `WeightsResults` has nothing to populate; MCNNM sets that precedent.

## Caveats on this spike

* Two teams, not the paper's 31. Indianapolis has 84 counties and Baltimore 17,
  so the small-N regime is thinly covered.
* PPCA here is EM where upstream is NUTS. The GAP-versus-PPCA comparisons carry
  implementation risk that the GAP-versus-rSC ones do not, since rSC is
  reproduced to 4.4e-08.
* The Gibbs sampler is validated by construction (`validate_gibbs.py`: recovers
  the generating rate to 5%, beats an intercept-only fit, degrades at the wrong
  rank, recovers held-out cells) and not against upstream's NUTS, which has
  no shipped output to match.
* 150 posterior draws per arm. The estimand is a posterior mean, so this is not a
  binding approximation, but it is not the paper's 2,000 either.

## Files

| file | what it does |
|---|---|
| `extract_panels.py` | recovers the authors' panels from their tensors |
| `mosc_port.py` | Algorithm 1: GAP Gibbs, PPCA EM, outcome regression, `p_pop` |
| `dgp.py` | the semi-synthetic DGP, both published forms |
| `run_bakeoff.py` | the 48-cell comparison |
| `analyse_bakeoff.py` | answers the two questions, writes `results.json` |
| `validate_gibbs.py` | validates the sampler by construction |
| `validate_predictive_check.py` | calibrates `p_pop`, scores held-out density |
| `check_outcome_scale.py` | dispersion and conditional-independence diagnostics |

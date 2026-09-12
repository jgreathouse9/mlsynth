# Firpo–Possebom confidence sets — reference bundle

Oracle material for a port of the placebo-inverted confidence sets and the
assignment-probability sensitivity analysis of

> Firpo, S. & Possebom, V. (2018), "Synthetic Control Method: Inference,
> Sensitivity Analysis and Confidence Sets", *Journal of Causal Inference*
> 6(2), 20160026.

The port lives in `mlsynth/utils/vanillasc_helpers/placebo_cs.py`, reached as
`VanillaSC(inference="placebo_cs")`. Benchmark authoring is a separate
workstream, so there is no `benchmarks/cases` entry yet; this directory holds
what `mlsynth/tests/test_placebo_cs.py` validates against.

## What the reference is

`SCM.CS`, in the journal supplement's `function_SCM-CS_v07.R`. It inverts
Abadie's placebo test over a one-parameter class of effect paths — constant
`phi`, or linear `phi * (t - T0)` — and reports the parameter values the test
does not reject.

The step that distinguishes it from a quantile interval on a placebo
distribution is that the candidate effect is imposed across the whole panel
before the statistic is recomputed: for each placebo unit `j`, the path is added
to `j`'s own outcome and subtracted from the treated unit's column inside `j`'s
donor pool. The post/pre MSPE ratio is not invariant to the hypothesised effect,
so the reference distribution has to move with the null.

The sensitivity mechanism reweights the rank p-value by
`prob = softmax(phi_sens * v)` over units, with `v` a declared 0/1 vector. At
`phi_sens = 0` this is the uniform Abadie p-value.

## The scripts are not vendored here

`function_SCM-CS_v07.R`, `california_beta_testing_2018-08-12.R` and
`smoking_dataset.csv` are journal supplementary material and are not
redistributed in this repository. Fetch the supplement from the article;
`reference.R` expects the function file beside it, and `reference_authors.R`
takes the supplement directory as its first argument. The two `provenance*.json`
files record the sha256 of every supplement file used, so a future run can
confirm it has the same code.

## Two arms

`SCM.CS` consumes a `J x (J+1)` matrix of pre-computed placebo weights. They are
an input to the procedure under test, so the bundle pins the inversion twice,
against two different weight sets.

`reference_authors.R` is the decisive one. It is the authors' own California
driver — `Synth` fits all 39 placebo units under their predictor specification,
then `SCM.CS` inverts on the result — with the working directory and the
`doParallel` backend dropped and the sensitivity sweep appended. Everything
except the inversion is the authors' code. Their specification is not ADH 2010's:
`beer` sits in the plain predictor block averaged over 1980–1988, where ADH gives
it a special predictor over 1984–1988, so reconstructing the spec from the paper
produces different weights and different bounds.

`reference.R` is the arm that runs without R `Synth`. `Ymat.csv` and
`weightsmat.csv` are computed once by
`mlsynth.utils.bilevel.simplex.simplex_lstsq` (outcome-only simplex over
1970–1988) and handed to both sides, so any difference in the captured bounds is
attributable to the inversion and to nothing else. `SCM.CS` itself needs no CRAN
package — it takes the outcome matrix and the weights as arguments and the rest
is base R — so this arm runs on a stock installation.

State order is alphabetical in both, matching the authors' driver — it relies on
R factor-level order, under which `californiaid <- 3`, which both panels
reproduce.

## Installing Synth

CRAN is unreachable from some sandboxes, which is not the same as `Synth` being
unavailable. `benchmarks/R/install_sc_references.sh` documents the route, and
apt carries the three dependencies at the versions `synth_jhai_prop99` pinned:

```bash
apt-get install -y r-cran-kernlab r-cran-optimx r-cran-rgenoud r-cran-ggplot2
git clone --depth 1 https://github.com/j-hai/Synth && R CMD INSTALL Synth
```

## Contents

```
reference_authors.R       the authors' driver (Synth + SCM.CS), serialised
Ymat_authors.csv          31 x 39 outcomes, from Synth's dataprep
weightsmat_authors.csv    38 x 39 placebo weights, from synth(method = "BFGS")
gold_bounds_authors.csv   the captured output, 5 configurations
provenance_authors.json   versions, spec, hashes, and the measured deviation

reference.R        the Synth-free arm (sources the supplement's SCM.CS)
Ymat.csv           31 x 39 outcome matrix, from dataprep
weightsmat.csv     38 x 39 placebo weights, column j = unit j's donors
gold_bounds.csv    the captured output, 8 configurations
provenance.json    versions, spec, and the sha256 of script, archive and data
```

## The captured result — the authors' weights

Uniform assignment (`phi_sens = 0`), California, `significance = 4/39`, and then
the linear class tilted toward the treated unit. mlsynth reproduces every row to
7.1e-15 and refuses on the same one.

| class | phi_sens | lower | upper | zero inside? |
|---|---|---|---|---|
| constant | 0.0 | −27.881392962934704 | −9.588311133625828 | no |
| linear | 0.0 | −3.983798401337399 | −1.238873954311117 | no |
| linear | 0.5 | −4.139948245661779 | −1.078944579024872 | no |
| linear | 1.0 | −4.322418798490689 | −0.901968838437196 | no |
| linear | 2.0 | — | — | search fails |

## The captured result — outcome-only weights

Uniform assignment (`phi_sens = 0`), California, `significance = 4/39`:

| class | lower | upper |
|---|---|---|
| constant | −31.453690469626419 | −6.0579127580238854 |
| linear | −4.503544289801302 | −0.7857011601364281 |

Sensitivity on the linear class, tilting toward the treated unit:

| phi_sens | lower | upper | zero inside? |
|---|---|---|---|
| 0.0 | −4.5035 | −0.7857 | no |
| 0.5 | −4.7083 | −0.6146 | no |
| 1.0 | −6.9623 | +1.4684 | yes |
| 2.0 | — | — | search fails |

So on this specification the Proposition 99 conclusion absorbs a tilt of about
`phi_sens = 0.5` and loses its sign at `1.0` — where under the authors' weights
it survives `1.0`. The sensitivity verdict is the part of the output that moves
with the donor fit. That is the quantity
`docs/vanillasc.rst` describes as the Rosenbaum `Gamma` and defers as needing a
non-convex program: with `v` declared instead of optimised over, it is a sweep.

Tilting the other way — `v` marking the donors — leaves every bound unchanged.
That is a property of the discreteness, not a bug: with 39 units the rank
p-value lives on multiples of about 1/39 and the threshold is 4/39, so
redistributing mass away from the treated unit rarely crosses it. Any port
should reproduce this, and any docs page should say it.

## Regenerating

```bash
# the authors' arm: <supplement_dir> holds function_SCM-CS_v07.R and
# smoking_dataset.csv; <outdir> caches the 39 Synth fits, so a re-run of the
# confidence sets alone skips them
Rscript benchmarks/reference/fp_confidence_sets/reference_authors.R \
        <supplement_dir> benchmarks/reference/fp_confidence_sets

# the Synth-free arm: place function_SCM-CS_v07.R in this directory first
Rscript benchmarks/reference/fp_confidence_sets/reference.R
```

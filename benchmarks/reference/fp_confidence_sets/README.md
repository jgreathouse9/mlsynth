# Firpo–Possebom confidence sets — reference bundle

Oracle material for a port of the placebo-inverted confidence sets and the
assignment-probability sensitivity analysis of

> Firpo, S. & Possebom, V. (2018), "Synthetic Control Method: Inference,
> Sensitivity Analysis and Confidence Sets", *Journal of Causal Inference*
> 6(2), 20160026.

mlsynth has no such estimator yet (see issue #552), so there is no
`benchmarks/cases` entry. This directory holds what a port validates against.

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

## The script is not vendored here

`function_SCM-CS_v07.R` is journal supplementary material and is not
redistributed in this repository. Fetch the supplement from the article and
place the file beside `reference.R`; `provenance.json` records the sha256 of
both the script and the supplement archive so a future run can confirm it has
the same code.

`SCM.CS` needs no CRAN package. It takes the outcome matrix and the weights as
arguments and the rest is base R, so `reference.R` runs on a stock installation
— which matters here, because the authors' own driver reaches for `Synth` only
to *produce* the weights, and CRAN is not always reachable.

## Why the weights are an input, not part of the comparison

`SCM.CS` consumes a `(J) x (J+1)` matrix of pre-computed placebo weights. They
are therefore an input to the procedure under test. `Ymat.csv` and
`weightsmat.csv` here are computed once by
`mlsynth.utils.bilevel.simplex.simplex_lstsq` (outcome-only simplex over
1970–1988) and handed to both sides, so any difference in the captured bounds is
attributable to the inversion and to nothing else.

The consequence, stated plainly: these bounds are not the paper's published
confidence set. The authors' driver uses Synth's full ADH predictor
specification with `special.predictors`, which produces different weights. A
case that wanted to pin the *published* numbers would have to reproduce that
specification first. What is pinned here is the inversion.

State order is alphabetical, matching the authors' driver — it relies on R
factor-level order, under which `californiaid <- 3`, which this panel
reproduces.

## Contents

```
reference.R        the run (hand-written; sources the supplement's SCM.CS)
Ymat.csv           31 x 39 outcome matrix, from dataprep
weightsmat.csv     38 x 39 placebo weights, column j = unit j's donors
gold_bounds.csv    the captured output, 8 configurations
provenance.json    versions, spec, and the sha256 of script, archive and data
```

## The captured result

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
`phi_sens = 0.5` and loses its sign at `1.0`. That is the quantity
`docs/vanillasc.rst` describes as the Rosenbaum `Gamma` and defers as needing a
non-convex program: with `v` declared instead of optimised over, it is a sweep.

Tilting the other way — `v` marking the donors — leaves every bound unchanged.
That is a property of the discreteness, not a bug: with 39 units the rank
p-value lives on multiples of about 1/39 and the threshold is 4/39, so
redistributing mass away from the treated unit rarely crosses it. Any port
should reproduce this, and any docs page should say it.

## Regenerating

```bash
# place function_SCM-CS_v07.R in this directory first
Rscript benchmarks/reference/fp_confidence_sets/reference.R
```

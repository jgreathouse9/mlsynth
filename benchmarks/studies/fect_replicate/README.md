# FEct, demonstrated before it is built

Reference: Liu, Licheng, Ye Wang and Yiqing Xu (2022), "A Practical Guide to
Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional
Data", *American Journal of Political Science* 68(1):160-176, DOI
[10.1111/ajps.12723](https://doi.org/10.1111/ajps.12723). Section "Empirical
Examples" is what this reproduces. Replication package: AJPS Dataverse, DOI
[10.7910/DVN/ZVC9W5](https://doi.org/10.7910/DVN/ZVC9W5), which ships `fect`
0.4.1 and the stdout of the authors' own runs under `log/`.

This is a replication spike. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.fect_replicate.run
```

## This is not a benchmark

A benchmark case pins what an estimator's `.fit()` returns. Nothing here does,
because as of this commit **no mlsynth estimator can fit either panel**. The
numbers below come from `fect_fe.py`, a reimplementation that lives in this
directory and is not part of the library. It exists to answer one question
before a build starts — are the published numbers reachable from `basedata` at
all — and the answer is yes.

`run.py` records the refusals, which are the specification for what a FEct
estimator would have to support:

| panel | estimator | refusal |
| --- | --- | --- |
| hh2015 | `MCNNM` | Some unit is treated at the earliest period |
| hh2015 | `GSYNTH(r=0)` | needs absorbing adoption; treatment reverses for units 3215, 4034 |
| fm2015 | `MCNNM` | panel is unbalanced (missing unit-time cells) |
| fm2015 | `GSYNTH(r=0)` | panel is not strongly balanced |

All four hold on `fect`'s own `min.T0` sample too, so this is not something
preprocessing fixes. Applying `min.T0 = 5` to hh2015 still leaves units treated
in the first year, because `min.T0` counts untreated observations wherever they
fall and a unit that switches off again accumulates them after treatment.

That is the whole difficulty in one sentence: these panels have no clean
pre-period block, and every counterfactual estimator mlsynth currently ships
needs one.

## What was reproduced

Both of the paper's empirical examples, point estimates only, in NumPy.

### Example 1 — Hainmueller & Hangartner (2019), Swiss naturalisation

1211 municipalities, 1991-2009, staggered adoption with two reversals.

| quantity | authors | here | abs diff |
| --- | --- | --- | --- |
| TWFE coefficient | 1.339 | 1.33858 | 4.2e-04 |
| TWFE HC1 standard error | 0.161 | 0.161264 | 2.6e-04 |
| TWFE t | 8.301 | 8.30055 | 4.5e-04 |
| TWFE residual sd | 4.206 | 4.20626 | 2.6e-04 |
| TWFE residual dof | 21779 | 21779 | exact |
| FEct ATT, treated obs equally weighted | 1.767 | 1.76728 | 2.8e-04 |
| FEct ATT, treated units equally weighted | 1.249 | 1.24883 | 1.7e-04 |

### Example 2 — Fouirnaies & Mutlu-Eren (2015), English council grants

432 councils, 1992-2012, treatment switching on and off.

| quantity | authors | here | abs diff |
| --- | --- | --- | --- |
| TWFE coefficient | -0.015 | -0.0147484 | 2.5e-04 |
| FEct ATT, treated obs equally weighted | -0.03396 | -0.0337895 | 1.7e-04 |
| FEct ATT, treated units equally weighted | -0.07488 | -0.0747373 | 1.4e-04 |

Example 2 lands at 0.5% and 0.2%, against example 1's 0.02%. Its script also
passes `pre.periods = c(-6, 0)` and `group = "cohort"`, neither of which this
reimplementation reads; that is the likely residual and it is not chased here.

## Three details that decide the numbers

**`min.T0` counts untreated observations, not a leading run.** `fect` 0.4.1
defaults it to 5 (`R/default.R:61`) and silently drops treated units below it —
294 of 811 on example 1. Counting the leading untreated run instead is the
obvious reading, and it gives 1.7786 against the published 1.767. Counting
untreated cells wherever they fall gives 1.7673. The two only differ when
treatment reverses, which is exactly the design this paper is about.

**The published standard error is HC1, not clustered.** `fastplm`'s "Robust"
is heteroskedasticity-robust. Clustering on the municipality gives 0.185 against
the published 0.161, which reads as a failed replication and is not one.

**The two ATT weightings are not interchangeable.** `fect` prints both; they
differ by a factor of 1.4 on example 1 and 2.2 on example 2. A port that reports
one of them without saying which is reporting an ambiguous number.

## What was not reproduced

Recorded in the JSON as `unchecked`, with the log line each target sits on, so
the next spike does not have to find them again:

- the 1000-draw block bootstrap standard errors (FEct 0.2021 on example 1);
- the placebo test (placebo ATT 0.1455, p = .422) and the equivalence tests;
- the F-test for pretrends (p = .182);
- IFEct (`method="ife"`, r=2 on example 2, ATT 0.006397);
- MC, whose cross-validation picks `lambda.norm* = 1` on example 1 — maximum
  regularisation, so it collapses onto FEct and returns the same 1.767, which is
  the paper's own claim and is checkable once MC can run.

## Recommendation

Build FEct. It is small — the estimator is two-way fixed effects
fitted on untreated cells and extrapolated onto treated ones, about thirty lines
before the inference. What earns it a place is not the point estimate but the
regime: it accepts panels with treatment reversal, with units treated from
the first period, and unbalanced, none of which `MCNNM` or `GSYNTH` will take.
That is a gap in coverage, measured above and not asserted.

The diagnostics are the larger prize and the larger job. The placebo test, the
no-carryover test, the equivalence tests and the F-test are this paper's actual
contribution, and they would apply to every estimator in the library, not only
this family. A grep for `TOST`, `equivalence test` or `carryover` across
`mlsynth/` returns nothing.

Order: FEct first, since the diagnostics need an estimator that can run these
panels at all. Then the diagnostics, on their own branch. Only then can this
directory become a case, with `FECT(...).fit()` pinned against the targets
above.

## Files

```
run.py       the checks and the coverage probe; writes results/fect_replicate.json
fect_fe.py   FEct and TWFE in NumPy -- a demonstration, not library code
results/     fect_replicate.json
```

Data: `basedata/hh2015_naturalization.parquet` and
`basedata/fm2015_council_grants.parquet`, rebuilt from the Dataverse package by
`benchmarks/data/build_fect_examples.py`.

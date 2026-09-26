# Where SL's random-forest expert differs between R and scikit-learn

Reference: Daniel Viviano and Jelena Bradic, "Synthetic learner: model evaluation
with limited overlap", *Journal of Econometrics* 234(2):691-713, DOI
10.1016/j.jeconom.2022.07.005, and their released application package:
`libraries/library.R` lines 107-116 for the forest, `analyze_main_text.R` line
403 for the block whose numbers are Table 4's second half.

`docs/replications/sl.rst` said the residual gap between mlsynth's effect and
their Table 4 was "the one member that cannot be matched across languages". This
study measures that. It is a study and not a benchmark case: both forests are
random, so every figure here is a mean over a seed sweep with its spread beside
it, and nothing is pinned.

```
python -m benchmarks.studies.sl_forest_languages.run
python -m benchmarks.studies.sl_forest_languages.run --employment <employment_BFRSS.txt>
```

The second form adds the arm that needs their `employment_BFRSS.txt`, 300 rows by
51 columns, which is in their package and is not vendored here. Without it the
study runs the language comparison, which needs only `basedata/`.

## The answer

The two implementations can be matched. Given the same design and the same
hyperparameters they are exchangeable draws from one distribution:

| comparison | within R | within scikit-learn | across | 30-seed mean paths |
| --- | --- | --- | --- | --- |
| 13 predictors, matched settings | 0.00291 | 0.00328 | 0.00309 | 0.00084 |
| 57 predictors, matched settings | 0.00331 | 0.00321 | 0.00327 | 0.00047 |
| 13 predictors, each on its own defaults | 0.00305 | 0.00328 | 0.00695 | 0.00541 |
| 57 predictors, each on its own defaults | 0.00302 | 0.00321 | 0.00403 | 0.00277 |

Each cell is the mean over seed pairs of the largest per-period difference
between two forest paths, on an outcome running from 0.048 to 0.235. Matched, the
distance between an R path and a scikit-learn path is the distance between two R
paths. Averaged over 30 seeds each, the two mean paths agree to 4.7e-04.

What does differ is the defaults. R's `randomForest` regression default is
`mtry = floor(p/3)` and `nodesize = 5`; scikit-learn's `RandomForestRegressor`
considers every feature and splits to `min_samples_leaf = 1`. On the 13-predictor
design that moves the paths by 2.2 times the seed spread, which is a real
difference and is a difference of settings, not of algorithms.

## What the gap to Table 4 actually is

Effect over their window, periods 52 to 88, mean over 30 seeds, the other three
experts held at mlsynth's (they agree with the R reference to 2.3e-12, 7.0e-07
and 4.9e-11, so only the forest column moves):

| step | effect | change |
| --- | --- | --- |
| mlsynth as shipped: 13 predictors, scikit-learn defaults, `eta` 48.25 | 5.3513 | |
| predictor set 13 -> 57 | 5.2662 | -0.0851 |
| `eta` 48.25 -> 51.43 | 5.2173 | -0.0489 |
| scikit-learn -> R, their exact call | 5.2186 | +0.0013 |
| their published Table 4 | 5.2227 | residual +0.0041 |

The residual is 0.42 of that arm's seed standard deviation of 0.0097, so their
published number is reproduced. The three steps explain -0.1326 of an observed
-0.1286.

Two causes, and the forest implementation is neither of them:

1. The predictor set, 66 percent of the gap. Their forest reads the six donor
   outcomes plus all 51 columns of `employment_BFRSS.txt`, which is employment
   for 50 states and Tennessee. mlsynth's covariate block comes from `dataprep`
   per column, so it holds the panel's own seven units and nothing else: 13
   predictors against 57.
2. The learning rate, 38 percent. mlsynth computes the paper's
   `1/(sqrt(T) var(y))` with the panel's T of 100 and gets 48.25; their script
   runs `1/(sqrt(88) var(med_ts))` and gets 51.43.
3. The implementation, -1 percent, inside the seed noise of either language.

## How the predictor set moves the effect

Through the weight, not the path, and the two pull in opposite directions. One
seed, holding each factor in turn:

| forest path | weights from | effect |
| --- | --- | --- |
| 13 predictors | 13 predictors | 5.3340 |
| 13 predictors | 57 predictors | 5.2139 |
| 57 predictors | 13 predictors | 5.4162 |
| 57 predictors | 57 predictors | 5.2740 |

The wider design fits the weighting window worse, mean in-window SSR 0.0395
against 0.0326, so Equation 12 gives the forest 0.155 of the weight instead of
0.204 and the ensemble leans on the other three. Swapping only the path raises
the effect by 0.082; swapping only the weights lowers it by 0.120.

## Their Table 4, through the public API

`external_covariates` makes their design expressible, so the study also runs the
whole thing through `SL(...).fit()` with their learning rate and reads the effect
off their window. 57 predictors, `eta` 51.4307, forest weight 0.145, mean over
three seeds:

| m | window | SL statistic | theirs | SL effect | theirs |
| --- | --- | --- | --- | --- | --- |
| 0 | 52-88 | 0.6919 | 0.6910 | 5.2298 | 5.2227 |
| 1yr | 56-88 | 0.6238 | 0.6225 | 5.3679 | 5.3624 |
| 2yr | 60-88 | 0.6262 | 0.6247 | 5.5227 | 5.5167 |
| 3yr | 64-88 | 0.6123 | 0.6108 | 5.5935 | 5.5870 |

The effect agrees to 0.10 to 0.14 percent and the statistic to 0.13 to 0.25
percent. What is left is the forest's seed.

## What this does not say

Their forest is not better for reading 44 states outside the donor pool, and
mlsynth's is not better for reading fewer. Neither is validated against a known
answer here; the study measures what each produces and which difference accounts
for which part of the gap.

It also does not say a single pair of seeds agrees. It does not: two paths differ
by about 0.003 whatever the language, which is why the cross-validation case
`benchmarks/cases/sl_tennessee.py` compares the three deterministic experts and
leaves this one out. What the study settles is that the disagreement between one
R path and one scikit-learn path is the disagreement between two R paths.

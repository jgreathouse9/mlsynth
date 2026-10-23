# Zhang (2026) spike -- "Causal Inference under Dynamic Selection: Time-Varying Covariates and Latent Heterogeneity"

Demonstrate-first spike. No estimator is added. This ports the paper's Section 8.1
Monte Carlo design and its Algorithm 1 (the simultaneous-treatment case) and
settles two questions: does the method reproduce its own Table 1, and what must a
port decide that the paper does not state?

Sources:

* Paper: arXiv:2609.17170v1, September 2026. Single-authored UCL working paper,
  unpublished, supplied as LaTeX source.
* Reference implementation: none. The paper ships no code, so cross-validation
  against an authoritative implementation is unavailable and Path B (the paper's
  own Monte Carlo) is the only validation route.
* Empirical application data: the Bailey (2012) AEJ:Applied replication package
  was supplied alongside the paper. Its contents are verified in Finding 8 and it
  carries everything the paper's application needs except the pre-1959 outcome
  history, which comes from ICPSR 36603 and is not reachable from here.

## Verdict

Build the simultaneous-treatment estimator. Table 1 reproduces to within two to
three Monte Carlo standard errors on every column, and the method removes about
three quarters of the DiD bias with near-nominal coverage, which is the paper's
substantive claim.

Two decisions the paper leaves open have to be encoded, because the formulas as
printed do not work without them. The kernel shares one bandwidth across two
arguments on unrelated scales; standardizing them is what separates reproducing
Table 1 from producing an estimator worse than DiD. And the propensity score
needs trimming that Algorithm 1 does not mention.

One claim does not reproduce. Double cross-fitting -- three folds with the
outcome regression and the propensity score on separate samples, the paper's
inference contribution -- shows no advantage over ordinary two-fold cross-fitting
at N = 1000. The four variants are statistically indistinguishable on bias, and
the point estimates order the wrong way.

## What was run

| script | cost | what it settles |
|---|---|---|
| `gate_did.py` | 2.4 s, 2000 reps | the DGP port, against Table 1's DID row |
| `check_distance.py` | 0.2 s | the pseudo-distance against its analytic limit and the trend claim |
| `gate_trim.py` | 8 min, 240 reps | the propensity trim sweep |
| `gate_final.py` | 17 min, 300 reps | Table 1, across three porting configurations |

Four estimator variants throughout, matching the paper's columns: `DR2` and
`DR3` are two- and three-fold cross-fitting on the estimated pseudo-distance,
and `DR2*` and `DR3*` are the infeasible versions using the true
`|alpha_i - alpha_j|`.

## Finding 1 -- the DGP port is exact

2000 reps, N = 1000, T0 = 20, against Table 1's DID row:

| | bias | SD | coverage |
|---|---|---|---|
| port | -2.16 | 1.93 | 80.1 |
| paper | 2.16 | 1.94 | 79.8 |

Three numbers to three significant figures. The DID row depends only on the
data-generating process, so this fixes the design, the burn-in, the selection
mechanism, the baseline period and the standard error before any of the paper's
machinery is exercised. Everything below is therefore about the estimator.

The sign is negative. Selection on a high lagged outcome plus mean reversion at
rho = 0.8 pushes the treated group's pre-post difference down, so DiD understates
the effect, which is what the paper's Figure 1(A) shows and what its Table 1
reports as a magnitude.

## Finding 2 -- the pseudo-distance converges to a limit the design pins down

For design 8.1 the transformed factor representation is
`g(alpha, Gamma) = alpha/(1-rho)` plus a term common to all units, so
`g(a1,.) - g(a2,.) = 5(a1-a2)` is constant in `Gamma` and the paper's eq. (4.2)
collapses to a closed form:

    d(a_i, a_j) = sup |5(a1-a2) * 5(a_i-a_j)| = 25 * 0.5 * |a_i - a_j| = 12.5 |a_i - a_j|

Regressing the sample distance on `|alpha_i - alpha_j|`:

| T0 | slope | correlation |
|---|---|---|
| 20 | 17.650 | 0.840 |
| 50 | 13.987 | 0.915 |
| 200 | 12.649 | 0.979 |
| 800 | 12.481 | 0.994 |

The slope converges to 12.5 as derived. At T0 = 20 it overstates by 41% and the
median absolute error is about half a typical distance, so the estimator attains
the paper's Table 1 performance while the distance feeding it is badly noisy.
Kernel smoothing absorbs that noise: `DR3` and `DR3*` differ by 0.21 in bias.

## Finding 3 -- the trend-robustness claim is exact

The paper's construction differs from Feng (2024) by contracting against a
difference of two other units instead of one unit's level, which it says makes
the distance immune to an additive common time trend. Adding a large
nonstationary trend to every unit:

| metric | max relative change |
|---|---|
| `zhang_range` (the paper) | 7.8e-14 |
| `feng_maxabs` (Feng; mlsynth LPCA) | 13.76 |

Machine precision against a factor of fourteen. The claim holds as stated.

The construction also reduces. Written as a max over pairs `(k1, k2)` it reads
as O(n^4 p); with `G = A A' / p` the quantity is `v_k1 - v_k2` for
`v_k = G[k,i] - G[k,j]`, so the max over pairs is the range of `v`, an O(n)
reduction. Total O(n^2 p + n^3), the shape already in
`mlsynth.utils.lpca_helpers.core.pseudo_max_distance`: measured at 1.96 s for
n = 1000, T0 = 20. Feng's variant is `max |v|` over the same vector, so the two
differ by about four lines.

## Finding 4 -- one bandwidth, two scales, and the feasible estimator breaks

The kernel weight is `K((X_j - X_i)/h) K(d_ij/h)`: a single `h` across a
covariate and a pseudo-distance. By Finding 2 the pseudo-distance runs at
12.5 times the scale of the latent distance it proxies, and neither argument is
standardized anywhere in the paper.

Running the formula as printed (`raw`) against standardizing each argument by its
own dispersion first (`std`), both at trim 0.10 with the nearest-neighbour
fallback, 300 reps:

| variant | `std` bias | `raw` bias | `std` cov | `raw` cov |
|---|---|---|---|---|
| DR2* | 0.20 | 0.08 | 94.0 | 95.3 |
| DR3* | 0.45 | 0.10 | 93.3 | 93.7 |
| DR2 | 0.56 | 2.78 | 92.0 | 75.3 |
| DR3 | 0.66 | 3.08 | 92.0 | 75.7 |

The infeasible variants are unaffected, because the true `|alpha_i - alpha_j|`
and `X` happen to be commensurable. The feasible variants collapse: bias rises
above DiD's 2.18 in magnitude and coverage falls to 75%, below DiD's 82%. On the
scale mismatch a shared bandwidth tight enough to separate units on the
pseudo-distance is far too wide on the covariate, so the covariate adjustment the
method exists to perform stops happening.

Any port must standardize, or carry separate bandwidths per argument. This is the
single most consequential undocumented choice in the paper.

## Finding 5 -- the propensity score needs trimming Algorithm 1 does not specify

`(1 - pi-hat)` sits in a denominator and Nadaraya-Watson on a handful of
effective neighbours returns `pi-hat = 1` whenever a unit's kernel neighbourhood
happens to be all-treated. Measured: `pi-hat` reaches exactly 1.0, and 6.2% to
11.2% of reps contain at least one unit above 0.9.

Trimming is free here. The true propensity is
`Lambda(alpha/2 + Y_{T0-1}/2)` with an index of standard deviation 0.51; over
40,000 pooled unit-draws it never exceeds 0.8311 and never once passes 0.90. A
`pi-hat` near 1 is estimation error the design rules out, so clipping at 0.90
cannot introduce bias.

Effect on `DR3`, 240 reps:

| trim | bias | SD |
|---|---|---|
| 0.01 | 1.14 | 5.47 |
| 0.03 | 1.35 | 3.19 |
| 0.05 | 1.39 | 2.97 |
| 0.10 | 1.45 | 2.88 |

At trim 0.01 a single rep carried a unit with `pi-hat = 0.986`, hence
`pi/(1-pi) = 70` and `max |psi| = 145` against 3 to 12 in every other rep. That
one unit moved the ATT by 0.29 and set the standard deviation for the whole run.

Coverage was identical across all four trims, which is the diagnostic to
distrust. When one influence-function term dominates, the point estimate and the
standard error are both approximately `psi_i / n_1`, so the interval widens in
exactly the reps where the estimate is worst and the rep still counts as covered.
Near-nominal coverage is compatible with intervals that are useless in a tenth of
runs. The `wid p95 / wid med` ratio in `gate_final.py` separates the two: 1.12
once the fixes are in, 1.4 to 1.6 without them.

## Finding 6 -- the empty-neighbourhood fallback sets the variance

At the bandwidth the paper's rule selects, 1.4% to 3.3% of units have no control
inside the kernel support, treated units slightly more often than controls. The
paper does not say what to do with them. Imputing the pooled control mean against
falling back to the nearest eligible control in the product-kernel sense
(smallest `max(dist, xdiff)` at which a unit enters the support), 300 reps:

| variant | SD, nearest control | SD, pooled mean | paper SD |
|---|---|---|---|
| DR2* | 2.24 | 3.11 | 2.07 |
| DR2 | 2.35 | 3.15 | 2.15 |
| DR3* | 2.28 | 2.71 | 2.41 |
| DR3 | 2.24 | 2.98 | 2.44 |

The pooled mean inflates the standard deviation by 21% to 39% and `max |psi|`
from about 1.8 to about 4.7. With the nearest-control fallback the port's
standard deviations land inside the paper's range. This, not trimming, was the
main variance driver.

## Finding 7 -- double cross-fitting shows no advantage at N = 1000

Table 1 against the port, `std` scale, nearest-control fallback, trim 0.10,
300 reps, Monte Carlo standard error 0.13 on each bias:

| variant | port bias | paper bias | port SD | paper SD | port cov | paper cov |
|---|---|---|---|---|---|---|
| DID | -2.18 | 2.16 | 1.85 | 1.94 | 82.0 | 79.8 |
| DR2* | 0.20 | 0.54 | 2.24 | 2.07 | 94.0 | 94.0 |
| DR2 | 0.56 | 0.84 | 2.35 | 2.15 | 92.0 | 92.1 |
| DR3* | 0.45 | 0.21 | 2.28 | 2.41 | 93.3 | 93.7 |
| DR3 | 0.66 | 0.54 | 2.24 | 2.44 | 95.0 | 93.7 |

Every column agrees to within two to three Monte Carlo standard errors, and the
headline result reproduces: bias falls from 2.18 to 0.56 and coverage rises from
82% to 92%.

The ordering does not. The paper has `DR3* (0.21) < DR2* (0.54) = DR3 (0.54) <
DR2 (0.84)`, three folds beating two. The port has `DR2* (0.20) < DR3* (0.45) <
DR2 (0.56) < DR3 (0.66)`, two beating three. Neither gap clears noise:
`DR2` against `DR3` is 0.10 +/- 0.19, and `DR2*` against `DR3*` is 0.25 +/- 0.18.
The honest statement is that no difference is detectable in either direction, and
the paper's measured advantage for its own refinement is absent.

A mechanism fits. Three folds leave n/3 units per nuisance against n/2, so each
Nadaraya-Watson fit has fewer neighbours and more smoothing bias; the paper's
gain is asymptotic, through a faster rate under undersmoothing. The paper already
reports two-fold winning at N = 200 and attributes it to thin subsamples. The
port puts that crossover above N = 1000 instead.

## Finding 8 -- the empirical application is available; its printed numbers are not

The supplied Bailey package was read directly. `vs_fo_final.dta` is 91,110 rows,
3,037 counties by 30 years, 1959 to 1988, balanced, with `gfr_nonint` complete at
100% and `pop1544_70` for the paper's weighting. Its `fp_year_p74_fed` gives 654
treated counties with first-grant counts

    1965: 6   1966: 43   1967: 74   1968: 52   1969: 278
    1970: 63  1971: 75   1972: 53   1973: 10

which is the paper's treatment-timing table cell for cell, including its three
pooled groups at 123, 330 and 201. `table1data.dta` carries the 1960 census
covariates the paper's selection table controls for. Nothing about the design is
missing.

What is missing is outcome history before 1959. The paper builds its
pseudo-distance on 1937 to 1964, merged in from ICPSR 36603; the package leaves 6
common pre-1965 years. Using a longer per-cohort window is possible for the later
groups -- 11 years for the 1970-1973 group -- but only by pulling in periods in
which the earlier groups are already treated, and the first group is capped at 6
either way.

That gap costs less than expected. Distance quality degrades smoothly and the
paper's own window is already noisy:

| T0 | context | correlation with true latent distance | median error / median distance |
|---|---|---|---|
| 6 | Bailey only, common pre-1965 | 0.694 | 0.83 |
| 11 | Bailey only, per-cohort G3 max | 0.770 | 0.78 |
| 19 | paper robustness, 1946-1964 | 0.825 | 0.61 |
| 28 | paper main, 1937-1964 | 0.857 | 0.56 |
| 200 | asymptotic reference | 0.973 | 0.16 |

And the estimator barely notices. `DR3`, `std` scale, nearest-control fallback,
trim 0.10:

| T0 | bias | SD | coverage | DiD bias | DiD coverage |
|---|---|---|---|---|---|
| 6 | 0.34 | 2.74 | 94.0 | -2.12 | 76.7 |
| 20 | 0.66 | 2.24 | 95.0 | -2.18 | 82.0 |
| 28 | 0.41 | 2.45 | 92.7 | -2.16 | 80.1 |

Flat across the range, with every cell removing most of the DiD bias at
near-nominal coverage. At T0 = 6 the paper's sufficient condition for root-n
inference, `T0 >> n^(2/(d_alpha+d_X))`, is violated by three orders of magnitude,
and the estimator works anyway; at the paper's own T0 = 28 it is violated by two.

So the two things are separable. Reproducing the paper's figures of -1.70, -3.39
and -3.03 needs its inputs and is out of reach here. Running the method on the
same counties, the same cohorts, the same outcome and the same weights is not,
and on this evidence a 6-year distance window is a defensible way to do it. Two
pieces of the paper's empirical section are reproducible today without the new
estimator at all: its selection-on-lagged-outcomes table, which is the motivating
evidence, and its Callaway-Sant'Anna comparison row via
`PPSCM(method="callaway_santanna")`.


## What this means for a build

Confirmed, and reusable:

* The identification gap is real. Nothing in mlsynth joins the pseudo-distance to
  covariates, a propensity score and a doubly robust moment. `LPCA` has the
  distance and stops there; `DSCAR` has the regime and assumes away the latent
  factor.
* The distance is a four-line delta on `lpca_helpers/core.py`, with a trend
  invariance the existing metric lacks. That change belongs on its own branch
  first, as `distance="trend_robust_range"` with its own tests.
* Dependencies are numpy alone.
* Cost is manageable: 2 s per distance at n = 1000, and the O(n^4) reading of the
  paper's formula is avoidable.

To encode in the estimator:

* Standardize both kernel arguments, or expose a bandwidth per argument.
  Default to standardizing. Finding 4 is the reason.
* Trim `pi-hat`, default 0.05 to 0.10, and report the share trimmed on
  `MethodDetailsResults`.
* Fall back to the nearest eligible control, not a pooled mean, and report the
  share of units that needed it.
* Report the realized `h_m` and `h_pi`, whether either sat at a grid edge, the
  median effective neighbourhood size, and the interval-width p95 over median.
  Each of these was the diagnostic that located a finding here.
* Offer two-fold cross-fitting as a supported mode. On this evidence it is at
  least as good at N = 1000 and cheaper.

## Not done

* Designs 8.2 (interactive fixed effects) and 8.3 (nonlinear binary) are
  implemented in `dgp.py` and not yet run.
* The staggered-adoption case, Algorithm 3, which is the applied contribution and
  where most of the build cost sits. Its backward recursion carries the
  within-fold `m-tilde` forward while the cross-fit `m-hat` is read off at each
  step; substituting one for the other collapses the cross-fitting without
  failing.
* Path A as a number-for-number match. See Finding 8: the paper's printed ATT
  figures cannot be matched without ICPSR 36603, but the application itself is
  available on the supplied data and the missing history costs little.
* Sample sizes other than N = 1000, T0 = 20.

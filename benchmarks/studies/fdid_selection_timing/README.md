# What Li's forward selection costs, in four implementations

Reference: Kathleen T. Li, "Frontiers: A Simple Forward Difference-in-Differences
Method", *Marketing Science* 43(2):267-279, DOI 10.1287/mksc.2022.0212, and her
released replication package.

Forward DiD's cost is its selection. The search adds one donor at a time and
scores every remaining candidate at every step, so the path is quadratic in the
donor count before any arithmetic. What differs between implementations is what a
candidate score costs.

This is a study, not a benchmark case. Timings are machine-dependent and nothing
here is pinned; the case that pins Forward DiD's *numbers* is
`benchmarks/cases/fdid_hongkong.py`.

```
python -m benchmarks.studies.fdid_selection_timing.run --reps 15 --scaling
```

## What is here

| file | what it does |
| --- | --- |
| `run.py` | the four implementations, the two panels, the exponent sweep |
| `time_selection.R` | Li's `Fun_FDID.R` selection loop, lines 12-56, timed |
| `time_selection.m` | Li's `FDID_newR2.m` selection loop, lines 9-57, under Octave |
| `results/timings.json` | the run behind the tables below |

Only the selection is timed. Both of Li's functions select and then fit in one
call, and the fit is a constant that does not scale in the donor count, so timing
the whole call would fold it in.

## Correctness first

All four implementations select the same donors in the same order on both panels.
A faster answer to a different question is not faster, so the study asserts this
before it reports a single time, and `run.py` records any disagreement.

## The two panels

| panel | donors | pre-periods | source |
| --- | --- | --- | --- |
| `hcw_hongkong` | 24 | 44 | Hsiao, Ching and Wan (2012), Li's own released companion panel |
| `fspda_china_watches` | 87 | 35 | Shi and Huang's fsPDA anti-corruption panel, p > n |

Median of 15 repetitions, on `Linux-6.18.44-fc-v37-x86_64-with-glibc2.39`,
Python 3.11.15, R 4.3.3, Octave 8.4.0:

| implementation | Hong Kong (24) | vs mlsynth | Watches (87) | vs mlsynth |
| --- | --- | --- | --- | --- |
| mlsynth `forward_did_select` | 0.0009 s | 1.0x | 0.0040 s | 1.0x |
| the same algorithm, naive, in Python | 0.0037 s | 4.0x | 0.0573 s | 14.5x |
| Li's R, `Fun_FDID.R` | 0.0060 s | 6.5x | 0.1080 s | 27.3x |
| Li's MATLAB under Octave, `FDID_newR2.m` | 0.0986 s | 106.1x | 1.2160 s | 307.6x |

The naive Python column is the one that isolates the algorithm from the language:
it is Li's rebuild-every-candidate loop written in the same language and on the
same arrays as mlsynth's. At 24 donors it costs 4.0x, at 87 donors 14.5x, so most
of the gap to her R is the algorithm and not the interpreter. Her MATLAB is slower
than her R by an order of magnitude here, which is Octave and not MATLAB: Octave's
interpreter is not MathWorks' JIT, and the loop is interpreted per candidate.

The language factor, read off the same rows at fixed algorithm, is roughly but not
exactly constant: R costs 1.62x the naive Python loop on the 24-donor panel and
1.88x on the 87-donor one, Octave 26.7x and 21.2x. Close enough to say the total
gap factorises into an algorithmic part that grows with the donor count and a
language part that does not; not close enough to quote one number for either
language.

## Where the difference comes from

Both loops were instrumented and the totals match their closed forms exactly, so
these are counts and not estimates:

| | Li's rebuild | mlsynth |
| --- | --- | --- |
| candidate scores over the path | `N(N+1)/2` | `N` steps |
| array elements touched | `N(N+1)(N+2) T0 / 6` | `N^2 T0` |
| Hong Kong, `N` 24, `T0` 44 | 300 scores, 114,400 elements | 24 steps, 25,344 elements |
| Watches, `N` 87, `T0` 35 | 3,828 scores, 3,974,740 elements | 87 steps, 264,915 elements |

Li's two implementations rebuild each candidate's donor average from scratch:
`rowMeans(x[1:t1, c(selected, j)])` for every remaining `j` at every step, so the
work per score grows with the selected set. mlsynth centres the donors once,
caches each donor's squared norm and its cross-product with the treated unit, and
keeps a running centred sum of the selected set; a step is then one matvec that
scores every candidate at once.

The two counts give two different predictions for the gap, `(N+1)/2` and
`(N+1)(N+2)/6N`, and they differ by a factor of three. Neither is the answer on
its own. On the two panels the score-count ratio reads 12.5 and 44.0 against
measured gaps of 4.0 and 14.5, while the element-count ratio reads 4.5 and 15.0;
at 320 donors on the synthetic panels it is the other way round, the measured
155.7 sitting near the score count's 160.5 and far above the element count's
53.8. A score costs a fixed interpreter charge plus a per-element charge, and
which one dominates moves with the donor count.

## Two terms, and the exponents they imply

Fitting both terms at once, minimising relative error over the sweep below:

```
naive    t = 1.74e-08 N^3 + 6.14e-06 N^2 + 4.17e-04     rel rms 1.5%
mlsynth  t = 1.41e-08 N^2 + 1.74e-05 N   + 7.08e-04     rel rms 2.7%
```

against 18.5% and 24.4% for the best single power law, which is the sense in
which one exponent cannot describe either series. The coefficients are unit
costs: 2.61 ns per element touched for the naive loop against 0.351 ns for the
matvec, and 12.3 us of fixed cost per candidate score against 17.3 us per
mlsynth step. mlsynth pays more per interpreter round-trip and 7.4 times less per
element.

So the asymptotic gap is `1.24 N`, which is the element-count ratio `N/6`
multiplied by the per-element cost ratio, and both series reach their asymptotics
late. The leading term overtakes the rest at `N` about 353 for the naive loop and
about 1274 for mlsynth.

## The exponents are approached, not reached

Synthetic panels at `T0 = 40`, median of 15:

| donors | mlsynth | naive | ratio |
| --- | --- | --- | --- |
| 20 | 0.00108 s | 0.00299 s | 2.8x |
| 40 | 0.00142 s | 0.01159 s | 8.2x |
| 80 | 0.00209 s | 0.04907 s | 23.5x |
| 160 | 0.00409 s | 0.22434 s | 54.9x |
| 320 | 0.00755 s | 1.17475 s | 155.7x |
| 640 | 0.01754 s | 7.08454 s | 403.8x |
| 1280 | 0.04584 s | 47.23252 s | 1030.3x |

Fitted over that range the naive slope is 2.32 where the arithmetic predicts
3.00, and mlsynth's is 1.38 over 320 to 2560 donors where the arithmetic predicts
2.00. The local slopes climb toward those asymptotics as the crossovers are
passed: naive 1.95, 2.08, 2.19, 2.39, 2.59, 2.74 from 20-to-40 up to
640-to-1280, and mlsynth 1.11, 1.44, 1.57 over 320 to 2560. The earlier version
of this study stopped the shared sweep at 320, entirely below the naive
crossover at 353, and fitted 2.16 for a cubic algorithm.

The check on all of this is the ratio, which the fit never saw. Measured, its
local slope runs 1.56, 1.52, 1.22, 1.50, 1.38, 1.35 across the sweep; the two
fitted models predict 1.49, 1.48, 1.42, 1.39, 1.37, 1.33. Both sit well above the
asymptotic 1.00 because the two crossovers are a factor of 3.6 apart, so over
this range the naive loop is entering its cubic regime while mlsynth is still
overhead-bound. The model puts the ratio's slope back at 1.04 only near 20,000
donors.

The fit also extrapolates. Fitted on 20 to 320 alone it predicted the naive cell
at 640 and 1280 as 6.8 s and 44.3 s; measured, 7.08 s and 47.23 s.

Those local slopes need repetitions to be readable. At five they came out
non-monotone for mlsynth and moved by up to 1.0 between runs, because the small
cells run in 8 to 146 milliseconds; the figures above are medians of 15.

## What is not claimed

Her MATLAB timing is Octave's, not MATLAB's, and MATLAB's JIT would close much of
that particular gap. Nothing here says her code is wrong: it selects exactly what
mlsynth selects on both panels, and `benchmarks/cases/fdid_hongkong.py` pins
mlsynth against it to 3e-13. The selection is also rarely the binding cost in
applied work at these donor counts -- a millisecond against a hundred is a
difference that matters inside a bootstrap or a placebo sweep, where the fit is
repeated thousands of times, and nowhere else.

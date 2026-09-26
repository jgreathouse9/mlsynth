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
| mlsynth `forward_did_select` | 0.0009 s | 1.0x | 0.0033 s | 1.0x |
| the same algorithm, naive, in Python | 0.0038 s | 4.3x | 0.0525 s | 16.1x |
| Li's R, `Fun_FDID.R` | 0.0060 s | 7.0x | 0.0850 s | 26.0x |
| Li's MATLAB under Octave, `FDID_newR2.m` | 0.0969 s | 112.3x | 1.2799 s | 391.2x |

The naive Python column is the one that isolates the algorithm from the language:
it is Li's rebuild-every-candidate loop written in the same language and on the
same arrays as mlsynth's. At 24 donors it costs 4.3x, at 87 donors 16.4x, so most
of the gap to her R is the algorithm and not the interpreter. Her MATLAB is slower
than her R by an order of magnitude here, which is Octave and not MATLAB: Octave's
interpreter is not MathWorks' JIT, and the loop is interpreted per candidate.

## Where the difference comes from

Li's two implementations rebuild each candidate's donor average from scratch:
`rowMeans(x[1:t1, c(selected, j)])` for every remaining `j` at every step. Over
the whole path that is `O(N^3 T0)` in arithmetic and `O(N^2)` candidate
evaluations.

mlsynth centres the donors once, caches each donor's squared norm and its
cross-product with the treated unit, and keeps a running centred sum of the
selected set. A step is then one matvec that scores every candidate at once:
`O(N T0)` arithmetic and one interpreter-level operation, so `O(N^2 T0)` and
`O(N)` over the path.

## The exponents are approached, not reached

Synthetic panels at `T0 = 40`, donors 20 to 320, median of 15:

| donors | mlsynth | naive | ratio |
| --- | --- | --- | --- |
| 20 | 0.00086 s | 0.00292 s | 3.4x |
| 40 | 0.00120 s | 0.01162 s | 9.7x |
| 80 | 0.00196 s | 0.04669 s | 23.8x |
| 160 | 0.00456 s | 0.23070 s | 50.6x |
| 320 | 0.00749 s | 1.16613 s | 155.8x |

Fitted over that range the naive slope is 2.16 where the arithmetic predicts
3.00, and mlsynth's is 1.36 over 320 to 2560 donors where the arithmetic
predicts 2.00. Both fall short, and by about the same amount, which is the
signature of a cost that is part arithmetic and part fixed per-operation overhead.
Three measurements support that reading:

- Raising `T0` from 40 to 400 moves the naive slope from 2.30 to 2.80. More
  arithmetic per candidate, and the cubic asymptotic comes into view.
- The naive local slopes climb monotonically with the donor count: 1.99, 2.01, 2.30, 2.34
  over 20 to 40, 40 to 80, 80 to 160, 160 to 320. Approaching 3 from below.
- So do mlsynth's: 1.13, 1.33, 1.64 over 320 to 640, 640 to 1280, 1280 to 2560. Approaching 2 from below.

Those local slopes need repetitions to be readable. At five they came out
non-monotone for mlsynth and moved by up to 1.0 between runs, because the cells
run in 8 to 146 milliseconds; the figures above are medians of
15, where both series are monotone.

So the honest statement of the difference is the count of interpreter-level
operations, `O(N)` against `O(N^2)`, which is what the measured exponents separate
by. The arithmetic asymptotics are the limit both approach and neither reaches at
the sizes a panel study actually uses.

## What is not claimed

Her MATLAB timing is Octave's, not MATLAB's, and MATLAB's JIT would close much of
that particular gap. Nothing here says her code is wrong: it selects exactly what
mlsynth selects on both panels, and `benchmarks/cases/fdid_hongkong.py` pins
mlsynth against it to 3e-13. The selection is also rarely the binding cost in
applied work at these donor counts -- a millisecond against a hundred is a
difference that matters inside a bootstrap or a placebo sweep, where the fit is
repeated thousands of times, and nowhere else.

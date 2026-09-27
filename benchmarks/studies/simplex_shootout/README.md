# Simplex shootout: which solver belongs behind `solve_weights`

Nine algorithms for

    min_w  ||A - B w||^2    subject to    w >= 0,  sum(w) = 1

over 39 panels in 6 families, to answer one question: which one should the
library reach for by default, and where does a different one win by enough to
matter.

Run with `python benchmarks/studies/simplex_shootout/run.py`. Timings are the
best of repeated solves per panel; `results/raw.json` holds one row per
(panel, algorithm) with microseconds, iterations, terminal status, relative
objective excess against the best objective any solver reached, feasibility, and
the support size.

## The families

| family | panels | what it stands for |
| --- | --- | --- |
| `classic` | Basque, German reunification, Proposition 99 | the ordinary published panel: more matching rows than donors, support 3 to 7 |
| `factor` | low-rank designs with a handful of factors | supports 8 to 14 |
| `gaussian` | random wide designs | many donors against few rows, supports 6 to 127 |
| `montecarlo` | simulation DGPs the library's own cases use | repeated solves on one design |
| `ridged` | SDID's ridge-augmented programs | supports 14 to 65 |
| `degenerate` | exact-fit and coincident-donor designs | the optimum is a face, not a point |

## Correctness

The tolerance decides the verdict here, so it is stated and not assumed. A
solve counts as wrong when it returns an infeasible point, or when its objective
exceeds the best objective reached on that panel by more than the threshold.
Panels wrong out of 39:

| algorithm | 1e-12 | 1e-10 | 1e-8 | 1e-6 | 1e-4 |
| --- | --- | --- | --- | --- | --- |
| `active_set` | 0 | 0 | 0 | 0 | 0 |
| `active_set_cold` | 0 | 0 | 0 | 0 | 0 |
| `osqp` | 0 | 0 | 0 | 0 | 0 |
| `pairwise_smo` | 0 | 0 | 0 | 0 | 0 |
| `away_frank_wolfe` | 1 | 0 | 0 | 0 | 0 |
| `becker_kloessner` | 1 | 0 | 0 | 0 | 0 |
| `fista_projected` | 10 | 1 | 0 | 0 | 0 |
| `clarabel` | 31 | 20 | 0 | 0 | 0 |
| `frank_wolfe` | 29 | 29 | 29 | 25 | 0 |

One algorithm fails and it is `frank_wolfe`: 29 panels at 1e-8 and 25 still at
1e-6, which is a wrong answer at any tolerance a caller would set. Its away-step
variant fixes it, at 1 panel at 1e-12 and none beyond.

`clarabel` is the row to read carefully. Its worst relative excess over all 39
panels is 5.28e-09 and its median is 1.00e-10, so it clears every threshold from
1e-8 out. Counting it as wrong on 9 panels is what comes of reading the table at
1e-9, and at that threshold the quantity being measured is the conic solver's
interior-point stopping rule and not an error a caller would see. An earlier note
of ours quoted that 9 beside `frank_wolfe`'s 29 without saying that the two came
from different thresholds; they are not comparable that way.

## Speed

Total microseconds per family, relative to that family's own best, so 1.00 is the
winner of that column. The last column is each algorithm's worst family, which is
what a general-purpose default has to survive.

| algorithm | classic | degenerate | factor | gaussian | montecarlo | ridged | worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `active_set` | 1.02 | 1.07 | 1.00 | 1.98 | 1.17 | 1.21 | 2.0 |
| `osqp` | 2.32 | 2.60 | 3.22 | 13.78 | 1.14 | 1.06 | 13.8 |
| `clarabel` | 8.86 | 15.96 | 1.12 | 27.65 | 1.74 | 2.56 | 27.6 |
| `active_set_cold` | 1.00 | 1.00 | 2.14 | 28.87 | 1.70 | 1.27 | 28.9 |
| `pairwise_smo` | 135.76 | 82.50 | 14.57 | 1.00 | 1.00 | 1.00 | 135.8 |
| `frank_wolfe` | 233.19 | 253.00 | 22.28 | 48.65 | 33.86 | 43.45 | 253.0 |
| `away_frank_wolfe` | 319.96 | 242.40 | 36.83 | 3.29 | 4.46 | 3.68 | 320.0 |
| `fista_projected` | 424.73 | 216.77 | 47.87 | 22.10 | 62.08 | 69.70 | 424.7 |
| `becker_kloessner` | 54.52 | 88.42 | 79.19 | 1253.45 | 72.37 | 69.71 | 1253.4 |

## What it recommends

Keep the active set as the general-purpose solver. It wins `factor` outright, sits
within 7 per cent of the best on `classic` and `degenerate`, and its worst family
costs 2.0x. The next best worst-case is `osqp` at 13.8x, so the gap is not close.

`pairwise_smo` is the one alternative that earns a regime test. It wins `gaussian`,
`montecarlo` and `ridged`, and on `gaussian` the seeded active set costs 1.98x
against it. Those are the wide designs where donors outnumber rows, which is the
shape the simulation studies and the ridged SDID programs solve in bulk. Against
that, it costs 135.8x on `classic`, so it can only be a regime choice and never a
replacement.

`becker_kloessner` is the MSCMT sunny-donor reduction. Its 1253x on `gaussian` is
the cost of computing the reduction on a design where it prunes nothing: the
`min alpha` LP runs per donor, and on a random wide panel almost every donor is
sunny. It pays where a panel has many shady donors, which the families here do
not.

`clarabel` and `osqp` are the conic and ADMM references. Both are correct at every
practical tolerance and both are beaten on speed by the active set on five of six
families, which is the argument for a dedicated simplex routine over a
general-purpose QP.

## Limits

The timings are single-threaded wall clock on one machine, so read the ratios and
not the absolute microseconds. `degenerate` measures how fast a solver reaches a
face and says nothing about which point of that face it returns; which member
comes back is the subject of the identified-set work, not of this study.

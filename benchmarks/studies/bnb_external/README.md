# The certified branch and bounds against external solvers (#847)

#847 asks how fast the two certified searches are against established solvers
on the same instances, not only against our own enumeration. This study runs
SYNDES's two-way search against BiqCrunch and against SCIP, and MAREX's exact
backend against its own MIQP backend (SCIP), above the candidate limit.

This is a study, not a benchmark case. Nothing is pinned. The node counts are
deterministic; the seconds are one machine's.

## What is compared

The two-way design chooses its treated set by maximising `sigma'R sigma` over
sign vectors with `K` entries `+1` (`mlsynth/utils/syndes_helpers/gram.py`). The
design's value at a set is at least `4 alpha / sigma'R sigma`, with equality
where the set's closed form holds, so that maximisation is the problem's
combinatorial core: a binary quadratic program with one cardinality
constraint, the class BiqCrunch solves (Krislock, Malick and Roupin, ACM TOMS
43(4), 2017).

- Ours: `syndes_helpers.branch.branch_and_bound` with no work limit, on one
  thread, timed as process CPU. It certifies the design itself (the core plus
  the sets whose closed form breaks), every discard on a bound certified in
  exact arithmetic.
- BiqCrunch 2.0, built with gcc 13.3 and gfortran at `-O3` against reference
  LAPACK and BLAS, run as an external program (`biqcrunch.py`; it is GPL v3, so
  nothing of it is in this repository). Its input is the core in 0/1 variables
  (`sigma = 2z - 1`) with integer coefficients, its own requirement: they are
  multiplied by a per-instance scale and rounded, so no set of size `K` can pass
  2^28, under its 32-bit integers. A set's rounded value is within
  `(K(K-1)/2 + K) / 2` units of `scale * sigma'R sigma`, under 1e-6 of
  `sigma'R sigma` on every instance here. The cardinality constraint is given
  with its products by each variable, `z_i (sum_j z_j - K) = 0` (its paper's
  sec. 5.3), which our lift's `X r = 0` already holds. Its pruning runs in
  floating point. Times are its own CPU.
- `seeded`: BiqCrunch given the value the swap search's 32 restarts reach,
  less one, as ours starts from them.
- SCIP 10.0 through cvxpy 1.9.3: the library's `backend="mip"` on the whole
  two-way design, with a time limit of 1,200 seconds.

Every solve ran on one core of a machine with four, up to three at a time, so
the load average was 0.6 to 5.7 (each row records it); BLAS ran on one thread.
Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1.

```
python -m benchmarks.studies.bnb_external.run syndes noise 40 8 ours
BIQCRUNCH=/path/to/BiqCrunch/problems/generic/biqcrunch \
    python -m benchmarks.studies.bnb_external.run syndes noise 40 8 biqcrunch 1200
python -m benchmarks.studies.bnb_external.run syndes noise 40 8 scip 1200
python -m benchmarks.studies.bnb_external.run marex geolift 8 mip 1200
```

## SYNDES two-way, against BiqCrunch

The #834 panels at 40 units (`benchmarks/studies/syndes_global_bnb`): a panel
with no structure (`noise`), a four-factor panel (`factor`) and GeoLift's 40
markets, with 6, 8 and 10 treated: 3.8 million to 848 million sets each.

| panel | treated | ours: CPU s | nodes | BiqCrunch: CPU s | nodes | BiqCrunch / ours |
|---|---:|---:|---:|---:|---:|---:|
| noise | 6 | 8.3 | 500 | 91.1 | 693 | 11.0 |
| noise | 8 | 17.7 | 1,814 | 159.1 | 1,091 | 9.0 |
| noise | 10 | 28.5 | 285 | 144.7 | 831 | 5.1 |
| GeoLift | 6 | 4.5 | 293 | 38.3 | 287 | 8.6 |
| GeoLift | 8 | 16.9 | 653 | 123.6 | 625 | 7.3 |
| GeoLift | 10 | 11.8 | 423 | 66.0 | 307 | 5.6 |
| factor | 6 | 6.4 | 96 | 237.4 | 1,823 | 36.9 |
| factor | 8 | 82.4 | 378 | 823.0 | 4,743 | 10.0 |
| factor | 10 | 144.1 | 6,304 | 732.4 | 3,743 | 5.1 |

- Both certify every instance, and BiqCrunch returns our design on all nine.
- Over the nine, ours takes 321 CPU seconds and BiqCrunch 2,416: 7.5 times
  less in total, 5.1 to 36.9 times less per instance.
- `seeded` gives BiqCrunch the same nodes as its defaults on every instance,
  and times within 2.5%: the seed changes nothing it does.
- The two spend their time differently. BiqCrunch runs its bounding procedure
  at every node it evaluates and branches down to single sets. Ours bounds most
  nodes by the trust region alone, lifts where its cost model expects the lift
  to pay, and prices whole subtrees in closed form at 2.4 microseconds a set in
  that model. On the four-factor panel, where the bounds prune little, it
  prices almost every set: all 3,838,380 at six treated, 76.7 million of 76.9
  million at eight (16 lifts in 378 nodes).
- BiqCrunch prunes in floating point; ours discards a node only on a bound
  certified in exact arithmetic, and is faster anyway.

The root bounds, as a gap over the optimum's `sigma'R sigma` (lower is tighter):

| panel | treated | ours (the lift) | BiqCrunch, with the products | without |
|---|---:|---:|---:|---:|
| noise | 6 | 0.735% | 6.115% | 18.606% |
| noise | 8 | 2.106% | 5.443% | 6.041% |
| noise | 10 | 2.264% | 3.671% | 5.314% |
| GeoLift | 6 | 0.196% | 1.150% | 1.887% |
| GeoLift | 8 | 0.186% | 0.500% | 0.598% |
| GeoLift | 10 | 0.092% | 0.458% | 0.489% |
| factor | 6 | 0.429% | 0.894% | 0.747% |
| factor | 8 | 0.408% | 0.419% | 0.765% |
| factor | 10 | 0.332% | 0.312% | 0.392% |

- Our lift is BiqCrunch's bounding procedure, read from its source for the
  procedure and its defaults (`syndes_helpers/branch.py`). It ends tighter on
  eight of nine, in 0.25 to 1.60 CPU seconds against 0.18 to 1.37.
- The product constraints tighten BiqCrunch's root bound on eight of nine,
  most on noise with six treated (18.6% to 6.1%). On four-factor with six
  treated it is looser with them (0.894% against 0.747%).

## SYNDES two-way, against SCIP

`backend="mip"` with a time limit of 1,200 seconds. The gap is SCIP's, between
its design and its own bound when the limit stopped it; the last column is its
design's value over the optimum's.

| panel | treated | ours: CPU s | SCIP: CPU s | SCIP's gap | SCIP's design |
|---|---:|---:|---:|---:|---:|
| noise | 6 | 8.3 | 1,154 | 16.8% | +1.13% |
| noise | 8 | 17.7 | 1,156 | 9.9% | the optimum |
| noise | 10 | 28.5 | 1,157 | 12.6% | +0.77% |
| GeoLift | 6 | 4.5 | 1,178 | 29.4% | +2.75% |
| GeoLift | 8 | 16.9 | 1,181 | 13.4% | +1.59% |
| GeoLift | 10 | 11.8 | 1,181 | 26.8% | +1.33% |
| factor | 6 | 6.4 | 1,185 | 6.6% | the optimum |
| factor | 8 | 82.4 | 1,186 | 5.3% | +0.16% |
| factor | 10 | 144.1 | 1,187 | 6.2% | +0.15% |

- SCIP stops at its limit on all nine, 5.3% to 29.4% from closing; ours
  certifies each in 4.5 to 144 CPU seconds.
- SCIP's design is the optimum on two of nine and 0.15% to 2.75% worse on the
  other seven. On the two where it holds the optimum it cannot show it.

## MAREX, exact against MIP

`MAREX(..., backend="exact")` against `backend="mip"` with `time_limit=1200`,
inference off: the Walmart case's configuration
(`benchmarks/cases/marex_walmart.py`) with 6 and 8 treated, and GeoLift's 40
markets with `T0 = 90` and 8 treated. The last column is the MIP design's
objective over the exact backend's, which is the optimum; the two backends
report the same objective (`mlsynth/tests/test_marex_exact_backend.py` compares
them).

| panel | treated | exact: CPU s | MIP: CPU s | MIP's objective over the optimum |
|---|---:|---:|---:|---:|
| Walmart | 6 | 5.2 | 1,186 | +51.3% |
| Walmart | 8 | 205.0 | 1,189 | +65.8% |
| GeoLift | 8 | 39.1 | 1,190 | +27.1% |

- The MIP runs to its limit on all three: 1,202 seconds of wall time against a
  limit of 1,200. The row does not record SCIP's status, so the wall time is how
  it is read here.
- Its designs share 2, 0 and 1 treated units with the optimum.

## Instruments

- `biqcrunch.py`: the 0/1 form, its integer rounding (`Instance`), the BC
  file, the parameter file and the summary BiqCrunch prints.
  `test_biqcrunch.py` holds the form to `sigma'R sigma` at every set, the
  rounding to its stated error, the BC file to the problem it states (read
  back with a reader written from the user's guide), and the parser to both
  kinds of summary. Where `BIQCRUNCH` names a built binary, it also checks that
  BiqCrunch returns the enumeration's optimum on three 12-unit panels.
- `run.py`: one JSON line per solve in `results/`. `test_run.py` checks each
  command's row and that an unknown solver or backend is refused.

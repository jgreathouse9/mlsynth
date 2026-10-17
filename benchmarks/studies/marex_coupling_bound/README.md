# MAREX's weakly-targeted search: a bound that sees the control pool

Issue #770 asks for a lower bound on a weakly-targeted candidate's objective
that includes some of the coupling term `beta ||B_S w - B_C v||^2`, so that the
exact search can discard candidates before fitting them. This study measures
the issue's three leads on GeoLift and prototypes the one that works: a bound on
the distance between the treated hull and the control hull, computed by Wolfe's
nearest-point method. At `beta = 100` it leaves 579 coupled fits where the
shipped search makes 82,296, and the search returns the same design. Run in
lockstep over batches of candidates, the same bound takes that search from 332 s
to 20 s; with four worker processes bounding and fitting batches as they free
up, it takes 5.3 s.

This is a study, not a benchmark case, and it changes nothing in the library.
`bound.py` is a prototype; `search.py` puts it in front of `coupled_fit` from
the outside. Timings are machine-dependent and nothing here is pinned.

```
python benchmarks/studies/marex_coupling_bound/search.py
python benchmarks/studies/marex_coupling_bound/leads.py
python benchmarks/studies/marex_coupling_bound/borrowed.py
python benchmarks/studies/marex_coupling_bound/profile_bound.py
python benchmarks/studies/marex_coupling_bound/batched_curve.py
python benchmarks/studies/marex_coupling_bound/batched_search.py
python benchmarks/studies/marex_coupling_bound/levers.py
python benchmarks/studies/marex_coupling_bound/pool.py
python benchmarks/studies/marex_coupling_bound/parallel.py
```

Run them from the repository root, `search.py` first: the others read the
optimum it records.

## What is here

| file | what it does |
| --- | --- |
| `panel.py` | the GeoLift panel as one MAREX cluster, and the optimum `search.py` recorded |
| `bound.py` | the prototype: the hull-distance bound by Wolfe's method, and lead 1's dual completion |
| `search.py` | the exact search as shipped and with the bound in front of each fit, end to end |
| `leads.py` | each lead per candidate: its share of the objective, its cost, the fits it leaves |
| `borrowed.py` | lead 2: separating directions and a dual pair borrowed from fitted candidates |
| `profile_bound.py` | where the per-candidate bound's time goes |
| `batched.py` | the bound for a batch of candidates per numpy call: Wolfe's method, and Frank-Wolfe |
| `batched_curve.py` | the batched bounds per candidate: how many they clear, their cost, their validity |
| `batched_search.py` | the search driven by the batched bound, beside the per-candidate and shipped searches |
| `contiguous.py` | the batched bound with the running candidates contiguous, warm starts, and a pool worker |
| `levers.py` | two levers on the batched bound: fewer copies, and warm starts |
| `pool.py` | the batched bound across 1, 2 and 4 worker processes: the bound pass, and the whole search |
| `parallel.py` | the fits in the workers too, in rounds and streamed, against one process |
| `results/*.json` | the run behind the tables below |

## The bound

For a treated set `S` with control pool `C`, write `P = conv(B_S)` and
`Q = conv(B_C)`. The weakly-targeted objective is
`f(S) = min over t in P, c in Q of ||A - t||^2 + beta ||t - c||^2`, and a
minimum of a sum is at least the sum of the minimums, so

```
f(S) >= g(S) + beta * dist(P, Q)^2,
```

with `g(S)` the treated fit the search computes before any coupled fit. The
constant the issue left open is 1.

`P - Q` is the convex hull of the differences `b_i - b_j`, so `dist(P, Q)` is
the norm of that polytope's point nearest the origin. For any `x` in it, Wolfe's
Theorem 2.2 gives

```
dist(P, Q) >= (min over i in S of x'b_i - max over j in C of x'b_j) / |x|,
```

the separating-hyperplane dual of the distance (Boyd and Vandenberghe 2004,
sec. 8.2.3; it follows from Fenchel's Duality Theorem, Rockafellar 1970,
Theorem 31.1). The right side reads the data
only through the products `x'b_j`, so any `x` certifies: an iterate stopped
early or computed inaccurately cannot make the bound invalid, which is the
issue's floating-point requirement. Wolfe's method moves `x` toward the nearest
point and stops as soon as `g + beta * bound^2` clears the cutoff. A step takes
the extreme point of `P - Q` along `-x`, which is `b_i - b_j` with `i` the
minimiser of `x'b_i` over `S` and `j` the maximiser over `C`, so each step costs
one product `B'x` and never forms the `m (n - m)` differences. The derivation
and the implementation notes are in `bound.py`.

## The panel and the run

GeoLift's 40 markets as one cluster, the 90 pre-periods of
`basedata/geolift_market_data.csv` as the fit window, no covariates, four
treated markets: 91,390 candidates. The shipped search fits 82,296 of them at
`beta = 100`; the issue reports 82,415 on a configuration it does not state, so
this is the same regime, not necessarily the same panel.

Python 3.13.16 and NumPy 2.5.3, BLAS threads pinned to one, one process at a
time, after one untimed run of each search.

## The search, end to end

`search.py` replaces `exact.coupled_fit` for the second run with a guard that
computes the bound first and discards the candidate when the bound clears the
cutoff, which is the fit's own way of discarding. The incumbent and the order
of candidates are the search's.

| beta | coupled fits, shipped | with the bound | discarded by the bound | seconds, shipped | with the bound | design |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 234 | 5 | 229 of 233 | 0.92 | 0.55 | identical |
| 100 | 82,296 | 579 | 81,717 of 82,295 | 329 | 132 | identical |

Identical means the same treated set and the same objective to the last digit.
An earlier run in the same container, without the thread pinning, took 199 s
and 109 s at `beta = 100`; the speedup is 1.8 to 2.5 across the two runs, and
the counts are the same in both.

The fits are no longer where the time goes. The 82,295 candidates that reach
the guard each pay for the bound, about 1.6 ms at `beta = 100` in this run,
which is most of the 132 s.

## The leads, per candidate

At `beta = 100`, `leads.py` samples 1,000 of the 87,439 candidates the
closed-form bound leaves; 926 of them have a treated fit under the optimum and
reach the coupled fit. At `beta = 1` it takes all 234. Shares are of the
candidate's exact objective, as the 10th, 50th and 90th percentiles. "Left to
fit" counts the candidates a bound leaves under the optimum, scaled to the whole
search: the count once the incumbent is optimal, which is why the end-to-end
search fits more. Every bound stayed at or under the exact objective on every
sampled candidate.

`beta = 100`:

| lead | share of f | left to fit | ms per candidate |
| --- | --- | ---: | ---: |
| 1, value less the Frank-Wolfe gap at the treated-only start | -3.77, -1.43, 0.05 | 77,209 | 2.9 |
| 1, completed as a dual bound (Hazimeh et al. 2022, sec. 3.2) | 0.00, 0.003, 0.32 | 71,438 | the same computation |
| 3, hull distance by Wolfe's method, stopped once it clears | 0.23, 0.35, 0.53 | 437 | 1.6 |
| 3, exact hull distance (Clarabel) | 0.46, 0.72, 0.97 | 437 | 14.3 |
| the cutoff-stopped coupled fit the bound replaces | | | 4.3 |

`beta = 1`:

| lead | share of f | left to fit | ms per candidate |
| --- | --- | ---: | ---: |
| 1, value less the Frank-Wolfe gap | 0.14, 0.42, 0.69 | 204 | 2.2 |
| 1, completed as a dual bound | 0.33, 0.51, 0.71 | 192 | the same computation |
| 3, hull distance by Wolfe's method | 0.60, 0.68, 0.77 | 5 | 1.9 |
| 3, exact hull distance | 0.76, 0.88, 0.97 | 5 | 14.8 |
| the cutoff-stopped coupled fit | | | 3.1 |

Wolfe's method takes a median of 12 major cycles at `beta = 100` and 14 at
`beta = 1`, never more than 31. Its share is a floor, since it stops once it
clears; it leaves the same candidates as the exact distance at about a ninth
of the cost.

Lead 1 measures the coupled objective at the treated-only start, which at
`beta = 100` sits far from the optimum, and its gap is larger than its value.
The dual completion fixes the sign but not the size.

## Lead 2: borrowed certificates

`borrowed.py` tries the cheapest form of the issue's second lead: certificates
from candidates already fitted, applied to the rest. Both kinds are valid for
every candidate; the question is whether they carry over.

| beta | under the optimum, closed form | with 1 borrowed direction | 16 | 64 | 256 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 234 | 234 | 234 | 225 | 125 |
| 100 | 87,439 | 87,439 | 87,438 | 87,430 | 87,382 |

The directions are the residuals of the 256 candidates with the smallest
closed-form bound. A dual pair from the optimal design is exact at that design
(its share is 1 to 11 digits) and certifies nothing on 300 random candidates at
`beta = 100` (largest share 0.0). At `beta = 100` a certificate fitted to one
treated set does not transfer: the bound that discards a candidate is that
candidate's own.

## The bound in batches

The per-candidate bound is not slow arithmetic. `profile_bound.py` puts it at
1.4 ms a candidate, 12.8 major cycles and 15.9 least-squares calls on average,
with 43% of the time inside those calls and 32% in the method's own Python. One
least-squares solve on a 90-by-8 corral takes 40.7 microseconds as a numpy call
for about 12,000 floating-point operations; one product `B'x` takes 1.8. The
shipped search prices every candidate in about a microsecond because it batches
20,000 of them per numpy call, and `batched.py` does the same for this bound.

Batched, the bound uses the closed-form treated bound `lb` in place of the
treated fit `g`, so a candidate needs no simplex fit to be bounded. That costs
nothing at `beta = 100`: with exact distances, `lb + beta * dist^2` and
`g + beta * dist^2` leave the same candidates, 1 of 1,000 sampled. A candidate
needs its bound to reach 61% of the exact distance at the median, 78% at the
90th percentile and 94% at the 99th (`batched_curve.py`), so a method has to
come close to the distance, not only start toward it.

Frank-Wolfe does not. Run in lockstep for at most 200 iterations on all 87,439
candidates the closed-form bound leaves under the optimum at `beta = 100`:

| method | cleared within 10 | 50 | 100 | 200 | left after 200 | microseconds per candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frank-Wolfe | 266 | 9,877 | 26,695 | 53,412 | 34,027 | 373 |
| pairwise Frank-Wolfe | 48 | 660 | 2,603 | 9,039 | 78,400 | 666 |

The pairwise variant moves both simplexes by one step size, which the block
with less mass to move limits; separate steps per block might do better, and
are not measured. Both fall far short of Wolfe's method, whose exact affine
steps reach the distance in a few cycles.

So `wolfe_bound` batches Wolfe's method. Every quantity goes through the Gram
matrix of the target-centred markets: a corral point is a pair of indices,
`b_i - b_j`, its products with every market are `G[i] - G[j]`, and each
candidate keeps those products and its corral's Gram, updated when a point is
added. It clears 86,855 of the 87,439 and leaves 584: 422 converge without
clearing and 162 fill their 24 corral slots. The cost per candidate depends on
the batch size, because each lockstep iteration copies arrays that grow with
it:

| batch | microseconds per candidate at `beta = 100` | at `beta = 1` |
| ---: | ---: | ---: |
| 512 | 174 | 296 |
| 2,048 | 178 | 239 |
| 8,192 | 213 | 237 |

At `beta = 1` it clears 229 of the 234 candidates, the same 5 left as the exact
distance.

The bound reads the data through `G`, so its rounding is that of forming `G` and
of the products. Run to convergence on 300 sampled candidates at `beta = 100`,
it matches the per-candidate exact distance to 2.9e-11 relative at the median,
and its largest value is 2.7e-12 above it: rounding between two evaluation
routes, six orders of magnitude inside the search's margin. The margin, not the
arithmetic, keeps it valid, so a version in the library needs the a priori
rounding bound pinned, as the bordered-pricing study does for the treated bound.

## The search with the batched bound

`batched_search.py` runs a search driven by the batched bound beside the other
two, in one process. It fits the first 8 candidates in `lb` order to set an
incumbent, then bounds each batch of 512 against the cutoff at the batch's
start and fits the rest in `lb` order, as the shipped search would.

| beta | search | coupled fits | discarded by a bound | seconds | design |
| ---: | --- | ---: | ---: | ---: | --- |
| 100 | shipped | 82,296 | | 332 | identical |
| 100 | per-candidate bound | 579 | 81,717 | 131 | identical |
| 100 | batched bound | 754 | 86,685 | 20 | identical |
| 1 | shipped | 234 | | 0.73 | identical |
| 1 | per-candidate bound | 5 | 229 | 0.51 | identical |
| 1 | batched bound | 11 | 223 | 0.24 | identical |

The batched search fits more candidates than the per-candidate one: its first
incumbent comes from 8 fits, it bounds with `lb`, and a batch uses the cutoff
it started with. At `beta = 100` the bound takes about 15 of the 20 s
(87,431 candidates at about 174 microseconds) and the fits about 3.

## Three levers, measured

`levers.py` and `pool.py` measure three ways to cut the batched bound's cost:
fewer copies, warm starts, and a pool of worker processes. Timings of the same
computation move by about 15% between runs: `levers.py` bounds the same
candidates, in the same order, with the same function twice (the contiguous
bound at batches of 512, and the cold start the warm starts are compared with)
and records 180 and 156 microseconds a candidate. Effects smaller than that are
reported as such.

### Fewer copies

`contiguous.py` keeps the running candidates contiguous at the front of the
batch, runs the major step's products on a view of that block with masks,
gathers only the slices an append needs, and compacts the block when a quarter
of it has finished. The statuses equal the committed batched bound's everywhere,
at both `beta`. At `beta = 100` and batches of 512 it takes 180 microseconds a
candidate against 188, and 161 against 175 in an earlier run: 4 to 8%, inside
the run-to-run spread. The copies were not the cost.

### Warm starts

Candidates sharing their first three markets form a chain in lexicographic
order, and each starts from its predecessor's final corral with the points that
stay valid (Wolfe 1976, sec. 9). At `beta = 100`, on the 87,439 candidates:

| start | major steps a candidate | inherited points | microseconds a candidate | cleared |
| --- | ---: | ---: | ---: | ---: |
| cold | 13.68 | | 155.6 | 86,855 |
| the predecessor's whole corral | 6.77 | 8.69 | 188.5 | 86,853 |
| its 3 heaviest points | 11.19 | 3 | 138.9 | 86,855 |
| its heaviest point | 13.28 | 1 | 157.4 | 86,853 |

Inheriting the whole corral halves the major steps and still costs more: every
lockstep step then works on arrays as wide as an 8.7-point corral, and the minor
cycles that prune it come first. Three points come out 11% below the cold start,
near the run-to-run spread. Warm starts also need candidates in lexicographic
chains, while the search visits them in bound order, so they are not combined
with the search here. At `beta = 1`, with 234 candidates in chains of up to 13,
every warm start is slower than the cold one: the later rounds are small
batches.

### A pool of worker processes

Issue #768's pool, on a machine with 4 cores. The workers fork, run
single-threaded, and get the Gram matrix and pair distances once. The bound
pass alone at `beta = 100`, all 87,439 candidates in batches of 512:

| workers | seconds | speedup | statuses equal the single process's |
| ---: | ---: | ---: | --- |
| 1 | 15.9 | | |
| 2 | 7.27 | 2.19 | yes |
| 4 | 3.86 | 4.12 | yes |

The speedups exceed the ideal because this run's single process, 182
microseconds a candidate, sits at the slow end of the spread; an earlier run
gave 14.3, 7.28 and 4.02 seconds, speedups of 1.96 and 3.55.

The whole search, with each round bounding as many batches as there are workers
against the cutoff at the round's start, then fitting what they leave in the
parent:

| beta | search | seconds | coupled fits | design |
| ---: | --- | ---: | ---: | --- |
| 100 | committed batched bound | 18.25 | 754 | identical |
| 100 | contiguous, 1 worker | 17.85 | 754 | identical |
| 100 | contiguous, 2 workers | 10.72 | 781 | identical |
| 100 | contiguous, 4 workers | 7.68 | 837 | identical |
| 1 | committed batched bound | 0.21 | 11 | identical |
| 1 | contiguous, 1 worker | 0.09 | 11 | identical |
| 1 | contiguous, 2 workers | 0.13 | 11 | identical |
| 1 | contiguous, 4 workers | 0.13 | 11 | identical |

The fits rise with the workers because a round's batches share one cutoff. At
`beta = 1` the bound pass is 0.06 s and the pool's overhead outweighs it.

## The fits in the workers too

With the bound in the workers and the fits in the parent, the fits are about
half of the `beta = 100` search at four workers. `parallel.py` moves them into
the workers, as issue #768 proposes: each worker bounds a batch of 512 against
the best value known when the batch was sent, fits what the bound leaves in
`lb` order, and returns the batch's best design; the parent merges designs with
the search's tie rule. Two schedules: rounds of one batch per worker, each round
waiting for all of them, and streaming, with a fixed number of batches in
flight and the next sent, with the current best, as soon as one returns. The
baseline is the same logic run in one process.

| beta | run | seconds | speedup | coupled fits | design |
| ---: | --- | ---: | ---: | ---: | --- |
| 100 | one process | 19.42 | | 754 | identical |
| 100 | rounds, 2 workers | 10.89 | 1.78 | 781 | identical |
| 100 | streaming, 2 workers, 2 in flight | 9.72 | 2.00 | 796 | identical |
| 100 | streaming, 2 workers, 4 in flight | 9.73 | 2.00 | 858 | identical |
| 100 | rounds, 4 workers | 6.47 | 3.00 | 837 | identical |
| 100 | streaming, 4 workers, 4 in flight | 5.26 | 3.69 | 891 | identical |
| 100 | streaming, 4 workers, 8 in flight | 5.65 | 3.44 | 968 | identical |

Rounds leave workers idle: the candidates that survive the bound cluster in a
few batches, and a round lasts as long as its slowest one. Streaming keeps every
worker busy and reaches 3.69 times one process on four. Keeping more batches in
flight than workers only sends them with staler bests, so they fit more.

The design is the same in every run, here and at `beta = 1`, so it does not
depend on the workers or the schedule; the fit count does. Under streaming it
also moves between runs of the same configuration, 891 here and 863 in an
earlier run, since which batch returns first decides the best the next one is
sent with. At `beta = 1` every run takes 0.09 to 0.13 s and one process is the
fastest: the pass is too small for a pool to pay.

Four cores were all this machine had; issue #768's eight workers are not
measured.

## What is left

At four workers, streamed, the `beta = 100` search takes about 5 s against the
shipped 332 s. None of these is measured.

* Narrower lockstep arrays. A batch's arrays are as wide as its largest
  corral; grouping candidates by corral size would narrow them.
* Factor updates. The batched affine step solves the corral's normal equations
  afresh each time; Wolfe's sec. 5 updates a triangular factor instead.
* More workers, on a machine with more cores.

The per-candidate affine step is solved on the corral's edges,
`min ||z_0 + E alpha||`. The bordered Gram system that states it most directly
is unusable at GeoLift's scale: its Gram entries are of order 1e7 against a
border of 1, a least-squares solve returns `w = 0`, and the method stalls at its
first step. The batched step solves the edges' normal equations, measured from a
reference point, with a ridge of 1e-13 of their largest diagonal; a version in
the library needs a test that pins the scale either way.

## Sources

* Wolfe, P. (1976). Finding the nearest point in a polytope. *Mathematical
  Programming* 11, 128-149. The algorithm, the bound (Theorem 2.2), two point
  sets (sec. 10), warm starts (sec. 9).
* Gilbert, E. G., Johnson, D. W. and Keerthi, S. S. (1988). A fast procedure for
  computing the distance between complex objects in three-dimensional space.
  *IEEE Journal of Robotics and Automation* 4(2), 193-203. The support mapping of
  a Minkowski difference.
* Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*, sec. 8.2.3. The
  duality between distance and separation.
* Rockafellar, R. T. (1970). *Convex Analysis*. Princeton University Press.
  Fenchel's Duality Theorem (Theorem 31.1), from which the distance equals the
  best separation and the dual bound is exact at a fitted candidate, and
  Fenchel's inequality in its proof, the weak duality the bounds rest on; the
  support function of a finite convex hull and of a sum of sets (Corollaries
  16.5.1 and 16.4.1); the indicator and support function of a closed convex set
  as conjugates (Theorem 13.2); strong separation of disjoint polytopes
  (Corollary 11.4.2).
* Hazimeh, H., Mazumder, R. and Saab, A. (2022). Sparse regression at scale:
  branch-and-bound rooted in first-order optimization. *Mathematical
  Programming* 196, 347-388. Dual bounds completed from inexact primal points.
* Lacoste-Julien, S. and Jaggi, M. (2015). On the global linear convergence of
  Frank-Wolfe optimization variants. *Advances in Neural Information Processing
  Systems* 28. The pairwise variant, and the rate behind Wolfe's short cycle
  counts.
* Das Gupta, S., Van Parys, B. P. G. and Ryu, E. K. (2023). Branch-and-bound
  performance estimation programming: a unified methodology for constructing
  optimal optimization methods. arXiv:2203.07305v5. Its pool of lazily added
  cuts (sec. 4.2.2) is the form lead 2 takes here.
* Montanari, M., Petrinic, N. and Barbieri, E. (2017). Improving the GJK
  algorithm for faster and more reliable distance queries between convex
  objects. *ACM Transactions on Graphics* 36(3), article 30. Its subalgorithm is
  specific to three dimensions and does not apply at 90.

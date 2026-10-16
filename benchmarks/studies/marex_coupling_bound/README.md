# MAREX's weakly-targeted search: a bound that sees the control pool

Issue #770 asks for a lower bound on a weakly-targeted candidate's objective
that includes some of the coupling term `beta ||B_S w - B_C v||^2`, so that the
exact search can discard candidates before fitting them. This study measures
the issue's three leads on GeoLift and prototypes the one that works: a bound on
the distance between the treated hull and the control hull, computed by Wolfe's
nearest-point method. At `beta = 100` it leaves 579 coupled fits where the
shipped search makes 82,296, and the search returns the same design.

This is a study, not a benchmark case, and it changes nothing in the library.
`bound.py` is a prototype; `search.py` puts it in front of `coupled_fit` from
the outside. Timings are machine-dependent and nothing here is pinned.

```
python benchmarks/studies/marex_coupling_bound/search.py
python benchmarks/studies/marex_coupling_bound/leads.py
python benchmarks/studies/marex_coupling_bound/borrowed.py
```

Run them from the repository root, `search.py` first: the other two read the
optimum it records.

## What is here

| file | what it does |
| --- | --- |
| `panel.py` | the GeoLift panel as one MAREX cluster, and the optimum `search.py` recorded |
| `bound.py` | the prototype: the hull-distance bound by Wolfe's method, and lead 1's dual completion |
| `search.py` | the exact search as shipped and with the bound in front of each fit, end to end |
| `leads.py` | each lead per candidate: its share of the objective, its cost, the fits it leaves |
| `borrowed.py` | lead 2: separating directions and a dual pair borrowed from fitted candidates |
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

the separating-hyperplane dual of the distance. The right side reads the data
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

## What would cut the bound's cost

None of these is measured.

* Warm starts. The search visits candidates in bound order, but in
  lexicographic order neighbours share three of their four treated markets, and
  Wolfe's method restarts from a solved corral when points are added or deleted
  (Wolfe 1976, sec. 9); Hazimeh et al. warm-start every node from its parent.
  Bounding in lexicographic order, then fitting in bound order, keeps the
  search's order for the fits.
* Factor updates. Each affine step solves a fresh least-squares problem on the
  corral's edges; Wolfe's sec. 5 updates a triangular factor instead.
* Issue #768's process pool. The bound needs only the cutoff, so the pass over
  candidates divides across processes as the fits would.

The affine step is solved on the corral's edges, `min ||z_0 + E alpha||`. The
bordered Gram system that states it most directly is unusable at GeoLift's
scale: its Gram entries are of order 1e7 against a border of 1, a
least-squares solve returns `w = 0`, and the method stalls at its first step. A
version in the library needs the factored form of Wolfe's sec. 5 and a test that
pins the scale.

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
* Hazimeh, H., Mazumder, R. and Saab, A. (2022). Sparse regression at scale:
  branch-and-bound rooted in first-order optimization. *Mathematical
  Programming* 196, 347-388. Dual bounds completed from inexact primal points.
* Lacoste-Julien, S. and Jaggi, M. (2015). On the global linear convergence of
  Frank-Wolfe optimization variants. *Advances in Neural Information Processing
  Systems* 28. The rate behind the short cycle counts.
* Das Gupta, S., Van Parys, B. P. G. and Ryu, E. K. (2023). Branch-and-bound
  performance estimation programming: a unified methodology for constructing
  optimal optimization methods. arXiv:2203.07305v5. Its pool of lazily added
  cuts (sec. 4.2.2) is the form lead 2 takes here.
* Montanari, M., Petrinic, N. and Barbieri, E. (2017). Improving the GJK
  algorithm for faster and more reliable distance queries between convex
  objects. *ACM Transactions on Graphics* 36(3), article 30. Its subalgorithm is
  specific to three dimensions and does not apply at 90.

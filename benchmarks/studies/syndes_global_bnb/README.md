# SYNDES two-way design above the candidate limit: node bounds and a search

`mode="two_way_global"`'s exact backend enumerates every admissible treated set
under the closed-form bound of `gram.py` and refuses to certify past twenty
million sets, where it falls back to a swap search reported against the
Rayleigh bound. #803 asks for a certified search above that limit. This study
measures what a node bound must reach to prune, builds the bounds that reach
it, solves the strongest one without an interior-point solver, and measures
where a branch and bound spends its time against enumeration as it stands on
`main` after #828. It applies `agents/agents_combinatorial.md`: measure the
ceiling first (rule 5), bound only where pricing costs more (rule 6), and
certify what a solver returns (rule 3).

This is a study, not a benchmark case. Nothing here is pinned, and the timings
are machine-dependent.

## The node and its bounds

A node fixes a set `I` of treated units and a set `O` of controls and leaves
`F` free. With `sigma` the design's +/-1 indicator, the closed form of
`gram.py` bounds each set, `f(S) >= 4 alpha / sigma'R sigma`, so a node is
bounded by `4 alpha / U` for any `U >= max sigma'R sigma` over its subtree.
Splitting `sigma` into its decided and free parts,

    sigma'R sigma = c + 2 b'x + x'A x,    x in {-1, +1}^n,  1'x = s = 2r - n.

One-way's optimum is at least two-way's, so every bound here bounds one-way too.

- `trs_upper` relaxes `x` to the sphere `|x|^2 = n` cut by the plane
  `1'x = s`: a trust-region subproblem in an orthonormal basis of the plane's
  direction, whose Lagrangian dual `const + mu rho^2 + g'(mu I - B)^-1 g` is a
  bound for every `mu` above `B`'s top eigenvalue. Newton's method on the
  secular equation (Moré and Sorensen 1983), started left of the root, stays
  left of it, so every iterate's dual is a bound. At the root it is the
  Rayleigh bound `certify` reports.
- `spectral_upper`, `c + 2 |b|_1 + lambda_max(A) n`, takes the linear term
  over the box. Neither it nor `trs_upper` dominates on small nodes, so
  `node_upper` takes the smaller.
- `lift_upper` is Shor's relaxation of the node with the cardinality on the
  lift and the four products `(1 +- x_i)(1 +- x_j) >= 0`, the sign-vector form
  of the doubly nonnegative relaxation, solved by Clarabel. Its value is
  rebuilt from the solver's multipliers as `const + (n + 1) lambda_max(M)`,
  valid for any multipliers, since a design's lifted matrix is semidefinite
  with trace `n + 1` (rule 3; the construction of MAREX's `persp_n.certified`,
  in floating point).
- `dual_upper` (`dual.py`) bounds the same relaxation without an
  interior-point solve. The certificate above is a function of the
  multipliers, so it is minimised directly, in the form of BiqCrunch's bounding
  procedure: L-BFGS-B on a regularised dual (`const + |C_+|_F^2 / (2 alpha) +
  alpha (n + 1)^2 / 2`, valid since a design's lifted matrix has Frobenius norm
  `n + 1`), `alpha` halved from 0.1 to 5e-5, triangle inequalities added as
  cuts when violated by more than 0.05 and dropped when slack with a zero
  multiplier. The products are the triangle inequalities through the
  homogenising index; `cuts="triangles"` also separates those among the free
  units, every one valid at a design. Every evaluation reports the exact
  eigenvalue bound, so the method stops as soon as the node prunes. An optional
  give-up rule, off by default (`PATIENCE = inf`), abandons a node whose bound
  would take more than ten rounds at its last round's rate to reach the target;
  "Weighting the lift, and its cost" below has why it is off.

`test_bounds.py` and `test_dual.py` check every bound against brute force over
every completion: on random nodes of factor panels, on any symmetric matrix
(random, low rank, zero, negative definite, repeated eigenvalues) as a
Hypothesis property, at outcome scales `1e-3` and `1e3`, and both certificates
for arbitrary multipliers. They also pin tightness: the trust-region dual is
minimised (a 400-step bisection is the reference), the bound is exact with two
free units, the certified lift is the relaxation's value when the solve is
accurate, and the dual with the products alone reaches the lift's value.

## How tight a bound must be (`quality.py`)

Every set of a panel small enough to enumerate is priced, and the
include/exclude tree is walked with the incumbent at the optimum. For a bound at
a fraction `rho` of every subtree's minimum, the sets left to price:

| panel | sets | rho 0.98 | 0.99 | 0.995 | 0.999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor `N = 24, K = 4` | 10,626 | 5.1-10.2% | 1.0-3.2% | 0.3-0.5% | 0.0-0.3% |
| factor `N = 30, K = 5` | 142,506 | 3.6-9.6% | 0.3-2.0% | 0.0-0.4% | 0.0% |
| factor `N = 40, K = 4` | 91,390 | 11.4-15.9% | 1.8-3.4% | 0.3-0.6% | 0.0% |
| GeoLift `N = 40, K = 4` | 91,390 | 15.5% | 2.1% | 0.3% | 0.0% |
| noise `N = 24, K = 4` | 10,626 | 0.1-0.2% | 0.0-0.1% | 0.0-0.1% | 0.0% |
| noise `N = 30, K = 5` | 142,506 | 0.0% (45-48 sets) | 0.0% | 0.0% | 0.0% |

(`results/ceiling_*.log`; index order shown, the relaxed order is similar.)

The two kinds of panel differ in the shape of the objective near the optimum.
On factor panels and GeoLift it is flat: 0.9 to 3.1% of all designs lie within
2% of the optimum, and a bound must reach 99.5% of each subtree's minimum to
leave under 1% of the sets. On noise panels it is steep: at most 35 designs
lie within 2%, and a 98% bound leaves a few dozen sets.

## How tight the bounds are

Each bound's node bound over the subtree's minimum, at eight random nodes of
24-unit panels with four treated (`results/ratios.log`, `results/duals.log`):

| panel | trust region | Shor | lift | dual, products | dual, triangles |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor | 0.983-0.985 | 0.985-0.991 | 0.9925-0.9999 | 0.9919-0.9982 | 0.9925-0.9990 |
| noise | 0.880-0.911 | 0.915-0.961 | 0.9969-1.0000 | 0.9958-1.0000 | 1.0000 |
| GeoLift | 0.971-0.982 | 0.985-0.994 | 0.9954-1.0000 | 0.9943-0.9998 | 0.9955-1.0000 |

The trust-region bound sits at 97 to 98.5% on flat panels, where 98% is not
enough, and at 88 to 91% on steep ones, where 98% would be. The lift is at
99.25% or more on every panel, and 99.5% or more on GeoLift and the noise
panel; Shor's relaxation alone adds 0.2 to 0.6 points on
factor panels, so the products carry the bound, as entrywise nonnegativity
carried MAREX's (`agents_combinatorial.md`, rule 7). The triangles among the
free units add the rest on the noise panel.

At 32 to 40 free units on 40-unit panels, each dual's `U` over the lift's (above
1 is looser), and the seconds (`results/duals.log`):

| panel | lift | dual, products | dual, triangles |
| --- | ---: | ---: | ---: |
| GeoLift, 10 treated | 1.2-2.6 s | +0.06 to +0.20%, 0.23-0.74 s | -0.01 to +0.14%, 0.27-0.78 s |
| factor, 6 treated | 1.3-2.3 s | +0.10 to +0.36%, 0.30-1.26 s | +0.08 to +0.32%, 0.27-1.11 s |
| noise, 6 treated | 1.1-2.1 s | +0.17 to +0.42%, 0.77-1.83 s | -0.29 to -0.04%, 0.77-2.09 s |

The dual runs its whole schedule here (no target): one to seven times faster
than Clarabel, within 0.4% of its bound, and tighter than it on the noise panel.
In a search it stops sooner, at the prune target.

## The search (`search.py`)

Depth-first over the units, in the order of the root relaxation's maximiser.
A node is enumerated when pricing it costs less than a bound times the chance
the bound prunes, estimated as the search runs per subtree-size bucket;
otherwise it is bounded by the trust region, then lifted if pricing its subtree,
weighted by the lift's own prune rate at that size, costs more than a lift, then
split. The lift's cost starts from a calibration and is learned from the lifts
the search pays for (`LiftCost`). Leaves are priced in batches by the library's
closed form, and sets whose closed form breaks its sign conditions are solved by
projected gradient at the end. `test_search.py` checks the answer against the
library's enumeration on 16 panels, under every cost model, with forced and
forbidden units, from a poor incumbent, on a panel where no set has a closed
form, and with either lift.

The interior-point lift, unweighted as the search first ran, against the
library's enumeration on `main` after #828 (`results/search.log`,
`results/enumeration.log`; the library run with its candidate limit raised):

| panel | sets | enumeration | search | lifts | sets priced |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor `N = 40, K = 6` | 3.8M | 3.0 s | 33.2 s | 22 | 3,838,149 |
| noise `N = 40, K = 6` | 3.8M | 3.1 s | 11.0 s | 5 | 1,845,633 |
| factor `N = 60, K = 6` | 50.1M | 61.3 s | 101.4 s (no lift) | 0 | 50,062,682 |
| noise `N = 60, K = 6` | 50.1M | 65.1 s | 110.2 s (no lift) | 0 | 50,063,860 |
| GeoLift `N = 40, K = 8` | 76.9M | 63.4 s | 88.4 s | 46 | 3,817,331 |
| GeoLift `N = 40, K = 10` | 847.7M | 698.1 s | 150.2 s | 88 | 1,094,371 |
| noise `N = 40, K = 10` | 847.7M | 724.6 s | 266.9 s | 278 | 16,044 |
| factor `N = 40, K = 10` | 847.7M | 710.6 s | 1,167.8 s | 1,284 | 33,428 |

Every search returned the enumeration's optimum where both ran. Enumeration
prices 0.8 µs a set at 40 units and 1.2 µs at 60, so below the candidate limit
the search loses at every size measured: pricing is cheap enough that the bound
only adds cost. Above it, the search wins where its bounds prune: with ten
treated, 4.6 times faster than enumeration on GeoLift and 2.7 times on the noise
panel. On the flat factor panel 1,284 lifts leave it 1.6 times slower than
enumerating all 848 million sets.

## Where the time goes (`stages.py`)

Each stage's own seconds, the interior-point lift (`results/stages.log`):

| panel | seconds | in the lift | lifts | nodes pruned | the rest |
| --- | ---: | ---: | ---: | ---: | --- |
| factor `N = 40, K = 6` | 31.7 | 85.2% | 18 | 1 | pricing 10.5% |
| noise `N = 40, K = 6` | 24.1 | 96.8% | 10 | 9 | |
| GeoLift `N = 40, K = 8` | 99.1 | 94.0% | 52 | 714 | trust region 2.7% |
| GeoLift `N = 40, K = 10` | 75.3 | 97.4% | 47 | 22 | |
| factor `N = 60, K = 6`, no lift | 77.8 | | 0 | 73 | closed form 74.4%, sign matrices 14.5%, trust region 5.0% |

Whenever the lift runs it takes 85 to 97% of the search, at 1 to 2.9 s a lift
at 30 to 40 free units: the price of one to three million sets. The calibration
overstated a lift's cost by up to a factor of two, and since it is timed, the
same panel took 75.3 s with 47 lifts and 150.2 s with 88 (`stages.log`,
`search.log`).

## The lift without an interior-point solve (`dual.py`)

The semidefinite branch-and-bound codes for +/-1 quadratic programs solve their
node bounds this way. BiqMac (Rendl, Rinaldi and Wiegele) handles its triangle
inequalities by a bundle method on their Lagrangian. BiqCrunch (Krislock,
Malick and Roupin) minimises a regularised dual by L-BFGS-B and adds its
inequalities as cuts. Burer and Vandenbussche use an augmented Lagrangian whose
bound is valid at every iterate, solve loosely near the root, and warm-start a
node from its parent's multipliers. Buchheim, Montenegro and Wiegele run
coordinate ascent on the dual, whose constraint matrices have rank at most two,
as this lift's do. Liuzzi, Locatelli and Piccialli find a slightly weaker but
much cheaper bound the better choice inside a branch and bound, so what counts
is the search's time, not the bound's tightness. `dual.py` follows BiqCrunch's
procedure and parameters.

In the search, the dual with its triangles takes the lift's place
(`results/search_dual.log`, the lift unweighted, as in `search.log`):

| panel | interior-point lift | dual |
| --- | ---: | ---: |
| GeoLift `N = 40, K = 10` | 150.2 s, 88 lifts | 72.7 s, 251 lifts |
| GeoLift `N = 40, K = 8` | 88.4 s, 46 lifts | 110.0 s, 347 lifts |
| noise `N = 40, K = 10` | 266.9 s, 278 lifts | 59.1 s, 178 lifts |
| noise `N = 40, K = 6` | 11.0 s, 5 lifts | 8.9 s, 8 lifts |
| factor `N = 40, K = 6` | 33.2 s, 22 lifts | 63.0 s, 245 lifts |
| factor `N = 40, K = 10` | 1,167.8 s, 1,284 lifts | 1,128.7 s, 7,527 lifts |

Where lifts prune the dual is the faster lift, most on the noise panel, where its
triangles make it tighter than the interior-point lift as well. Where they do not
(the factor panel), a cheaper lift is a lift the cost rule pays for more often:
245 lifts, and five nodes pruned.

## Weighting the lift, and its cost

The cheap tier has been weighted by its prune rate since the first search's fix
(the third wrong turn below). The lift tier now is too: it lifts when its subtree's pricing,
times the lift's prune rate at that subtree size, costs more than a lift. Two
things had to change with it.

- The lift's cost. The calibration times a lift without a prune target, so the
  dual runs its whole schedule, while in the search it stops at the target.
  Weighted by a prune rate, the overstated cost drove the lift out. `LiftCost`
  starts from the calibration and learns from the lifts the search pays for.
- The give-up rule. With it on, lifts were cut short before they could prune,
  every abandoned lift counted against the prune rate, and the weight stopped
  lifting (`results/search_isolate.log`; its rows marked `patient` ran with the
  rule off, as everything does now, and NumPy's warning lines are removed):

  | panel | give-up off, weighted | give-up on, unweighted |
  | --- | --- | --- |
  | GeoLift `N = 40, K = 10` | 70.3 s, 64 of 176 lifts prune | past 900 s |
  | GeoLift `N = 40, K = 8` | 95.3 s, 93 of 239 | 344.5 s, 0 of 14,367 |
  | noise `N = 40, K = 10` | 61.9 s, 72 of 151 | past 900 s |
  | factor `N = 40, K = 6` | 11.5 s, 0 of 26 | 18.9 s, 0 of 644 |

  Without the weight, the rule alone ran GeoLift with ten treated past 900 s,
  against 77.6 s with the rule off. It is off by default (`PATIENCE = inf`) and
  kept as an option (`run.py ... giveup`) so the table can be reproduced.

The search as it stands, both lifts weighted by their prune rate, the lift's cost
learned, the give-up rule off (`results/final_ipm.log`, `results/final_dual.log`;
enumeration from `results/enumeration.log` and `results/enumeration_k10.log`).
Lifts are counted with the nodes they pruned; the rest of the prunes are the
trust region's:

| panel | sets | enumeration | interior-point lift | dual |
| --- | ---: | ---: | --- | --- |
| factor `N = 40, K = 6` | 3.8M | 3.0 s | 22.3 s, 10 lifts (0) | 13.9 s, 30 lifts (0) |
| noise `N = 40, K = 6` | 3.8M | 3.1 s | 7.4 s, 3 lifts (1) | 7.7 s, 4 lifts (1) |
| GeoLift `N = 40, K = 8` | 76.9M | 63.4 s | 83.4 s, 27 lifts (8) | 104.6 s, 281 lifts (119) |
| GeoLift `N = 40, K = 10` | 847.7M | 698.1 s | 90.5 s, 46 lifts (19) | 64.5 s, 187 lifts (84) |
| noise `N = 40, K = 10` | 847.7M | 724.6 s | 274.6 s, 253 lifts (123) | 72.6 s, 170 lifts (82) |
| factor `N = 40, K = 10` | 847.7M | 710.6 s | 1,122.3 s, 1,099 lifts (471) | 1,274.8 s, 8,881 lifts (3,959) |

Every run returned the enumeration's optimum. The weight brought the factor panel
at six treated from 33.2 s to 22.3 s and 13.9 s, still four to seven times
enumeration's 3.0 s: the prune rate is estimated per subtree size, with a prior of
one half, so each size pays a lift or two before the weight learns that lifts do
not prune there. Pooling the estimate across sizes is the next lever.

## Wrong turns

- The first validity test asserted the spectral bound sits above the
  trust-region bound. It does not on small nodes: one takes the linear term over
  the box (`|b|_1`), the other over the sphere (up to `sqrt(n) |b|_2`). Both
  were valid all along; the node takes the smaller.
- The first trust-region code zeroed "the constant direction" by a heuristic
  test on eigenvectors and kept its zero eigenvalue in the spectrum. That
  forced `mu > 0` and loosened the bound wherever the quadratic is negative on
  the plane (-12.18 against an exact -14.77 with two free units); an explicit
  basis of the plane removes the direction.
- The first search lost to enumeration: 25.2 s against 12.7 s at `N = 40,
  K = 6`. A bound cost 0.42 ms, most of it a 200-step bisection (Newton:
  0.26 ms); it pruned 21% of the nodes it bounded; and pricing a small subtree
  cost about 12 µs a set, because each subtree paid a call's overhead (MAREX's
  RCA, cause 5, again). Buffering subtrees into one call and weighting the
  bound's cost by its prune rate fixed all three.
- The lift was first solved in the outcome's units. `R` scales as those units
  to the power -4, near 1e-11 on GeoLift, under the solver's absolute
  tolerances: CLARABEL reported `optimal` at a value 12% under the true
  maximum, a "bound" 11 to 14% above the subtree's minimum. Scaling fixed the
  value; the certificate from multipliers makes the bound valid whatever the
  solver's accuracy.
- The search's chunked enumeration dropped one set per chunk: the helper that
  took the next `k` items pulled a `k + 1`-th and returned without yielding it.
  The first full enumeration of GeoLift with ten treated priced 847,618,148 of
  847,660,528 sets and still found the search's optimum, so only the count
  showed it. `islice` stops at the count; a test forces the chunked path and
  counts, and a mutant restores the loop.
- The first certificate wrote the objective's constant as `c X_00`. It is the
  same number on a design, so the bound stayed valid, but the solver's
  multipliers are stationary only for the constant form, and the certificate
  came out at 4.40 where the relaxation's value is 1.88.
- The search was first compared with the library's enumeration as it stood
  before #828, at 3.3 to 5.3 µs a set. The table then had the search 2.2 to 3.3
  times faster on three of its four rows at six and eight treated. After #828 the
  library prices at 0.8 to 1.3 µs a set and enumerates 3.8 million sets in
  3.0 s: the search loses at every size enumeration can reach.
- The first first-order dual smoothed `lambda_max` by log-sum-exp with a fixed
  schedule `mu = 1, 0.1, 0.01, 0.001`. The parts are divided by their largest
  entry, so the top eigenvalue is near 0.5, and the first two stages moved
  nothing; at 13 free units it took 537 evaluations to come within 8e-4 of the
  lift. BiqCrunch's regularised dual and schedule replaced it.
- Warm starts from the parent's multipliers, restarted at `alpha = 0.1`, saved
  4% of the evaluations over eight children; at 1e-2, 12%; at 1e-4 they cost
  27% more. They start at 1e-2, and node by node they can still cost more.
- Dual fixing: the parent's multipliers read on each child bound it for one
  eigendecomposition (0.03 to 0.06 s for every free unit of a 40-unit node).
  At the nodes that do not prune they fixed no unit; at nodes that prune anyway,
  5 to 16. MAREX's study found the same of inherited bounds
  (`benchmarks/studies/marex_bnb/results/inherit_probe_geolift_m6.log`).
- The give-up rule first fired without a target: the goal is then minus
  infinity, so every node looked hopeless, and five tests caught it.
- The learned lift cost first averaged rates arithmetically: two pseudo-lifts at
  the calibration's rate, a hundred times too high, outweighed two hundred real
  lifts. It averages logarithms.
- The `restrict` mutant that keeps the parent's cut type survived its first
  run: the hand example's re-indexing left every cut's type unchanged. A
  property now checks that every inherited cut takes the same value on the
  child's lifted matrix as the parent's cut on the parent's.

## Findings

1. Below the candidate limit enumeration wins. After #828 it prices 0.8 µs a set
   at 40 units and 1.2 µs at 60, and enumerated 3.8 to 77 million sets in 3.0 to
   65 s, where the best search took 7.4 to 104 s. A node bound only adds cost
   there, and the library's dispatch (enumerate under the limit) stands.
2. Above the limit the search wins where its bounds prune, and the regularised
   dual is the faster lift. At 848 million sets: GeoLift 64.5 s against
   enumeration's 698.1 s, the noise panel 72.6 s against 724.6 s. On the flat
   factor panel no bound of this family separates the sets near the optimum: the
   search took 1,122.3 s with the interior-point lift and 1,274.8 s with the dual,
   against 710.6 s enumerating.
3. When the lift runs it takes 85 to 97% of the search. The dual cuts a lift's
   cost one to seven times at 30 to 40 free units and stays within 0.4% of the
   interior-point bound; with the triangles among the free units it is tighter
   on steep panels.
4. A lift needs its own prune-rate weight, and the weight needs the lift's real
   cost and a lift allowed to finish. The timed calibration starved the dual
   (GeoLift with ten treated past 1,000 s), the give-up rule starved it again
   (0 of 14,367 lifts pruned at eight treated), and with both fixed the search
   is back at 64.5 s.
5. Dual fixing and warm starts help little: no unit fixed at a node that does not
   prune, 4 to 12% fewer evaluations from a warm start. MAREX's study measured
   the same of inherited bounds.
6. The same tree serves one-way (#820): one-way's optimum is at least two-way's,
   and the two-way lift reached 98.5 to 99.98% of one-way subtree minima on
   GeoLift, factor and noise panels, where the two optima differ by 0.01 to
   0.3%. Leaves are priced by one-way's ridge bound and fits, as its backend
   prices them now.

## What to build

- Below the limit, nothing: enumeration as `main` has it after #828.
- #803, above the limit: this search with the dual as its lift, triangles on,
  weighted by its prune rate with the cost learned and the estimate pooled
  across sizes, the give-up rule off. The dual's final multipliers are certified
  with `lambda_max` bounded under controlled rounding, as MAREX's
  `lambda_min_lower` does; here the bound is floating point. A node limit returns
  the incumbent with the smallest open bound, which is never below the Rayleigh
  bound the fallback reports now. On flat panels above the limit the search costs
  about what enumeration would, and the node limit is what reports it.
- #820 on the same tree, with one-way's leaf pricing.
- MAREX: weighting its lift by the prune rate (#832), and a study of the
  regularised dual for its persp+N bound (#833).

## Sources

- C. Buchheim, M. Montenegro and A. Wiegele, SDP-based branch-and-bound for
  non-convex quadratic integer optimization, J. Global Optim. 73 (2019),
  485-514.
- S. Burer and D. Vandenbussche, A finite branch-and-bound algorithm for
  nonconvex quadratic programming via semidefinite relaxations, Math. Program.
  113 (2008), 259-282.
- F. de Meijer and R. Sotirov, On integrality in semidefinite programming for
  discrete optimization, arXiv:2306.09865v2 (2023).
- N. Krislock, J. Malick and F. Roupin, BiqCrunch 2.0, source
  (`bounding_procedure.c`, `bound.c`, `inequalities.c`, `biq_crunch.param`),
  read for the bounding procedure and its default parameters; GPL, so nothing
  here is copied from it.
- G. Liuzzi, M. Locatelli and V. Piccialli, A new branch-and-bound algorithm for
  standard quadratic programming problems, Optim. Methods Softw. 34 (2019),
  79-97.
- H. Luo, S. Chen and H. Wu, A new branch-and-cut algorithm for non-convex
  quadratic programming via alternative direction method and semidefinite
  relaxation, Numer. Algorithms 88 (2021), 993-1024.
- F. Matter and M. E. Pfetsch, Presolving for mixed-integer semidefinite
  optimization (2022).
- J. J. Moré and D. C. Sorensen, Computing a trust region step, SIAM J. Sci.
  Stat. Comput. 4 (1983), 553-572.
- F. Rendl, G. Rinaldi and A. Wiegele, A branch and bound algorithm for Max-Cut
  based on combining semidefinite and polyhedral relaxations (2006).

## Running it

From the repository root:

```
python -m pytest benchmarks/studies/syndes_global_bnb/ -q
python -m benchmarks.studies.syndes_global_bnb.quality ceiling factor noise geolift
python -m benchmarks.studies.syndes_global_bnb.ratios geolift 24 4
python -m benchmarks.studies.syndes_global_bnb.duals ratios geolift 24 4
python -m benchmarks.studies.syndes_global_bnb.duals nodes geolift 40 10 237545.0339
python -m benchmarks.studies.syndes_global_bnb.stages geolift 40 10
python -m benchmarks.studies.syndes_global_bnb.run geolift 40 10 dual
python -m benchmarks.studies.syndes_global_bnb.run geolift 40 10 library
```

`results/` holds the logs, each from the code that wrote it:

- `ceiling_*.log`, `quality_factor.log`: `quality.py`;
- `ratios.log`: `ratios.py`;
- `search.log`, `search_dual.log`, `stages.log`, `duals.log`, `enumeration.log`:
  the search before the lift was weighted (commit `210630f8`);
- `search_weighted.log`: the lift first weighted, its cost still timed and the
  give-up rule on, beside other jobs (the 1,095.9 s GeoLift run);
- `search_isolate.log`: the give-up rule's isolation, written while the rule was
  on by default;
- `final_ipm.log`, `final_dual.log`, `enumeration_k10.log`: the search as it
  stands.

GeoLift is `basedata/geolift_market_data.csv`, read from the repository root.
Two or three of these runs sometimes shared the machine; the logs of the final
search ran three at a time.

One machine with four cores visible: Python 3.13.16, NumPy 2.5.3, cvxpy 1.9.3,
Clarabel 0.11.1, SciPy 1.18.1, BLAS pinned to one thread.

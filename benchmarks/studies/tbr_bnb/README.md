# TBR design: a certified bound for the shared-slope control group

Time-based regression (TBR) designs a geo experiment by choosing a treatment
group and a control group of geos. Its reference search, Au (2018),
Algorithm 1, is a greedy hill climb; #964 added a batched local search after
it. Neither says how far its design is from the best one. This study, #965,
asks whether a branch and bound can prove a design optimal, starting with the
control group for a fixed treatment group.

This is a study, not a benchmark case. Nothing here is pinned, and the timings
are machine-dependent: they were taken four processes at a time on four cores,
with BLAS on one thread.

## The problem

With the treatment group fixed, its aggregate `y` and the candidate controls'
series `X` are centred over the scoring window, and the best control group
solves

    min over C, b of || y - b X u_C ||^2,

every selected control carrying the same slope `b`. That is the regression of
the treatment aggregate on the control aggregate, Kerman, Wang and Vaver
(2017), eq. (1), with the intercept absorbed by the centring. Au's required
impact is proportional to `sqrt(RSS / (n - 2))`. With `T` fixed, `S_yy` is
fixed and `RSS = S_yy (1 - corr^2)`, so the RSS-optimal control group also has
the highest correlation and the highest rounded correlation, and it is the
reference key's optimum for that treatment group whenever it passes the four
gates. A design whose correlation is not positive fails the correlation gate,
so only positive slopes matter.

With `w = b u` and `b` in a slope piece `[lo, hi]`, size `m = |C|`, every
feasible `w` lies in `{0 <= w <= hi, m lo <= sum w <= m hi}`.

## What bounds it

Root gap below the enumerated optimum on a ten-geo GeoLift subset (seed 7),
slope pieces of ratio 1.25:

| treatment group | hull QP | Shor | + perspective | + products | joint lift |
| --- | --- | --- | --- | --- | --- |
| phoenix | 10-33% (all four) | 3.76% | 1.09% | 1.09% | 0 |
| baltimore, columbus | | 4.24% | 3.42% | 2.60% | 0 |
| baltimore, minneapolis, phoenix | | 1.15% | 1.15% | 1.15% | 0 |
| baltimore, columbus, honolulu, miami | | 2.28% | 0.70% | 0.70% | 0 |

- The hull is 10-33% below, and does not tighten as the pieces narrow: at
  `m = 1` it is `b` times the simplex, a synthetic control, which always fits
  at least as well as an equal-weight group.
- The Shor lift (Shor 1987) with the bound-factor products (Sherali and Adams
  1999) gives 1.2-4.2%; the perspective (De Rosa and Khajavirad 2026) and the
  products `(hi - w_i) w_j`, `(hi - w_i)(hi - w_j)` close part of the rest.
- The joint lift of `(u, w)` with every product of the bound factors `u`,
  `1 - u`, `w - lo u`, `hi u - w`, and the shared-slope identity `V = V'`
  (`u_i w_j = b u_i u_j = w_i u_j`, derived here) reaches the enumerated
  optimum on every group and every piece tested, to the solver's accuracy.

What the tightness rests on:

- Without `V = V'` the gap reopens to 12.0-12.3%. Removing any one family of
  products changes nothing on ten geos.
- The slope-free lift (no pieces, no `lo`/`hi`) is exact on ten geos but leaves
  2.8% at `m = 4` on 40 geos (Dallas): the ten-geo ablation had no power to show
  the products are needed, since the pool there is 6 to 9 geos.
- On 40 geos the joint lift closes the root gap for three of the four reference
  groups; for chicago, cincinnati, dallas and denver it leaves 0.79%, which
  branching on geos closes (below).
- The pairwise cones of Han, Gómez and Atamtürk (2023), eq. (5), written in
  this lift's variables, `[[U_ij, V_ij], [V_ij, W_ij]] >= 0` and
  `[[u_i - U_ij, w_i - V_ij], [., W_ii - W_ij]] >= 0`, valid because the slope
  is shared, close 12% of that 0.79% (to 0.70%) and cannot replace the
  products (1.47% with the cones and no products).

No published hull covers this set. Atamtürk and Gómez (2025), Thm 2, and Wei,
Gómez and Küçükyavuz (2022), Thm 1, take the continuous variables as free once
the support is chosen; the proof of the former minimises over `beta` with
`beta(T)` fixed (its eq. (11)), which a shared coefficient breaks. Their hulls
are valid here, not tight. With `sum u = m` exactly, Wei et al.'s strengthening
is empty (the remark after their Thm 1).

## The certificate

Every prune compares an exact lower bound with the incumbent's exact RSS:

- Slope cap. A design with RSS below `y'y` has `b <= ||y|| / sqrt(lambda_min(G))`
  (Cauchy-Schwarz and the Rayleigh quotient, derived here), with
  `lambda_min(G)` bounded below exactly by the library's `lambda_min_lower`. The
  pieces cover `(0, cap]`; the cap is 53 to 85 times the incumbent's slope.
- Hull QP. `f(wbar) + min_P grad f(wbar)'(w - wbar)` is a bound at any point
  (Boyd and Vandenberghe 2004, eq. (3.2)); the linear minimum is a continuous
  knapsack (Dantzig 1957), solved exactly.
- Shor and joint lift. Any multipliers, inequality ones clipped at zero, give
  an affine Lagrangian `<S, Z> + k`, and `obj >= k + min(0, lambda_min(S)) tr_max`
  (Jansson, Chaykin and Keil 2007, Thm 3.2, with a trace bound in place of the
  eigenvalue bound), `S` and `k` summed in rationals and `lambda_min` bounded
  exactly. The joint lift's trace bound `1 + m + m hi^2` follows from the
  products' diagonal (derived here).
- The incumbent's own node. Its minimum is the incumbent, so no lower bound
  exceeds it, and the first certified run left that node open in three groups
  at certified gaps of 1.5e-5 to 2.3e-4 after branching eight levels deep. The
  no-good cut `sum_C u - sum_notC u <= |C| - 1` (Balas and Jeroslow 1972)
  excludes the incumbent and no other design; with it every node closes.

Checked against enumeration on small panels (the tests): the certificates
never exceed the piece optimum, with the solver's multipliers, corrupted ones,
fixings and the cut; with the cut they never exceed the second-best design,
and they exceed the best in 10 of 10 cases. The solver's own value is often
above the optimum (by up to 5e-6 relative); the certificate is a median 2e-8
below it.

## Result: four groups proven optimal on 40 geos

| treatment group | pairs | after hull | after Shor | joint-lift nodes | Clarabel pipeline | L-BFGS-B pipeline |
| --- | --- | --- | --- | --- | --- | --- |
| dallas | 1989 | 13 | 1 | 1 / 3 | 76 s | 85 s |
| dallas, denver | 1900 | 18 | 3 | 3 / 5 | 134 s | 95 s |
| cincinnati, dallas, denver | 1924 | 36 | 11 | 11 / 15 | 440 s | 191 s |
| chicago, cincinnati, dallas, denver | 1872 | 41 | 15 | 21 / 35 | 714 s | 365 s |

For each group every control group with a positive shared slope has RSS at
least the incumbent's. The joint-lift stage alone fell from 40, 100, 393 and
666 s to 26, 39, 113 and 285 s. Dallas is slower in total only because its
hull stage took 25 s against 14 s, the cores being shared; step 5 of the
roadmap below removes that stage's solver cost.

## The node solver

On the hardest group's 15 root nodes, target the incumbent, with the no-good
cut, cold starts:

| | Clarabel | smoothed L-BFGS-B | spectral bundle |
| --- | --- | --- | --- |
| nodes pruned (of 13 Clarabel prunes) | 13 | 11 | 0 |
| time per node | 25-38 s | 4.3-13 s | 37-61 s, at its limit |
| all 15 nodes | 480 s | 117 s | |

- Smoothed L-BFGS-B is the bounding procedure of BiqCrunch (Krislock, Malick and
  Roupin), as `syndes_helpers/branch.py` uses it, on the certified dual: its
  objective is exactly the certificate (Liao, Ding and Zheng 2026, Prop. 3.3).
  The exact certificate of its final multipliers takes 0.4-0.5 s.
- The spectral bundle (Helmberg and Kiwiel 2002, Algorithm 1, with the
  trace-bounded model set of Liao, Ding and Zheng, eq. (44)) is a negative
  result at this size: each iteration solves a conic master QP over about
  26,000 residuals, and at 79 x 79 a full eigendecomposition costs under a
  millisecond, which removes the method's reason to exist. A first prototype
  stalled because its model value was computed as `L(W*, y+)`, which equals the
  model's minimum only when the inner Gauss-Seidel loop has converged; the
  model's minimum has a closed form, `k + T min(0, lambda_min(P'SP), <S, Wbar>)`.
- Reusing one solve's multipliers on other slope pieces prunes about 2% of
  them: once `lo` and `hi` move, `lambda_min(S)` turns strongly negative.
- No product family is dead: all 16 carry multipliers of at least 1e-3 at some
  node, so trimming the lift has to be entry by entry (Fischer, Gruber, Rendl
  and Sotirov 2006, Sect. 3.3), not by family.

### The schedule and the stall rule

The smoothing stages ran 400 evaluations each, and the bound only jumped
between them: the first 600-800 evaluations of every node bought nothing.
Starting at a small smoothing parameter is worse (ill-conditioned); a growing
stage budget, 60 x 2^k up to 800, halves an easy prune's cost (2.3 s against
5.6 s). With it, at 900 evaluations every node that never pruned sat below
-0.9% of the incumbent and every node that pruned later above -1.6%.

Handing a stalled node to Clarabel costs 30-38 s per node and, for the nodes
that need branching, buys only a sharper bound before branching anyway. The
node branches instead, on its most fractional geo read from the eigenvector
behind the bound, and the children warm-start from the parent's multipliers.
On the 15 root nodes: 33 nodes, 24 certified prunes, 9 branches, no open node,
157 s against Clarabel's 666 s.

A first version stopped exactly at the incumbent, leaving margins near 1e-8,
which the exact certificate can lose; a child warm-started from such a parent
stopped at the same margin again, and one node grew into 511 open leaves. The
solver now aims 1e-6 past the incumbent and, if a certificate still fails,
continues toward a higher target before branching.

### The incumbent during the search

`certify(y, X, restarts=r)` on each group, the eight runs sharing four cores
(`results/incumbent.log`):

| treatment group | restarts | pairs after hull, Shor | nodes | adopted | time |
| --- | --- | --- | --- | --- | --- |
| dallas | 50 | 13, 1 | 3 | 0 | 60 s |
| dallas | 1 | 65, 43 | 53 | 1 | 436 s |
| dallas, denver | 50 | 18, 3 | 5 | 0 | 77 s |
| dallas, denver | 1 | 66, 50 | 52 | 1 | 302 s |
| cincinnati, dallas, denver | 50 | 36, 11 | 15 | 0 | 196 s |
| cincinnati, dallas, denver | 1 | 89, 64 | 80 | 4 | 598 s |
| chicago, cincinnati, dallas, denver | 50 | 41, 15 | 35 | 0 | 406 s |
| chicago, cincinnati, dallas, denver | 1 | 79, 57 | 77 | 4 | 576 s |

Every run is certified at the same design. From one restart the search finds
the optimum itself, adopting one to four designs on the way, at 1.4 to 7.3 times
the cost: a poor start costs time, not the proof. Fifty restarts already reach
the optimum, so with them nothing is adopted, and the times match the table
above within the noise of shared cores. The update is a guarantee for a start
the local search gets wrong, not a saving when it gets it right.

## Tests

`pytest benchmarks/studies/tbr_bnb/` (80 tests, about 40 s) checks every bound
against enumeration on panels of up to eight geos, with hypothesis properties
over panels, slope pieces, sizes, fixings and scrambled multipliers, and
`certify()` against the enumerated positive-slope optimum. Coverage is 100%
but for two lines marked with their reasons. The 27 mutants of the eleven
`tbr-bnb-*` targets in `tools/mutation/targets.toml` are killed. The first run
left four certificate mutants alive, each for an assertion too weak to see it:
the two trace bounds without the size (with the solver's multipliers
`lambda_min(S)` is near zero, so the trace barely moves the bound), a negative
product multiplier kept, and the fixing's constant with the wrong sign. The
trace bounds are now checked against every design's lifted matrix, with
equality at `b = hi`; clipping is asserted as an invariant; and a binding
fixing is certified to the solver's value.

## Roadmap

1. Node solver: smoothed L-BFGS-B with stall-and-branch (done here); Clarabel
   for nodes left at the depth limit.
2. Warm starts parent to child (done here). Piece to neighbouring piece, of the
   same size, was measured and dropped: it saves 6% of the solver's evaluations
   and no time.
3. Certificate cost: a floating-point check with a rigorous rounding bound if
   the rational one dominates.
4. Trim the lift entry by entry (a working set).
5. The hull stage on the library's active-set solvers. (a), done here: the band
   without the per-geo cap (`hull.Band`), one `nnls` per group and
   `solve_simplex_qp(s X, y)` at the band's nearer end, its point certified by
   the capped hull's first-order bound (valid for the piece at any point, and at
   least the band's value there), then the capped hull only on what
   the band cannot rule out (`search.hull_survivors`). On the four groups the
   band rules out 94-95% of the pairs with 1,531-1,653 simplex QPs, the capped
   hull runs on 108-127, and the stage takes 6.7-7.9 s against 19.7-24.9 s for
   the capped hull alone, with the same survivors (measured with the cores
   shared; the band's exact certificates are most of what remains). (b): upper
   bounds in `solve_simplex_qp`, on its own branch, after which the capped tier
   is one exact call per pair.
6. Branching rules and node order.
7. Incumbent from #964's local search, improved during the search (done here).
   Each node rounds its eigen-primal to a design of its size within its
   fixings, descends from it (`search.Incumbent.offer`, on the positive-slope
   RSS of `hull.positive_rss`), and a better design becomes the incumbent with
   its own no-good cut beside the earlier ones (`oracle.Oracle(cuts=...)`,
   `certificate.joint_bound(nogood=[...])`). See the table below.
8. The gates, handled lazily by no-good cuts.
9. The search over treatment groups.
10. Tightness on other panels and at more geos.
11. This study.
12. The build: `search="exact"` with `certify`.
13. Shared helpers and `agents/agents_combinatorial.md`.

## Files

- `panel.py`: the scoring window and the four groups.
- `hull.py`, `lifts.py`: the hull QP, the band without the per-geo cap, and the lift levels.
- `certificate.py`: every exact bound, the slope cap, fixings and the cut.
- `oracle.py`, `dual.py`: the certified dual and its maximisers.
- `search.py`: the branch and bound on geos and `certify()`.
- `brute.py`: enumeration, the truth the tests check against.
- `run.py`: `python -m benchmarks.studies.tbr_bnb.run certify` and `... lifts`.
- `results/`: the logs of the runs above.

## Sources

- Au, T. C. (2018). A time-based regression matched markets approach for
  designing geo experiments. Google technical report.
- Kerman, J., Wang, P. and Vaver, J. (2017). Estimating ad effectiveness using
  geo experiments in a time-based regression framework. Google technical report.
- Shor, N. Z. (1987). Quadratic optimization problems. Soviet J. Comput. Systems
  Sci. 25, 1-11.
- Sherali, H. D. and Adams, W. P. (1999). A Reformulation-Linearization
  Technique for Solving Discrete and Continuous Nonconvex Problems. Kluwer.
- Balas, E. (1985). Disjunctive programming and a hierarchy of relaxations for
  discrete optimization problems. SIAM J. Algebr. Discrete Methods 6, 466-486.
- Balas, E. and Jeroslow, R. (1972). Canonical cuts on the unit hypercube. SIAM
  J. Appl. Math. 23, 61-69.
- De Rosa, A. and Khajavirad, A. (2026). Explicit convex hull description of
  bivariate quadratic sets with indicator variables. Math. Program. 218, 169-211.
- Atamtürk, A. and Gómez, A. (2025). Rank-one convexification for sparse
  regression. J. Mach. Learn. Res.
- Wei, L., Gómez, A. and Küçükyavuz, S. (2022). Ideal formulations for
  constrained convex optimization problems with indicator variables. Math.
  Program. 192, 57-88.
- Han, S., Gómez, A. and Atamtürk, A. (2023). 2x2-convexifications for convex
  quadratic optimization with indicator variables. Math. Program. 202, 95-134.
- Jansson, C., Chaykin, D. and Keil, C. (2007). Rigorous error bounds for the
  optimal value in semidefinite programming. SIAM J. Numer. Anal. 46, 180-200.
- Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge.
- Lawson, C. L. and Hanson, R. J. (1974). Solving Least Squares Problems.
  Prentice-Hall.
- Dantzig, G. B. (1957). Discrete-variable extremum problems. Oper. Res. 5,
  266-288.
- Horn, R. A. and Johnson, C. R. (2013). Matrix Analysis, 2nd ed. Cambridge.
- Helmberg, C. and Kiwiel, K. C. (2002). A spectral bundle method with bounds.
  Math. Program. 93, 173-194.
- Liao, F.-Y., Ding, L. and Zheng, Y. (2026). An overview and comparison of
  spectral bundle methods for primal and dual semidefinite programs.
  arXiv:2307.07651v2.
- Nesterov, Y. (2007). Smoothing technique and its applications in semidefinite
  optimization. Math. Program. 110, 245-259.
- Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory
  algorithm for bound constrained optimization. SIAM J. Sci. Comput. 16,
  1190-1208.
- Fischer, I., Gruber, G., Rendl, F. and Sotirov, R. (2006). Computational
  experience with a bundle approach for semidefinite cutting plane relaxations
  of Max-Cut and Equipartition. Math. Program. 105, 451-469.
- Dong, H. (2016). Relaxing nonconvex quadratic functions by multiple adaptive
  diagonal perturbations. SIAM J. Optim. 26, 1962-1985. Read and not used: it
  keeps only the diagonal products, which loses `V = V'`.
- Chen, Y., Lan, G., Ouyang, Y. and Zhang, W. (2019). Fast bundle-level methods
  for unconstrained and ball-constrained convex optimization. Comput. Optim.
  Appl. 73, 159-199. Read and not used: it covers unconstrained and
  ball-constrained problems, not sign-constrained multipliers.

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

- Slope cap. A design with RSS below `y'y` has `b <= ||y|| / sqrt(L)` with
  `L <= x'x = u'Gu` for every nonzero binary `u` (Cauchy-Schwarz, derived here).
  When no entry of `G` is negative, `L = min_j G_jj`, since `u'Gu` sums the entries
  of a principal submatrix and so is at least any one diagonal entry in it; else `L`
  is `lambda_min(G)` bounded below exactly by the library's `lambda_min_lower` (the
  Rayleigh quotient). The diagonal is never below `lambda_min(G)`. The pieces cover
  `(0, cap]`; on the four GeoLift groups of `results/certify_lbfgs.log` the cap is 7
  to 11 times the incumbent's slope, 53 to 85 times with `lambda_min(G)`.
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

### Where the time goes, and two settings

A profile of `certify` on the hardest group (406 s) put 302 s in the node
solver, of which the dual's own evaluations were 57 s: the rest was L-BFGS-B's
bookkeeping over the joint lift's 26,015 multipliers (24,336 sign-constrained).
The exact joint-lift certificates were 45 s and the Shor stage 50 s. On Dallas
(55 s) the node solver was 24 s, 17.7 s of it in L-BFGS-B's core routine, and
the Shor stage 16 s.

- The node solver now stores 10 correction pairs (`dual.LBFGS_MEMORY`), not 20.
  On the four groups that gives the same nodes, prunes and branches as 20 at
  70-87% of the time: 40, 58, 145 and 272 s against 50, 67, 193 and 391 s. With 5
  the hardest group is not certified (47 nodes, one left open): its multipliers
  converge too slowly within the evaluation budget.
- The Shor stage solves at Clarabel tolerance 1e-5 (`search.SHOR_TOL`), not the
  default 1e-8: the same survivors at about 89% of the time. At 1e-4 the hardest
  group keeps 18 pairs against 15.
- A working set (step 4) would trim less than it promises: at a node's final
  multipliers 15% of the sign-constrained ones are zero and 46% are below 1e-3 of
  the largest, so it could at most halve L-BFGS-B's bookkeeping.
- Moving the Shor stage onto the smoothed L-BFGS dual is a negative result. An
  oracle in `shor_bound`'s layout, checked against Clarabel's multipliers (equal
  to 1e-8), keeps 5 more pairs than Clarabel on Dallas and on the hardest group:
  those pairs sit just above the incumbent, and the first-order dual does not
  close that margin within its budget, with or without the stall rule. Run first
  and handed to Clarabel where it falls short, it keeps Clarabel's survivors and
  saves 3.5 s and 1.5 s, which does not pay for a second Shor solver.
- Andrei (2017), Sect. 6.4-6.5, compares L-BFGS-B with the spectral projected
  gradient method (Birgin, Martinez and Raydan): about even in CPU time over 800
  test problems (Tables 6.2-6.3), but far behind on ill-conditioned
  40,000-variable problems (Tables 6.4-6.5, 83 s against 1.9 s), which is the
  regime of the smoothed dual at small smoothing. It is not tried here.

All of it is in `results/solver_settings.log`.

## Without the slope

At the best slope `b = c'u / u'Gu` (with `c = X'y`, `G = X'X` and `u` the
indicator of the control group) the residual sum of squares is

    RSS(C) = y'y - (c'u)^2 / u'Gu      when c'u > 0,

so the best control group maximises the ratio `(c'u)^2 / u'Gu` over binary `u`
with a positive slope (derived here from the normal equation of `b`). That is
a fractional 0-1 program, and its parametric form (Dinkelbach 1967; Borrero,
Gillen and Prokopyev 2017, Sect. 4.2) turns certification into one question
per size `m`: at the incumbent's ratio `lam*`, does any design of size `m` have
`u'(cc' - lam* G)u > 0`? `slopefree.py` answers it with a semidefinite
relaxation in `u` alone, a 37 by 37 matrix on the hardest group against the
joint lift's 73 by 73. Its constraints are `diag U = u`, the products of
`0 <= u <= 1` (Sherali and Adams 1990), the size and the slope sign `c'u >= 0`
with their products, a no-good cut on the incumbent at every size, and triangle
inequalities (Padberg 1989) added where the relaxation's solution violates
them. The certificate is the Lagrangian dual at Clarabel's multipliers (the
weak-duality argument of the S-procedure; Boyd, El Ghaoui, Feron and Balakrishnan
1994, Sect. 2.6.3), summed
in rationals, with the trace of the lifted matrix exactly `1 + m`. The slope
pieces, the Shor stage and the joint lift's node solver are not used; the hull
stage runs first and rules out every size none of whose pieces survive it.

On the four GeoLift groups every size the hull stage leaves is certified at
the root, with no branching (`results/slope_free.log`):

| treatment group | sizes after the hull | `certify` (joint lift, `results/joint_float.log`) | `certify_slope_free` |
| --- | --- | --- | --- |
| dallas | 6 | 38 s | 18 s |
| dallas, denver | 7 | 48 s | 21 s |
| cincinnati, dallas, denver | 14 | 135 s | 35 s |
| chicago, cincinnati, dallas, denver | 17 | 273 s | 54 s |

Each certifies the same design at the same RSS as `certify`. On the hardest
group sizes 8, 9 and 10 sit above zero at the root, with bounds 0.98%, 1.33%
and under 0.01% below the incumbent's RSS, and one round of about 400 triangle
inequalities closes each. Branching (`_branch`, on the most fractional geo) is
implemented and tested on small panels against a poor incumbent; the GeoLift
groups never reach it.

The certificate first read a different matrix than the one it built.
`lambda_min_lower` takes its matrix through mlsynth's `_from_fractions`, which
reads every entry as a dyadic rational, and a ratio such as `9/10 lam*` is not
one. The bound came out 11% loose on a test panel, and it could as well have
come out too low. `lam` now enters as the largest double below it
(`dyadic_below`), which proves the stronger statement that no design exceeds
it, and `bound` refuses a non-dyadic matrix.

Two other routes were read and ruled out before building. The accelerated
Dinkelbach method (Chen, Zheng and Xia 2025) speeds the outer root-finding,
which is one step here because the incumbent's ratio is already the root to
test, and its rates assume a twice-differentiable parametric function, which a
finite design set does not have. A bound from the zonotope of a low-rank part
of `G` (Allemand, Fukuda, Liebling and Steiner 2001; Ferrez, Fukuda and
Liebling 2005) needs the objective to be convex, and neither the ratio nor its
parametric form is: the ratio's superlevel sets `{c'u >= sqrt(lam u'Gu)}` are
convex (derived here), so its maximum over the hull need not be at a vertex.
Their reported times (rank 4, 35 generators: 13,160 cells in 171 s) would have
ruled it out at the ranks the GeoLift Gram matrix needs anyway.

The reference key's four gates are not in this formulation, as they are not in
the joint lift's; the next section adds them.

## The gates

The reference implementation ranks designs by the key `(corr_test, aa_test,
bb_test, dw_test, round(corr, 2), 1 / required_impact)`, read lexicographically
(`mlsynth.utils.tbr_helpers.design.objective.score_split`). For a fixed
treatment group `RSS = y'y (1 - corr^2)` on the centred window, so among the
designs that pass all four gates the one of least RSS also has the greatest
correlation, hence the greatest rounded correlation, and the least required
impact among designs that share it, because the required impact rises with the
residual scale (derived here from `required_impact`'s formula). The key's
optimum is therefore the least-RSS passing design.

The gates are tests on the fitted residuals, not constraints on `u` a
relaxation can carry, so `gated.py` handles them lazily. `certify_gated`
certifies the RSS-best design not yet excluded (`certify_slope_free(...,
exclude=...)`), scores it with the library's own gates, and when a gate fails
excludes the design with a no-good cut and certifies again. Each round's design
is the next by RSS, so the first that passes is the passing optimum, and every
earlier round is a certified failure. The cut
`sum_C u - sum_notC u <= |C| - 1` admits every design but `C` at every size
(at size `m` with `t` geos in common it reads `2t - m <= |C| - 1`, which fails
only when the design is `C`; derived here), so one cut per excluded design is
added to every size's relaxation. The correlation gate ends the search when it
fails: the RSS-best remaining design has the greatest correlation of all
remaining ones, so none of them reaches 0.8 either, and the answer is a
certificate that no design passes. A search that excludes every positive-slope
design ends the same way.

On the four GeoLift groups with the benchmark's A/A window (`n_test = 14`) the
RSS-best design passes all four gates in the first round, so the gated optimum
is the slope-free one and the gates cost one scoring call
(`results/gated.log`):

| treatment group | rounds | controls | key's last entry, `1 / required_impact` | seconds |
| --- | --- | --- | --- | --- |
| dallas | 1 | 5, 9, 18, 38 | 0.00128 | 17 |
| dallas, denver | 1 | 3, 5, 17, 37 | 0.00055 | 18 |
| cincinnati, dallas, denver | 1 | 3, 5, 6, 16, 28, 33, 36 | 0.00049 | 34 |
| chicago, cincinnati, dallas, denver | 1 | 15, 18, 19, 27, 28, 32, 35 | 0.00038 | 47 |

The exclusions are exercised on simulated AR(1) panels (`brute.panel_ar`),
where designs fail the gates: the tests check the rounds against enumeration of
the key (`brute.gated_optimum`) on seeds drawn by hypothesis, the exclusion order, the
correlation stop, the no-design-passes certificate and the round limit. Before
the gates the incumbent's cut was added only at the incumbent's size, which is
not enough once excluded designs of other sizes must stay out; the cut now goes
to every size. The first mutation run left one mutant alive, the excluded
designs' cuts dropped, because the branch and bound still found the right
answer by branching around them. With the cuts the excluded design's size
closes in one to three nodes on the test panels, and without them in eight to
ten, so the test pins the node count at three or fewer.

## The treatment groups

Within one treatment group the least-RSS passing design has the key's greatest
value (The gates). Across treatment groups the key compares the rounded
correlation and then `1 / required_impact`, and the required impact is a constant
times `sqrt(RSS / (n - 2))` on the shared window, so the best design with `k`
treatment geos is the best of the groups' own optima, compared on
`round(corr, 2)` and then on RSS (derived here). `treatment.py` searches every
group of the size, every geo eligible for every role:

1. An incumbent from the library's greedy search and local search (#964), with
   its own group's optimum certified (`gated.certify_gated`).
2. An exact floor on the RSS of every positive-slope design of every group
   (`rss_floor`). With `w = b u` every such design is a point of
   `{0 <= w <= cap}`, so the group's nonnegative least squares over that box is
   at most its RSS; the float hull certificate (`certificate.hull_bound_float`,
   with `lo = 0` and the size constraint slack) bounds it from below at the NNLS
   point. The cap is `ExactData.slope_cap`'s, with one eigenvalue floor for every
   group: a group's control Gram matrix is a principal submatrix of the full one,
   whose smallest eigenvalue is at least the full matrix's (the interlacing
   inequalities, Hogben 2007, Sect. 8.2, Fact 5, once per deleted row). The
   floor sits within 1e-8 (relative) of the NNLS value. Each group's `y'y` is
   computed once, exactly, in integers (`square_sum`: every finite double is an
   integer over a power of two); as a sum of Fractions, computed twice, it was 70%
   of the stage. The stage, `y'y` and the floor for each of the 91,390 groups at
   four geos, takes 17 s, 0.18 ms a group, where it took 49 s.
3. A group whose floor rules it out is skipped (`can_compete`): every design's
   `corr^2 = 1 - RSS / y'y`, so the floor bounds the group's correlation, and a
   group that cannot reach the incumbent's rounded class, or can reach it only
   with a larger RSS, cannot have a larger key. The class edges are half a
   rounding step either side, less a slack of 1e-9 for the key's correlation
   being a double; the slack is a tolerance, not derived.
4. Every other group is certified against its ceiling, the RSS below which one of
   its designs could beat the incumbent (`ceiling`). Each size's relaxation is
   tested at `y'y - ceiling` instead of at the group's own optimum
   (`certify_slope_free(..., ceiling=...)`), and the group's gated search ends,
   certified, at the first round whose RSS-best design is not below the ceiling.
   A group whose optimum beats the incumbent's key replaces it.

On GeoLift (`n_test = 14`, `results/treatment.log`):

| treatment geos | groups | ruled out by the floor | certified one by one | seconds | the key's optimum | `round(corr, 2)` | `1 / required_impact` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 39 | 1 | 14 | dallas | 0.99 | 0.001283 |
| 2 | 780 | 770 | 10 | 172 | dallas, milwaukee | 1.00 | 0.000652 |
| 3 | 9,880 | 9,869 | 11 | 410 | austin, dallas, jacksonville | 1.00 | 0.000631 |
| 4 | 91,390 | 91,297 | 93 | 1,117 | austin, dallas, minneapolis, reno | 1.00 | 0.000542 |

The reference implementation's designs (`benchmarks/cases/tbrmm.py`) round to
0.99 at every size, with `1 / required_impact` of 0.000999, 0.000450, 0.000433
and 0.000369. At one geo the reference's treatment group is the optimum's and
the certified control group is 28% better on that entry. From two geos up the
optimum is another treatment group whose correlation rounds to 1.00, which the
key ranks above any design at 0.99.

Three things were found on the way, each from a run that came back uncertified.

- Certifying each surviving group's own optimum left groups open. At two geos
  some groups' RSS-ordered designs kept failing the gates and ran to
  `certify_gated`'s 50-round limit at 140-300 s each, and one group (denver,
  jacksonville) branched to the depth limit at size 9. None of them could reach
  the incumbent's class: the second group's own best correlation is 0.99482. A
  stop rule read before each round's gates certified the first kind; only the
  ceiling, which asks each relaxation the narrower question, certified both. With
  it two geos take 259 s against 1,147 s uncertified, and three take 516 s
  against 1,098 s.
- The node left open was a node with one design. `_branch` evaluated a node
  exactly only once every geo was fixed, and that node had nine of its nine geos
  fixed in and three out, so it was bounded and split on geos with one value left
  until the depth limit. A node with `m` geos fixed in, or `p - m` fixed out, is
  now evaluated as its one design; with that, the group's own optimum certifies
  in 81 s, and its branching finds a design better than the local search's (RSS
  2.754e6 against 2.759e6).
- At four geos two runs certified 92 of the 93 survivors and left one open
  (austin, dallas, jacksonville, reno, 149-170 s): its turn came against an
  earlier incumbent's ceiling. Against the final incumbent's, lower and so a
  weaker claim, it closes in 34 s. The search now certifies every group left open
  again once all groups are read, when the ceiling has fallen since its turn; the
  groups that closed on their turn were proved against ceilings at least as high,
  so their certificates stand. The four-geo row is a single search with one such
  retry.

## The queue

Most of the four-geo search above certifies groups that cannot win: the
winner (austin, dallas, minneapolis, reno) was the 89th of 93 survivors
certified, so most were certified against an earlier incumbent's ceiling, a
higher one. With the final incumbent from the start, the floor alone rules out
61 of the 93, and 382 of the 383 sizes that reach the relaxation close at the
root (measured on the 93, at the final ceiling, without triangles). The bound
is tight enough; the incumbent arrives late.

So before any group is certified, every group the start incumbent leaves gets
a local search of `pass_restarts` random starts (the study's RSS descent,
`search.incumbent`), the best passing design becomes the incumbent, the groups
are certified best local design first (the reference key's order: rounded
correlation, then RSS), and each certification starts from its group's local
design (`certify_slope_free(..., start=)`) instead of searching again. A
certification still proves its group's claim against the ceiling of its turn,
so the order changes the work, never the answer.

Ten restarts are the fewest that give the four-geo optimum's own design: on
the 93 survivors, 10 restarts reach the 50-restart design in 48 groups and the
same best design overall, 5 restarts reach the winning group with a weaker
design (`1 / required_impact` 0.000531 against 0.000542), and one descent from
the rounded support of the group's nonnegative least squares reaches another
group (0.000526): its support is a weighted fit's, and rounded to equal weights
it is a poor start. At four geos (`results/queue.log`):

| | total | certifications | groups certified | incumbent replaced | groups retried |
| --- | --- | --- | --- | --- | --- |
| without the pass | 1,117 s | 1,043 s | 93 | 3 | 1 |
| with the pass, 10 restarts | 371 s | 274 s | 33 | 0 | 0 |

The pass searches 90 groups in about 45 s, and its best design is already the
certified optimum, so the incumbent is never replaced and no group is left
open. A third of that was scoring: each group's local design was scored through
the group's whole exact Gram (`certificate.ExactData`), 0.17 s a group, though
the score needs only the design's `c'u`, `u'Gu` and `y'y`. `design_moments` gives
those exactly from integers over a power of two, as `square_sum` does for `y'y`,
in 0.4 ms. On one host, with the cascade and the dual tier in both runs
(`results/pass.log`), the pass takes 39.7 s, where it took 66.6 s (21.2 s of it
in `ExactData`), and the four-geo search 159.5 s, where it took 192.7 s; that
host runs the whole search about 1.45 times slower than the one behind the
other tables here. On 15 markets with three treated (455 groups) both searches certify the
optimum found by enumeration of all 1.9 million designs (austin, baton rouge,
dallas; 0.99 and 0.000518) in 1.4-1.6 s, the start incumbent there being
already near it.

## Pricing

A node whose subtree holds few designs is priced instead of bounded, as in
MAREX's branch and bound (`search_skeleton.cheaper_to_price`): every design's
ratio `(c'u)^2 / u'Gu` is computed in doubles for the whole subtree at once
(`slopefree.price_subtree`), with an upper bound on its rounding, twice the
dot-product bound (Bjorck 2015, Lemma 1.4.2) on the sum `c'u` and on the Gram sum
`u'Gu` with the underflow allowance of `certificate._dot_err`. A design is
discarded only when that bound is at or below the incumbent's ratio, and the rest
are evaluated exactly, best bound first. The limit is the cost model's
break-even: one relaxation takes 1.5-1.9 s at the root on the GeoLift groups
(4.3 s with a round of triangles), one priced design 0.6-2.0 us, so 750,000
designs, which prices sizes up to 5 of 38 at their roots (`results/pricing.log`):

| treatment group | every node bounded | small subtrees priced | nodes priced |
| --- | --- | --- | --- |
| dallas | 16.7 s | 9.4 s | 5 of 6 |
| dallas, denver | 17.2 s | 10.5 s | 5 of 7 |
| cincinnati, dallas, denver | 34.2 s | 25.9 s | 5 of 14 |
| chicago, cincinnati, dallas, denver | 46.3 s | 36.0 s | 5 of 17 |

Each certifies the same design either way. The saving is about five relaxations
a group, the smallest and largest sizes; the middle sizes, which hold up to
`10^10` designs at the root and most of the time on the larger groups, are still
bounded. Of the 440,000-670,000 designs priced per group, at most one needed
exact arithmetic. The relaxation's own tests bound every node, since on their
small panels pricing closes every size at its root; `test_pricing.py` checks the
priced path against enumeration and that both paths certify the same design.

## The cascade

At four geos on GeoLift the queue certifies 33 groups, and a stage profile of
that run (379 s on a quiet machine, BLAS on one thread) put 224 s in Clarabel:
242 root sizes, one relaxation each, 241 of them closed in round 0 and none
branched. Their bounds were far below zero, which left room for weaker,
cheaper certificates. The Lagrangian bound holds at any multipliers and for any
subset of the rows, so `size_certificate` tries three, cheapest first:

1. The previous root size's multipliers at this size's rows. They are screened
   in doubles (`_screen`, an eigenvalue of the same matrix) and, when the screen
   is at or below zero, evaluated exactly by `bound`. No solve.
2. The Shor relaxation (`Rows(..., level="shor")`): the equalities (diag U = u,
   the size and its products, the fixings), `c'u >= 0` and the cuts, without the
   products of the bounds and of `c'u`. A solver failure here hands the size on.
3. The full relaxation, with its rounds of triangles, as before.

On the 242 recorded sizes, reused multipliers closed 43 of the 239 that had a
previous size, at about 0.02 s each, and the Shor relaxation closed 212 of 242 at
0.37 s a solve, against 1.07 s for the full relaxation. A variant of the Shor
relaxation with the `u_i c'u` products kept could not be solved by Clarabel on
the sizes Shor left open, so the third tier is the full relaxation.

The four-geo search certifies the same design (austin, dallas, minneapolis,
reno) in 230 s, where it took 379 s: the 242 sizes take 100 s, where they took
246 s, with 207 solves, 90 s of them in Clarabel. Each tier's certificate is
the exact bound, so a size is closed only when that bound is at or below zero;
`test_cascade.py` checks each tier against enumeration and that a tier which
does not close, or whose solver fails, hands the size to the next. The full
relaxation hands a failure to the branch and bound: the size is left open, with the
last round's relaxation point when one was solved, and is branched. On a simulated
panel of 100 geos at four treated geos, Clarabel failed on one size's full
relaxation while one group's search branched, and before this the error ended the
whole treatment search (`test_solver_failure.py`).

### The dual tier

None of the three certificates above needs the relaxation's optimum, only
multipliers at which the exact bound is at or below zero. So the cascade has a
tier between the reused multipliers and the Shor relaxation that looks for them
directly (`dual_multipliers`): L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995), with the
inequality multipliers bounded at zero, on a smoothing of the certificate

    F(y) = b'y + (1 + m) max(0, lambda_max(S(y))),

namely `max {<Z, S> - a/2 ||Z||^2 : Z >= 0, tr Z <= 1 + m}`, within `a (1 + m)^2 / 2`
of it, whose maximiser projects the eigenvalues of `S / a` onto
`{z >= 0, sum z <= 1 + m}` (derived here). It starts from the previous size's
multipliers, shrinks `a` tenfold per stage, stops at the first iterate whose
unsmoothed `F` is below zero, and gives up after 400 evaluations. The rows
carrying `c` are divided by a power of two near `max |c|`, as `solve` divides
them for Clarabel: without it the dual stalled above zero on every GeoLift
group tried, since those rows are orders of magnitude above the others.

On the 242 recorded root sizes it closed 240 (209 of 210 warm-started, 31 of 32
cold), at a mean of 0.04 s (median 0.017 s) and at most 309 evaluations, against
0.37 s for a Shor solve; the exact bound at its multipliers costs another 0.04 s.
The two it left open went on to Clarabel. With it, the four-geo search
(`results/cascade.log`, with the y'y of the floor stage in integers):

| run | wall | size certificates | Clarabel solves |
| --- | --- | --- | --- |
| before the cascade | 379 s | 246 s | 243, 224 s |
| reuse, Shor, full | 213 s | 112 s | 207, 103 s |
| reuse, dual, Shor, full | 131 s | 28 s | 7, 6 s |

The same design is certified in each.

## One certificate for every size

The treatment search's floor (`rss_floor`) is a nonnegative least-squares fit:
any nonnegative weighting of the controls, a relaxation of the shared slope on
an equal-weight control group. On the 32 groups that survive it at four geos
the floor sits 15-40% below each group's best design. Two tighter relaxations in
the weights alone, the convex hull of the equal-weight designs of each size and
the same hull with a slope cap for each size (taken optimistically from a greedy
bound on u'Gu), gave the NNLS value on all 32 and ruled out none. The convex hull
of {b u} holds every mixture of one-control designs, the NNLS fit is one, and a
relaxation in w scores the objective at the mixture, below the mixed designs'
own values (derived here). So any relaxation that admits unequal weights is at
most the floor: the rank-one and 2 x 2 convexifications for indicators
(Atamtürk and Gómez 2025; Han, Gómez and Atamtürk 2023) relax a problem whose
optimum, without a cardinality constraint, is the NNLS value.

A tighter bound has to know u_j^2 = u_j. The slope-free search already holds
one, size by size: the ratio test u'(cc' - lam G)u <= 0 (Dinkelbach 1967) over
the semidefinite relaxation with the bound products. Asked only whether any
design of a group is below a ceiling, `certify_slope_free` now first tries that
relaxation once for every size (`group_certificate`): the rows of `Rows(c,
None)`, in which sum u >= 1 and its products with 1 - u_i, (sum u - 1)(1 - u_i)
>= 0, replace the size equality and its products, and a trace bound of 1 + p.
The first-order multipliers are tried first and Clarabel's after, and the exact
certificate decides. When it is at most zero no design of any size is below
the ceiling, and neither the hull stage nor a size is searched. Without the
bound products the same relaxation ruled out none of the 32 groups.

On the 32 recorded survivors the certificate closed 31, 27 of them through the
first-order multipliers at about 0.25 s a group and 4 through Clarabel at about
2 s. The one it leaves open is the group of the certified optimum, which no
certificate can close. End to end on GeoLift at four geos
(`results/group_screen.log`, `main` and this branch on one host):

| `n_jobs` | `main` | with the group screen | the design |
| --- | --- | --- | --- |
| 1 | 168.6 s | 121.8 s | austin, dallas, minneapolis, reno |
| 4 | 72.9 s | 60.4 s | the same, key and RSS equal |

The screen runs inside the worker that certifies the group, in parallel with the
rest of its wave. The gain at four processes, 17% against 28% at one, has not
been decomposed.

### The screen's schedules

On the simulated 100-geo panel at four treated geos (97 candidate controls), the
groups the screen closed cost 95 to 145 s each, nearly all of it one Clarabel
solve: 53 interior-point iterations at about 2.3 s each on the 98 x 98
semidefinite block (profiled on group (4, 21, 35, 58)). Dropping every pair row
left 290 rows and 28 iterations at the same cost each, so the row count is not
the cost. The k-cluster literature bounds problems of this size with first-order
methods instead (Malick and Roupin 2012, 2013; Krislock, Malick and Roupin 2016),
and the screen's first-order tier is one. It gave up on these groups because of
its schedule, not its bound: L-BFGS-B at 150 evaluations a smoothing stage, the
smoothing shrinking tenfold a stage, stops far from the relaxation's optimum. On
(4, 33, 35, 58) its best certificate after 1,143 evaluations was 3,725 times y'y; at
3,000 evaluations a stage it was below zero in the second stage.

On the 19 groups that search certified, each screened at its recorded ceiling:

| schedule | groups closed | seconds a closed group |
| --- | --- | --- |
| 150 evaluations a stage, 400 in all (the size certificates') | 4 | 1.3-1.6 |
| 1,000 a stage, 8 stages, 8,000 in all | 16 | 4.3-9.3 |
| 3,000 a stage, 4 stages, 12,000 in all | 17 | 8.1-25.3 |
| Clarabel, full rows | 17 | 78.4-140.1 |

The 17 are the same groups in the last two rows. The two left open, (4, 35, 58,
60) and (4, 21, 35, 58), are left open by Clarabel too, and go on to the sizes.
`group_certificate` now tries the three schedules in turn and hands what none
closes to Clarabel; the result's `rung` names the schedule that closed it.

The longer schedules run from 48 candidate controls (`SCREEN_LADDER_FROM`). On
simulated 90-period panels, where the second schedule closed a group, it took
0.7-1.5 s against Clarabel's 0.8-1.8 s at 36 controls, the GeoLift groups' size;
1.5-2.6 s against 4.6-8.7 s at 56; and 2.8-8.1 s against 23-47 s at 76 (24
groups, four at each size and ratio). End to end on simulated 90-period panels
(`results/screen_ladder.log`, `main` and this branch on one host, three
processes):

| panel | treated geos | `main` | with the schedules | the design |
| --- | --- | --- | --- | --- |
| 90 geos | 3 | 477.3 s (561.1 s certifying) | 369.3 s (325.6 s) | the same, key equal |
| 80 geos | 4 | 2,587.0 s (4,460.3 s) | 1,093.0 s (1,284.9 s) | the same, key equal |

The slowest group of each run kept its time (287 s and 448 s), and the gain is
in the rest. A schedule that
does not close costs its whole budget: on five 30-geo groups (26 controls, 20
periods) the two longer schedules spent 1.9-4.6 s, where the whole screen with
Clarabel took 0.35-0.71 s. Below 48 controls the screen is unchanged.

### The screen's last tier

When no schedule closes, the screen solves the size-free relaxation with Clarabel.
That relaxation is looser than any one size's, since a fractional point can blend
designs of different sizes, and at 96 and 98 candidate controls its solve costs
more than the sizes' own certificates. From `SCREEN_SOLVE_BELOW` (90) candidate
controls the screen ends open after its schedules and the sizes decide
(`screen_solves`; `group_certificate(..., final_solve=...)` overrides it).

On the hard group of the 100-geo, four-treated search, (4, 27, 33, 58) of
`brute.geo_panel(0, n=90, p=100)`, at the ceiling of a 1% gap, the screen spent
1.3, 15.2 and 26.5 s on its three schedules and 129.3 s on the solve, and closed
nothing; every size then closed at its root. Run alone at the ceilings of 3%, 10%
and 30% gaps the screen closed at its third or second schedule in 43.2, 5.6 and
5.3 s, so it proves an RSS floor of 1,273.9 and not one of 1,324.8, against the
group's optimum of 1,336.16 (`results/screen_solve.log`).

With and without the solve, four processes, each run alone
(`results/screen_solve.log`):

| case | candidate controls | with the solve | without | what the solve did |
| --- | --- | --- | --- | --- |
| GeoLift, 4 treated, exact | 36 | 21.8 s | 24.0 s | closed 7 screens, 11.8 s |
| GeoLift, 4 treated, gap 1% | 36 | 14.0 s | 18.6 s | closed 6 screens, 9.6 s |
| GeoLift, 5 treated, exact | 35 | 45.3 s | 50.1 s | closed 10 screens, 14.9 s |
| GeoLift, 5 treated, gap 1% | 35 | 39.4 s | 43.8 s | closed 5 screens, 8.1 s |
| `geo_panel(0)`, 60 geos, 4 treated, exact | 56 | 219.9 s | 219.4 s | never reached |
| `geo_panel(0)`, 60 geos, 4 treated, gap 1% | 56 | 60.4 s | 65.6 s | closed 1 screen, 25.1 s |
| `geo_panel(0)`, 100 geos, 2 treated, gap 1% | 98 | 327.5 s | 208.8 s | closed 1 screen, 204.4 s |
| the hard 100-geo group, gap 1% | 96 | 299.5 s | 180.3 s | closed nothing, 120.9 s |

Every pair certified the same design. The hard group's runs solved the sizes on the
SCS workspace of the step 2 spike, which is not in this repository. Below 90
controls the solve pays for itself and stays; at 98 it closed its screen in 204.4 s
where the sizes took 45 s; and on the 19 groups of the 100-geo, four-treated search
measured for the schedules (above) it closed none the schedules left open.

### Skipping a rung, measured and not built

Each rung of the screen pays only when it closes what a later rung would close at
more cost. To see where a rung never pays, every screen of four runs was repeated
with each rung run on its own (`results/screen_rungs.log`, gap 1%):

| case | candidate controls | screens | rung 0 closes | rung 1 closes | rung 2 closes | the screen |
| --- | --- | --- | --- | --- | --- | --- |
| GeoLift, 4 treated | 36 | 25 | 19 (5.0 s) | not run | not run | 14.7 s |
| `geo_panel(0)`, 60 geos, 4 treated | 56 | 15 | 9 (6.4 s) | 14 (22.0 s) | 14 (53.9 s) | 38.2 s |
| `geo_panel(0)`, 80 geos, 4 treated | 76 | 135 | 3 (100.3 s) | 132 (292.3 s) | 133 (693.8 s) | 526.9 s |
| the hard 100-geo group | 96 | 1 | 0 (1.2 s) | 0 (14.9 s) | 0 (27.7 s) | 47.2 s |

Rung 2 closed one screen rung 1 left open, at 76 controls; below 90 the
Clarabel solve after the rungs would close it too, at 23-47 s against rung 2's
8 s, so dropping rung 2 there costs more than it saves. Rung 0 never pays at 76
controls: skipping it there saves its 0.74 s on the 132 screens rung 1 closes and
costs rung 1's 2.2 s on the 3 it closed, about 91 CPU-seconds of the search. At 36
and 56 controls it closes most screens, and at 96 (4 of the 19 groups of the
100-geo search, above) it breaks even.

So the closure rate of rung 0 is a property of the panel, not of the number of
controls, and a rule fixed by `p` would misfire. The rule that fits is adaptive:
a rung tried some number of times in a search that closed few of them is skipped
for the rest of it, a skip changing what the screen costs and never what is
certified. Kept deterministic across worker counts, it needs the counts carried
between processes wave by wave, through `treatment.py`, `gated.py` and
`slopefree.py`, for about 6% of the 80-geo search and nothing on the other three.
Not built; the logs are here for the case that changes the balance.

### The sizes' schedules and first-order triangle rounds

What the screen cannot close is a different kind of hard. Profiled alone, the
100-geo group (4, 35, 58, 60) took 2,321 s. It is the incumbent's own group (its
best design is the incumbent, RSS 1,413.44 with 25 controls), so the screen does
not apply and every size is certified against the incumbent's ratio. All 41 sizes
survive the hull, 29 close on the first-order tier at about 1 s each and three
are priced, and nine around the optimum's size, 19 to 27, go to Clarabel, 2,250 s
of the total. None branches. Each Clarabel solve costs 100-145 s, and each round
of triangles one more solve.

The same two levers apply as at the screen, now per size. On those nine sizes,
first-order multipliers alone (no Clarabel):

| size | `main` (Clarabel) | first-order |
| --- | --- | --- |
| 19, 20, 27 | 144-149 s, no triangles | the longer schedules close them |
| 21, 22, 26 | 244-270 s, one round of 400 triangles | one first-order round closes them |
| 23, 24 | 345-354 s, two rounds | two first-order rounds close them |
| 25 | 392 s, two rounds | three first-order rounds leave it open (829, 159, 27, 6 times y'y) |

A first-order round adds the 400 most violated triangles at the dual's smoothed
point (`dual_multipliers` returns it when it gives up) and runs the last schedule
from zero. The triangle rounds inside a first-order method are those of
Krislock, Malick and Roupin (2016); their warm start between rounds does not carry
over here. Started from the last round's multipliers, sizes 23, 24 and 25 were
still open after six rounds (4.6, 53 and 98 times y'y), where two rounds from zero
closed 23 and 24: as at the screen, a start near the boundary begins the smoothing
too fine.

From 90 candidate controls (`SIZE_LADDER_FROM`) `size_certificate` now tries the
screen's three schedules (the previous size's multipliers given to the first only),
then up to `FIRST_ORDER_ROUNDS = 3` rounds, before the Shor and full relaxations;
the result's tier names `triangles` when a round closes it. Profiled again, the
group took 1,272 s:

| sizes | tier | seconds a size |
| --- | --- | --- |
| 19, 20, 27 | dual | 10.3, 62.8, 16.0 |
| 21, 22, 26 | triangles, one round | 92.4-94.5 |
| 23, 24 | triangles, two rounds | 137.7, 149.4 |
| 25 | full, after three first-order rounds | 584.5 (392 on `main`) |

Why 90 and not the screen's 48: on the 80-geo, four-treated search the incumbent's
group, (7, 9, 26, 48), has 76 candidate controls, and there Clarabel's first solve
took 43-48 s, about what a first-order round costs. Profiled on one process:

| the group's sizes 21-27 | `main` | schedules and rounds | schedules alone |
| --- | --- | --- | --- |
| 21, 23, 27 | 43-48 s each, Clarabel | under 8 s each, dual | under 8 s each, dual |
| 24 | 67.6 s, one round | 70.2 s, one first-order round | 118.9 s, Clarabel after the schedules |
| 25 | 131.0 s, two rounds | 272.7 s, Clarabel after three rounds | 162.7 s |
| 26 | 85.9 s, one round | 100.3 s, two first-order rounds | 128.0 s |
| the group | 439.3 s | 479.0 s | 446.8 s |

The schedules close the sizes the dual can close and cost about 50 s on each size
that goes to Clarabel anyway; at 76 controls the two balance. Below 90 controls a
size's cascade is the one on `main`. The threshold rests on two groups, one either
side of it.

The other hard group of that search, (4, 27, 33, 58), is hard in a third way: the
screen does not close it (193 s), and size 21 is open at the root after six
Clarabel rounds of triangles (798 s), so it branches, at 148-249 s a node. A
profile was cut short by a host restart after five nodes; cheaper nodes and the
branching rule are its levers, not measured here.

### The dual's memory

At 96 candidate controls a first-order schedule costs tens of seconds, and the
smoothed certificate it evaluates is the smaller part. Profiled on a size-24 root
(18,627 multipliers), one schedule of 1,000 evaluations a stage took 27.8 s: 18.1 s
(65%) in L-BFGS-B's own work per call (scipy's `setulb`), 7.6 s in the
certificate and its gradient, 4.5 s of that the eigendecomposition. L-BFGS-B's work
per call grows with the number of correction pairs it stores times the number of
multipliers (Byrd, Lu, Nocedal and Zhu 1995), and the dual stored twenty.

It now stores five (`DUAL_MEMORY`). On the three size-24 nodes of the 100-geo group
(4, 27, 33, 58) that the first-order triangle rounds closed in the profile, each
re-run alone at the profile's ratio (`results/dual_memory.log`):

| correction pairs | node seconds | rounds to close |
| --- | --- | --- |
| 20 (before) | 118.0, 118.5, 120.7 | 1, 1, 1 |
| 10 | 101.2, 89.8, 92.7 | 1, 1, 1 |
| 5 | 63.7, 67.4, 66.6 | 1, 1, 1 |
| 3 | 106.3, 74.7, 64.4 | 2, 1, 1 |

At five pairs a node costs about 55% of what it did; at three the multipliers
converge too slowly for one of the three, which takes a second round. On the five
test beds, both memories under the same load:

| bed | nodes | Clarabel solves | dual seconds | group seconds |
| --- | --- | --- | --- | --- |
| `s15_k3` | 974 → 974 | 151 → 151 | 22.1 → 16.9 | 64.0 → 53.8 |
| `s20_k3` | 1,122 → 1,122 | 203 → 203 | 39.4 → 30.7 | 102.7 → 92.0 |
| `s20_k4` | 525 → 525 | 79 → 79 | 13.2 → 10.9 | 35.5 → 32.7 |
| `s30_k10` | 685 → 691 | 139 → 139 | 27.5 → 23.8 | 68.6 → 65.7 |
| `s30_k4` | 1,404 → 1,406 | 255 → 260 | 94.3 → 67.3 | 252.7 → 222.3 |
| all | 4,710 → 4,718 | 827 → 832 | 196.5 → 149.6 | 523.5 → 466.5 |

The dual takes 24% less time and the beds 11% less, for 8 more nodes and 5 more
Clarabel solves; every group certifies the same design. The memory steers only the
search for multipliers: a size is certified by the exact bound at whatever
multipliers that search returns, so no memory changes what is proved. The node
solver of the shared-slope search (`dual.py`) keeps its own memory, 10, measured
on its own problem.

### The Shor root, measured and not built

A size's root holds every design of that size. Once the first-order tiers leave it
open, it solves the Shor relaxation, then the full relaxation with up to six
rounds of triangle inequalities, and branches on the full relaxation's point. At 96
candidate controls that is the most expensive step of a size: on the 100-geo group
(4, 27, 33, 58), size 21, one Shor solve took 39.9 s, one full solve (18,434 rows)
121.5 s, and the root with its triangle rounds 954 s, neither relaxation closing it.

Stopping the root at the Shor relaxation and branching on its point
(`results/shor_root.log`) cut the beds' full solves by 30% (827 to 575) for
6% more nodes (4,710 to 4,970), and every group certified the same design. It did
not cut their time. Under the same load:

| bed | full-solve seconds | group seconds |
| --- | --- | --- |
| `s15_k3` | 2.9 → 1.7 | 44.2 → 42.9 |
| `s20_k3` | 11.9 → 6.6 | 94.7 → 86.8 |
| `s20_k4` | 3.7 → 2.8 | 36.4 → 35.8 |
| `s30_k10` | 13.6 → 7.7 | 70.0 → 62.2 |
| `s30_k4` | 58.5 → 36.7 | 253.3 → 267.3 |
| all | 90.6 → 55.5 | 498.6 → 495.0 |

The extra nodes' first-order work took what the cheaper roots saved. On the
100-geo group, profiled side by side, sizes 21-23 (three to five nodes each) took
1,894 s against 4,506 s, and size 24 reversed it:

| | `main` | Shor root |
| --- | --- | --- |
| root branched on | control 17 | control 77 |
| size 24 | closed, 7 nodes, 2,738 s | open after 24 nodes, 6,059 s |

By 8,350 s into each run the Shor root had spent the 2,612 s it saved and was
still inside size 24.

The first explanation, that Shor's point is near-uniform and so carries no
information about which control to branch on, measured wrong. At size 24's root
Shor's u has 67 of 96 controls fractional and a standard deviation of 0.334
against the full point's 0.345, and both points put the same eight controls
nearest 1. Re-run alone, the full root does not pick 17 either: its first full
solve picks 11, and its second triangle round raises `SolverError` after 123 s,
which `size_certificate` turns into an open size branched on the first round's
point. In the profile that solve succeeded, and the sixth round's point picked 17.
On the beds the full point is integral at every branching root (227 of 227), so
its nearest-a-half pick is a tie decided by rounding, and branching on Shor's
point instead changed the beds' nodes by -0.9% to +1.6%.

With the full relaxation still solved at the root and the incumbents held equal,
the beds decompose the Shor root's extra nodes (4,718 to 4,994 at the five-pair
dual memory) into three parts that add up: branching on Shor's point, +76;
rounding the root's point from Shor's, +124; not solving the full relaxation at
the root, +76. The root keeps the full relaxation.

### Branching on the parent's point

A node of the branch and bound that the first-order tiers leave open goes to the
Shor and full relaxations, and on small simulated panels of the same form those
solves almost never close it. On 21 branching groups of 20 simulated markets (four
treated, 20 or 40 periods, enumeration turned off so every node is bounded), 127
nodes reached the full relaxation and none closed there. The solve still buys the
point the next control is branched on, the free control whose relaxed value is
nearest a half, and no cheaper point did as well: branching on the dual's smoothed
point instead took 1,029 nodes to the 487 of `main` and left eight groups at the
depth limit; the most decided control, 1,243 nodes and 21 open; the first free
control, 1,087 and 19.

So a node now solves the relaxations only at depths that are multiples of
`clarabel_every` (`CLARABEL_EVERY = 2`), and wherever the parent had no point; in
between it runs the first-order tiers (`size_certificate(..., relaxations=False)`)
and branches on its parent's point. What is proved does not change: a node is
pruned by its own certificate, and the rule only picks the control. On the three
beds (`results/node_reuse.log`):

| bed | rate | nodes | Clarabel solves | open at the depth limit |
| --- | --- | --- | --- | --- |
| 20 markets, 4 treated, 21 groups | every node (`main`) | 487 | 128 | 0 |
| | every second level | 523 | 78 | 0 |
| | every third level | 619 | 80 | 1 |
| | each size's root only | 607 | 28 | 3 |
| 30 units, 10 treated, 20 periods, 13 groups | every node (`main`) | 783 | 296 | 2 |
| | every second level | 871 | 193 | 3 |
| | every third level | 883 | 145 | 4 |
| 30 units, 4 treated, 20 periods, 25 groups | every node (`main`) | 1,450 | 492 | 3 |
| | every second level | 1,524 | 293 | 2 |
| | every third level | 1,660 | 247 | 6 |

Every second level takes 35-40% fewer Clarabel solves for 5-11% more nodes, and
no group that both runs certified has a different design. On these beds a solve
costs about a second, so the wall time moves little (140 s to 129 s, 316 s to
262 s); at 97 candidate controls a solve costs 100-145 s and a first-order node a
few seconds, so the saving scales with the solves. Every third level saves few
more solves and leaves more groups open.

### Rounding the relaxed point

A node that solves the relaxations and stays open rounds its point to a design of
its size (`rounded_design`): the controls it fixed in, and the free controls with
the largest relaxed values, ties to the lower index. The design is evaluated
exactly, and one better than the incumbent replaces it, so the nodes after it are
bounded at the higher ratio. Huang et al. (SoCS 2025) fix a relaxation's most decided
variables the same way (Relax-Search, Algorithm 2). Rounding prunes nothing by
itself, and what is proved does not change. `certify_slope_free(rounding=True)` is
the default, and the result's `rounded` counts the improvements.

On the two 30-unit beds, with the relaxations every second level
(`results/node_rounding.log`):

| bed | rounding | nodes | Clarabel solves | open at the depth limit | better incumbents |
| --- | --- | --- | --- | --- | --- |
| 10 treated, 13 groups | off | 871 | 193 | 3 | |
| | on | 685 | 139 | 3 | 24 |
| 4 treated, 25 groups | off | 1,524 | 293 | 2 | |
| | on | 1,404 | 255 | 3 | 48 |

Together 13% fewer nodes and 19% fewer Clarabel solves, and no group that both runs
certified has a different design. The start's local search often stops short of a
group's optimum, and each improvement raises the ratio every later node is tested
at. One group, seed 21 with treatment (8, 11, 18, 21), certified in 92 nodes without
rounding and reached the depth limit in 74 with it: a different incumbent changes
which control is branched on, and on these beds groups cross the depth limit both
ways between settings.

Two variants measured no better. Rounding the inherited parent's point at the nodes
in between gave the same counts. Re-checking a node at the new ratio after an
improvement closed none of 72 nodes and cost up to a Clarabel solve each, so the
build does not re-check.

### A cut for every incumbent

The search cut the start design from every size's relaxation with its no-good
cut, and no design adopted after it, by rounding, at a leaf or by pricing. The node
whose subtree holds an adopted design bounds that design at the ratio it set, so its
certificate cannot fall below zero, and the path to the design runs to the depth
limit. On `main`, every node bounded, 10 of the 14 nodes left open at the depth limit
on the beds held the final incumbent, and 10 of the 11 uncertified groups had one
(`results/incumbent_cuts.log`). The two variants above that measured no better are
this defect: the group of seed 21 that reached the depth limit with rounding, group
13 of `s30_k4`, held its final incumbent at the open node, and a re-check at the new ratio could not close a node that
still held the design that set it.

Each bounded node now cuts the incumbent it is bounded against, beside the caller's
cuts (`_branch`). A cut design is the incumbent or one it replaced, whose ratio is
below the incumbent's, so no design the claim is about is cut. Against a ceiling the
nodes are bounded at `y'y - ceiling`, above the incumbent's ratio, and its cut
excludes a design the claim does not need. On the beds:

| bed | nodes | full relaxations | open at the depth limit | uncertified |
| --- | --- | --- | --- | --- |
| `s15_k3` | 960 → 616 | 149 → 79 | 0 → 0 | 0 → 0 |
| `s20_k3` | 1,114 → 588 | 197 → 76 | 7 → 0 | 5 → 0 |
| `s20_k4` | 533 → 285 | 81 → 27 | 1 → 0 | 1 → 0 |
| `s30_k10` | 681 → 307 | 131 → 40 | 4 → 0 | 3 → 0 |
| `s30_k4` | 1,400 → 774 | 258 → 113 | 2 → 0 | 2 → 0 |
| all | 4,688 → 2,570 | 816 → 335 | 14 → 0 | 11 → 0 |

Every group certifies, and every group of the enumerated beds at the enumerated
optimum. The group seconds went from 369.6 to 253.9, the first on three workers and
the second on four, so the time comparison is indicative; the counts do not depend
on the machine.

With the cut, a re-check closes nodes. Bounding the node whose rounding adopted a
design again, under the design's cut, took 2,068 nodes against 2,570 and 397 full
relaxations against 335, in 259 s against 287 s on the same machine. At 100 geos a
full relaxation costs 100-145 s and a first-order node a few seconds, so the extra
solves cost more than the nodes save, and the build does not re-check.

### The relaxation on its face

Every point Z of a size's relaxation has null directions the rows force: v =
(-m, 1, ..., 1), since sum u = m and its products say Zv = 0; e_j for a control
fixed out; e_0 - e_j for one fixed in. So no point is positive definite, the
relaxation has no interior (Burer 2012, Prop. 8.3, in the Handbook on Semidefinite,
Conic and Polynomial Optimization), its dual's optimal set is unbounded, and an
interior-point solver loses accuracy on it. On the beds Clarabel returned
`optimal_inaccurate` on 539 of 766 Shor solves and 682 of 832 full solves.

`face.py` writes Z = T W T' with T integer (Krislock and Wolkowicz, facial
reduction, in the same Handbook): row 0 is e_0, a free control's row other than the
pivot's is its unit vector, the pivot's row is q e_0 minus the other free controls',
a control fixed in maps to e_0 and one fixed out to zero, with q = m less the
controls fixed in. Clarabel solves for W >= 0, and the rows the face satisfies
identically are dropped: sum u = m becomes m W_00 = m, its products become 0 = 0,
and the fixings vanish. A structural row the face reduces to a constant it does not
meet means the node holds no design; `reduce` raises `FaceError` naming the row and
the fixings, since the search never builds such a node.

The certificate stays exact and gains a second form. For any multipliers <S, Z> =
<T'ST, W> and tr W = 1 + sum u_j over the free controls but the pivot, at most 1 + q,
so every design is at most k + (1 + q) max(0, lambda_max(T'ST)). T'ST is `T'NT / 2**e`
for the dyadic `S = N / 2**e`, computed in integers. T is not orthonormal, so at
multipliers far from the face's this can exceed the full-space bound; `bound` takes
the smaller of the two. The first build formed T'ST in Python fractions, which took
2.9 s of every 3.7 s of the face's term and made five s30_k4 groups 8% slower than
the full space (`results/face.log`); in integers they ran 2% faster.

On the beds, both arms under the same load (`results/face.log`):

| bed | nodes | full solves | Shor `optimal` | full `optimal` | solve seconds | uncertified |
| --- | --- | --- | --- | --- | --- | --- |
| `s15_k3` | 974 → 970 | 151 → 152 | 84 → 149 of 149 | 80 → 98 | 8.3 → 7.5 | 0 → 0 |
| `s20_k3` | 1,122 → 1,100 | 203 → 194 | 64/192 → 185/185 | 39 → 72 | 27.7 → 21.7 | 4 → 5 |
| `s20_k4` | 525 → 525 | 79 → 79 | 27/78 → 77/77 | 20 → 32 | 10.9 → 8.7 | 1 → 0 |
| `s30_k10` | 691 → 679 | 139 → 132 | 29/125 → 122/122 | 5 → 36 | 31.7 → 24.6 | 3 → 4 |
| `s30_k4` | 1,406 → 1,404 | 260 → 259 | 23/222 → 221/221 | 6 → 60 | 142.0 → 122.9 | 3 → 2 |
| all | 4,718 → 4,678 | 832 → 816 | 227/766 → 754/754 | 150/832 → 298/816 | 220.6 → 185.4 | 11 → 11 |

The table is the face with its substituted rows. With the sparse rows described below,
the recorded baseline (`results/beds_baseline.json`) has 4,688 nodes, 816 full solves,
11 uncertified groups and no optimum missed; each bed's node count is within 1.5% of
the table's, as round-off breaks the branching ties differently.

Every Shor solve on the face is `optimal`, and the face solve's multipliers certify
its value: over 216 solves on panels of five to seven controls the exact bound at
them is within 1e-7 relative of the solve's value. Solving takes 16% less time, the
groups 4% less, and every group that certifies certifies the enumerated optimum.
The full level stays mostly inaccurate, 298 of 816 `optimal`: its McCormick pair
rows leave a second degeneracy at a rank-one optimum, which facial reduction does
not touch. The uncertified count is unchanged and the groups in it move both ways
between beds, each with a node left open at the depth limit. The full point is
integral at every branching root of the beds (227 of 227, the Shor root above), so
the most fractional control is a tie that round-off decides, and a more accurate
solve breaks the ties differently. Group 0 of `s30_k10`,
the node-rounding tests' former fixture, is one: on the face it stops at one open
node of size 8 at depth 12 with and without rounding, so the tests moved to group
10, which certifies both ways (112 nodes without rounding, 72 with).

At 96 candidate controls the face costs time. At the size-21 and size-24 roots of
the 100-geo group (4, 27, 33, 58), both arms at once (`results/face.log`):

| root | Shor seconds | Shor status | full seconds | full status | full-level gap |
| --- | --- | --- | --- | --- | --- |
| size 21 | 35.6 → 42.3 | inaccurate → `optimal` | 118.9 → 149.6 | inaccurate → inaccurate | 6.2e-4 → 4.8e-6 |
| size 24 | 37.9 → 50.3 | inaccurate → `optimal` | 107.9 → 137.8 | inaccurate → inaccurate | 3.3e-4 → 7.3e-5 |

The gap is the exact bound at the solve's multipliers less the solve's value,
relative to the value; neither root closes at either level, on or off the face.
Substituted, the full level's pair rows that touch the pivot densify, since the
pivot's row of T is q e_0 less every other free control: 262,300 nonzeros against
73,825, 574 rows with more than 50 entries. So the solve carries that row as a
vector a, a_b = Z_rb, with one definition row per entry, and every other entry of Z
is one entry of W: no row grows past twice its nonzeros (155,146 in all, 27,649 of
them in the definitions). It cut a full-level iteration from 3.08 s to 2.64 s, level
with the full space's 2.63 s. What remains is the number of iterations: at the
size-21 root 16 against 14 at the Shor level and 53 against 45 at the full level,
at 2.5 to 3 s each, the cost of the 97 x 97 cone, on the face or off it. Clarabel's
time at 100 geos is set by the cone and the iterations, not by the rows.

A node whose fixings leave no design with c'u >= 0 has an infeasible relaxation.
Off the face Clarabel reports it with no point; the face solve reports the same.
`size_certificate` closes such a node at its dual tier before any solve.

`FACE = True` is the default; `certify_slope_free(face=False)` solves in the full
space.

### The Shor tier, off

The cascade was built as reuse, Shor, full, and the Shor relaxation then closed 212
of the 242 recorded GeoLift sizes. The dual tier came after it and bounds the full
rows, so a size the weaker Shor relaxation could close it closes first. Measured
again on the face (`results/shor_tier.log`): over every bed group the Shor
relaxation closed 0 of the 756 sizes it was asked to, and on the GeoLift window at
k = 4 0 of 3; each went on to the full relaxation. The Shor solves were 32.1 s of the
beds' 367.3 s in size certificates.

`SHOR_TIER = False` is the default; `size_certificate(shor=True)` and
`certify_slope_free(shor=True)` solve it. With it off the beds have the same nodes,
full solves, open nodes and uncertified groups as the recorded baseline, and no
Shor solve. At the 100-geo group's roots a Shor solve is about 40 s against about
120 s for a full one.

### The pair rows loosened, measured and not built

At an integral point three of a pair's four McCormick rows are tight whatever u_i
and u_j are: at 50 geos with 5 treated, 2,970 of the 4,051 rows (three for each of
the 990 pairs) carry a multiplier at the full level's optimum. Loosening the four
rows by eps removes that degeneracy, and any multipliers still certify the original
rows. It does not pay (`results/pair_perturb.log`). At 50 geos the full solves took
26 to 59 iterations loosened against 16 to 53 as they are, with bounds the same or
looser. At the 100-geo size-21 root it took 48 iterations at eps = 1e-4 and 42 at
1e-3 against 53 (118 s and 101 s against 128 s), but the exact bound rose 0.8% and
15.6%: the bound loses about eps times the pair multipliers' sum, 1.7e11 there, which
is more than the margin of the nodes that decide a search.

### SCS, an option, not the default

At 100 geos Clarabel's time is the 97 x 97 cone's factorization at each
interior-point iteration. SCS (O'Donoghue, Chu, Parikh and Boyd 2016) splits the
cone from the rows by ADMM on the homogeneous self-dual embedding, so an
iteration is one projection onto the cone and one sparse solve. Its multipliers
go to the same exact certificate. `solve(solver="scs")` and
`certify_slope_free(solver="scs")` use it, in the full space at
`SCS_EPS = 1e-6`: on the face the auxiliary rows slowed it (44,525 iterations
against 4,250 at the 100-geo size-21 root). A solve SCS does not answer (a limit,
an error) is reported and solved again by Clarabel, and every solve event and
`Solve` names the solver that answered. `beds --solver scs` runs a bed under it.

At the 100-geo group's roots, sizes 21 to 26, both arms at once
(`results/scs.log`), SCS solved the Shor level 117 times faster (2.4 s against
274.5 s over the six sizes) and the full level 11 times faster (64.7 s against
722.0 s), every solve `optimal`, with an exact bound at most 7.3e-5 above
Clarabel's. No root closes either way. With the Shor tier off by default (above),
only the full level's comparison applies to a search; the bed runs below were
measured with it on.

On the beds it is worse on every count:

| | Clarabel | SCS |
| --- | --- | --- |
| nodes | 4,688 | 6,446 |
| uncertified groups | 11 | 72 |
| group seconds | 429.2 | 3,679.0 |
| optima missed | 0 | 0 |

The nodes that decide a search sit within about 0.1 of zero on an objective
scale near 1e7 (on `s20_k4` group 0, 8.4e6). There SCS's value is off by up to
2.7 from Clarabel's, with its sign flipped at four of eleven, so its point, which
branching and rounding read, is a different point: that group takes 46 nodes and
leaves one open at the depth limit, where Clarabel certifies it in 22. Three of
its Shor solves at nodes with fixings took SCS 32,450 to 97,850 iterations (1.5
to 4.1 s), against about 0.02 s for Clarabel. So `SOLVER = "clarabel"`.

### The SCS workspace

Through cvxpy, SCS factors its linear system again at every node and starts every
node from zero. `workspace.py` calls SCS directly and changes three things:

1. One workspace for a size, its cuts and its triangle rows. A node's fixing
   `u_j = v` is two box rows, `u_j <= hi_j` and `-u_j <= -lo_j` with
   `lo_j = hi_j = v`; a free geo has `lo_j = 0` and `hi_j = 1`, which every design
   satisfies (`Rows(box=True)`). Fixings change only the right-hand side, so every
   node of a size reuses one factorization and starts from the last node's iterate.
2. Two stages, a small dual scale (`scale=0.01`) for two chunks and then SCS's
   default, each node running both from its warm start.
3. A certificate after every chunk of 200 iterations: the float screen, and when it
   passes, the exact certificate on the node's own rows. The first chunk that
   closes the node ends the solve.

SCS's stopping test decides nothing that is proved. Each solve reports one of
`certified`, `converged`, `cap` (every stage spent its chunks) or `infeasible`, and
an SCS error is reported and the node solved by Clarabel.

A triangle round builds a workspace with the new rows and starts it from the last
round's iterate, the new rows' multipliers at zero. Every size starts from a pool,
the triangle rows active at the last root solved (a multiplier above 1e-7 of the
largest); `Workspaces` holds one search's workspaces and its pool. In the spike the
pool was module state, and on the beds a group with fewer controls, run after a
larger one in the same process, read triangle indices past its matrix
(`test_two_groups_in_one_process_do_not_share_a_pool`). The triangles were valid
rows, only of another group.

Under the workspace a size takes the first first-order schedule and no first-order
triangle rounds: a relaxation on it costs about what a longer schedule does and
closes more. The group screen and the Shor tier solve one relaxation each, so they
stay on Clarabel. `certify_slope_free(solver=None)` takes `default_solver(p)`: the
workspace from `WORKSPACE_FROM = 40` candidate controls, Clarabel below.

Measured with the code of this branch, both arms of a row at once, four processes on
four cores (`results/workspace.log`); every pair certified the same design.

| case | candidate controls | question | Clarabel | workspace | nodes |
| --- | --- | --- | --- | --- | --- |
| `geo_panel(0, n=90, p=100)`, treated (4, 35, 58, 60) | 96 | exact | 1,122.3 s | 225.3 s | 41, 41 |
| the same panel, treated (4, 27, 33, 58) | 96 | below RSS 1,324.85, the 1% gap's target | 833.4 s | 377.1 s | 47, 47 |
| six `geo_panel(s, n=90, p=50)` groups, 5 treated | 45 | exact | 105.0 s | 65.4 s | the same |
| `geo_panel(0, n=90, p=60)`, `certify_treatment` at 4 treated | 56 | exact | 244.7 s | 117.5 s | |
| the same search | 56 | 1% gap | 68.9 s | 70.2 s | |

The two 100-geo rows include the group screen, whose Clarabel solve at 96 candidate
controls takes 129-163 s in either arm; #1015 drops that solve from 90 candidate
controls. On the 50-geo groups the workspace was faster in all six (6.4 to 15.3 s
against 6.6 to 31.5 s). In the treatment search the time is the one group that
branches: 75.4 s against 206.4 s exact. At the 1% gap that group closes in about 28 s
either way, and the rest is the floors and the screens.

On the five test beds (15 to 30 units, below the default's threshold), with
`solver="workspace"` forced, every group was certified at the enumerated optimum, as
under Clarabel:

| bed | Clarabel nodes | workspace nodes | Clarabel | workspace |
| --- | --- | --- | --- | --- |
| `s15_k3` | 616 | 616 | 35.9 s | 39.0 s |
| `s20_k3` | 588 | 594 | 64.7 s | 68.7 s |
| `s20_k4` | 285 | 285 | 21.1 s | 21.6 s |
| `s30_k10` | 307 | 307 | 27.1 s | 28.3 s |
| `s30_k4` | 774 | 786 | 113.9 s | 103.8 s |

The two are even there, 262.7 s against 261.3 s in all, and the workspace pays from
the 45 controls of the 50-geo groups; the threshold of 40 sits between them. SCS
through cvxpy, on the same beds, left 72 groups uncertified (above).

The first mutation run left four of the workspace's fourteen mutants alive, each for
an assertion too weak to see it: the fixings dropped from the box rows (the tightness
test's fixings never bound the relaxation), the certificate tried only once SCS
converges (on the easy instance SCS converged in its first chunk), the pool set at a
node with fixings (its active triangles matched the root's), and the iterate of a
failed solve kept (the error came at the first solve, when there was none). Each test
now has a case the mutant changes, and all fourteen are killed.

### Where the workspace's time goes

SCS reports its own timers. At two roots of the 100-geo group (4, 35, 58, 60), the
incumbent's no-good cut in place, at the default settings (`results/solver_internals.log`):

| size | iterations | node | cone projection | linear system | acceleration | outside SCS |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 6,375 | 16.2 s | 9.8 s | 4.4 s | 0.6 s | 0.6 s |
| 21 | 4,025 | 9.9 s | 5.6 s | 2.8 s | 0.4 s | 0.5 s |

An iteration costs about 2.4 ms, three fifths of it the projection onto the
semidefinite cone, a full eigendecomposition of the 97 x 97 matrix. The workspace's
own work (the multipliers, the screen and the exact certificate) is under 5%, and
building the workspace 0.4 s. So the levers inside the solve are a cheaper
projection and fewer iterations.

### A low-rank projection, measured and not built

The projection needs only the eigenpairs on one side of zero, so it is cheap when the
iterate is close to low rank. At 97 x 97 a full eigendecomposition takes 0.92 ms; the
top eigenpair 0.28 ms, the top three 0.40 ms and the top ten 0.66 ms. The same two
roots solved to 1e-9 (21,125 and 24,125 iterations) end with Z of rank 37 and 35 at a
cutoff of 1e-6 of the largest eigenvalue: one large eigenvalue (17.1 and 20.6), then
1.5 and 1.4, and some thirty more above the cutoff. About 35 eigenpairs cost no less
than all 97, so the projection cannot be made cheaper this way.

The dual matrix has rank 54 and 58, so the two ranks sum to 91 and 93, not 97: strict
complementarity fails at these roots (at the same cutoff, on approximate solutions).
Local linear convergence of ADMM rests on it (Ding, Lu and Yang 2026), which fits the
4,000 to 6,400 iterations a root takes at the working tolerance.

### A working set of pair rows, measured and not built

At those roots 3,703 and 3,731 of the 18,240 pair rows carry a multiplier, against 2
and 7 of the other 194 inequalities. Primal-dual methods speed up once the active rows
are identified, without strict complementarity (Díaz, Izquierdo Lehmann, Lu and Yang
2026), and the certificate holds with a dropped row's multiplier at zero, so a
workspace built on the active rows alone loses no soundness.

The spike solved each size's root and two children (the most fractional geo fixed to
one and to zero), sizes 21 to 26 of the group (4, 35, 58, 60) exact and sizes 24 to
29 of the hard group (4, 27, 33, 58) at its 1% target. The working-set arm kept the
pair rows active at the previous size's root; where a node stayed open and its point
violated dropped rows, it added them and solved again. Every bound was certified on
the full rows (`results/working_set.log`):

| group | arm | closed | iterations | time |
| --- | --- | --- | --- | --- |
| (4, 35, 58, 60) | all rows | 6 of 18 | 95,625 | 238.1 s |
| (4, 35, 58, 60) | working set | 6 of 18 | 83,300 | 529.8 s |
| (4, 27, 33, 58), 1% | all rows | 2 of 18 | 107,075 | 275.1 s |
| (4, 27, 33, 58), 1% | working set | 2 of 18 | 74,350 | 484.9 s |

The active rows do not carry over: a size's 3,000 to 4,000 active pair rows are a
different set from its neighbour's, and at 28 of the 30 nodes that kept a previous
set the point violated 159 to 891 of the dropped rows. Nearly every node paid for a
second solve, and the working set took about twice as long with the same closures.
The second solve started cold in a new workspace; a warm start would recover part of
it, but the first pass is spent either way, so the ceiling is about a quarter to a
third of the solves' time.

### The certificate in floating point

`bound()` sums the certificate's `S` and `k` in rationals, then bounds
`lambda_max(S)` exactly. On the profiled 100-geo groups that took 13.7 s and 15.7 s of
181 s and 170 s, about 0.33 s a call, and the rational sum (`_slack`, 638,000
`Fraction` operations on four easy sizes) was most of it. `fbound.py` decides the
same certificate in doubles, as a verification method in the sense of Rump (2010):
a theorem whose assumptions the computer checks, with every rounding bounded
(`agents/agents_solvers.md`, rules 17 to 27).

- The slack in doubles, `fl(S)` and `fl(k)`, with `|fl(S) - S| <= E` and
  `|fl(k) - k| <= e_k` entrywise. Each entry of `S` is `A`'s entry less a sum of at
  most `N` products, so its rounding is at most `gamma_{N+3}` times the sum of the
  moduli plus `(N + 3) eta` for underflow (Rump (2.3), (2.4), Theorem 2.1), plus
  `A`'s own rounding. A coefficient that rounds into the subnormal range is not
  within a relative `u` of its double, and turns the floats off for those rows.
- The upper bound: `lambda_max(S) <= lambda_max(fl(S)) + ||E||_inf` by Weyl's
  inequality (Rump Lemma 10.12), `lambda_max(fl(S))` from the Cholesky-residual
  bound of `certificate.lambda_min_lower_float`. The face's `T'ST` is bounded the
  same way, `T` integer, and combined with the full-space bound as `bound()`
  combines them.
- The lower bound: the Rayleigh quotient at the top eigenvector, its rounding and
  `|v|'E|v|` subtracted. It is at most the dual function, and so at most `bound()`.

`certificate()` closes a size when the upper bound is at most zero, rules it out
when the lower bound is above zero, and calls `bound()` only between them, so it
closes every size `bound()` closes. Every tier of `size_certificate`, the group
screen and the SCS workspace decide through it. The float screen it replaces
(`_screen`) was unrounded and decided only whether to try `bound()`; the
certificate costs about what the screen did and is a proof.

On the same two groups (`results/float_bound.log`, the spike run beside the exact
bound at every call):

| group | gap | calls | exact `bound()` | float certificate | closures | in the band |
| --- | --- | --- | --- | --- | --- | --- |
| (4, 35, 58, 60) | exact | 41 | 13.7 s | 1.8 s | 38 = 38 | 0 |
| (4, 27, 33, 58) | 1% | 45 | 15.7 s | 2.0 s | 44 = 44 | 0 |
| (0, 1, 2, 3), 60 geos | exact | 16 | 1.9 s | 0.26 s | 16 = 16 | 0 |

The float bound was within six digits of `bound()` at every call and never below
the exact lower bound on the dual function. No call fell between the float bounds,
so the exact tier ran nowhere in these runs; it is there for the cases the error
bound cannot separate from zero (rule 18). The saving is about 7% of a 100-geo
group's time; the rest is SCS and the dual tier.

The first mutation run left one of the thirteen `tbr-bnb-fbound` mutants alive: the
face's error bound without the slack's own error carried through `T`. The face's
enclosure test compared the face against the exact face of the exact slack, where
the slack's error is a few units in the last place and the rounding term covers it.
A property now checks the face against every matrix within `E` of the slack, with
`E` as wide as the slack, and kills it.

### The rows by index arithmetic

`Rows` built its table of rows as dicts of exact coefficients at every construction,
and `Rows.matrices` turned the table into sparse matrices entry by entry. On the 100-geo
group (4, 35, 58, 60) the search built `Rows` 102 times (17.3 s) and called
`matrices` 101 times (4.5 s). The solvers and the float certificate read only the
counts, the right-hand sides, the rows that carry `c` and the sparse matrices, so
`Rows` now computes those directly from the row families' index patterns: a build
takes about 0.1 ms and the matrices under 2 ms. The dict table, which the exact
certificate (`_slack`) and `holds` read, is built the first time something asks for it.
`test_fast_rows.py` pins the two paths equal: every count, right-hand side and row
index, and the matrices bit for bit at any scale of the `c` rows, as a hypothesis
property over sizes, levels, cuts, triangles, fixings and the box rows.

Against main, both at once (`results/fast_rows.log`):

| group | main | fast rows | |
| --- | --- | --- | --- |
| (4, 35, 58, 60), exact | 158.9 s | 139.6 s | 42,600 SCS iterations and 41 nodes each |
| (4, 27, 33, 58), 1% | 148.6 s | 128.0 s | 17,925 SCS iterations and 47 nodes each |

Both certified the same designs.

### Ending a triangle round early

The full tier solves a size's relaxation on the SCS workspace, separates the most
violated triangle inequalities at its point and solves again, up to `rounds` times.
On the 100-geo groups most rounds ran SCS to its cap, although the point had shown,
chunks earlier, the triangles the next round would add. So a round that another
round follows ends after `SEPARATE_AFTER` (5) chunks, counted over its stages, when
its point violates a triangle by more than `SEPARATE_TOL` (1e-2): the outcome is
`"separate"`, and the next round adds the triangles and starts warm. The last round
runs to its end. The certificate is computed at every chunk and decides first, so a
round that closes still reports `"certified"`, and an early end changes only which
rows the next solve has, never what is proved.

The two thresholds came from a sweep on the two 100-geo groups (spike runs on main,
`results/early_triangles.log`):

| chunks, violation | (4, 35, 58, 60), exact | (4, 27, 33, 58), 1% |
| --- | --- | --- |
| no early end | 176.6 s, 42,600 SCS iterations | 167.7 s, 17,925 |
| 10, 1e-2 | 125.5 s | 148.1 s |
| 10, 1e-3 | 135.5 s | 148.2 s |
| 5, 1e-3 | 145.3 s (47 nodes) | 136.6 s |
| 5, 1e-2 | 108.2 s, 13,000 | 137.2 s, 6,000 |

Built, against its base, both at once:

| case | base | early end | |
| --- | --- | --- | --- |
| (4, 35, 58, 60), exact | 160.8 s | 94.3 s | 42,600 to 13,000 SCS iterations, 41 nodes each |
| (4, 27, 33, 58), 1% | 150.2 s | 124.3 s | 17,925 to 6,000, 47 nodes each |
| 60 geos, 4 treated, exact | 101.7 s | 84.9 s | same treatment |
| the five beds | 160.8 s | 154.9 s | 2,588 and 2,578 nodes; 315 and 343 full solves |

Every case certified the same design, and no bed group missed its enumerated optimum.

### Accelerating SCS, measured and not built

Three levers on SCS itself, each measured at the setting above or beside it:

- Anderson acceleration. SuperSCS (Sopasakis, Menounou and Patrinos 2019) owes its
  speed to Anderson and restarted Broyden directions, and SCS 3 applies Anderson
  acceleration by default (memory 10). At memory 0, 5, 10 and 20 the exact group
  took 109.5, 114.0, 108.2 and 109.0 s and the 1% group 138.0, 139.3, 137.2 and
  139.6 s; type-I acceleration (memory -10) ran past 750 s on both and was
  stopped. SCS's safeguard accepted 7 of 619 accelerated steps: on this relaxation
  the lever is spent.
- A warm start from the neighbouring size's last iterate. On the exact group 42,600
  to 38,300 iterations, on the 1% group 17,925 to 27,200, size 22 hitting its cap.
- A first-order prediction from the derivative of the solution map (Agrawal, Barratt,
  Boyd, Busseti and Moursi 2019). The derivative needs a unique, strictly
  complementary solution; this relaxation has no interior and an unbounded dual
  optimal set (`face.py`), and in the paper's own example one derivative cost about
  one solve (191.7 s against 195.7 s). Not tried.

An SCS iteration costs 2.1 to 2.5 ms here, about 55% a full eigendecomposition of
the 97 x 97 block and 35% the back-solve, so what is left inside SCS is the number
of iterations, which the early end cuts. An interior-point method pays a Schur
complement of order `n^2 mu^2 + n^3` a step, with about 18,000 rows and `mu = 97`
(Nesterov and Nemirovskii 1994, section 6.4.1), which is why Clarabel took 100 to
145 s a full solve.

### The dual tier and the screen, measured and not built

With the early end and the fast rows, the dual tier is 75.6 s of the 1% group's
113.8 s under the profiler: `setulb` (L-BFGS-B's own step) 28.6 s over 50,236
calls, the evaluations 34.2 s, of which the 97 x 97 eigendecomposition is 22.0 s.
On the same group the group screen is 44.9 s, three calls that do not close.
Four levers were measured against that (`results/dual_tier.log`), every one on
the dual's own calls, replayed with the same start, smoothing schedule, budget and
stopping rule:

- A cheaper step. Spectral projected gradient (Birgin, Martinez and Raydan 2000)
  with Fletcher's (2005) non-monotone window costs 1.4 ms an evaluation against
  L-BFGS-B's 2.8 to 3.1, and closes 9 to 11 calls where L-BFGS-B closes 14 and 16;
  given twice the budget, the same time, it closes 10 against 14. A projected
  L-BFGS with the two-loop direction on the variables off their bound (Kim, Sra
  and Dhillon 2010) closes 10 and 12. Neither closes a call L-BFGS-B leaves open,
  and each call left open costs a relaxation solve, so L-BFGS-B's generalized
  Cauchy point and subspace step pay for themselves.
- Giving up early. Over 2,273 recorded calls (both 100-geo searches and the five
  beds), the calls that never close spend 123,371 evaluations. Giving up at
  evaluation 100 when the best certificate is above 0.25 of its start loses no
  closure and saves 15,351 of them, a few percent of the time; at 0.20 it loses
  two, at 0.201 and 0.239. The margin is too thin for the gain.
- Skipping the screen's longer rungs. On the 56 screens of the 60 and 80-geo,
  four-treated searches at 1%, the screens a later rung closed ended rung 0 at
  best ratios up to 0.048, and those no rung closed at 0.052 to 0.064 (0.051 on
  the hard 100-geo group). One rung later, rung 2 closed a screen whose rung 1
  had reached 0.0080 and left open one at 0.0081. Neither rung predicts the next.
- A partial eigendecomposition. At the dual's iterates S(y) has a median of 12
  positive eigenvalues of 97, but LAPACK's `syevr` restricted to them takes
  0.89 ms against 0.86 ms for the full decomposition: the reduction to tridiagonal
  form, which both pay, is the cost at this size.

## Several processes

`certify_treatment(..., n_jobs=)` computes the floors in chunks over a pool of
forked processes, and runs the pass and the certifications in waves of `n_jobs`
groups. A wave's groups are screened and given their ceilings from the incumbent
as the wave fills, and their results are applied in queue order, a group
replacing the incumbent only with a strictly larger key. A certificate proved
against a ceiling that the wave's later results lower still holds, since no
design below a ceiling means none below a lower one, so the certified design is
the sequential search's at every `n_jobs`, ties included; `n_jobs = 1` starts no
pool and is that search. `test_parallel.py` checks the two agree as a hypothesis
property, against enumeration, and that a group left open in a wave is retried
and a worker's error reaches the caller.

Clarabel runs on one thread in every solve of the study (`max_threads=1`). Its
automatic threading starts a native thread pool on a relaxation of the GeoLift
groups' size, six threads in the parent after one solve against two on one
thread, and a worker forked after that inherits the pool's state but not its
threads: its next solve never returns. The first four-process run hung that way
in its first certification wave that reached Clarabel, every process asleep and
the stuck worker inside `slopefree.solve`; the small test panels had never
reached Clarabel in a worker. `test_parallel.py` solves a relaxation of the
recorded size in the parent and then in a forked worker.

On GeoLift at four geos (`results/parallel.log`, one host, BLAS on one thread):

| `n_jobs` | wall | pass groups searched | groups certified | the design |
| --- | --- | --- | --- | --- |
| 1 | 163.1 s | 90 | 33 | austin, dallas, minneapolis, reno |
| 4 | 64.5 s | 95 | 33 | the same, key and RSS equal |

The four-process pass searches five more groups: a wave admits its groups against
the incumbent as it starts, so a group the wave's own better design would have
screened out is searched anyway. The search before the pool (the library's
greedy and local search) runs in one process; the start group is certified in a
wave like every other group (next section).

## Halving the four-geo run

A profile of the four-geo GeoLift search on one process (98 s on that host)
put 36 s in the pass, 34 s in the certifications (11.4 s of them the start
group's), 16 s in the floors and 3.4 s in the start. Three changes halve it:

- The start group is certified in its turn. The search used to certify the
  start group's own optimum before any incumbent existed. The design that chose
  the start group is now the first incumbent without a certificate (the
  library search's design, which passes its gates, or the group's local design
  when the caller gives the start), and the start group joins the queue, to be
  certified against the incumbent of its turn like every other group. A design
  is an incumbent whatever proves it, and the start group's own certificate is
  still required.
- ExactData builds its Gram, cross products and y'y from the doubles' integers
  over one power of two, one division each: 8 ms a group against 223 ms for the
  Fraction products, the same rationals.
- The incumbent's RSS descent (`search.incumbent`) scores every move of a step
  at once from c = X'y, G = X'X and v = G u: an add, drop or swap changes c'u and
  u'Gu by a few entries of c, v and G. Scored from the series one move at a
  time it took 0.48 s a group; from the Gram, 0.011 s, with the same designs on
  the 73 groups measured.

On GeoLift at four geos (`results/gram_descent.log`, `main` and this branch on
one host):

| `n_jobs` | `main` | this branch | the design |
| --- | --- | --- | --- |
| 1 | 110.2 s | 47.8 s | austin, dallas, minneapolis, reno |
| 4 | 50.5 s | 23.0 s | the same, key and RSS equal |

With the start in its turn the pass searches 98 groups, not 90: its incumbent is
the library's design, weaker than the start group's certified optimum it used to
start from, so fewer groups are screened out before the pass. One more group is
screened out before certification and one fewer certified (91,358 and 32, against
91,357 and 33). `results/start_last.log` has the timing of the first two changes
alone: 94.8 s and 37.6 s.

Three other changes were measured and not made:

- Fewer pass restarts. At 10 restarts the run took 94.8 s and certified 32
  groups; at 5, 331.7 s and 58; at 2, 551.9 s; at 1, 571.6 s; with no pass,
  431.5 s and 100. A weaker pass leaves a weaker incumbent, so more groups
  survive the screen and each is certified against a higher ceiling. On four
  simulated 14-geo panels at three geos every setting ran in about a second.
- Warm-starting the group screen's first-order multipliers from the previous
  group's. On the 31 recorded survivors it took 6,804 evaluations against 6,655
  from zero, and closed the same 29: the multipliers belong to each group's
  objective, and a start near the boundary begins the smoothing too fine.
- Floors from the global Gram. The floor's parts take 7.8 s (the hull
  certificate), 7.2 s (NNLS), 4.8 s (the exact y'y) and 2.3 s (the split and
  the cap) on one process. The exact y'y has to be the floating-point sum's, which
  the Gram of the panel does not give, and the hull certificate's rounding
  analysis is written for the series; a Gram form needs its own, for at most
  15 s of the run.

## The y'y bracket and the screen in doubles

After the halving, the floors' exact y'y (the sum of squares in Python integers)
took 6.1 s of the 48 s four-geo run, and the screen's Fraction comparisons took 2.4 s
more. A group needs y'y twice: for its floor's slope cap, and for the screen
against the incumbent's ceiling. Neither needs it exactly.

- `yy_bracket` bounds y'y in doubles, lo <= y'y <= hi, from `math.fsum` of the
  squares, widened by 2^-48 either side. Each square of an entry that is zero
  or between 2^-500 and 2^500 is a normal double within a relative 2^-53 of the
  exact square, and `fsum` rounds their sum once more, so the sum is within
  about 2^-52 of y'y relative, sixteen times inside the margin (derived here). A series with an entry outside that range, a
  non-finite entry or more than 2^20 periods gets no bracket and keeps the exact
  sum.
- The slope cap reads the bracket's top: the cap needs y'y from above.
- The ceiling is nondecreasing in y'y, so a group competes for every y'y in the
  bracket when its floor is below ceiling(lo), and for none when the floor is at
  least ceiling(hi). `compete_bracket` decides both in doubles across a relative
  margin of 2^-48. The ceiling's terms and products are each rounded once, which
  moves either side by under 5 * 2^-53 (derived here). A ceiling below 2^-1000,
  where rounding is no longer relative, is left undecided.
- An undecided group, and every group that is certified, reads the exact y'y,
  computed once a group. The certificate's ceiling is the exact one.

On the four-geo GeoLift run, 91,390 brackets take 1.2 s and the 182,779 screens
take 0.4 s, none of them undecided. The exact y'y is computed for the 32 certified
groups only. `results/float_screen.log` has two alternating runs of each branch on
one host:

| `n_jobs` | `main` after #989 | this branch | the design |
| --- | --- | --- | --- |
| 1 | 46.8 s, 47.5 s | 42.8 s, 40.8 s | austin, dallas, minneapolis, reno |
| 4 | 22.1 s, 22.2 s | 19.4 s, 18.1 s | the same, key and RSS equal |

The counts are the same: 98 groups searched in the pass on one process (103 on
four), 91,358 screened out and 32 certified.

On a panel scaled by 2^-520, no group has a bracket, so the search takes the exact
route throughout. Its answer there is not the unscaled panel's. The key's
correlation is computed as s_xy / sqrt(s_xx s_yy), and that product leaves the
double range below a scale of about 2^-270 and above 2^270 (#990).

## The floor's per-call costs

After the float screen, the floors took 16.1 s of the 42 s four-geo run, about
0.18 ms a group: NNLS 6.4 s, the hull certificate 6.8 s, the bracket 1.2 s, and the
split and the cap 1.7 s. A spike over sampled groups measured what each part of a
floor costs and what would replace it:

- Floors from the Gram. The panel is 90 periods by 36 controls, so the Gram does
  not shrink the problem: NNLS through a Cholesky factor of the group's Gram took
  88 us against 65 us on the series, and slicing the group's Gram out of the
  panel's took 32 us.
- A first-order point in place of NNLS. The floor adds cap |g_j| for each
  coordinate whose gradient is still negative, and the cap is a median 80 times
  the largest weight, so the floor is only useful at a point that meets the
  optimality conditions almost exactly. Batched FISTA, with the box projection
  (a clip; the hull's simplex projection has nothing to do here) and restarts,
  screened out none of 4,000 groups after 200 steps (140 us a group) and 412 after
  1,000 (730 us), against 3,998 for NNLS (124 us under the same load). Plain
  projected gradient and a clipped least-squares point did no better. The
  groups' X'X have condition numbers near 1.7e4.
- The certificate's own costs. With lo = 0 and the size at the number of controls
  the knapsack's sum constraint is implied by the box, so the greedy fill (6 us)
  is not needed: `box_bound_float` takes the multiplier as 0, valid at any
  multiplier and the box minimum there. The last step's four rationals (10 us)
  are three additions each rounded one step down (`add_down`), and the cap's
  Fractions (6 us) are doubles where the quotient y'y / lambda is a normal double
  (`slope_cap`). The bound is the same to 9e-16 relative on the groups sampled.

On GeoLift at four geos (`results/lean_floor.log`, two alternating runs of each
branch on one host):

| `n_jobs` | `claude/tbr-float-screen` | this branch | the design |
| --- | --- | --- | --- |
| 1 | 40.8 s, 41.4 s | 37.4 s, 35.8 s | austin, dallas, minneapolis, reno |
| 4 | 20.3 s, 19.6 s | 17.7 s, 18.0 s | the same, key and RSS equal |

The counts are the same.

The exact cap that remains for the other cases read float(y'y / lambda), which
raised past the doubles, and below them left a zero cap that its widening loop
could not raise. It is now an integer root (`root_above`), at most 2^-64 above the
square root, rounded up to a double, or to inf, a cap whose bound clears nothing.

## The floors in one compiled loop

With the floor's per-call costs gone, a floor still cost about 0.16 ms, almost all of
it the Python and NumPy around a few thousand floating-point operations. With numba
installed (the optional extra `fast`), `fastfloor.floors` computes a chunk of groups
in one compiled loop, and `treatment._floor_chunk` uses it; without numba the old
path computes every group. For each group the kernel computes what `rss_floor`
computes:

- the series, summed left to right in ascending column order as numpy sums fewer
  than eight columns, so it is `_split`'s bit for bit (eight or more go to
  `rss_floor`);
- a bracket on y'y from the plain sum of squares and a gamma_{n+2} error bound
  (numba has no `math.fsum`; the bracket is wider than `yy_bracket`'s and the screen
  reads it the same way);
- the slope cap in doubles, as `slope_cap` computes it;
- an NNLS point by Lawson and Hanson's active set on the group's Gram, started from
  the previous group's support carried over by geo (11.1 solves a group against
  22.1 from cold, on the four-geo GeoLift groups);
- `box_bound_float`'s certificate at that point, term for term
  (`fastfloor._certificate`, whose pieces the tests compare with `certificate.py`'s).

A group the kernel cannot do in doubles comes back with a nan floor and takes
`rss_floor`'s path. The certificate holds at any point, so the solver decides only
how tight a floor is: on every four-geo group the kernel's floors are `rss_floor`'s
to 1e-10 relative.

The active set ends where a Cholesky solve on it fails, at its last feasible point,
with the column just added left out. The first version stepped toward the solution
of that failed solve, which is whatever an earlier solve left in the scratch; when
those entries were positive nothing left the set, and the same solve failed again
without end (`test_a_singular_active_set_ends_the_solve`, a singular three-column
Gram). The floors of real panels never reached it, since their c is the Gram's own
B'y, but a floor that can hang the search is a defect whatever its inputs.

numba keys its disk cache on the source file and records the module's name in it.
The suite imports the module as `studies.tbr_bnb.fastfloor` and a script as
`benchmarks.studies.tbr_bnb.fastfloor`, and a cache written under the first failed
to load under the second. The kernel is cached only under the second name and
compiles in memory under any other (3.3 s, once a process).

On the 91,390 four-geo GeoLift groups the floors take 2.0 s against 14.2 s, 22 us a
group against 156 us. End to end (`results/fast_floor.log`, two alternating runs
of each on one host):

| | main | with the kernel |
| --- | --- | --- |
| one job | 38.3 s, 36.5 s | 24.4 s, 24.1 s |
| four jobs | 19.2 s, 18.9 s | 15.1 s, 15.1 s |

Both certify austin, dallas, minneapolis and reno with the same key, 32 survivors,
91,358 groups screened out, and 98 groups searched on one job (103 on four).

Two changes to the smoothed dual come with this one. Each evaluation built S(y) and
decomposed it twice, `eigh` for the smoothing and `eigvalsh` for the unsmoothed value
that decides whether the dual has closed; the value now comes from `eigh`'s largest
eigenvalue. The bounds, a list of a few thousand pairs that every `minimize` call
converted, are one `Bounds` object per dual. The 49 dual calls of the four-geo run
replay with the same closures and the same 7,828 evaluations, in 6.7 s against 7.2
to 7.7 s.

Measured and not built:

- Cython. The same kernel compiled with Cython ran in 18.9 us a group against
  numba's 19.4 us, with identical weights. It would make the package build compiled
  extensions, with wheels per platform and Python version, for no gain in speed.
- A compiled solver for the smoothed dual. The dual spends about 0.5 ms an
  evaluation inside L-BFGS-B, outside the objective. A projected L-BFGS with its
  two-loop update in numba costs almost nothing per evaluation, but on the 49
  recorded dual calls it closed 15 (17 with L-BFGS-B's first step, interpolated
  backtracking and its memory kept across stages) against L-BFGS-B's 45, and took
  half again as many evaluations where both closed. About 2,630 of the 2,667
  multipliers are bounded and many sit at zero, the case L-BFGS-B's Cauchy point and
  subspace minimisation are built for.

## A lean NNLS in the kernel

On 80 geos the floors cover 1,581,580 four-geo groups and took 41.3 s of a 111.5 s
run on four jobs. Inside the kernel most of a group's time was the NNLS, and most of
that was work the solve did not need:

- each group copied its 76 x 76 block of the panel's Gram before solving;
- each step summed the gradient c - Gw over all 76 columns, though w is zero off the
  support;
- each column that joined the active set refactored the whole support's Cholesky
  factor from scratch.

The lean solve (`fastfloor._nnls_gram`) reads the Gram through the controls' index
`rest`, without a copy, sums the gradient over the support only, and extends the
factor by one row when a column joins (`_extend`, O(k^2) against O(k^3)). The factor
is rebuilt only after a column leaves. A column whose extended pivot is not positive
lies in the span of the support in doubles; the solve refuses it and stops at its
last feasible point, as the full refactor did.

Carrying the support over from the previous group changes the order in which columns
join, and so the factor's arithmetic, and so the point in its last bits. A floor that
depends on the group before it would make the search depend on how the groups are
cut into chunks and spread over processes. The solve therefore ends by re-solving
the final support in ascending column order (`_canonical`), the order the support's
own factor has from scratch: the point is then a function of the support alone. A
re-solve that is not positive, or a support that is not positive definite in
doubles, leaves the point as it was. The support itself is carried by geo (`held`),
not by position in the controls' list, so a group starts from the geos the last
group used.

The point only decides how tight a floor is, since the certificate holds at any
point of the box. `test_lean_nnls.py` keeps the old active set as a reference and
pins the lean solve to it: the same objective to 1e-10 relative, cold and warm, the
KKT conditions, `rss_floor`'s floors to 1e-9 on three-geo groups of an 18-geo panel,
and (a property) a group's floor bit for bit the same alone and after any run of
groups before it.

On 50,000 consecutive four-geo groups of the 80-geo panel, taken at three points of
the enumeration, the kernel takes 66, 54 and 62 us a group against main's 97, 73 and
91, and its floors are main's bit for bit on all 150,000. Starting every group cold
instead of from the last group's support takes 84, 72 and 72 us, with the same
floors. End to end on 80 geos, 150 periods, k = 4, a 1% gap and four jobs
(`results/lean_nnls.log`, two alternating rounds on one host):

| | main | lean NNLS | with the worker screen too |
| --- | --- | --- | --- |
| wall | 107.7 s, 104.9 s | 94.8 s, 95.6 s | 83.3 s, 88.4 s |
| the floors | 36.5 s, 36.8 s | 26.0 s, 26.2 s | 24.2 s, 25.2 s |
| the start | 52.2 s, 49.0 s | 50.1 s, 51.1 s | 49.9 s, 53.4 s |

Every run certifies the same treatment, controls, RSS and key, searches 7 groups,
screens out 1,581,572 and keeps 8 survivors. The start's 50 restarts in one process
are now the larger part of the run.

Mutants (`tbr-bnb-lean-nnls`): 10 on the lean solve, all killed but `the-support-is-not-carried`, which
starts every group cold. It gives the same floors by construction (`_canonical`) and
only costs time, so no test can tell it from the original; its cost is in
`results/lean_nnls.log`.

## The start's restarts on the pool

The search's first incumbent is the library's design: the greedy search, then
`local_search`'s descent from the greedy design and from 50 random designs. On 80
geos the greedy takes 1.1 s and the 50 restarts 47.7 s, about 1 s each, all in one
process, while the pool the search made sat idle.

`local_search` draws every restart from one generator seeded 0, and each descent
depends only on its own draw. With a pool, `_start` makes the draws itself in
`local_search`'s order (`_random_design`), descends them on the pool (`_descend`,
`local_search` from that design with no restarts), and takes the best in draw order:
the greedy's descent first, a later descent replacing the best only when its key is
strictly greater, which is `local_search`'s own rule. The result is
`local_search`'s design, and `test_parallel_start.py` pins it on the pool and in
this process, at fixed panels and as a property over panels, sizes and restart
counts, together with the tie rule and the draw that is no design.

On four processes the start takes 15.5 s against 53.4 s on 80 geos at k = 4, 5.4 s
against 20.2 s on 60 geos at k = 3, and 1.5 s against 4.6 s on 40 geos at k = 4,
each time with the same design. End to end on 80 geos, 150 periods, k = 4, a 1% gap
and four jobs, with the lean NNLS and the worker screen in place
(`results/parallel_start.log`, two alternating rounds):

| | the floors' two changes | with the pooled start |
| --- | --- | --- |
| wall | 90.1 s, 83.3 s | 52.6 s, 49.8 s |
| the start | 51.2 s, 49.0 s | 17.2 s, 15.8 s |
| the floors | 28.3 s, 24.9 s | 26.8 s, 24.7 s |

Every run certifies the same treatment, controls and key. The floors are now half of
the run.

Mutants (`tbr-bnb-parallel-start`): 8, all killed.

## Five geos

At five geos GeoLift has 658,008 treatment groups. On one process the search took
about 410 s and certified austin, dallas, jacksonville, portland and reno; 136 groups
survived the screen and their certifications took 379 s, 300 s of it one group's.
That group was the eventual optimum, certified from its pass design:

| start of the certification | RSS | sizes branched | nodes | open at the depth limit | time |
| --- | --- | --- | --- | --- | --- |
| the pass design (10 restarts, 20 controls) | 5,089,437.7 | 13 to 17 | 224 | 4 | 300.5 s |
| the RSS descent (50 restarts, 17 controls) | 4,860,705.3 | none | 23 | 0 | 10.3 s |

The second row is the group's retry at the final ceiling; from the same design at the
first ceiling the certification took 10.1 s. Both ceilings are above the group's
optimum, so both certifications prove that optimum, not a ceiling.

The pass gives every group the RSS descent's best of `pass_restarts` (10) starts, and
the certification started from that design without searching again. Its ten starts
stopped at a local optimum; fifty reach the group's optimum. The slope-free search
tests every size at its start's ratio, y'y less the start's RSS, and at the pass
design's ratio sizes 13 to 17 do not close at the root. Their branches found the
optimum but left four nodes open at depth 12, so the certificate came back open (the
gated search's break on an open certificate had been marked unreachable), and the
group was certified on its retry, which ran the descent.

`certify_slope_free` now runs its descent whether or not it is given a start, and
starts from the design with the higher ratio, the given one on a tie. The descent
costs a median of 50 ms a certification.

The first attempt at this fix was wrong. The slow group was taken to be the start
group, certified from the library search's design (chosen by the key, not the RSS),
and the start group was given the descent; at five geos that measured 426 and 408 s
against 402 and 423 s on `main`. The start group's certification took 0.2 s. The
record of each certification's start (`results/best_start.log`) showed the slow group
was a pass group.

On GeoLift (`results/best_start.log`, one host; `main` from two alternating runs with
the first attempt, this branch from two runs after them):

| | `main` | this branch |
| --- | --- | --- |
| five geos, one process | 402.4 s, 423.4 s | 65.0 s, 65.4 s |
| five geos, four processes | 371.1 s, 389.1 s | 39.6 s, 40.4 s |
| four geos, one process | 26.0 s, 25.8 s | 27.2 s, 27.3 s |
| four geos, four processes | 16.2 s, 16.2 s | 15.9 s, 15.8 s |

Every run certifies the same design with the same key. At five geos 71 groups survive
the screen against 136 (the optimum's own certification no longer delays the
incumbent), 657,937 are screened out against 657,872, and none is retried. On `main`
four processes barely help at five geos, since the one certification took 300 s. At
four geos the descents add about 1.3 s on one process, 32 certifications at a median
of 50 ms each (94 ms at most over 40 groups).

## More geos than periods

On simulated panels of 90 periods (`brute.geo_panel(0, n=90, p)`, centred per geo)
the search at three treated geos took 46.9 s with 60 geos and 123.6 s with 80, and at
100 geos had not finished after an hour. The floors explain it. With 90 or 100 geos
M'M has rank 89 (centring takes one), so its smallest eigenvalue is zero and
`lambda_min_lower` finds no positive bound; the slope cap needs one, so every floor
was zero (117,480 of 117,480 groups at 90 geos, 161,700 of 161,700 at 100), the
screen admitted every group, and the pass searched them one by one. The hull stage of
`certify_slope_free` reads the same cap and was skipped.

The cap needs `L <= x'x = u'Gu` for every nonzero binary `u`, not an eigenvalue.
When no entry of `G` is negative, `u'Gu` is the sum of a principal submatrix's
entries and so at least each of its diagonal entries, and `L = min_j G_jj` (derived
here). The diagonal is never below `lambda_min(G)`, so the new cap is never looser.
Every panel measured has a Gram matrix with no negative entry: GeoLift's smallest
entry is 8,391,624.9, and the simulated panels' 1,337 to 1,783. With a negative entry
the eigenvalue bound stands.

With the diagonal bound no floor is zero at 90 or 100 geos, and their quantiles are
those of the panels at 60 and 80 (`results/diag_cap.log`); at 40, 60 and 80 geos the
floors are unchanged. On four processes, one run each (`results/diag_cap.log`):

| panel | treated geos | groups | `main` | this branch | survivors | screened out |
| --- | --- | --- | --- | --- | --- | --- |
| 60 geos | 3 | 34,220 | 46.9 s | 39.6 s | 10 | 34,210 |
| 80 geos | 3 | 82,160 | 123.6 s | 113.2 s | 4 | 82,156 |
| 90 geos | 3 | 117,480 | not run | 389.9 s | 8 | 117,472 |
| 100 geos | 3 | 161,700 | over 3,600 s | 1,361.5 s | 10 | 161,690 |
| 80 geos | 4 | 1,581,580 | 1,910.6 s | 1,907.0 s | 145 | 1,581,435 |

At 60 and 80 geos the floors and the survivors are the same on both, and the times
differ from run to run. At 100 geos the start found the design the search certified
(g16, g21, g58), and 1,039 s of the 2,000 s of certification went to one group. At
100 geos and four treated geos the search on this branch ended about 50 minutes in
on a solver failure in one group's full relaxation, which a separate change reads as
an open size.

The cap also binds inside each group's certification. On the four GeoLift groups of
`results/certify_lbfgs.log` it is 7 to 11 times the incumbent's slope, against 53 to
85 with `lambda_min(G)`. It does not change the GeoLift timings. With `main` and this
branch run in turn on a quiet host (four processes), four geos took 12.0 and 11.9 s
against 12.2 and 12.2 s, five geos 30.5 and 29.5 s against 31.4 and 28.3 s, and six
geos 274.7 s against 267.2 s; `geo_panel(0)` with 60 geos at four took 206.3 s
against 204.6 s. Each pair certifies the same design with the same key, survivors
and screen.

## The floor-guided search

`guided.py` is `certify_treatment` without its certificates. It takes the library's
start, computes every treatment group's exact floor, and searches the controls of
every group the floors cannot rule out, best floor first and the start group first
of all. The answer is a design, not a proof, with a bound from the floors on how far
it can be from the optimum.

Each group's search is the study's RSS descent (`_local`, 50 restarts) and, when the
RSS descent's design fails one of the key's gates, a descent on the key itself from
that design: the library's local search with the treatment group fixed. The start
group is also descended from the library's design.

The bound: a group searched whose floor can still beat the reported design is open,
since its own optimum may be better than what the search found in it. Every other
group cannot beat the design. When no open group's correlation bound can round above
the design's class, the optimum is in that class, its RSS is at least the least floor
of the open groups, and its required impact is at least the design's impact times
the square root of that floor over the design's RSS (derived here; `guided.py`'s
docstring).

On one process (`results/guided.log`, a quiet host), against each case's certified
optimum:

| panel | treated geos | groups | searched | time | design | required impact | proven at least |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GeoLift | 3 | 9,880 | 11 | 4.6 s | the certified optimum | 1,585.8 | 1,322.4 |
| GeoLift | 4 | 91,390 | 91 | 8.0 s | the certified optimum | 1,845.8 | 1,690.8 |
| GeoLift | 5 | 658,008 | 489 | 32.6 s | the certified optimum | 2,047.3 | 1,831.4 |
| `geo_panel(0)`, 60 geos | 3 | 34,220 | 10 | 17.8 s | the certified optimum | 29.9 | 26.1 |
| `geo_panel(0)`, 80 geos | 3 | 82,160 | 4 | 48.1 s | the certified optimum | 24.5 | 23.0 |

The certified search took 27.2 s at four geos and 65.0 s at five on one process
(`results/best_start.log`). The design's impact is 6.7% to 19.9% above the bound the
floors give; the certificates are what close that.

On the same three-geo GeoLift case the library's hill climbs (`results/guided.log`)
need a larger effect to detect: the greedy alone 2,310.2 (46% above the optimum);
the greedy and the local search with 10 restarts 2,159.8, and with 50 restarts
1,593.4, the optimum's treatment group with 17 controls against its 14. Over 20
seeds of the local search none reached the optimum, at 10 restarts or at 50. At four
and five geos, over 10 seeds with 50 restarts, the closest were 3.1% and 2.2% above
the optimum and the furthest 18.3% and 20.5%.

The first version of this search read the pass of `certify_treatment` and stopped
where its certificates begin. At three geos it reported 1,602.4 against the optimum's
1,585.8, a ratio of 1.0105, for two causes. The pass leaves out the start group,
which `certify_treatment` certifies last, and the start group held the optimum; so
the group kept the library's design, whose RSS (2,977,639.8 against the optimum's
2,916,143.5) accounts for the whole ratio (the square root of the RSS ratio is
1.0105). And the pass's descent minimises RSS alone: in that group its least-RSS
design fails the Breusch-Godfrey gate, at 10 restarts and at 50, so the group
reports no passing design. Either cause alone leaves the miss. With both removed,
the key's descent from the library's design reaches the optimum from one seed in
ten and stops 0.12% above it from the others, a local optimum of the library's
moves; the RSS descent with 50 restarts, then the key's descent, reaches it in the
search above. The first version's bound also left out the start group's floor; at
four and five geos that floor was not the least, and the bound did not change.

## A certificate within a gap

The groups the floors leave are certified one by one, and on large panels one or two
of them take most of the time. They are the groups whose own optimum is close to the
incumbent, so close that a relaxation cannot separate them from it without branching
deep. A group that holds the optimum, or a near-tie, cannot be screened out by any
floor.

`certify_treatment(..., gap=g)` asks a weaker question. Every screen and every
ceiling reads a target RSS, the incumbent's RSS over (1 + g)^2, in place of the
incumbent's own. A certified search then proves that the reported design needs at
most 1 + g times the optimum's required impact: every design with a larger key is in
the reported design's correlation class, with an RSS at least the target, and the
required impact is a constant times the square root of the RSS (derived in
`treatment.py`'s docstring). A design that could round above the class is still
excluded exactly, since the ceiling's first term does not read the RSS. With `g = 0`
the search is the exact one. `test_gap.py` checks the claim against enumeration.

At 1% (`results/gap.log`, four processes, each run alone):

| panel | treated geos | groups | exact | gap 1% | survivors | slowest group | design |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GeoLift | 4 | 91,390 | 21.1 s | 14.7 s | 32, 25 | 8.1 s, 1.9 s | the certified optimum |
| GeoLift | 5 | 658,008 | 44.9 s | 38.7 s | 71, 51 | 15.1 s, 15.1 s | the certified optimum |
| `geo_panel(0)`, 60 geos | 4 | 487,635 | 241.4 s | 68.6 s | 15, 15 | 202.2 s, 26.7 s | the certified optimum |

Each pair reported the same design, the optimum. On the 60-geo panel the gap screened
out no more groups; the time came out of the one group that took 202.2 s, which closed
in 26.7 s.

### More restarts, measured and not built

A stronger start in each group shortens its certificate: on the hard 100-geo group
the RSS descent with 50 restarts stopped at an RSS above the group's ceiling, and with
500 it found the group's optimum in 1.7 s, where the branch and bound reached it by
rounding late in the search. The certificate still branched, since the near-ties
around the optimum are what it has to separate. Across a whole search the restarts
are paid in every group: on the 60-geo panel at four treated geos, `restarts=500`
took 380.1 s against 241.4 s with 50, its slowest group 184.6 s against 202.2 s, with
the same design and survivors (`results/gap.log`).

### Pentagonal cuts, measured and not built

With `s = 2u - 1`, every design is a vector of signs, and the full relaxation is a
relaxation of the cut polytope. Beside the triangle inequalities it already carries,
the next family of that polytope's valid inequalities is the pentagonal one (a
hypermetric inequality on five points, Deza and Laurent 1997, ch. 28): for any five
geos and any signs `b` on them, `sum_{a<c} b_a b_c s_a s_c >= -2`. In the lift each
is a linear row in `Z`, valid for every design, so its multiplier enters the exact
certificate like any other row. They were separated by a local search over signed
five-sets of the relaxed point (`results/pentagons.log`).

At the roots they close what triangles do not. On the hard 100-geo group, at the
optimum's own ratio with the optimum cut out, sizes 28 and 29 stayed open after 1,200
triangles; 400 pentagons closed size 28, and 738 closed size 29. Size 26 under SCS
closed with 400 after 1,200 triangles. The size-free screen at the 1% gap's target
needed no pentagons: it closed with 800 triangles.

End to end, with pentagons added from the second round of a size's relaxation, they
changed nothing at the gap the search would be run at:

| case | gap | without | with | nodes |
| --- | --- | --- | --- | --- |
| hard 100-geo group | 1% | 643.7 s | 655.7 s | 47, 47 |
| `geo_panel(0)`, 60 geos, 4 treated | 0 | 140.1 s | 138.7 s | |
| `geo_panel(0)`, 60 geos, 4 treated | 1% | 85.5 s | 85.7 s | |

On the hard group at 1% every node closed before the rounds that add them: 33 at the
first-order dual, 9 at triangles, 2 by reuse. On the 60-geo search they were added at
ten sizes, which closed in the same round with or without them. The exact certificate
on the hard group, where the roots say they would bind, ran past two hours in both
arms with four processes on four cores and was stopped, so that comparison is open.
Every pair certified the same design.

### Near-ties, not symmetry

A certificate branches where designs other than the optimum come within the
relaxation's slack of it. On the hard 100-geo group the optimum has 29 controls and
RSS 1,336.16, and of its 2,039 neighbours one move away (one control dropped, added
or swapped), none is better and 3 are within 0.1%, 9 within 0.5%, 37 within 1% and 106
within 2%; the closest is 0.0146% above it (`results/near_ties.log`).

These are not symmetric copies of the optimum. Symmetry handling in branch and bound
(orbital fixing and isomorphism pruning, as in Linderoth, Margot and Thain 2007, and
for semidefinite relaxations Hojny and Pfetsch 2022) needs a permutation of the
variables that leaves the problem unchanged; almost-symmetry detection (Knueven,
Ostrowski and Pokutta 2018) needs one that leaves it nearly unchanged. Here a swap of
controls `i` and `j` is a symmetry only if their series are equal, and none are close:

| group | control pairs | largest correlation | pairs above 0.99 | smallest swap distance |
| --- | --- | --- | --- | --- |
| hard 100-geo group | 4,560 | 0.928 | 0 | 0.381 |
| GeoLift, treated (0, 1, 2, 3) | 630 | 0.988 | 0 | 0.213 |

The swap distance is `||x_i - x_j|| / max(||x_i||, ||x_j||)`. The near-ties come from
the sum: a design's series adds about 29 controls, so one move changes it by about a
29th of the two series' difference, and different moves land at nearly the same RSS
for no structural reason. Nothing in the data gives an orbit to prune. The levers are
the ones above: a gap, so the search stops separating designs it cannot tell apart,
and a tighter relaxation, so fewer of them survive a node.

## The test beds

Every change to the search is measured first on five fixed beds of simulated
panels (`beds.py`), not on the 100-geo panel: a certificate there runs for hours,
and on panels of the same form, a few dozen units wide, the counts that decide a
change move the same way in seconds. The form matters more than the size: groups with
more candidate controls than periods branch the most.

| bed | units | treated | periods | candidate controls | groups | checked against enumeration |
| --- | --- | --- | --- | --- | --- | --- |
| `s15_k3` | 15 | 3 | 20 | 12 | 62 | yes |
| `s20_k3` | 20 | 3 | 20 | 17 | 45 | yes |
| `s20_k4` | 20 | 4 | 20 or 40 | 16 | 21 | yes |
| `s30_k10` | 30 | 10 | 20 | 20 | 13 | no |
| `s30_k4` | 30 | 4 | 20 | 26 | 25 | no |

Each bed holds the groups of a scan of random treatment groups whose certificates
branch on `main` with every node bounded (`PRICE_LIMIT = 0`): `s15_k3` 62 of 600
groups, `s20_k3` 45 of 320. A run certifies each group and counts nodes, Clarabel
and Shor solves, nodes left open at the depth limit and incumbents found by rounding.
Where a group has at most 17 candidate controls (`ENUMERATION_LIMIT`, 131,071
designs) the design is checked against the enumerated optimum, and a group certified
at any other design is reported unsound and fails the run.

```bash
python -m benchmarks.studies.tbr_bnb.beds --workers 4            # every bed, against the baseline
python -m benchmarks.studies.tbr_bnb.beds --bed s15_k3 s20_k3    # the enumerated beds
python -m benchmarks.studies.tbr_bnb.beds --record --workers 4   # on main, after a change merges
```

The baseline, `results/beds_baseline.json`, holds every group's counts on `main`,
and a run prints each total against it. A change reports its bed tables in its pull
request, and the baseline is re-recorded on `main` after it merges. Wall times are
recorded too, but they depend on the machine and its load; the counts do not.

## Progress reporting

The search reports what it is doing at four levels (`progress.py`):

| level | events |
| --- | --- |
| 1 | the treatment search and each treatment group, each gated round, the group screen, the hull, each size, each new incumbent (from a leaf, a priced subtree or rounding) |
| 2 | every node of a size's branch and bound: its depth, the controls fixed in, its outcome (leaf, priced, pruned, open or branched), the tier that decided it and the control it branched on |
| 3 | every first-order schedule, first-order triangle round, Shor or full relaxation solve, and priced subtree, each with its time |
| 4 | adds Clarabel's own iteration log, on stdout in Clarabel's format |

`certify_treatment`, `certify_gated`, `certify_slope_free` and `guided_treatment` take
`verbose` (0-4: print to stderr, one line an event, indented by level and depth),
`progress` (a function called with each `Event`) and `log` (a directory) with
`log_level` (1-3, default 3). A log is a parquet dataset with one column per declared
field; each process writes its own `part-<pid>-<run>-<n>.parquet` files, flushed every
`FLUSH_EVERY = 20000` events, every `FLUSH_SECONDS = 60` and at the end of each treatment
group, so a long run can be read as it goes, and `read_log`
returns the whole run as one pandas DataFrame. Forked workers report through the
same settings, and every group's events carry the group as the tag `treatment`.

```python
from benchmarks.studies.tbr_bnb.progress import read_log
from benchmarks.studies.tbr_bnb.treatment import certify_treatment

out = certify_treatment(M, 4, n_test=14, n_jobs=4, verbose=1, log="run_log")
df = read_log("run_log")
df[df.kind == "solve"].groupby("relaxation").seconds.describe()   # what the relaxations cost
df[df.kind == "node"].groupby(["m", "outcome", "tier"]).size()    # how each size's tree closed
```

Without `verbose`, `progress` or `log` nothing is reported, and reporting never changes
what is computed: a property test checks the certificate is the same at every level,
and that the node events add up to the search's nodes. An event of a kind or with a
field that is not declared raises, so a log's columns are fixed.

## Tests

`pytest benchmarks/studies/tbr_bnb/` (1,071 tests, about 165 s with four workers and one BLAS thread each) checks every bound
against enumeration on panels of up to eight geos, with hypothesis properties
over panels, slope pieces, sizes, fixings and scrambled multipliers, and
`certify()` against the enumerated positive-slope optimum. Coverage is 100%
but for the lines marked `pragma: no cover`, each with its reason. The 503 mutants of the 40
`tbr-bnb-*` targets in `tools/mutation/targets.toml` are killed, but for six recorded as equivalent with their reasons. The first run
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
3. Certificate cost (done here for the hull): `certificate.hull_bound_float`
   reads the hull's first-order certificate in doubles and subtracts a rigorous
   bound on every rounding error, the dot-product bound of Bjorck (2015),
   Lemma 1.4.2 and eq. (1.4.10), with an allowance for underflow derived from
   Russinoff (2022), Lemma 5.11 and Definition 6.9 (every double is a multiple
   of the smallest denormal); the last combination is done in rationals and rounded
   down. `certificate.hull_clears` decides on it when it clears the incumbent
   and on the exact certificate otherwise, so the stage keeps exactly the exact
   one's survivors. On the four groups the hull stage takes 1.2-1.5 s against
   6.5-8.2 s, with the same survivors and no pair left to the exact
   certificate (`results/float_cert.log`); the float bound sits a median 7e-14
   of `y'y` below the exact one. Hypothesis found the bound failing at
   subnormal slopes (`lo = hi = 1.1e-308`), where the relative rounding bound
   does not hold; the underflow allowance closes it.

   The joint lift's certificate is done the same way
   (`certificate.joint_bound_float`, decided by `certificate.joint_clears`).
   S and k are assembled in doubles by an oracle in the data's units
   (`oracle.Oracle(unscaled=True)`), and a second pass over the same assembly
   sums the moduli of each entry's terms (`Oracle.S_k_moduli`), which bounds
   their rounding by the same dot-product bound; the data's own rounding
   (`fl(X'X)`, `fl(X'y)`, `fl(y'y)`) is bounded beside it. `lambda_min(S)` is
   bounded below by the steps of mlsynth's `lambda_min_lower` (a Cholesky
   factor of `S - sigma I`, then Weyl's theorem and a Gershgorin row bound,
   Golub and Van Loan 2013, Cor. 8.1.6 and Thm 8.1.3), with the residual
   bounded instead of computed exactly, and Weyl's theorem once more moves it
   from `fl(S)` to `S`. Each rounding claim is checked exactly, in rationals,
   by a hypothesis property, and the whole bound against an exact upper bound
   on the dual function (the Rayleigh quotient at the computed eigenvector).
   Hypothesis found one hole: with a subnormal `lo`, `lo^2` underflows to zero
   and the coefficient has no relative bound, so there the float bound gives
   up and the exact certificate decides. The first mutation run left two
   assertions too weak: unclipped inequality multipliers passed a validity
   check they happened to satisfy, and dropping the assembly's own rounding
   term passed because random sums rarely lose more than 2u of their moduli.
   A negative multiplier is now checked to give the bound of a zero one, and a
   constant that loses all 24 of its terms to rounding is checked exactly. On
   the small panels every prune is decided by the float bound
   (`test_certify_prunes_on_the_float_certificate`). On the four GeoLift
   groups every one of the 2, 4, 13 and 25 prunes is decided by the float
   bound and none falls back; the certificates take 0.01-0.07 s against
   2.0-18.4 s, and `certify` 38, 48, 135 and 273 s against 42, 53, 138 and
   284 s, with the same nodes, prunes and designs (`results/joint_float.log`).
   What remains is the node solver: 221 s of the hardest group's 273 s.
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
   shared; the band's exact certificates are most of what remains). (b), done
   here: the capped tier is one call of `solve_simplex_qp(..., upper=...)` per
   pair (#967), through a slack weight with a zero column capped at `B - A`, which
   makes the band an equality (`hull.ControlBound`). Moving onto it found a
   defect in the solver, a cycle at a vertex of the capped simplex, whose
   vertices here are the designs; it is fixed in #970. On the four groups the
   survivors are the same as Clarabel's (`results/capped.log`), the solves alone take 9.7-12.6 s against
   10.5-14.8 s for all 1,872-1,989 pairs, and the exact certificates take about
   as long again (7-10 s), so the stage gains an exact tier and no cvxpy more than
   it gains time; the two-tier stage is unchanged (7.6-9.5 s against 8.0-9.6 s).
6. Branching rules and node order.
7. Incumbent from #964's local search, improved during the search (done here).
   Each node rounds its eigen-primal to a design of its size within its
   fixings, descends from it (`search.Incumbent.offer`, on the positive-slope
   RSS of `hull.positive_rss`), and a better design becomes the incumbent with
   its own no-good cut beside the earlier ones (`oracle.Oracle(cuts=...)`,
   `certificate.joint_bound(nogood=[...])`). See the table below.
8. The gates, handled lazily by no-good cuts (done here, on the slope-free
   search; see The gates).
9. The search over treatment groups (done here, by enumeration of the groups
   with an exact floor and a ceiling for each one; see The treatment groups).
10. Tightness on other panels and at more geos.
11. This study.
12. The build: `search="exact"` with `certify`.
13. Shared helpers and `agents/agents_combinatorial.md`.

## Files

- `panel.py`: the scoring window and the four groups.
- `hull.py`, `lifts.py`: the capped hull as one capped simplex QP, the band without the per-geo cap, and the lift levels.
- `certificate.py`: every exact bound, the slope cap, fixings and the cut, and the float certificates of the hull and the joint lift.
- `oracle.py`, `dual.py`: the certified dual and its maximisers.
- `search.py`: the branch and bound on geos and `certify()`.
- `face.py`: the face Z = T W T' of a size's relaxation, its reduced rows and its exact bound.
- `slopefree.py`: the slope-free relaxation, its certificate, the cascade of certificates for one size
  (`size_certificate`) and `certify_slope_free()`.
- `fbound.py`: the slope-free certificate in floating point with rigorous rounding bounds, and `certificate()`, which calls the exact bound only between the float bounds.
- `workspace.py`: a size's full relaxation on one SCS workspace, and `Workspaces`, one search's
  workspaces and triangle pool.
- `gated.py`: the reference key's four gates on the slope-free search, `certify_gated()`.
- `treatment.py`: the search over treatment groups, `certify_treatment()`.
- `guided.py`: the floor-guided search without certificates, `guided_treatment()`, and its bound.
- `fastfloor.py`: the floors of a chunk of groups in one compiled loop, with numba installed.
- `brute.py`: enumeration, the truth the tests check against, the simulated AR(1) panels, and
  `treatment_optimum`, every treatment and control group of a small panel scored by the library's key.
- `run.py`: `python -m benchmarks.studies.tbr_bnb.run certify` and `... lifts`.
- `beds.py`: the five test beds, `run_bed`, and `python -m benchmarks.studies.tbr_bnb.beds`; the baseline in `results/beds_baseline.json`.
- `progress.py`: the search's progress events, `reporting`, the stderr printer, the parquet log and `read_log`.
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
- Rump, S. M. (2010). Verification methods: Rigorous results using floating-point
  arithmetic. Acta Numerica 19, 287-449. Cambridge University Press.
  doi:10.1017/S096249291000005X.
- Jansson, C., Chaykin, D. and Keil, C. (2007). Rigorous error bounds for the
  optimal value in semidefinite programming. SIAM J. Numer. Anal. 46, 180-200.
- Boyd, S., El Ghaoui, L., Feron, E. and Balakrishnan, V. (1994). Linear Matrix
  Inequalities in System and Control Theory. SIAM.
- Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge.
- Lawson, C. L. and Hanson, R. J. (1974). Solving Least Squares Problems.
  Prentice-Hall.
- Dantzig, G. B. (1957). Discrete-variable extremum problems. Oper. Res. 5,
  266-288.
- Hogben, L., ed. (2007). Handbook of Linear Algebra. Chapman & Hall/CRC.
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
- Golub, G. H. and Van Loan, C. F. (2013). Matrix Computations, 4th ed. Johns
  Hopkins. Cor. 8.1.6 (Weyl) and Thm 8.1.3 (Gershgorin), the eigenvalue steps
  of the joint lift's float certificate.
- Björck, Å. (2015). Numerical Methods in Matrix Computations. Springer.
  Lemma 1.4.2 and eq. (1.4.10), the dot-product rounding bound the float
  certificate uses, under the standard model (1.4.5), which holds only if no
  underflow or overflow occurs.
- Russinoff, D. M. (2022). Formal Verification of Floating-Point Hardware
  Design, 2nd ed. Springer. Lemma 5.11, Definitions 5.19 and 6.9: denormals
  are the multiples of the smallest positive denormal, and denormal rounding
  rounds to that grid, from which the underflow allowance is derived.
- Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, 2nd ed.
  Springer. Sect. 16.5, the active-set method of `solve_simplex_qp` that the
  capped hull calls.
- Andrei, N. (2017). Continuous Nonlinear Optimization for Engineering
  Applications in GAMS Technology. Springer. Sect. 6.4 (the spectral projected
  gradient method, Algorithm 6.2, Theorem 6.5) and Sect. 6.5 (L-BFGS-B), with
  their comparison in Tables 6.2-6.5.
- Dinkelbach, W. (1967). On nonlinear fractional programming. Manag. Sci. 13,
  492-498.
- Borrero, J. S., Gillen, C. and Prokopyev, O. A. (2017). Fractional 0-1
  programming: applications and algorithms. J. Global Optim. 69, 255-282.
  Sect. 4.2, the parametric method.
- Padberg, M. (1989). The boolean quadric polytope: some characteristics,
  facets and relatives. Math. Program. 45, 139-172.
- Sherali, H. D. and Adams, W. P. (1990). A hierarchy of relaxations between
  the continuous and convex hull representations for zero-one programming
  problems. SIAM J. Discrete Math. 3, 411-430.
- Chen, H., Zheng, C. and Xia, Y. (2025). Accelerated Dinkelbach method.
  arXiv:2510.26257. Theorems 2.3, 2.4 and 3.14 assume a twice continuously
  differentiable parametric function.
- Allemand, K., Fukuda, K., Liebling, T. M. and Steiner, E. (2001). A
  polynomial case of unconstrained zero-one quadratic optimization. Math.
  Program. 91, 49-52.
- Ferrez, J.-A., Fukuda, K. and Liebling, T. M. (2005). Solving the fixed rank
  convex quadratic maximization in binary variables by a parallel zonotope
  construction algorithm. Eur. J. Oper. Res. 166, 35-50. Table 1.
- Deza, M. M. and Laurent, M. (1997). Geometry of Cuts and Metrics. Springer.
  Ch. 28, hypermetric inequalities; the pentagonal inequality is the
  five-point case.
- Linderoth, J., Margot, F. and Thain, G. (2007). Improving bounds on the
  football pool problem via symmetry reduction and high-throughput computing.
  Technical report.
- Hojny, C. and Pfetsch, M. E. (2022). Handling symmetries in mixed-integer
  semidefinite programs. Preprint.
- Knueven, B., Ostrowski, J. and Pokutta, S. (2018). Detecting almost
  symmetries of graphs. Math. Program. Comput. 10, 143-185.
- Ding, L., Lu, H. and Yang, J. (2026). On the local linear convergence of
  operator splitting methods for conic programming. arXiv:2608.16054v2.
- Díaz, M., Izquierdo Lehmann, P., Lu, H. and Yang, J. (2026). Active set
  identification and rapid convergence for degenerate primal-dual problems.
  arXiv:2602.10436v1.

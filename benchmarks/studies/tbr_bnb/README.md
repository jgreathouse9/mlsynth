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
open. On 15 markets with three treated (455 groups) both searches certify the
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
does not close, or whose solver fails, hands the size to the next.

## Tests

`pytest benchmarks/studies/tbr_bnb/` (343 tests, about 80 s with four workers) checks every bound
against enumeration on panels of up to eight geos, with hypothesis properties
over panels, slope pieces, sizes, fixings and scrambled multipliers, and
`certify()` against the enumerated positive-slope optimum. Coverage is 100%
but for the lines marked `pragma: no cover`, each with its reason. The 136 mutants of the 22
`tbr-bnb-*` targets in `tools/mutation/targets.toml` are killed, but for one recorded as equivalent with its reason. The first run
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
- `slopefree.py`: the slope-free relaxation, its certificate, the cascade of certificates for one size
  (`size_certificate`) and `certify_slope_free()`.
- `gated.py`: the reference key's four gates on the slope-free search, `certify_gated()`.
- `treatment.py`: the search over treatment groups, `certify_treatment()`.
- `brute.py`: enumeration, the truth the tests check against, the simulated AR(1) panels, and
  `treatment_optimum`, every treatment and control group of a small panel scored by the library's key.
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

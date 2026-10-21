# A node bound that is tight and cheap: persp+N

The certified search of this study had a tight node bound, the doubly
nonnegative lift, that cost 30 to 250 seconds per node, so on GeoLift it lost to
the library's enumeration at every treated count. This note records what was
read and measured to find a bound that is both, the bound that came out of it,
and what it does to the search.

## What a node bound has to contain

Twenty-seven papers on quadratics with indicator variables, cardinality and
semidefinite relaxations were read against this problem, each with a check on
synthetic panels where the claim needed one; they are listed under References
with what each contributed. The reports converge on five requirements, each
forced by a failure:

1. Second moments. The relaxation of `(z, w, v)` alone scores exactly 0: the
   point `z = (m/n) 1, w = v = (1/n) 1` is the average of designs (uniform
   weights on a treated set and on its complement, over all treated sets), and
   it scores 0 because `G 1 = 0`. No inequality in `(z, w, v)` can remove it.
   The node bound of Bienstock (1996), the one reformulation of ours that fits
   Kılınç-Karzan, Küçükyavuz and Lee's conic hulls, and the best bound the dual
   of Takapoui et al. (2020) certifies are all this relaxation, and all score 0.
   A bound has to see the products `W = w w'`, where that point is the average
   of the designs' objectives. With them the problem is a linear program over
   completely positive matrices (Burer 2009); every lift in this study relaxes
   that one.
2. Entrywise nonnegativity of `W`, the "doubly nonnegative" half of Burer's
   relaxation. Dropping it took the full lift from 99% to 29.5% of the optimum
   at GeoLift's root. Adding it to a Shor-type lift took synthetic panels from
   35-65% (the semidefinite form of Dong, Chen and Linderoth, eq. (24) of Wei,
   Gómez and Küçükyavuz 2022) to within a few points of the full lift. The
   reader of De Rosa and Khajavirad (2026) found the same from the other side:
   their bivariate hull beats the diagonal perspective only through its
   `X_12 >= 0` term, which needs no pairs.
3. The cardinality at the level of `W`: the perspective `W_jj z_j >= w_j^2` of
   Frangioni and Gentile (2006) and Günlük and Linderoth (2010), placed on the
   lifted diagonal as the 2x2 block `[[z_j, w_j], [w_j, W_jj]] >= 0` of the
   Dong-Chen-Linderoth form. With `sum z = m` it gives `sum W_jj >= 1/m`, the
   concentration of a design's weights on `m` markets. A PSD block with linear
   constraints and no perspective scores exactly 0: the averaged point of
   requirement 1 lifts to the rank-one `W = w w'`, which meets every such
   constraint, and only the perspective cuts it off (`W_jj z_j = m / n^3` is
   below `w_j^2 = 1 / n^2`).
4. A global PSD coupling. Every decomposition of `G` into small pieces with
   known indicator hulls is exactly 0 once the markets outnumber the
   predictors: the diagonal perspective and its best diagonal (Zheng, Sun and
   Li 2014), 2x2 pieces (Frangioni, Gentile and Hungerford 2020; Han, Gómez and
   Atamtürk 2023), pairwise and M-matrix hulls (Atamtürk and Gómez 2018;
   Atamtürk, Gómez and Han 2021), bivariate hulls (De Rosa and Khajavirad 2026)
   and rank-one hulls (Wei, Gómez and Küçükyavuz 2022). The reason, derived and
   checked by the readers: on `null(B)` restricted to zero-sum directions the
   shifted quadratic vanishes, so every PSD piece must vanish there, which
   forces a piece on one or two markets to zero once `n >= T + 2` or `T + 3`.
   Linear relaxations without a PSD block (McCormick and RLT products, after
   Gupte et al. 2013 and 2020, Nagarajan et al. 2018, Najman, Bongartz and
   Mitsos 2021 and Beach et al. 2024) are negative, by 26 to about 50,000 times
   the optimum on the readers' synthetic panels. Replacing the PSD block by its
   2x2 minors or by second-order cone outer approximations (Sherali and
   Fraticelli 2002; Bertsimas and Cory-Wright 2020) leaves the bound negative.
   Eigenvector cuts reach the semidefinite value only once they include about
   ten mixed weight-indicator directions with eigenvalues 1e-3 to 1e-6 of the
   largest, which only the semidefinite program's own dual supplies, and cuts
   taken from a root solve do not adapt at its children.
5. Per-node multipliers on the nonnegativity. Freezing a root solve's
   multipliers keeps the root value. That is the quadratic convex
   reformulation (QCR) of Wu, Sun, Li and Zheng (2017), whose best root bound
   is Shor's plus the perspective and has no `W >= 0`, and the convex
   quadratic relaxations of Nohra, Raghunathan and Sahinidis (2022): following
   their duality argument, one convex QP in `(w, v, z)` built from the root DNN
   dual reproduces the root bound in milliseconds, but returns exactly the root
   value at every depth-1 child. Multipliers on equalities survive freezing,
   since the equalities become exact when a market is fixed; multipliers on
   `W >= 0` do not.

The smallest object meeting all five is persp+N (`persp_n.py`): one PSD block
`[[1, x'], [x, X]]` per side, of side `|U| + 1` with `U` the markets that side
can use, `X >= 0`, `X 1 = x`, the perspective block `[[z_j, x_j], [x_j, X_jj]]`
per free market (`1 - z_j` on the control side), `x_j <= z_j`, and
`sum_F z = r`. It is the Dong-Chen-Linderoth semidefinite form (the primal of
Zheng, Sun and Li's best-diagonal program) with Burer's entrywise
nonnegativity added, written once for each of the design's two sides with a
shared `z`. The indicators never enter a PSD block, which is what makes it
cheap: two blocks of 41 rows at GeoLift's root, where the treated lift has 79
and the full lift about 116.

## The certificate

`persp_n.certified` turns the solver's multipliers into a bound no rounding can
break. It is the rigorous lower bound of Jansson, Chaykin and Keil (2007) for
semidefinite programs, the analogue of Neumaier and Shcherbina's (2004) for
linear ones, computed in exact arithmetic where they use interval arithmetic:
the Lagrangian with multipliers clipped to their cones is linear in the
lifted variables, a design has `tr M <= 2` per side and `z` in
`{0 <= z <= 1, sum z = r}`, so the bound is the constant plus
`2 min(0, lambda_min(S))` per side plus the sum of the `r` smallest
coefficients of `z`, with `S` assembled from the panel's exact Gram matrix and
`lambda_min` bounded below exactly (a floating-point Cholesky factor, its
residual computed exactly, then Weyl's and Gershgorin's theorems). On small panels it matches the solver to
about 1e-7 relative and stays under the brute-force subtree minimum where the
solver's own value does not (1.000000065 of it, once). On GeoLift's nodes it
sits 3e-8 to 6e-4 below the solver's value (the last column of
`persp_n_geolift_*.log`), more where the solver reports an inaccurate optimum,
and never by enough to change which nodes prune. `test_persp_n.py` checks
it against enumeration over random panels and nodes.

## Bound quality on GeoLift (`persp_n_geolift_*.log`)

Twelve nodes, four treated, as a fraction of each subtree's minimum; persp+N
values are the solver's, and the certified bounds are within 3e-4 of them.
"Need" is the fraction that prunes.

| node | sets | need | persp+N treated | s | persp+N both | s | treated lift | s | full lift | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| root | 91,390 | 1.000 | 0.9753 | 1.5 | 0.9850 | 3.7 | 0.9811 | 35.1 | 0.9915 | 252.2 |
| lex d1 q=0 | 9,139 | 0.743 | 0.9601 | 1.7 | 0.9689 | 3.8 | 0.9739 | 32.0 | 0.9877 | 205.7 |
| out: 1 optimal | 82,251 | 0.874 | 0.8915 | 1.7 | 0.9009 | 3.5 | 0.8774 | 26.3 | 0.9027 | 170.3 |
| in 1, out 1 optimal | 8,436 | 0.816 | 0.9887 | 1.4 | 0.9955 | 3.8 | 0.9856 | 25.8 | 0.9972 | 169.0 |

Every node the treated or full lift prunes, both persp+N variants prune (the
full table is in the logs). Root gaps at more treated:

| treated | persp+N treated | persp+N both | treated lift | full lift |
|---:|---:|---:|---:|---:|
| 6 | 15.7% (1.8 s) | 12.2% (3.3 s) | 13.0% (31 s) | 8.6% (198 s) |
| 8 | 23.4% (1.4 s) | 12.1% (3.5 s) | 22.0% (34 s) | 9.2% (210 s) |

On the nodes of more than five thousand sets, where a bound has the most to
save, the two-sided bound is 6 to 10 times faster than the treated lift and 44
to 68 times faster than the full lift. It is stronger than the treated
lift at eight of the twelve nodes and at both larger roots, equal at one node,
and weaker at three (by 0.5, 0.6 and 2.5 points, at nodes that all four bounds
prune). It is within 4.2 points of the full lift everywhere, and within one
point at four of the five largest nodes.

## The search (`hybrid_persp_n_*.log`, `exact_reference_*.log`)

`hybrid.search(..., bounder="persp_n")` bounds a node by the certified two-sided
persp+N when pricing its subtree would cost more than the bound
(`persp_n_cost`, fitted to the twelve nodes), and enumerates it otherwise.

| panel | treated | sets | persp+N search | bounds | sets enumerated | library search | earlier certified search |
|---|---:|---:|---:|---:|---:|---:|---:|
| GeoLift | 4 | 91,390 | 0.1 s | 0 | 91,390 | 0.30 s | 0.1 s |
| GeoLift | 6 | 3,838,380 | 8.7 s | 2 | 575,757 | 3.94 s | 6.9 s |
| GeoLift | 8 | 76,904,685 | 39.9 s | 11 | 812,889 | 105.1 s | 205.5 s |
| GeoLift | 10 | 847,660,528 | 33.4 s | 10 | 324,632 | 1,866.7 s | not run |
| Walmart | 6 | 8,145,060 | 13.6 s | 2 | 1,086,008 | 7.5 s | not run |
| Walmart | 8 | 215,553,195 | 165.0 s | 34 | 6,224,400 | 305.2 s | not run |

Every persp+N result is certified (gaps 1e-13 to 2e-12, nothing unresolved).
The library's search is not certified and refuses more than twenty million
sets; its column was run with the limit raised (`exact_reference.py`). The two
searches return the same optimum, value and set, in every row.
Optima: GeoLift 8 treated 181,943.3763 at (0, 2, 10, 12, 18, 20, 24, 27);
10 treated 127,404.7767 at (1, 21, 22, 23, 28, 29, 30, 35, 37, 39); Walmart 6
treated 0.03101918869 at (1, 2, 11, 26, 31, 39); 8 treated 0.02248592328 at
(1, 2, 11, 13, 26, 31, 33, 39) (column positions in the panel).

The branch and bound now wins where the subtree counts are large. At eight
treated on GeoLift it is 2.6 times faster than the uncertified enumeration and
about five times faster than certified enumeration. At ten treated, where
enumeration takes 31 minutes with its limit raised and the library refuses
without it, it takes half a minute, 56 times faster. On Walmart it is 1.8 times
faster at eight treated. At six treated, on both panels, a bound costs more
than enumerating the subtrees it would save, and enumeration is about twice as
fast.

## The penalized design (`penalized_relaxation_geolift.log`)

Abadie and Zhao's (2026) penalized design (online appendix, section OA.1)
adapts the penalty of Abadie and L'Hour (2021): it charges each unit's weight
its squared distance to the target. That charge is exactly what the
spread-out points of requirement 1 cannot pay, so the plain continuous
relaxation, one convex QP with no lift, tightens with the penalty. GeoLift, four
treated:

| lambda | relaxation / optimum | markets per side | relaxation | library search |
|---:|---:|---:|---:|---:|
| 0 | 0.0% | 40 | 0.02 s | 0.08 s |
| 0.01 | 58.9% | 14 | 0.02 s | 10.0 s |
| 0.1 | 79.7% | 9 | 0.02 s | 49.4 s |
| 1 | 84.4% | 4 | 0.02 s | 285.7 s |
| 10 | 95.2% | 3 | 0.02 s | 306.1 s |
| 100 | 99.4% | 2 | 0.02 s | 295.4 s |

What separates the relaxation from the optimum at large lambda is mostly a
market carrying weight on both sides (`w_j + v_j <= 1` in the relaxation, where
a design needs `w_j v_j = 0`), which branching on those few markets settles.
The library's search slows by three orders of magnitude under the penalty
because its bound sees neither penalty side; #777 and its PR #780 fix that.

## Open

- Hundreds to thousands of markets. persp+N has about `n^2` variables per node.
  It took 4 s at GeoLift's 40 markets here, and 150-186 s at 100 synthetic
  markets in a reader's run on a loaded machine; at 2,000 markets no
  interior-point method solves it per node. Certifying that no `m` markets
  nearly cancel is a sparse null-space property, and computing the constants of
  the null-space and restricted-isometry properties is NP-hard in general
  (Tillmann and Pfetsch 2014). At that scale the route is structure: small `m`
  by certified enumeration (three treated among 2,000 markets is about 1.3e9
  sets), clusters or screening to bring each search to tens or low hundreds of
  markets, and the penalized design's QP relaxation.
- The control side at large n. When every halfspace through the target holds
  more than `m` markets (the target's halfspace depth, Tukey 1975, exceeds
  `m`), the control fit is zero for every treated set; on synthetic panels with
  100 predictors it was zero for every tested set at 500 and 2,000 markets and
  nonzero at 40. It can be certified by `m + 1` disjoint groups of markets each
  surrounding the target. Not yet tested on a real panel.
- A persp+N search for the penalized design, branching on the markets the
  relaxation shares between the sides.

## References

Read for this study. The first twenty-seven are the papers behind the five
requirements; each line says what the reader found against this problem.

- Atamtürk, A., and Gómez, A. (2018). "Strong formulations for quadratic
  optimization with M-matrices and indicator variables." Mathematical
  Programming 170:141-176. Pairwise hulls for M-matrices; zero at the averaged
  point, and `G` falls outside the one split it uses (requirement 4).
- Atamtürk, A., and Gómez, A. (2020). "Submodularity in conic quadratic mixed
  0-1 optimization." Operations Research 68(2):609-630. Needs the binaries
  inside a concave function; ours enter only through `w_j <= z_j`.
- Atamtürk, A., Gómez, A., and Han, S. (2021). "Sparse and smooth signal
  estimation: convexification of L0-formulations." Journal of Machine Learning
  Research 22(52):1-43. Pairwise-difference hulls; 14-19% of the optimum on
  synthetic panels (requirement 4).
- Atamtürk, A., and Narayanan, V. (2010). "Conic mixed-integer rounding cuts."
  Mathematical Programming 122:1-20. Redundant here: every cone row has a zero
  constant.
- Beach, B., Burlacu, R., Bärmann, A., Hager, L., and Hildebrand, R. (2024).
  "Enhancements of discretization approaches for non-convex mixed-integer
  quadratically constrained quadratic programming: Part I." Computational
  Optimization and Applications 87:835-891. Discretizes continuous products;
  its relaxation is negative here (requirement 4).
- Bertsimas, D., and Cory-Wright, R. (2020). "On polyhedral and second-order
  cone decompositions of semidefinite optimization problems." Operations
  Research Letters 48:78-85. Outer approximations of the PSD block stay
  negative (requirement 4).
- Bertsimas, D., Cory-Wright, R., and Pauphilet, J. (2022). "Mixed-projection
  conic optimization: a new paradigm for modeling rank constraints."
  Operations Research 70(6):3321-3344 (read as arXiv 2009.10395v2). Its
  sparsity relaxation is the ridge perspective, no stronger than the diagonal.
- Bienstock, D. (1996). "Computational study of a family of mixed-integer
  quadratic programming problems." Mathematical Programming 74:121-140. Its
  node bound is the natural relaxation, 0 here (requirement 1).
- De Rosa, A., and Khajavirad, A. (2026). "Explicit convex hull description of
  bivariate quadratic sets with indicator variables." Mathematical Programming
  218:169-211. Pair hulls add 0.2-3 points over persp+N at 25-40 times the
  cost; their nonnegativity term carries what helps (requirements 2 and 4).
- Del Pia, A. (2024). "Convex quadratic sets and the complexity of mixed
  integer convex quadratic programming." arXiv 2311.00099v2. Complexity theory;
  fixed-parameter tractable in the number of integer variables, which is `n`.
- Frangioni, A., Gentile, C., and Hungerford, J. (2020). "Decompositions of
  semidefinite matrices and the perspective reformulation of nonseparable
  quadratic programs." Mathematics of Operations Research 45(1):15-33. 2x2
  decompositions; exactly 0 once `n >= T + 3` (requirement 4).
- Günlük, O., and Linderoth, J. (2010). "Perspective reformulations of mixed
  integer nonlinear programs with indicator variables." Mathematical
  Programming 124:183-205 (read as the 2009 technical report). The perspective
  hull of the separable piece (requirement 3); its only tool for a
  non-separable quadratic is a diagonal split.
- Gupte, A., Ahmed, S., Cheon, M. S., and Dey, S. (2013). "Solving mixed
  integer bilinear problems using MILP formulations." SIAM Journal on
  Optimization 23(2):721-744. With binaries it reduces to McCormick
  (requirement 4).
- Gupte, A., Kalinowski, T., Rigterink, F., and Waterer, H. (2020). "Extended
  formulations for convex hulls of some bilinear functions." Discrete
  Optimization 36:100569. With our sign pattern the hull is McCormick
  (requirement 4).
- Han, S., Gómez, A., and Atamtürk, A. (2023). "2x2-convexifications for
  convex quadratic optimization with indicator variables." Mathematical
  Programming 202:95-134. Its pairwise relaxation is 0 as written and, with
  the simplex shift, about as strong as the perspective with `W >= 0`
  (requirement 4).
- Kılınç-Karzan, F., Küçükyavuz, S., and Lee, D. "Conic mixed-binary sets:
  convex hull characterizations and applications." Preprint, read from
  Optimization Online; published with Shafieezadeh-Abadeh, S., in Operations
  Research 73(1):251-269 (2025). The one reformulation that fits it is the
  natural relaxation (requirement 1).
- Nagarajan, H., Sundar, K., Hijazi, H., and Bent, R. (2018). "Convex hull
  formulations for mixed-integer multilinear functions." arXiv 1807.11007v1.
  McCormick scaled by binaries; no better than the RLT bound (requirement 4).
- Najman, J., Bongartz, D., and Mitsos, A. (2021). "Linearization of McCormick
  relaxations and hybridization with the auxiliary variable method." Journal
  of Global Optimization 80:731-756. Zero or negative in every factorable form
  tried (requirement 4).
- Nohra, C. J., Raghunathan, A. U., and Sahinidis, N. V. (2022). "SDP-quality
  bounds via convex quadratic relaxations for global optimization of
  mixed-integer quadratic programs." Mathematical Programming 196:203-233.
  Its duality argument gives a convex QP that reproduces the root DNN bound and
  barely moves when frozen (requirement 5).
- Park, J., and Boyd, S. (2015). "Concave quadratic cuts for mixed-integer
  quadratic problems." arXiv 1510.06421v2. Triangle inequalities on `z z'`
  close the DNN gap at `m = 4` on small panels, but only through the full PSD
  block.
- Sherali, H. D., and Fraticelli, B. M. P. (2002). "Enhancing RLT relaxations
  via a new class of semidefinite cuts." Journal of Global Optimization
  22:233-261. RLT without the PSD block stays negative, with or without its
  2x2 minors; eigenvector cuts recover the bound only when taken from the
  solved semidefinite program (requirement 4).
- Takapoui, R., Moehle, N., Boyd, S., and Bemporad, A. (2020). "A simple
  effective heuristic for embedded mixed-integer quadratic programming."
  International Journal of Control 93(1):2-12. Cheap incumbents at 2,000
  markets, no better than single swaps; no bound (requirement 1).
- Wei, L., Gómez, A., and Küçükyavuz, S. (2022). "Ideal formulations for
  constrained convex optimization problems with indicator variables."
  Mathematical Programming 192:57-88. Its eq. (24) is the Dong-Chen-Linderoth
  semidefinite form persp+N builds on; its strengthening is empty under
  `sum z = m` (requirements 2 to 4).
- Wu, B., Sun, X., Li, D., and Zheng, X. (2015). "Tight MIQP reformulations
  for semi-continuous quadratic programming: lift-and-convexification
  approach." arXiv 1507.05708v1. A root reformulation reused at every node; its
  root equals the best diagonal perspective (requirement 5).
- Wu, B., Sun, X., Li, D., and Zheng, X. (2017). "Quadratic convex
  reformulations for semicontinuous quadratic programming." SIAM Journal on
  Optimization 27(3):1531-1553 (read as the final preprint). Its best root
  bound is Shor's with the perspective and no `W >= 0`, and it is exactly 0
  once `n >= T + 3` (requirement 5).
- Yu, Q., and Küçükyavuz, S. (2025). "Convexification of classes of
  mixed-integer sets with L-natural convexity." arXiv 2511.19754v2. Needs
  submodularity, which `g` lacks.
- Zheng, X., Sun, X., and Li, D. (2014). "Improving the performance of MIQP
  solvers for quadratic programs with cardinality and minimum threshold
  constraints: a semidefinite program approach." INFORMS Journal on Computing
  26(4):690-703. The best diagonal for the perspective; its primal is Shor
  with `W 1 = w` and the perspective, the form persp+N adds `W >= 0` to
  (requirements 3 and 4).

Read earlier in the study, for the lifts and the certificates.

- Abadie, A., and Zhao, J. (2026). "Synthetic Controls for Experimental
  Design." Working paper, and its online appendix. The design, and the
  penalized design of section OA.1.
- Burer, S. (2009). "On the copositive representation of binary and continuous
  nonconvex quadratic programs." Mathematical Programming 120:479-495. The
  completely positive form and its doubly nonnegative relaxation
  (requirements 1 and 2).
- Jansson, C., Chaykin, D., and Keil, C. (2007). "Rigorous error bounds for
  the optimal value in semidefinite programming." SIAM Journal on Numerical
  Analysis 46(1):180-200 (read as the preprint by Jansson and Keil). The
  certificate of `persp_n.certified`.
- Kim, S., Kojima, M., and Toh, K.-C. (2016). "A Lagrangian-DNN relaxation: a
  fast method for computing tight lower bounds for a class of quadratic
  optimization problems." Mathematical Programming 156:161-187. Assessed as a
  faster way to solve the doubly nonnegative lift.
- Neumaier, A., and Shcherbina, O. (2004). "Safe bounds in linear and
  mixed-integer linear programming." Mathematical Programming 99:283-296. A
  valid bound from any dual vector, with the rounding controlled.
- Sun, X., Zheng, X., and Li, D. (2013). "Recent advances in mathematical
  programming with semi-continuous variables and cardinality constraint."
  Journal of the Operations Research Society of China 1:55-77. The survey
  behind `perspective.py`.
- Wiegele, A., and Zhao, S. (2024). "Tight SDP relaxations for
  cardinality-constrained problems." arXiv 2107.11338v2. The indicator lift of
  `certified_dnn.py`.
- Wolfe, P. (1976). "Finding the nearest point in a polytope." Mathematical
  Programming 11:128-149. The dual of one fit, used by every certificate here.

Cited and not read here.

- Abadie, A., and L'Hour, J. (2021). "A penalized synthetic control estimator
  for disaggregated data." Journal of the American Statistical Association
  116(536):1817-1834.
- Dong, H., Chen, K., and Linderoth, J. "Regularization vs. relaxation: a
  conic optimization perspective of statistical variable selection." arXiv
  1510.06083, as given in Wei, Gómez and Küçükyavuz (2022).
- Frangioni, A., and Gentile, C. (2006). "Perspective cuts for a class of
  convex 0-1 mixed integer programs." Mathematical Programming 106:225-236.
- Tillmann, A. M., and Pfetsch, M. E. (2014). "The computational complexity of
  the restricted isometry property, the nullspace property, and related
  concepts in compressed sensing." IEEE Transactions on Information Theory
  60(2):1248-1259.
- Tukey, J. W. (1975). "Mathematics and the picturing of data." Proceedings of
  the International Congress of Mathematicians (Vancouver, 1974), vol. 2,
  523-531.

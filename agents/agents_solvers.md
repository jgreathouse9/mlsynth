# agents_solvers.md

# Standards for Optimization Solvers in `mlsynth`

This file states what a numerical solver in `mlsynth` must do, whether it is
written here (`utils/solvers/`, `utils/bilevel/`, a study's LP code) or called
from a package (`cvxpy`, `scipy.optimize`). It exists so that "this tolerance is
fine" becomes a claim with a citation and a test, not a matter of taste.

Eight books supply the rules:

- Nocedal and Wright, *Numerical Optimization*, 2nd ed. (Springer, 2006), cited
  as NW with page numbers. Appendix A (floating point, conditioning, stability),
  Chapter 10 (least squares), Chapters 13-14 (simplex and interior-point
  numerics), Chapters 3 and 11 (safeguarded steps and stopping tests).
- Russinoff, *Formal Verification of Floating-Point Hardware Design*, 2nd ed.
  (Springer, 2022), cited as R with lemma numbers. What one IEEE rounding can
  and cannot do: exactness (Sterbenz, Corollary 4.16), the error of a single
  rounding (Lemma 6.45), gradual underflow (Lemma 6.101).
- Björck, *Numerical Methods in Matrix Computations* (Springer, 2015), cited as
  B with page numbers. Section 2.2.2 (the least squares condition number,
  Theorem 2.2.6), Section 2.7.3 (least squares with linear inequality
  constraints, the least distance reduction, rank-deficient solutions),
  Sections 1.4.6 and 2.3.8 (iterative refinement). Also Section 2.2.1 (the
  pseudoinverse solution and the effective condition number, pp. 229-232),
  Section 2.2.3 (Eckart-Young-Mirsky, singular value interlacing, pp. 237-239),
  Section 2.2.5 (principal angles computed from sines, pp. 243-244), Section
  2.4.1 (numerical rank at the data's error level, pp. 291-292), Sections 2.4.2
  and 2.3.8 (pivoted QR does not reveal rank; inverse iteration does,
  pp. 285-286, 296), Section 2.4.6 (updating, and which direction loses rank,
  pp. 305-311), Section 2.4.7 (stepwise regression, pp. 311-314), Section 2.6.1
  (filter factors, GCV, L-curve, pp. 345-348), Section 2.7.2 (equality
  constraints by elimination, pp. 368-369) and Section 4.5.6 (semiconvergence,
  pp. 733-734).
- Björck, *Numerical Methods for Least Squares Problems* (SIAM, 1996), cited as
  B96 with page numbers. The earlier and narrower book: Section 1.4.6 (the
  backward error of an approximate least squares solution, pp. 34-36), Sections
  1.4.3 and 2.2.3 (column scaling and reparametrisation, pp. 31, 49-51),
  Section 2.9.3 (iterative refinement in working precision, pp. 124-125),
  Section 5.1.5 (equality constraints imposed by a weight, pp. 192-194) and
  Section 5.2 (active set methods for inequality-constrained least squares,
  pp. 196-203). Also Section 1.4.2 (the pseudoinverse is continuous only where
  the rank is, pp. 26-27), Section 2.7 (numerical rank, Tikhonov as a filter,
  the representative of a rank-deficient problem, rank-revealing QR,
  pp. 99-110), Sections 3.1-3.3 (modifying and downdating QR and Cholesky,
  pp. 127-143), Section 4.6 (total least squares, pp. 176-180), Section 5.1.1
  (uniqueness under equality constraints, p. 187), Section 5.3 (quadratic
  constraints and GCV, pp. 203-213) and Section 7.7.2 (regularisation by
  stopping an iteration, pp. 314-316).
- Golub and Van Loan, *Matrix Computations*, 4th ed. (Johns Hopkins, 2013),
  cited as GvL with page numbers. Section 3.5.3 (mixed-precision iterative
  improvement, Heuristic III, p. 140). Also Sections 5.4.3-5.4.4 (the Kahan
  matrix; pivoted QR as a heuristic rank indicator, pp. 279-280), Section 5.5
  (the rank-deficient problem, pp. 288-294), Section 6.1.3 (column weighting
  moves the rank estimate, pp. 306-307), Section 6.1.4 (ridge and its
  leave-one-out identity, pp. 307-308), Section 6.2.3 (equality constraints by
  a change of variables, pp. 315-316), Section 6.4.3 (principal angles,
  p. 331) and Section 6.5 (updating, and hyperbolic downdating, pp. 334-345).
- Yanai, Takeuchi and Takane, *Projection Matrices, Generalized Inverse
  Matrices, and Singular Value Decomposition* (Springer, 2011), cited as YTT
  with page numbers. Projectors and their inclusion (Theorems 2.11, 2.24, 2.25,
  pp. 36-50), the g-inverse families and the minimiser set (Theorem 3.19 and
  (3.89), p. 80), equality constraints by a null-space basis (Theorem 4.10,
  p. 101), the Tikhonov inverse (4.123), p. 120, and the SVD forms of the
  g-inverses (pp. 138-148).
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd
  ed. (Springer, 2009), cited as ESL with page numbers. Ridge, PCR and the lasso
  as filters (pp. 64-80), forward selection and its inference (pp. 57-60, 78-79),
  and cross-validation of a multistep procedure (Sections 7.10-7.12,
  pp. 241-255).
- Bueno Cachadiña and Pérez Álvaro, *Matrix Methods in Data Analysis*
  (Springer, Texts in Applied Mathematics 80, 2025), cited as BP with page
  numbers. Pivoted QR and the rank-deficient solution set (pp. 407-410,
  481-483), regularised and constrained least squares (pp. 489-491), QR
  updating (pp. 424-428), the pseudoinverse as a ridge limit (pp. 842-843) and
  rank from Gram eigenvalues (pp. 843-844).

Two papers supply what the books leave to low-rank iterations:

- Hastie, Mazumder, Lee and Zadeh (2015). Matrix completion and low-rank SVD
  via fast alternating least squares. *Journal of Machine Learning Research*
  16, 3367-3402. Cited as HMLZ with page and equation numbers: the
  rank-restricted soft-thresholded SVD (Theorem 1, p. 3370), when a factored
  problem equals the nuclear-norm problem (pp. 3369, 3383, 3385), the
  stationarity condition (47), p. 3384, and the stopping measure (Theorem 4,
  p. 3381).
- Wang, Zhang, Li and Zhang (2025). High-dimensional low-rank matrix regression
  with unknown latent structures. arXiv:2510.22106v1. Cited as WZLZ: an
  estimator defined as the T-th iterate of scaled gradient descent, with local
  convergence only (Algorithm 1 and Theorem 3.3, pp. 9-10), and a rank ratio
  with an additive constant that is not scale invariant ((3.2), p. 12).

A paper supplies the rules for proofs computed in floating point (17 to 27):

- Rump, S. M. (2010). Verification methods: Rigorous results using
  floating-point arithmetic. *Acta Numerica* 19, 287-449. Cambridge University
  Press. doi:10.1017/S096249291000005X. Cited as Rump with page, equation,
  theorem and algorithm numbers. Part 1 (Sections 1 to 9: the principles of
  verification methods, the floating-point model with underflow, error-free
  transformations, directed rounding, interval operations and their
  overestimation), Section 10 (linear systems: the failure of interval Gaussian
  elimination, preconditioning, Krawczyk's operator with epsilon-inflation,
  inner inclusions, positive definiteness by one Cholesky factorization, the
  spectral norm), Section 13 to 13.1 (nonlinear systems, existence and
  exclusion), Section 13.4 (eigenvalue clusters) and Section 14 (linear,
  convex and semidefinite programming). Sections 11, 12, 13.2, 13.3 and Part 3
  (automatic differentiation, quadrature, multiple roots, differential
  equations) treat problems no `mlsynth` solver poses, and bind nothing here.

Rules 1 to 10 were extracted while auditing the RRMA study's hand-written LP solvers
and distribution functions (`benchmarks/studies/rrma_ablation/`, commit
`ef9da44`). Every rule below comes with the failure it caused there, measured,
and the test that now pins it. The books give the principle; the numbers say
what breaking it cost here.

The RRMA study was not merged: its code, the tests named under rules 1 to 10
and the `rrma-study-*-numerics` mutation targets live on the branch
`claude/vigilant-euler-cszefb` at `ef9da44`. The rules bind on `main`
regardless; that branch is where their measurements can be rerun.

The thesis in one line:

> A solver's answer must not depend on the units of its data, and a solver must
> say what it found: certified optimum, cap reached, infeasible, unbounded, or
> not identified. Everything else in this file is how to make those two
> statements true in floating point.

And where a statement is a proof, not an estimate (a pruned node, a certified
optimum, a positive definite matrix), Rump's design principle governs it:

> "Mathematical theorems are formulated whose assumptions are verified with the
> aid of a computer" (Rump (1.1), p. 292).

A certificate in `mlsynth` is such a theorem: weak duality with a trace bound,
Weyl's inequality, a Cholesky factorization that completes. Rules 17 to 27 say
how to state it, how to verify its assumptions in floating point, and what a
failed verification means.

---

# The rules

Status, as in `agents_unix.md`: binding means new code must follow it and
existing code that breaks it is backlog (counted at the end); adopted means this
file makes it a rule where it was not one.

Numbers 11 to 16 belong to the testing refinements under "Testing a solver: the
floor of rounding", which other files cite by number, so rules added after the
verification block continue at 28.

| # | Rule | Status | Book basis |
| --- | --- | --- | --- |
| 1 | Every outcome is explicit | binding (invariant 7) | NW p. 57, p. 370, p. 607 |
| 2 | Optimality is certified on the original data | adopted | NW p. 370, pp. 381-382, p. 386; B96 pp. 34-36 |
| 3 | Tolerances are relative to what they compare, with no floor | adopted | NW pp. 26-27, p. 131; GvL p. 331 |
| 4 | A computed zero is a zero only to the rounding of its terms | adopted | R Lemma 6.45, Lemma 6.101 |
| 5 | Do not difference what you can avoid differencing | adopted | NW p. 615; B96 pp. 50-51; B p. 244 |
| 6 | Do not square the condition number; name rank deficiency | adopted | NW pp. 251-253, p. 610, p. 617; B p. 235, pp. 291-292, p. 296, pp. 371-373; B96 pp. 26-27, p. 49, pp. 99-107, pp. 192-203; GvL pp. 279, 288-292, 307; YTT p. 80; BP pp. 482-483, 843-844 |
| 7 | Iterations are safeguarded and end | adopted | NW p. 37, p. 61, p. 62; HMLZ p. 3381, p. 3384 |
| 8 | Repairs to a solver's output are reported | binding (invariant 7) | none; invariant 7 |
| 9 | Form quantities so the failure mode is underflow | adopted | NW p. 616 |
| 10 | Degeneracy: perturb to pivot, evaluate and certify on the original | adopted | NW pp. 381-382 |
| 17 | A certificate is a theorem whose assumptions the computer verifies | adopted | Rump (1.1), pp. 292-293, Theorem 1.1, p. 299 |
| 18 | Verify only well-posed claims; equality and singularity are decided exactly | adopted | Rump (1.2), p. 293, p. 300, p. 309 |
| 19 | Bound every rounding: the floating-point model with underflow | adopted | Rump (2.3), (2.4), Theorem 2.1, (2.8)-(2.10), pp. 302-304 |
| 20 | Utilize input data: verify from the data, not from computed intermediates | adopted | Rump (5.13), p. 312; Sections 6, 10.1, pp. 315-337 |
| 21 | Move a bound from the computed matrix to the exact one by perturbation theory | adopted | Rump Lemma 10.12, (10.54), (10.61), pp. 357-359 |
| 22 | Positive definiteness: one factorization that completes, or a rigorous residual | adopted | Rump Lemma 10.13, (10.56)-(10.60), Table 10.10, pp. 357-358 |
| 23 | A bound on a linear solve comes from a preconditioned residual, never from the elimination | adopted | Rump Theorems 10.2, 10.6, 10.8, Algorithm 10.7, (10.14), pp. 338-345 |
| 24 | Range bounds: each variable once, and the inclusion of an image is not an inclusion of a preimage | adopted | Rump (5.16), (5.17), Theorem 8.1, (9.4)-(9.8), pp. 314-327 |
| 25 | Bounds in matrix form, by products of magnitudes | adopted | Rump (9.10)-(9.14), Algorithm 9.1, Tables 9.1-9.3, pp. 327-331 |
| 26 | A relaxation's bound is postprocessed from any multipliers, never read from the solver | binding (rule 2) | Rump Section 14, Theorem 14.2, pp. 392-397 |
| 27 | A directed bound is rounded in its own direction, and says which | adopted | Rump (4.1)-(4.3), Theorem 4.2, pp. 308-310 |
| 28 | A rewrite is the solver's only if it keeps the set of minimisers | adopted | B96 p. 101, p. 187; BP pp. 489-491; YTT p. 80, p. 120; B p. 369; ESL p. 66 |
| 29 | Regularisation is a filter, and its parameter is part of the estimand | adopted | B pp. 345-348, pp. 733-734; B96 pp. 101-102, pp. 211-213, pp. 314-316; ESL pp. 66-80; BP pp. 842-843 |
| 30 | Remove a row with Q in hand, or refit; never downdate R alone | adopted | B96 pp. 129-143; B pp. 305-311; GvL pp. 335-340 |
| 31 | Cross-validation contains every step that reads the outcome | adopted | ESL pp. 60, 231, 242, 245-248 |

## 1. Every outcome is explicit

A solver ends in one of a few states: optimal (and certified, rule 2), iteration
cap reached, infeasible, unbounded, numerically singular, or not identified (a
face of minimisers, rule 6). It returns or reports which. NW's procedures name
their exits: a line search "either terminates ... or determines that such a
step length does not exist" (p. 57); the simplex step stops at optimal or
unbounded (Procedure 13.1, p. 370); LU stops with the matrix singular (Algorithm
A.1, p. 607).

The report is a typed field on the result or a `warnings.warn` with its own
category (`CLAUDE.md` invariant 7). A module-level counter is neither: nobody
reads it, and under `multiprocessing` each worker has its own.

What it looked like. The RRMA solvers bumped `UNCERTIFIED[0]` when a certificate
failed and returned the point anyway; `one_rep` silenced every warning. The
study's tables could not say whether any number came from a failed solve. Now
`CertificateWarning` and `UnidentifiedFitWarning` carry a `problems` count, and
`simulate.counting` records both per replication.

Test pattern: force the failure (monkeypatch the certificate to fail) and assert
the warning arrives with the right count, and that a clean solve is silent with
warnings as errors (`test_rrma_certificate_reporting.py`).

## 2. Optimality is certified on the original data

After a solve, check a certificate computed from the returned point on the data
the caller passed, not on whatever the iteration worked with:

- for an LP, primal feasibility, dual feasibility, stationarity and a zero
  duality gap (weak duality turns these into a proof; NW p. 386 for free
  variables, whose dual constraint is an equality);
- for a QP, the KKT residual, scaled as in rule 3 (`utils/weights/solve.py` and
  `utils/solvers/minnorm.py` do this).

The certificate re-checks every condition itself. A pivot loop that stops when
its last test passes has checked the vertex it believed it was at; the
certificate checks the one it returns.

A small stationarity residual is evidence, not a definition. For least squares,
"for x to have a small backward error it is sufficient that either (b - Ax) or
A^T(b - Ax) be small, but neither of these conditions is necessary" (B96 p. 36).
On a nearly consistent problem (an exact fit, a treated unit inside the donor
hull) the computed residual is rounding noise and is not orthogonal to the
range of `A`, so `A'r` measured against `||A|| ||r||` alone refuses a sound
answer. Scale the stationarity residual by the size of the terms it is computed
from, `s_1 (s_1 ||x|| + ||b||)` with `s_1` the largest singular value, as
`lsq.kkt_backward_error` does; that reads as a backward error and stays bounded
when the residual vanishes. The exact normwise backward error (Waldén, Karlsson
and Sun, B96 Theorem 1.4.8, p. 35) exists in closed form, but the book calls the
formula "unsuitable for computation since [it] can suffer from cancellation"
(p. 35); mlsynth's certificates use the scaled KKT residual and say so.

What it looked like. The pair-free certificate checked stationarity and the gap
but took the dual's feasibility from the pivot loop. A `phi` built to meet both
at a vertex 0.1 percent from the optimum passed. It now applies Hoffman's cut
test to `phi` on all the periods.

Test pattern: hand the certificate a worse point and assert it refuses. A
certificate that accepts everything is green forever, so a test that only checks
"the optimum is certified" proves nothing (`TestTheCertificateAtEveryScale`,
`TestTheGapTestAtEveryScale`). To test one condition, hold the others: keep the
optimal dual fixed and move the primal point to test the gap, or build a dual
that meets every condition but one. A stubbed pivot loop that returns a vertex
that is not optimal tests the reporting path, not the gap: that vertex fails on
its multipliers or its cuts first. Keeping the certificate in its own function
(`lp._rank_certified`, `lp._lad_certified`) is what makes those tests possible.

## 3. Tolerances are relative to what they compare, with no floor

A tolerance is `TOL` times the size of the quantity it compares, in that
quantity's units. The size is either the magnitude of the terms the quantity is
computed from (for `a'x`, `|a|'|x|`), or the problem's own scale (for a cut
violation in a pairwise programme, `h(G)^2 / 4`). Two rates in different units
are compared each against its own scale, never against one shared tolerance.

`max(1, |x|)` and `(1 + |x|)` are floors: below 1 they make the test absolute,
and then the units of the data decide the answer. NW: "Algorithms that are not
sensitive to scaling are preferable" (pp. 26-27). NW's own `(1 + |f|)` test on
p. 131 is the stopping rule of one experiment on problems scaled to order 1,
not a recommendation.

What it looked like, measured before the fix:

- data at a scale of 1e-7: the simplex-weight solves stopped 0.3 to 0.6 percent
  from the optimum;
- weights scaled by 1e-6: `simplex_lad` stopped 19 percent from it (a floor of
  1 under the tolerance on rates of size 1e-6, one tolerance for rates in units
  of `c` and of `c y`);
- at a scale of 1e-6 the certificate accepted a point 3e-6 from the optimum.

Test pattern: a `hypothesis` property per solver that multiplies the outcome,
the design and the weights by `10^k` (k from -8 to 8) and asserts the same
answer, certified (`TestUnits` in `test_rrma_lp.py` and
`test_rrma_pairfree_solver.py`).

Place a tolerance on the quantity that is well conditioned at the boundary it
tests. Two subspaces are nested when the sines of their principal angles are
zero; a computed cosine `sigma >= 1 - delta` (GvL's "intelligently chosen small
parameter delta", p. 331) bounds `sin^2` by `2 delta` only, so a cosine test at
`delta = 1e-9` cannot see an angle below `sqrt(2e-9)`, about `4.5e-5` (derived).
State subspace tolerances on `sin^2`, relative to the dimension the statistic
sums over.

## 4. A computed zero is a zero only to the rounding of its terms

One rounding errs by at most half a unit in the last place of its result (R
Lemma 6.45), so a residual computed from operands of size `m` is known to about
`m eps`. Testing `e == 0.0`, `x != 0` or `t == 0.0` on a computed value
classifies rounding noise; compare against `64 eps` times the operands'
magnitudes instead.

Two refinements, both measured:

- An entry of a linear solve's output errs by about `eps max|x|` (normwise), not
  `eps |x_i|`. A weight held at zero by an active constraint reads `-1e-17`; a
  feasibility test scaled by that weight's own size rejected 50 of 400
  degenerate problems.
- Tests on stored data are exact. `x_i - x_j == 0` holds exactly when
  `x_i == x_j` under IEEE gradual underflow, so tie detection on input data
  needs no tolerance. The book does not state this; it follows from R Lemma
  5.11 (doubles are multiples of the smallest subnormal) and Lemma 6.101
  (rounding near zero), and held on 300,000 pairs including subnormal ones. It
  fails under flush-to-zero.

What it looked like. The LAD ratio test clipped slack with `max(slack, 0)` and
tested `t_hit == 0.0`: a constraint one rounding inside its boundary gave a step
of 4e-16, not 0 (measured, `TestTheRatioTest`), and the degenerate-vertex
check, which tests for that exact zero, would then raise on a sound edge
(inferred, not observed). Now a slack within the rounding of its terms is set to
zero, so a degenerate step is exactly 0, and a slack below `-TOL` times that
scale is an error, not a silent repair.

## 5. Do not difference what you can avoid differencing

"One should avoid taking the difference of two similar numbers if at all
possible" (NW p. 615). In practice:

- Remove a common level when the problem is invariant to it. A pairwise or
  simplex-weighted objective reads only differences; centre before solving.
  Centring is a reparametrisation that changes the conditioning, not the
  problem: fitting `a + b t`, moving `t` to its mean makes the two columns
  orthogonal, and "ill-conditioning is caused by an unsuitable formulation of
  the problem ... a different choice of parametrization can significantly
  reduce the condition number" (B96 Example 2.2.3, pp. 50-51). On the simplex
  the shift is exact: `1'w = 1` gives `y - D w = (y - c) - (D - c) w` for any
  level `c`, per row if need be.
- Centre before a sum whose weights sum to zero: `a'e = a'(e - c)` when
  `sum(a) = 0`.
- Near a root, form `log(f / t)` as `log1p((f - t) / t)`, not `log f - log t`.
- Measure a small angle by its sine, not by `sqrt(1 - cos^2)`. "Since small
  angles theta_k are not well defined by cos theta_k, it is preferable to
  compute sin theta_k more directly": the sines are the singular values of
  `(I - P_F) Q_G` (B Definition 2.2.3 and the SVD after it, p. 244). GvL's
  distance `sqrt(1 - cos(theta_p)^2)` (p. 331) differences two numbers equal to
  about `u`. Measured (2026-11-25, `D1` in the reading notes): on nested
  subspaces the sine route returns `1.1e-15` and the cosine route `3.3e-8`, the
  `sqrt(2u)` floor; at a true angle of `1e-9` the cosine route returns 0. A
  statistic `||(I - H) H'||_F^2` is the sum of squared sines (it equals
  `rank(H') - tr(H H')`, YTT Theorem 2.24, p. 48), so it is computed from the
  residual block, never from cosines.
- Compute a small tail directly, not as `1 - (something near 1)`. The
  subtraction itself is exact for `1/2 <= p <= 2` (Sterbenz, R Corollary 4.16);
  the loss is in the stored operand, which in `[1/2, 1)` already errs by up to
  `2^-54`.

What it looked like:

- the LP perturbation was sized by `max|y|`; at a level of 1e8 it was 0.1, as
  large as the data's variation, and the solves stopped 18 to 26 percent from
  the optimum;
- the uncentred pairwise objective at a level of 1e8 cancelled to 1.5e-9
  relative, above the certificate's tolerance;
- `solve_simplex_qp_least_norm` sizes its tie-breaking ridge by the design's
  mean column energy, level included. With donors `N(0, 1)` plus a level and a
  treated unit inside their hull plus noise of 0.3, the loss it returned
  exceeded the optimum by up to 4.9e-8 relative at levels 1 to 1e2, by up to
  110 percent at 1e2 to 1e4, and by up to 800 percent at 1e4 to 1e6 (300
  problems, measured 2026-10-28). The ridge was sized by the level, so it
  outweighed the variation it was meant to break ties within. The same solve on
  data centred by `mean(y)` matched the optimum to 6e-16 at every level, and
  the plain `solve_simplex_qp` on the raw data to 6.4e-16; no warning was
  raised (rule 1);
- Newton for the chi-square quantile stepped on `log f - log q`; near the root
  the two logs agreed to their last bit, the step was 0, and bisection did the
  work in 57 evaluations. With `log1p` the grid's worst is 6.

Test pattern: a level property (add `c` up to 1e8 to everything; assert the same
answer within the error model of the stored data, rule 11 below), and an exact
reference (`fractions.Fraction`, a closed form) where one exists.

An argument about error is a hypothesis; measure it. The complement `1 - P`
multiplies the relative error of `P` by `P / Q`, 6.4 at the edge of the series'
region, so taking `Q` from the continued fraction when `P > 1/2` looked better.
Measured against `erfc`, the fraction far below `a + 1` errs by 5e-15 to 1e-13,
the complement by at most 7e-16, and the quantile then missed R's by 1.06e-13.
It was reverted, and the measurement is kept as a mutant so the next person to
have the idea sees it fail.

## 6. Do not square the condition number; name rank deficiency

- Least squares by QR or SVD, not the normal equations: forming `J'J` squares
  the condition number (NW pp. 251-253; SVD "the most robust and reliable of
  all", p. 253).
- Cholesky on `A'A` "may fail, i.e., square roots of negative numbers may
  arise, even when `kappa(A)` is of the order of `1/sqrt(u)`" (B96 p. 49):
  about 1e8 in double, a condition number SC designs reach.
- Scaling the columns to unit length nearly minimises the condition number over
  all diagonal scalings (van der Sluis, B96 Theorem 1.4.7, p. 31), but it
  "changes the norm in which the error in x is measured" (p. 31). A column-scaled
  solve is certified on the original columns (rule 2).
- A linear solve by LU with partial pivoting is backward stable, and its forward
  error grows with the condition number (NW p. 617, A.32). Either estimate the
  condition number (from the triangular factor, NW p. 610) or let a certificate
  computed from the result (rule 2) measure it, and say which.
- Rank is a decision about the data's error, and `max(n, p) eps` times the
  largest singular value is only its floor. The numerical delta-rank counts
  singular values above `delta`; with entry errors `|e_ij| <= e`, "an
  appropriate choice is `delta = (mn)^(1/2) e`", and where the error sizes
  differ widely the rows and columns are scaled first (B96 Definition 2.7.1,
  pp. 99-100; B Definition 2.4.1, pp. 291-292). The rank is well determined
  only across a gap: "There are matrices for which no gap exists for any k"
  (B96 p. 100), and "overestimating the rank of A can lead to a computed
  solution of very large norm" (p. 99). A rounding-level cutoff counts noise
  directions as signal. NumPy's defaults are all rounding-level: `pinv` cuts at
  `1e-15` times the largest singular value unless `rcond` or `rtol` is passed,
  `lstsq(rcond=None)` and `matrix_rank` at `max(m, n) eps` (NumPy 2.5 source).
  A cutoff passed to them is the rank decision and is reported with the result.
- The rounding floor is measured against the size of the entries when the
  matrix has been centred: a column of ones centres to noise near 1e-16, which
  a threshold relative to the centred matrix alone counts as a direction.
- Column scaling moves the rank. "Column weighting affects singular values.
  Consequently, a scheme for determining numerical rank may not return the same
  estimate when applied to A and AG^-1" (GvL p. 307). One donor scaled by
  `1e8` turns NumPy's default rank 5 into 4 on the same column space (measured,
  2026-11-25). So the rank is decided on the matrix whose minimiser is reported,
  after its scaling and centring are fixed; a rank chosen on one matrix and
  applied to another (a standardised matrix for the rule, the raw one for the
  truncation) is a modelling choice and is reported as one (backlog).
- The diagonal of a pivoted QR is not a rank certificate. `|r_nn|` bounds
  `sigma_n` from above only, and Kahan's matrix has `sigma_n = 0.368e-8` with
  `r_nn = 0.133` (B96 Example 2.7.1, p. 105; GvL p. 279; B Example 2.4.2,
  p. 296). Decide rank from singular values, or certify a pivoted QR by one or
  two steps of inverse iteration on `R'R`, about `2 n^2` flops (B (2.3.83)-
  (2.3.84), pp. 285-286), or by Chan's ordering, `|r_nn| <= sqrt(n)
  sigma_min` (GvL Theorem 5.4.1, pp. 279-280). Rank read from eigenvalues of
  `A'A` loses everything below about `sqrt(eps)` times the largest singular
  value (BP Example 48.10, pp. 843-844).
- The pseudoinverse is continuous only where the rank is: if `rank(A + E) !=
  rank(A)`, `||(A + E)^+ - A^+||` can be `1/||E||` (B96 Theorem 1.4.1 and
  Corollary 1.4.1, pp. 26-27; GvL p. 290: "not even a continuous function of
  the data"). Refits on perturbed data (placebo, conformal, leave-one-out)
  compare each refit's rank with the full fit's and report a change.
- A rank-deficient problem has a family of minimisers (NW p. 253): every
  minimiser is `A^+ b + (I - A^+ A) z`, the fitted values `A x` are the same for
  all of them, and only the component in `Ker(A)` is free (YTT (3.89), p. 80;
  BP Theorem 32.13, p. 481). Name the family and choose the representative in
  the specification, not in the solver: the minimum-norm point `A^+ b` (what
  `pinv` and `gelsd` return; GvL Theorem 5.5.1, p. 289), a basic solution with
  at most `r` nonzeros (what pivoted QR and `gelsy` return, "not the minimal
  2-norm solution unless the submatrix R12 is zero", GvL p. 292; BP p. 482), or
  the minimiser of `||B x||` for a stated `B`, since minimum norm "is not
  always a good way to resolve rank deficiency" (B96 p. 107). Report which, and
  the rank (rule 1). A small ridge reaches the minimum-norm point only in the
  limit (rule 28).

Test pattern: on a design with more donors than periods, `pinv`, `gelsd`,
`gelsy` and a ridge of `1e-8` return the same fit and different weights; the
spec's representative is the one returned, and the result names it. A
`hypothesis` property scales one column by `10^k` and asserts the reported rank
is the one the spec's rule gives on the scaled matrix, not the default's.

What it looked like. RRMA's Algorithm 1 on the Basque data puts 10 donors in one
group at `m = 6` and fits it on GWCV windows of 10 to 19 pre-periods; on the
first, 10 periods identify 9 directions for 10 coefficients. HiGHS had
returned an arbitrary vertex; the NumPy solver raised an opaque error. `fit()`
now solves in the identified coordinates and warns.

### Constrained least squares: reductions, the problem's own condition, and the face

The exact backend of VanillaSC (`utils/bilevel/exact/lsq.py`) solves, on each
face of the donor hull, `min ||E y - f||` subject to `G y >= h`. It reduces the
problem to a least distance problem through the SVD `E = U S V'` and one NNLS
call, Lawson and Hanson's Chapter 23 route, which Björck gives as Cline's
reduction (B pp. 372-373). Four statements bear on it, and they separate what
the problem costs from what the method costs.

- The weights' own condition is `kappa_LS = kappa(E) (1 + ||r|| / (sigma_n
  ||y||))`, and the fitted values' is about `kappa(E)` (B Theorem 2.2.6, p. 235,
  (2.2.29)-(2.2.31)). With a nonzero residual the weights carry a
  `kappa^2`-sized term that no algorithm removes; the fitted values do not.
  The estimate overstates the true condition number by at most `sqrt(2)`
  (Malyshev, B p. 235), and only the singular values whose left singular
  vectors meet the target enter it (the effective condition number, B p. 230).
  Certify what the caller needs on the scale its condition allows: a check on
  `y` at `1e-7` is out of reach once `kappa^2 ||r|| / ||y||` times `eps` passes
  it, while the same point's fit and loss may still be accurate to `kappa eps`.
- A reduction that multiplies the constraints by `T^{-1}` (Cline's
  `G_1 = E_1 T^{-1}`, B p. 373) forms the transformed problem explicitly, and
  "explicitly forming `B^{-1} A` may result in a loss of precision when `B` is
  ill-conditioned" (B p. 373, on the generalized QR factorization). Quadratic
  programming methods for the same problem "should preferably be adapted to
  work directly with `A`", since a computed `A'A` can turn indefinite (B
  p. 372). The earlier book is blunter: using a quadratic programming code on
  the cross-product matrix "is not a numerically safe approach" (B96 p. 196). The direct route is an active-set method on the original problem
  (Stoer's, B p. 371; LSSOL, B p. 373) whose working-set factorization is
  updated as constraints enter and leave (NW pp. 478-479), with no inverse
  formed.
- Iterative refinement recovers digits that a method lost and the problem did
  not: with `u = 10^-d` and `kappa = 10^q`, `k` refinement steps whose residuals
  are formed in precision `u^2` leave about `min(d, k(d - q))` correct digits
  (GvL Heuristic III, p. 140; B Sections 1.4.6, 2.3.8). It does not reach below
  the problem's own condition, the first bullet's `kappa_LS`. In working
  precision refinement buys stability, not digits: for least squares it
  "improves a componentwise measure of backward stability" (Higham, reported in
  B96 p. 125), and one step after Gaussian elimination gives a componentwise
  backward error at machine precision when the matrix is not too ill-conditioned
  and the components of `|A||x|` do not vary too much (Skeel, B96 p. 124). The digits of Heuristic III need the residual in higher
  precision.
- A rank-deficient problem with a nonempty constraint set has a bounded
  solution, and `E y` is uniquely determined even when `y` is not (Lötstedt,
  B Theorem 2.7.1, p. 371). The loss of the face is therefore defined when the
  face has more free weights than rows. The minimum-norm solution is a least
  distance problem (B p. 372), with `z_2` free in Cline's form when
  `rank(E) < n` (B p. 373). Refusing such a face refuses a well-posed problem;
  the documented representative of this rule's last bullet applies.
- On when the face is a point, B96 states that with `A` rank deficient "there
  will still be a unique solution if all active constraints at the solution
  have nonzero Lagrange multipliers. Otherwise there is an infinite manifold M
  of optimal solutions with a unique optimal value", whose minimum-norm member
  is a least distance problem (p. 200). Read as an algorithm, the condition is
  on the directions left free: the constraints with nonzero multipliers stay
  active along any move between minimisers, and the minimiser is unique when
  `A`, restricted to the remaining coordinates and with the remaining active
  rows appended, has full column rank. `minnorm.simplex_optimum_is_unique`
  tests exactly that for the simplex.
- A linearly dependent working set "can cause possible cycling", and the remedy
  in use is "to enlarge the feasible region of the offending constraint by a
  small quantity" (B96 p. 201). That is rule 10's perturbation, with rule 10's
  obligation: certify on the unperturbed constraints.
- An active-set method changes one constraint per iteration, so it "can be slow
  to converge when the set of active constraints cannot be guessed well" (B96
  pp. 202-203). Along a sequence of nearby problems the previous answer is that
  guess: seeding SparseSC's 4,000 inner solves on the Basque panel with the
  previous weights cut the least-norm solve from 627 to 94 microseconds
  (jgreathouse9/mlsynth#940). The start changes the work, not the contract; on
  a face it may change which member the tie-break reaches, within the tie-break's
  own accuracy.

### Equality constraints imposed by a weight

Appending `gamma 1'` to the design and `gamma` to the target makes an NNLS code
solve a simplex problem, Lawson and Hanson's method of weighting (B96 Section
5.1.5, pp. 192-194). Three statements bound it.

- The constraint holds only to `O(gamma^-2)`: the residual of the weighted rows
  is proportional to `gamma^-2` for large `gamma` (p. 192).
- "Except in special cases, the method of normal equations will fail for large
  values of `gamma`" (p. 192). The weighted problem is solved by QR, with the
  weighted rows first, and column pivoting is needed for accuracy when the
  constraint columns are dependent: in Van Loan's example, `gamma = 1e9` gave
  full precision with pivoting and errors of order `1e-9` without (p. 193).
- `gamma` is sized by the variation of the data, not their level (rule 5), and
  closing the remaining `O(gamma^-2)` gap by renormalising is a repair (rule 8):
  allowed within the tolerance a certificate has checked, reported beyond it.
- Elimination is exact and is the books' first choice: "The most efficient way
  to solve problem LSE is to derive an equivalent unconstrained least squares
  problem of lower dimension" (B p. 369; GvL Algorithm 6.2.2, p. 316; YTT
  Theorem 4.10, p. 101). For `1'w = 1`, write `w = 1/J + Z v` with `Z` an
  orthonormal basis of `{1'd = 0}`, as `utils/weights/solve.py` `_affine` does.
  The minimiser is unique if and only if `Ker(B)` and `Ker(1')` meet only in 0
  (B96 (5.1.2)-(5.1.3), p. 187), that is, `[B; 1']` has full column rank (BP
  Theorem 33.6(b), p. 491), that is, `B Z` has full column rank. A design with
  `B 1 = 0` is rank deficient and still has a unique sum-to-one minimiser.

What it looked like. CTSC's `_simplex_ls` (`utils/ctsc_helpers/estimate.py`)
sets `gamma = 1e3` times the root mean square of the donors, level included,
then renormalises. On 400 problems (measured 2026-10-28) the weights summed to
one within 1.9e-6 at levels 1 to 1e2, 1.6e-8 at 1e2 to 1e4 and 2.2e-10 at 1e4 to
1e6; the gap shrinks with the level because `gamma` grows with it. After
renormalisation the loss was within 6.6e-11 of the optimum. The answer is
accurate; the repair is unchecked and unreported.

What it looked like. On Abadie and Gardeazabal's Basque specification the
exact backend refuses three times, and each refusal is a different statement
above.

- The 13 predictors span 12 dimensions: the four schooling shares, averaged
  over 1964-1969, sum to 83.333 for every region. The hull is not
  full-dimensional, and `hull.py` refuses without naming the predictors.
- With `school.higher` dropped (`K = 12`), a face has 11 free weights and the
  outcome fit over 1960-1969 has 10 rows. The face is rank-deficient; the loss
  is defined (Lötstedt), and `_lsi` refuses it.
- With the fit widened to 1955-1969 (15 rows), the first face has
  `kappa(E) = 1.1e4` against `COND_MAX = 1e4`, and both predictor-weight classes
  refuse in 0.1 to 0.2 seconds. With `COND_MAX` raised to `1e6` in a scratch
  run, every returned point still passed the `1e-7` constraint check, and the
  search finished: 21.6 seconds for the diagonal class, 0.2 for the positive
  definite class (measured, 2026-10-27).

The cap at `1e4` was the method's choice, not the problem's limit (measured).
The search on the Basque specification makes 88,631 face fits, 429 above the
cap and none above `2.62e4`. Against a 50-digit reference, the KKT conditions
of the equality-constrained problem on each answer's active rows verified in 50
digits, the reduction's answers on 492 faces were right at every condition
number: weights within `9.1e-9` (growing like `kappa^2 eps`, the first bullet),
fits within `1.3e-11`, losses within `1.2e-11`. The cap is replaced by a
certificate, `lsq.kkt_backward_error`, the KKT conditions' backward error on the
face's own data. On 4,192 captured answers it was at most `1.3e-14` at every
condition number, and an answer moved `1e-6` along a free direction scored at
least `1.6e-10`; `CERTIFY = 1e-12` sits between them (rule 11). The reduction's
point then names the active rows and the fit is solved on them directly (the
third bullet's direct route, restricted to the final working set): without that
step the point missed an active row by about `1e-9`, its loss fell below the
optimum, and the clipped weights were not the certified ones. The certificate
costs about a quarter on the diagonal class (California's fit 0.22 s to 0.28 s).

## 7. Iterations are safeguarded and end

- Every loop has a cap, and reaching it raises or warns (rule 1). Where a loop
  has no cap, its termination argument is written beside it (the kinetic line
  search removes one inversion per swap, so it ends within `n (n - 1) / 2`).
- A Newton or interpolation step is taken only inside the bracket and only when
  steps contract: NW requires a safeguarded contraction factor in
  `[rho_lo, rho_hi]` inside `(0, 1)` (p. 37) and an interpolated step kept away
  from the bracket's ends (p. 62). The rule used here is Brent's: shorter than
  half the step before last, otherwise bisect, or expand while no upper end is
  known (NW p. 61: steps must grow fast enough to reach the limit).
- Stop on a relative change near machine precision (NW p. 62) and a residual at
  the rounding level of the function, together. A step test alone is fooled by
  a wrong derivative; a residual test alone never fires when the function
  carries more error than the threshold.
- A bound used as a bracket end is lowered by its own rounding error. The
  chi-square leading-term root is a lower bound in exact arithmetic; at
  `q = 1.7e-305` its exponent errs by 702 eps, the computed bound landed above
  the root, and the iteration converged onto the wrong end.

- Stop on a certificate where the problem has one, and report it. A
  convergence test on the size of a step says the iteration slowed down, not
  that it arrived. Nuclear-norm completion has an exact one: `Z` solves the
  problem if and only if `Z = S_lambda(P_O X + P_O^perp Z)`, the soft-thresholded
  SVD of the data filled with `Z`'s own entries (HMLZ (47), p. 3384, with
  Theorem 1, p. 3370; derived), and the paper's step measure goes to zero at
  rate `1/K` (Theorem 4, (39), p. 3381). A factored or nonconvex solver's
  stationary point need not solve the convex problem (HMLZ p. 3383), so its
  answer passes the convex certificate before it is called an optimum.
- An iteration whose answer depends on where it stops is an estimator with the
  stopping index as a parameter (rule 29). A step or gradient test with a cap
  is a convergence test; the cap reached is an outcome (rule 1), and the
  iteration count, the test that stopped it and the certificate residual are
  fields on the result.

Test pattern: count evaluations over a grid and bound the worst; corrupt the
derivative (multiply it by 1000) and assert convergence to the right root. For a
fixed-point iteration, assert that the reported certificate residual is the one
recomputed from the returned point on the caller's data, and that a run capped
at two iterations reports the cap.

What it looked like. MC-NNM stops when `||L - L_prev|| / ||L_prev||` falls
below `1e-5`, with a cap of 400 and no flag (`utils/mcnnm_helpers/completion.py`).
On 8 synthetic panels (2026-11-25) the default stop left a fixed-point residual
of `6.7e-7` to `8.5e-6` and an ATT `1.4e-6` to `1.2e-3` (relative) from a
`1e-12` stop; the cap was not reached, and the result could not have said so if
it had.

## 8. Repairs to a solver's output are reported

Clipping a solver's output (`np.clip(w, 0, None)` then renormalising) is allowed
only within the tolerance the certificate already checked. Beyond it the solve
failed, and that is reported, not repaired: a weight of -0.3 renormalised away
is an estimate nobody can trace.

## 9. Form quantities so the failure mode is underflow

When a ratio can overflow, use its reciprocal: a Newton step `d / (pdf / P)`
overflows when the root is subnormal; `d * (P / pdf)` underflows to a harmless
0. When a product can be formed directly, do not form it as `exp` of a sum of
logs: the exponent's absolute error `eps |sum|` becomes a relative error in the
result (NW p. 616). Through logs, the chi-square prefactor cost the quantile
7 eps at `q = 1.5e-52` and 188 eps at the edge of the normal range; formed
directly unless a piece overflows, the worst over 15,000 points is
`6 eps (1 + kappa)`, with `kappa` its condition number (rule 13).

## 10. Degeneracy: perturb to pivot, evaluate and certify on the original

A degenerate vertex makes pivoting ambiguous. Perturb the data to pivot (NW
pp. 381-382), then reset the solution from the final basis on the original data
(NW p. 382: "resetting x_B = B^-1 b for the final basis B and the original
right-hand side b") and certify there (rule 2). The perturbation is sized by the
data's variation, not their level (rule 5), and must survive the rounding of
`y + shift` (R Lemma 6.45): a shift of `1e-9 * spread * xi_i` on centred data
survives when `|xi_i|` exceeds about 1.1e-7, and for the RRMA hash the smallest
`|xi|` for `n <= 500` is 1.06e-3.

None of the LP texts consulted gives an anti-cycling rule for a bounded dual
simplex. Where the pivot rule carries no guarantee, the certificate carries the
claim of optimality, and the docstring says so.

---

# Verification: rigorous results in floating point

Rules 1 to 10 make a solver's answer accurate and its outcome explicit. Rules
17 to 27 cover the stronger claim a branch and bound, an exact backend or a
certified optimum makes: that a bound holds, proved. Rump (2010) is the source
throughout. Its subject is "methods providing rigorous results which, in
particular, use floating-point arithmetic" (p. 291), and its claim is that such
methods need neither exact arithmetic nor a slow one: the goal is "to derive
verification algorithms which are, for certain classes of problems, not much
slower than the best numerical algorithms, say by at most an order of
magnitude" (p. 293).

Verification is not reliability. Stochastic error estimates, careful iterative
refinement and a solver that "proves reliable in millions of examples" are
"approaches increasing reliability"; "none of these approaches claims to
produce always true results" (p. 293). A tolerance-based certificate (rule 2's
KKT residual at most `TOL`) belongs to that class. A verification certificate
is a proof, conditional only on the hardware, compiler and operating system
working to their specification (p. 304).

## 17. A certificate is a theorem whose assumptions the computer verifies

"Mathematical theorems are formulated whose assumptions are verified with the
aid of a computer" (Rump (1.1), p. 292). Three consequences bind.

- State the theorem first, in the docstring, with its source: the assertion,
  the hypotheses, and which hypotheses the code checks numerically. The
  slope-free certificate's theorem is weak duality with a trace bound (Jansson,
  Chaykin and Keil 2007, Rump Theorem 14.2); its verified hypotheses are an
  upper bound on `lambda_max(S)` and the rounding of `S` and `k`.
- "Mostly those theorems are sufficient criteria: if the assumptions are
  satisfied, the assertions are true; if not, nothing can be said" (p. 292).
  A verification that fails reports the node or the claim open (rule 1), never
  false, and never retries with a loosened hypothesis.
- Formulate the theorem so that its assumption "is likely to be satisfied for
  an approximate [quantity] calculated in floating-point arithmetic. This is
  the interplay between the mathematical theorem and the practical
  application" (p. 300). In Theorem 1.1 the matrix `R` "is arbitrary, so
  neither the 'quality' of R as an approximate inverse nor its non-singularity
  need to be checked" (p. 300). In `mlsynth` the multipliers play `R`'s part:
  any multipliers give a valid bound (rule 26), so the solver that proposed
  them is never trusted and never checked, only the bound.

"All numerical experience should be used to design the mathematical theorem,
the assumptions to be verified, and the way to compute R" (p. 300): Rump's
example is computing `R` as a left inverse, for which `rho(I - RA) < 1` is
likely even when `A` is ill-conditioned. The choice of the theorem decides how
often verification succeeds; the floating-point code decides only whether it
is valid.

## 18. Verify only well-posed claims; equality and singularity are decided exactly

"Verification methods solve well-posed problems" (Rump (1.2), p. 293). A claim
is verifiable when it holds on a neighbourhood of the data: "if the assumption
`rho(I - RA) < 1` is satisfied, then it is satisfied for all matrices in a small
neighbourhood of A" (p. 300). Its negation at a boundary is not: "the proof of
singularity is outside the scope of verification methods because in every open
neighbourhood of a matrix there is a non-singular matrix" (p. 293); "the
verification of singularity excludes the use of estimates: it is only possible
using exact computation" (p. 309). The exceptions Rump names are integer input
data and semidefinite programs that are not at zero distance from infeasibility
on both sides (p. 293).

In `mlsynth`:

- Prove strict statements with room. A size is closed when its bound is at most
  zero; the slope-free search certifies at `lam' = dyadic_below(lam) <= lam`, a
  stronger statement than the one at `lam`, so the claim survives rounding of
  `lam` itself.
- An exact tie (two designs of equal RSS, a matrix exactly singular, a bound
  exactly zero) is decided in rationals or not at all. The float tier of a
  certificate decides only the cases its error bound separates from the
  boundary; between its upper and lower bounds the exact certificate decides
  (`certificate.joint_clears` on `main`; the slope-free certificate's float tier
  on #965's branch), and an exact zero is a closure there.
- An ill-posed problem is regularized first and the regularized problem
  verified (p. 293), with the regularization reported (rule 8).

## 19. Bound every rounding: the floating-point model with underflow

IEEE 754 in rounding to nearest defines every basic operation exactly (Rump
(2.1)-(2.2), p. 302), and "basic floating-point operations ... according to
IEEE 754, are defined precisely, and are accurate to the last bit" (p. 294).
Standard library functions (`exp`, `log`, `erf`) are not: "there is no proof of
that property", so they cannot be used "as is" in a verification (p. 294). A
verified quantity is built from `+ - * /` and `sqrt` only, or from a function
with its own proved error bound.

The model, with `u = 2^-53` and the underflow unit `eta = 2^-1074` (Rump (2.3),
p. 302):

    a op b = fl(a op b) (1 + eps) + delta,  |eps| <= u, |delta| <= eta / 2,
    eps delta = 0,

and for addition and subtraction "the result is exact if underflow occurs"
(Rump (2.4)): `a +- b = fl(a +- b)(1 + eps)`. Taking an absolute value is exact.

Sums and dot products (Rump Theorem 2.1, p. 303, after Higham 2002, (3.2) and
Lemma 8.2), evaluated "in any order":

    |fl(sum x_i) - sum x_i| <= gamma_{n-1} sum |x_i|,                  (2.5)
    |fl(c - x'y) - (c - x'y)| <= gamma_n |x|'|y|,  no underflow,        (2.6)

with `gamma_n = n u / (1 - n u)`. "In any order" is what makes the bound usable
on BLAS and sparse products, whose summation order is not specified; and "for
composite operations, such as dot products, intermediate results are sometimes
accumulated in some extended precision ... However, the error estimates ...
remain valid" (p. 301).

Three refinements are needed before (2.6) is a computable bound.

- The right-hand side is itself computed. With `e~ = fl(|x|'|y|)`,
  `|e| <= |e~| + gamma_n |e|`, so the error is at most
  `gamma_{2n} / 2 * e~` (Rump (2.7)-(2.8)), and with the computed constant
  `gamma~_n = fl(n u / (1 - n u))` it is at most `fl(gamma~_{2n+2} / 2 * e~)`
  (Rump (2.9)). `certificate.inflate` covers the same rounding by a relative
  `1e-6`; either is valid, and the docstring says which is used.
- Underflow. (2.6) holds "provided no underflow has occurred". Each product
  contributes at most `eta / 2` more by (2.3) and sums add nothing by (2.4), so
  an `n`-term dot product errs by at most `gamma_n |x|'|y| + n eta`
  (`certificate._dot_err`). A data entry or coefficient that converts to a
  subnormal double is within `eta / 2` of its value, not within `u` of it: a
  conversion bound `u |x|` holds only for a result at least the smallest normal
  double (`np.finfo(float).tiny`; below it `eta / 2` is `u` times it at
  most), and code that relies on it checks that.
- Overflow. The model holds for `|x| <= realmax` (Rump (2.2)). A bound that
  overflows is no bound: return `+inf` (an upper bound that closes nothing) or
  `-inf`, and let the exact certificate or the open state decide.

"For each operation the worst-case error is (and has to be) assumed" (p. 304):
an a priori bound such as (2.10) is pessimistic by design, by about `n u`
relative, which is why the float tier of a certificate leaves a band around zero
to the exact tier (rule 18).

Error-free transformations make the rounding error itself a double: for
`+ - *`, "the error of every floating-point operation is itself a
floating-point number" (Rump (3.1), p. 304). Knuth's `TwoSum` (Algorithm 3.1,
six operations, any inputs) and Dekker's `FastTwoSum` (Algorithm 3.3, three
operations, `|a| >= |b|`) return `x = fl(a + b)` and `y` with `x + y = a + b`
exactly (Theorems 3.2, 3.4); `VecSum` (Algorithm 3.5) carries a vector to one
with the same exact sum (Theorem 3.6), and Neumaier's `Sums` (Algorithm 3.8)
gives a sum "as if computed in quadruple precision", within
`u|s| + gamma_n^2 sum |p_i|` (Rump (3.10)). Dekker's `Split` (Algorithm 10.3)
cuts a double into two halves whose products are exact, the basis of the
improved residual (rule 23). Python's `math.fsum` returns the correctly rounded
sum of its arguments (its documentation; not in Rump), and is the
transformation to reach for in a scalar sum that decides a certificate.

## 20. Utilize input data: verify from the data, not from computed intermediates

"Avoid re-use of computed data; use input data where possible" (Rump (5.13),
p. 312). The reason is data dependency: an inclusion of `X - X` has twice the
width of `X` (Rump (5.9), p. 311), "absolute errors add" under subtraction
(5.10), and each consecutive operation on computed bounds widens them. Replacing
each operation of an algorithm by its interval operation "will almost certainly
fail by producing wide and useless bounds" (p. 315): the naive interval Newton
iteration diverges from `[1.4, 1.5]` to `[-12.2, 8.5]` in five steps (Algorithm
6.1); interval Gaussian elimination on random matrices loses all accuracy by
dimension 60 to 70 (Table 10.2, 96 percent failures at 60, 100 at 70), "solely
[from] the number of consecutive interval operations, permanently violating the
Utilize input data principle" (p. 336); a product of random 3 x 3 matrices in
interval arithmetic loses about a factor 2 per step, so a relative error of 1
"can be expected after some 50 iterations" (p. 337). "Interval arithmetic is
not to blame for this: the result would be the same with power set operations
applied in this way" (p. 336).

The rule for `mlsynth`: a certificate is computed in one pass from the
problem's data and the candidate answer, never propagated through the
algorithm that produced the answer. Rule 2 already requires certifying on the
original data; this is why a running error bound carried through a pivot loop,
an elimination or an iteration is not a substitute. The slope-free and joint
certificates assemble `S` from the data and the multipliers directly; the
solver's internal matrices, scalings and iterates enter nothing that is proved.

"The No inaccurate cancellation principle (NIC)" of Demmel, Dumitriu, Holtz and
Koev (reported by Rump, p. 312) states when a formula is accurate without
verification: additions only of same-sign data or of input data, subtractions
only of opposite-sign data or of input data. A modulus sum `|x|'|y|` (the
right-hand side of (2.6)) satisfies NIC, which is why its own rounding is a
small relative factor.

## 21. Move a bound from the computed matrix to the exact one by perturbation theory

A symmetric matrix computed in floating point differs from the exact one by an
error bounded entrywise (rule 19). Weyl's inequality moves an eigenvalue bound
across that difference: for symmetric `A`, `E`,

    lambda_i(A) + lambda_n(E) <= lambda_i(A + E) <= lambda_i(A) + lambda_1(E)

(Rump Lemma 10.12, p. 357, from Wilkinson 1965). With `|S - S~| <= E`
entrywise, `|lambda_i(S) - lambda_i(S~)| <= ||S - S~||_2 <= || |S - S~| ||_2
<= ||E||_2` (the middle step is used in the proof of Rump Lemma 10.14, p. 359),
and `||E||_2 <= ||E||_inf` for symmetric `E`. A sharper bound for a
nonnegative `E` is Collatz's: for any positive vector `x`,
`||E||_2 <= sqrt(max_i (E'(Ex))_i / x_i)` (Rump (10.61), p. 359); `x` the
computed Perron vector makes it nearly tight.

The same lemma turns any approximate factor into a bound: with `G` any matrix
and `E := A - a I - G'G`, `lambda_n(A) >= a - ||E||_2` (Rump (10.54)). `G`
need not be a good factor; only `||E||_2` has to be bounded rigorously, from
the input data (rule 20). This is `mlsynth`'s `lambda_min_lower` (exact
residual, `marex_helpers/branch.py`) and `certificate.lambda_min_lower_float`
(residual bounded a priori).

## 22. Positive definiteness: one factorization that completes, or a rigorous residual

Rump gives two verified routes to `lambda_min(A) >= a`, and `mlsynth` uses both.

- The residual route (rule 21): factor `A - a I` approximately, bound the
  residual `A - a I - G'G` rigorously, subtract its norm. The exact residual in
  rationals is a cubic count of big-integer products; the a priori residual
  bound of `certificate.cholesky_residual` costs one matrix product.
- The factorization route (Rump Section 10.8.1, after Demmel 1989). If a
  floating-point Cholesky factorization runs to completion, then, barring over-
  and underflow, `G~'G~ = A + DA` with
  `||DA|| <= sum_j phi_{j+1} a_jj`, `phi_k = gamma_k / (1 - gamma_k)`
  (Rump Lemma 10.13, (10.60)). So `A` is proved positive definite by
  factoring `A - a~ I`, with `a~` at least that bound plus an underflow term,
  and checking only that the factorization completes. No residual is formed;
  "it is much faster than (10.54), albeit with a more restricted range of
  applicability" (p. 357): it verifies down to a smallest singular value of
  about `1.5e-13 ||A||` at dimension 100 (Table 10.10), where the residual
  route reaches "almost of size u" (p. 359). An interval matrix
  `[M - R, M + R]` is positive definite when `M - cI` is and `||R||_2 <= c`
  (Rump Lemma 10.14).

`benchmarks/studies/bnb_node_cost/parts.rump_lower` implements the factorization
route and measured it 4.5 to 6.5 times faster than the exact residual (7.0 to
20.7 ms to 1.4 to 3.6 ms) with bounds within `3.5e-16` of the exact ones
(`benchmarks/studies/bnb_node_cost/README.md`). Which route a certificate
takes is a cost choice; both are proofs. The near-zero band where neither
separates the bound from zero is the exact tier's (rule 18).

## 23. A bound on a linear solve comes from a preconditioned residual, never from the elimination

Interval Gaussian elimination fails (rule 20), and "there seems little
potential for converting floating-point Gaussian elimination into a
verification algorithm, because known error estimates need an upper bound for
the condition number" (Rump p. 338); a reliable condition estimate costs as
much as a reliable inverse (Demmel, Diament and Malajovich, reported p. 338).
Rump's own example: Gaussian elimination with partial pivoting on Wright's
trapezoid-rule system of dimension 65, condition number below 30, returns
components `-23.27, 0, -93.09, 1.00` where the solution is all ones, "no warning
is given by MATLAB, so a user might trust the computed values" (p. 338); one
residual iteration in working precision repairs it.

Verification inverts the usual advice: "It is an unwritten rule in numerical
analysis never to compute the inverse of a matrix ... For the purpose of
verification other rules apply, in particular to rely only on the input data if
possible" (p. 338). With any `R` and any approximation `x~`,

    ||A^{-1} b - x~|| <= ||R(b - A x~)|| / (1 - ||I - RA||)   if ||I - RA|| < 1

(Rump Theorem 10.2, (10.9)), which also proves `A` and `R` nonsingular. The
stronger form is Krawczyk's operator as an existence test: if
`R(b - A x~) + (I - RA) X` lies in the interior of `X`, then `A^{-1} b` is in
`x~ + R(b - A x~) + (I - RA) X` (Rump Theorem 10.6 with (10.14)), computed by
iteration with epsilon-inflation, `Y := X [0.9, 1.1] + [-e, e]` (Rump (10.13),
Algorithm 10.7). Three things carry over to every verification by fixed point,
nonlinear and eigenvalue problems included (Rump Theorems 13.3, 13.9):

- include the error of an approximate solution, not the solution (p. 343);
- inflate before testing inclusion, since without it the test can fail on a
  problem it should pass (p. 343, the 1 x 1 example with `1 - RA = -0.6`);
- the test succeeds essentially when `rho(|I - RA|) < 1` (p. 345), so it fails
  only on problems whose floating-point solution has no correct digit.

The bound's quality follows the residual: "the quality of the bound in (10.9)
depends directly on the size of the residual" (p. 339). An improved residual
(Rump Algorithm 10.4, `lssresidual`, by Dekker's `Split`; or an accurate dot
product, Section 3) buys accuracy that iterative refinement in working
precision does not (rule 6's GvL Heuristic III). For an M-matrix,
`||A^{-1}||_inf <= ||y~|| / (1 - ||e - A y~||_inf)` for an approximate solution
`y~` of `Ay = e`, an `O(n^2)` bound (Rump (10.62), p. 361).

`mlsynth` has no verified linear solve on `main`. A future exact backend that
certifies a weight vector by solving a system cites this rule, and its
certificate is (10.9) or Algorithm 10.7, not a residual check at `TOL`.

## 24. Range bounds: each variable once, and the inclusion of an image is not an inclusion of a preimage

The inclusion property (Rump (5.16), p. 314) and principle (5.17) make every
interval evaluation an outer bound on the range of a function: "the result will
always be true; however, much overestimation may occur" (p. 314). A branch and
bound's node bounds are range bounds, so the rules of Sections 8 and 9 apply.

- "If f is evaluated by some arithmetic expression and each variable `x_i`
  occurs at most once", the interval evaluation is the exact range (Rump
  Theorem 8.1, p. 321). `x^2 - 4x` over `[1, 4]` evaluates to `[-15, 12]`,
  `x(x - 4)` to `[-12, 0]`, `(x - 2)^2 - 4` to the exact `[-4, 0]` (p. 320).
  Rewrite a node bound so each free variable appears once before bounding it.
  "There seems to be no general recipe" (p. 321), so the rewrite is measured.
- A matrix with interval entries times a point vector is exact (Rump (9.4)); a
  point matrix times an interval vector is exact componentwise (9.6), not as a
  set (9.5).
- The pitfall: from `b` in `AX` it does not follow that `X` contains
  `A^{-1} b` (Rump (9.8), p. 327). A bound on the image of a box says nothing
  about which point of the box maps to a given value. A certificate that a
  solution lies in a region is a fixed-point argument (rule 23), not an
  evaluation.
- Monotonicity removes the overestimation: where an inclusion of the
  derivative excludes zero, the range is the image of the endpoints (Rump
  (8.8)); otherwise bisect (Table 8.1). A local optimizer finds an inner
  inclusion and can miss the global extremum (`fminbnd` underestimates the
  range in Table 8.2), so it can propose a bound, never prove one.

## 25. Bounds in matrix form, by products of magnitudes

Branches are slow and rounding-mode switches slower (Rump Section 9.1, Table
9.1: total pivoting runs at 0.1 to 0.2 GFLOPs where matrix multiplication runs
at 121 to 166). A rigorous bound is computed as a few dense or sparse matrix
products: in midpoint-radius form, `A B` for an interval `A = <mA, rA>` is
`<mA B, rA |B|>` (Rump (9.10)), and for two interval matrices
`<mA mB, rA(|mB| + rB) + |mA| rB>` (9.14), at the cost of three or four real
products (Tables 9.2, 9.3). The a priori bounds of rule 19 have the same shape:
the error of `M y` is bounded by `|M| |y|`, one more product.
`certificate.joint_assembly` bounds the joint lift's slack this way, and the
slope-free certificate's float tier (#965) bounds its slack by `|M|' |y|` and
its face by `|T|' |S| |T|`: 2 ms a certificate where the rational sum took 120
to 330 ms, with the same closures on two 100-geo groups (the #965 spike,
2026-11-10). Loops over entries in Python, and `Fraction` arithmetic in
particular, are a last resort for the band rule 18 leaves.

## 26. A relaxation's bound is postprocessed from any multipliers, never read from the solver

Linear programs are not safe: 72 percent of the NETLIB problems "are
ill-conditioned ... many commercial solvers fail" (Ordóñez and Freund, reported
by Rump p. 393); "preprocessing in particular may change the status of the
linear program from feasible to infeasible, and vice versa" (p. 393); "it is not
easy to program a robust simplex algorithm. Even for small problems it is not
unlikely that an incorrect branch will lead to a sub-optimal result or apparent
infeasibility" (p. 393); and "nonlinear programming solvers often fail even in
solving convex problems" (Tawaralani and Sahinidis, p. 393). In a branch and
bound, "ill-conditioning is ... likely to take place in combinatorial
optimization when branch-and-cut procedures sequentially generate linear or
semidefinite programming relaxations. Rigorous results in combinatorial
optimization can be obtained when solving the relaxations rigorously" (p. 395).
SDPT3 gave no warnings for 27 of the 32 ill-posed SDPLIB problems (p. 397).

The remedy is postprocessing: "obtain rigorous bounds by postprocessing already
computed data" (p. 395). For the semidefinite program
`min <C, X>` subject to `<A_i, X> = b_i`, `X >= 0`, and any `y~`, with
`D = C - sum y~_i A_i`, `d <= lambda_min(D)` and `lambda_max(X) <= x` on
the feasible set,

    p* >= b'y~ + s min(0, d) x

(Rump Theorem 14.2, (14.8), p. 395, from Jansson, Chaykin and Keil 2007, for
each block of size `s`); if no bound on `X` is known, `d >= 0` is required.
The proof needs only `<E, Y> >= k min(0, d) x` for `d <= lambda_min(E)` and
`0 <= Y`, `lambda_max(Y) <= x` (Rump (14.10)), and the same proof gives
`<E, Y> >= min(0, d) tr(Y)`, so a trace bound `tr(Y) <= T` replaces `s x`
(derived here; the slope-free and joint certificates use the trace).
`lambda_min(D)` is bounded by rule 22.

So: a solver's status, objective value or "optimal" flag is never a bound. The
multipliers it returns are a proposal; the bound is the theorem evaluated at
them, rigorously, on the problem's own data (rule 2). Where the theorem's
`d >= 0` condition fails, Jansson's remedy is to relax the constraint and solve
again (p. 396); `mlsynth`'s certificates instead carry a trace bound, so every
multiplier gives a finite bound and a loose one simply does not close.

## 27. A directed bound is rounded in its own direction, and says which

IEEE 754 defines rounding toward minus and plus infinity (Rump (4.1)), valid for
every operation "including possible over- or underflow" (p. 308); a dot product
computed entirely in one mode is a one-sided bound (Rump (4.3)), and two passes
give an enclosure at the cost of two products (Algorithm 4.1). The trap: in
`I - RA` "the multiplication and subtraction must be computed in opposite
rounding modes to obtain valid results" (p. 310). The sign of each term decides
its rounding direction.

NumPy and the BLAS it calls do not expose the rounding mode reliably, so
`mlsynth` uses the rounding-to-nearest route of rule 19 and rounds only the
final scalar in its direction: `certificate.round_down` (the largest double at
most a rational, decided by an exact comparison) and `add_down` (one step down
with `nextafter`). Every bound in a docstring says which side it is on ("a
double at most ...", "at least ..."), and a mutant flips the direction once.

## Testing a verification

Verification code allows "a kind of testing which is hardly possible for
numerical algorithms": an approximate `3.14159265358978` gives no reason to
doubt, "but the wrong 'inclusion' `[3.14159265358978, 3.14159265358979]` would
reveal an error" (Rump p. 300). The test is the enclosure, not the value.

- Assert the enclosure against an exact reference (`fractions.Fraction`, rule
  15): the computed bound is on the right side of the exact value for every
  input a `hypothesis` property draws, including subnormal and huge magnitudes,
  mixed signs and cancellation (`test_joint_float.py`'s
  `test_property_the_assembly_error_bounds_its_rounding` checks
  `|fl(S) - S| <= E` entrywise in rationals).
- Assert tightness separately, at a fixture, as a relative distance to the
  exact certificate: a valid bound that is never tight enough to close is a
  defect of performance, not of soundness, and is tested as such.
- Assert the failure mode: a verification that fails returns open (rule 17),
  and a non-finite input returns the bound that closes nothing (rule 19).
- One mutant per hypothesis the code verifies: drop the rounding term, flip the
  rounding direction, skip the exact tier between the float bounds.

---

# Least squares weight problems: layers, filters, refits (28 to 31)

Rules 1 to 10 make one solve accurate and its outcome explicit. Rules 28 to 31
cover what sits around a least squares weight fit: which transformations belong
to the solver and which change the estimator, how a regularisation parameter is
carried, how a fit is refitted after a row leaves, and what a cross-validated
error measures. They were extracted from a reading of B, B96, GvL, YTT, ESL,
BP, HMLZ and WZLZ done for the single least squares front door (#1065).

## 28. A rewrite is the solver's only if it keeps the set of minimisers

A weight fit has three layers: the design (which matrix and target go in: rows
kept, truncation, filling, centring, scaling), the problem (objective and
feasible set), and the solver (how a minimiser is found and certified). A
transformation belongs to the solver when the transformed problem has the same
minimisers as the stated one; otherwise it is a modelling choice and lives in
the specification and the result.

- Stacking a ridge is a solver rewrite of the penalised problem:
  `||b - Ax||^2 + lambda ||x||^2 = ||[b; 0] - [A; sqrt(lambda) I] x||^2` for
  every `x`, so the two have the same unique minimiser and the stacked matrix
  has full column rank (B96 p. 101; BP Theorem 33.4 and (33.5), pp. 489-490).
  The ridge itself is a problem choice: at every `lambda > 0` its minimiser
  differs from every unpenalised minimiser and its fitted values
  `U diag(sigma^2 / (sigma^2 + lambda)) U'b` differ from `P_A b` (YTT (4.123),
  p. 120; ESL (3.47), p. 66). It tends to the minimum-norm point only as
  `lambda -> 0` (BP Theorem 48.9, pp. 842-843). A tiny ridge added "to make the
  Gram invertible" therefore changes the problem and names no representative.
- Eliminating an equality constraint through a null-space basis is a change of
  variables and keeps the minimisers (B p. 369; YTT p. 101). Centring away an
  unpenalised common level keeps them (B96 pp. 50-51, rule 5). Centring or
  scaling before a truncation, a ridge or a rank rule does not: principal
  components "depend on the scaling of the inputs" (ESL p. 79), and PCA is
  defined on the centred matrix (BP pp. 868-870).
- Truncating singular values, filling missing cells, and stopping an iteration
  early change the minimisers (rule 29).

The post-period counterfactual inherits the face. The minimisers are
`A^+ b + Ker(A)` (YTT (3.89), p. 80), so `B_post w` is the same for every
minimiser if and only if `B_post` annihilates `Ker(B_pre)`, that is, every row
of `B_post` lies in the row space of `B_pre` (derived; the subspace inclusion
of #1156). On the sum-to-one set the face is unique if and only if `[B; 1']`
has full column rank (rule 6's equality bullet). `WeightSolution.unique` and
`WeightSolution.identifies(B_post)` (`utils/weights/solution.py`) implement
both tests; their null space is taken at the rounding floor, and rule 6 says
where the decision belongs.

Failure it prevents: two code paths that state the same problem and return
different counterfactuals because one used `pinv`, one `lstsq` and one a Gram
plus `1e-8 I`, none of which named its tie-break. Measured (2026-11-25): stacked
`lstsq` and the SVD ridge agree to `1.7e-15` at `lambda = 1`; the ridge and the
minimum-norm point differ by `0.15` at `lambda = 1`, `2.3e-4` at `1e-3` and
`2.3e-9` at `1e-8`.

Test pattern: for each rewrite a solver performs, a `hypothesis` property that
the original and rewritten objectives agree at random points (an identity of
functions, not of answers). For a problem-layer choice, a test that the result
records it. For the counterfactual, a design with post rows inside the pre row
space (`identifies` true, counterfactual equal across representatives) and one
with a generic post row (`identifies` false, counterfactuals differ).

## 29. Regularisation is a filter, and its parameter is part of the estimand

On the SVD of the design, every regulariser in use multiplies the component
along `v_i` by a filter factor: 0 or 1 for truncation (PCR, hard singular value
thresholding), `sigma_i^2 / (sigma_i^2 + mu^2)` for ridge and Tikhonov
("called filter factors or, in statistical applications, shrinkage factors",
B p. 346; B96 (2.7.9), p. 101; ESL p. 66 and p. 80), `(sigma_i - lambda)_+ /
sigma_i` for nuclear-norm shrinkage (HMLZ Theorem 1, p. 3370), and soft
thresholding for the lasso on an orthonormal design (ESL Table 3.4, p. 71). An
iteration stopped early is one more: Landweber after `k` steps "is roughly
similar to truncating the SVD expansion for sigma_i <= k^(-1/2)", and LSQR and
CGLS share the semiconvergence (B p. 733); "it is essential to stop the
iterations after the optimal number of steps" (B96 p. 316). PLS after `m` steps
is the `m`-th conjugate gradient iterate (ESL p. 82).

So the family, the parameter (`k`, `mu`, `lambda`, the iteration count) and the
rule that chose it are fields on the result, and the default is documented as a
choice. The rules are not interchangeable: the discrepancy principle needs the
noise level; GCV needs none, "can have a very flat minimum" (B p. 347) and
cancels for small `mu` (B96 p. 213); the L-curve corner is a heuristic (B96
p. 102); cross-validation with the one-standard-error rule picks the simplest
model within one standard error (ESL p. 244). A cumulative-variance threshold
sums squared singular values; a sum of singular values is a different rule (BP
uses both, pp. 855 and 869-870), and the result names which. A stopping index
chosen to regularise is distinct from a convergence test (rule 7) and is
reported as a parameter.

Failure it prevents: an estimate that changes with an undocumented default (a
`1e-5` step test, a `0.95` energy threshold) while the result reports the
number as if it had none. The truncation error is computable and reported with
the rank: `sigma_{k+1}` in the spectral norm and the tail sum of squares in the
Frobenius norm (B Theorem 2.2.11, p. 239), unique only when `sigma_k >
sigma_{k+1}` (BP p. 854).

Test pattern: the parameter and its rule appear on the result; a mutant that
changes the default threshold changes a reported field; for a truncation, the
reported tail equals the discarded singular values of the matrix truncated.

## 30. Remove a row with Q in hand, or refit; never downdate R alone

Refit loops (leave-one-out, placebo, conformal, forward selection) change one
row or column at a time. "Deleting a column and adding a row are 'easy'
operations, whereas adding a column and deleting a row are more delicate ...
the modified matrix may become rank deficient" (B96 p. 132; B p. 305). The
rules:

- Delete a row with the orthogonal factor available (B96 Section 3.2.6,
  p. 137; BP Section 28.2.2, pp. 425-427), or refit from the data. Downdating a
  Cholesky or `R` factor alone is at best backward stable for `R'R`, not for
  the data, and can return `R = 0` without an error (B96 pp. 141-142); the
  LINPACK downdate is the less stable variant (B p. 311); hyperbolic downdating
  has "clearly numerical issues if |x1| is just slightly greater than |x2|"
  (GvL p. 340).
- In a sliding window, add the new row before deleting the old (B p. 306).
- Sherman-Morrison and Woodbury updates of an inverse "will not always lead to
  stable methods" (B96 p. 129). The recursive least squares deletion divides by
  `1 - w'(A'A)^{-1} w`, one minus the deleted row's leverage (B96 (3.1.19),
  p. 131; derived); a leverage near 1 is a rank drop (rule 6) and is reported,
  not divided by.
- Cost: with `Q` stored explicitly a rank-one QR update costs `O(m^2)` against
  `O(m n^2)` for a refactorization, `O(m n)` with the Gram-Schmidt factor, "if
  n is small updating may still be more expensive than computing the
  decomposition from scratch" (B96 pp. 130, 133); an exact SVD update costs as
  much as a new SVD (B96 p. 145). For linear smoothers the leave-one-out
  residuals need no refit: `(y_i - yhat_i) / (1 - S_ii)` (ESL (7.51), p. 244;
  GvL (6.1.16)-(6.1.17), p. 308), which does not apply to a simplex or L1 fit.

Failure it prevents: a refit distribution built from factors that are not the
factors of any nearby data, with no error raised. No code on `main` downdates;
the rule binds the per-call work #1065 plans for refit loops.

Test pattern: the downdated solution against a refit from the data on a case
where the deleted row dominates (B96 Example 3.3.1), and a property over random
designs that the two agree within the rule 13 tolerance for the refit's
condition number.

## 31. Cross-validation contains every step that reads the outcome

"With a multistep modeling procedure, cross-validation must be applied to the
entire sequence of modeling steps. In particular, samples must be 'left out'
before any selection or filtering steps are applied" (ESL p. 245); screening on
all samples gave a cross-validated error near 3 percent where the true error was
50 percent (ESL pp. 245-246), and "the model must be completely retrained for
each fold" (p. 247). Unsupervised steps (a rank rule on the donors alone) may
precede the folds; a rank, penalty or donor choice that reads the treated
unit's outcome may not.

Three corollaries. Standard errors computed on a selected model "are not valid,
since they do not account for the search process" (ESL p. 60). The complexity
of a selected fit exceeds the count of what was selected: "by choosing the
best-fitting model with d inputs, the effective number of parameters fit is
more than d" (ESL p. 231), so a count is not a degrees-of-freedom in Cp, AIC,
BIC or GCV. And a cross-validated error estimates the procedure's average error
over training sets, not the error of the weights fitted to this panel (ESL
p. 242, p. 254).

Test pattern: assert that the selection function runs inside the fold loop (a
counter, or a mutant that hoists it out and is killed by a null simulation
whose cross-validated error then falls below its true error).

---

# Testing a solver: the floor of rounding

The four levels and two instruments of `agents_tests.md` apply. These
refinements are specific to numerical code.

11. A tolerance in a test comes from an error model or a measured spread, never
    from a guess. When the data are stored at level `c`, each entry errs by
    `eps c / 2`, a residual by `eps c (1 + |beta|_1) / 2`, and a pairwise
    objective by `h(.)^2` times that; the solver's point, scored on the
    unshifted data, may exceed the optimum by twice the objective's error. Two
    guessed bounds failed under `hypothesis`; the derived one has held.
12. A tolerance is absolute in the units of what it bounds when the reference
    can be zero. An exact fit has optimum zero, computed as rounding noise; a
    test relative to the optimum alone fails there. Add the rounding floor of
    the objective's terms.
13. A tolerance on an inverse problem includes its condition number. A quantile
    is as accurate as its tail divided by `kappa = P / (x pdf)`; the bound is
    `c eps (1 + kappa)` with `c` measured (6.0 at worst over 15,000 points, so
    12).
14. Pin every counterexample `hypothesis` finds with `@example`, so the kill
    does not depend on a local example database. The same holds for a mutant:
    a kill that depends on a random draw is not a kill. The shared-rate-scale
    mutant was killed in one run and survived the next; its witness turned up
    at the 229th seed of a direct search. Search against the mutated code (exec
    the mutated source as its own module) and pin the witness. A re-executed
    module defines its own warning and exception classes, so match the
    witness's warnings by name or base class, not by the original's class
    object.
15. Prefer an exact reference to another floating-point implementation:
    `fractions.Fraction` for sums of products, a closed form (`chi2_2` is
    exponential), brute-force enumeration for small discrete cases, `mpmath`.
16. Each rule a change relies on gets a mutant in `tools/mutation/targets.toml`
    that reintroduces the old defect: the floor, the exact zero, the uncentred
    sum, the silent counter. The RRMA targets are `rrma-study-*-numerics`.

---

# The backlog

A read-only sweep of `mlsynth/` (tests excluded) at commit `ef9da44`, every hit
read by hand. The classified counts below come from reading; the greps after
them are the rough check and over-count.

| Rule | Pattern | Count |
| --- | --- | --- |
| 3 | tolerance floors, `max(1, x)` form | 15 |
| 3 | tolerance floors, `tol * (1 + abs(x))` form | 12 |
| 3 | absolute literal thresholds in loops | 44, plus 12 unscaled `tol` |
| 4 | `== 0.0` / `!= 0` on computed values | about 10 of 43 |
| 7 | loops with no cap | 2 |
| 1, 7 | caps reached with a silent return | 7 |
| 6 | normal-equation solves (`solve`/`inv`/`cholesky` of `X'X`) | 29, plus 9 `pinv(X'X)` |
| 1 | `cvxpy` solves | 51: 19 check `status`, about 23 only `value is None`, about 9 nothing |
| 1 | `solve_simplex_qp` callers asking for `converged` | 5 of 43 |
| 8 | `np.clip(..., 0, None)` on solver output | 56, none reported |

A second sweep, for rules 3, 6, 7 and 28 to 31, at commit `1b35b871`
(2026-11-25), read the same way: grep, then every hit on an estimate or
interval path read by hand.

| Rule | Pattern | Count |
| --- | --- | --- |
| 6 | normal-equation solves, the sweep below, re-run | 25, plus 9 `pinv(X'X)` |
| 6 | `pinv` with NumPy's default cutoff (`1e-15 sigma_1`) | 50 of 52; 1 decides a reported rank |
| 6 | `lstsq(rcond=None)` with the rank discarded | 74 of 75 |
| 28 | a tiny ridge added to a Gram as a rank fix | 11 |
| 28, 29 | a rank chosen on one matrix and applied to another | 1 (shared by CLUSTERSC, SI, SNN) |
| 1, 7 | iterations ending at a cap with no flag on the result | 3, plus 1 that reports only the count |
| 3 | absolute thresholds in outcome or covariate units on an estimate or interval path | 4 |
| 30 | QR or Cholesky downdating | 0 |

Representative sites, for whoever takes the backlog:

- Floors: `utils/solvers/simplex.py:274` (`prev - obj <= tol * max(1.0,
  abs(prev))`, a squared residual, so small-scale data make it absolute);
  `utils/solvers/nnls.py:92`
  (`tol = max(tol, 1e-12)` floors a scaled tolerance);
  `utils/bilevel/mscmt.py:268`; `utils/bilevel/determine_v.py:308`.
- Silent caps: `utils/solvers/simplex.py:73` defaults `warn` to False and 8 of
  its 9 callers leave it there; `utils/solvers/nnls.py:114-116`.
- Unchecked status: `utils/ssc_helpers/weights.py:89` (then clips and
  renormalises at 90-92); `utils/syndes_helpers/certificate.py:114` checks only
  `value is None`, so `optimal_inaccurate` reaches a lower bound that an inexact
  relaxation can invalidate.
- Normal equations: `utils/solvers/ridge_augment.py` `_residualized_matching`
  (`inv(Zc.T @ Zc)`, reused twice); `utils/rolldid_helpers/pipeline.py:56`.
- Absolute floor on a standard deviation (rule 3): `utils/solvers/ridge_augment.py`
  `build_matching` (`sdz < _EPS`), the covariate path of the ridge layer.
- No cap: `utils/drosc_helpers/estimation.py:75` (`while beta is None:`).
- Ridge sized by the level (rule 5): `utils/solvers/active_set.py`
  `solve_simplex_qp_least_norm` scales by the raw design's column energy (the
  rule 5 measurement). Its caller `utils/scmo_helpers/solvers.py:51` then clips
  and renormalises (rule 8).
- Equality by weighting, with the weight sized by the level and the repair
  unreported (rules 5, 8): `utils/ctsc_helpers/estimate.py:64`.
- Rank deficiency refused where a representative is defined (rule 6):
  `utils/bilevel/exact/lsq.py` `_lsi` refuses a face with more free weights
  than rows. `utils/bilevel/exact/hull.py` refuses a hull of deficient affine
  rank without naming the predictors that carry the dependency (rule 1). The
  condition cap `COND_MAX = 1e4` that sat beside them is gone: each face fit is
  certified instead (rule 6, the Basque measurement).

- Rank chosen on one matrix, truncation on another (rules 6, 28, 29):
  `utils/pcr/rank.py` `select_rank` picks `k` on the column-standardised
  (centred, scaled) donor matrix, and `utils/pcr/core.py` `hsvt` truncates the
  raw one, whose first singular direction on level data is the level. On 300
  synthetic panels (`T0 = 30`, `J = 40`, level 100, three factors) the raw
  rank-`k` truncation's time space kept a median 95.3 percent (minimum 77.8)
  of the factor signal, the centred matrix at the same `k` 99.9 percent
  (minimum 98.9) (measured 2026-11-25).
- Cap with no flag and a step test where a certificate exists (rules 1, 7):
  `utils/mcnnm_helpers/completion.py` `mcnnm_fit` (cap 400, relative step
  `1e-5`) and `_two_way_fe` (cap 100, absolute `1e-7`); the measurement is under
  rule 7. `utils/clustersc_helpers/rpca/hqf.py:173` stops on an absolute RMSE
  decrease of `1e-6` with a cap of 1000 reported only as `iterations`;
  `utils/gsynth_helpers/ife.py:221` stops on an absolute `||delta beta||` and
  reports `n_iterations` with no flag.
- Absolute thresholds in data units (rule 3):
  `utils/clustersc_helpers/pcr/inference.py` `_threshold` zeroes residual
  entries below `1e-10` (a port of the reference `var.py`); the standard errors
  divided by the data scale agree to six digits at scales 1 and `1e-6`, move by
  13 percent at `1e-10` and are exactly zero at `1e-12` (measured
  2026-11-25). `utils/gsynth_helpers/ife.py:203` drops a covariate whose
  absolute sum is below `1e-5`.
- Default `pinv` cutoff deciding a reported rank (rule 6):
  `utils/clustersc_helpers/pcr/inference.py` computes
  `R = round(trace(Y0 pinv(Y0)))` on the rank-`k` reconstruction. Its trailing
  singular values were at most `1.9e-16 sigma_1` over 2,000 draws (`T0` 15 to
  40, `J` 16 to 100) against the `1e-15` cutoff, and `R = k` held in 400 of 400
  calls: safe at these sizes by a factor of five, and the rank is already known
  from the SVD taken beside it.
- Different representatives for a point and its placebo distribution (rules 6,
  28): `utils/ssc_helpers/estimation.py` solves the point estimate by LU on the
  Gram, falling back to `lstsq` only on an exact `LinAlgError`, and the placebo
  draws by `pinv` of the same Gram. They part only when the Gram is singular,
  which the paper's Assumption 2.1 excludes and `gram_min_eig` records.
- Tiny ridges (rule 28): `utils/rrsc_helpers/pipeline.py:98` (`1e-8`,
  absolute, on the fixed-N effect regression), `utils/fma_helpers/fit.py:67`
  (`1e-10` times `max(mean diagonal, 1)`, a floored relative ridge, on the
  counterfactual's normal equations), and nine more listed in the 2026-11-25
  audit.

Already right, and the models to copy:

- `utils/weights/solve.py` `_affine` eliminates the sum-to-one row through a
  null-space basis (rule 6's equality bullet), and `utils/weights/solution.py`
  `unique` and `identifies(B_post)` are rule 28's uniqueness verdict and
  inclusion test.
- `utils/pda_helpers/fs/estimation.py` ranks forward-selection candidates by
  Elden's criterion, the reduction in residual sum of squares from each
  orthogonalised column (B pp. 312-313), and refits the near-tied leaders
  exactly; refit loops elsewhere refit from the data (rule 30).

- `benchmarks/studies/tbr_bnb/certificate.py`: `hull_bound_float`,
  `joint_bound_float` and `lambda_min_lower_float` verify in floating point with
  the a priori bounds of rule 19, and `hull_clears` / `joint_clears` call the
  exact certificate only when the float one falls short (rules 17 to 19, 27).
- `mlsynth/utils/marex_helpers/branch.py` `lambda_min_lower`: a floating-point
  Cholesky factor, its residual computed exactly, Weyl and Gershgorin (rules 21,
  22).

- `utils/bilevel/mscmt.py:339-341`: counts unconverged inner solves into
  `metadata["inner_unconverged"]` and warns.
- `utils/weights/solution.py`: a result whose `kkt_residual` and `status` are
  recomputed from the returned weights and scaled so rescaling the data does not
  move them.
- `utils/solvers/sunny.py:258-266`: checks `res.success` and warns with the
  solver's message.
- `utils/bilevel/penalized.py:243-258`: a solver failure becomes
  `MlsynthEstimationError`, not a fallback.
- `utils/pcr/core.py:100-108`: singular values cut relative to the largest, and
  rank 0 raises.
- `utils/solvers/active_set.py` `solve_simplex_qp` (#829, #960): the pivot loop
  runs on the program in difference form at unit scale (a power-of-two division,
  so data that differ by a power of two give the same bits), and `converged` is
  `simplex_kkt_error` on the caller's data, at most `CERTIFY_TOL`. Before it, a
  scale of 1e-6 or a level of 1e8 left the loop 37% above the optimum, reported
  converged at 1e-6. A caller that does not read `info` gets
  `SimplexQPUncertifiedWarning`.
- `utils/solvers/ridge_augment.py` (#960): the ridge correction through the
  thin SVD of `B` (the Gram form erred by 1.2e-6 to 3.5e-6 against an exact
  rational solve at condition 1e8, the SVD by at most 8.5e-10), and a cvxpy
  fallback that reads each status, raises when no solver returns an optimum,
  and reports itself, its repairs and its certificate in a
  `SimplexFallbackWarning`.

Sweeps (documentation-level, like those in `agents_unix.md`; they over-count):

```bash
grep -rn "max(1\.0,\|max(1,\|np.maximum(1\.0," mlsynth/ --include=*.py | grep -v /tests/   # rule 3
grep -rn "\.solve(" mlsynth/ --include=*.py | grep -v /tests/ | grep -v linalg             # rule 1: then read for .status
grep -rn "np.clip(.*0.*None)" mlsynth/ --include=*.py | grep -v /tests/                    # rule 8
grep -rnE "(solve|inv|cholesky)\([A-Za-z_]+\.T @" mlsynth/ --include=*.py | grep -v /tests/ # rule 6
```

---

# What the books do not support

- No tolerance values. NW states the principle of scaling; R bounds one
  rounding. `TOL = 1e-9` and the `64 eps` floors are choices, defended by the
  units properties and the error models above, not by a citation.
- No bound on accumulated error through a pivot loop or a triangular solve.
  The dot-product and summation bounds are Rump's Theorem 2.1 (rule 19, after
  Higham's *Accuracy and Stability of Numerical Algorithms*, (3.2) and Lemma
  8.2), with underflow by Rump (2.3)-(2.4), and Cholesky's by Rump Lemma 10.13.
  Rump argues against propagating error bounds through an elimination at all
  (rule 20, Table 10.2): the bound is computed once, from the data and the
  answer. B, B96 and GvL give the least squares perturbation bounds, the
  refinement heuristic and the least distance reduction (B96 Theorem 5.2.1,
  p. 197, by citation to Lawson and Hanson), not a rounding analysis of that
  reduction; its digit loss is measured, not derived.
- No size for a tie-breaking ridge or a constraint weight. B96 gives the
  `gamma^-2` rate and the need for pivoting, not a value; the ridge and the
  `1e3` are choices, and rule 5 says what they are measured against.
- No anti-cycling rule for a bounded dual simplex, and no safeguarded Newton
  method for a scalar root (NW's Chapter 11 globalises with a merit function,
  p. 287). Brent's halving rule is named as Brent's.
- No claim that a tolerance-based certificate is a proof. A KKT residual at
  most `TOL` increases reliability (Rump p. 293) and is worded "certified to a
  relative tolerance". A proof needs either rational arithmetic or Rump's
  route: a theorem whose hypotheses are verified with every rounding bounded
  (rules 17 to 27). Floating point suffices for a rigorous inequality with room
  to spare; equality, a tie or singularity needs exact arithmetic (rule 18).
- No directed rounding from NumPy. Rump's interval algorithms switch the
  processor's rounding mode (Section 4, Algorithm 9.1); NumPy and the BLAS do
  not expose that reliably, so `mlsynth` uses the a priori bounds of rounding to
  nearest (rule 19) and rounds a final scalar in its direction (rule 27).
- No verified method for sparse or iterative linear algebra in general: "for
  general sparse linear systems such an algorithm is not known, one major open
  problem of verification algorithms", and "the field of verification
  algorithms based on iterative methods is basically a blank slate" (Rump
  p. 356). A first-order or ADMM solver's iterate is a proposal (rule 26); only
  the postprocessed bound is proved.
- No tolerance for the band where a float certificate defers to the exact one.
  Its width is the a priori error bound itself (rule 19), not a choice.
- No value for a rank tolerance. B96 and B give its form, `delta = (mn)^(1/2)
  e` with `e` the entries' error (rule 6), and `e` is the analyst's; GvL, YTT,
  ESL and BP give none ("small enough", BP p. 410). No source here justifies a
  cumulative-variance threshold for a truncation rank.
- No complete theory for stopping a Krylov iteration as a regulariser: "a
  complete understanding of the regularizing effect of Krylov subspace methods
  is still lacking" (B96 p. 316). No global optimality for alternating least
  squares: each block solve lowers the objective (GvL pp. 734-737; HMLZ
  Theorem 3, p. 3379), and a limit point is stationary only (HMLZ Theorem 5,
  p. 3382). WZLZ's estimator converges locally, from a start inside a ball set
  by unknown constants (Theorem 3.3, pp. 9-10).

Mark claims accordingly in docstrings and reviews: what a book or paper states
(with the page or equation), what was measured (with the number), and what is
inferred or derived here.

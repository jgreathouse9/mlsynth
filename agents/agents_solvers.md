# agents_solvers.md

# Standards for Optimization Solvers in `mlsynth`

This file states what a numerical solver in `mlsynth` must do, whether it is
written here (`utils/solvers/`, `utils/bilevel/`, a study's LP code) or called
from a package (`cvxpy`, `scipy.optimize`). It exists so that "this tolerance is
fine" becomes a claim with a citation and a test, not a matter of taste.

Two books supply the rules:

- Nocedal and Wright, *Numerical Optimization*, 2nd ed. (Springer, 2006), cited
  as NW with page numbers. Appendix A (floating point, conditioning, stability),
  Chapter 10 (least squares), Chapters 13-14 (simplex and interior-point
  numerics), Chapters 3 and 11 (safeguarded steps and stopping tests).
- Russinoff, *Formal Verification of Floating-Point Hardware Design*, 2nd ed.
  (Springer, 2022), cited as R with lemma numbers. What one IEEE rounding can
  and cannot do: exactness (Sterbenz, Corollary 4.16), the error of a single
  rounding (Lemma 6.45), gradual underflow (Lemma 6.101).

The rules were extracted while auditing the RRMA study's hand-written LP solvers
and distribution functions (`benchmarks/studies/rrma_ablation/`, commit
`ef9da44`). Every rule below comes with the failure it caused there, measured,
and the test that now pins it. The books give the principle; the numbers say
what breaking it cost here.

The thesis in one line:

> A solver's answer must not depend on the units of its data, and a solver must
> say what it found: certified optimum, cap reached, infeasible, unbounded, or
> not identified. Everything else in this file is how to make those two
> statements true in floating point.

---

# The rules

Status, as in `agents_unix.md`: binding means new code must follow it and
existing code that breaks it is backlog (counted at the end); adopted means this
file makes it a rule where it was not one.

| # | Rule | Status | Book basis |
| --- | --- | --- | --- |
| 1 | Every outcome is explicit | binding (invariant 7) | NW p. 57, p. 370, p. 607 |
| 2 | Optimality is certified on the original data | adopted | NW p. 370, pp. 381-382, p. 386 |
| 3 | Tolerances are relative to what they compare, with no floor | adopted | NW pp. 26-27, p. 131 |
| 4 | A computed zero is a zero only to the rounding of its terms | adopted | R Lemma 6.45, Lemma 6.101 |
| 5 | Do not difference what you can avoid differencing | adopted | NW p. 615 |
| 6 | Do not square the condition number; name rank deficiency | adopted | NW pp. 251-253, p. 610, p. 617 |
| 7 | Iterations are safeguarded and end | adopted | NW p. 37, p. 61, p. 62 |
| 8 | Repairs to a solver's output are reported | binding (invariant 7) | none; invariant 7 |
| 9 | Form quantities so the failure mode is underflow | adopted | NW p. 616 |
| 10 | Degeneracy: perturb to pivot, evaluate and certify on the original | adopted | NW pp. 381-382 |

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
- Centre before a sum whose weights sum to zero: `a'e = a'(e - c)` when
  `sum(a) = 0`.
- Near a root, form `log(f / t)` as `log1p((f - t) / t)`, not `log f - log t`.
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
- A linear solve by LU with partial pivoting is backward stable, and its forward
  error grows with the condition number (NW p. 617, A.32). Either estimate the
  condition number (from the triangular factor, NW p. 610) or let a certificate
  computed from the result (rule 2) measure it, and say which.
- Rank is decided by the usual threshold, `max(n, p) eps` times the largest
  singular value, measured against the size of the entries when the matrix has
  been centred: a column of ones centres to noise near 1e-16, which a threshold
  relative to the centred matrix alone counts as a direction.
- A rank-deficient problem has a family of minimisers (NW p. 253). Name it,
  choose a documented representative (the minimum-norm point of the face), and
  report it (rule 1).

What it looked like. RRMA's Algorithm 1 on the Basque data puts 10 donors in one
group at `m = 6` and fits it on windows of 7 to 10 pre-periods: the pairwise
objective identifies `beta` only on a 6-to-9-dimensional subspace. HiGHS had
returned an arbitrary vertex; the NumPy solver raised an opaque error. `fit()`
now solves in the identified coordinates and warns.

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

Test pattern: count evaluations over a grid and bound the worst; corrupt the
derivative (multiply it by 1000) and assert convergence to the right root.

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

Representative sites, for whoever takes the backlog:

- Floors: `utils/solvers/simplex.py:274` (`prev - obj <= tol * max(1.0,
  abs(prev))`, a squared residual, so small-scale data make it absolute);
  `utils/solvers/active_set.py:287`; `utils/solvers/nnls.py:92`
  (`tol = max(tol, 1e-12)` floors a scaled tolerance);
  `utils/bilevel/mscmt.py:268`; `utils/bilevel/determine_v.py:308`.
- Silent caps: `utils/solvers/simplex.py:73` defaults `warn` to False and 8 of
  its 9 callers leave it there; `utils/solvers/nnls.py:114-116`.
- Unchecked status: `utils/ssc_helpers/weights.py:89` (then clips and
  renormalises at 90-92); `utils/syndes_helpers/certificate.py:114` checks only
  `value is None`, so `optimal_inaccurate` reaches a lower bound that an inexact
  relaxation can invalidate.
- Normal equations: `utils/solvers/ridge_augment.py:479`
  (`inv(Zc.T @ Zc)`, reused twice); `utils/rolldid_helpers/pipeline.py:56`.
- No cap: `utils/drosc_helpers/estimation.py:75` (`while beta is None:`).

Already right, and the models to copy:

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
- No bound on accumulated error through a pivot loop, a triangular solve or a
  dot product. R stops at single operations; the standard dot-product bound is
  Higham's (*Accuracy and Stability of Numerical Algorithms*), which was not
  among the sources read.
- No anti-cycling rule for a bounded dual simplex, and no safeguarded Newton
  method for a scalar root (NW's Chapter 11 globalises with a merit function,
  p. 287). Brent's halving rule is named as Brent's.
- No claim that a tolerance-based certificate is exact. Exactness would need
  rational arithmetic; the wording is "certified to a relative tolerance".

Mark claims accordingly in docstrings and reviews: what a book states (with the
page), what was measured (with the number), and what is inferred.

# The simplex-QP audit, and why its classifier reads algebra

`tools/simplex_qp_audit.py` answers one question for every cvxpy problem in the
library: does this site solve

    min ||A - B w||^2   subject to   w >= 0,  sum(w) = 1

so that `bilevel/active_set.py::solve_simplex_qp` could replace it? The answer
is an allowlist, pinned by name in `mlsynth/tests/test_simplex_qp_audit.py`, so
a new cvxpy simplex problem has to be classified by a person instead of joining
a count nobody re-derives.

Of the 40 problems carrying a sum-to-one constraint, 24 are that program: 14
swap the solver call with no other change, and 10 need the caller to reshape
the data first. One minimises something else, one cannot be read at all, and 14
have a feasible set that is not the simplex.

## Six ways to misread a solve

Each of these produced a wrong verdict on this library, and each is a rule in
the module now.

A text window around `sum(w) == 1` cannot see a constraint list assembled in
pieces, so the list is followed through its name, its `+=` and its `.append`.

Reading only the constraint list misses non-negativity declared as
`cp.Variable(n, nonneg=True)`, which is how most of this library writes it.
Reading both sources changes the count by more than a factor of two.

Following a name by walking the whole module and keeping the last assignment
crosses function boundaries. SpSyDiD binds `objective` and `constraints` in two
functions, and its time-weight solve was read as its unit-weight one: wrong
variable, wrong constraint list, wrong objective, and a verdict that happened
to agree, which is what kept it invisible. DSC binds `w` in two functions, one
declared non-negative and one not, and the audit reported
`solve_sum_to_one_weights` as the probability simplex. That is the expensive
direction of this error: the audit saying "swap the solver here" about weights
that may go negative.

Matching any callable whose name ends in `Problem` matches a class. LAXSCM
caches compiled programs behind `_PenalizedProblem` and `_RelaxedProblem`, and
both constructors entered the audit as solves with nothing to read.

A constraint appended inside an `if` holds on one branch. SHC appends `w >= 0`
under one, so the feasible set is read twice: whole, for what it can contain,
and without its conditional pieces, for what it always contains. Only the
second establishes non-negativity.

The sixth is the reason the module is mostly a small expression algebra. An
objective matched as text is matched on its syntax, and syntax is not the
program. `cp.norm(r, 2)` and `cp.sum_squares(r)` have the same minimiser.
`quad_form(r, V)` is a row scaling, `quad_form(w, Q)` a factorisation. A ridge
is extra design rows carrying no target. An intercept is profiled out by
centring. A penalty on a constraint the program already enforces is a constant
on the feasible set. Ten sites were filed as "a different objective" on those
grounds, and all ten are the simplex least-squares program.

So the objective is split into a sum of terms, and each term is matched against
what the solver can carry: a squared residual, a term linear in the weights
(the solver's `linear` argument), a ridge the design absorbs, a penalty that is
zero on the feasible set. A site is eligible when exactly one term is the fit
and every other term is one of the rest, and the reshaping the caller needs is
recorded per site as `transform`.

The three answers are also three: eligible, a different objective, and an
objective the audit cannot read. The third used to be folded into the second,
which asserted that four sites solve something else when nobody had checked;
three of the four turned out to be the simplex least-squares program.

## The ten transforms, checked

`agents/spike_simplex_audit_transforms.py` builds each site's cvxpy program on
eight random panels, applies the transform the audit names, solves through the
active set, and compares. Weight gap is the largest absolute difference in `w`;
objective excess is the active set's objective minus cvxpy's, relative, at the
worst panel.

| Site | Transform | Weight gap | Objective excess |
| --- | --- | --- | --- |
| `clustersc_helpers/pcr/convex.py:60` | square the objective | 5.03e-06 | -4.01e-12 |
| `cscm_helpers/engine.py:124` | scale rows by the square root of the metric | 8.68e-09 | +1.50e-13 |
| `dscar_helpers/weights.py:72` | the same, and the sum-to-one penalty is zero on the feasible set | 2.22e-08 | +7.30e-14 |
| `fast_scm_helpers/fast_scm_bb_helpers.py:192` | factor the Gram as R'R, take B = R | 7.10e-07 | -3.53e-14 |
| `hsc_helpers/formulation.py:157` | the same, recovering the target from the linear term | 1.50e-06 | +0.00e+00 |
| `mlsc_helpers/crossval.py:131` | augment the design with a multiple of the identity | 9.53e-09 | +6.38e-14 |
| `mlsc_helpers/crossval.py:154` | augment with the penalty's square-root factor | 9.66e-09 | +6.24e-15 |
| `mlsc_helpers/optimization.py:129` | the same | 1.76e-08 | +3.18e-15 |
| `spsydid_helpers/weights.py:54` | centre the design and the target | 2.76e-07 | +2.53e-14 |
| `spsydid_helpers/weights.py:136` | centre, and augment with a multiple of the identity | 1.78e-05 | +3.51e-15 |

The two columns disagree, and the objective is the one to read. SpSyDiD's
unit-weight solve has the widest weight gap in the table and an objective
excess of 3.5e-15: the program has a flat direction there, so the two
minimisers differ in `w` and agree in what `w` achieves. On two sites the
active set reached a lower objective than cvxpy did. The largest excess
anywhere is 1.5e-13 relative.

The transforms are algebra, and where they have a citation it is:

- Zou and Hastie (2005), Lemma 1: an L2 penalty is `[X; sqrt(lambda) I]` with
  the target padded by zeros, exact under a rank condition on the augmented
  design. `mlsynth/utils/bilevel/ridge_augment.py` carries the same reduction.
- A monotone transform of the objective preserves the argmin, which is why
  `cp.norm(r, 2)` and `cp.sum_squares(r)` pick the same weights. CLARABEL's
  tolerance on the unsquared norm is looser near a flat optimum, and that is
  the 5e-06 in the first row.

## Running it

```bash
python tools/simplex_qp_audit.py                    # the classification
python agents/spike_simplex_audit_transforms.py     # the table above
pytest mlsynth/tests/test_simplex_qp_audit.py -q    # the pins
```

The pins are keyed `(file, variable)` with a count, not line numbers, so an
edit above a site does not break them. Two files appear in more than one list:
`bilevel/penalized.py` is eligible and migrated, because its two programs are
the same shape and only `penalized_weights` moved; `dsc_helpers/weights.py` is
eligible and ineligible, because its two solves are the module's two weight
options and only `_refine_exact` takes the simplex.

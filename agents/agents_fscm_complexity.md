# FSCM's forward scan: what its cost is made of, and what is left to take out

A spike, prompted by a question: is FSCM's complexity already as low as it
goes? It is not, and the answer separates into one thing that is provably
irreducible, one that is measured here, and three that are not yet measured.

`agents/spike_fscm_pruning.py` is the measurement.

## Why the scan cannot be an inner product

fsPDA (Shi and Huang) selects by projecting onto the span of the chosen donors.
Projection onto a subspace is linear, so the gain from adding donor `j` has a
closed form: residualise `j` on the current set and the reduction is
`(r'x_j)^2 / ||x_j||^2`. Every candidate's gain comes out of one Gram, the Gram
updates by rank one as donors enter, and no problem is ever re-solved. Selection
is screening against an exact criterion.

FSCM projects onto the convex hull. That projection is not linear in the target,
so there is no closed-form gain, no rank-one update of the answer, and the
optimum on the enlarged set is not a function of the optimum on the smaller one.
Each candidate needs its own quadratic program. The scan is `O(J^2)` solves
where fsPDA's is `O(J^2)` inner products, and that gap is the constraint the
method exists to impose.

This is visible in the solver. `solve_simplex_qp` forms `G = B'B` and uses it at
exactly one line, `g = G @ w - c`, the dual test deciding which pinned donor to
release. The primal solve never touches it: the free set is factored from the
design directly through `gelsy`, which is what keeps the condition number from
being squared. So the Gram is present, and it prices; it does not select.

The evidence that the gap is real and not an artefact of how the scan is
written: a Frank-Wolfe shortlist, the simplex analogue of fsPDA's screening,
reproduces the exact selection on 2 percent of random panels at shortlist 5.
For OLS the screening gain is the exact residual reduction. On the simplex it is
only the initial slope.

## What can be taken out: bounded pruning

The minimum over the simplex on `S + {j}` is bounded below by the minimum over
any relaxation of it, and two relaxations have closed forms from the Gram. If a
candidate's bound already exceeds the best score at that step, its program
cannot win and need not be solved.

This is a bound and not an approximation, which is the difference from the
Frank-Wolfe attempt: where a candidate strictly wins, pruning cannot change
which one.

| panel | relaxation | solved | pruned |
| --- | --- | ---: | ---: |
| Proposition 99 | unconstrained | 640 / 741 | 13.6% |
| Proposition 99 | affine, `1'w = 1` | 601 / 741 | 18.9% |
| Basque | unconstrained | 108 / 136 | 20.6% |
| Basque | affine, `1'w = 1` | 99 / 136 | 27.2% |

The affine relaxation is tighter because it keeps the constraint the simplex
binds through. Both are cheap: one small solve against the Gram, no access to
the design.

Nineteen to twenty-seven percent is real and is not a change of order. The hull
does enough work that relaxing it loosens the bound considerably, which is the
same fact as the paragraph above seen from the other side.

## What the measurement turned up instead

Writing the invariance check for the pruning is what exposed this. The two
relaxations selected different donors from step 7 on Proposition 99, which two
valid lower bounds cannot do — so either a bound was wrong or something else
was.

Something else was. At step 7 the fit has saturated and every remaining
candidate scores identically:

| panel | saturates at | candidates exactly tied there | `optimal_size` |
| --- | ---: | ---: | ---: |
| Proposition 99 | step 7 | 32 of 32 | 3 |
| Basque | step 4 | 13 of 13 | 2 |

Not nearly tied. Identical to the last digit — `52.129583432871584` for all 32,
a second-best gap of `0.00e+00`. The treated unit is already in the hull of the
donors selected by then, so no further donor can change the fit at all.

So FSCM's greedy order is a well-defined object for six steps on Proposition 99
and three on Basque, and past that it is whichever candidate the loop happened
to evaluate first among thirty that are indistinguishable. The estimator
computes and reports all 38 steps in `selection_path`.

This also closes an open question from the root-cause ladder on #622, which
found the greedy order reordering from step 6 under a 1e-5 change in the solver
and put it down to near-ties at depth. It is not near-ties. It is exact
saturation, and the divergence begins precisely at the first tied step.

Both panels choose a size well inside the determined region, so neither
estimate moves. What the tie reaches is how `selection_path` reads -- and, as
below, the donor count itself.

### The tie is in sample only

"Nothing reported depends on this" holds for the in-sample score and fails for
the number the estimator picks. `optimal_size` is the argmin of the rolling
cross-validation curve, and that curve refits on windows shorter than the
pre-period, where the added donor is not rejected. So it keeps moving across
sizes the in-sample score cannot separate at all:

| size | in-sample SSE | rolling CV |
| ---: | ---: | ---: |
| 6 | 52.129583433 | 2.816021810 |
| 7 | 52.129583433 | 2.892861947 |
| 8 | 52.129583433 | 2.894976120 |

Running past saturation therefore lets evaluation order reach the chosen donor
count through the CV curve. That makes the stop a correctness change and not
the speed win this note first called it. #628 implements it; both panels still
choose three and two, so the estimates are unchanged and the exposure was to
the selection, not to these two results.

## Not measured

Three further reductions, none of which change the order:

- **Gram reuse.** Every candidate at step `k` shares the first `k` columns, so
  its Gram is a submatrix of one `X'X` maintained across the sweep.
  `solve_simplex_qp` rebuilds `B'B` per call at `O(T0 k^2)`. Free to take, grows
  with the pool.
- **Rank-one updates.** Each candidate's design is the incumbent plus one
  column, so its factorisation is a QR column update at `O(T0 k)` against
  `O(T0 k^2)` to refactor. The rolling CV is the same on the other axis, where
  nested windows differ by one row.
- **Stopping at saturation.** The scan runs to `J` and the last thirty-two steps
  on Proposition 99 are solving programs whose answer is already known to be the
  same. Stopping when the best score stops improving would remove them outright,
  and it is a larger saving than the pruning above. It changes what
  `selection_path` contains, which is a decision about the reported object and
  not only about cost -- and, per the correction above, about which donor count
  is chosen.

## Recommendation

Take the saturation stop first. It is the largest of the four and the
simplest, and the steps it removes carry no in-sample information. Leaving them
in is not free, though: the CV curve still separates them, so evaluation order
reaches `optimal_size`. It needs a decision about `selection_path`'s contract,
which is why it is a recommendation and not a patch.

Take Gram reuse and the rank-one updates second. Both are mechanical, neither
touches the answer, and they compose with everything else.

Take bounded pruning last, or not at all. It is the most interesting of the four
and the least valuable: a fifth of the programs, at the cost of a second code
path through the scan, in a scan that a saturation stop would already have
shortened by four fifths.

Do not revisit screening. The 2 percent figure settles it, and the reason is
structural.

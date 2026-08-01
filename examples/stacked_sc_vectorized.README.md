# Batching a stacked synthetic-control design

A note on the technique in `stacked_sc_vectorized.py`, which solves the 566
per-county synthetic controls of allsynth Example 12 as six matrix problems
instead of 566 solver calls. 60 seconds to 3.9. It is worth writing down because
the trick is not specific to this panel — but the conditions it needs are real,
and they are easy to break without noticing.

## What it is

A stacked (event-study) synthetic-control design fits one donor-weight vector per
treated unit: treated unit *j*, adopting at *g*, gets its own convex combination
of the never-treated donors, and the per-unit gaps are then averaged on a common
event clock. Written as a loop that is *ι* independent constrained least-squares
problems.

It is not *ι* problems. It is one problem per adoption cohort, with many
right-hand sides. Every treated unit in cohort *g* solves

```
min_w  || A_g w - b_j ||²      s.t.  w ≥ 0,  1'w = 1
```

against the same `A_g` — the donor pre-treatment block — and differs only in
`b_j`, its own pre-treatment path. Stack the `b_j` as columns of `B_g` and the
whole cohort is a single multiple-right-hand-side program whose solution is a
weight matrix `W_g`, one column per treated unit.

## How

Generate the event-time index up front, `e = year - super_year`, and group by
cohort. Then per cohort:

1. Form the Gram matrix `G = A_g'A_g` (39 × 39 here) and the step size
   `1 / (2 λ_max(G))` — once, not once per unit.
2. Run accelerated projected gradient (FISTA) on all columns at once. Each
   iteration is one `(39, 39) @ (39, N_g)` matmul for the gradient, plus a
   projection of every column onto the probability simplex.
3. Recover the counterfactuals for the entire cohort with a second matmul,
   `D_g W_g`, and subtract to get the gap block.

The only piece that needs writing is the column-wise simplex projection. The
scalar version in `mlsynth/utils/bilevel/simplex.py` sorts, takes a cumulative
sum, and finds the last index satisfying a threshold condition; the batched form
does the same along `axis=0` and locates the threshold per column with a single
`argmax` over the reversed condition array. No Python loop over units anywhere.

One implementation detail that is not optional here: judge convergence on the
objective, not on the step norm. When `A_g` is rank deficient the iterate keeps
drifting along the optimal face long after the objective has settled, so a
step-norm rule never fires and the solver silently runs to `max_iter`.

## Why it works here

Three conditions have to hold. All three do in this design, and each is worth
checking before reaching for the technique elsewhere.

Every treated unit faces the same donor pool. The comparison group is the
never-treated set, shared by construction. Example 12 does restrict each county's
donors to other commuting zones, which would break this — but the rule binds for
23 of 566 counties, each losing one of 39 donors, and moves the aggregate by
0.001pp. A restriction that bound seriously would force a per-unit design matrix
and the batching would be lost.

The donors' transformation depends on the cohort, not on the unit. This is the
subtle one, and it is what makes the event-time reshape the right move.
allsynth's `transform(..., normalize)` indexes every series — treated and donor
alike — to the treated county's final pre-treatment year. That looks per-unit,
but the base year is *g − 1*, a property of the cohort. So the donor block takes
six distinct values, not 566. Any per-unit transformation that is genuinely a
function of the unit (say, scaling donors by a distance to that specific unit)
would break this.

The per-unit information enters only through the right-hand side. Outcome-only
matching puts the treated unit's path in `b_j` and nothing else. Covariate
matching in the `synth` tradition does not have this property: the predictor
weighting matrix *V* is chosen per treated unit by nested optimisation, so the
effective Gram matrix `A'VA` differs unit by unit and the shared factorisation
is gone. A common *V* across the cohort restores it. Worth flagging because
Example 12 does specify a predictor list; the runs here are outcome-only.

Note that these are conditions on the *design*, not on the estimator. SDID
satisfies them too, and satisfies more: its time weights and its ridge penalty
`zeta` are fit on the donors alone, so they are identical for every treated unit
in a cohort — verified in section 4 of the script, where cohort 1996 returns
`lambda = [0 0 0 0 0 1]` for every county in it. A batched SDID is one
time-weight solve per cohort plus one multi-RHS unit-weight solve, with the
intercept concentrated out by centring the pre-period rows and `zeta` folded into
the Gram diagonal.

## What it costs

Nothing, in aggregate. The event-time path is unchanged:

| | −5 | −4 | −3 | −2 | −1 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batched (3.84s) | 0.01 | 0.11 | 0.03 | 0.05 | −0.00 | 0.24 | 0.05 | −0.24 | −0.53 | −0.83 | −1.10 |
| looped (59.64s) | 0.01 | 0.11 | 0.03 | 0.05 | −0.00 | 0.25 | 0.04 | −0.25 | −0.54 | −0.85 | −1.14 |

Per county, though, the two disagree by up to 6.7pp — and that is not a bug in
either. Each cohort has 5 to 10 pre-treatment periods against 39 donors, so
`A_g` has a null space of 29 dimensions or more:

| cohort | treated | T_pre | donors | rank(A_g) | null space |
|---|---|---|---|---|---|
| 1995 | 96 | 5 | 39 | 5 | 34 |
| 1996 | 99 | 6 | 39 | 6 | 33 |
| 1997 | 76 | 7 | 39 | 7 | 32 |
| 1998 | 89 | 8 | 39 | 8 | 31 |
| 1999 | 92 | 9 | 39 | 9 | 30 |
| 2000 | 114 | 10 | 39 | 10 | 29 |

Moving the weights along that null space changes the post-treatment prediction
without touching the pre-treatment fit, so the optimum is a face rather than a
point and two solvers that both reach it land in different places. The average is
pinned down; no individual county's weight vector should be read as *the*
synthetic control for that county.

A ridge restores uniqueness and converges far faster, but it is not a solver
knob — it shrinks toward uniform weights and moves the estimand:

| ridge ρ | e = 0 | 1 | 2 | 3 | 4 | 5 | time | iterations |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.24 | 0.05 | −0.24 | −0.53 | −0.83 | −1.10 | 3.88s | 10050 |
| 1e−4 | 0.24 | 0.00 | −0.32 | −0.65 | −0.96 | −1.23 | 1.41s | 4325 |
| 1e−2 | 0.19 | −0.21 | −0.75 | −1.26 | −1.67 | −1.93 | 0.23s | 650 |

At ρ = 1e−2 the path lands on SDID's own −1.93, which is the more useful reading
of the exercise than the speedup: the gap between the SC and SDID answers in
`sdid_allsynth_walmart.py` is this penalty, doing the identification work that a
five-year pre-window cannot.

## Status

The two primitives — `project_simplex_cols` and `simplex_lstsq_batch` — live in
the example, not the library. Promoting them belongs next to
`mlsynth/utils/bilevel/simplex.py` on its own branch, test-first: the batched
projection against the scalar version on random input, batch-versus-loop
equivalence on a well-conditioned design, the rank-deficient case asserting equal
objective rather than equal weights, and the degenerate single-donor branch.

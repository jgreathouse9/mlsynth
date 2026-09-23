# The weight-solver problem

A design spike, not a plan of record. It asks what a shared solver layer for
mlsynth would cover, what it would refuse, and what licenses those boundaries.
No code is proposed here beyond signatures.

## What is there now

Counted over `mlsynth/`, excluding tests:

| Measure | Count |
| --- | --- |
| Files importing cvxpy | 54 |
| `cp.Problem` call sites | 62 |
| Of those pinning `solver=CLARABEL` | 55 |
| Sites building `sum(w) == 1` | 50 |
| Helper packages with a sum-to-one solve | 19 |
| Separately named QP helper functions | 14 |
| Simplex QP implementations inside `bilevel/` alone | 3 |
| Solver abstraction layers | 0 |

By objective, across the 59 `cp.Minimize` sites: 28 least squares with or
without a ridge term, 3 explicit `quad_form`, 3 L1, 4 entropy or log or KL,
21 that the classifier could not place. Three files reach for `cp.ExpCone`,
three for `PSD` or `SOC`, three declare boolean or integer variables.

Two facts frame everything below. The first is that the three simplex solvers
already in `bilevel/` -- `active_set.solve_simplex_qp`,
`ridge_augment.simplex_qp` and `penalized._simplex_qp` -- agree to 1e-10 or
better on well-conditioned panels, on `J > T0`, at 1e4 scale and on collinear
donors. Three implementations that provably compute the same thing are one
implementation with two copies.

The second is that CLARABEL, which carries 55 of the 62 solves, is the least
reliable option available for this program. Against `solve_simplex_qp` on the
three canonical panels it returns an identical objective to six significant
figures while leaving weights as low as -2.17e-09 where the active set returns
exact zeros; and at 1000x and 1e6x the West German GDP scale it reports the
simplex INFEASIBLE. A simplex is never empty. `spannability.py` already
carries a scale-normalisation workaround written for that failure, and a
`1e-5 * scale` negligibility floor written for the residual dust.

## The mathematical frame

### The objective is a one-parameter family

Liao, Shi and Zheng (2026, Remark following their eq. for `g`-SCM-relaxation)
observe that the Cressie-Read discrepancies (Cressie and Read, 1984)

    g(x) = (x^(gamma+1) - 1) / [gamma (gamma+1)]

index the objectives the synthetic-control literature actually uses, and are
strictly convex for every `gamma` in `[-1, 1]`:

| gamma | `g(x)` | Name | Cone |
| --- | --- | --- | --- |
| 1 | `x^2` | L2 | quadratic |
| 0 | `x log x` | entropy (Hainmueller 2012) | exponential |
| -1 | `-log x` | empirical likelihood (Owen 1988) | exponential |

They add that all three are Bregman divergences from the uniform weights
`1_J / J`: `D_psi(w, 1_J/J)` for the respective `psi`. The L2 case is
symmetric; entropy and EL penalise deviations above and below the simple
average differently.

This gives the scope boundary for free. At `gamma = 1` the problem is a
quadratic program over a polyhedron. Everywhere else in the family it is
exponential-cone. That is a difference in kind, not in degree, and it is the
line the layer should be drawn along.

### The constraint set is a short hierarchy

Every weight program in the library is one of these, possibly with a free
intercept coordinate:

| Set | Definition | Used by |
| --- | --- | --- |
| Simplex `Delta_J` | `w >= 0`, `1'w = 1` | Abadie-Gardeazabal; mlsynth's `simplex` objectives |
| Cone | `w >= 0` | Bayani (2021) eq. 1.30; `nnls` |
| Affine | `1'w = 1` | Doudchenko-Imbens style |
| Box | `0 <= w <= 1` | several helper packages |
| Free | `w` in `R^J` | Rho et al. (2025) Algorithm 2; Amjad (2018) |
| Relaxed simplex | `Delta_J` and `||Sigma w - Upsilon||_inf <= eta` | Liao-Shi-Zheng |

All six are polyhedral. With a quadratic objective, all six are QPs.

### Why an active set, and not an interior point

Spielman and Teng (2004) introduce smoothed analysis to explain why Dantzig's
simplex method -- which walks vertex to vertex on a polyhedron and is
exponential in the worst case -- is fast in practice: its smoothed complexity
is polynomial in the input size and in the standard deviation of a Gaussian
perturbation of the input. Their subject is linear programming, and the word
"simplex" there names the algorithm and not the constraint set, so the result
does not transfer to our programs as a theorem.

What transfers is the argument. `bilevel/active_set.py` is a primal active-set
method: it pivots on the faces of `Delta_J` exactly as Dantzig's method pivots
on the vertices of a polyhedron, and it inherits the same worst-case
pathology and the same practical behaviour. The module's own docstring
records the empirical counterpart already measured in this repository: cold
pivot counts run 0.6 to 0.9 times `J` from `J = 20` to `J = 320`, and a FISTA
warm start drops them to 0 or 1. An exact finite-termination pivoting method
with that profile is the right default for `gamma = 1`; a general conic
interior-point solver is not, and the dust and the spurious infeasibility
above are what paying for generality looks like.

### Slater holds on the simplex unconditionally, so KKT certifies

Boyd and Vandenberghe (2004, 5.5.3) state the result the certificate rests
on: for a convex problem with differentiable objective and affine
constraints, any point satisfying the KKT conditions is primal and dual
optimal with zero duality gap, and under Slater's condition those conditions
are necessary as well as sufficient.

Every program in the hierarchy above qualifies, and it qualifies always.
The objective at `gamma = 1` is `||A - Bw||^2`, differentiable with Hessian
`2 B'B >= 0`. The constraints are affine. And Slater's condition -- a
strictly feasible point -- is satisfied on `Delta_J` by the uniform weights
`1_J / J`, whose coordinates are all strictly positive and which satisfies
the equality. The probability simplex has nonempty relative interior for
every `J`, so there is no instance of this problem where Slater fails.

Two consequences follow, and the design rests on both.

A residual computed from the returned weights is a complete certificate of
optimality, not a heuristic check. It needs no appeal to what the backend
reported about itself, which is what makes a dispatching layer safe: the
layer can verify any backend's answer in the same currency.

And CLARABEL's INFEASIBLE verdict at 1000x the West German GDP scale is
definitively a defect, not a hard instance. Strong duality holds for every
instance of this program; a solver reporting infeasibility is reporting
something that cannot be true of the feasible set. The earlier note in this
document that "a simplex is never empty" understated it.

### Uniqueness has a textbook condition, and the library already needs it

Boyd and Vandenberghe (2004, Example 3.2): a quadratic `f(x) = (1/2)x'Px +
q'x + r` is convex if and only if `P >= 0` and strictly convex if and only if
`P > 0`. For `||A - Bw||^2` the Hessian is `2 B'B`, positive definite exactly
when `B` has full column rank.

So the minimiser is unique if and only if the donor block has full column
rank, and otherwise the solver returns one point of a continuum that all
achieve the same fit. This is not a corner case in this library. Under
`J > T0` -- the regime Rho et al. (2025) motivate ClusterSC by, and the one
Liao, Shi and Zheng work in -- it never holds. Nor does it hold after a
low-rank denoiser: `spannability.py` already reports
`weights_identified` for exactly this reason, and measured across three
panels and six denoiser-clustering combinations it is False in eleven of
twelve.

A solver layer should carry that condition instead of leaving each call
site to rediscover it. `WeightSolution` reporting whether the minimiser is unique
costs one rank computation and tells a caller whether the weight vector is an
answer or an arbitrary representative of one.

### What Boyd does not license

The book is an argument for interior-point methods, and it is not a source
for preferring an active set over one. The algorithm choice above rests on
Spielman and Teng plus this repository's own pivot measurements, and on the
exactness of finite termination; Boyd supplies the problem-class boundary and
the optimality certificate, which are different claims. Citing it for the
algorithm would be citing it for something it does not say.

## What the layer covers, and what it refuses

Covers: `gamma = 1` on any of the six polyhedral sets, with an optional
ridge term and an optional free intercept. That is 28 least-squares plus 3
`quad_form` objective sites, against roughly 50 constraint sites -- the bulk
of the library.

Refuses, explicitly and by raising: `gamma < 1`. The entropy and EL
relaxations in `laxscm_helpers` are exponential-cone programs and stay on
cvxpy, as do the `PSD`, `SOC`, boolean and integer sites. Ten files, and
forcing them through a QP interface would be a worse abstraction than none.

The refusal is the design. A layer that covers the quadratic corner exactly
and hands everything else back is honest about where an exact method exists.
A layer that covers everything is cvxpy, which the library already has.

## Shape

Two frozen specifications and one entry point. The specification is what to
solve; the solver is how; the solution carries enough to audit it.

```python
@dataclass(frozen=True)
class WeightConstraint:
    nonneg: bool = True
    sum_to_one: bool = True
    upper: float | None = None          # box, when set
    intercept: bool = False             # a free, sign-unconstrained coordinate
    balance: BalanceRelaxation | None = None   # ||Sigma w - Upsilon||_inf <= eta

@dataclass(frozen=True)
class WeightObjective:
    ridge: float = 0.0                  # gamma = 1 with an L2 penalty
    toward: np.ndarray | None = None    # shrinkage target; uniform by default

@dataclass(frozen=True)
class WeightSolution:
    weights: np.ndarray
    intercept: float                    # 0.0 when not fitted
    objective: float
    kkt_residual: float                 # solver-independent certificate
    solver: str                         # which backend ran
    status: str

def solve_weights(B, A, constraint, objective=...) -> WeightSolution: ...
```

Three properties earn their place.

The intercept is a field and not a convention. When `intercept=True` the
design gains a ones column whose coefficient is free in sign and outside any
sum-to-one or non-negativity restriction. Getting that wrong is not
hypothetical: `srho1/ClusterSC`'s `SimplexLinearRegression` places the
intercept inside the simplex, so it is bounded in `[0, 1]` and competes with
the donors for the unit budget -- inert at any realistic outcome scale. A
typed field with one implementation removes the chance of each call site
inventing its own.

`kkt_residual` is solver-independent. Today a caller cannot tell whether
CLARABEL returned `optimal` or `infeasible`, and several call sites do not
check. A certificate computed from the returned weights, and not from the
backend's self-report, is what makes the dispatch safe.

`toward` defaults to uniform because Liao-Shi-Zheng shrink toward
`1_J / J` and so do the Lasso, Ridge and Group Lasso comparators in their
Section 5. Making the target explicit keeps a shrinkage estimator from
silently meaning two different things in two packages.

## Staging, and the risk

The risk is concrete: 103 estimators, and many benchmark cases pinned
value-for-value against current output. The differences measured so far sit
at 1e-10 to 1e-13, which should move nothing, but "should" is a hypothesis
about the tight-tolerance cases and not a finding. It gets checked before a
sweep.

1. Consolidate the three `bilevel/` simplex QPs behind one entry point, with
   the parity and KKT tests `active_set.py` already carries extended to cover
   the other two. No call-site churn, and the duplication is measured, so
   this stage can be verified against itself.
2. Route CLUSTERSC's simplex paths through it -- `pcr/convex.py` and
   `spannability.py` -- both of which then drop workarounds that exist only
   for CLARABEL. This is what unblocks the pending defaults change, since
   making `simplex` the default routes every default run onto the solver with
   the scale failure.
3. Migrate the remaining packages one branch at a time, each keeping its own
   benchmarks green.

An audit of pinned benchmark tolerances precedes stage 3, not stage 1.

## Open questions

Whether `BalanceRelaxation` belongs in the same entry point or beside it. It
is polyhedral and quadratic, so it fits the frame, but `laxscm_helpers`
already has a tuned OSQP path and a cache keyed on the problem shape, and
folding that in may cost more than it saves.

Whether the box and affine sets have live callers or are incidental. The
classifier placed 21 objective sites as "other", and those need reading
before the constraint hierarchy above can be called complete.

Whether an exact active set remains the right default at the `J` that
disaggregated panels reach. Rho et al. motivate ClusterSC by individual-level
data with hundreds of donors; the measured pivot counts scale with the pool
and not with the support, and the FISTA warm start is what holds that down.
The crossover, if there is one, has not been measured.

## References

Cressie, N. and Read, T. R. C. (1984). Multinomial goodness-of-fit tests.
Journal of the Royal Statistical Society B 46(3), 440-464.

Liao, C., Shi, Z. and Zheng, Y. (2026). A Relaxation Approach to Synthetic
Control.

Owen, A. (1988). Empirical likelihood ratio confidence intervals for a single
functional. Biometrika 75(2), 237-249.

Hainmueller, J. (2012). Entropy balancing for causal effects. Political
Analysis 20(1), 25-46.

Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge
University Press. Sections 4.4, 5.5.3 and Example 3.2.

Spielman, D. A. and Teng, S.-H. (2004). Smoothed analysis of algorithms: why
the simplex algorithm usually takes polynomial time. Journal of the ACM
51(3), 385-463.

Duchi, J., Shalev-Shwartz, S., Singer, Y. and Chandra, T. (2008). Efficient
projections onto the L1-ball for learning in high dimensions. ICML.

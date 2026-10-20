# agents_combinatorial.md

# Combinatorial Optimization in `mlsynth`

This file states how `mlsynth` searches over discrete choices: which units are
treated, which donors a unit may use, which set of a given size scores best. It
covers every exact design backend in the library and any search added beside
them, and it exists so that "a MIP solver will handle it" and "branch and bound will be
faster" become claims with a measurement behind them.

The searches it covers:

- MAREX: per cluster, a treated set of size `m` with simplex weights on each side
  (`mlsynth/utils/marex_helpers/exact.py`, and `branch.py` above the candidate
  limit).
- SYNDES: a treated set of size `K` in three modes, two-way (`exact.py`),
  one-way (`one_way_exact.py`) and per-unit (`per_unit_exact.py`).

FSCM's forward scan is a greedy selection with its own cost analysis
(`agents_fscm_complexity.md`). The numerical rules that each fit inside a search
obeys (relative tolerances, zeros to rounding, certification on the original
data) are a separate doctrine, `agents_solvers.md`, proposed in commit `abe222f`
and not yet on `main`; this file is about the search around the fits.

Where the rules come from:

- The MAREX branch-and-bound study, `benchmarks/studies/marex_bnb/`, and its two
  notes: `results/persp_n.md` (what a tight node bound has to contain, with the
  twenty-seven papers read to find it) and `results/rca_enumeration_vs_bnb.md`
  (why the first certified search lost to enumeration).
- The SYNDES studies, `benchmarks/studies/syndes_per_unit_exact/` and
  `benchmarks/studies/syndes_one_way_exact/`, which measured the searches against
  SCIP on 156 runs each.
- The books and papers under References, read against these problems by
  readers who checked each claim on synthetic panels where it needed one.

The thesis in one line:

> A design search is exact when everything it discards is discarded on a valid
> lower bound above a valid incumbent, and fast when its bounds cost less than
> the sets they save. Neither property comes from a general-purpose MIP solver,
> and both can be measured before anything is built.

---

# The shape of the problem

Every design problem here has the same two layers. The outer layer chooses a set
`S` of `K` units from `N`, subject to restrictions. Once `S` is named, the inner
layer is convex: one or a few least-squares fits over simplices, with an exact
oracle, `solve_simplex_qp`, that returns a fit and a Frank-Wolfe gap bounding its
distance to the optimum.

| design | `f(S)` once `S` is named |
| --- | --- |
| MAREX `standard`, `penalized`, `unit_penalized` | `g(S) + h(C(S))`: a treated fit and a control fit that share nothing |
| MAREX `weakly_targeted` | one coupled fit over two simplices |
| SYNDES two-way | `u'Mu` over signed weights, `M = G + lam I`; a closed form when the sign conditions are slack |
| SYNDES one-way | one control fit with the treated weights fixed at `1/K` |
| SYNDES per-unit | `K` fits, one per treated unit, each over the controls |

So the problem is a search over sets with an exact oracle, not a mixed-integer
program. The mixed-integer formulation is a way to write the problem down; it
is not the algorithm.

The vocabulary. To price a set is to evaluate `f(S)` exactly. A node fixes some
units into the treated set (`I`), some out of it (`O`), and leaves the rest free
(`F`); its subtree is every admissible set of size `K` that contains `I` and
avoids `O`, and `r = K - |I|` units remain to choose. A bound on a node is a
lower bound on the minimum over its subtree. The incumbent is the best priced
set so far.

## Four facts about this structure

These decide which of the textbook tools can work. Each was established on
MAREX. SYNDES has not been checked against facts 1, 2 and 4; its per-unit search
rests on the pool monotonicity at the end of fact 3.

1. The linear part is already its own convex hull. MAREX's constraint matrix is
   totally unimodular (Schrijver 1986, Theorems 19.1 and 19.3, through
   Ghouila-Houri's two-colouring; checked numerically on every square minor at
   `n = 2, 3, 4`), so every vertex of the continuous relaxation is integral and
   no linear cut tightens it at any node (Theorem 23.1). The entire gap between
   the relaxation and the optimum sits in the quadratic objective. Cutting
   planes, the main tool of a MIP solver, have nothing to cut.
2. The continuous relaxation scores the average of designs. When the selection
   is relaxed to `0 <= z <= 1`, the point that spreads the weights evenly over
   every unit is an average of designs, and its objective is the objective of
   the average, not the average of the objectives. In MAREX the target is the
   mean of the markets, so that point scores exactly 0 (`persp_n.md`,
   requirement 1). A bound that does not see the cardinality, through products
   of the weights, sees this point.
3. Adding a treated unit improves its own side. A partial selection says
   nothing about its completions unless the bound uses "exactly `K`". This is
   the opposite of problems like the minimum `k`-enclosing ball (Cavaleiro and
   Alizadeh 2022), where a partial set's radius bounds every completion and
   whole subtrees fall near the root. Removing donors runs the other way: a fit
   over a smaller pool is never better, so a fit over a larger pool bounds it.
4. The set function has no exploitable regularity. MAREX's treated fit is
   neither supermodular nor submodular: on a random 12-market panel,
   supermodularity failed in 974 of 5,940 checks and submodularity in 4,788. Its
   costs do not add over units. So greedy guarantees and the
   cardinality-aware bounds built on them (Korte and Vygen 2018; Du et al.
   2022), contribution bounds (Martí and Reinelt 2022; Brusco and Stahl 2005)
   and column generation (Uchoa, Pessoa and Moreno 2024; Petris et al. 2025) do
   not apply.

---

# The rules

Status, as in `agents_unix.md`: binding means the library's searches already
do it and tests pin it; adopted means this file makes it a rule for new
searches, and existing code that departs from it is backlog.

| # | Rule | Status |
| --- | --- | --- |
| 1 | Write the oracle before the search | binding |
| 2 | Discard only on a valid bound above a valid incumbent | binding |
| 3 | Certify a bound on the original data; any multipliers give one | binding (MAREX), adopted |
| 4 | Every search ends in a named outcome and reports its own bound | binding (SYNDES), adopted |
| 5 | Measure the ceiling before designing a bound | adopted |
| 6 | Bound a node only when pricing its subtree costs more | binding (MAREX), adopted |
| 7 | A node bound has to see the cardinality | adopted |
| 8 | Batch the bound; price set by set only what it leaves | binding |
| 9 | Restrictions live inside the search, and the certificate sees the same ones | binding |
| 10 | Reuse is exact or it is a bound | adopted |
| 11 | Singular panels are a regime of the search, not an error | binding |
| 12 | The answer does not depend on the outcome's units | binding |

## 1. Write the oracle before the search

Name `S`, write `f(S)` as the convex problem it leaves, and solve that with the
library's own solver, reading its convergence flag and gap. Only then decide
how to search over `S`.

The mixed-integer route hides this structure from the solver that needs it.
SCIP on SYNDES's per-unit MIP, at the library's defaults of 60 s and a 5% gap,
on 78 instances (`syndes_per_unit_exact/README.md`):

| outcome | instances |
| --- | ---: |
| a design | 57 |
| `solver_error` at the limit, no design | 15 |
| hung past its time limit | 4 |
| crashed with heap corruption | 2 |

Of the 57 designs, 31 were more than 10% worse than the optimum, and the median
excess among the non-optimal ones was 56%. The branch and bound on units closed
every instance in at most 28 s. On 60 small BLS draws SCIP reported `optimal`
on all 60 and was wrong on 2; rescaling the panel to unit variance fixed both,
so its "optimal" depended on the units of the outcome (#814). One-way fared
better, since its program has `N` binaries where per-unit's has `N^2` weights,
but still returned designs up to 10.1% worse at its limit.

Test pattern: price every set of a small panel by brute force and assert the
search returns the same set and value; this is the oracle for every other test
of a search.

## 2. Discard only on a valid bound above a valid incumbent

A set leaves the search in one of two ways: it is priced, or a lower bound on it
(or on a subtree containing it) reaches the incumbent. That is the whole
correctness argument, and it holds with any valid bound: a weak bound costs time,
never the answer.

Two consequences. First, the stop rule is a bound: a search that visits sets in
bound order stops when the next bound reaches the incumbent, which proves the
incumbent optimal (the one-way and per-unit searches). Second, a fit can stop
early: the Frank-Wolfe gap makes `value - gap` a lower bound at every iterate,
so a fit handed the incumbent as a cutoff discards its set as soon as that bound
clears it (Büsing, Gersing and Koster 2023 do the same per node; MAREX's coupled
fit does it per set).

The comparison is relative to the values compared, with no absolute floor
(`agents_solvers.md`, rule 3). The zero of a nonnegative objective is the one
place a different threshold enters, and rule 11 covers it.

Test pattern: a property that the bound is at most every completion's price,
over random panels and random nodes, checked by brute force over the subtree;
equality at a leaf; one mutant per pruning comparison.

## 3. Certify a bound on the original data; any multipliers give one

Weak duality gives a valid lower bound from any dual-feasible point, not only an
optimal one (Nocedal and Wright 2006). So a relaxation never has to be solved to
optimality to bound a node; the solver's multipliers, clipped to their cones,
define a Lagrangian whose minimum over the designs is a bound. Accurate
multipliers give a tight bound, inaccurate ones a valid weaker one.

The bound is evaluated on the original data, with its rounding controlled. MAREX
assembles the Lagrangian in exact arithmetic from the panel's exact Gram matrix
and bounds the smallest eigenvalue below exactly: a floating-point Cholesky
factor, its residual computed exactly, then Weyl's and Gershgorin's theorems.
This is the rigorous bound of Jansson, Chaykin and Keil (2007) for semidefinite
programs, the analogue of Neumaier and Shcherbina (2004) for linear ones, with
exact arithmetic in place of their interval arithmetic. The solver's own value
once sat above the brute-force subtree minimum, at 1.000000065 of it; the
certified bound never did (`persp_n.md`, "The certificate").

Where the search prices in floating point, the rounding is bounded instead. The
certified enumeration reads the floating-point Gram and bounds its error as
`|Gf - G| <= gamma_T |B|'|B|`, which kept it at 1.0 to 2.6 µs per set where
rebuilding each Gram block from the data had cost about 1,000 flops per set
(`rca_enumeration_vs_bnb.md`, cause 6). Wolfe's (1976) Theorem 2.2 turns any
direction into a lower bound on the distance between two hulls, so a stopped or
inaccurate nearest-point solve still certifies (MAREX `hull_separation`).

Test pattern: a hostile near-collinear panel on which the uncertified value
crosses the true minimum and the certified one does not; a mutant that drops
each slack term. A slack term whose mutant survives because a larger term masks
it stays, with the reason recorded (MAREX's `product-rounding-ignored`, masked by
the Gram term whenever `T >= m`).

## 4. Every search ends in a named outcome and reports its own bound

A search ends optimal (closed: nothing open under the incumbent), at a limit
(a node limit, a candidate limit, a price-all limit) with its best design and
the bound it reached, or refused with the reason and what would let it proceed.
In SYNDES the outcome is a typed field (`solve_outcome`), and a limit also
issues a `SYNDESSolveWarning`; neither is printed (invariant 7).

The certificate reads that bound. It does not compute a second, weaker answer
to the question the search already answered (#822): SYNDES's per-unit
certificate had reported `certified=False` on a continuous relaxation about 70%
loose, while its own branch and bound held the optimum.

When a search runs over several sizes (`K=None`), the bound is the smallest over
the sizes searched, and the design is certified only when every size's search
was. The best size's bound alone is not a bound on the union. On every panel
checked the best design sat at the size with the smallest bound, so no example
test could see the difference; a crafted outcome per size could.

Test pattern: force each limit and assert the outcome, the bound, the gap and the
warning; assert a clean search is silent with warnings as errors.

## 5. Measure the ceiling before designing a bound

Before building a node bound, price every set of the panels it is for and
measure how tight a bound must be to prune. For a bound reaching a fraction
`rho` of every subtree's minimum, with the incumbent already at the optimum,
count the sets that still reach pricing (`marex_bnb/ceiling.py`). On GeoLift
with four treated (91,390 sets):

| `rho` | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sets left to price | 71,770 | 39,554 | 4,782 | 386 | 15 |

A bound at a third of each subtree's minimum leaves more than 40% of the sets.
Subtree minima sit close to the optimum: at depth one their median is 1.75 times
it on GeoLift. A set can carry units at zero weight, so a node fixing `d` units
has a minimum no larger than about the best design with `K - d` units, which
caps how far above the optimum any subtree can sit. Measure this before choosing
which bound family to build.

The same discipline applies to a whole search: compute what it must beat. The
MAREX enumeration prices a set in about 1.4 µs with a batched closed-form bound,
so a node bound costing a second must save about 700,000 sets to break even.

## 6. Bound a node only when pricing its subtree costs more

Each node is either priced set by set or bounded and split; the choice is a cost
comparison, never a fixed size. In MAREX's first certified search a fixed
64-set threshold sent every node above it to a semidefinite lift that cost
70.7 s, about 52 million priced sets, so it lost to enumeration by a factor of
2,500 at four treated (`rca_enumeration_vs_bnb.md`). With the rule (bound a
node when its sets times the price per set exceed the bound's cost), the
search with persp+N is 56 times faster than enumeration at ten treated on
GeoLift and about twice as slow at six, where a bound costs more than the
subtrees it saves.

The costs are calibrated on the enumeration's own path. Timing a 20,000-set
batch, where fixed overheads dominate, overstated the price per set at 21 to 37
µs against 6.5 µs, and sent the root of a six-treated search to 93 s of lifts
where enumeration cost 25 s.

The costs decide where the search spends time and never what it proves. A leaf
is never bounded: a node with no free unit is one set, and lifting it raised an
`IndexError` that the fixed threshold had hidden.

Test pattern: Hypothesis properties that the rule is monotone in the subtree's
size and that the search's answer is the same under any cost model; mutants for
a fixed threshold, a rule computed and ignored, and a bounded leaf.

## 7. A node bound has to see the cardinality

From facts 2 and 3, a relaxation that keeps the selection fractional and the
weights linear sees the averaged point and scores near 0. What MAREX's node bound
needed, each forced by a failure (`persp_n.md`, requirements 1 to 5):

1. second moments, the products `W = w w'`, without which the averaged point
   scores 0;
2. entrywise `W >= 0` (Burer 2009), without which the root bound fell from 99%
   to 29.5% of the optimum;
3. the perspective `W_jj z_j >= w_j^2` (Frangioni and Gentile 2006; Günlük and
   Linderoth 2010), which turns `sum z = m` into concentration of the weights;
4. a global semidefinite coupling: every decomposition into pieces on one or two
   units is exactly 0 once the units outnumber the periods by two or three;
5. multipliers solved per node: freezing the root's keeps the root's value at
   every child (the quadratic convex reformulation of Wu et al. 2017; the convex
   quadratic relaxations of Nohra, Raghunathan and Sahinidis 2022).

The smallest object meeting all five, persp+N, keeps one semidefinite block per
side of `|U| + 1` rows and no indicators in a block, and certified the root in
1.5 to 3.7 s at 97.5 to 98.5% of the optimum, where the full doubly nonnegative
lift took 252 s for 99.15%.

For SYNDES the cardinality enters differently. Two-way, once the sign vector
`sigma` is fixed, has the closed form `lb(S) = 4 alpha / sigma'R sigma`, and
`sigma` carries `K` exactly; Rayleigh's inequality over every `sigma` with `K`
positive entries bounds the whole problem at once, at 88.8 to 92.0% of the
optimum. Per-unit's node bound carries `K` through the sum of the `K - |F|`
smallest single-unit fits.

## 8. Batch the bound; price set by set only what it leaves

A bound evaluated over thousands of sets in one array operation costs about a
microsecond a set. A bound computed one set at a time through library calls
costs a millisecond, nearly all of it call overhead. MAREX's treated bound
`(2 + t)^2 / (4 s) - u / 4`, batched over blocks of 20,000 sets from a
precomputed Gram, priced GeoLift's 76.9 million eight-treated sets at 1.15 µs
each and left 22 for fits. Its hull-distance bound for the weakly-targeted
design spent 1.29 ms per candidate when run one candidate at a time (43% in
least-squares calls on 90 x 8 matrices whose arithmetic is about 3 µs); run in
lockstep over a batch, it took the `beta = 100` search from 82,296 fits and 237
s to 677 fits and 16 s.

The tightest batched bounds make a search cost one fit. SYNDES one-way's ridge
bound `d'(C'HC)^{-1} d`, one `(K + 1) x (K + 1)` solve per set, equalled the
optimum to rounding on 126 of 132 runs, so the search priced one set on all but
one run; its cost is the enumeration.

## 9. Restrictions live inside the search, and the certificate sees the same ones

Forced and forbidden units start the root's `I` and `O`. Stratum quotas and
forced units decide how many sets a node holds, and a node with none is dropped
(`count_candidates`). A unit included at a node sends its conflict partners out.
Every priced set passes the filter (`SubsetFilter`). Donor exclusions shrink a
control pool, which can only raise a fit, so a bound computed without them stays
valid and only the priced fit honours them.

A restriction applied after the search, as a filter on its output, returns the
best admissible design among those the search happened to keep, which is not the
admissible optimum.

The certificate must search the same problem. Under holdout selection SYNDES's
certificate runs its own search on the whole pre-period, and a mutant that
dropped the restrictions from that call survived a first mutation run because
every test was in-sample, where the certificate reads the design's own search
(#822). A holdout test that forbids the unrestricted optimum's units kills it.

## 10. Reuse is exact or it is a bound

A child differs from its parent by one fixed unit, so its work overlaps its
parent's. Reuse that work only where the reuse is exact, or treat the result as
a lower bound and certify it as one.

Exact reuse:

- A fit stays optimal on a smaller pool whenever the removed donors carried no
  weight in it (SYNDES per-unit; Nocedal and Wright on active sets).
- A parent's fit is dual feasible for a child after one bound change, so a dual
  step warm-starts the child (Atamtürk and Gómez 2019 report about 15 times per
  node over an interior-point solver).

Reuse as a bound:

- A parent's multipliers, folded onto the child's rows, give the child a valid
  certified bound with no solve (`marex_bnb/inherit.py`). On GeoLift it
  recovered a median 5.8% of the gain a fresh lift gives and excluded 0 of 34
  excludable markets. Valid and nearly useless: requirement 5 of rule 7 again.

## 11. Singular panels are a regime of the search, not an error

With fewer pre-periods than units, a penalty at zero, or a unit that is an exact
combination of others, `G + lam I` is singular and every closed form that needs
its inverse disappears. This is a common panel, not a corner case. The search
handles it:

- The objective is nonnegative, so a set priced at zero, to the rounding of its
  terms, ends the search. The threshold scales with the matrix
  (`256 eps max|M|` in SYNDES two-way, #821): an absolute `1e-12` read a positive
  optimum of about `1e-14`, on a panel scaled by `1e-6`, as zero and stopped
  with the wrong design labelled certified.
- Otherwise every set is priced by a method that needs only a semidefinite
  matrix (projected gradient), the open sets are priced again to a tighter
  tolerance, and the bound is `max(0, min(value - gap))`.
- `M = 0` (every unit equal to its period's mean) makes every design score 0 and
  the projected-gradient step `1 / (2 lambda_max)` divide by zero; the search
  returns the first admissible design.
- Past a price-all limit it raises, and the message names what would help (a
  positive penalty, more pre-periods), not another solver.

Eigenvalue shifts behave the same way. Bienstock's (2010) bound for a convex
quadratic over the simplex with at most `K` nonzeros needs a positive smallest
eigenvalue on the free units, so it is off whenever the free units exceed
`T + 1`. Where it was on, on 40 markets, it reached 4 to 21% of the subtree
minimum with 18 or more markets free, and about 50% with 10.

## 12. The answer does not depend on the outcome's units

Fits run on the panel centred within each period and divided by its root mean
square (#812, #817), stopping rules are relative, and the zero floor of rule 11
scales with the data. The best designs nearly cancel: on GeoLift the optimum
sits about 2,000 times below the objective's scale. That is why every
first-order semidefinite solver tried at MAREX's nodes failed to certify (ADMM
reached 85% of the bound, a spectral bundle method 5%, LoRADS did not converge,
SCS certified nothing) and why SCIP's `optimal` on SYNDES's BLS draws depended on
their units.

Test pattern: a Hypothesis property over scales `10^k`, `k` from -6 to 6, and
level shifts up to `1e6`: the same design, the objective scaled by `s^2`.

---

# Bound families and where each stands here

What each family needs, and what it gave on these problems. Measured on MAREX
unless the row says otherwise; `persp_n.md` has the readers' details.

| family | sources | needs | here |
| --- | --- | --- | --- |
| Continuous (natural) relaxation | Bienstock 1996; Takapoui et al. 2020; Kılınç-Karzan, Küçükyavuz and Lee | nothing | 0 at MAREX's root (fact 2) |
| McCormick, RLT, discretized products | Gupte et al. 2013, 2020; Nagarajan et al. 2018; Beach et al. 2024 | bounded continuous products | negative, by 26 to about 50,000 times the optimum |
| Lagrangian relaxation of the linking constraints | Martin 1999; Korte and Vygen; Crainic, Gendreau and Frangioni 2024 | a subproblem without the integrality property | equals the continuous relaxation (Martin's integrality property) |
| Lagrangian relaxation of `sum z = K` | Martin 1999 | a tractable remainder | the best set of any size, no easier |
| Column generation, branch-price-and-cut | Uchoa, Pessoa and Moreno; Petris et al. | costs additive over units | not additive (fact 4) |
| Greedy and diminishing-returns bounds | Korte and Vygen; Du et al. | sub- or supermodularity | neither (fact 4) |
| Frank-Wolfe branch and bound | Buchheim et al. 2018; Mexi et al. 2026 (Boscia) | a hull of integer points tighter than the relaxation | the linear part is its hull (fact 1), so the continuous bound |
| Benders master over tangent cuts | Ghezzi et al. 2026 (CAMINO) | many cuts | converges, in the worst case after as many sets as enumeration; never rose above the relaxation in the paper's 72-binary run; not measured here |
| Diagonal perspective and small-piece hulls | Frangioni and Gentile; Günlük and Linderoth; Zheng, Sun and Li 2014; Han, Gómez and Atamtürk 2023; De Rosa and Khajavirad 2026 | a usable diagonal or small pieces | 0 once the units outnumber the periods by two or three |
| Eigenvalue shift, S-lemma | Bienstock 2010; Li 2025 | positive definite on the free units | off near the root when `N > T + 1` (rule 11) |
| Semidefinite with the cardinality on the lift | Rendl in Jünger et al. 2010; Burer 2009 | a solve per node | persp+N: 97.5 to 98.5% at the root in seconds, certified (rule 7) |
| Closed form on a fixed assignment | SYNDES `gram.py`; MAREX `exact.py` | the sign or side pattern named | MAREX treated bound; SYNDES two-way `4 alpha / sigma'R sigma` |
| Ridge relaxation (drop only the sign conditions) | SYNDES one-way study | `G + lam I` invertible | equals the optimum to rounding on 126 of 132 runs |
| Pool monotonicity | SYNDES per-unit study | a fit that can only rise as its pool shrinks | per-unit branch and bound closed all 156 runs |
| Whole-problem Rayleigh bound | SYNDES `gram.py` | `G + lam I` invertible | 88.8 to 92.0% of the optimum, one eigenvalue |

---

# The searches in the library

| search | below the candidate limit | above it | certificate |
| --- | --- | --- | --- |
| MAREX `standard`, `penalized`, `unit_penalized` | enumeration under the batched treated bound (`exact.py`) | branch and bound with the persp+N node bound and the cost rule (`branch.py`) | exact-arithmetic bracket `lower <= f* <= upper` (`certify.py`) |
| MAREX `weakly_targeted` | enumeration under the treated bound plus the certified hull distance | refused | refused: the coupled objective does not split into the two fits the bracket bounds |
| SYNDES two-way | enumeration under `4 alpha / sigma'R sigma` | swap search, uncertified, against the Rayleigh bound | the search's bound (#822) |
| SYNDES one-way | enumeration in ridge-bound order, stopped by the bound | swap-search fallback against the Rayleigh bound | the search's bound |
| SYNDES per-unit | depth-first branch and bound on units, node limit 5,000,000 | the same, with its open bound at the limit | the search's bound |

The open row is SYNDES's two-way and one-way searches above their limit, where
no node bound has been built (#803, #820).

---

# Testing a search

`agents_tests.md` sets the four levels and the instruments. For a search, the
standing tests are:

- Brute force over every admissible set on small panels is the reference for the
  search's answer; brute force over every completion of a node is the reference
  for a node bound.
- A bound's validity is a property, not an example: at most every completion's
  price, over random panels and random nodes; equal at a leaf; at the root, equal
  to the closed form where one is known.
- Panels with power. A check that passes because two candidates coincide has not
  passed: the `K=None` aggregation of rule 4 was invisible on every natural
  fixture. Build the fixture where the candidates differ. The standing set is
  near-collinear columns, tied optima (with the tie order asserted), panels
  without structure where every bound sits near every other (the slowest case in
  the MAREX and per-unit studies), `T < N`, a duplicated unit, and `M = 0`.
- One mutant per pruning comparison, per stop rule, per reuse rule, per
  restriction path, and per limit. A surviving mutant is either equivalent, and
  the code it mutates goes, or masked, and the reason is recorded. Three
  survivors in the SYNDES one-way build were resolved that way: a rescaling that
  measured as doing nothing was removed, the centring was pinned by a test that
  prices every set of a panel at level `1e8`, and the stop rule by a panel built
  so that a donor exclusion moves the optimum.
- Timings live in studies, never in tests. A test pins what a search proves,
  not how long it takes.

---

# Open

- A node bound for SYNDES two-way and one-way above the candidate limit (#803,
  #820). Every two-way bound is also a one-way bound, since one-way's constraints
  imply two-way's and its optimum is no lower. The candidate under study bounds `sigma'R sigma` over the free
  signs of a node by a trust-region subproblem; its validity is not yet
  established.
- Hundreds to thousands of units. persp+N has about `n^2` variables per node and
  took 4 s at 40 markets and 150 to 186 s at 100; at 2,000 no interior-point
  method solves it per node. The route at that scale is structure: small `K` by
  certified enumeration, and clusters or screening to bring each search to tens
  or low hundreds of units.
- MAREX's weakly-targeted design above the limit: its objective couples the two
  sides through the synthetic treated unit, and persp+N splits them.
- Warm starts for the per-unit search's fits, whose time on the hardest instances
  is the fits (204,000 on a 20-unit noise panel).

---

# References

Books.

- Brusco, M. J., and Stahl, S. (2005). *Branch-and-Bound Applications in
  Combinatorial Data Analysis*. Springer. The lexicographic subset tree.
- Crainic, T. G., Gendreau, M., and Frangioni, A. (eds.) (2024). *Combinatorial
  Optimization and Applications: A Tribute to Bernard Gendron*. Springer,
  International Series in Operations Research and Management Science 358. A
  node relaxation costing ten times more per node rarely pays (ch. 3);
  reliability branching (ch. 4); no perspective reformulation.
- Du, D.-Z., Pardalos, P. M., Hu, X., and Wu, W. (2022). *Introduction to
  Combinatorial Optimization*. Springer Optimization and Its Applications 196.
  Its cardinality-aware bounds need sub- or supermodularity.
- Jünger, M., Liebling, T., Naddef, D., Nemhauser, G., Pulleyblank, W., Reinelt,
  G., Rinaldi, G., and Wolsey, L. (eds.) (2010). *50 Years of Integer
  Programming 1958-2008*. Springer. Rendl's chapter: the semidefinite
  relaxation with the cardinality on the lifted matrix.
- Korte, B., and Vygen, J. (2018). *Combinatorial Optimization: Theory and
  Algorithms*, 6th ed. Springer. The fixed-in, fixed-out node; greedy results
  that assume linear costs.
- Martí, R., and Reinelt, G. (2022). "Branch-and-Bound." Chapter 4 of *Exact and
  Heuristic Methods in Combinatorial Optimization*, 2nd ed. Springer.
- Martin, R. K. (1999). *Large Scale Linear and Integer Optimization: A Unified
  Approach*. Kluwer. The integrality property that makes the obvious Lagrangian
  bound equal the continuous relaxation.
- Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.
  Springer. Bounds from dual-feasible points; active sets and warm starts.
- Schrijver, A. (1986). *Theory of Linear and Integer Programming*. Wiley.
  Theorems 19.1, 19.3 and 23.1: MAREX's linear part is its own hull.

Papers.

- Atamtürk, A., and Gómez, A. (2019). "Simplex QP-based methods for minimizing a
  conic quadratic objective over polyhedra." Mathematical Programming
  Computation 11:311-340. Warm starts at nodes.
- Bienstock, D. (2010). "Eigenvalue techniques for convex objective, nonconvex
  optimization problems." IPCO 2010, Lecture Notes in Computer Science 6080,
  29-42.
- Buchheim, C., Caprara, A., and Lodi, A. (2010). "An effective branch-and-bound
  algorithm for convex quadratic integer programming." IPCO 2010, Lecture Notes
  in Computer Science 6080. Cheap nodes from precomputed inverses of a positive
  definite matrix, which these problems lack.
- Buchheim, C., De Santis, M., Rinaldi, F., and Trieu, L. (2018). "A Frank-Wolfe
  based branch-and-bound algorithm for mean-risk optimization." Journal of
  Global Optimization 70:625-644.
- Büsing, C., Gersing, T., and Koster, A. M. C. A. (2023). "A branch and bound
  algorithm for robust binary optimization with budget uncertainty."
  Mathematical Programming Computation 15:269-326. Stop a node's solve when its
  bound clears the incumbent.
- Cavaleiro, M., and Alizadeh, F. (2022). "A branch-and-bound method for the
  minimum k-enclosing ball problem." Operations Research Letters 50:274-280.
- Ghezzi, A., Van Roy, W., Sager, S., and Diehl, M. (2026). "A sequential
  Benders-based mixed-integer quadratic programming algorithm and its
  implementation in the CAMINO toolbox." Mathematical Programming Computation.
- Gilbert, E. G., Johnson, D. W., and Keerthi, S. S. (1988). "A fast procedure
  for computing the distance between complex objects in three-dimensional
  space." IEEE Journal of Robotics and Automation 4(2):193-203. The support
  mapping behind the lockstep hull bound.
- Hazimeh, H., Mazumder, R., and Saab, A. (2022). "Sparse regression at scale:
  branch-and-bound rooted in first-order optimization." Mathematical
  Programming 196:347-388. Node bounds from first-order dual certificates.
- Jansson, C., Chaykin, D., and Keil, C. (2007). "Rigorous error bounds for the
  optimal value in semidefinite programming." SIAM Journal on Numerical Analysis
  46(1):180-200.
- Lacoste-Julien, S., and Jaggi, M. (2015). "On the global linear convergence of
  Frank-Wolfe optimization variants." Advances in Neural Information Processing
  Systems 28.
- Li, Y. (2025). "The augmented factorization bound for maximum-entropy
  sampling." IPCO 2025, Lecture Notes in Computer Science 15620, 412-426.
- Locatelli, M., Piccialli, V., and Sudoso, A. M. (2024). "Fix and Bound: an
  efficient approach for solving large-scale quadratic programming problems
  with box constraints." arXiv 2211.08911v2. Fixing variables once a bound on
  departures reaches the incumbent.
- Mexi, G., Hendrych, D., Designolle, S., Besançon, M., and Pokutta, S. (2026).
  "A Frank-Wolfe-based primal heuristic for quadratic mixed-integer
  optimization." Mathematical Programming Computation.
- Montanari, M., Petrinic, N., and Barbieri, E. (2017). "Improving the GJK
  algorithm for faster and more reliable distance queries between convex
  objects." ACM Transactions on Graphics 36(3):30. Its robustness warning
  applied: a Gram-based nearest-point step stalled at GeoLift's scale, and the
  centred least-squares form fixed it.
- Neumaier, A., and Shcherbina, O. (2004). "Safe bounds in linear and
  mixed-integer linear programming." Mathematical Programming 99:283-296.
- Petris, M., Archetti, C., Cattaruzza, D., Ogier, M., and Semet, F. (2025). "A
  tutorial on branch-price-and-cut algorithms." 4OR 23(1):1-52.
- Uchoa, E., Pessoa, A., and Moreno, L. (2024). *Optimizing with Column
  Generation: Advanced Branch-Cut-and-Price Algorithms*, Part I. Cadernos do
  LOGIS 2024(3).
- Wolfe, P. (1976). "Finding the nearest point in a polytope." Mathematical
  Programming 11:128-149. Theorem 2.2: any direction bounds the distance.

The twenty-seven papers behind rule 7's five requirements, and the lifts and
certificates read for them (Burer 2009 among them), are listed with what each
contributed in `benchmarks/studies/marex_bnb/results/persp_n.md`.

The designs. Abadie, A., and Zhao, J. (2026), "Synthetic Controls for
Experimental Design" (MAREX); Doudchenko, N., Khosravi, K., Pouget-Abadie, J.,
Lahaie, S., Lubin, M., Mirrokni, V., Spiess, J., and Imbens, G. (2021),
"Synthetic design: an optimization approach to experimental design with
synthetic controls," Advances in Neural Information Processing Systems 34
(SYNDES).

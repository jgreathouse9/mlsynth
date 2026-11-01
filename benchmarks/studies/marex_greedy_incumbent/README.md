# MAREX's base design at 18 markets: a greedy incumbent, and what prunes

The question: on a small panel of 18 markets, can a greedy incumbent for
MAREX's base design prune the exact search and give a valid bound on the
optimum?

The answer, measured on five 18-market panels at every treated count from 1 to
17 and unconstrained, 90 settings in all:

1. It gives a valid bound. The greedy design is admissible. Its objective,
   evaluated exactly at its fitted weights, is an upper bound on the optimum
   that floating-point rounding cannot break. It held in all 90 settings.
2. It cannot prune the library's search, and no incumbent can. The search fits
   treated sets in increasing order of a lower bound. A walk in that order
   reaches the same sets whatever incumbent it starts from; the proof is below.
   - Seeded with the greedy design, the search walked exactly the sets it walks
     unseeded at all 85 fixed treated counts.
   - Seeded with the optimum itself, the best incumbent there is, it walked the
     same sets at 82 of the 85.
   - The other three, and the five unconstrained designs, saved at most 2.9% of
     the sets walked. All of that saving falls in the search's second pass,
     whose order is by block.
3. The greedy is not free. It evaluates 18 to 450 treated sets, two simplex
   fits each. At 40 of the 85 fixed counts that is more sets than the library's
   whole search walks.
4. What prunes is the bound. The library's bound sees only the treated side of
   a design. The control side of the base design is the same fit over the
   complement, so the same closed form bounds it. With that term added, the
   search walks 1 to 141 sets at each treated count where the library walks up
   to 40,968. Unconstrained, it walks 2 to 68 where the library walks 20,007 to
   179,597, in about a second where the library takes 4 to 30.

SCIP, the MIQP backend, does not walk in bound order, and there a warm start
does change the work: it moved the node count by -69% to +35% against a cold
start, with fewer nodes in most settings, the optimum as the start included.
The exact backend solved the same designs 11 to 39,000 times faster. At the
library's settings SCIP also discards MAREX's `warm_start` on the two panels
with many fit periods, without a message: the start fixes too small a share of
SCIP's variables for its completion heuristic to accept it.

This is a study, not a benchmark case, and it changes nothing in the library.
`search.py` runs the library's own search with one global swapped for the
length of the call. Timings are machine-dependent and nothing here is pinned.

```
python -m pytest benchmarks/studies/marex_greedy_incumbent -q -p no:cacheprovider
python -m benchmarks.studies.marex_greedy_incumbent.run enumeration section5 geolift walmart noise factor
python -m benchmarks.studies.marex_greedy_incumbent.run scip section5 2,3,4,5,6,7,8,9
python -m benchmarks.studies.marex_greedy_incumbent.run seedcheck section5 geolift walmart
python -m benchmarks.studies.marex_greedy_incumbent.run bnb section5 walmart
python -m benchmarks.studies.marex_greedy_incumbent.run table
```

Run from the repository root. `enumeration` caches each panel's brute force
under `results/` (`brute_*.npz`, ignored by git).

## What is here

| file | what it does |
| --- | --- |
| `panels.py` | the five 18-market panels |
| `brute.py` | the treated fit on every set of markets, so the optimum at every size by brute force |
| `greedy.py` | the greedy incumbent and the exact bracket of a design's objective |
| `search.py` | the library's search with its incumbent seeded, or with the two-sided bound |
| `run.py` | the runner, and the tables below from its output |
| `test_greedy_incumbent.py` | the claims below as tests and properties, on panels small enough to enumerate |
| `results/enumeration_*.jsonl` | one line per panel and treated count: the optimum, the greedy, and four searches |
| `results/scip_*.jsonl` | one line per panel and treated count: the MIQP cold and warm-started |
| `results/seedcheck.jsonl` | whether SCIP uses MAREX's warm start at the library's settings |
| `results/bnb.jsonl` | the certified branch and bound at 18 markets, beside the enumeration |
| `results/tables.md` | every row, as `run table` prints it |

## The base design and the search

Abadie and Zhao's base design, their program (7), chooses treated weights `w`
and control weights `v`, each on the simplex and never both positive on one
market, to bring both weighted averages of the predictors `X_j` to the
population average `X̄`:

```
min ||X̄ - sum_j w_j X_j||^2 + ||X̄ - sum_j v_j X_j||^2,   m_lo <= ||w||_0 <= m_hi.
```

In MAREX this is `design="standard"`. The exact backend searches each cluster
over its treated sets `S`. Naming `S` splits the objective into two fits that
share nothing,

```
f(S) = g(S) + g(S^c),
```

with `g(U)` the least squared distance from the target `X̄` to the convex hull
of the markets in `U`. Without donor exclusions or penalties the control fit is
the treated fit's own problem over the complement. A set is admissible when its
fit gives every market in it a positive weight and its complement is not empty:
Abadie and Zhao count the nonzero treated weights.

Some vocabulary. An incumbent is the best design found so far, and its
objective is the cutoff. A lower bound on a set is a number at most its
objective. To prune a set is to discard it because its lower bound reaches the
cutoff, so it cannot beat the incumbent. A bound is valid when it is never
above the objective it bounds, and an incumbent is valid when it is a design the
problem admits; with both valid, pruning never discards the optimum.

The library's enumeration (`exact.search_cluster`) prices every candidate with a
lower bound `lb(S)` and fits candidates in increasing order of it, a best-first
walk. The bound is the closed form of the treated fit with the sign constraints
dropped, the squared distance from the target to the affine hull of `S`, plus a
floor under every control fit. For the base design the floor is zero, since the
whole cluster reproduces its own mean. The walk stops at the first candidate
whose bound reaches the cutoff. A candidate the walk reaches has its treated
fit computed, and its control fit too unless the treated fit alone already
reaches the cutoff. The tables count both: walked sets, and fitted sets.

At 18 markets the backend enumerates: an unconstrained cluster holds 262,142
treated sets, far under the twenty-million limit above which the certified
branch and bound takes over.

## The greedy incumbent

`greedy.py` builds the incumbent in two passes, each evaluating `f` exactly with
the library's simplex solver.

1. Forward. Start with no treated market. At each step add the market whose
   admissible extension of the current set has the least objective (when no
   extension is admissible, the least of any). Continue to the largest treated
   count searched and keep the best admissible set met at a count searched. At
   `m` treated markets this evaluates `n + (n - 1) + ... + (n - m + 1)` sets.
2. Exchange. From that set, make the best exchange of one treated market for
   one control market among admissible sets, as long as it lowers the
   objective. Each round evaluates `m (n - m)` sets.

At one treated market the forward pass evaluates every market and is exact.
Past one it carries no guarantee. The treated fit is neither submodular nor
supermodular (`agents/agents_combinatorial.md`, fact 4), so no greedy
approximation result applies. Measured, the greedy design's objective was 1.00
to 2.58 times the optimum.

The bound it provides. Any admissible design has `f* <= f(S)`, and
`greedy.bracket` evaluates `f(S)` in exact rational arithmetic at the fitted
weights, clipped and rescaled to the simplex exactly
(`ExactData.interval`, the machinery of `certify_cluster`). The upper end is the
objective at those weights, so it is at least `f(S)` and at least `f*`, whatever
the accuracy of the fits. In all 90 settings it was at least the exact lower end
of the optimum's own bracket, and within `1.1e-14` of the greedy's
floating-point objective.

## Why no incumbent prunes this search

Write the candidates in the order the walk takes them, by increasing bound. The
cutoff `c` starts at the seed's objective, or at infinity without one, and falls
to the objective of each better design the walk fits. The walk stops at the
first candidate with `lb(S) >= c + delta(c)`, where
`delta(c) = 1e-6 (1 + |c|)` is the library's margin.

Claim. If the bound is valid and the seed is an admissible design, or there is
no seed, the walk reaches exactly the candidates with `lb(S) < f* + delta(f*)`.

Proof. The cutoff is always the objective of an admissible design, so it never
falls under `f*`, and `c + delta(c)` grows with `c`. A candidate with
`lb(S) < f* + delta(f*)` therefore cannot stop the walk, and neither can any
candidate before it, whose bound is no larger; it is reached. An optimal set
`S*` has `lb(S*) <= f(S*) = f*`, so it comes before every candidate whose bound
is at least `f* + delta(f*)`, and it is reached before them. Its treated fit is
at most `f*`, so either it is fitted or the cutoff is already `f*` (the seed is
optimal, and the treated fit of `S*` reaches it); either way `c = f*` from then
on.
The first candidate with `lb(S) >= f* + delta(f*)` stops the walk. Neither step
used the seed. ∎

What a seed can change, then, is limited to two things:

- For candidates the walk reaches before `S*`, whether the control fit is
  skipped after the treated fit. The greedy seed skipped no control fit the
  unseeded search made: the fitted counts were equal at all 85 fixed counts.
- The order outside the first pass. The walk is first taken over the 512
  candidates with the smallest bounds. If those do not settle the search, a
  second pass prices all candidates again in blocks of 20,000 and walks each
  block in its own bound order. Across blocks the order is not by bound, and a
  block before the one holding `S*` can have candidates whose bound lies
  between `f*` and the cutoff of the moment; a seed under that cutoff skips
  them. The second pass ran in 47 of the 90 settings, and the eight where the
  optimum seed saved anything are all among them.

A property in `test_greedy_incumbent.py` checks the claim over random panels
small enough for one pass: under any admissible seed the walk reaches exactly
the candidates whose bound is under `f* + delta(f*)`. On the 18-market panels
the optimum-seeded walk reached exactly those candidates in all 90 settings,
and the unseeded walk in 82.

This is the classical property of best-first search: it expands the nodes whose
bound is under the optimum, which any search with the same bound must expand to
prove the optimum, and no node whose bound is over it (Dechter and Pearl 1985
for A*; Morrison et al. 2016 survey it for branch and bound). An incumbent
decides what a depth-first search prunes, and what a search stopped early can
report. It does not decide what a best-first search prunes.

The certified branch and bound (`branch.py`) obeys the same argument at this
size. Its cost model charges a persp+N bound 1.27 modelled seconds at 18 markets
and pricing 48,620 sets 0.05, so it prices the root set by set, in blocks of
20,000 in bound order. Run directly on `section5` and `walmart` at 4, 9 and 11
treated (`run bnb`, `results/bnb.jsonl`), it made no bound and fitted exactly
the designs the enumeration fitted (249, 2,861, 2,811 and 8,331 at 9 and 11).

## The measurements

The five panels, each 18 markets, the target their mean (`panels.py`):

- `section5`: Abadie and Zhao's Section 5 process with 18 markets in place of
  15, drawn with NumPy; fit on the first 20 of 30 periods, on outcomes, as
  `benchmarks/cases/marex_section5_mc.py` configures MAREX.
- `geolift`: 18 of GeoLift's 40 markets over 90 days.
- `walmart`: 18 of the 45 stores as the exact backend sees the paper's Walmart
  design (100 standardised weeks and four covariates).
- `noise` and `factor`: the `marex_bnb` study's synthetic panels at 18 markets
  and 40 periods.

Per panel, over the 17 fixed treated counts. `first / f*` is the objective of
the library's own first incumbent, the first admissible set in its bound order,
over the optimum. `walked` sums the sets reached over the 17 counts. Seconds are
the searches' own, summed.

| panel | greedy / f* | sets the greedy evaluates | first / f*, at most | walked: library | greedy seed | optimum seed | two-sided | seconds: library | two-sided |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| section5 | 1.00 to 2.58 | 18 to 401 | 20.7 | 36,551 | 36,551 | 36,551 | 20 | 6.4 | 1.7 |
| geolift | 1.00 to 1.53 | 18 to 325 | 148.5 | 105,982 | 105,982 | 105,982 | 19 | 22.1 | 1.9 |
| walmart | 1.00 to 2.50 | 18 to 450 | 251.2 | 65,753 | 65,753 | 65,753 | 18 | 12.9 | 1.6 |
| noise | 1.00 to 1.78 | 18 to 273 | 1.7 | 100,741 | 100,741 | 100,741 | 18 | 15.6 | 2.8 |
| factor | 1.00 to 1.29 | 18 to 394 | 1.7 | 184,410 | 184,410 | 182,824 | 508 | 34.0 | 1.7 |

The unconstrained design, every treated count from 1 to 17 in one search:

| panel | sets | first / f* | greedy / f* | walked: library | greedy seed | optimum seed | two-sided | fitted: library | two-sided | seconds: library | two-sided |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| section5 | 262,142 | 2,661 | 1.86 | 20,007 | 20,005 | 19,961 | 2 | 6,749 | 2 | 4.0 | 1.1 |
| geolift | 262,142 | 801 | 1.02 | 105,290 | 104,863 | 104,467 | 2 | 79,361 | 2 | 21.5 | 1.1 |
| walmart | 262,142 | 3,102 | 1.00 | 56,894 | 56,616 | 56,616 | 2 | 28,088 | 2 | 11.4 | 1.3 |
| noise | 262,142 | 18 | 1.07 | 98,690 | 98,095 | 97,335 | 2 | 95,669 | 2 | 14.9 | 1.1 |
| factor | 262,142 | 2 | 1.05 | 179,597 | 179,554 | 178,931 | 68 | 41,608 | 19 | 30.5 | 1.2 |

Findings:

1. The library's first incumbent can be poor. At a fixed count it reached 251
   times the optimum (walmart, 16 treated), and unconstrained 3,102 times. The
   walk does not depend on it: seeded with the optimum, the walk is the same at
   82 of 85 fixed counts. The order does the work the incumbent would.
2. The greedy seed changed nothing at any fixed count, walked or fitted.
   Unconstrained it saved 0.01% to 0.60% of the sets walked; the optimum seed,
   the ceiling for any seed, saved 0.23% to 1.37%. At fixed counts the optimum
   seed saved anything only on `factor`, at 8, 10 and 11 treated, by 1.3%,
   2.9% and 0.1%.
3. The work sits where `m` approaches half the cluster and passes it. At small
   counts the complement is large, its hull holds the target, and the treated
   bound alone is close to the objective: the library walks 1 to 17 sets up to
   3 treated. From 6 to 10 treated on, depending on the panel, the complement is
   small, its fit is poor, and the treated bound cannot see it: more than a
   thousand sets have a bound under the optimum, up to 40,968, and any search
   with that bound walks every one.

## What prunes: the control side of the bound

The closed form that bounds the treated fit bounds `g` on any set of markets,
so for the base design

```
f(S) = g(S) + g(S^c) >= lb(S) + max(h0, lb(S^c)),
```

with `h0` the library's control floor. `search.two_sided_bound` prices every
candidate with it, using the library's own `treated_fit_bound` on the
complements; nothing else in the search changes. It is valid with donor
exclusions too, since an exclusion shrinks a control pool to a subset of the
complement and a fit over fewer markets is no better. With a control penalty
the closed form takes the penalty as its linear term, as the treated side does
(not measured here).
A property in the tests checks it under the objective on every set of random
panels, and the search under it returned the brute-force optimum in all 90
settings.

Measured (tables above, column two-sided): 1 to 141 sets walked at each fixed
count, 18 to 508 summed over the 17 counts against 36,551 to 184,410, and 2 to
68 unconstrained against 20,007 to 179,597. The two-sided walk also needs no
incumbent: it reached exactly the candidates whose two-sided bound is under
`f* + delta(f*)` in all 90 settings, and no setting needed a second pass.

The price is the complement pass. With the library's pricing code the
two-sided search took up to half a second per treated count at 18 markets. That
is more than the library's whole search at small counts (up to 4 to 9 treated,
depending on the panel), where the library takes at most 0.17 seconds, and far
less from 5 to 10 treated on, where the library takes up to 7.4 seconds. A
build would apply the complement bound where it can pay: under the cost rule of
`agents_combinatorial.md` (rule 6), or as a screen on the candidates the treated
bound leaves under the cutoff, the way the weakly-targeted search screens with
its coupling bound. A screen inside the walk gives the incumbent a role again,
for the candidates the walk reaches before `S*`.

## The MIQP backend

`backend="mip"`, the default, hands the program to SCIP. SCIP's search dives
depth first between best-estimate choices, so the argument above does not
apply to it, and a warm start can change what it prunes.

At the library's settings the warm start does not reach SCIP on two of the
three panels run. MAREX's `warm_start` fixes the 18 selection variables and
leaves SCIP's `completesol` heuristic to complete the rest, and SCIP drops a
partial solution when more than 85% of its variables are unknown
(`heuristics/completesol/maxunknownrate`). On the two panels with many fit
periods the share passes that limit: SCIP reported 0.933 on `walmart` (104 fit
periods) and 0.925 on `geolift` (90), and never called `completesol`. On
`section5` (20 periods) it completed the start and took it as its first
solution (`run seedcheck`, `results/seedcheck.jsonl`, with the optimum as the
start at 2 treated). The `warm_start` field describes a start SCIP completes;
on these two panels SCIP discards it and nothing reports that.

`run scip` therefore raises that limit to 1 in its warm-started solves, so the
comparison below is of starts SCIP used, and records per solve whether SCIP
found a solution from the start (`seeds used`, greedy and optimum). It solves
each treated count three ways: cold, from the greedy design, and from the
optimum. GeoLift ran at 2 to 5 treated only, its solves taking a minute or more
each.

| panel | m | exact seconds | greedy / f* | cold nodes | greedy-seed nodes | optimum-seed nodes | seeds used | cold seconds | greedy-seed seconds | optimum-seed seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| section5 | 2 | 0.002 | 1.14 | 165 | 135 | 101 | yes/yes | 2.4 | 2.4 | 2.2 |
| section5 | 3 | 0.002 | 2.01 | 393 | 451 | 435 | yes/yes | 4.0 | 4.9 | 4.9 |
| section5 | 4 | 0.005 | 2.42 | 986 | 1,049 | 989 | yes/yes | 8.8 | 9.4 | 10.0 |
| section5 | 5 | 0.012 | 1.36 | 1,386 | 1,149 | 1,008 | yes/yes | 12.9 | 9.5 | 7.6 |
| section5 | 6 | 0.043 | 2.31 | 1,517 | 1,417 | 1,513 | yes/yes | 11.2 | 11.2 | 11.9 |
| section5 | 7 | 0.087 | 2.58 | 1,323 | 1,217 | 693 | yes/yes | 10.9 | 12.7 | 5.0 |
| section5 | 8 | 0.103 | 1.70 | 702 | 273 | 215 | yes/yes | 6.5 | 2.7 | 2.4 |
| section5 | 9 | 0.231 | 1.80 | 678 | 425 | 349 | yes/yes | 5.4 | 4.1 | 3.2 |
| geolift | 2 | 0.002 | 1.17 | 108 | 108 | 108 | no/no | 64.7 | 64.2 | 63.5 |
| geolift | 3 | 0.003 | 1.53 | 186 | 222 | 237 | yes/yes | 62.9 | 26.4 | 20.6 |
| geolift | 4 | 0.005 | 1.12 | 1,177 | 728 | 634 | yes/yes | 107.4 | 90.0 | 30.3 |
| geolift | 5 | 0.014 | 1.20 | 1,813 | 1,425 | 799 | yes/yes | 85.3 | 66.0 | 45.4 |
| walmart | 2 | 0.002 | 2.50 | 75 | 76 | 55 | yes/yes | 10.3 | 10.6 | 6.8 |
| walmart | 3 | 0.002 | 1.29 | 548 | 449 | 416 | yes/yes | 19.1 | 17.0 | 16.7 |
| walmart | 4 | 0.005 | 1.34 | 737 | 654 | 726 | yes/yes | 23.5 | 22.4 | 21.8 |
| walmart | 5 | 0.012 | 1.27 | 954 | 1,245 | 946 | yes/yes | 28.4 | 34.4 | 27.1 |
| walmart | 6 | 0.033 | 1.02 | 1,664 | 963 | 859 | yes/yes | 50.5 | 25.9 | 23.8 |
| walmart | 7 | 0.074 | 1.02 | 687 | 930 | 635 | yes/yes | 19.1 | 29.8 | 17.9 |
| walmart | 8 | 0.322 | 1.06 | 438 | 338 | 245 | yes/yes | 14.3 | 12.4 | 10.1 |
| walmart | 9 | 0.984 | 1.00 | 616 | 313 | 287 | yes/yes | 22.4 | 13.6 | 11.2 |

Every solve ended optimal, at the exact backend's objective.

Findings:

1. A warm start helps SCIP more often than not, by an amount it cannot
   predict. Against a cold start the optimum as the start moved the node count
   by -69% to +27%, with fewer nodes in 16 of the 19 settings where SCIP used
   it; the greedy design by -61% to +35%, fewer in 13. On `geolift` with 2
   treated SCIP found no solution from either start, even with the limit
   raised, and solved node for node as from cold.
2. The incumbent is not what SCIP lacks. With the optimum installed at 0.1 s on
   `section5` with 2 treated, its root dual bound was `-5e-10`: the continuous
   relaxation scores the design that spreads the weights over every market at
   zero (`agents_combinatorial.md`, fact 2). Nothing is pruned until branching
   has raised the bound, and that is where its nodes go.
3. The exact backend solved each of these designs in 0.002 to 0.98 seconds,
   11 to 39,000 times faster than SCIP from any start.

## Recommendations

- Do not add a greedy incumbent to the exact backend. At this size it cannot
  prune the enumeration beyond its second pass, by the argument above, and
  there the measured ceiling is under 3% of the sets walked. It costs up to
  450 set evaluations, more than the whole search at small treated counts.
- Build the control side of the bound for the separable designs (`standard`,
  `penalized`, `unit_penalized`), on its own branch and issue: the complement's
  closed form in the search, with its certified counterpart in
  `ExactData.treated_bounds` for `certify=True`, applied where the cost rule
  says it pays. Its measured ceiling at 18 markets is 1 to 141 sets fitted per
  treated count where the library fits up to 40,968.
- Make MAREX's `warm_start` reach SCIP, on its own branch: let SCIP complete a
  start that fixes only the selection (`maxunknownrate` at 1), or hand it a
  complete solution with the weights the two fits give, and report when SCIP
  discards a start. As it stands the field does nothing on panels with many
  fit periods.
- On an 18-market cluster, prefer `backend="exact"` to the MIQP for the base
  design. Where the MIQP is used, the greedy design is a cheap start (0.002 to
  0.14 s) that cut SCIP's nodes in 13 of 19 settings and added to them in 6.

## How it was measured

One machine with four cores visible: Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
cvxpy 1.9.3, PySCIPOpt 6.3.0. BLAS threads pinned to one. The two enumeration
runs shared the machine with two other jobs, and the three SCIP runs with each
other, so a time here can be off by a few tens of percent; the comparisons
drawn are between times a factor of three or more apart. Counts are
deterministic.

## References

- Abadie, A. and Zhao, J. (2026). Synthetic Controls for Experimental Design.
  Working paper, February 2026.
- Dechter, R. and Pearl, J. (1985). Generalized best-first search strategies and
  the optimality of A*. Journal of the ACM 32(3): 505-536.
- Morrison, D. R., Jacobson, S. H., Sauppe, J. J. and Sewell, E. C. (2016).
  Branch-and-bound algorithms: A survey of recent advances in searching,
  branching, and pruning. Discrete Optimization 19: 79-102.

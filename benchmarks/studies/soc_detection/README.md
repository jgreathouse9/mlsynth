# Why SCIP's cone handler declines MAREX's objective

MAREX hands its design program to SCIP through cvxpy. Each `sum_squares`
term reaches SCIP as a second-order cone, and SCIP has a handler built for
exactly that shape, `nlhdlr_soc`, with the highest detection priority of any
nonlinear handler. On MAREX's program it takes nothing: every cone row goes
to the generic `nlhdlr_default`. This study is the root-cause analysis of
that, written down as tests in `benchmarks/tests/test_soc_detection_ladder.py`.

Everything here was measured on SCIP 10.0.2 (PySCIPOpt 6.2.1, cvxpy 1.9.3),
and the SCIP source cited is the `v10.0.2` tag, which is the code that ran.

## The answer

cvxpy writes `sum_squares(r) <= x` with the identity for a rotated cone,

    || (2r, 1 - x) ||  <=  1 + x,

so in the quadratic row SCIP receives,

    s_1^2 + ... + s_n^2 + u^2 - tau^2 <= 0,   u = 1 - x,   tau = 1 + x,

the cone's right side `tau` and one of its left components `u` are affine in
the same variable. Two simplifications then act in sequence.

1. Aggregation. Presolve reads the two defining equalities and substitutes
   one variable for the other, `tau = 2 - u`.
2. Expansion. The expression simplifier expands the square of a sum (rule
   POW7 in `expr_pow.c`, on while `expr/pow/expandmaxexponent >= 2`, the
   default), so `u^2 - (2 - u)^2` becomes `-4 + 4u`.

What reaches the handlers is `-4 + 4u + sum s_i^2 <= 0`: a sum of squares
bounded by a linear term, which is the convex quadratic cvxpy started from.
`nlhdlr_soc` detects a sum of squares minus one square, a sum of squares
minus one bilinear term, or a quadratic with exactly one negative eigenvalue
(`nlhdlr_soc.c`, the documentation of `detectSOC`). This row is none of the
three, so `nlhdlr_default` takes it, builds one auxiliary per square, cuts
each from below by tangents, and sums them in one linear row.

Neither simplification is a SCIP defect. Both are correct, and together they
undo an encoding that exists for solvers that consume cones natively.

## The ladder

The chain the reported statistic depends on, each link depending only on the
ones to its right:

    "soc DetectAll = 0"  <-  the handler's detect callback
      <-  the row as presolved  <-  presolve (aggregation, simplification)
      <-  the row as written  <-  cvxpy's encoding of sum_squares
      <-  MAREX's objective

### Rung 0 and 1: what was observed, and what moved with it

`detect.py` runs MAREX's own program, built through the library's
formulation helpers. At J = 12, 16 and 20 the SOC handler participates 0
times and `default` takes all 76 detections; the two cone rows cost 4,627,
15,554 and 38,605 cuts.

The first link cleared was the statistic itself. `DetectAll` reads like a
count of times the handler was asked. It is not: `nlhdlr.c` increments both
detection counters only under `if( *participating != SCIP_NLHDLR_METHOD_NONE )`,
so they count participations, and a zero cannot distinguish "never asked"
from "asked and declined".

The link that failed was the presolved row. Reading it directly, instead of
the row cvxpy wrote, shows the negative square is gone:

    as written:   <s40>*<s40> + ... + <s58>*<s58> - <s39>*<s39> <= 0
    presolved:    -4 + 4*<s40> + (<s41>)^2 + ... + (<s58>)^2 <= 0

with `s39 = 1 + x0` and `s40 = 1 - x0` in the written model.

### Rung 2: two causes, each necessary

`causes.py` switches each mechanism off alone on the written MAREX model
(`results/marex.cip`) and reads the presolved row back.

| configuration | negative squares after presolve | SOC participations | nodes | LP iterations | cuts |
|---|---|---|---|---|---|
| baseline | 0 | 0 | 36 | 9,864 | 4,989 |
| no aggregation (`presolving/donotaggr`) | 2 | 2 | 29 | 5,216 | 2,860 |
| no expansion (`expr/pow/expandmaxexponent = 1`) | 2 | 2 | 53 | 7,354 | 3,648 |
| neither | 2 | 2 | 29 | 5,216 | 2,860 |

Either correction alone restores the negative square and the handler takes
both rows, so the failure is the conjunction of the two. The causes are
conjunctive, not independent faults, so their contributions do not multiply:
removing either removes the whole effect.

`minimal.py` reproduces the failure in six squares and three continuous
weights, with no integers, no MAREX and no cvxpy, and pairs it with a twin
that differs in one feature: `u` is defined from a variable of its own,
`u = 1 - y`, so nothing ties it to `tau`.

| model | baseline | no aggregation | no expansion |
|---|---|---|---|
| tied (cvxpy's encoding) | 0 squares, SOC 0 | 1, SOC 1 | 1, SOC 1 |
| untied twin | 1, SOC 1 | 1, SOC 1 | 1, SOC 1 |

### Rung 3: the statement over the domain

Four `hypothesis` properties, and a one-off sweep of 400 random cases
(`n` from 1 to 20 squares, `k` from 1 to 5 weights, random coefficients) under
every configuration and both encodings, 3,200 solves with no violation:

- a tied cone never survives default presolve;
- either correction alone always restores it;
- an untied cone always survives;
- the optimum does not depend on the configuration.

One of them found a case, `n = 3`, `k = 2`,
where presolve flipped a row to `expr >= lhs`; the handler still took it,
but the square counter, which assumed every row is an upper bound, reported
two negative squares where the cone had one.

### Rung 4: the contracts the study ran without

The study's instruments were never tested against known answers, and five of
them were wrong:

1. the cut count read a wrong column of SCIP's constraint table;
2. `DetectAll` was read as a count of invocations (above);
3. a regular expression for negative squares could not see `-((2-<x>))^2`;
4. its replacement assumed every row is `<=`;
5. it also read only the `(<x>)^2` notation SCIP uses after presolve, while a
   row as written spells the same square `<x>*<x>`, so on every unpresolved
   row it counted zero.

All of them now live in `instruments.py`, tested against known rows and
against `results/stats_marex_J12.txt`, a captured statistics file that holds
three rows named `nonlinear` in three different tables, only one of which
carries the cut counts.

The second contract: a probe meant to stand for MAREX has to reproduce
MAREX's outcome before anything is concluded from it. The ladder enforces
it with the tied/untied pair.

### Rung 5: mutation

Eleven mutants in `tools/mutation/targets.toml` across three targets
(`soc-detection-reproducer`, `soc-detection-causes`,
`soc-detection-instruments`), one per defect found: the reproducer written as
a standard cone, the tie broken, each correction made a no-op, and each
instrument defect reintroduced, plus the over-correction that would count a
bilinear term as a square. All are killed.

## Confirming the bottom

For aggregation, and separately for expansion:

- Would the failure still have occurred without it? No. Switching it off
  alone restores participation on the MAREX model, the minimal model and all
  400 random cases.
- Will it recur if it alone is corrected? No, on the same evidence.

A third necessary condition sits beside the two, and it was found by reading,
not by correction: the handler has no case for a sum of squares bounded by a
linear term. It cannot be switched off without changing SCIP, so it is
recorded and not counted as a measured cause.

## What it costs

A handler that declines a row is a value that is wrong; whether it did damage
is a separate question. One instance has no power to answer it, since SCIP's
effort counts move by large factors under a reordering of rows and columns,
so `damage.py` solves nine MAREX programs (J = 12, 16, 20, three panels each)
under six orderings and compares each configuration to the baseline on the
same (program, ordering). Across the 54 pairs the optimum agrees to 2.3e-07.

| configuration | LP iterations | cuts | nodes | root bound higher |
|---|---|---|---|---|
| no aggregation | 0.66x, better in 51/54 | 0.63x, 52/54 | 0.88x, 35/54 | 23/54 (median -0.039) |
| no expansion | 0.70x, 48/54 | 0.66x, 48/54 | 0.93x, 31/54 | 26/54 |

Geometric means of the paired ratios. The SOC handler's relaxation is not
stronger -- the root bound is higher in fewer than half the pairs -- it is
cheaper to separate: a third fewer LP iterations and cuts for the same convex
set.

`damage.py` ran beside other solves, so its time column is contaminated and
not used. `timing.py` measures wall time alone, three alternating repeats per
pair, median kept: the handler taking the row runs at 0.857x on the geometric
mean, faster in 10 of 16 pairs, with individual ratios from 0.54 to 1.31.

## Blast radius

Downstream: the failure is confined to the relaxation SCIP builds. The
optimum is the same under every configuration, so nothing a MAREX result
reports is affected; the cost is search effort.

Sideways: the collapse needs a quadratic atom and cvxpy's SCIP interface
together. `exposure.py` checks the other estimators that reach SCIP.

- SYNDES's default for `two_way_global` is `backend='exact'`, which searches
  treated sets directly and makes no cvxpy solve at all, so it never reaches
  SCIP. Its MIP path does: with `backend='mip'`, and in `one_way_global` and
  `per_unit`, which force it, every captured program has two cone rows with
  one negative square each as written, none after presolve, and no SOC
  participation -- MAREX's signature exactly.
- PANGEO's MIP builds no quadratic atom, and PDA's HCW writes its model in
  PySCIPOpt directly, so neither can produce the tied encoding.

## Wrong turns

Each of these is the branch the next person would otherwise take.

- The first feature bisection (`feature_bisect.py`) rebuilt the cone with
  MAREX's features added one at a time and every case detected. The
  conclusion drawn -- that the cause was in cvxpy's coefficients and not in
  the shape -- was wrong. Every case encoded a standard cone, `sum s^2 <= t^2`,
  with no left component tied to `t`, so none of them could reach the cause.
- The wrong source. The SCIP source first read was `master`
  (11.0.0-dev). The runtime is 10.0.2.
- An aggregation hypothesis, measured and rejected. The first guess was
  that aggregating `t` against the objective variable turned `t^2` into an
  offset square the detector does not handle. A probe with exactly that tie
  still detects. Aggregation matters, but through the cancellation between
  two squares, not through an offset.
- A narrower fix that is worse. Letting the convex handler take rows
  whose root is a sum (`nlhdlr/convex/detectsum`, off by default) is not a
  lighter version of either correction (`alternatives.py`): nodes 2.96x,
  better in 1 of 54 pairs. It cuts the whole sum at once, and one aggregated
  gradient cut per round is far weaker than one per term.
- A false alarm. An untied row printed with a stray constant `1` looked
  like presolve had fixed a free variable at the wrong value. The row had
  been truncated in the printout; its tail reads `-2*<y>+(<y>)^2`, the same
  expansion rule applied to `(1 - y)^2`, and the optimum is identical with
  presolve on and off.

## What would change it, and what was not changed

Nothing in MAREX was changed here. The options, with what was measured:

- `presolving/donotaggr`: the handler takes the rows, LP iterations 0.66x. It
  is a global switch that stops aggregation everywhere in the program, and
  its effect on MAREX's other configurations (clusters, budgets,
  restrictions, covariates) is unmeasured.
- `expr/pow/expandmaxexponent = 1`: the handler takes the rows, LP
  iterations 0.70x. Also global, over every squared sum in the program.
- `nlhdlr/convex/detectsum`: worse, above.
- Not reaching SCIP at all. SYNDES's exact backend already searches treated
  sets directly for `two_way_global`; a MAREX design that did the same for
  the configurations where it applies would make the question moot there.

A solver-parameter change to MAREX is a change to a shared estimator and
belongs on its own branch, with the measurement repeated across MAREX's
configurations before it is adopted.

## Files

| file | what it does |
|---|---|
| `panels.py` | MAREX's program, built through the library's formulation helpers |
| `detect.py` | the observation, at J = 12, 16, 20 |
| `feature_bisect.py` | the first bisection, kept as the record of a wrong turn |
| `reread.py` | writes `results/marex.cip` and shows the zero reproduces from the file alone |
| `causes.py` | the 2x2: each mechanism switched off alone |
| `minimal.py` | the six-square reproducer and its untied twin |
| `instruments.py` | statistics parsing and the negative-square counter |
| `damage.py`, `analyze_damage.py` | effort, paired over 54 (program, ordering) pairs |
| `timing.py` | wall time on a quiet machine |
| `alternatives.py` | the convex-handler alternative |
| `exposure.py` | the other estimators that reach SCIP |

    cd benchmarks/studies/soc_detection
    python detect.py results/detect.csv
    python causes.py results/causes.csv
    python minimal.py
    python damage.py results/damage.csv && python analyze_damage.py results/damage.csv
    python timing.py results/timing.csv
    python alternatives.py results/damage.csv results/alternatives.csv
    python exposure.py
    python -m pytest ../../tests/test_soc_detection_ladder.py -q

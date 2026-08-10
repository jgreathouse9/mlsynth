# Spike: greedy forward selection for treated-unit design

De Geest, L. R. and Wang, A. (2025). "Designing Synthetic Control Experiments
with Forward Selection." CODE '25, Cambridge MA. Five pages, no replication
package, no public reference implementation.

Verdict: pass as a standalone estimator. The search is fast and it works, but
its objective and its stopping rule are the parts that decide the answer, and
both are weaker than what mlsynth already has. The search itself is cheap to
harvest as a backend if it is pointed at a different criterion.

## What the method is

A synthetic control experiment ("region split") treats whole markets instead of
individual users, so that interference between users — a discount in San
Francisco changes the experience of untreated San Francisco riders — does not
contaminate the control group. Which markets to treat is then a design choice,
and the space of choices is combinatorial.

The paper's proposal: pick the treated markets greedily. Score every single
market as the treated unit, keep the best by pre-treatment fit, then repeatedly
add whichever remaining market most improves the fit, stopping when the
improvement falls below a tolerance. Any synthetic control estimator can sit
inside the loop; the paper uses Forward DiD (Li 2024), which builds the
counterfactual as a simple average of selected donor markets plus an intercept.

## What was ported, and two departures from the paper as printed

`fwdsel.py` implements the flow chart on p.2 — the procedure the prose
describes. The inner estimator is mlsynth's own `forward_did_select`, so this is
a port of the design search, not a second Forward DiD.

Algorithm 1 on the same page is not that procedure. As printed it fits a fixed
target `y_pre` with a simple average over selected columns of `X_pre`, which is
Forward DiD's donor step (Table 1c). There is no inner synthetic control per
candidate treated set, so the flow chart's "Construct synthetic control(s) for
each experiment" has no counterpart in the pseudocode. An implementer has to
reconstruct the outer loop from prose.

Table 1(d) gives the treatment effect as the gap averaged over `t = 1..T1`, the
pre-treatment periods, with no intercept subtracted. Li's estimator averages
over the post-treatment periods and subtracts the fitted intercept. The port
uses mlsynth's validated version.

One further collision: `tau` is the stopping tolerance in Algorithm 1 and the
treatment effect in Table 1(d) and Table 2.

## Data and setup

`basedata/geolift_test_data.csv`: 40 US markets, 105 daily periods,
2021-01-01 to 2021-04-15. Ingested through `prepare_syndes_inputs`, the design
family's path. T0 = 91 pre-treatment days, a 14-day post window.

Reproduce with `python run_spike.py --stage all`.

### A measurement correction that changes the reading

`design_compare.compare_pareto` reports the raw contrast RMSE. That is right for
SYNDES, LEXSCM and MAREX, whose analyses are pure weighted contrasts, but it
charges a Forward DiD design for a constant level gap that the intercept
removes. Scored that way, greedy at K=5 looked worse than the median random
design. On the demeaned gap — what both analysis families actually see, and what
the MDE simulation already used internally — greedy is at the 100th percentile.
Every fit number below is demeaned.

## Findings

### Confirmed: the search beats random designs, and it is fast

Among 400 random treated sets of the same size, scored with the same estimator,
greedy sits at the 100th percentile on its own objective, the 99.8th–100th on
demeaned pre-period fit, and the 93.5th–99th on the minimum detectable effect
(MDE — the smallest lift the experiment would reliably catch). The paper's
central claim holds on real geo data.

It also holds against a stronger baseline than the paper uses. Given the same
number of Forward DiD evaluations that greedy spends (K × N) and the same
selection criterion, random search beats greedy's objective in 4% of runs at
K = 3 and 0% at K = 4 and K = 5.

Timing on this panel, 40 markets:

| Procedure | K = 3 | K = 5 |
| --- | --- | --- |
| greedy | 0.21s | 0.36s |
| exhaustive enumeration | 17.8s | infeasible |
| SYNDES annealed relaxation | 47.2s | 40.8s |
| SYNDES exact MIP (6-design pool) | did not finish in 15min of CPU | — |
| LEXSCM | — | 37.7s |

The premise that the mixed-integer program is slow is validated here.

### Confirmed: pooling selects more treated units than the separate approach

At a common tolerance of 1e-3, the pooled objective (one synthetic control for
the treated average) selects K = 5; the separate objective (one synthetic
control per treated unit) selects K = 1. This is the paper's p.4 claim.

### Not confirmed: pooling estimates better than separate

Section 4.1 and Table 2 claim pooling "performs much better." Holding the
treated set fixed across the two arms — the only way to attribute a difference
to the pooling choice — and rolling the pseudo-treatment date through the panel
for 16 placebo windows:

| Scenario | Arm | Mean | Bias | SD | RMSE |
| --- | --- | --- | --- | --- | --- |
| A/A, truth 0% | pooling | −0.73% | −0.73pp | 1.54 | 1.66 |
| A/A, truth 0% | separate | −0.78% | −0.78pp | 1.41 | 1.57 |
| +10% injected | pooling | +8.43% | −1.57pp | 1.40 | 2.07 |
| +10% injected | separate | +8.38% | −1.62pp | 1.28 | 2.04 |

Pooling is closer to the truth in 50% of windows under the injected effect and
31% under the placebo. The two arms are indistinguishable, and the separate
arm's variance is slightly lower — the opposite of Table 2's direction, where
`Var(tau_separate)` climbs from 3.4 to 15.6.

This does not refute the paper's simulation. These markets are highly
correlated and fit well (R² about 0.99), so the panel sits in the low-noise
regime where Table 2 also shows small differences. It does mean the claim should
not be carried over to a real geo panel without checking. Table 2 reports a
variance for the separate arm only, so the comparison it asserts cannot be
checked from the paper itself.

### New: the objective is a weak proxy for the decision

Greedy reliably wins its own objective and does not reliably win the quantity a
designer chooses on:

| K | Greedy R² | Random-search R² | Greedy MDE | Random-search MDE |
| --- | --- | --- | --- | --- |
| 3 | 0.99310 | 0.99040 | 1.25% | 2.25% |
| 4 | 0.99450 | 0.99210 | 2.25% | 2.00% |
| 5 | 0.99560 | 0.99290 | 1.75% | 1.75% |

Greedy wins the objective in 96–100% of matched-budget runs, and wins the MDE at
K = 3, loses at K = 4, ties at K = 5. Across random designs the rank
correlation between the objective and the MDE is only −0.57 (K = 3) and −0.63
(K = 5).

### New: greediness costs power, and the market list is unstable

Against exhaustive enumeration over every C(40, K) treated set, scored by the
same estimator:

| K | Exact R² | Greedy R² | Gap | Shared markets | Exact MDE | Greedy MDE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.97990 | 0.97990 | 0 | 1/1 | 3.25% | 3.25% |
| 2 | 0.99080 | 0.98920 | 1.6e−3 | 1/2 | 2.50% | 2.25% |
| 3 | 0.99370 | 0.99310 | 6e−4 | 0/3 | 0.75% | 1.25% |
| 4 | 0.99510 | 0.99450 | 6e−4 | 3/4 | 1.25% | 2.25% |

The objective gap is negligible and the consequences are not. At K = 3 the
optimum and greedy share no markets at all (atlanta + dallas + jacksonville
against denver + milwaukee + saint paul) and the optimum detects a 0.75% lift
where greedy needs 1.25%. At K = 4 the optimum needs 1.25% against greedy's
2.25%. Section 4.2 anticipates that an early pick can be the wrong one; it
starts biting at K = 2.

### New: the stopping rule does not select the best design on its own path

Along the greedy path the objective peaks at K = 10, so Algorithm 1's line 22
returns K = 10. The MDE-minimising size on that same path is K = 7 (0.50%
against 0.75%). The objective keeps creeping up — 0.99690, 0.99720 — while the
quantity the experiment lives on gets worse.

The design size is a function of a tolerance the paper never states:

| Tolerance | 0 | 1e−4 | 5e−4 | 1e−3 | 2e−3 | 5e−3 | 1e−2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K* | 10 | 10 | 5 | 5 | 3 | 2 | 1 |

## Recommendation

Pass on building this as a standalone estimator. mlsynth already covers the
design axis with SYNDES (the mixed-integer program this paper positions against,
including a simulated-annealing relaxation for when the solver is too slow),
LEXSCM (treated-unit selection by multi-start local search, with an explicit
power report), MAREX, SPCD, PANGEO and GEOX. Nothing here clears the replication
contract either: the empirical section is proprietary Lyft data, the simulation
does not fix the number of factors, the AR(1) coefficient, the two variance
components, T, T0, or the tolerance, and Table 2's caption labels the varying
parameter as the between-cluster variance while the text says it is the
idiosyncratic variance.

The part with value is the search, not the paper's use of it. A
`search="greedy"` backend on LEXSCM's Stage 1 or as a `SYNDES.mode` — beside the
existing `two_way_global_annealed`, already flagged as an mlsynth extension —
would give a sub-second path where the MIP takes minutes and enumeration is
impossible. It should select on an MDE-aware criterion, which
`design_compare.simulated_mde` already computes, and not on pre-treatment R².
That is a repo enhancement with its own justification, so it needs no
replication path.

## Unrelated finding: a SYNDES crash

`SYNDES(mode="two_way_global", top_K=1)` on this 40-market panel dies with

    TypeError: float() argument must be a string or a real number, not 'NoneType'

from `syndes_helpers/certificate.py:130`, where `_sdp_moment_bound_two_way`
returns `float(obj.value)` with no check that the SDP solved. The SCS solve on
the line above is capped at 8000 iterations and the relaxation is 120 × 120 at
N = 40, so it returns `None`. The caller is the acceleration path
(`accelerate.py:65`), and `accelerate` defaults to `True`, so this hits the
default configuration. The neighbouring `solve_synthetic_design` guards
`D.value is None` with an actionable message; this path has no equivalent guard,
and the natural fix is to skip the acceleration cut when the bound does not
solve. Separate scope from this spike.

# RCA: why the enumerative solver beat the certified branch and bound

## Incident

On GeoLift (40 markets, 90 pre-periods), the library's exact backend
(`search_cluster`, its 20 million candidate limit raised) and the certified
best-first search (`certified_bfs.search`) return the same design for every
treated count tried. Their times:

| treated | sets | exact backend | certified_bfs | ratio |
| --- | --- | --- | --- | --- |
| 4 | 91,390 | 0.30 s | 758 s | 2,500 |
| 6 | 3,838,380 | 3.94 s | 1,302 s | 330 |
| 8 | 76,904,685 | 105.1 s | 2,333 s | 22.2 |

Expectation: a branch and bound that bounds 20 nodes should not lose to one
that prices 77 million sets. Estimand: wall time to a design whose optimality
is established (certified for the branch and bound, floating point for the
backend).

## Reproduction

Deterministic: GeoLift is fixed data and both searches are deterministic. The
smallest reproduction is m = 4 (0.30 s against 758 s).

## Dependency chain

```
wall time  <-  units of work x cost per unit
           <-  (branch and bound) node bounds + priced sets + overhead
           <-  (enumeration) generated sets + priced sets + fitted sets
```

## Rung 2: which term carries the time (decomposition, m = 8)

Exact backend, instrumented (`side_fit`, `treated_fit_bound`, `_chunks`
wrapped):

| m | generate | price | fits | other | per set |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.01 s | 0.10 s (1.14 us/set) | 2 (0.18 s, first-call cost) | 0.01 s | 3.34 us |
| 6 | 0.60 s | 3.28 s (0.85 us/set) | 10 (0.002 s) | 0.06 s | 1.03 us |
| 8 | 15.66 s | 88.26 s (1.15 us/set) | 22 (0.004 s) | 1.15 s | 1.37 us |

The backend's closed-form bound, `1 / 1'G_S^-1 1` batched over blocks of
20,000 sets, rules out all but 22 of 77 million sets: its time is pricing.

certified_bfs at m = 8 (`results/certified_bfs_geolift_m8.log`): 33 lifts take
488 s (treated) + 1,845 s (full) = 2,333 s of 2,333 s; 49 priced sets and the
exact certificates are under a second.

Factors: units 33 / 76,904,685 = 4.29e-7; cost per unit 70.7 s / 1.37 us =
5.18e7. Product 22.2, the observed ratio.

## Branch A: why a lift costs 70.7 s

| # | candidate | result | evidence |
| --- | --- | --- | --- |
| A1 | cvxpy model building | cleared | 0.0-0.1 s per node (`node_solvers.log`) |
| A2 | the 13,000 nonnegativity constraints | cleared | without any, the root still takes 96.5 s |
| A3 | the exact certificate | cleared | 0.28 s at GeoLift's size |
| A4 | the iteration count | cleared | 28-54 iterations, ordinary for interior point |
| A5 | per-iteration cost of the PSD block | cause | 0.22 s at K = 56, 0.66 s at K = 76, 4.5 s at K = 116 |
| A6 | the full lift's share | contributing | 13 of 33 lifts, 79% of the time, at a 30% margin |

Below A5: first-order solvers, which avoid the dense Newton block, do not
certify on this objective (ADMM, a spectral bundle method, LoRADS, SCS;
`node_solvers.log`). The optimum is about 2,000 times smaller than the
objective's scale.

## Branch B: why a priced set costs 1.37 us

| # | candidate | result | evidence |
| --- | --- | --- | --- |
| B1 | the bound is batched over a precomputed Gram | cause | one m x m solve per set, 20,000 sets per call |
| B2 | the bound is tight, so fits are rare | cause | 22 fits of 77 million (2.9e-7) |

Neither is a fault: they are why the backend is fast.

## Branch C: why the branch and bound paid for 33 lifts

| # | candidate | result | evidence |
| --- | --- | --- | --- |
| C1 | too many nodes | cleared | 20 nodes; the tree is small |
| C2 | the bound-or-enumerate rule | cause | a fixed 64-set threshold, no units; a lift is worth about 52 million priced sets, so every lift after the root bounded a subtree cheaper to enumerate |
| C3 | its own enumeration is per-set exact pricing | contributing | two fits and two exact certificates per set, milliseconds each; with the threshold raised it would have been the bottleneck |

## Causes found while fixing

Removing the 64-set threshold exposed three more, each by a test:

4. **Lifts accepted a node with no free market** (rung 4: a contract never
   enforced). Such a node is one treated set; `treated_lift` built a PSD block
   one column short and raised `IndexError`. The threshold had kept leaves
   away from the lifts by accident. Now `_require_free` refuses them and
   `should_bound` never bounds a single set.
5. **The price calibration overstated the per-set cost** (21-37 us against 6.5
   us actual), because it timed a 20,000-set batch where fixed overheads
   dominate. At m = 6 this made the rule bound the root (93 s of lifts) where
   enumeration costs 25 s. Now it times the enumeration's own path on 200,000
   sets.
6. **The certified enumeration rebuilt each Gram block from the data** (about
   1,000 flops per set at m = 6, against the backend's m^2 reads). Now it reads
   the panel's floating-point Gram and bounds the Gram's own rounding,
   `|Gf - G| <= gamma_T |B|'|B|`; its price is 1.0-2.6 us per set.

## Confirmation

| cause | absent, does the gap remain? | corrected alone, does it recur? |
| --- | --- | --- |
| A5 lift cost | no: at 3.2 s per lift, 33 lifts beat 105 s | no |
| C2 no cost rule | no: subtrees under the break-even are enumerated | no |
| C3, 6 per-set exact pricing | no: with the Gram-based certificate the enumeration is in the backend's class | no |
| 5 calibration | no: m = 6 goes to enumeration (6.9 s) | no |
| 4 leaf lifts | (a crash, not a slowdown) | no |

A5 is structural: no rule makes a 70 s bound cheap. C2 is the fault of omission
the search could act on: the invariant "bound a node only when enumerating it
would cost more" was never stated.

## Blast radius

Study code only (`benchmarks/studies/marex_bnb`). The library's exact backend
is the reference and is untouched. Downstream of C2 is only the search's time,
never its answer: every run returned the certified optimum.

## Fix and verification

`hybrid.py`: `should_bound(sets, price_cost, lift_cost, kappa)` and
`certified_enumerate`. Tests in `test_hybrid.py` (21; 100% line coverage of
`hybrid.py` with one unreachable branch marked):

- rung 3, the cost rule: Hypothesis properties for exactness and monotonicity;
- rung 3, the enumeration: never above the exact fit, and as tight as the
  library's closed form;
- the hostile near-collinear panel;
- rungs 0 and 4, end to end: the brute-force optimum, ties reported,
  infeasible children, leaf lifts refused.

Mutants (applied in place, tests run, file restored):

| mutant | models | result |
| --- | --- | --- |
| cost-rule-is-a-fixed-threshold | the 64-set rule | killed |
| never-enumerates | the rule computed and ignored | killed |
| leaf-bounded | a single set bounded | killed |
| leaf-lift-accepted | a lift built with no free market | killed |
| kappa-unchecked | a kappa below one accepted | killed |
| enumeration-reads-the-data | Gram blocks rebuilt from the data | killed |
| all-slack-dropped | the floating-point dual taken as exact | killed |
| gram-rounding-ignored | the certificate trusts the floating-point Gram | killed (hostile collinear panel) |
| product-rounding-ignored | the products with c taken as exact | survived |
| closed-form-taken-as-certified | 1/1'x used in place of the dual | survived |

The two survivors are not equivalent: the proof needs the product term, and
1/1'x is not a certified quantity. Each is masked by a larger slack term. The
Gram term dominates the product term whenever T >= m, since
`|B|'|B| >= |G|` entrywise and `gamma_T >= gamma_m`. That slack also exceeded
the solve's error in every panel probed: 400 tiny-T ill-conditioned panels and
1,080 hostile ones. No constructed panel distinguishes them; the terms stay
in.

Data, GeoLift, the cost-ruled search (`run_hybrid.py`,
`results/hybrid_geolift.log`):

| treated | exact backend | certified_bfs | hybrid | lifts | certified gap |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.30 s | 758 s | 0.1 s | 0 | 1.7e-13 |
| 6 | 3.94 s | 1,302 s | 6.9 s | 0 | 1.6e-13 |
| 8 | 105.1 s | 2,333 s | 205.5 s | 7 | 2.8e-13 |

At 8 treated the search lifts 7 nodes (193 s) and enumerates the 5,085,465
sets they do not prune (12 s). The pruned 72 million sets would have cost about
192 s to enumerate at 2.67 us each, so the lifts broke even, which is what
`kappa = 1` asks for. The remaining factor against the backend,
205.5 / 105.1 = 1.96, is the certified price against the uncertified one,
2.67 / 1.37 = 1.95. Certified enumeration of all 77 million sets would take
the same 205 s.

On this panel the search cannot beat enumeration. A lift is worth about 50
million priced sets, the whole tree holds 77 million, and the lifts that prune
save about what they cost. It pays only where subtrees are many times a lift's
worth of pricing and the bound prunes them: more markets, or more treated.

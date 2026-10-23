# The cost of a node in SYNDES's two-way search (#844)

#844 asks where a node's time goes in the two-way branch and bound
(`mlsynth/utils/syndes_helpers/branch.py`) and which of four candidates cut it
without changing what the search proves: separating the triangles in
vectorised form, stopping the lift sooner at its prune target, warm-starting a
child's lift from its parent, and certifying in floating point. This study
times the library's search as it stands on `main`, stage by stage, and
measures the candidates beside the steps they would replace.

This is a study, not a benchmark case. Nothing is pinned and the seconds are
machine-dependent: one machine with four cores visible, Python 3.13.16, NumPy
2.5.3, SciPy 1.18.1, BLAS held to one thread by the instruments. The node
counts, the modelled work and the bounds are deterministic; the seconds moved
with load. Two or three of the stage profiles shared the machine; the lean
lift and the L-BFGS-B overhead ran alone.

## The ledger (`stages.py`)

`stages.profile` runs `branch_and_bound` with its stages wrapped. A wrapped
call is charged its own seconds, its time less the wrapped calls it makes, so
every second is charged once; a generator (the enumeration) is charged for
each step, not for its call. A stage that also runs inside another is labelled
by its caller (`lift > trust region`), so the tree's own trust regions stay
apart from the lift's fallback. Beside each stage is the cost model's charge
for what it did, the charge the search spends its work limit against. A test
holds the wrapped search to the library's decisions count for count, and the
modelled charges to the library's reported work.

| panel | units | treated | work limit | seconds | lift | pricing | certificate | trust region | other | lift / model | pricing / model | certificate / model |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLS | 40 | 20 | default | 43.6 | 93.4% | 3.8% | 2.2% | 0.2% | 0.4% | 6.4 | 4.5 | 1.5 |
| noise | 40 | 10 | none | 78.6 | 96.6% | 2.0% | 1.2% | 0.1% | 0.1% | 5.4 | 3.3 | 1.5 |
| GeoLift | 40 | 10 | none | 51.3 | 94.4% | 4.1% | 1.2% | 0.1% | 0.2% | 5.5 | 2.9 | 1.5 |
| GeoLift | 40 | 20 | default | 38.4 | 93.4% | 3.8% | 2.3% | 0.2% | 0.3% | 6.0 | 3.6 | 1.5 |
| noise | 40 | 20 | default | 118.5 | 95.7% | 0.5% | 3.3% | 0.3% | 0.2% | 6.2 | 4.6 | 1.5 |
| BLS | 60 | 30 | default | 231.6 | 96.5% | 0.0% | 3.1% | 0.2% | 0.1% | 4.0 | | 1.7 |
| factor | 40 | 10 | none | 1,997.5 | 1.7% | 95.1% | 0.0% | 0.0% | 3.1% | 5.4 | 2.2 | 1.6 |

`model` is the cost model's charge for the same calls. "Other" is the swap
search that seeds the incumbent, settling designs, and the tree loop.

Wherever the bounds prune, the lift is 93 to 97% of the search. On the
four-factor panel at 40 units and 10 treated (the #803 instance, no bound
separates its sets) pricing is 95%: 847,660,528 sets at 2.24 µs each, of which
the closed form is 1.48 µs, the sign matrices 0.50 and the enumeration 0.26.

### Inside the lift

| panel | units | treated | evaluations | ms per evaluation | eigendecomposition | rest of the evaluation | cut matrix | L-BFGS-B | rest of the lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLS | 40 | 20 | 37,811 | 1.08 | 0.28 (26%) | 0.28 (26%) | 0.13 (12%) | 0.34 (31%) | 0.05 (5%) |
| noise | 40 | 10 | 74,599 | 1.02 | 0.32 (31%) | 0.28 (28%) | 0.11 (11%) | 0.28 (27%) | 0.02 (2%) |
| GeoLift | 40 | 10 | 45,442 | 1.07 | 0.32 (30%) | 0.29 (27%) | 0.12 (12%) | 0.30 (28%) | 0.03 (2%) |
| GeoLift | 40 | 20 | 35,380 | 1.01 | 0.27 (26%) | 0.28 (27%) | 0.11 (11%) | 0.31 (30%) | 0.05 (5%) |
| noise | 40 | 20 | 104,442 | 1.09 | 0.32 (30%) | 0.29 (27%) | 0.11 (10%) | 0.31 (29%) | 0.05 (5%) |
| BLS | 60 | 30 | 142,249 | 1.57 | 0.56 (36%) | 0.35 (23%) | 0.16 (10%) | 0.40 (26%) | 0.10 (6%) |
| factor | 40 | 10 | 33,551 | 1.02 | 0.33 (32%) | 0.28 (28%) | 0.10 (10%) | 0.27 (27%) | 0.04 (4%) |

An evaluation of the lift's dual costs about 1.0 ms at 40 free units, against
the cost model's 0.17 to 0.24 ms. The eigendecomposition, which the model
prices, is a quarter to a third of it. The rest is the evaluation's other
NumPy (assembling the matrix, `X`, the gradients, the cut values), the cut
matrix's six `np.add.at` calls, and SciPy's L-BFGS-B outside the objective.
Separation, the trust region and the round loop together are under 6%.

`parts.lbfgsb_overhead` measures SciPy's L-BFGS-B on an objective that costs
nothing, with the lift's options: 0.09 ms an evaluation at 82 unknowns, 0.23
ms at 380 and 0.35 ms at 700. A lift carries its 82 multipliers and 180 to 560
cuts' weights at 40 units, so that is most of the
L-BFGS-B column, and it is SciPy's Python layer, not ours.

## The candidates (`parts.py`)

At five nodes each (0 to 8 units fixed) of the factor, GeoLift and noise
panels with 40 units and 10 treated:

- The cut matrix by one `bincount` over the upper triangle
  (`parts.cut_matrix`): the library's matrix bit for bit at all 15 nodes and
  over a property test, 1.3 to 2.5 times faster (0.06 to 0.21 ms to 0.03 to
  0.10 ms).
- The dual's evaluation with fewer temporaries (`parts.theta`: the diagonal by
  a strided view, the rows of `X r = 0` as one outer product added to its
  transpose, the cut matrix above, the cut values by one-dimensional takes):
  the library's five outputs bit for bit over a property test, and a lift run
  with it gives the library's bound, multipliers and evaluation count bit for
  bit at all 15 nodes. On a quiet machine, fastest of seven runs, it takes a
  median 5.5% off a lift (-1.8 to 15.8% across the nodes): most of the
  evaluation's own NumPy is work, not overhead.
- A dyadic matrix rounded to doubles by integer division (`parts.to_float`)
  in place of a `Fraction` per entry: the same doubles at all 15 nodes and over
  a property test that reaches overflow and subnormals, 5.5 to 10 times
  faster (2.8 to 6.0 ms to 0.35 to 1.1 ms). The rounding is a fifth to two
  fifths of a certificate.
- Rump's floating-point test (`parts.rump_lower`; Rump, "Mathematically
  rigorous global optimization in floating-point arithmetic", p. 12, Theorem
  4.1) in place of the exact residual: 4.5 to 6.5 times faster than
  `lambda_min_lower` (7.0 to 20.7 ms to 1.4 to 3.6 ms), its bound within
  3.5e-16 of the library's at every node, above it at some and below at
  others. A
  property test checks it in exact arithmetic: the matrix less the bound is
  positive semidefinite by exact elimination.
- Stopping the lift sooner at its prune target: measured by #837
  (`syndes_global_bnb/stopping.py`). A forecast stop held back until `alpha`
  is 1e-4 keeps every prune and saves 3 to 6% of the evaluations.
- Warm-starting a child's lift from its parent's multipliers: not measured
  here. It needs a copy of the lift's round loop that takes a starting point;
  `stopping.lift` (#837) is that copy.

## Findings

1. Where the bounds prune, the search's time is the lift (93 to 97%), and the
   lift's time is not its eigendecompositions: those are 26 to 36% of it. The
   certificate is 1 to 3%, pricing under 5%.
2. On the flat panel the search's time is pricing (95%), and three quarters of
   pricing is the closed form. No change to a node's cost moves that instance.
3. The cost model under-charges the lift 4 to 6 times and pricing 2 to 5 times
   on this machine, so it favours lifting over pricing by roughly 2.5 times
   where a lift competes with a large subtree. That is #845's recalibration.
4. Of a 1.0 ms evaluation, about 0.4 ms is NumPy the library owns, and a
   bit-identical rewrite recovers a median 5.5% of the lift: most of it is the
   matrix products and the gradient, not temporaries. SciPy's L-BFGS-B layer
   is 0.2 to 0.35 ms that only a hand-driven `setulb` (a private interface)
   would remove.
5. The certificate is small wherever measured; the integer-division rounding
   takes 17 to 38% off it without changing a bit, and Rump's test would take
   most of the rest at a bound that differs in the sixteenth digit.

## What to build

- Nothing for the lift's speed alone. The bit-identical pieces take about 5%
  off a lift and 17 to 38% off a certificate, which is about 5% of a search
  where the bounds prune and nothing on the flat panel. They are safe to land
  (the gate pins modelled work, so its counts do not move) and small, and are
  best carried by the next change to `branch.py` and to
  `marex_helpers/branch.py`'s rounding (shared with MAREX), not built for
  their own sake.
- Rump's test is the author's decision (#844 chose exact arithmetic on
  purpose); at 1 to 3% of a search it buys little.
- What moves a search that lifts is fewer evaluations, and the evaluations
  SciPy's driver costs. Warm starts (#844's third candidate) are the
  measurement left, on `stopping.lift` from #837.
- The flat panel's lever is the closed form's cost per set, and the cost
  model's constants (#845), not the node.

## Running it

From the repository root:

```
python -m pytest benchmarks/studies/bnb_node_cost/ -q
python -m benchmarks.studies.bnb_node_cost.stages geolift 40 20 limit
python -m benchmarks.studies.bnb_node_cost.stages factor 40 10
python -m benchmarks.studies.bnb_node_cost.parts candidates factor 40 10
python -m benchmarks.studies.bnb_node_cost.parts lean geolift 40 10
python -m benchmarks.studies.bnb_node_cost.parts overhead 380
```

`results/stages.jsonl` holds one row per search, `results/parts_*.jsonl` the
candidates. The panels are `syndes_global_bnb`'s (`factor`, `noise`,
`geolift`) and `bnb_scale`'s (`bls`).

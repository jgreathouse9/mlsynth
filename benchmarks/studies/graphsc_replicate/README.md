# Graph-SC, demonstrated before it is built

Reference: Yanlin Zhang, Sungyong Chung and Alireza Talebpour (2026),
"Graph-Regularized Synthetic Control for Stable Counterfactual Estimation under
Donor Redundancy", arXiv:2610.24113v1. Equation (2) is the estimator, Algorithm
1 the donor graph, Algorithm 2 the solver, Table 1 the Prop 99 benchmark,
Section 4.1 and Appendix C the simulation.

This is a replication spike. It produces a recommendation, not estimator code.

```
python -m benchmarks.studies.graphsc_replicate.run      # about 8 minutes on 4 cores
```

Results land in `results/graphsc.json`; the tests are
`benchmarks/tests/test_graphsc_replicate.py`.

## The method

Graph-SC chooses donor weights `w` by

    min  1/2 ||x1 - X0 w||^2 + lambda_1 ||w||_1 + lambda_g/2 w'Lw + lambda_sum/2 (1'w - 1)^2

where `L` is the normalized Laplacian of a k-nearest-neighbour graph of the
donors, built from their pre-period outcomes. The graph term pulls the weights
of similar donors together; the last term replaces the simplex's sum-to-one
constraint with a penalty. The paper fixes `lambda_1 = 0.05`, `lambda_g = 10`,
`lambda_sum = 50` and `k = 5` throughout and does not tune them.

The paper is not consistent about the sign of the weights. Definition 3.4
minimises over all of `R^J`; Algorithm 2 starts in `R^J_+` and its proximal step
keeps `w >= 0`, as does the proof of Theorem 3.9. With `w >= 0` the l1 term is
`lambda_1 1'w`, a linear term, and the problem is exactly a non-negative least
squares whose target sum is `1 - lambda_1 / lambda_sum` (0.999 at the paper's
values): the "sparsity" penalty does nothing else. Without the sign constraint
it is a lasso. The oracle solves both readings exactly and certifies each solve
by its KKT residual.

## Path A: Table 1 on Prop 99

Classic SC through `dataprep` reproduces Table 1's SC row in all five columns,
which pins the metric definitions. Three of the four rows the paper reports for
its own method and ablations reproduce to the printed digit; the headline row
does not.

| row | pre-RMSPE | post-RMSPE | mean post gap | active donors | entropy |
| --- | --- | --- | --- | --- | --- |
| SC, paper | 1.6564 | 20.6056 | 19.5136 | 6 | 1.4752 |
| SC, here | 1.6564 | 20.6056 | 19.5136 | 6 | 1.4752 |
| Sparse-only, paper | 0.0058 | 13.0945 | 12.3503 | 20 | 2.5791 |
| Sparse-only, here | 0.0058 | 13.0945 | 12.3503 | 20 | 2.5791 |
| Graph-only, paper | 0.0663 | 18.4235 | 17.3714 | 38 | 3.3940 |
| Graph-only, here | 0.0663 | 18.4235 | 17.3714 | 38 | 3.3940 |
| Graph-SC, paper | 0.0893 | 19.8918 | 18.6245 | 38 | 3.4112 |
| Graph-SC, here | 0.0696 | 18.1940 | 17.1577 | 38 | 3.3692 |

The matches require three departures from the paper's text:

- weights unrestricted in sign (Definition 3.4, not Algorithm 2); 17 of the 38
  Graph-SC weights are negative, so the counterfactual is a signed regression on
  the donors, not the "interpretable mixture of real donors" the paper
  describes;
- the raw panel, although Appendix B.3 says the features were "standardized
  (zero mean, unit variance) across donors"; four readings of that sentence
  match nothing;
- the entropy taken on `|w|` normalized to sum to one, since the printed
  `-sum w_j log(w_j + eps)` is undefined for negative weights.

With `w >= 0` only the SC row matches (Sparse-only then keeps 8 donors, not 20).

The Graph-SC row is not the optimum of eq. (2) at the stated settings. The
optimum keeps 87.9 percent of SC's post-period gap, below the 95 percent the
paper requires, so by the paper's own criteria Graph-SC meets two of three, as
Graph-only does; "the only method satisfying all three" rests on the
unreproduced row. FISTA truncated at checkpoints from 100 to 200,000 iterations
(signed weights, two starting points) and from 1 to 5,000 (`w >= 0`, three
starting points), and a grid over `k`, `lambda_1`, `lambda_g` and `lambda_sum`
around the stated values, come no closer than 11 percent on some column.

### The paper's figure is not its table

`python -m benchmarks.studies.graphsc_replicate.figure` draws
`results/prop99_overlay.png`: each arm's exact solve overlaid on classic SC and
California, and the Graph-SC line of the paper's Figure 2(a), digitized from the
image in the arXiv source (`results/paper_figure2a_digitized.json` records the
method; on the SC line it recovers the known pre-RMSPE as 1.69 against 1.656).

The plotted Graph-SC line has a pre-RMSPE of 1.80 and misses California by 6.3
packs in 1970 alone. Table 1 prints 0.0893 for the same method, which no curve
with that 1970 miss can reach. The plotted line is closest to SC itself (RMS
distance 1.27); every reading solved here, both sign conventions and three
standardizations, sits 1.77 or more from it, about four times the digitization error.
Its mean post-period gap, 17.75, also falls below the 95 percent criterion.

## Path B: the simulation, and the gate

Section 4.1 and Appendix C describe 200 donors in 10 clusters over 70 periods
(50 before treatment), Fourier-plus-trend cluster trajectories, noise with
`sigma = 1`, a treated unit built as a convex combination of one cluster's
donors, and an effect of 2. `simulate.py` fixes the details the paper leaves
open and documents them; the absolute numbers therefore differ from the paper's,
and the comparisons are what carry over.

Beside the paper's arms, two more decide whether the graph matters: Graph-SC on
a uniformly random graph with as many edges as the learned one (Appendix C's
100%-rewired graph), and on no graph at all (`L` replaced by the identity, which
is ridge at the same `lambda_g`; the normalized Laplacian's eigenvalues average
one, so both carry the same amount of shrinkage). 200 replications:

| arm | ATT-MSE, signed weights | ATT-MSE, `w >= 0` |
| --- | --- | --- |
| SC | 0.204 | 0.204 |
| Sparse-only | 0.944 | 0.257 |
| Graph-only | 2.135 | 0.255 |
| Graph-SC | 1.506 | 0.254 |
| Graph-SC, oracle graph (the true clusters) | 1.482 | 0.254 |
| Graph-SC, random graph | 0.696 | 0.261 |
| Ridge, same `lambda_g`, no graph | 0.687 | 0.261 |

Paired MSE ratios with 95 percent bootstrap intervals:

| ratio | signed weights | `w >= 0` |
| --- | --- | --- |
| learned graph / random graph | 2.17 [1.79, 2.62] | 0.973 [0.965, 0.981] |
| learned graph / ridge | 2.19 [1.82, 2.66] | 0.974 [0.966, 0.980] |
| learned graph / SC | 7.37 [5.36, 10.37] | 1.24 [1.12, 1.40] |
| oracle graph / learned graph | 0.98 [0.95, 1.02] | 1.000 [0.996, 1.004] |

In the reading the paper's numbers follow, the learned graph is more than twice
as bad as a random graph or as ridge, and seven times as bad as plain SC; across
`lambda_g` from 0.1 to 1000 it never beats either (`results/graphsc.json`,
`gate_sweep_signed`). Even the true clusters do no better. This reproduces the
paper's own Appendix C, whose Table 3 has SC (0.1438) and sparse-only (0.1074)
ahead of every Graph-SC variant (0.52 to 0.59) and a fully random graph (0.2561)
ahead of the learned one (0.5337), and contradicts its Section 4.1, which reports
the opposite order for the same design. With `w >= 0` the graph's content buys
2.6 to 2.7 percent over a random graph or ridge, and the estimator is still 24
percent worse than SC.

## The LaTeX source

The arXiv source adds no code and no simulation detail, and:

- prints Graph-SC's post-RMSPE as 19.8918 (a first reading of the PDF had
  19.8198);
- confirms the sign inconsistency (Definition 3.4 over `R^J`, Algorithm 2 and
  the proof of Theorem 3.9 over `R^J_+`), and its commented-out drafts disagree
  in the same way (a removed lemma claims strong convexity "on `R^J`"; a
  removed proof adds "the indicator of `w >= 0`");
- contains a removed version of the Waymo table in which SC and Graph-SC have
  the same pre-period fit (0.034 and 0.034 for headway; 0.080 and 0.082 for
  time-to-collision) and similar post-period RMSPE; the published Table 2
  reports SC at 1.1738 against Graph-SC's 0.0236. The source does not say what
  changed between the versions.

## Recommendation

Pass.

- The headline Prop 99 result does not reproduce, and the rows that do require
  signed weights, which removes the interpretability the method is sold on.
- The gate fails: in the paper's own reading the learned graph is worse than a
  random graph and than plain ridge, and far worse than SC; the paper's own
  appendix says the same.
- The l1 term is inert on non-negative weights, so the method's sparsity claim
  cannot hold in the reading that keeps the weights interpretable.
- mlsynth already has the penalized-SC family this belongs to (RESCM's
  SC / LASSO / RIDGE / ENET corner cases); a graph penalty would be one more
  corner case, and the evidence does not support adding it.

One finding outlives the verdict: RESCM's `LASSO` returns exactly the `SC`
weights on Prop 99 (largest difference 0.0), because its penalty
`lam * alpha * ||w||_1` is constant on the simplex. That is a defect in shipped
code and belongs on its own branch.

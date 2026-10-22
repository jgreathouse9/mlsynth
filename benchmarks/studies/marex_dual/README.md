# MAREX's node bound by a regularised dual (#833)

MAREX's branch and bound (`mlsynth/utils/marex_helpers/branch.py`) bounds a
node with the persp+N relaxation, solved by Clarabel, and certifies the
solver's multipliers in exact arithmetic. #833 asks whether BiqCrunch's
procedure (Krislock, Malick and Roupin 2014: L-BFGS-B on a regularised dual)
certifies the same bound faster. Four first-order methods were already ruled
out (`benchmarks/studies/marex_bnb/results/node_solvers.log`). The issue's bar:
at least 0.99 of Clarabel's certified bound in a fraction of its time, or the
study records why not.

This is a study, not a benchmark case. Nothing is pinned. The seconds are one
machine's, on one thread, with other jobs sometimes sharing it; the comparisons
below are between runs a factor of twenty or more apart.

## The method (`dual.py`)

The library's certificate is a function of the multipliers alone. Per side:

- `y0` on `M_00 = 1`, `alpha` on `1'x = 1`, `beta` on `X 1 = x`;
- `H >= 0` on `X >= 0`, and `mu >= 0` on `x_j <= z_j`;
- a semidefinite 2x2 `Lam_j` on each perspective block.

For any such values the Lagrangian is `const + <S, M>` per side plus `c'z`. A
design has `tr M <= 2` per side and `z` in the capped simplex, so

    bound = const + sum over sides of 2 min(0, lambda_min(S)) + (the r smallest c_j).

`dual.py` maximises that bound with L-BFGS-B through a smooth lower bound of it,
valid for every smoothing `a > 0`:

- per side, `<S, M> >= -2 |S_-|_F >= -|S_-|_F^2 / (2a) - 2a`;
- for the indicators, `min c'z >= min (c'z + a |z|^2 / 2) - a r / 2`, a capped
  simplex projection.

Every iterate is a valid bound. `a` starts at 0.1 and is halved round by round,
BiqCrunch's schedule.

Parametrisation:
- `H` by its upper triangle, entries kept nonnegative by bounds;
- `mu` kept nonnegative by bounds;
- each `Lam_j` by a Cholesky factor, so it is semidefinite for every value.

The multipliers with the best exact bound go to the library's `_certificate`.
The bound reported is therefore certified in exact arithmetic exactly as
Clarabel's is.

The tests hold:
- the float bound to the library's exact certificate of the same multipliers;
- every value under every design of the node, as a property over multipliers
  and smoothing;
- the gradient to the smooth bound's, by finite differences;
- the bound to the interior-point bound on eight-market nodes.

## Against Clarabel

`compare` runs both on the issue's 12-market factor root and on the twelve
nodes of GeoLift's tree with four treated (`persp_n_geolift.nodes`), at 40
markets. The issue's three GeoLift cases (K = 56, 76, 116) are block sizes of
the earlier indicator relaxation; the library's persp+N has blocks of at most
41 rows, and these twelve nodes are where its search spends them.

Bounds are certified bounds over the subtree's minimum. `need` is the ratio a
bound must reach to prune the node, the optimum over the subtree minimum. The
dual ran to 40,000 evaluations, and in a second run stopped once its exact bound
reached the prune target.

| node | sets | need | Clarabel | seconds | dual | of Clarabel's | seconds | evaluations | to the target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| factor 12, root | 220 | 1.0000 | 0.9519 | 0.23 | 0.9517 | 0.9998 | 88.3 | 30,894 | not reached |
| GeoLift root | 91,390 | 1.0000 | 0.9849 | 7.48 | 0.9663 | 0.981 | 191.5 | 40,000 | not reached |
| first unit in, 0 | 9,139 | 0.7432 | 0.9687 | 6.05 | 0.9333 | 0.964 | 192.7 | 40,000 | 97.5 s |
| first unit in, 5 | 5,984 | 0.6442 | 0.9785 | 5.40 | 0.9680 | 0.989 | 176.1 | 38,269 | 94.7 s |
| first unit in, 10 | 3,654 | 1.0000 | 0.9999 | 4.16 | 0.9870 | 0.987 | 164.9 | 39,904 | not reached |
| first unit in, 15 | 2,024 | 0.5258 | 0.9808 | 3.01 | 0.9743 | 0.993 | 160.5 | 40,000 | 71.5 s |
| first unit in, 20 | 969 | 0.4649 | 0.9806 | 2.06 | 0.9752 | 0.995 | 147.0 | 40,000 | 67.7 s |
| first unit in, 25 | 364 | 0.4140 | 0.9560 | 1.76 | 0.9510 | 0.995 | 122.4 | 34,811 | 61.1 s |
| first unit in, 30 | 84 | 0.3825 | 0.9992 | 1.77 | 0.9937 | 0.995 | 107.6 | 31,603 | 48.7 s |
| first unit in, 35 | 4 | 0.0009 | 1.0000 | 1.92 | 1.0000 | 1.000 | 71.1 | 22,859 | 2.1 s |
| six non-optimal out | 46,376 | 1.0000 | 0.9894 | 4.49 | 0.9719 | 0.982 | 174.1 | 40,000 | not reached |
| one optimal out | 82,251 | 0.8740 | 0.9007 | 7.04 | 0.8676 | 0.963 | 182.3 | 40,000 | not reached |
| one optimal in, one out | 8,436 | 0.8161 | 0.9954 | 6.35 | 0.9801 | 0.985 | 185.3 | 39,418 | 98.4 s |

"First unit in, q" treats the market with the q-th best single fit and leaves
out the q markets that fit better (`persp_n_geolift.nodes`).

- The dual certifies 0.963 to 1.000 of Clarabel's bound, median 0.987. It
  reaches 0.99 on six of the thirteen nodes.
- It takes 71 to 193 seconds where Clarabel takes 1.8 to 7.5: 26 to 71 times
  longer at 40 markets, and 380 times at 12.
- Stopped at the prune target, it prunes 8 of the 9 nodes Clarabel prunes. It
  misses the node with one optimal market out (0.868 against a need of
  0.874). It takes 2.1 to 98 seconds against Clarabel's 1.8 to 7.0, so it
  is not faster even where a node prunes early.

## Why

- The bound is a small difference of large numbers. At GeoLift's root the
  subtree minimum is 6.45e-5 of the Gram matrix's scale: the best combinations
  of markets nearly cancel. The smoothing costs `4a + a r / 2` in the same
  units, so a bound within 1% needs `a` near 1e-7. At that smoothing the smooth
  bound's curvature is of order `1 / a`, and L-BFGS-B on 2,044 multipliers
  moves slowly. This is the cause `node_solvers.log` records for the other
  first-order methods, which needed relative accuracy near 1e-6.
- The cost is in evaluations, not in any single step. An evaluation at 40
  markets takes about 5 ms: two eigendecompositions of 41-row matrices, the
  assembly of `S` from 820 entrywise multipliers per side, and the 2x2
  blocks. The dual needs 20,000 to 40,000 evaluations, where Clarabel takes a
  few dozen Newton steps. On the 12-market factor root, where the minimum is
  0.053 of the scale and the smoothing is no obstacle, the dual still took
  30,894 evaluations (88 s) against Clarabel's 0.23 s.
- SYNDES's lift, where the same procedure won (#803), is a ±1 problem with a
  unit diagonal and one block. MAREX's is a simplex lift on a face, with the
  entrywise nonnegativity carrying most of the bound (`persp_n.md`). Its
  multipliers are where the bound lives, and they converge slowly.

## What to build

Nothing. The dual misses the bar on both counts: under 0.99 of Clarabel's bound
on seven of thirteen nodes, and 26 to 71 times its time at 40 markets. Clarabel stays MAREX's
node solver. This joins FaceADMM, the spectral bundle method, LoRADS and SCS in
`node_solvers.log` as a first-order method that does not certify MAREX's bound
at the needed accuracy in useful time.

What would change the picture is a smaller node problem, not a different solver
for this one. A bound whose dual has fewer multipliers, or that is solved at
fewer nodes, is the lever, and #832 found MAREX's searches already bound few
nodes (2 to 22).

## Running it

From the repository root:

```
python -m pytest benchmarks/studies/marex_dual/ -q
python -m benchmarks.studies.marex_dual.dual factor12
python -m benchmarks.studies.marex_dual.dual geolift
```

`results/compare.jsonl` holds one row per node. The GeoLift run reads the leaf
cache of `benchmarks/studies/marex_bnb` (`results/leaves_geolift_m4.npz`, built
on first use).

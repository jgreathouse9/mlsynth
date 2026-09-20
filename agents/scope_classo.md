# Scope: C-Lasso donor selection for synthetic control

> **Status: scoped, no library code.** This branch is a replication spike. It
> produces a recommendation and the measurements behind it. The estimator, if
> built, lands test-first on its own branch per `CLAUDE.md`.

## Source

> Su, L., Shi, Z. & Phillips, P. C. B. (2016). "Identifying Latent Structures in
> Panel Data." *Econometrica* 84(6), 2215-2264.

Reference implementations: `zhentaoshi/C-Lasso` (the authors' MATLAB, CVX +
MOSEK) and `zhan-gao/classo` (R, CVXR + ECOS / Rmosek). The numerical side is
documented in

> Gao, Z. & Shi, Z. (2021). "Implementing Convex Optimization in R: Two
> Econometric Examples." *Computational Economics* 58, 1127-1135.

## The method

For a panel with unit-specific slopes, C-Lasso estimates

```
Q(b, a) = (1/NT) sum_i sum_t psi(w_it; b_i, mu_i(b_i))
          + (lambda/N) sum_i prod_{k=1..K} || b_i - a_k ||
```

The penalty is additive over units and multiplicative over groups. Standard
Lasso shrinks a coefficient to zero; C-Lasso shrinks each unit's coefficient to
one of `K` unknown centres, which are themselves optimised. Classification and
estimation happen in one step, the classification is uniformly consistent, and
for penalized least squares the estimator has the oracle property: post-Lasso
group estimates are asymptotically equivalent to knowing the groups.

The objective is not convex in `b`, so the authors' Supplement S3.1 gives an
iterative scheme in which each sub-step is convex: freeze the other centres,
form `gamma_i = prod_{j != k} || b_i^(j) - a_j ||`, and solve a second-order
cone program in `(b, a_k)`. Cycle `k` to numerical convergence, assign each unit
to its nearest centre, refit within groups. `K` and `lambda` are chosen jointly
by their information criterion (2.9),

```
IC(K, lambda) = post-Lasso MSE + rho * p * K,     rho = (2/3) (NT)^(-1/2)
```

## Why it belongs in mlsynth

mlsynth has donor-pool selection by clustering (`CLUSTERSC`, via FPCA k-means or
fGRC) and by low-rank denoising. All of these cluster on the outcome matrix
itself, in some feature space. None classifies on estimated *dynamics*.

Pearson correlation, the metric underneath most donor-similarity machinery, is
invariant to the time ordering: permute the periods identically across units,
or run the panel backwards, and the ranking it produces is unchanged. It has no
temporal content by construction. C-Lasso does: put lagged outcomes on the
right-hand side and the classification is on persistence and response speed,
with shrinkage toward group centres supplying the regularization that an
unpenalized clustering of per-unit estimates lacks.

## The port validates

`agents/prototypes/classo_pls_spike.py` transcribes `PLS.cvxr` and reproduces
the reference package's published output on its own `sample_data.rda` (SSP's
DGP 1, N=200, T=25, K=3, `lambda = 0.5 var(y) / T^(1/3)`):

| | group coefficients | misclassified |
|---|---|---|
| this port | `[0.4017 1.6014] [1.0388 0.9987] [1.6197 0.3614]` | 5 / 200 |
| `zhan-gao/classo` README | `[0.4017 1.6014] [1.0388 0.9987] [1.6197 0.3614]` | 5 / 200 |

Four decimals, and the same five units. Per-unit standardization follows their
`master.m` exactly (demean, divide by the population standard deviation, `y` and
each column of `X` separately).

## The Basque demonstration

Abadie & Gardeazabal (2003), 17 Spanish regions, 1955-1997, `basedata/basque_data.csv`.
C-Lasso is fit on the pre-treatment window only, so the grouping carries no
post-treatment information. Five specifications were tried, each over the
authors' geometric `lambda` grid (10 points, 0.2 to 2.0 times the reference
scaling).

### Finding 1: the IC cannot select K at this sample size

Every specification returns `K = 1`, at every `lambda`. The reason is arithmetic,
not empirical. With `N = 17` and `T = 14`, `rho = (2/3)(238)^(-1/2) = 0.0432` per
parameter per group:

| spec | p | MSE(K=1) | rho·p | can the IC pick K>1? |
|---|---|---|---|---|
| lag gdpcap | 1 | 0.0148 | 0.0432 | no: penalty exceeds the whole MSE |
| lag + invest | 2 | 0.0147 | 0.0864 | no: penalty exceeds the whole MSE |
| lag + schooling | 3 | 0.0142 | 0.1296 | no: penalty exceeds the whole MSE |
| invest + schooling | 3 | 0.3425 | 0.1252 | needs a 37% MSE drop; got 2.85% |
| growth | 2 | 0.8889 | 0.0897 | needs a 10% MSE drop; got 3.30% |

For the three dynamic specifications the per-group penalty is three to nine
times the entire post-Lasso MSE, so `K = 1` is a foregone conclusion. SSP's IC is
calibrated for their `N = 56, T = 15` savings panel and their `N = 100-200`
simulations. Synthetic control panels are smaller, and the IC does not transfer.

`K` therefore has to be an explicit user choice in any mlsynth implementation,
with a guard that reports when the IC is structurally unable to choose.

### Finding 2: at a forced K, restriction helps, and for the expected reason

Convex SCM (`VanillaSC`, outcome-only, simplex weights) on the full pool against
C-Lasso pools from the `lag + invest` specification:

| donor pool | ATT | pre-RMSE | 1997 gap | weights |
|---|---|---|---|---|
| all 16 | −0.8946 | 0.0756 | −1.0124 | Madrid 0.483, Baleares 0.311, Rioja 0.206 |
| C-Lasso K=2 (5) | −0.5847 | 0.0888 | −0.8219 | Cataluna 0.831, Madrid 0.169 |
| C-Lasso K=3 (8) | −0.5441 | 0.1791 | −0.8748 | Cataluna 1.000 |
| C-Lasso K=4 (4) | −0.5441 | 0.1791 | −0.8748 | Cataluna 1.000 |

Pre-treatment fit necessarily worsens: the simplex over a subset of donors is
contained in the simplex over all of them, so a restricted pool can only fit the
pre-period weakly worse. Pre-RMSE is the wrong criterion for judging donor
restriction, and a benchmark that pins it would encode the wrong objective.

What restriction is supposed to buy is less interpolation bias, which shows up
out of sample. Scoring by placebo-in-time -- fit up to a pre-treatment date,
predict the remaining untreated years, where the true effect is zero:

| donor pool | 1962 | 1963 | 1964 | 1965 | 1966 | 1967 | mean | beats full pool |
|---|---|---|---|---|---|---|---|---|
| all 16 | 0.3845 | 0.3367 | 0.2792 | 0.2525 | 0.1799 | 0.0542 | 0.2479 | — |
| C-Lasso K=2 (5) | 0.3845 | 0.3716 | 0.2788 | 0.2525 | 0.2123 | 0.1455 | 0.2742 | 1/6 |
| C-Lasso K=3 (8) | 0.1758 | 0.1865 | 0.1958 | 0.1987 | 0.1865 | 0.1766 | 0.1866 | 4/6 |
| C-Lasso K=4 (4) | 0.1758 | 0.1865 | 0.1958 | 0.1987 | 0.1865 | 0.1766 | 0.1866 | 4/6 |
| C-Lasso K=5 (3) | 1.3172 | 1.3231 | 1.3265 | 1.3210 | 1.3026 | 1.2913 | 1.3136 | 0/6 |

The `K = 3` pool is 25% better on average, and the gain is in stability: its
held-out error sits in 0.176-0.199 across every placebo date while the full
pool's swings over 0.054-0.385. The full pool wins at 1966 and 1967 only, where
the held-out window is three years or fewer.

The economics are legible. The unrestricted fit leans on Baleares, an island
tourism economy, which fits the Basque pre-period well and extrapolates badly.
C-Lasso's pool drops it and the weight moves to Cataluna, the other industrial
region. `K = 5` drops Cataluna too and the result collapses, which shows how much
of this rests on one donor.

### What this does not establish

One treated unit, one dataset, six placebo dates, one specification, and `K` set
by hand. The Basque study is a demonstration that the mechanism works as the
theory says it should, not evidence that it generalises. `N = 17` is far below
anything SSP's asymptotics address.

## Proposed architecture

A new estimator, not a `CLUSTERSC` cluster method. `CLUSTERSC`'s existing
`cluster_method` values take the outcome matrix and nothing else; C-Lasso needs a
regression specification (which regressors, how many lags), a penalty, and a
group count. That is a different input contract, and the group-level slope
estimates are an output in their own right. Invariants 4 and 6 point to a
separate package.

```
mlsynth/estimators/classosc.py                  # thin; .fit()
mlsynth/utils/classosc_helpers/
    config.py                                   # CLASSOSCConfig
    setup.py                                    # dataprep wrapper; builds y, X
    pls.py                                      # the SOCP iteration
    selection.py                                # IC over (K, lambda) + the guard
    pipeline.py                                 # classify -> pool -> SCM -> results
    structures.py                               # CLASSOSCResults
    plotter.py                                  # group trajectories + the SCM fit
```

Config fields the demonstration forces:

- `n_groups: int | None` -- required in practice. `None` attempts the IC and
  raises `MlsynthConfigError` when `rho * p >= MSE(K=1)`, naming the arithmetic.
- `spec: Literal["lag", "lag+covariates", "covariates"]` and `n_lags: int` --
  what goes on the right-hand side.
- `lambda_grid` -- default the authors' geometric grid.
- `backend` -- the convex SCM to run on the selected pool, defaulting to
  `VanillaSC` outcome-only.
- `min_pool_size: int` -- guard against the `K = 5` collapse; raise when the
  treated unit's group is a singleton.

The classification is fit on pre-treatment periods only. This is an invariant,
not an option.

A second, smaller artifact may be the more useful one: expose the classifier
alone as `mlsynth/utils/classo.py`, so any estimator can ask "what latent groups
does this panel have" without buying a whole estimator. `CLUSTERSC` could then
gain `cluster_method="classo"` for the outcome-lag specification as a follow-up.

## Replication contract

Path A, plus a cross-validation, both already met by the spike:

- Cross-validation against `zhan-gao/classo` on `sample_data.rda` -- the
  four-decimal agreement above. This is the primary target; it pins the
  algorithm, not an application.
- Path A on SSP's own savings application (`app_saving_PLS`, N=56, T=15, their
  Table 6) once the estimator exists.

Benchmark cases are a separate workstream and a separate branch. Candidates:
`benchmarks/cases/classo_dgp1.py` (the cross-validation) and
`benchmarks/cases/classo_saving.py` (Path A). The Basque demonstration is not a
benchmark -- its numbers are a measurement of this design choice, not a
replication of a published result -- and pinning its pre-RMSE would encode the
wrong objective, per Finding 2.

## Test plan

Test-first, per `CLAUDE.md`. Beyond smoke, unit, edge and failure levels:

- The SOCP sub-step recovers the known solution on a two-group synthetic panel.
- Per-unit standardization matches `master.m` (population standard deviation,
  `y` and each `X` column separately).
- The IC guard fires: construct a panel where `rho * p >= MSE(K=1)` and assert
  `MlsynthConfigError` with the arithmetic in the message.
- Classification uses no post-treatment period: assert that perturbing
  post-treatment outcomes leaves the labels unchanged.
- A singleton treated group raises instead of returning an empty donor pool.
- Convergence failure is reported on the result, never swallowed (invariant 7).
- Label permutation invariance: relabelling groups does not change the pool.

## Risks

1. **Sample size.** SSP's theory wants `N` and `T` large. SCM panels are short
   and narrow. The estimator will often be asked for something its asymptotics
   do not cover, so the diagnostics have to say so on the result object.
2. **`K` is a researcher degree of freedom.** With the IC unusable at these
   sizes, `K` becomes a tuning knob that moves the ATT -- from −0.89 to −0.54 on
   Basque. Any docs page has to lead with this, and the result should carry the
   placebo-in-time held-out error so the choice is defensible.
3. **Solver dependency.** The sub-step is an SOCP. `cvxpy` is the natural route;
   check what mlsynth already depends on before adding one.
4. **Dynamic-panel bias.** The spike omits the split-panel jackknife the
   references ship. It is common across units and does not change a grouping,
   but the library version should port `SPJ_PLS` for the reported coefficients.

## Recommendation

Build it, in two steps. First the classifier as a utility with the DGP-1
cross-validation as its benchmark -- that is a self-contained, fully validated
unit, and it is the piece other estimators can reuse. Then the `CLASSOSC`
estimator on top, if the utility earns its keep.

Hold the estimator until there is a second dataset. One treated unit and six
placebo dates is enough to say the mechanism behaves as the theory predicts;
it is not enough to default anyone's donor pool to it.

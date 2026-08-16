# CSC (Correlated Synthetic Controls) — demonstrate-first spike

Replication spike for Tzvetan Moev, "Correlated Synthetic Controls"
(arXiv:2507.08918), reference code
[`tzvetanmoev/Correlated-Synthetic-Controls`](https://github.com/tzvetanmoev/Correlated-Synthetic-Controls).
Nothing here is wired into `mlsynth`; this is the evidence behind the
build/park decision, kept so it does not have to be re-derived.

CSC targets panels with many treated units, a short pre-period and one common
adoption date. It writes each treated unit's untreated outcome as
`y_it(0) = eta_i + sum_j w_ij y_jt(0) + e_it` and makes the donor weights a
correlated-random-coefficients function of the treated unit's time-invariant
covariates, `w_ij = omega_j + x_i alpha_j`, still on the simplex. Treated
units with similar covariates get similar (correlated) synthetic controls.

## What is here

| file | what it does |
| --- | --- |
| `core.py` | the estimator: the paper's eq. (24) matrix program in cvxpy, plus an optional ridge tie-break |
| `dgp.py` | port of the paper's Monte Carlo DGP (`Simulation/1_Simulate_Data_Int_Fixed_Effects_Model.R`) |
| `baselines.py` | feasible DiD, infeasible DiD, and Abadie-L'Hour PSC via mlsynth's own `penalized_weights` |
| `ingest.py` | whether `dataprep` supplies CSC's inputs on a many-treated panel (it does) |
| `mc.py` | the Path-B grid, under both factor-draw protocols |
| `fixed_factors.py` | the protocol experiment: factors redrawn per replication vs held at the paper's printed draw |
| `identified_set.py` | the width of the set of ATTs consistent with an optimal CSC fit |
| `reference_form.py` | the reference's stacked construction, transliterated, against the compact eq. (24) |
| `solver_check.py` | cross-solver agreement on the same program |
| `tiebreak.py` | whether a ridge tie-break restores a single answer, and what it costs |
| `feasibility.py` | whether continuous covariates really make the program infeasible |

Run any of them directly (`python3 mc.py --fixed-factors`); they only need
`mlsynth` and its own dependencies.

## Findings

1. The port solves and behaves. Weights are non-negative and sum to one per
   treated unit; the program is a convex QP that solves in ~0.1 s at the
   paper's dimensions (n0 = 85, n1 = 15, T0 = 4, K = 5). The compact eq. (24)
   form and a transliteration of the reference's own stacked construction --
   long residual vector, replicated donor block,
   `vec(t(X_pre) %*% alphas %*% H)`, `O1_n1Tn1` intercept design -- reach the
   same optimum to `1.4e-9` relative over ten panels (`reference_form.py`).

2. CSC needs no ingestion module. `dataprep`'s cohort branch already returns
   the treated block, the donor block and `pre_periods`, and `covariates=[...]`
   returns a per-unit covariate matrix; `ingest.py` round-trips all four
   against the simulator.

3. The estimator is weakly identified in the regime it advertises. The fit
   has `n1 * T0` residuals against `n0 * (1 + K) + n1` parameters — 60-ish
   equations for ~500 parameters at the paper's baseline — so the minimiser is
   a face, not a point. `identified_set.py` maximises and minimises the ATT
   over the near-optimal set: at the paper's baseline the set is about
   `[-0.79, +1.78]` wide against a true effect of 1.0. Four solvers handed the
   identical program return ATTs differing by 0.12 in median and 0.70 at worst
   (`solver_check.py`). The set narrows as equations per parameter rise and
   closes entirely by ~0.76.

   This is the multiplicity of solutions the paper says CSC removes ("by
   construction, there does not exist a SC which exactly matches both of their
   time series at the same time", Section 3.e). Pooling the weights reduces
   the parameter count relative to a separate SC per treated unit; it does not
   make the program determined.

4. A ridge tie-break fixes it for free. `ridge * (||omega||^2 + ||alpha||_F^2)`
   at `1e-2` collapses cross-solver disagreement from `1.3e-1` to `2.9e-3` in
   the weights and the ATT spread to 0.002, with no measurable loss of
   pre-treatment fit — the face is flat, so selecting its minimum-norm point
   costs nothing (`tiebreak.py`).

5. Continuous covariates do not make the program infeasible. Section 3.f says
   they do, and the reference code thresholds continuous predictors into
   dummies to avoid it. Adding up requires `X1 S = (1 - W0) 1` where `S` is
   the column sums of `alpha`; `S = 0` with `sum_j omega_j = 1` satisfies it
   for any `X1`, and gives the pooled synthetic control. `feasibility.py`
   solves a continuous covariate, dummies plus a continuous covariate, and two
   non-exhaustive binaries; all four solve, and the continuous designs come
   back with `sum_j alpha_jk = 0` to machine precision, exactly as the algebra
   predicts. What a continuous covariate costs is the level freedom, which
   halves the weight heterogeneity here — not feasibility.

6. The paper's headline reproduces, but only under the paper's stated protocol
   (`fixed_factors.py`, 150 panels, true tau = 1.0):

   | protocol | CSC | fDiD | PSC | iDiD |
   | --- | --- | --- | --- | --- |
   | factors redrawn each replication (what the driver does) | +0.02 / 0.78 | −0.01 / 1.77 | −0.43 / 0.85 | +0.01 / 0.22 |
   | factors fixed at eq. (26)-(27) (what Appendix A.f says) | +0.01 / 0.38 | −1.54 / 1.54 | −0.57 / 0.67 | −0.00 / 0.22 |

   (signed mean error / mean absolute error.)

   With the factors held fixed, DiD carries the large negative bias the paper
   reports (−1.54 here, −0.94 in Table 1) and CSC is nearly unbiased (+0.01 vs
   the paper's −0.11) — the ordering and rough magnitudes reproduce. With the
   factors redrawn, DiD's bias changes sign across replications and its signed
   mean goes to −0.01, so on the paper's own metric DiD looks unbiased. Its
   mean absolute error, 1.77, is the worst of the four in both protocols, and
   CSC's advantage over DiD survives in that metric under either protocol.
   Footnote 27 of the paper says the absolute version was not run for lack of
   time; it is the metric that carries the claim.

   The grid under the fixed-factor protocol, 250 replications per row
   (`mc.py --fixed-factors`, signed mean error / mean absolute error,
   true tau = 1.0):

   | row | CSC | CSC + ridge | fDiD | fDiD (reference) | PSC | iDiD |
   | --- | --- | --- | --- | --- | --- | --- |
   | baseline | −0.04 / 0.39 | −0.04 / 0.35 | −1.54 / 1.54 | −1.35 / 1.35 | −0.61 / 0.69 | −0.00 / 0.24 |
   | N = 200 | −0.10 / 0.35 | −0.10 / 0.26 | −1.59 / 1.59 | −1.38 / 1.38 | −0.45 / 0.51 | −0.03 / 0.15 |
   | T = 4 | +1.01 / 1.09 | +0.98 / 1.03 | +4.04 / 4.04 | +2.02 / 2.02 | −0.34 / 0.59 | −0.02 / 0.24 |
   | T = 7 | −0.81 / 0.93 | −0.81 / 0.92 | −0.73 / 0.77 | −0.82 / 0.82 | −0.77 / 0.89 | −0.03 / 0.23 |
   | F = 4 | −0.23 / 0.42 | −0.26 / 0.43 | −1.01 / 1.02 | −1.01 / 1.01 | −0.39 / 0.58 | +0.04 / 0.26 |
   | E\[pi] = 0.40 | −0.05 / 0.38 | −0.04 / 0.34 | −1.47 / 1.48 | −1.30 / 1.30 | −0.58 / 0.68 | −0.01 / 0.23 |
   | assignment at random | −0.01 / 0.31 | −0.02 / 0.32 | −0.03 / 0.43 | −0.34 / 0.42 | −0.02 / 0.38 | +0.01 / 0.26 |
   | selection x 0.50 | −0.05 / 0.32 | −0.06 / 0.31 | −1.18 / 1.18 | −1.11 / 1.11 | −0.30 / 0.44 | −0.01 / 0.23 |
   | selection x 0.00 | −0.02 / 0.26 | −0.05 / 0.26 | +0.00 / 0.36 | −0.39 / 0.42 | −0.01 / 0.30 | −0.00 / 0.21 |

   What reproduces:

   * The mechanism. Dialling the dependence of assignment on the factor
     loadings from full to half to none moves fDiD from −1.54 to −1.18 to
     +0.00 while CSC sits between −0.02 and −0.06 throughout. This is
     Proposition 2 and the paper's central claim, and it comes through
     cleanly.
   * The ordering. CSC beats fDiD on mean absolute error in eight of nine
     rows and PSC in seven of nine, with the paper's own ranking
     (CSC better than PSC better than fDiD, all worse than the infeasible
     oracle) holding at the baseline: 0.39 / 0.69 / 1.54 / 0.24.
   * The short-panel collapse. The paper's `T = 4` row has every feasible
     estimator failing; here CSC returns +1.01 on a true effect of 1.0 and
     fDiD +4.04.

   What does not:

   * `T = 7`. The paper has CSC improving and beating fDiD as the pre-period
     lengthens (0.09 against 0.50 at `T = 8`); here CSC is the one row where
     fDiD wins (0.93 against 0.77). Under the fixed-factor protocol every
     cell is conditional on one realised `lambda`, so a single row flipping
     is what that protocol produces — which is itself the argument against
     reading the cells as properties of the DGP.
   * PSC's magnitude. The paper's `T = 4` row puts PSC worst at 4.71; here it
     is the best feasible estimator in that row at 0.59. mlsynth's
     `penalized_weights` is used, not the paper's PSC harness, and that
     harness stacks the treatment indicator itself among the matching
     predictors (`run_pscm` slices `covariates_continous[, 1:2]`, whose
     second column is `treatment_indicator`).

   The ridge tie-break costs nothing in accuracy anywhere in the grid, and
   helps at `N = 200` (0.26 against 0.35).

7. The reference's DiD marks one donor as treated. `run_did` builds its
   treatment column as
   `D[seq(from = length(Y_long) - TT * num_treated, to = length(Y_long), by = TT)] <- 1`,
   which starts at index `T * n0` — the last donor's post-treatment cell — so
   `n1 + 1` cells are treated and one belongs to a control unit. The cost is
   cell-dependent and runs both ways: at the redrawn baseline it turns +0.03
   into −0.32; at the fixed-factor `T = 4` row it halves DiD's bias from
   +4.04 to +2.02; under random assignment it manufactures −0.34 where the
   corrected version reports −0.03. Either way the comparator CSC is measured
   against is wrong by a fifth to a whole effect size.
   `baselines.twfe_did(..., reference_offbyone=True)` reproduces it.

## What could not be done here

The authors' CVXR implementation could not be executed: CRAN is unreachable
from this environment (the proxy answers 403 to `cloud.r-project.org`), so
`CVXR` cannot be installed even though R itself can. The usual cell-for-cell
cross-validation against `run_rwscm` is therefore missing. The substitutes are
`reference_form.py` (the reference's stacking, rebuilt in cvxpy, agreeing with
eq. (24) to `1.4e-9`), the four-solver agreement check, and the reference's
intercept convention (`intercept="pin_first"`). None of them exercises the
authors' solver. Anyone with CRAN access should run `run_rwscm` against
`core.csc_weights` on identical inputs before a build is committed to.

Path A (the Mariel Boatlift / PSID application) was not attempted: it needs
PSID registration and a five-script `psidR` extraction, PSID redistribution is
restricted, and the headline is a hold-out RMSE table on 42 treated workers.

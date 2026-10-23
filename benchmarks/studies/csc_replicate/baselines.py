"""The three comparators the CSC paper's Table 1 puts CSC against.

* feasible DiD (fDiD)   -- two-way fixed effects on the observed panel;
* infeasible DiD (iDiD) -- the same after subtracting the demeaned
  interactive fixed effects, which no analyst can do; the paper's oracle
  (Definition 1, Section 5.b);
* penalized SC (PSC)    -- Abadie & L'Hour (2021), the paper's closest
  sibling. Taken from mlsynth's own implementation, which is already
  cross-validated against the authors' ``wsoll1``
  (``docs/replications/pensynth.rst``), instead of re-porting the paper's
  R harness.

``run_did`` in the reference (``Simulation/2_Code_CSC_IDiD_FDiD.R``) builds
its treatment column as

    D[seq(from = length(Y_long) - TT * num_treated, to = length(Y_long), by = TT)] <- 1

which starts one unit too early: index ``T * n0`` is the *last donor's*
post-treatment cell, so the reference marks ``n1 + 1`` cells as treated and
one of them belongs to a control unit. ``twfe_did(..., reference_offbyone
=True)`` reproduces that; the default does not.
"""

from __future__ import annotations

import numpy as np

from mlsynth.utils.bilevel.penalized import penalized_weights


def twfe_did(
    Y: np.ndarray,
    treated_mask: np.ndarray,
    *,
    reference_offbyone: bool = False,
) -> tuple[float, np.ndarray]:
    """Two-way fixed effects DiD with treatment in the final period.

    Returns ``(att, counterfactual)`` where the counterfactual is
    ``alpha + gamma_i + delta_T`` for each treated unit -- the reference's
    ``predicted`` -- and the ATT is the mean gap on the treated post cells,
    also as the reference computes it.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    treated = np.flatnonzero(treated_mask > 0.5)
    n1 = treated.size

    # Design: intercept, unit dummies (unit 0 omitted), time dummies (period 0
    # omitted), treatment indicator.
    rows = N * T
    unit_idx = np.repeat(np.arange(N), T)
    time_idx = np.tile(np.arange(T), N)

    D = np.zeros(rows)
    if reference_offbyone:
        # T*n0, T*(n0+1), ..., T*N in 1-based indexing -> one donor cell plus
        # every treated post cell, in the reference's donors-then-treated order.
        n0 = N - n1
        flat = np.arange(T * n0, N * T + 1, T) - 1
        D[flat[flat < rows]] = 1.0
    else:
        D[(np.isin(unit_idx, treated)) & (time_idx == T - 1)] = 1.0

    X = np.zeros((rows, 1 + (N - 1) + (T - 1) + 1))
    X[:, 0] = 1.0
    for u in range(1, N):
        X[unit_idx == u, u] = 1.0
    for t in range(1, T):
        X[time_idx == t, N - 1 + t] = 1.0
    X[:, -1] = D

    coef, *_ = np.linalg.lstsq(X, Y.reshape(-1), rcond=None)
    alpha = coef[0]
    gamma = np.concatenate([[0.0], coef[1:N]])
    delta = np.concatenate([[0.0], coef[N : N + T - 1]])

    counterfactual = alpha + gamma[treated] + delta[T - 1]
    att = float(np.mean(Y[treated, -1] - counterfactual))
    return att, counterfactual


def idid(
    Y: np.ndarray,
    treated_mask: np.ndarray,
    mu: np.ndarray,
    lam: np.ndarray,
    mu_mean: float,
    lambda_mean: float,
    **kwargs,
) -> tuple[float, np.ndarray]:
    """Infeasible DiD: TWFE after removing the demeaned factor structure.

    The reference demeans by the DGP's mean parameters (``mu_mean``,
    ``lambda_mean``), not by sample means; kept.
    """
    correction = (mu - mu_mean) @ (lam - lambda_mean).T
    return twfe_did(np.asarray(Y, dtype=float) - correction, treated_mask, **kwargs)


def psc_weights_all(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    *,
    lambdas: tuple[float, ...] = (0.4, 0.8, 6.4),
) -> tuple[np.ndarray, float]:
    """Abadie-L'Hour penalized SC, one weight vector per treated unit.

    ``lambda`` is chosen by the reference's holdout rule: fit on all but the
    last pre-period, score on the last pre-period, take the lambda with the
    smallest RMSE across treated units (``run_pscm`` in
    ``4_add_PSC_to_Simulation.r``).
    """
    Y1_pre = np.asarray(Y1_pre, dtype=float)
    Y0_pre = np.asarray(Y0_pre, dtype=float)
    n1 = Y1_pre.shape[0]

    train1, test1 = Y1_pre[:, :-1], Y1_pre[:, -1]
    train0, test0 = Y0_pre[:, :-1], Y0_pre[:, -1]

    best_lam, best_rmse = lambdas[0], np.inf
    for lam in lambdas:
        W = np.vstack(
            [penalized_weights(train1[i], train0.T, lam) for i in range(n1)]
        )
        err = test1 - W @ test0
        rmse = float(np.sqrt(np.mean(err**2)))
        if rmse < best_rmse:
            best_lam, best_rmse = lam, rmse

    W = np.vstack(
        [penalized_weights(train1[i], train0.T, best_lam) for i in range(n1)]
    )
    return W, best_lam

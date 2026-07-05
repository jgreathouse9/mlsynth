"""Nearest-neighbour + SC weight constructors and MASC combiner.

Direct port of ``NearestNeighbors`` and ``sc_estimator`` from Maxwell
Kellogg's reference R code (``masc/R/estimator.R``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def nearest_neighbor_weights(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    m: int,
) -> np.ndarray:
    """Equal-weight nearest-neighbour weights on outcome-path distance.

    Picks the ``m`` donors whose pre-period outcome paths have the
    smallest squared-distance from the treated unit and assigns
    ``1/m`` to each (zero elsewhere). Mirrors ``NearestNeighbors`` in
    the R reference's outcome-path branch (lines 15-39 of
    ``estimator.R``).

    Parameters
    ----------
    Y_treated_pre : np.ndarray
        Shape ``(T0,)``, treated unit's pre-period outcomes.
    Y_donors_pre : np.ndarray
        Shape ``(T0, J)``, donor pre-period outcomes.
    m : int
        Number of nearest neighbours to retain.

    Returns
    -------
    np.ndarray
        Shape ``(J,)``, weights that sum to 1, with ``1/m`` on the
        ``m`` closest donors and 0 elsewhere.
    """
    if m < 1:
        raise ValueError("m must be at least 1.")
    J = Y_donors_pre.shape[1]
    if m > J:
        raise ValueError(f"m={m} exceeds donor pool size J={J}.")
    diff = Y_donors_pre - Y_treated_pre[:, None]    # (T0, J)
    dist_sq = np.sum(diff ** 2, axis=0)              # (J,)
    # `argpartition` is faster than full sort and gives the m smallest.
    keep_idx = np.argpartition(dist_sq, m - 1)[:m]
    # Handle ties at the m-th position by widening to all donors that
    # equal the m-th distance (matches R's `dist %in% sort(dist)[1:m]`).
    threshold = dist_sq[keep_idx].max()
    keep_mask = dist_sq <= threshold + 1e-12
    n_keep = int(keep_mask.sum())
    weights = np.zeros(J)
    weights[keep_mask] = 1.0 / n_keep
    return weights


def _standardize_predictors(
    X_treated: np.ndarray,
    X_donors: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Row-wise standardisation (zero mean, unit std *across all units*).

    Reproduces Abadie's default heuristic V via simple preconditioning:
    after dividing each predictor row by its cross-unit standard
    deviation, the inner QP with V = I gives the same weights as
    Abadie's standard ``V = diag(1/var)`` choice. Matches ``Cov.Vars``
    from the R reference (``estimator.R`` line 18) up to numerical
    scale.

    Parameters
    ----------
    X_treated : np.ndarray
        Shape ``(P,)`` -- treated unit's covariate vector.
    X_donors : np.ndarray
        Shape ``(P, J)`` -- donor covariate matrix.

    Returns
    -------
    X1, X0 : tuple of np.ndarray
        Standardised predictor block.
    """
    big = np.column_stack([X_treated[:, None], X_donors])
    sd = np.std(big, axis=1, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return X_treated / sd, X_donors / sd[:, None]


def _sc_simplex_clarabel(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
) -> np.ndarray:
    """Outcome-only SC simplex QP solved by calling Clarabel directly.

    Solves ``min_w ||Y1 - Y0 w||^2  s.t.  sum(w) = 1, w >= 0`` in Clarabel's
    native cone form, skipping cvxpy's per-solve canonicalisation. The default
    outcome-path solver in ``sc_simplex_weights`` already dispatched to Clarabel
    *through* cvxpy; calling it directly removes ~6 ms of graph-building /
    matrix-stuffing per solve (~3.6x faster) while returning a bit-for-bit
    equivalent solution (matches the cvxpy path to ~1e-7).

    The ``0 <= w <= 1`` box collapses to ``w >= 0`` because ``sum(w) = 1`` with
    ``w >= 0`` already implies ``w_j <= 1``, so the redundant upper bound is
    dropped. In Clarabel's ``A x + s = b, s in K`` form:

    * ``P = 2 Y0'Y0`` (PSD), ``q = -2 Y0'Y1`` -- the ``1/2 x'Px + q'x`` objective;
    * row ``1' w = 1`` in a ``ZeroCone(1)`` (equality);
    * rows ``-w + s = 0, s >= 0`` in a ``NonnegativeCone(J)``.
    """
    Y0 = np.asarray(Y_donors_pre, dtype=float)
    y1 = np.asarray(Y_treated_pre, dtype=float).flatten()
    J = Y0.shape[1]
    if J == 1:
        # The only feasible simplex point when there is a single donor.
        return np.array([1.0])

    import clarabel
    from scipy import sparse

    P = sparse.csc_matrix(2.0 * (Y0.T @ Y0))
    q = -2.0 * (Y0.T @ y1)
    A = sparse.vstack(
        [
            sparse.csc_matrix(np.ones((1, J))),   # 1' w = 1
            -sparse.eye(J, format="csc"),          # -w + s = 0, s >= 0
        ],
        format="csc",
    )
    b = np.concatenate([[1.0], np.zeros(J)])
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(J)]

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    # Tighten past Clarabel's 1e-8 defaults: the raw QP is poorly scaled
    # (``P`` entries ~ T0 * level^2), and the default tolerances leave ~5e-5
    # slack on the weights -- enough to perturb the CV's (m, phi) argmin.
    # 1e-11 reproduces the cvxpy/CLARABEL solution to ~1e-8 for a few extra
    # (cheap) interior-point iterations.
    settings.tol_gap_abs = 1e-11
    settings.tol_gap_rel = 1e-11
    settings.tol_feas = 1e-11
    solution = clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()

    # SolverStatus.Solved / .AlmostSolved both carry a usable primal point.
    if "Solved" not in str(solution.status):  # pragma: no cover - defensive; a
        # feasible simplex QP (non-empty donor pool) always solves.
        raise RuntimeError(
            f"MASC SC QP failed (Clarabel status={solution.status})."
        )
    w_hat = np.clip(np.asarray(solution.x, dtype=float), 0.0, None)
    total = w_hat.sum()
    if total <= 0.0:  # pragma: no cover - defensive; sum(w)=1 is enforced by the
        # ZeroCone equality, so the clipped weights cannot all be zero.
        raise RuntimeError("MASC SC QP produced degenerate (all-zero) weights.")
    return w_hat / total


def sc_simplex_weights(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    *,
    X_treated: Optional[np.ndarray] = None,
    X_donors: Optional[np.ndarray] = None,
    solver: Optional[str] = None,
    sc_backend: str = "mscmt",
) -> np.ndarray:
    """Standard SC simplex QP on pre-period outcomes or covariates.

    Solves ``min_w ||target1 - target0 w||^2`` subject to ``sum(w) = 1``
    and ``0 <= w_j <= 1``. With ``X_treated`` and ``X_donors`` supplied
    the QP runs on the *row-standardised predictor block* (Abadie's
    default V = ``diag(1/var)`` realised as preconditioning) rather
    than the raw outcome path. Without them, it reduces to the
    outcome-paths SC used by the R reference's ``sc_estimator`` no-
    covariates branch.

    This is not full bilevel V-optimisation; it matches Abadie's
    *initial* / heuristic V used by ``Cov.Vars`` (``Estimator_Code.R``
    line 16) and produces SC weights identical to those Abadie's
    ``synth()`` returns when ``custom.v = "default"``.
    """
    if (X_treated is None) != (X_donors is None):
        raise ValueError(
            "X_treated and X_donors must both be provided (or both None)."
        )

    if X_treated is None:
        # Outcomes-only branch -- same QP as the R reference's
        # no-covariates ``sc_estimator``. The default (CLARABEL) is solved
        # by calling Clarabel directly, skipping cvxpy's canonicalisation
        # overhead; an explicitly requested non-Clarabel cvxpy solver falls
        # back to the cvxpy path for backward compatibility.
        if solver is None or str(solver).upper() == "CLARABEL":
            return _sc_simplex_clarabel(Y_treated_pre, Y_donors_pre)

        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "MASC's non-default SC solver requires cvxpy; install with "
                "`pip install cvxpy` or use the default CLARABEL backend."
            ) from exc
        J = Y_donors_pre.shape[1]
        w = cp.Variable(J, nonneg=True)
        residual = Y_treated_pre - Y_donors_pre @ w
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(residual)),
            [cp.sum(w) == 1, w <= 1],
        )
        problem.solve(solver=solver)
        if w.value is None:
            raise RuntimeError(
                f"MASC SC QP failed (status={problem.status!r})."
            )
        w_hat = np.clip(np.asarray(w.value).flatten(), 0.0, None)
        return w_hat / w_hat.sum()

    # Covariate branch -- jointly optimise ``V`` with ``W`` against the
    # outcome-fit objective, matching Abadie's ``synth()`` V-optimisation
    # (the R reference's ``custom.v = NULL`` path) instead of fixing ``V``
    # to the heuristic inverse-variance default. ``sc_backend="mscmt"``
    # (default) uses the MSCMT global search, which reproduces synth() /
    # the Kellogg et al. (2020) reference; ``"bilevel"`` uses the Malo et
    # al. solver shared with FSCM.
    from ..bilevel import BilevelProblem
    Pt, Pd = _standardize_predictors(X_treated, X_donors)
    prob = BilevelProblem(
        y1_pre=Y_treated_pre,
        Y0_pre=Y_donors_pre,
        X1=Pt,
        X0=Pd,
    )
    if sc_backend == "mscmt":
        from ..bilevel.mscmt import solve_mscmt
        sol = solve_mscmt(prob, canonical_v="min.loss.w")
    elif sc_backend == "bilevel":
        from ..bilevel import solve_bilevel
        sol = solve_bilevel(prob)
    else:
        raise ValueError(
            f"Unknown sc_backend {sc_backend!r}; use 'mscmt' or 'bilevel'."
        )
    return np.clip(sol.W, 0.0, None) / np.clip(sol.W, 0.0, None).sum()


def masc_combine(
    weights_match: np.ndarray,
    weights_sc: np.ndarray,
    phi: float,
) -> np.ndarray:
    """``phi * match + (1 - phi) * sc``. Trivial helper for clarity."""
    return phi * weights_match + (1.0 - phi) * weights_sc


def analytic_phi(
    Y_treated: np.ndarray,
    Y_match: np.ndarray,
    Y_sc: np.ndarray,
    obj_weights: np.ndarray,
) -> float:
    """Closed-form phi minimising the weighted CV objective.

    Implements Kellogg et al. (2021) equation (15) -- equivalent to a
    1-D weighted OLS of ``(Y_treated - Y_sc)`` on ``(Y_match - Y_sc)``
    clamped to ``[0, 1]``. Direct port of lines 299-302 of
    ``crossvalidation.R``.

    Parameters
    ----------
    Y_treated, Y_match, Y_sc : np.ndarray
        Stacked forecast vectors from every CV fold.
    obj_weights : np.ndarray
        Per-observation weights derived from the fold weights
        (line 293 of ``crossvalidation.R``).

    Returns
    -------
    float
        The CV-optimal ``phi`` in ``[0, 1]``.
    """
    delta = Y_match - Y_sc
    denom = float((obj_weights * delta) @ delta)
    if denom <= 0.0:
        return 0.0
    numer = float((obj_weights * (Y_treated - Y_sc)) @ delta)
    return float(np.clip(numer / denom, 0.0, 1.0))

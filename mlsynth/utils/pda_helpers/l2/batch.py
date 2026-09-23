"""Batched L2-relaxation solver via a single OSQP factorization.

When several treated units share the **same** control pool (the
multiple-treated-units PDA), the L2-relaxation primal

    min_beta  (1/2) ||beta||^2   s.t.   || eta_j - Sigma beta ||_inf <= tau

has the *same* ``Sigma = X'X/T1`` for every treated unit ``j`` and every
penalty ``tau`` -- only the moment vector ``eta_j`` and the bound ``tau``
change. OSQP factorises its KKT system from ``(P = I, A = Sigma)`` **once**; we
then sweep every ``(j, tau)`` by updating only the constraint bounds
``l = eta_j - tau``, ``u = eta_j + tau`` and re-solving warm-started -- no
re-factorisation. This turns hundreds of conic solves into hundreds of cheap
ADMM updates (and matches a per-problem ``cvxpy`` solve to solver precision).

Two tolerances, because the sweep has two callers
-------------------------------------------------
The ``tau`` cross-validation and the final fit ask for different things. The
final fit is the estimate that gets reported and pinned, so it is solved to
``_SINGLE_EPS``. The sweep behind cross-validation only has to *order* its
candidates by validation error -- its winner is refit afterwards -- and at
``_SINGLE_EPS`` that ordering costs essentially the whole runtime of an ``l2``
fit, because the small-``tau`` end of the grid drives ADMM to its iteration cap.
``_GRID_EPS`` is the tolerance for that sweep; on the Hong Kong pre-period at
every window length, and on simulated panels with the donor pool at or above the
pre-period length, it selects the same ``tau`` as a ``_SINGLE_EPS`` sweep.

Reading the solver's answer
---------------------------
OSQP reports a status alongside ``x``, and ``x`` is finite whatever the status
says. On an ill-conditioned pre-period -- 24 Hong Kong donors on 24 periods, so
``Sigma`` has rank 23 -- it returns a *primal infeasibility certificate* at the
small-``tau`` end, whose ``x`` has entries of order 1e9. The certificate is
false there (the smallest reachable ``||eta - Sigma beta||_inf`` is ~1e-12, far
below the ``tau`` it fires at), but true or false it is not a primal solution,
and a finiteness check alone admits it. :func:`_usable_status` is what separates
an imprecise solution, which is kept, from an answer to a different question,
which is refused as ``NaN`` for the callers to translate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Statuses carrying a primal iterate for the problem that was posed. The first
# is a converged solve; the rest stopped early and are imprecise, which the
# validation ranking tolerates and the single solve reports through its own
# feasibility check. Every other status -- the infeasibility certificates,
# ``problem non convex``, ``interrupted``, OSQP's initial ``unsolved`` -- is
# either about a different problem or about no problem at all.
_USABLE_STATUSES = frozenset({
    "solved",
    "solved inaccurate",
    "maximum iterations reached",
    "run time limit reached",
})


def _usable_status(status: str) -> bool:
    """Does this OSQP status carry a primal iterate for the problem posed?"""
    return str(status).strip().lower() in _USABLE_STATUSES


def l2_relax_batch(
    Sigma: np.ndarray, Eta: np.ndarray, taus: np.ndarray,
    eps: float = 1e-5, max_iter: int = 4000,
) -> np.ndarray:
    """Solve the L2-relaxation primal for every ``(unit, tau)``.

    Parameters
    ----------
    Sigma : np.ndarray
        Shared ``(N, N)`` second-moment matrix.
    Eta : np.ndarray
        ``(N, J)`` moment vectors, one column per treated unit.
    taus : np.ndarray
        ``(K,)`` penalty grid.
    eps, max_iter : float, int
        OSQP tolerance and iteration cap (warm-started, no polish).

    Returns
    -------
    np.ndarray
        ``(J, K, N)`` coefficients ``beta[j, k]`` for unit ``j`` at ``taus[k]``.
        A ``(j, k)`` whose solve returned no primal iterate -- see
        :func:`_usable_status` -- is ``NaN``, which is how the callers learn the
        pair has no fit. Zeros would read as a legitimate one.
    """
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the batched L2-relaxation solver needs `osqp` and `scipy`."
        ) from exc

    Sigma = np.asarray(Sigma, dtype=float)
    Eta = np.asarray(Eta, dtype=float)
    taus = np.asarray(taus, dtype=float)
    N, J = Eta.shape
    K = taus.shape[0]
    order = np.argsort(-taus)                 # descending -> warm-start path

    prob = osqp.OSQP()
    prob.setup(
        P=sp.eye(N, format="csc"), q=np.zeros(N),
        A=sp.csc_matrix(Sigma), l=Eta[:, 0] - taus[0], u=Eta[:, 0] + taus[0],
        eps_abs=eps, eps_rel=eps, max_iter=max_iter,
        polish=False, warm_starting=True, verbose=False,
    )
    out = np.full((J, K, N), np.nan)
    for j in range(J):
        for k in order:
            prob.update(l=Eta[:, j] - taus[k], u=Eta[:, j] + taus[k])
            res = prob.solve()
            if (res.x is not None and _usable_status(res.info.status)
                    and np.all(np.isfinite(res.x))):
                out[j, k] = res.x
    return out


# Tight OSQP tolerance for the single-treated path: the benchmarks pin
# value-for-value results, so we solve the unique primal optimum to ~1e-8, which
# matches an interior-point (cvxpy/CLARABEL) solve. (At this tolerance OSQP's
# active-set polishing changes nothing, so it stays off -- and silent.)
_SINGLE_EPS = 1e-9
_SINGLE_MAX_ITER = 50000

# Tolerance for the cross-validation sweep, which ranks candidate taus by
# validation error and hands its winner to a _SINGLE_EPS refit. Solving the
# sweep to _SINGLE_EPS is where an l2 fit spends nearly all of its time: at
# N = 400 donors on 150 pre-periods the 80-tau grid takes 271s against 25s here,
# and the beta it returns agrees to 1.5e-04.
_GRID_EPS = 1e-6
_GRID_MAX_ITER = 50000


def l2_relax_solve(
    Sigma: np.ndarray, eta: np.ndarray, tau: float,
) -> np.ndarray:
    """One L2-relaxation primal solve (standardised scale) via tight OSQP.

    Returns the ``(N,)`` coefficient vector solving
    ``min ||beta||^2 / 2  s.t.  ||eta - Sigma beta||_inf <= tau``, or ``NaN``
    when the solver returned no primal iterate.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    return l2_relax_grid(Sigma, eta, np.asarray([float(tau)]),
                         eps=_SINGLE_EPS, max_iter=_SINGLE_MAX_ITER)[0]


def l2_relax_grid(
    Sigma: np.ndarray, eta: np.ndarray, taus: np.ndarray,
    eps: Optional[float] = None, max_iter: Optional[int] = None,
) -> np.ndarray:
    """L2-relaxation primal for one unit across a ``tau`` grid (shared Sigma).

    Returns ``(K, N)`` coefficients, one row per ``taus[k]``, ``NaN`` where the
    solver returned no primal iterate. A single KKT factorization is reused
    across the grid (the cross-validation hot loop). ``eps`` defaults to
    :data:`_GRID_EPS`, the ranking tolerance; :func:`l2_relax_solve` passes
    :data:`_SINGLE_EPS` for the fit that gets reported.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    return l2_relax_batch(
        Sigma, eta[:, None], np.asarray(taus, dtype=float),
        eps=_GRID_EPS if eps is None else eps,
        max_iter=_GRID_MAX_ITER if max_iter is None else max_iter,
    )[0]

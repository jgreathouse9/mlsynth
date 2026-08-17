"""NSC inner optimisation: eigenvalue-scaled (a, b) + simplex-affine QP.

The NSC weight problem (Tian 2023, eq. 7) is

.. math::

   \\min_w \\; \\| Z_1 - Z_0' w \\|^2
            + a \\sum_j |w_j| \\, \\| Z_1 - Z_0[j] \\|
            + b \\sum_j w_j^{\\,2}
   \\quad \\text{s.t.} \\quad \\sum_j w_j = 1.

The raw penalty multipliers ``a`` and ``b`` are scaled by the
eigenvalues of :math:`Z_0 Z_0'` so the dimensionless tuning
parameters ``a_star, b_star \\in [0, 1]`` (cf. paper paragraph
discussing eq. 7):

* ``b`` scales as :math:`b^* \\cdot \\lambda_{\\lceil n b^* \\rceil}`,
  where :math:`\\lambda_1 \\le \\dots \\le \\lambda_n` are the non-zero
  eigenvalues of :math:`Z_0 Z_0'`. At ``b_star = 1`` the L2 dominates
  and the weights are roughly uniform.
* ``a`` scales analogously by the eigenvalues of
  :math:`Z_0 Z_0' + \\mathrm{diag}(b)`. At ``a_star = 1`` only the
  nearest neighbour receives weight.

The QP is solved directly through Clarabel (one cached
constraint-skeleton per donor count) to skip the CVXPY
canonicalisation overhead — matches the pattern used in SparseSC,
SIV, and SYNDES.
"""

from __future__ import annotations

from typing import Tuple

import clarabel
import numpy as np
import scipy.sparse as sp

from ...exceptions import MlsynthEstimationError


# ---------------------------------------------------------------------------
# Eigenvalue-based (a, b) scaling
# ---------------------------------------------------------------------------

# NSC.R filters both spectra at this level before indexing them, so the same
# threshold decides how many eigenvalues ``a`` and ``b`` are chosen from.
_EIG_TOL = 1e-7


def design_eigenvalues(Z0: np.ndarray) -> np.ndarray:
    """Ascending non-zero eigenvalues of the *stacked* donor Gram.

    The matrix is the one the reference program is posed on. ``NSC.R`` writes
    the L1 term by splitting each weight into a non-negative pair, which stacks
    the design as ``Z* = [Z_0; -Z_0]`` and makes the quadratic form
    ``Z* Z*'`` -- a ``(2J, 2J)`` matrix -- and it is that spectrum the
    dimensionless ``b`` indexes.

    It is not built here. Writing ``A = Z_0 Z_0'``,

    .. math::

       Z^* Z^{*\prime} = \begin{pmatrix} A & -A \\ -A & A \end{pmatrix}
                        = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
                          \otimes A,

    and the left factor has eigenvalues ``{0, 2}``, so the Kronecker product's
    spectrum is ``{0}`` together with ``{2 lambda_i(A)}``. The non-zero part is
    therefore twice the non-zero spectrum of ``A``, available from one ``(J, J)``
    decomposition instead of a ``(2J, 2J)`` one. Equality with the literal
    construction is asserted in ``test_nsc_penalty_scaling``.

    Taking the spectrum of ``A`` itself -- which the auxiliary-variable encoding
    of the same L1 term suggests, since it never forms ``Z*`` -- halves every
    ``b``, and through ``b`` every ``a``, so the paper's tuning parameters buy
    half the penalty they name (#463).

    Parameters
    ----------
    Z0 : np.ndarray
        ``(J, p)`` matching matrix of donor units.
    """
    if Z0.size == 0:
        return np.asarray([], dtype=float)
    M = Z0 @ Z0.T
    M = 0.5 * (M + M.T)
    vals = 2.0 * np.linalg.eigvalsh(M)
    vals = vals[vals > _EIG_TOL]
    return np.sort(vals)


def _scale_param(p_star: float, eigvals: np.ndarray) -> float:
    """Translate the dimensionless ``p_star`` into the raw multiplier.

    Implements the paper's scaling: with ``n = len(eigvals)``,
    ``p = p_star * eigvals[ceil(n * p_star) - 1]``. Boundary handling:
    ``p_star == 0`` -> raw multiplier ``0`` (no penalty).
    """
    if p_star <= 0.0 or eigvals.size == 0:
        return 0.0
    n = eigvals.size
    idx = int(np.ceil(n * float(p_star))) - 1
    idx = max(0, min(idx, n - 1))
    return float(p_star) * float(eigvals[idx])


def scale_b(b_star: float, eigvals: np.ndarray) -> float:
    """Eigenvalue-scaled L2 multiplier."""
    return _scale_param(b_star, eigvals)


def scale_a(a_star: float, Z0: np.ndarray, b_raw: float) -> float:
    """Eigenvalue-scaled L1-discrepancy multiplier.

    Indexes the spectrum of the ridged stacked Gram ``Z* Z*' + b I`` -- the
    matrix ``NSC.R`` forms once ``b`` is fixed -- so the L1 anchor maxes out at
    the nearest-neighbour solution when ``a_star = 1`` whatever the L2 penalty.

    Adding ``b`` to the diagonal makes it full rank, so the spectrum has ``2J``
    entries and not ``rank(Z_0)`` of them: the ``2J - rank`` zero eigenvalues of
    ``Z* Z*'`` become ``b``, and the rest become ``2 lambda_i + b``. That
    multiplicity is why the error this corrects was not a constant. Indexing the
    ``(J, J)`` spectrum instead leaves ``J - rank`` copies of ``b`` where there
    should be ``2J - rank``, so ``ceil(a^* n)`` lands on the ridge value in both
    when the rank is low -- and on structurally different elements once the rank
    approaches ``J``, where the multipliers differ by 1.36 to 1.63 and not by
    two (#463).

    The tiny ``1e-8`` is the reference's, added so the quadratic form is
    positive definite for its solver; it is kept so the filtering threshold
    below decides the same way.
    """
    if a_star <= 0.0:
        return 0.0
    J = Z0.shape[0]
    if J == 0:
        return 0.0
    M = Z0 @ Z0.T
    M = 0.5 * (M + M.T)
    lam = 2.0 * np.linalg.eigvalsh(M)
    ridge = float(b_raw) + 1e-8
    nonzero = lam[lam > _EIG_TOL]
    # The (2J, 2J) ridged spectrum, assembled from the (J, J) one: the stacked
    # Gram's non-zero eigenvalues shifted, and one copy of the ridge for each of
    # its 2J - rank zeros.
    vals = np.concatenate([nonzero + ridge,
                           np.full(2 * J - nonzero.size, ridge)])
    vals = np.sort(vals[vals > _EIG_TOL])
    if vals.size == 0:
        return 0.0
    idx = int(np.ceil(vals.size * float(a_star))) - 1
    idx = max(0, min(idx, vals.size - 1))
    return float(a_star) * float(vals[idx])


# ---------------------------------------------------------------------------
# Inner QP
# ---------------------------------------------------------------------------

_CACHE: dict[int, tuple] = {}


def _affine_skeleton(J: int):
    """Return cached ``(A, b, cones, settings)`` for the adding-up constraint.

    NSC drops non-negativity; the only constraint is ``sum(w) = 1``.
    """
    cached = _CACHE.get(J)
    if cached is not None:
        return cached
    A = sp.csr_matrix(np.ones((1, J))).tocsc()
    b = np.asarray([1.0])
    cones = [clarabel.ZeroConeT(1)]
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.tol_gap_abs = 1e-9
    s.tol_gap_rel = 1e-9
    s.tol_feas = 1e-9
    s.max_iter = 500
    _CACHE[J] = (A, b, cones, s)
    return _CACHE[J]


def solve_nsc_weights(
    Z1: np.ndarray,
    Z0: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Solve the NSC weight QP for one (a, b) pair.

    The L1 penalty :math:`a \\sum_j |w_j| \\, d_j` is rewritten using
    auxiliary non-negative variables :math:`u_j \\ge |w_j|`; the full
    QP becomes

    .. math::

        \\min_{w, u} \\quad \\| Z_1 - Z_0' w \\|^2 + a \\, d^\\top u
                          + b \\, w^\\top w
        \\quad \\text{s.t.} \\quad
            \\sum_j w_j = 1, \\;
            u_j \\ge w_j, \\; u_j \\ge -w_j.

    Parameters
    ----------
    Z1 : np.ndarray
        Treated matching vector of length ``p``.
    Z0 : np.ndarray
        Donor matching matrix of shape ``(J, p)``.
    a : float
        Raw L1-discrepancy multiplier (post eigenvalue scaling).
    b : float
        Raw L2 multiplier (post eigenvalue scaling).

    Returns
    -------
    np.ndarray
        Donor weight vector of length ``J``, summing to 1.
    """

    Z1 = np.asarray(Z1, dtype=float).flatten()
    Z0 = np.asarray(Z0, dtype=float)
    J, p = Z0.shape

    # Pairwise discrepancies d_j = ||Z_1 - Z_0[j]||, mean-normalized so
    # the dimensionless ``a`` multiplier sits on the same scale as the
    # reference R implementation (NSC.R, ``dist_J / mean(dist_J)``).
    d_raw = np.linalg.norm(Z0 - Z1[None, :], axis=1)
    d_mean = float(d_raw.mean())
    d = d_raw / d_mean if d_mean > 0 else d_raw

    if a <= 0.0:
        # No L1 penalty: tighter QP without the auxiliary variables.
        H = Z0 @ Z0.T + float(b) * np.eye(J)
        H = 0.5 * (H + H.T)
        diag_mean = float(np.trace(H) / max(J, 1))
        if diag_mean <= 0:
            diag_mean = 1.0
        H = H + 1e-10 * diag_mean * np.eye(J)
        q = -2.0 * (Z0 @ Z1)
        P_mat = sp.csc_matrix(2.0 * H)
        A, b_rhs, cones, settings = _affine_skeleton(J)
        solver = clarabel.DefaultSolver(P_mat, q, A, b_rhs, cones, settings)
        sol = solver.solve()
        if str(sol.status) not in {"Solved", "AlmostSolved"}:
            raise MlsynthEstimationError(
                f"NSC inner QP (no-L1 path) failed: {sol.status}"
            )
        return np.asarray(sol.x, dtype=float)

    # Stacked variables x = [w; u], length 2J.
    # Objective: w' (Z0 Z0' + b I) w - 2 (Z0 Z1) w + a d' u
    H_top = Z0 @ Z0.T + float(b) * np.eye(J)
    H_top = 0.5 * (H_top + H_top.T)
    diag_mean = float(np.trace(H_top) / max(J, 1))
    if diag_mean <= 0:
        diag_mean = 1.0
    H_top = H_top + 1e-10 * diag_mean * np.eye(J)
    P_full = np.zeros((2 * J, 2 * J))
    P_full[:J, :J] = 2.0 * H_top
    q_full = np.concatenate([-2.0 * (Z0 @ Z1), float(a) * d])

    # Constraints: sum_j w_j = 1, u_j - w_j >= 0, u_j + w_j >= 0.
    A_eq = sp.csr_matrix(
        np.concatenate([np.ones(J), np.zeros(J)])[None, :]
    )
    I = np.eye(J)
    A_u_minus_w = sp.csr_matrix(np.hstack([-I, -I]))   # -(u - w) <= 0  -> for nonneg cone
    A_u_plus_w = sp.csr_matrix(np.hstack([I, -I]))     # -(u + w) <= 0
    A_stack = sp.vstack([A_eq, A_u_minus_w, A_u_plus_w]).tocsc()
    b_stack = np.concatenate([[1.0], np.zeros(J), np.zeros(J)])
    cones = [
        clarabel.ZeroConeT(1),
        clarabel.NonnegativeConeT(J),
        clarabel.NonnegativeConeT(J),
    ]

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-9
    settings.tol_feas = 1e-9
    settings.max_iter = 500

    solver = clarabel.DefaultSolver(
        sp.csc_matrix(P_full), q_full, A_stack, b_stack, cones, settings
    )
    sol = solver.solve()
    if str(sol.status) not in {"Solved", "AlmostSolved"}:
        # Fall back to a stronger ridge and try once more.
        H_top_ridge = H_top + 1e-6 * diag_mean * np.eye(J)
        P_full[:J, :J] = 2.0 * H_top_ridge
        settings.tol_gap_abs = 1e-6
        settings.tol_gap_rel = 1e-6
        settings.tol_feas = 1e-6
        sol = clarabel.DefaultSolver(
            sp.csc_matrix(P_full), q_full, A_stack, b_stack, cones, settings
        ).solve()
        if str(sol.status) not in {"Solved", "AlmostSolved"}:
            raise MlsynthEstimationError(
                f"NSC inner QP failed: {sol.status}"
            )

    x = np.asarray(sol.x, dtype=float)
    return x[:J]


def fit_nsc(
    Z1: np.ndarray,
    Z0: np.ndarray,
    a_star: float,
    b_star: float,
    eigvals: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Convenience wrapper: scale ``(a_star, b_star)`` and solve the QP.

    Returns
    -------
    w : np.ndarray
        Donor weights.
    a_scaled : float
        Raw L1 multiplier used.
    b_scaled : float
        Raw L2 multiplier used.
    """
    if eigvals is None:
        eigvals = design_eigenvalues(Z0)
    b_raw = scale_b(b_star, eigvals)
    a_raw = scale_a(a_star, Z0, b_raw)
    w = solve_nsc_weights(Z1, Z0, a_raw, b_raw)
    return w, a_raw, b_raw

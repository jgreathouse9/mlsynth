"""Inner W-weight QP for SparseSC.

Given V-weights ``v`` (a vector of predictor importance), the donor
weights ``w`` solve the canonical SCM simplex QP

    min over w >= 0, sum(w) = 1:  || V^(1/2) (X1 - X0 w) ||^2.

On the simplex ``X1 - X0 w = -(X0 - X1 1') w``, so the program is solved in
difference form: the design ``B = V^(1/2) (X0 - X1 1')`` against a zero target,
by the library's active-set solver. Three things follow (agents_solvers.md,
rules 2 to 6):

* the level of each predictor drops out, so a predictor recorded far from zero
  (a year, a population in levels) does not move ``w``;
* no Gram matrix ``X0' V X0`` is formed, so the condition number is not squared;
* inactive weights are exact zeros.

When the selected predictors let the treated unit be fitted exactly, or more
generally when the minimisers form a face, ``w`` is not identified by the fit,
and the outer V-search then optimises whichever member the inner solver hands
back. On a Basque fit (13 predictors, treated from 1975) about 6 percent of
inner solves land on a face, and the tie-break decided the estimate: the
least-norm member moved the fit to one predictor and an ATT of -0.526, against
-0.673 from Clarabel's undocumented interior point. The face is resolved by
Abadie and L'Hour's pairwise penalty in its vanishing limit: among the
minimisers, the one minimising ``sum_j w_j ||X1 - X0_j||_V^2``, which puts
weight on the donors closest to the treated unit. It is a linear programme over
the face, solved only when the face is not a point, so every unique answer is
unchanged. Donors with identical predictors are interchangeable in this
programme and share their weight equally. With this rule the Basque fit returns
Clarabel's donor weights and ATT, -0.6726, and the Proposition 99 benchmark is
unchanged (#934).

Each answer is certified by ``kkt_residual`` on ``B`` and refused above
``KKT_TOL`` with ``MlsynthEstimationError``. This replaced a Clarabel solve in
Gram form with a trace-scaled ridge and a looser retry, whose status was
accepted unchecked: on 20,000 inner solves sampled from a Basque fit its KKT
residual had median 3.7e-6 and maximum 0.38, and 199,478 weights sat in
(0, 1e-6). The active-set solve has median 2.4e-17 and maximum 2.8e-10 there.

The inner QP is called hundreds of thousands of times per fit, along L-BFGS-B
iterates that move ``v`` a little at a time, so each solve can start from the
previous answer: pass a ``WarmStart`` and it carries ``w`` from one call to the
next. On the same sequence a warm solve takes 106 microseconds; Clarabel took
410. The whole Basque fit takes about as long as before (192-219 s against
196-203 s, back to back): the outer search takes a different path, and faces
add a linear programme.

The ``solver`` argument is retained for API compatibility and selects nothing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.optimize import linprog

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from ..solvers.active_set import solve_simplex_qp
from ..weights.solve import KKT_TOL, kkt_residual
from ..weights.spec import WeightConstraint, WeightObjective

_SIMPLEX = WeightConstraint()
_PLAIN = WeightObjective()

# A donor is a candidate for the face when its reduced gradient is within this
# fraction of the design's largest column energy of the multiplier. The face of
# minimisers is exactly {w in the simplex : B w = B w*}, since the loss reads w
# only through B w, so admitting an extra donor costs at most a linear programme
# that does not move w, while leaving one out can miss the face's optimum. The
# active-set solve is exact (KKT residual 2.4e-17 median on a Basque sweep), so
# reduced gradients on a face are rounding. Measured: on 4,000 random exact-fit
# problems no face missed its optimum at 1e-9, and a Basque sweep ran the
# programme on 5.5 percent of solves, against 18 percent at 1e-6. With the
# least-norm solve as the first stage, whose ridge leaves gradients of about
# 1e-8 on a face, 1e-9 missed the optimum on 27 of the 4,000.
_FACE_ACTIVE_TOL: float = 1e-9
# A direction is flat when its singular value, relative to the largest, is below
# this: moving the whole simplex along it changes the loss by at most 1e-16 of
# its scale.
_FACE_RCOND: float = 1e-8


class WarmStart:
    """Carries the previous inner answer into the next solve of a sequence."""

    __slots__ = ("w",)

    def __init__(self) -> None:
        self.w: Optional[np.ndarray] = None


def _face(B: np.ndarray, w: np.ndarray) -> tuple:
    """The candidate donors ``S`` and an orthonormal basis ``Z`` of the face's directions.

    Every minimiser is ``w`` plus a direction ``d`` supported on ``S`` with
    ``B d = 0`` and ``1' d = 0``; ``Z`` spans them, empty when ``w`` is the only
    minimiser. Both tests are relative to the design's own scale.
    """
    scale = float(np.max(np.sum(B * B, axis=0)))
    if scale == 0.0:
        S = np.arange(w.size)
        M = np.ones((1, S.size))
    else:
        g = B.T @ (B @ w)
        S = np.flatnonzero((w > 0.0) | (np.abs(g - float(g @ w)) <= _FACE_ACTIVE_TOL * scale))
        M = np.vstack([B[:, S] / np.sqrt(scale), np.ones((1, S.size))])
    _, sv, Vt = np.linalg.svd(M, full_matrices=True)
    rank = int(np.sum(sv > _FACE_RCOND * sv[0]))
    return S, Vt[rank:].T


def _pairwise_member(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Among the minimisers, the one minimising ``sum_j w_j ||B_j||^2`` (Abadie and L'Hour).

    ``B_j`` is donor ``j``'s V-weighted discrepancy from the treated unit, so this
    is their pairwise penalty as it vanishes. The programme is posed over the
    face's directions, ``w_S = w*_S + Z y``, and its objective ``c_S' Z`` is
    divided by its largest entry, so HiGHS's tolerances are relative to what the
    programme compares (rule 3). With the raw costs, which span several orders of
    magnitude, HiGHS failed on 5 of 179 Basque face LPs; with the costs divided by
    their largest entry it returned a member 1.1e-5 above the optimum on a face
    whose two best donors differ by 2e-5.

    ``w`` is kept when it is already optimal, so a face whose members all cost
    the same keeps the active-set answer; ``_share_duplicates`` then makes it a
    function of the panel when the tie is between identical donors. A tie between
    distinct donors at exactly the same pairwise distance would leave the choice
    to HiGHS's pivot order; with continuous predictors it has probability zero.
    """
    S, Z = _face(B, w)
    if Z.shape[1] == 0:
        return w
    c = np.sum(B * B, axis=0)
    q = c[S] @ Z
    if not np.any(q):
        return w                      # every member costs the same
    q = q / np.abs(q).max()
    res = linprog(q, A_ub=-Z, b_ub=w[S], bounds=[(None, None)] * Z.shape[1], method="highs")
    if res.status != 0:
        raise MlsynthEstimationError(
            f"SparseSC: the inner QP's tie-break on a face of minimisers failed ({res.message})."
        )
    if float(q @ res.x) >= -1e-9:
        return w                      # w is already optimal for the programme
    out = np.zeros_like(w)
    out[S] = w[S] + Z @ res.x
    # HiGHS meets the bounds to its feasibility tolerance; within it a weight is zero
    # (rule 4), and the certificate in ``solve_w`` checks the repaired point.
    out[out < 1e-9] = 0.0
    return out / out.sum()


def _share_duplicates(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Spread each group of identical donors' total weight equally across the group.

    Identical columns of ``B`` (donors with the same predictors) are
    interchangeable in the inner problem and in the tie-break, so which of them
    the solver loads is a property of donor order. The equal split is the
    symmetric choice and the least-norm one. Columns are compared exactly: they
    are formed from stored data by the same operations (rule 4).
    """
    _, inverse, counts = np.unique(B.T, axis=0, return_inverse=True, return_counts=True)
    if counts.max() == 1:
        return w
    inverse = inverse.ravel()
    total = np.bincount(inverse, weights=w)
    return total[inverse] / counts[inverse]


def solve_w(
    v: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    solver: Any = None,  # noqa: ARG001 -- kept for API compatibility
    *,
    warm_start: Optional[np.ndarray] = None,
    warm: Optional[WarmStart] = None,
) -> np.ndarray:
    """Return the certified donor-weight vector ``w`` on the simplex.

    A unique minimiser is returned as the active-set solve finds it; on a face
    of minimisers, the member closest to the treated unit in Abadie and
    L'Hour's pairwise sense, with identical donors sharing their weight.

    Parameters
    ----------
    v : np.ndarray
        Length-``P`` V-weight vector, non-negative.
    X1 : np.ndarray
        Length-``P`` treated predictor vector.
    X0 : np.ndarray
        ``(P, N)`` donor predictor matrix.
    solver : Any, optional
        Unused. Retained for backwards compatibility.
    warm_start : np.ndarray, optional
        A starting point for the active-set solve; a point off the simplex is
        ignored there.
    warm : WarmStart, optional
        A holder whose ``w`` is used as the start when ``warm_start`` is not
        given, and which receives the answer.

    Raises
    ------
    MlsynthDataError
        If a predictor is not finite.
    MlsynthEstimationError
        If ``v`` has a negative or non-finite entry, the tie-break on a face
        fails, or the answer fails its KKT certificate.
    """
    v = np.asarray(v, dtype=float).ravel()
    X1 = np.asarray(X1, dtype=float).ravel()
    X0 = np.asarray(X0, dtype=float)
    if not (np.all(np.isfinite(X0)) and np.all(np.isfinite(X1))):
        raise MlsynthDataError("SparseSC: the predictors must be finite.")
    if not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise MlsynthEstimationError("SparseSC: the V-weights v must be finite and non-negative.")

    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    target = np.zeros(B.shape[0])
    start = warm_start if warm_start is not None else (warm.w if warm is not None else None)
    w = _share_duplicates(B, _pairwise_member(B, solve_simplex_qp(B, target, warm_start=start)))
    kkt = kkt_residual(B, target, w, 0.0, _SIMPLEX, _PLAIN)
    if not kkt <= KKT_TOL:
        raise MlsynthEstimationError(
            f"SparseSC: the inner simplex QP's answer fails its KKT certificate "
            f"(residual {kkt:.2e} against {KKT_TOL:.0e})."
        )
    if warm is not None:
        warm.w = w
    return w

"""Mode-aware optimality certificate for the SYNDES design.

The SYNDES design MIP is NP-hard; SCIP's cost is almost entirely in *proving*
optimality, because its dual bound is the continuous (McCormick) relaxation,
which is very loose for the two-way objective. This module supplies a valid
lower bound on the optimal objective *without* branch-and-bound, so a fitted
design can be reported with a validated optimality gap (``LB <= optimum <=
incumbent``) -- the rigorous form of "close enough, stop".

The right lower bound depends on the mode's geometry:

* ``global_equal_weights`` (one-way): the treated weights are pinned to ``1/K``,
  so ``D`` enters the residual linearly -- no bilinear term -- and the plain
  continuous relaxation is already tight (a single convex QP).
* ``global_2way`` (two-way): the ``q = w*D`` bilinear makes the continuous
  relaxation useless (~80% gap). The SDP / moment (Shor--Lasserre level-1)
  relaxation adds ``D_i^2 = D_i`` and ``w_i D_i = q_i`` as exact second moments
  and closes ~90% of the gap -- but it is ``O(N^3)`` and gated by ``sdp_n_max``.
* ``per_unit``: the weights are an ``(N, N)`` matrix, so the SDP lift is
  ``O(N^4)`` (intractable) and the continuous relaxation is very loose (~70%).
  There is no cheap tight bound; the certificate is returned ``certified=False``.

The returned lower bound is always *valid* (a relaxation optimum), modulo the
SDP solver's numerical tolerance (SCS is first-order); ``certified`` flags
whether it is tight enough to be a useful certificate for the mode.

This certificate is an mlsynth addition, not part of Doudchenko et al. (2021):
the paper proves the design NP-hard and gives the mixed-integer program but
derives no relaxation-based lower bound. It is a design-time diagnostic computed
on the pre-treatment panel and does not change the design SYNDES returns.

References
----------
Bomze, Peng, Qiu & Yildirim (2023), "On Tractable Convex Relaxations of Standard
Quadratic Optimization Problems under Sparsity Constraints" (arXiv:2310.04340) --
the Shor and RLT relaxations of the cardinality-constrained mixed-binary QP this
lift instantiates. Han, Gomez & Atamturk (2022), "The Equivalence of Optimal
Perspective Formulation and Shor's SDP for Quadratic Programs with Indicator
Variables" (Oper. Res. Lett. 50) -- for indicator quadratics the perspective
reformulation and Shor's SDP coincide, so lifting is the general tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthConfigError
from .formulation import build_syndes_problem_components
from .optimization import estimate_lambda

_ONE_WAY = "global_equal_weights"
_TWO_WAY = "global_2way"
_PER_UNIT = "per_unit"
_MODES = (_ONE_WAY, _TWO_WAY, _PER_UNIT)


@dataclass(frozen=True)
class SYNDESCertificate:
    """A validated optimality gap for a fitted SYNDES design.

    Attributes
    ----------
    lower_bound : float or None
        A valid lower bound on the optimal objective (``None`` only if the
        bound solve failed).
    optimality_gap : float or None
        ``(incumbent - lower_bound) / |incumbent|`` clamped at 0 -- the fraction
        by which the incumbent design may exceed the true optimum. ``None`` when
        there is no bound.
    certified : bool
        Whether ``lower_bound`` is tight enough to be a meaningful certificate
        for this mode (True for one-way and in-range two-way; False for per-unit
        and out-of-range two-way, where the bound is valid but loose).
    method : str
        ``"continuous_relaxation"`` or ``"sdp_moment"``.
    note : str
        Human-readable caveat (empty when fully certified).
    """

    lower_bound: Optional[float]
    optimality_gap: Optional[float]
    certified: bool
    method: str
    note: str = ""


def _continuous_relaxation_bound(Y: np.ndarray, K: int, mode: str, lam: float) -> float:
    """Continuous relaxation optimum (``D`` in ``[0, 1]``): a valid lower bound."""
    _, N = Y.shape
    D = cp.Variable(N, nonneg=True)
    comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode=mode)
    prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints) + [D <= 1])
    prob.solve(solver=cp.CLARABEL)
    return float(comp.objective.value)


def _sdp_moment_bound_two_way(Y: np.ndarray, K: int, lam: float) -> float:
    """SDP / moment (Shor level-1) lower bound for the two-way objective.

    Lifts ``x = [w; q; D]`` to a moment matrix and adds the constraints the
    McCormick relaxation drops: ``D_i^2 = D_i`` and ``w_i D_i = q_i = q_i D_i``.
    """
    T, N = Y.shape
    G = Y.T @ Y
    n = 3 * N
    M = cp.Variable((n + 1, n + 1), PSD=True)
    x = M[0, 1:]
    X = M[1:, 1:]
    w, q, D = x[:N], x[N:2 * N], x[2 * N:]
    Xww = X[:N, :N]
    Xqq = X[N:2 * N, N:2 * N]
    Xqw = X[N:2 * N, :N]
    cons = [
        M[0, 0] == 1,
        cp.sum(q) == 1, cp.sum(w) == 2,
        q <= D, q <= w, q >= w - (1 - D),
        cp.sum(D) == K, w >= 0, q >= 0, D <= 1,
    ]
    for i in range(N):
        cons += [X[2 * N + i, 2 * N + i] == D[i],   # D_i^2 = D_i
                 X[i, 2 * N + i] == q[i],           # w_i D_i = q_i
                 X[N + i, 2 * N + i] == q[i]]        # q_i D_i = q_i
    obj = (1.0 / T) * (4 * cp.sum(cp.multiply(G, Xqq))
                       - 4 * cp.sum(cp.multiply(G, Xqw))
                       + cp.sum(cp.multiply(G, Xww))) + lam * cp.trace(Xww)
    cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.SCS, max_iters=8000, eps=1e-5)
    return float(obj.value)


def _gap(incumbent: float, lb: float) -> float:
    denom = abs(incumbent) if abs(incumbent) > 1e-12 else 1.0
    return max(0.0, (incumbent - lb) / denom)


def syndes_certificate(
    Y: np.ndarray,
    K: int,
    mode: str,
    incumbent_obj: float,
    *,
    lam: Optional[float] = None,
    sdp_n_max: int = 120,
) -> SYNDESCertificate:
    """Certify a SYNDES design's optimality gap with a mode-appropriate bound.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix ``(T, N)`` -- the same matrix the design
        was optimized on.
    K : int
        Number of treated units.
    mode : {"global_equal_weights", "global_2way", "per_unit"}
        SYNDES formulation the design used.
    incumbent_obj : float
        Objective value of the fitted design (``SYNDESDesign.objective_value``).
    lam : float, optional
        Regularization; estimated from ``Y`` when ``None`` (must match the fit).
    sdp_n_max : int, optional
        Largest ``N`` for which the two-way SDP bound is attempted (it is
        ``O(N^3)``); above it the two-way certificate falls back to the loose
        continuous bound with ``certified=False``.

    Returns
    -------
    SYNDESCertificate
    """
    if mode not in _MODES:
        raise MlsynthConfigError(
            f"syndes_certificate: unknown mode {mode!r}; expected one of {_MODES}.")
    _, N = Y.shape
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)

    if mode == _ONE_WAY:
        lb = _continuous_relaxation_bound(Y, K, mode, lam_value)
        return SYNDESCertificate(lb, _gap(incumbent_obj, lb), True,
                                 "continuous_relaxation")

    if mode == _TWO_WAY:
        if N <= sdp_n_max:
            lb = _sdp_moment_bound_two_way(Y, K, lam_value)
            return SYNDESCertificate(lb, _gap(incumbent_obj, lb), True, "sdp_moment")
        lb = _continuous_relaxation_bound(Y, K, mode, lam_value)
        return SYNDESCertificate(
            lb, _gap(incumbent_obj, lb), False, "continuous_relaxation",
            note=(f"N={N} exceeds sdp_n_max={sdp_n_max}; the two-way SDP bound was "
                  "skipped and the continuous bound is loose (not a tight certificate)."))

    # per_unit: no cheap tight bound (SDP is O(N^4); continuous relaxation ~70% loose)
    lb = _continuous_relaxation_bound(Y, K, mode, lam_value)
    return SYNDESCertificate(
        lb, _gap(incumbent_obj, lb), False, "continuous_relaxation",
        note=("per-unit has an (N,N) weight matrix, so the SDP lift is intractable "
              "and the continuous relaxation is loose; the gap is not a tight "
              "certificate."))

"""MAREX's exact backend: the design as a bounded search over treated sets.

MAREX's MIQP chooses, per cluster, a treated set ``S``, treated weights ``w`` on
the simplex over ``S`` and control weights ``v`` on the simplex over the rest of
the cluster. For the ``standard``, ``penalized`` and ``unit_penalized`` designs
the objective is a sum of a treated-side term and a control-side term,

    ||A - B w||^2 + a'w   +   ||A - B v||^2 + b'v,

with ``A`` the cluster's predictor mean, the columns of ``B`` its markets, and
``a``, ``b`` non-negative penalty vectors (zero for ``standard``). Once ``S`` is
named the two terms share nothing, so

    f(S) = g(S) + h(C(S)),

two simplex-constrained least-squares fits, each solved exactly by
:func:`mlsynth.utils.solvers.active_set.solve_simplex_qp`. ``C(S)`` is the
cluster less ``S`` and less any donor a treated market excludes. The design is
a search over treated sets with an exact oracle, not a mixed-integer program.

The search. Both terms are non-negative, so ``f(S) >= g(S) >= lb(S)`` for any
lower bound ``lb`` on the treated fit. Every admissible candidate is priced by
``lb`` in batches, and only candidates whose bound is under the best total found
reach the oracle. A single streaming pass keeps the ``BUFFER`` best-bounded
candidates and evaluates them in bound order, which sets the incumbent; if more
candidates than the buffer held have a bound under it, a second pass collects
and evaluates those. Every candidate is therefore either solved exactly or
discarded on a bound that clears the incumbent, and the result is the optimum.

The bound. Because the weights sum to one, subtracting the target from every
column changes nothing: ``||A - B w|| = ||(B - A 1') w||``. On those centred
columns the target is zero, and dropping only ``w >= 0`` leaves a closed form,

    min { w'G w + a'w : 1'w = 1 } = (2 + t)^2 / (4 s) - u / 4,

with ``G = B~'B~`` restricted to ``S``, ``s = 1'G^-1 1``, ``t = 1'G^-1 a`` and
``u = a'G^-1 a`` (``1 / s`` when ``a = 0``). It equals ``g(S)`` whenever its
minimiser is already non-negative. SYNDES's exact backend bounds its own
objective the same way (:mod:`mlsynth.utils.syndes_helpers.gram`).

The weakly-targeted design. Its objective ties the synthetic control to the
synthetic treated instead of to the target,

    ||A - B w||^2 + beta ||B w - B v||^2,

so naming ``S`` does not split it: the treated and control weights still share
a term. What naming ``S`` leaves is a convex problem over a product of two
simplexes, ``min_w phi(w)`` with ``phi(w) = ||A - B_S w||^2 + beta d(B_S w)^2``
and ``d`` the distance to the control hull ``conv(B_C)``. :func:`coupled_fit`
minimises ``phi`` by Newton steps over the faces of the control simplex. On a
face ``F`` the projection of ``t`` onto the face's affine hull is
``c0 + P (t - c0)``, with ``P`` the projector onto the span of the face's edge
directions, so the residual is ``(I - P)(t - c0)`` and ``phi`` restricted to the
face is one least-squares fit over the treated simplex, with rows
``[B_S; sqrt(beta) (I - P) B_S]`` and target ``[A; sqrt(beta) (I - P) c0]``.
That fit is the step. A step is kept only if the true ``phi``, with the
controls refitted exactly, falls; otherwise it is halved, and when no halving
helps the fit takes one alternating sweep (the treated fit to
``(A + beta B v) / (1 + beta)``, then the controls) instead.

Each step solves the current face exactly, so the step count does not grow with
``beta``: across random problems it was at most three. The alternation it
replaced converges at the rate ``beta / (1 + beta)`` and took a median of 1,750
sweeps per candidate at ``beta = 100`` on GeoLift's 40 markets. The fit stops
on the Frank-Wolfe gap, ``sum over blocks of grad'x - min_j grad_j``, which
bounds the distance to the optimum from above, and raises when the gap has not
closed within ``COUPLED_MAX_ITER`` steps.

The same gap gives a certified lower bound at every step, the value less the
gap. The search hands each fit the incumbent as a cutoff, and a fit stops and
discards its candidate as soon as that lower bound clears it. The coupling term
is non-negative, so the treated-side fit and its bound still sit under the
objective and prune before any fit. At large ``beta`` they prune little,
because the coupling dominates the objective and they cannot see it.

The coupling bound sees it. The synthetic treated lies in ``conv B_S`` and the
synthetic control in ``conv B_C``, so

    f(S) >= g(S) + beta (1 + lam) dist(conv B_S, conv B_C)^2,

with ``lam`` the control penalty below (zero without it). Wolfe's (1976)
nearest-point method, run in lockstep over a batch of candidates
(:func:`_wolfe_lockstep`), finds a direction ``z``, and his Theorem 2.2 turns
any direction into a lower bound on the distance, which :func:`hull_separation`
certifies against the data with the rounding bounded a priori.
:class:`CouplingBound` adds it to the closed-form treated bound, so pricing a
candidate takes no fit. The search screens each batch of candidates under the
incumbent with it before fitting any of them, and the cutoff keeps each fit it
leaves to about one step. On GeoLift's 40 markets with four treated, at
``beta = 100``, it left 677 of 91,390 candidates to the coupled fit.

The control penalty. Vives-i-Bastida (2022, program 1) adds Abadie and
L'Hour's penalty on the controls, ``beta lam sum_j v_j ||B_S w - b_j||^2``.
Because ``sum v = 1`` it equals ``||t||^2 - 2 t'B_C v + sq'v`` with
``sq_j = ||b_j||^2``, and the objective stays convex exactly when
``beta lam (1 + lam) <= 1``. With every point measured from the target, the
best controls for ``t = B_S w`` are the simplex fit to ``(1 + lam) t`` with
the linear term ``lam sq``; on a face the controls' part is
``||(I - P)((1 + lam) t - b_1)||^2 + lam (1 + lam) q't`` up to a constant,
``q = D (D'D)^+ g`` with ``D`` the face's edges and ``g`` the differences of
their squared norms, so the step is one fit with rows ``sqrt(1 - kappa) B_S``
and ``sqrt(beta) (1 + lam) (I - P) B_S`` and the linear term
``kappa B_S'q``, ``kappa = beta lam (1 + lam)``. It took at most four steps
over random problems from ``beta = 0.01`` to 100 and up to 0.99 of the bound.
By convexity of the square the penalty is at least ``beta lam ||t - B_C v||^2``,
which is the ``1 + lam`` in the coupling bound. The bound does not see the
spread of the controls around ``B_C v``, which the penalty also charges, so it
prunes less of this objective.

Restrictions. Forced units, forbidden units and stratum quotas decide which
candidates exist, so :class:`~mlsynth.utils.syndes_helpers.enumeration.SearchSpace`
builds candidates inside them and counts them exactly; conflict pairs are a
test on a finished candidate (:class:`~mlsynth.utils.syndes_helpers.exact.SubsetFilter`).
Donor exclusions shrink each candidate's control pool, which the bound never
sees, since it reads only the treated side. Clusters are searched independently,
so a conflict pair or a stratum that spans two clusters couples their searches
and is refused.

What this backend does not solve. A cost budget constrains the treated
weights; ``relaxed`` and the branch-and-bound knobs (``solver``, ``warm_start``,
``time_limit``) belong to the MIQP; a negative ``beta`` makes the coupling
concave. The config refuses each with ``backend="exact"``. Above
``CANDIDATE_LIMIT`` candidates in a cluster, enumeration stops being affordable
and the search raises with the MIQP as the way out.
"""

from __future__ import annotations

import dataclasses
import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..solvers.active_set import solve_simplex_qp
from ..syndes_helpers.enumeration import SearchSpace
from ..syndes_helpers.exact import SubsetFilter
from .formulation import (
    build_membership_mask,
    compute_cluster_means_members,
    get_per_cluster_param,
    per_unit_reproducibility,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
    validate_scm_inputs,
)
from .optimization import _augment_fit, finish_design

#: Designs whose objective splits into a treated term and a control term.
SEPARABLE_DESIGNS = frozenset({"standard", "penalized", "unit_penalized"})

#: Designs the backend solves: the separable ones, and the coupled one.
EXACT_DESIGNS = SEPARABLE_DESIGNS | {"weakly_targeted"}

#: The coupled fit stops when its Frank-Wolfe gap is under this, relative.
COUPLED_TOL = 1e-10

#: Steps the coupled fit may take before it raises uncertified.
COUPLED_MAX_ITER = 20_000

#: Above this many candidates in one cluster the search refuses and names the MIQP.
CANDIDATE_LIMIT = 20_000_000

#: Candidates priced per batch.
CHUNK = 20_000

#: Best-bounded candidates kept from the first pass.
BUFFER = 512

#: A candidate is discarded only when its bound clears the cutoff by this much,
#: relative; it covers the floating-point error of the closed-form bound.
PRUNE_MARGIN = 1e-6

#: Corral slots per candidate in Wolfe's method; a candidate whose corral fills
#: is fitted.
COUPLING_CORRAL = 24

#: Lockstep iterations of Wolfe's method before the candidates still running are
#: fitted.
COUPLING_MAX_ITER = 200

#: A corral weight at or under this is zero, and a relative gap at or under it
#: has converged.
COUPLING_TOL = 1e-12

#: Candidates fitted unscreened while no incumbent exists for the coupling bound
#: to aim at.
SCREEN_SEED = 8

#: Candidates the coupling bound prices per lockstep batch.
SCREEN_BATCH = 512

#: Why Wolfe's method stopped on a candidate: its bound reached what was asked;
#: it converged, or met a support point already in its corral, short of that;
#: its corral filled; or the batch ran out of iterations.
WOLFE_CLEARED, WOLFE_CONVERGED, WOLFE_CORRAL_FULL, WOLFE_OUT_OF_ITERATIONS = (
    1, 2, 3, 4)

_UNIT_ROUNDOFF = 2.0 ** -53
_SMALLEST_SUBNORMAL = 2.0 ** -1074


@dataclass(frozen=True)
class ClusterDesign:
    """One cluster's design: the treated set, its fits and its weights.

    Indices are local to the cluster: positions in the ``B`` the search was
    given.
    """

    value: float
    treated: Tuple[int, ...]
    control: Tuple[int, ...]
    w: np.ndarray
    v: np.ndarray


# ----------------------------------------------------------------------
# the oracle and the bound
# ----------------------------------------------------------------------


def side_fit(B: np.ndarray, A: np.ndarray, cols: Sequence[int],
             linear: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """``min ||A - B[:, cols] w||^2 + linear'w`` over the simplex, exactly.

    Returns the minimum and the weights, in the order of ``cols``.
    """
    cols = list(cols)
    w = solve_simplex_qp(B[:, cols], A, linear=linear)
    r = A - B[:, cols] @ w
    value = float(r @ r)
    if linear is not None:
        value += float(np.dot(linear, w))
    return value, w


class CoupledFitNotCertified(MlsynthEstimationError):
    """The coupled fit ran out of sweeps before its gap closed."""


class SearchNotClosed(MlsynthEstimationError):
    """The branch and bound ran out of bounds before its gap closed."""


def _coupled_point(Bs: np.ndarray, Bc: np.ndarray, A: np.ndarray, beta: float,
                   w: np.ndarray, v_start: Optional[np.ndarray],
                   lam: float = 0.0, sq: Optional[np.ndarray] = None
                   ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """At treated weights ``w``: the best controls, the value and its gap.

    The controls are the exact simplex fit to ``B_S w``. The Frank-Wolfe gap,
    ``sum over blocks of grad'x - min_j grad_j``, bounds the value's distance to
    the joint minimum from above.

    With a control penalty ``lam`` the objective gains
    ``beta lam sum_j v_j ||B_S w - b_j||^2``. On the simplex that term is
    ``||t||^2 - 2 t'B_C v + sq'v`` with ``sq_j = ||b_j||^2``, so the best
    controls are the simplex fit to ``(1 + lam) t`` with the linear term
    ``lam sq``.
    """
    t = Bs @ w
    if lam:
        v = np.asarray(solve_simplex_qp(Bc, (1.0 + lam) * t, warm_start=v_start,
                                        linear=lam * sq), dtype=float)
    else:
        v = np.asarray(solve_simplex_qp(Bc, t, warm_start=v_start), dtype=float)
    c = Bc @ v
    r, d = t - A, t - c
    value = float(r @ r + beta * (d @ d))
    grad_w = 2.0 * (Bs.T @ (r + beta * d))
    grad_v = -2.0 * beta * (Bc.T @ d)
    if lam:
        dist = ((Bc - t[:, None]) ** 2).sum(axis=0)
        value += float(beta * lam * (v @ dist))
        grad_w = 2.0 * (Bs.T @ (r + beta * (1.0 + lam) * d))
        grad_v = grad_v + beta * lam * dist
    gap = float(grad_w @ w - grad_w.min()) + float(grad_v @ v - grad_v.min())
    return value, gap, v, c


def check_control_penalty(beta: float, control_penalty: float) -> None:
    """Refuse a control penalty that is negative or breaks convexity.

    On the control simplex the weakly-targeted objective with the penalty has
    quadratic part ``(1 + beta + beta lam) ||u||^2 - 2 beta (1 + lam) u's +
    beta ||s||^2`` in ``u = B_S w`` and ``s = B_C v``, positive semidefinite
    exactly when ``beta lam (1 + lam) <= 1``.
    """
    lam = float(control_penalty)
    if lam < 0:
        raise MlsynthConfigError(
            f"control_penalty must be >= 0; got {lam}.")
    kappa = float(beta) * lam * (1.0 + lam)
    if kappa > 1.0 + 1e-12:
        raise MlsynthConfigError(
            f"beta * control_penalty * (1 + control_penalty) = {kappa:.6g} "
            f"exceeds 1, where the weakly-targeted objective with the control "
            f"penalty stops being convex and has no minimum to certify. Lower "
            f"beta or control_penalty.")


def coupled_fit(B: np.ndarray, A: np.ndarray, treated: Sequence[int],
                control: Sequence[int], beta: float, *,
                w0: Optional[np.ndarray] = None, tol: Optional[float] = None,
                max_iter: Optional[int] = None, cutoff: Optional[float] = None,
                return_info: bool = False, control_penalty: float = 0.0):
    """``min ||A - B_S w||^2 + beta ||B_S w - B_C v||^2`` over two simplexes.

    With ``control_penalty`` ``lam > 0`` the objective gains Abadie and
    L'Hour's control penalty, ``beta lam sum_j v_j ||B_S w - b_j||^2``
    (Vives-i-Bastida 2022, program 1); see the module docstring for the face
    step it takes. ``beta lam (1 + lam)`` must not exceed 1.

    Newton steps over the faces of the control simplex (see the module
    docstring), each accepted only if the value falls, with an alternating
    sweep when no step does. Stops once the Frank-Wolfe gap is under
    ``tol * (1 + value)``; the value returned is then within that of the joint
    minimum. ``w0`` starts the treated weights, by default at the treated fit
    alone.

    With ``cutoff``, the fit returns ``None`` as soon as its certified lower
    bound, the value less the gap, exceeds it: the candidate cannot beat a
    design already found, and finishing the fit would only confirm that.

    Returns the value and the weights, in the order of ``treated`` and
    ``control``, and with ``return_info`` a dict counting ``iterations``,
    ``newton_steps`` and ``sweeps``.

    Raises
    ------
    CoupledFitNotCertified
        When ``max_iter`` steps leave the gap open. The value reached is above
        the minimum by an unknown amount, and ranking candidates on it could
        discard the optimal design.
    MlsynthConfigError
        When ``control_penalty`` is negative or ``beta lam (1 + lam) > 1``.
    """
    tol = COUPLED_TOL if tol is None else float(tol)
    max_iter = COUPLED_MAX_ITER if max_iter is None else int(max_iter)
    beta = float(beta)
    lam = float(control_penalty)
    check_control_penalty(beta, lam)
    root = np.sqrt(beta)
    Bs = B[:, list(treated)]
    Bc = B[:, list(control)]
    sq = None
    if lam:
        # Every term is unchanged by moving all points by the same vector, so
        # the penalised fit works around the target: A becomes 0 and the
        # squared norms in the linear term are distances from it.
        Bs, Bc, A = Bs - A[:, None], Bc - A[:, None], np.zeros_like(A)
        sq = (Bc ** 2).sum(axis=0)
        kappa = beta * lam * (1.0 + lam)
    w = (np.asarray(solve_simplex_qp(Bs, A), dtype=float) if w0 is None
         else np.asarray(w0, dtype=float))
    value, gap, v, c = _coupled_point(Bs, Bc, A, beta, w, None, lam, sq)
    newton = sweeps = 0
    for it in range(max_iter + 1):
        if cutoff is not None and value - gap > cutoff:
            return None
        if gap <= tol * (1.0 + abs(value)):
            if return_info:
                return value, w, v, {"iterations": newton + sweeps,
                                     "newton_steps": newton, "sweeps": sweeps}
            return value, w, v
        if it == max_iter:
            break
        # On the face F of the control simplex the projection of t onto the
        # face's affine hull is c0 + P (t - c0), P the projector onto the span
        # of the face's edge directions, so the residual is (I - P)(t - c0)
        # and the value on F is one least-squares fit over the treated simplex.
        face = Bc[:, v > 0]
        c0 = face.mean(axis=1)
        edges = face[:, 1:] - face[:, :1]
        U, sv, Vt = np.linalg.svd(edges, full_matrices=False)
        keep = sv > 1e-12 * max(1.0, sv.max(initial=0.0))
        span = U[:, keep]
        off_face = lambda X: X - span @ (span.T @ X)
        if lam:
            # On the face, with v = e_1 + sum_k a_k (e_k - e_1), minimising
            # ||(1 + lam) t - B_F v||^2 + lam sq'v over a leaves
            # ||(I - P)((1 + lam) t - b_1)||^2 + lam (1 + lam) q't plus a
            # constant, q = D (D'D)^+ g with D the edges and g the squared-norm
            # differences. With the -lam (1 + lam) ||t||^2 the controls' part
            # carries, the step is a fit with rows sqrt(1 - kappa) B_S and
            # sqrt(beta) (1 + lam) (I - P) B_S, and the linear term kappa B_S'q.
            sq_f = sq[v > 0]
            q = span @ ((Vt[keep] @ (sq_f[1:] - sq_f[0])) / sv[keep])
            design = np.vstack([np.sqrt(max(1.0 - kappa, 0.0)) * Bs,
                                root * (1.0 + lam) * off_face(Bs)])
            target = np.concatenate([A, root * off_face(face[:, 0])])
            w_face = np.asarray(solve_simplex_qp(
                design, target, warm_start=w, linear=kappa * (Bs.T @ q)),
                dtype=float)
        else:
            design = np.vstack([Bs, root * off_face(Bs)])
            target = np.concatenate([A, root * off_face(c0)])
            w_face = np.asarray(solve_simplex_qp(design, target, warm_start=w),
                                dtype=float)
        step, moved = 1.0, False
        for _ in range(8):
            w_try = w + step * (w_face - w)
            trial = _coupled_point(Bs, Bc, A, beta, w_try, v, lam, sq)
            if trial[0] < value - 1e-15 * (1.0 + abs(value)):
                w, (value, gap, v, c) = w_try, trial
                newton += 1
                moved = True
                break
            step *= 0.5
        if not moved:
            # Given v, the penalty is beta lam ||t - c||^2 plus a constant in t.
            pull = beta * (1.0 + lam)
            w = np.asarray(solve_simplex_qp(Bs, (A + pull * c) / (1.0 + pull),
                                            warm_start=w), dtype=float)
            value, gap, v, c = _coupled_point(Bs, Bc, A, beta, w, v, lam, sq)
            sweeps += 1
    raise CoupledFitNotCertified(
        f"The weakly-targeted fit of a candidate treated set was not certified "
        f"within {max_iter:,} steps (Frank-Wolfe gap {gap:.3g}). Its value "
        f"is an upper bound only, so the search cannot rank on it; use "
        f"backend='mip'.")


def treated_fit_bound(B: np.ndarray, A: np.ndarray, subsets: np.ndarray,
                      linear: Optional[np.ndarray] = None) -> np.ndarray:
    """A lower bound on the treated-side fit of every row of ``subsets``.

    ``min { w'G w + a'w : 1'w = 1 }`` on the centred columns, in closed form
    (see the module docstring). A block whose solve does not reproduce its
    right-hand side to ``1e-8`` relative, or whose ``1'G^-1 1`` is not positive, gets
    the bound ``0``, which never discards anything; so does a negative value,
    since the fit itself is non-negative.
    """
    subsets = np.asarray(subsets, dtype=np.int64)
    n_sub, m = subsets.shape
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    lin = None if linear is None else np.asarray(linear, dtype=float)
    out = np.zeros(n_sub)
    for lo in range(0, n_sub, CHUNK):
        idx = subsets[lo:lo + CHUNK]
        b = len(idx)
        Gs = G[idx[:, :, None], idx[:, None, :]]
        rhs = np.ones((b, m, 1 if lin is None else 2))
        if lin is not None:
            rhs[:, :, 1] = lin[idx]
        try:
            x = np.linalg.solve(Gs, rhs)
        except np.linalg.LinAlgError:
            x = np.full_like(rhs, np.nan)
            for i in range(b):
                try:
                    x[i] = np.linalg.solve(Gs[i], rhs[i])
                except np.linalg.LinAlgError:
                    pass
        resid = np.abs(Gs @ x - rhs).reshape(b, -1).max(axis=1)
        scale = np.maximum(1.0, np.abs(rhs).reshape(b, -1).max(axis=1))
        s = x[:, :, 0].sum(axis=1)
        good = (np.isfinite(s) & np.isfinite(resid) & (s > 0)
                & (resid <= 1e-8 * scale))
        val = np.zeros(b)
        if lin is None:
            val[good] = 1.0 / s[good]
        else:
            t = x[:, :, 1].sum(axis=1)
            u = np.einsum("bi,bi->b", rhs[:, :, 1], x[:, :, 1])
            val[good] = (2.0 + t[good]) ** 2 / (4.0 * s[good]) - u[good] / 4.0
        val[~np.isfinite(val)] = 0.0
        out[lo:lo + b] = np.maximum(val, 0.0)
    return out


# ----------------------------------------------------------------------
# the coupling bound
# ----------------------------------------------------------------------


def hull_separation(B: np.ndarray, Z: np.ndarray, treated_mask: np.ndarray,
                    control_mask: np.ndarray) -> np.ndarray:
    """A certified lower bound on ``dist(conv B_S, conv B_C)`` along each row of ``Z``.

    For any direction ``z``, Wolfe's (1976) Theorem 2.2 gives

        dist(conv B_S, conv B_C) >= (min_{i in S} z'b_i - max_{j in C} z'b_j) / |z|:

    every point of the first hull projects onto ``z`` at or above the smallest
    treated projection, every point of the second at or below the largest
    control projection, and a projection shortens distances. The bound reads
    the data only through ``z'b_j`` and ``|z|``, so any direction gives a valid
    bound, however it was found.

    Rounding. A product of length ``T``, summed in any order, is within
    ``gamma_T |z|'|b_j| + T eta`` of exact, with ``gamma_T = T u / (1 - T u)``,
    ``u`` the unit roundoff and ``eta`` the smallest subnormal, which covers
    gradual underflow (Higham 2002, sec. 3.1). Each projection is moved against
    the bound by eight times that error before the minimum and the maximum are
    taken. One share covers the products; the rest covers forming the error and
    the two subtractions, and leaves at least ``gamma_T`` times the numerator,
    which covers the rounding of ``|z|`` and the quotient. A row with no treated
    or no control market, a non-finite result, or a bound under the smallest
    normal number gets 0, which never discards anything.

    Parameters
    ----------
    B : np.ndarray, shape (T, n)
        The markets as columns.
    Z : np.ndarray, shape (b, T)
        One direction per row.
    treated_mask, control_mask : np.ndarray, shape (b, n), bool
        Each row's treated markets and control pool.
    """
    B = np.asarray(B, dtype=float)
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    T = B.shape[0]
    gamma = T * _UNIT_ROUNDOFF / (1.0 - T * _UNIT_ROUNDOFF)
    floor = T * _SMALLEST_SUBNORMAL
    with np.errstate(all="ignore"):
        P = Z @ B
        err = 8.0 * (gamma * (np.abs(Z) @ np.abs(B)) + floor)
        lo = np.where(treated_mask, P - err, np.inf).min(axis=1)
        hi = np.where(control_mask, P + err, -np.inf).max(axis=1)
        d = (lo - hi) / np.sqrt(np.einsum("bt,bt->b", Z, Z) + floor)
    ok = np.isfinite(d) & (d >= np.finfo(float).tiny)
    return np.where(ok, d, 0.0)


def _wolfe_lockstep(G: np.ndarray, pair_d2: np.ndarray, sets: np.ndarray,
                    control_mask: np.ndarray, need: np.ndarray, max_iter: int,
                    corral: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wolfe's nearest-point method on ``conv B_S - conv B_C``, a batch in lockstep.

    Every quantity goes through ``G``, the Gram matrix of the markets: a corral
    point is a pair ``(i, j)``, the difference ``b_i - b_j``, whose products with
    every market are ``G[i] - G[j]``. Each candidate keeps its corral points'
    market products ``R`` and their Gram ``Gc``, updated when a point is added;
    a dropped point leaves a hole, and the affine step measures from the first
    valid slot. At every major step the bound is Theorem 2.2 at the current
    point, read from those products, and a candidate leaves the batch once its
    square reaches ``need``. One candidate per numpy call would spend its time
    in call overhead, not arithmetic.

    The values here are not certified; :func:`hull_separation` certifies the
    direction behind each candidate's best one.

    Returns
    -------
    best : np.ndarray, shape (b,)
        Each candidate's best bound, uncertified.
    status : np.ndarray, shape (b,)
        One of the ``WOLFE_*`` codes.
    coef : np.ndarray, shape (b, n)
        The direction behind ``best`` as weights on the markets, ``z = B c``.
    """
    b, n = len(sets), G.shape[0]
    K = int(corral)
    rows = np.arange(b)
    M = np.zeros((b, n), dtype=bool)
    M[rows[:, None], sets] = True
    D = np.where(control_mask[:, None, :], pair_d2[sets], np.inf)
    flat = D.reshape(b, -1).argmin(axis=1)
    i0, j0 = sets[rows, flat // n], flat % n
    I = np.zeros((b, K), dtype=np.int64)
    J = np.zeros((b, K), dtype=np.int64)
    V = np.zeros((b, K), dtype=bool)
    L = np.zeros((b, K))
    R = np.zeros((b, K, n))
    Gc = np.zeros((b, K, K))
    I[:, 0], J[:, 0], V[:, 0], L[:, 0] = i0, j0, True, 1.0
    R[:, 0] = G[i0] - G[j0]
    Gc[:, 0, 0] = R[rows, 0, i0] - R[rows, 0, j0]
    major = np.ones(b, dtype=bool)
    best = np.full(b, -np.inf)
    coef = np.zeros((b, n))
    status = np.zeros(b, dtype=int)
    with np.errstate(all="ignore"):
        _wolfe_iterate(G, M, control_mask, need, int(max_iter), I, J, V, L, R,
                       Gc, major, best, coef, status)
    return best, status, coef


def _wolfe_iterate(G, M, control_mask, need, max_iter, I, J, V, L, R, Gc,
                   major, best, coef, status) -> None:
    """The iterations of :func:`_wolfe_lockstep`, updating its arrays in place.

    Non-finite data turns the arithmetic to NaN without harm: the certificate
    is what decides, and it gives such a row 0.
    """
    n = G.shape[0]
    K = I.shape[1]
    act = np.arange(len(I))
    for _ in range(max_iter):
        act = act[status[act] == 0]
        if not act.size:
            break
        # Major steps: the support point of the difference along -z, the
        # bound, and the point appended to the corral.
        mj = act[major[act]]
        if mj.size:
            Kw = int((K - np.argmax(V[mj][:, ::-1], axis=1)).max())
            Lm, Vm = L[mj, :Kw], V[mj, :Kw]
            proj = np.einsum("kq,kqn->kn", Lm, R[mj, :Kw])
            zz = np.einsum("kq,kqr,kr->k", Lm, Gc[mj, :Kw, :Kw], Lm)
            pS = np.where(M[mj], proj, np.inf)
            pC = np.where(control_mask[mj], proj, -np.inf)
            ifw, jfw = pS.argmin(axis=1), pC.argmax(axis=1)
            ar = np.arange(len(mj))
            ps, pc = pS[ar, ifw], pC[ar, jfw]
            val = (ps - pc) / np.sqrt(np.maximum(zz, 1e-300))
            up = val > best[mj]
            if up.any():
                g = mj[up]
                w = (Lm[up] * Vm[up]).ravel()
                rr = np.repeat(np.arange(g.size), Kw)
                c = np.zeros((g.size, n))
                np.add.at(c, (rr, I[g, :Kw].ravel()), w)
                np.add.at(c, (rr, J[g, :Kw].ravel()), -w)
                coef[g] = c
                best[g] = val[up]
            bb = best[mj]
            cleared = (bb > 0) & (bb * bb >= need[mj])
            conv = ~cleared & (zz - (ps - pc) <= COUPLING_TOL * zz)
            dup = ~cleared & ~conv & np.any(
                Vm & (I[mj, :Kw] == ifw[:, None]) & (J[mj, :Kw] == jfw[:, None]),
                axis=1)
            full = ~cleared & ~conv & ~dup & V[mj].all(axis=1)
            status[mj[cleared]] = WOLFE_CLEARED
            status[mj[conv | dup]] = WOLFE_CONVERGED
            status[mj[full]] = WOLFE_CORRAL_FULL
            go = ~(cleared | conv | dup | full)
            if go.any():
                g, i_, j_ = mj[go], ifw[go], jfw[go]
                s = np.argmax(~V[g], axis=1)
                I[g, s], J[g, s], V[g, s], L[g, s] = i_, j_, True, 0.0
                R[g, s] = G[i_] - G[j_]
                Rg = R[g]
                col = (np.take_along_axis(Rg, i_[:, None, None], 2)[..., 0]
                       - np.take_along_axis(Rg, j_[:, None, None], 2)[..., 0]
                       ) * V[g]
                Gc[g, :, s] = col
                Gc[g, s, :] = col
                major[g] = False
        # Affine steps: the corral's affine minimiser, by least squares on its
        # edges from a reference point, then Wolfe's minor cycle when a weight
        # turns non-positive.
        mn = act[(~major[act]) & (status[act] == 0)]
        if mn.size:
            Kw = int((K - np.argmax(V[mn][:, ::-1], axis=1)).max())
            Vm, Lm, Gm = V[mn, :Kw], L[mn, :Kw], Gc[mn, :Kw, :Kw]
            ref = np.argmax(Vm, axis=1)
            ar = np.arange(len(mn))
            gr = Gm[ar, :, ref]
            grr = Gm[ar, ref, ref]
            E = Gm - gr[:, :, None] - gr[:, None, :] + grr[:, None, None]
            r = -(gr - grr[:, None])
            edge = Vm.copy()
            edge[ar, ref] = False
            scale = np.maximum(
                np.abs(np.diagonal(E, axis1=1, axis2=2)).max(axis=1), 1.0)
            E = (np.where(edge[:, :, None] & edge[:, None, :], E, 0.0)
                 + np.eye(Kw)[None]
                 * np.where(edge, 1e-13 * scale[:, None], 1.0)[:, None, :])
            alpha = np.linalg.solve(E, np.where(edge, r, 0.0)[..., None])[..., 0]
            w = alpha * edge
            w[ar, ref] = 1.0 - w.sum(axis=1)
            ok = np.all(np.where(Vm, w > COUPLING_TOL, True), axis=1)
            Lnew = Lm.copy()
            Lnew[ok] = w[ok]
            bad = ~ok
            if bad.any():
                wb, lb_, Vb = w[bad], Lm[bad], Vm[bad]
                with np.errstate(divide="ignore", invalid="ignore"):
                    th = np.where(Vb & (wb <= COUPLING_TOL), lb_ / (lb_ - wb),
                                  np.inf)
                theta = np.clip(th.min(axis=1), 0.0, 1.0)
                lnew = lb_ + theta[:, None] * (wb - lb_)
                Vb = Vb & (lnew > COUPLING_TOL)
                lnew = np.where(Vb, lnew, 0.0)
                Lnew[bad] = lnew / lnew.sum(axis=1, keepdims=True)
                Vfull = V[mn]
                Vfull[bad, :Kw] = Vb
                V[mn] = Vfull
            Lfull = L[mn]
            Lfull[:, :Kw] = Lnew
            L[mn] = Lfull
            major[mn[ok]] = True
    status[act[status[act] == 0]] = WOLFE_OUT_OF_ITERATIONS


def _by_size(sets: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Row indices and an index array for each treated-set size in ``sets``."""
    if isinstance(sets, np.ndarray) and sets.ndim == 2:
        return [(np.arange(len(sets)), sets.astype(np.int64, copy=False))]
    groups: Dict[int, List[int]] = {}
    for r, S in enumerate(sets):
        groups.setdefault(len(S), []).append(r)
    return [(np.asarray(idx), np.asarray([tuple(sets[r]) for r in idx],
                                          dtype=np.int64))
            for idx in groups.values()]


class CouplingBound:
    """A lower bound on the weakly-targeted objective that sees the coupling.

    With Abadie and L'Hour's control penalty ``lam`` (zero without it), the
    penalty ``beta lam sum_j v_j ||t - b_j||^2`` is at least
    ``beta lam ||t - B_C v||^2`` by convexity of the square, and
    ``||t - B_C v||`` is at least the distance between the hulls, so

        f(S) >= g(S) + beta (1 + lam) dist(conv B_S, conv B_C)^2,

    where ``C`` is the candidate's own control pool, donor exclusions applied.
    The bound takes the closed-form treated bound ``lb <= g`` it is given, so
    pricing a candidate takes no simplex fit, and the distance from
    :func:`_wolfe_lockstep`, certified by :func:`hull_separation`.

    Raises
    ------
    MlsynthConfigError
        When ``beta`` or ``control_penalty`` is negative or not finite.
    """

    def __init__(self, B: np.ndarray, A: np.ndarray, beta: float,
                 control_penalty: float = 0.0,
                 donor_exclusion: Sequence[Tuple[int, int]] = ()):
        beta, lam = float(beta), float(control_penalty)
        if not np.isfinite(beta) or beta < 0:
            raise MlsynthConfigError(
                f"beta must be finite and >= 0; got {beta}.")
        if not np.isfinite(lam) or lam < 0:
            raise MlsynthConfigError(
                f"control_penalty must be finite and >= 0; got {lam}.")
        self.B = np.asarray(B, dtype=float)
        n = self.B.shape[1]
        centred = self.B - np.asarray(A, dtype=float)[:, None]
        self.gram = centred.T @ centred
        diag = np.diag(self.gram)
        self.pair_d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * self.gram,
                                  0.0)
        self.factor = beta * (1.0 + lam)
        self.excluded = np.zeros((n, n), dtype=bool)
        for i, j in donor_exclusion:
            self.excluded[int(i), int(j)] = True

    def _treated_mask(self, sets: np.ndarray) -> np.ndarray:
        mask = np.zeros((len(sets), self.B.shape[1]), dtype=bool)
        mask[np.arange(len(sets))[:, None], sets] = True
        return mask

    def control_mask(self, sets: np.ndarray) -> np.ndarray:
        """Each row's control pool: the cluster less the set and its exclusions."""
        sets = np.asarray(sets, dtype=np.int64)
        return ~self._treated_mask(sets) & ~self.excluded[sets].any(axis=1)

    def distance(self, sets: Any, need: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """A certified lower bound on each row's hull distance.

        Wolfe's method stops on a row once its bound's square reaches
        ``need`` (by default it runs to convergence). A row with no control
        market gets 0.
        """
        b = len(sets)
        need = (np.full(b, np.inf) if need is None
                else np.broadcast_to(np.asarray(need, dtype=float), (b,)))
        out = np.zeros(b)
        for idx, arr in _by_size(sets):
            cmask = self.control_mask(arr)
            live = np.flatnonzero(cmask.any(axis=1))
            for lo in range(0, live.size, SCREEN_BATCH):
                part = live[lo:lo + SCREEN_BATCH]
                _, _, coef = _wolfe_lockstep(
                    self.gram, self.pair_d2, arr[part], cmask[part],
                    need[idx[part]], COUPLING_MAX_ITER, COUPLING_CORRAL)
                out[idx[part]] = hull_separation(
                    self.B, coef @ self.B.T, self._treated_mask(arr[part]),
                    cmask[part])
        return out

    def bound(self, sets: Any, lb: np.ndarray,
              cutoff: Optional[float] = None) -> np.ndarray:
        """``lb + beta (1 + lam) d^2``, with Wolfe aimed at ``cutoff`` if given."""
        lb = np.asarray(lb, dtype=float)
        if self.factor == 0.0:
            return lb.copy()
        need = (None if cutoff is None
                else np.maximum((cutoff - lb) / self.factor, 0.0))
        d = self.distance(sets, need)
        return lb + self.factor * d * d

    def survivors(self, sets: Any, lb: np.ndarray, cutoff: float) -> np.ndarray:
        """Which rows the bound cannot discard against ``cutoff``."""
        return self.bound(sets, lb, cutoff) < cutoff


# ----------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------


def _space(n: int, size: int, filt: SubsetFilter) -> SearchSpace:
    return SearchSpace.build(n, size, forced=filt.forced_in,
                             forbidden=filt.forbidden, strata=filt.strata)


def count_candidates(n: int, sizes: Sequence[int], filt: SubsetFilter) -> int:
    """Admissible treated sets over ``sizes``, before conflict pairs remove any."""
    return sum(_space(n, s, filt).size for s in sizes)


def _chunks(n: int, sizes: Sequence[int], filt: SubsetFilter
            ) -> Iterator[np.ndarray]:
    """Admissible treated sets as ``(b, size)`` arrays, ``b <= CHUNK``.

    The walk already honours forced units, forbidden units and disjoint strata.
    Conflict pairs, and strata the walk could not resolve into groups, are
    tested on each finished candidate; with none of them the candidates go
    straight into an array without a per-candidate call.
    """
    check = bool(filt.conflict_pairs or filt.strata or filt.forbidden_sets)
    for size in sizes:
        space = _space(n, size, filt)
        if not space.size:
            continue
        walk = space.iter_subsets()
        if not check:
            while True:
                flat = np.fromiter(
                    itertools.chain.from_iterable(itertools.islice(walk, CHUNK)),
                    dtype=np.int64)
                if not flat.size:
                    break
                yield flat.reshape(-1, size)
            continue
        block: List[Tuple[int, ...]] = []
        for subset in walk:
            if not filt.accepts(subset):
                continue
            block.append(subset)
            if len(block) == CHUNK:
                yield np.asarray(block, dtype=np.int64)
                block = []
        if block:
            yield np.asarray(block, dtype=np.int64)


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


class _TopK:
    """The ``k`` best designs so far, and the cutoff they set."""

    def __init__(self, k: int):
        self.k = int(k)
        self.items: List[ClusterDesign] = []

    @property
    def cutoff(self) -> float:
        return self.items[-1].value if len(self.items) == self.k else np.inf

    def offer(self, design: ClusterDesign) -> None:
        if design.value >= self.cutoff:
            return
        self.items.append(design)
        self.items.sort(key=lambda d: (d.value, d.treated))
        del self.items[self.k:]


def search_cluster(
    B: np.ndarray,
    A: np.ndarray,
    sizes: Sequence[int],
    *,
    filt: SubsetFilter,
    linear_treated: Optional[np.ndarray] = None,
    linear_control: Optional[np.ndarray] = None,
    donor_exclusion: Sequence[Tuple[int, int]] = (),
    top_k: int = 1,
    candidate_limit: Optional[int] = None,
    beta: Optional[float] = None,
    control_penalty: float = 0.0,
) -> List[ClusterDesign]:
    """The ``top_k`` best designs of one cluster, best first.

    Parameters
    ----------
    B : np.ndarray, shape (T, n)
        The cluster's markets as columns, over the fit window.
    A : np.ndarray, shape (T,)
        The cluster's target.
    sizes : sequence of int
        Treated-set sizes to search.
    filt : SubsetFilter
        Forced, forbidden, conflict and stratum rules, in local indices.
    linear_treated, linear_control : np.ndarray, optional
        Non-negative penalty vectors over the cluster's markets.
    donor_exclusion : sequence of (i, j)
        ``j`` may not be a control when ``i`` is treated.
    top_k : int
        How many designs to return.
    candidate_limit : int, optional
        Refuse above this many candidates; defaults to ``CANDIDATE_LIMIT``.
    beta : float, optional
        The weakly-targeted coupling. When given, each candidate is priced by
        :func:`coupled_fit` instead of two independent fits, once
        :class:`CouplingBound` has failed to discard it; it takes no penalty
        vectors.
    control_penalty : float
        Abadie and L'Hour's control penalty inside the weakly-targeted design
        (Vives-i-Bastida 2022, program 1); needs ``beta``. It is non-negative,
        so the treated-side bound still sits under the objective, and it
        scales the coupling bound by ``1 + control_penalty``.

    Raises
    ------
    MlsynthConfigError
        Above the candidate limit, for a negative ``beta``, for ``beta`` with a
        penalty vector, or for a ``control_penalty`` without ``beta`` or past
        the convexity bound.
    MlsynthEstimationError
        When no admissible design exists.
    """
    n = B.shape[1]
    if control_penalty and beta is None:
        raise MlsynthConfigError(
            "control_penalty charges each control its distance from the "
            "synthetic treated unit inside the weakly-targeted design; it "
            "needs beta.")
    if beta is not None:
        check_control_penalty(beta, control_penalty)
        if beta < 0:
            raise MlsynthConfigError(
                f"beta must be >= 0; a negative beta makes the coupling term "
                f"concave and the design has no minimum to certify. Got {beta}.")
        if linear_treated is not None or linear_control is not None:
            raise MlsynthConfigError(
                "beta prices the weakly-targeted design, which has no linear "
                "penalty; a search given both would optimise an objective "
                "MAREX does not state.")
    limit = CANDIDATE_LIMIT if candidate_limit is None else int(candidate_limit)
    total = count_candidates(n, sizes, filt)
    if total > limit:
        raise MlsynthConfigError(
            f"The exact backend would enumerate {total:,} treated sets in one "
            f"cluster, above its limit of {limit:,}. Use backend='mip', or "
            f"narrow the design with restrictions or a smaller treated count.")

    excluded: Dict[int, set] = {}
    for i, j in donor_exclusion:
        excluded.setdefault(int(i), set()).add(int(j))
    lin_t = None if linear_treated is None else np.asarray(linear_treated, float)
    lin_c = None if linear_control is None else np.asarray(linear_control, float)

    best = _TopK(top_k)
    seen: set = set()
    screen = (CouplingBound(B, A, beta, control_penalty, donor_exclusion)
              if beta is not None and beta * (1.0 + control_penalty) > 0
              else None)

    def margin(cut: float) -> float:
        return PRUNE_MARGIN * (1.0 + abs(cut))

    def consider(S: Tuple[int, ...]) -> None:
        seen.add(S)
        cols = list(S)
        g, w = side_fit(B, A, cols, None if lin_t is None else lin_t[cols])
        if g >= best.cutoff:
            return
        drop = set(S)
        for i in S:
            drop |= excluded.get(i, set())
        control = [j for j in range(n) if j not in drop]
        if not control:
            return
        if beta is None:
            h, v = side_fit(B, A, control,
                            None if lin_c is None else lin_c[control])
            value = g + h
        else:
            cut = best.cutoff
            fit = coupled_fit(B, A, cols, control, beta, w0=w,
                              cutoff=cut + margin(cut) if np.isfinite(cut) else None,
                              control_penalty=control_penalty)
            if fit is None:
                return
            value, w, v = fit
        best.offer(ClusterDesign(value=value, treated=tuple(S),
                                 control=tuple(control), w=w, v=v))

    def walk(cands: List[Tuple[int, ...]], lbs: np.ndarray) -> None:
        """Consider candidates in bound order, stopping where the bound clears
        the incumbent. With the coupling bound, each batch of candidates still
        under the incumbent is screened against it first; while there is no
        incumbent, SCREEN_SEED at a time go unscreened."""
        start, total = 0, len(cands)
        while start < total:
            cut = best.cutoff
            if lbs[start] >= cut + margin(cut):
                return
            keep = None
            if screen is None:
                stop = total
            elif not np.isfinite(cut):
                stop = min(start + SCREEN_SEED, total)
            else:
                stop = min(start + SCREEN_BATCH, total)
                stop = start + int(np.searchsorted(lbs[start:stop],
                                                   cut + margin(cut)))
                keep = screen.survivors(cands[start:stop], lbs[start:stop],
                                        cut + margin(cut))
            for k in range(start, stop):
                if keep is not None and not keep[k - start]:
                    continue
                cut = best.cutoff
                if lbs[k] >= cut + margin(cut):
                    return
                consider(cands[k])
            start = stop

    # Pass 1: price every candidate, keep the BUFFER best-bounded.
    buf_lb = np.empty(0)
    buf_sets: List[Tuple[int, ...]] = []
    for block in _chunks(n, sizes, filt):
        lb = treated_fit_bound(B, A, block, lin_t)
        held = buf_lb.size
        all_lb = np.concatenate([buf_lb, lb])
        if all_lb.size > BUFFER:
            part = np.argpartition(all_lb, BUFFER - 1)[:BUFFER]
        else:
            part = np.arange(all_lb.size)
        buf_sets = [buf_sets[i] if i < held
                    else tuple(int(j) for j in block[i - held]) for i in part]
        buf_lb = all_lb[part]
    if not buf_sets:
        raise MlsynthEstimationError(
            "No admissible treated set exists in this cluster under the "
            "cardinality and the active restrictions.")
    order = np.argsort(buf_lb, kind="stable")
    walk([buf_sets[i] for i in order], buf_lb[order])

    # Pass 2, only when the buffer cannot certify the result: it was full, and
    # either fewer than top_k designs were found or its largest bound is still
    # under the cutoff, so a candidate outside it could still enter.
    cut = best.cutoff
    if len(buf_sets) == BUFFER and (
            not np.isfinite(cut) or buf_lb.max() < cut + margin(cut)):
        for block in _chunks(n, sizes, filt):
            lb = treated_fit_bound(B, A, block, lin_t)
            cut = best.cutoff
            cands: List[Tuple[int, ...]] = []
            lbs: List[float] = []
            for i in np.argsort(lb, kind="stable"):
                if lb[i] >= cut + margin(cut):
                    break
                S = tuple(int(j) for j in block[i])
                if S not in seen:
                    cands.append(S)
                    lbs.append(float(lb[i]))
            walk(cands, np.asarray(lbs))

    if not best.items:
        raise MlsynthEstimationError(
            "No admissible design exists in this cluster: every treated set "
            "the restrictions allow leaves no control market.")
    return list(best.items)


# ----------------------------------------------------------------------
# MAREX's program, solved by the search
# ----------------------------------------------------------------------


def _local_rules(restrictions: Any, members: np.ndarray, cluster_of: np.ndarray
                 ) -> Tuple[SubsetFilter, Tuple[Tuple[int, int], ...]]:
    """One cluster's restrictions in local indices, and its donor exclusions."""
    pos = {int(g): i for i, g in enumerate(members)}
    if restrictions is None or restrictions.is_empty:
        return SubsetFilter(), ()
    forced = frozenset(pos[j] for j in restrictions.forced_in if j in pos)
    forbidden = frozenset(pos[j] for j in restrictions.forbidden if j in pos)
    conflicts = tuple((pos[i], pos[j]) for i, j in restrictions.conflict_pairs
                      if i in pos and j in pos)
    strata = tuple((tuple(pos[s] for s in mem if s in pos), lo, hi)
                   for mem, lo, hi in restrictions.strata
                   if any(s in pos for s in mem))
    donors = tuple((pos[i], pos[j]) for i, j in restrictions.donor_exclusion
                   if i in pos and j in pos)
    filt = SubsetFilter(forced_in=forced, forbidden=forbidden,
                        conflict_pairs=conflicts, strata=strata)
    return filt, donors


def _refuse_spanning_rules(restrictions: Any, cluster_of: np.ndarray) -> None:
    """Clusters are searched apart, so no rule may tie two of them together."""
    if restrictions is None or restrictions.is_empty:
        return
    for i, j in restrictions.conflict_pairs:
        if cluster_of[i] != cluster_of[j]:
            raise MlsynthConfigError(
                "A border conflict links markets in different clusters. The "
                "exact backend searches each cluster on its own, so it cannot "
                "honour a rule that spans two of them; use backend='mip'.")
    for members, lo, hi in restrictions.strata:
        if len({int(cluster_of[s]) for s in members}) > 1 and (
                lo is not None or hi is not None):
            raise MlsynthConfigError(
                "A stratum quota counts markets in more than one cluster. The "
                "exact backend searches each cluster on its own, so it cannot "
                "honour a quota that spans two of them; use backend='mip'.")


def _sizes(n: int, lab: Any, m_eq, m_min, m_max) -> List[int]:
    me = get_per_cluster_param(m_eq, lab)
    if me is not None:
        return [int(me)] if 1 <= int(me) <= n - 1 else []
    mn = get_per_cluster_param(m_min, lab)
    mx = get_per_cluster_param(m_max, lab)
    lo = max(1, int(mn) if mn is not None else 1)
    hi = min(n - 1, int(mx) if mx is not None else n - 1)
    return list(range(lo, hi + 1))


def _penalties(design: str, k_idx: int, members: np.ndarray, X_fit: np.ndarray,
               D1: np.ndarray, lambda1: float, lambda2: float, xi: float,
               lambda1_unit: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The linear terms ``build_objective`` adds, restricted to one cluster."""
    if design == "penalized":
        a = lambda1 * D1[members, k_idx] if lambda1 > 0 else None
        b = lambda2 * D1[members, k_idx] if lambda2 > 0 else None
        return a, b
    if design == "unit_penalized":
        a = np.zeros(len(members))
        if xi > 0:
            a += xi * per_unit_reproducibility(X_fit, members)[members]
        if lambda1_unit > 0:
            a += lambda1_unit * D1[members, k_idx]
        return (a if (xi > 0 or lambda1_unit > 0) else None), None
    return None, None


def _k_best_combinations(lists: List[List[ClusterDesign]], k: int
                         ) -> List[Tuple[float, Tuple[int, ...]]]:
    """The ``k`` smallest sums choosing one design per cluster, best first."""
    start = tuple(0 for _ in lists)
    heap = [(sum(lst[0].value for lst in lists), start)]
    seen = {start}
    out = []
    while heap and len(out) < k:
        total, pick = heapq.heappop(heap)
        out.append((total, pick))
        for c, lst in enumerate(lists):
            if pick[c] + 1 < len(lst):
                nxt = pick[:c] + (pick[c] + 1,) + pick[c + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (total - lst[pick[c]].value
                                          + lst[pick[c] + 1].value, nxt))
    return out


def _designs(
    Y_full, T0, clusters, blank_periods=0, m_eq=None, m_min=None, m_max=None,
    exclusive=True, design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0,
    xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0, costs=None, budget=None,
    covariates=None, covariate_weight=1.0, standardize=False,
    restrictions=None, top_k=1, control_penalty=0.0, certify=False,
) -> List[Dict[str, Any]]:
    if design not in EXACT_DESIGNS:
        raise MlsynthConfigError(
            f"The exact backend solves the designs "
            f"{', '.join(sorted(EXACT_DESIGNS))}; got design={design!r}.")
    if costs is not None or budget is not None:
        raise MlsynthConfigError(
            "The exact backend does not take a cost budget, which constrains "
            "the treated weights; use backend='mip'.")
    if design == "unit_penalized" and lambda2_unit > 0:
        raise MlsynthConfigError(
            "lambda2_unit pairs the treated weights against the control weights "
            "and leaves no convex program to solve; use lambda1_unit.")
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit,
                        control_penalty=control_penalty)
    Y_full_np, clusters, N, cluster_labels, K, label_to_k = prepare_clusters(
        Y_full, clusters)
    Y_fit, Y_blank, T_fit = prepare_fit_slices(Y_full_np, T0, blank_periods)
    X_fit = _augment_fit(Y_fit, covariates, covariate_weight, standardize)
    M = build_membership_mask(clusters, label_to_k, N, K)
    Xbar_clusters, cluster_members = compute_cluster_means_members(
        X_fit, M, cluster_labels)
    D1, _ = precompute_distances(X_fit, Xbar_clusters, cluster_members)
    cluster_of = np.array([int(np.argmax(M[j])) for j in range(N)])
    _refuse_spanning_rules(restrictions, cluster_of)

    per_cluster: List[List[ClusterDesign]] = []
    certificates: Dict[str, Any] = {}
    for k_idx, lab in enumerate(cluster_labels):
        members = np.asarray(cluster_members[k_idx])
        sizes = _sizes(len(members), lab, m_eq, m_min, m_max)
        if not sizes:
            raise MlsynthEstimationError(
                f"MAREX design is infeasible in cluster {lab!r}: the treated "
                f"count leaves no control market (m_eq={m_eq}, m_min={m_min}, "
                f"m_max={m_max}, cluster size {len(members)}).")
        filt, donors = _local_rules(restrictions, members, cluster_of)
        a, b = _penalties(design, k_idx, members, X_fit, D1, lambda1, lambda2,
                          xi, lambda1_unit)
        B_k, A_k = X_fit[members].T, Xbar_clusters[k_idx]
        cert = None
        try:
            if (design in SEPARABLE_DESIGNS and
                    count_candidates(len(members), sizes, filt) > CANDIDATE_LIMIT):
                from .branch import branch_and_bound
                found = branch_and_bound(
                    B_k, A_k, sizes, filt=filt, linear_treated=a,
                    linear_control=b, donor_exclusion=donors, top_k=top_k)
                designs, cert = found.designs, found.certificate
            else:
                designs = search_cluster(
                    B_k, A_k, sizes, filt=filt,
                    linear_treated=a, linear_control=b, donor_exclusion=donors,
                    top_k=top_k,
                    beta=beta if design == "weakly_targeted" else None,
                    control_penalty=control_penalty)
        except (CoupledFitNotCertified, SearchNotClosed):
            raise
        except MlsynthEstimationError as exc:
            raise MlsynthEstimationError(
                f"MAREX design is infeasible under the active restrictions "
                f"(forced/forbidden units, border conflicts, size band) and the "
                f"cluster cardinality (m_min/m_max) in cluster {lab!r}: {exc}"
            ) from exc
        per_cluster.append(designs)
        if certify:
            if cert is None:
                from .certify import certify_cluster
                cert = certify_cluster(
                    B_k, A_k, sizes, filt, designs[0],
                    linear_treated=a, linear_control=b, donor_exclusion=donors)
            certificates[str(lab)] = dataclasses.replace(cert, unresolved=tuple(
                tuple(int(members[j]) for j in S) for S in cert.unresolved))

    raws = []
    for total, pick in _k_best_combinations(per_cluster, top_k):
        w_opt = np.zeros((N, K))
        v_opt = np.zeros((N, K))
        z_opt = np.zeros((N, K))
        for k_idx, choice in enumerate(pick):
            d = per_cluster[k_idx][choice]
            members = np.asarray(cluster_members[k_idx])
            treated = members[list(d.treated)]
            control = members[list(d.control)]
            w_opt[treated, k_idx] = d.w
            v_opt[control, k_idx] = d.v
            z_opt[treated, k_idx] = 1.0
        raws.append(finish_design(
            Y_full, Y_full_np, w_opt, v_opt, z_opt, Xbar_clusters,
            cluster_labels, cluster_members, N, K, T0, blank_periods, T_fit,
            Y_fit, Y_blank, design, clusters, float(total)))
    if certify:
        raws[0]["certificate"] = certificates
    return raws


def solve_design_exact(Y_full, T0, clusters, **kwargs) -> Dict[str, Any]:
    """MAREX's design by bounded search; the same dict as ``solve_design``.

    Takes ``solve_design``'s arguments less the branch-and-bound knobs
    (``solver``, ``verbose``, ``warm_start``, ``time_limit``, ``forbidden``).
    With ``certify=True`` the dict also holds ``"certificate"``, one
    :class:`~mlsynth.utils.marex_helpers.certify.ClusterCertificate` per
    cluster label, its unresolved sets in global unit indices.
    """
    return _designs(Y_full, T0, clusters, top_k=1, **kwargs)[0]


def solve_design_exact_pool(Y_full, T0, clusters, *, top_K=1, **kwargs
                            ) -> List[Dict[str, Any]]:
    """The ``top_K`` best distinct designs, best first; ``solve_design_pool``'s output.

    The MIQP pool re-solves with a no-good cut per design already found. Here
    each cluster's search keeps its ``top_K`` best treated sets, and the global
    designs are the ``top_K`` smallest sums of one choice per cluster.
    """
    return _designs(Y_full, T0, clusters, top_k=int(top_K), **kwargs)

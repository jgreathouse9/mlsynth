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

Pricing. Without restrictions candidates arrive in lexicographic order, so
consecutive ones share all but their last market. For ``S = P + {j}``, with ``H = G_P^-1``, ``g = G[P, j]``,
``z = H g`` and the Schur complement ``sigma = G_jj - g'z``, the Banachiewicz
inversion formula (Zhang 2005, (0.7.2)) gives
``e'G_S^-1 f = e_P'H f_P + (e_j - z'e_P)(f_j - z'f_P) / sigma``, so ``s``,
``t`` and ``u`` extend from the prefix in ``O(m^2)``, and each prefix's inverse is
formed once for its run (:func:`bordered_terms`).

An update need not be as accurate as a fresh solve, so the bound does not rest
on it. For any vector ``v``, ``||B~_S (w - v)||^2 >= 0`` gives
``g(S) >= -v'G_S v + min_i (2 (G_S v)_i + a_i)``, with equality at the affine
minimiser, where it is the closed form (Neumaier and Shcherbina 2004, sec. 3).
The search evaluates this at the minimiser bordering returns and subtracts an a
priori bound on the rounding of the evaluation and of forming ``G`` (Higham
2002, (3.5) and (3.13)), so an inaccurate update or an ill-conditioned ``G_S``
weakens a bound and never makes it exceed the fit (:func:`safe_treated_bound`).

The safe evaluation gathers ``G_S`` for every candidate, which costs about what
bordering saves, so the search chooses by size. Up to
``RIGOROUS_PRICING_LIMIT`` candidates in a cluster every bound is the safe form.
Above it a candidate takes the closed form straight from the update when its
conditioning is at most ``COND_GUARD``, and only the rest are evaluated in the
safe form; the admitted closed forms' error is covered by ``PRUNE_MARGIN``, as a
fresh solve's was.

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
objective and prune before any fit; at large ``beta`` they prune little,
because the coupling dominates the objective and they cannot see it, and the
cutoff is what keeps each surviving candidate to about one step.

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
#: relative. The bound accounts for its own rounding; the margin covers the
#: rounding in the fitted values that set the cutoff.
PRUNE_MARGIN = 1e-6

#: Unit roundoff of IEEE double precision.
UNIT_ROUNDOFF = 2.0 ** -53

#: On the fast path a candidate's closed form is taken from the update only when
#: its conditioning ``kappa`` (the prefix's condition estimate, and
#: ``G_jj / sigma``) is at most this. Against exact arithmetic on panels with
#: planted near-duplicate markets, the rows it admitted overestimated the
#: closed form by at most 7.9e-10 relative, three orders under PRUNE_MARGIN;
#: rows past it, by up to 47 times. On GeoLift every candidate is admitted.
COND_GUARD = 1e6

#: Up to this many candidates in a cluster every bound is the safe form; above
#: it the search prices on the fast path. On GeoLift the safe form costs 0.3 to
#: 0.5 microseconds more per candidate than the fast path, a third to half a
#: second per million, and about what a fresh solve per candidate costs.
RIGOROUS_PRICING_LIMIT = 1_000_000

#: The smallest positive subnormal double. A product that underflows is rounded
#: with an absolute error of up to half of it, which no multiple of the unit
#: roundoff covers.
SMALLEST_SUBNORMAL = 2.0 ** -1074


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


def centred_gram(B: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``G = B~'B~`` on the columns measured from the target, and their norms."""
    Bc = B - A[:, None]
    return Bc.T @ Bc, np.sqrt(np.einsum("ti,ti->i", Bc, Bc))


def prefix_runs(subsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The prefix of each run of consecutive rows that share one, and each row's run.

    A row's prefix is all of it but its last market. Lexicographic candidates
    share a prefix in long runs, so each prefix's inverse is formed once; rows in
    any other order are still grouped correctly, into shorter runs.
    """
    pre = subsets[:, :-1]
    new = np.ones(len(subsets), dtype=bool)
    new[1:] = np.any(pre[1:] != pre[:-1], axis=1)
    return pre[new], np.cumsum(new) - 1


def _inverse(M: np.ndarray) -> np.ndarray:
    """A batch of inverses; a singular matrix gets NaN instead of stopping the batch."""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        out = np.full_like(M, np.nan)
        for k in range(len(M)):
            try:
                out[k] = np.linalg.inv(M[k])
            except np.linalg.LinAlgError:
                pass
        return out


def bordered_terms(G: np.ndarray, subsets: np.ndarray,
                   linear: Optional[np.ndarray] = None):
    """``s``, ``t``, ``x = G_S^-1 1`` and ``y = G_S^-1 a`` for every row.

    Each row ``S = P + {j}`` is extended from its prefix's inverse ``H``: with
    ``g = G[P, j]``, ``z = H g`` and the Schur complement
    ``sigma = G_jj - g'z``, the Banachiewicz inversion formula gives
    ``e'G_S^-1 f = e_P'H f_P + (e_j - z'e_P)(f_j - z'f_P) / sigma``, which is
    ``O(m^2)`` per row. ``t`` and ``y`` are ``None`` without ``linear``.
    A row whose prefix has no inverse, or whose ``sigma`` is not positive, gets
    non-finite terms.
    """
    return _bordered(G, subsets, linear)[:4]


def _bordered(G: np.ndarray, subsets: np.ndarray,
              linear: Optional[np.ndarray] = None):
    """:func:`bordered_terms`, and each row's conditioning ``kappa``.

    ``kappa`` is the larger of the prefix's condition estimate
    ``||G_P||_1 ||H||_1`` and ``G_jj / sigma``. Together they track the
    condition of ``G_S``, which sets the update's error; ``kappa`` is NaN
    where the terms are, so no comparison can admit such a row.
    """
    subsets = np.asarray(subsets, dtype=np.int64)
    lin = None if linear is None else np.asarray(linear, dtype=float)
    j = subsets[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if subsets.shape[1] == 1:
            gjj = G[j, j]
            x = (1.0 / np.where(gjj > 0, gjj, np.nan))[:, None]
            s = x[:, 0]
            kappa = np.where(gjj > 0, 1.0, np.inf)
            if lin is None:
                return s, None, x, None, kappa
            y = lin[j][:, None] * x
            return s, lin[j] * s, x, y, kappa
        P, run = prefix_runs(subsets)
        GP = G[P[:, :, None], P[:, None, :]]
        H = _inverse(GP)
        kappa_P = np.abs(GP).sum(axis=1).max(axis=1) * np.abs(H).sum(axis=1).max(axis=1)
        xP = H.sum(axis=2)
        g = G[subsets[:, :-1], j[:, None]]
        z = np.einsum("lij,lj->li", H[run], g)
        sigma = G[j, j] - np.einsum("li,li->l", g, z)
        sigma = np.where(sigma > 0, sigma, np.nan)
        kappa = np.maximum(kappa_P[run], G[j, j] / sigma)
        xl = xP[run]
        c1 = (1.0 - np.einsum("li,li->l", g, xl)) / sigma
        s = xP.sum(axis=1)[run] + c1 * c1 * sigma
        x = np.concatenate([xl - z * c1[:, None], c1[:, None]], axis=1)
        if lin is None:
            return s, None, x, None, kappa
        aP = lin[P]
        yP = np.einsum("kij,kj->ki", H, aP)
        ca = (lin[j] - np.einsum("li,li->l", z, lin[subsets[:, :-1]])) / sigma
        t = np.einsum("ki,ki->k", xP, aP)[run] + c1 * ca * sigma
        y = np.concatenate([yP[run] - z * ca[:, None], ca[:, None]], axis=1)
    return s, t, x, y, kappa


def affine_minimiser(s, t, x, y) -> np.ndarray:
    """The minimiser of ``w'G_S w + a'w`` subject to ``1'w = 1`` alone, per row.

    ``x / s`` without a penalty; with one, ``(mu x - y) / 2`` with
    ``mu = (2 + t) / s``, from the stationarity condition ``2 G_S w + a = mu 1``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if t is None:
            return x / s[:, None]
        mu = (2.0 + t) / s
        return (mu[:, None] * x - y) / 2.0


def safe_rounding(d: np.ndarray, T: int, subsets: np.ndarray, v: np.ndarray,
                  linear: Optional[np.ndarray] = None) -> np.ndarray:
    """An a priori bound on the rounding of :func:`safe_treated_bound`'s value.

    Forming ``G`` from the data and evaluating the form are inner products of
    length at most ``k = T + 2m + 4``, so Higham's (3.5) and (3.13) bound the
    error by ``gamma_k`` times the products of absolute values, and
    Cauchy-Schwarz bounds those by the column norms ``d``:
    ``|v|'|B~_S|'|B~_S||v| <= (|v|'d_S)^2`` and each row of
    ``|B~_S|'|B~_S||v|`` by ``max d_S |v|'d_S``. That model holds without
    underflow. A product in the subnormal range is rounded with an absolute
    error of up to half the smallest subnormal ``eta`` (Higham 2002, (2.8));
    carried through the same inner products, those errors total at most
    ``k eta (1 + sum |v|)^2``. Below a data scale of about ``1e-153`` the
    products that form ``G`` are subnormal and this term is the one that
    holds. The factor 2 covers the rounding of ``d`` and of this bound itself.
    """
    m = subsets.shape[1]
    k = T + 2 * m + 4
    gamma = k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)
    dS = d[subsets]
    dv = np.einsum("li,li->l", np.abs(v), dS)
    total = dv * dv + 2.0 * dS.max(axis=1) * dv
    if linear is not None:
        total = total + np.abs(np.asarray(linear, dtype=float)[subsets]).max(axis=1)
    spread = 1.0 + np.abs(v).sum(axis=1)
    return 2.0 * (gamma * total + k * SMALLEST_SUBNORMAL * spread * spread)


def safe_treated_bound(G: np.ndarray, d: np.ndarray, T: int, subsets: np.ndarray,
                       v: np.ndarray, linear: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """``-v'G_S v + min_i (2 (G_S v)_i + a_i)``, less its rounding, for any ``v``.

    ``||B~_S (w - v)||^2 >= 0`` gives ``w'G_S w >= 2 v'G_S w - v'G_S v``, and the
    minimum of a linear function over the simplex is its smallest coefficient,
    so the form is at most the treated fit at every ``v`` (Neumaier and
    Shcherbina 2004, sec. 3; it is Fenchel's inequality). Less
    :func:`safe_rounding`, the value computed in floating point is too.
    """
    GS = G[subsets[:, :, None], subsets[:, None, :]]
    Gv = np.einsum("lij,lj->li", GS, v)
    lin = 0.0 if linear is None else np.asarray(linear, dtype=float)[subsets]
    val = -np.einsum("li,li->l", v, Gv) + (2.0 * Gv + lin).min(axis=1)
    return val - safe_rounding(d, T, subsets, v, linear)


def treated_fit_bound(B: np.ndarray, A: np.ndarray, subsets: np.ndarray,
                      linear: Optional[np.ndarray] = None, *,
                      rigorous: bool = True) -> np.ndarray:
    """A lower bound on the treated-side fit of every row of ``subsets``.

    ``min { w'G w + a'w : 1'w = 1 }`` on the centred columns (see the module
    docstring), its terms extended from each row's prefix by
    :func:`bordered_terms`.

    With ``rigorous`` every row is evaluated in the safe form at the affine
    minimiser, less its rounding (:func:`safe_treated_bound`). The safe form is
    a lower bound at any point, so an inaccurate update weakens a bound and
    never makes it invalid. Without it, a row whose conditioning ``kappa`` is at
    most ``COND_GUARD`` takes the closed form ``(2 + t)^2 / (4 s) - u / 4``
    straight from the update, which costs no gather of ``G_S``, and only the
    other rows are evaluated in the safe form; the closed form's error is then
    covered by ``PRUNE_MARGIN``, as a fresh solve's was. A row whose update
    breaks down, where a prefix has no inverse or a Schur complement is not
    positive, gets ``0``, which discards nothing; so does a negative value,
    since the fit itself is non-negative.
    """
    subsets = np.asarray(subsets, dtype=np.int64)
    n_sub = len(subsets)
    G, d = centred_gram(B, A)
    T = B.shape[0]
    lin = None if linear is None else np.asarray(linear, dtype=float)
    out = np.zeros(n_sub)
    for lo in range(0, n_sub, CHUNK):
        idx = subsets[lo:lo + CHUNK]
        s, t, x, y, kappa = _bordered(G, idx, lin)
        trust = np.zeros(len(idx), dtype=bool)
        if not rigorous:
            trust = (s > 0) & (kappa <= COND_GUARD)
        val = np.zeros(len(idx))
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if trust.any():
                st = s[trust]
                if lin is None:
                    val[trust] = 1.0 / st
                else:
                    u = np.einsum("li,li->l", lin[idx[trust]], y[trust])
                    val[trust] = (2.0 + t[trust]) ** 2 / (4.0 * st) - u / 4.0
            rest = ~trust
            if rest.any():
                v = affine_minimiser(s[rest], None if t is None else t[rest],
                                     x[rest], None if y is None else y[rest])
                val[rest] = safe_treated_bound(G, d, T, idx[rest], v, lin)
        val[~np.isfinite(val)] = 0.0
        out[lo:lo + len(idx)] = np.maximum(val, 0.0)
    return out


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
        :func:`coupled_fit` instead of two independent fits; it takes no
        penalty vectors.
    control_penalty : float
        Abadie and L'Hour's control penalty inside the weakly-targeted design
        (Vives-i-Bastida 2022, program 1); needs ``beta``. It is non-negative,
        so the treated-side bound still sits under the objective.

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
    rigorous = total <= RIGOROUS_PRICING_LIMIT
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

    # Pass 1: price every candidate, keep the BUFFER best-bounded.
    buf_lb = np.empty(0)
    buf_sets: List[Tuple[int, ...]] = []
    for block in _chunks(n, sizes, filt):
        lb = treated_fit_bound(B, A, block, lin_t, rigorous=rigorous)
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
    for i in np.argsort(buf_lb, kind="stable"):
        cut = best.cutoff
        if buf_lb[i] >= cut + margin(cut):
            break
        consider(buf_sets[i])

    # Pass 2, only when the buffer cannot certify the result: it was full, and
    # either fewer than top_k designs were found or its largest bound is still
    # under the cutoff, so a candidate outside it could still enter.
    cut = best.cutoff
    if len(buf_sets) == BUFFER and (
            not np.isfinite(cut) or buf_lb.max() < cut + margin(cut)):
        for block in _chunks(n, sizes, filt):
            lb = treated_fit_bound(B, A, block, lin_t, rigorous=rigorous)
            for i in np.argsort(lb, kind="stable"):
                cut = best.cutoff
                if lb[i] >= cut + margin(cut):
                    break
                S = tuple(int(j) for j in block[i])
                if S not in seen:
                    consider(S)

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
    restrictions=None, top_k=1, control_penalty=0.0,
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
        try:
            designs = search_cluster(
                X_fit[members].T, Xbar_clusters[k_idx], sizes, filt=filt,
                linear_treated=a, linear_control=b, donor_exclusion=donors,
                top_k=top_k,
                beta=beta if design == "weakly_targeted" else None,
                control_penalty=control_penalty)
        except CoupledFitNotCertified:
            raise
        except MlsynthEstimationError as exc:
            raise MlsynthEstimationError(
                f"MAREX design is infeasible under the active restrictions "
                f"(forced/forbidden units, border conflicts, size band) and the "
                f"cluster cardinality (m_min/m_max) in cluster {lab!r}: {exc}"
            ) from exc
        per_cluster.append(designs)

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
    return raws


def solve_design_exact(Y_full, T0, clusters, **kwargs) -> Dict[str, Any]:
    """MAREX's design by bounded search; the same dict as ``solve_design``.

    Takes ``solve_design``'s arguments less the branch-and-bound knobs
    (``solver``, ``verbose``, ``warm_start``, ``time_limit``, ``forbidden``).
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

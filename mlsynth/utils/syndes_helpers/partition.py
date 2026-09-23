"""The SYNDES inner solver: the quadratic program left once ``S`` is named.

:mod:`mlsynth.utils.syndes_helpers.gram` shows that naming the treated set
removes every binary from the two-way global MIP. What remains, for a given
``S``, is a strictly convex quadratic program over a product of two simplices:

.. math::

   f(S) = \\min \\{ u'(G + \\lambda I) u : u_S \\ge 0, \\textstyle\\sum_S u = 1;
   \\; u_{S^c} \\le 0, \\textstyle\\sum_{S^c} u = -1 \\}.

Geometrically that is the ridge-penalised distance between the convex hull of the
treated units' pre-treatment columns and the hull of the control units'. Its
minimiser is the pair of weight vectors the design uses.

Two facts shape the implementation. The feasible set is a product of two
simplices, so the Euclidean projection onto it is separable and exact -- a sort
per block. And the objective depends on the panel only through ``G``, so a batch
of designs shares one matrix and evaluates in one product per iteration. Together
they make projected gradient the right method: thousands of designs advance
together, each block projected in a single vectorised sort.

The iteration is FISTA at the fixed step ``1 / (2 lam_max(M))``. That step is
valid for every row at once because restricting the quadratic form to a
coordinate subset and flipping signs on part of it cannot raise the top
eigenvalue.

Every solve returns a Frank-Wolfe duality gap alongside the value. For a convex
objective over a compact convex set the gap bounds the distance to the optimum,
so ``value - gap`` is a valid lower bound and a caller can tell a converged
answer from an unconverged one without a reference solver. That matters here
because the conic solvers one would otherwise check against lose relative
accuracy as ``lam`` approaches zero, exactly where this program becomes hard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from .gram import TwoWayProblem


def project_simplex(V: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection of a ``(B, n)`` array onto the unit simplex.

    The standard sort-and-threshold construction: sort each row descending, find
    the largest prefix whose implied threshold keeps its own entries positive,
    and shift by that threshold.

    Raises
    ------
    MlsynthConfigError
        If ``V`` is not two-dimensional or has no columns.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise MlsynthConfigError(
            f"project_simplex expects a (B, n) array; got shape {V.shape}.")
    B, n = V.shape
    if n == 0:
        raise MlsynthConfigError("cannot project onto a simplex with no coordinates.")

    ordered = -np.sort(-V, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    cumulative -= 1.0
    position = np.arange(1, n + 1, dtype=float)
    condition = ordered * position > cumulative
    rho = n - 1 - np.argmax(condition[:, ::-1], axis=1)
    theta = cumulative[np.arange(B), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def split_indices(N: int, treated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``(B, K)`` treated indices into treated and control index arrays.

    Raises
    ------
    MlsynthConfigError
        If any row repeats a unit or names one outside ``0 .. N-1``.
    """
    treated = np.atleast_2d(np.asarray(treated, dtype=np.int64))
    B, K = treated.shape
    if K and (treated.min() < 0 or treated.max() >= N):
        raise MlsynthConfigError(
            f"treated indices must lie in 0..{N - 1}; got "
            f"[{treated.min()}, {treated.max()}].")

    mask = np.zeros((B, N), dtype=bool)
    np.put_along_axis(mask, treated, True, axis=1)
    if int(mask.sum()) != B * K:
        raise MlsynthConfigError("a treated set repeats a unit.")

    control = np.tile(np.arange(N), (B, 1))[~mask].reshape(B, N - K)
    return treated, control


@dataclass(frozen=True)
class PartitionSolution:
    """Batched inner-QP output.

    Attributes
    ----------
    value : np.ndarray
        ``(B,)`` objective at the returned weights.
    gap : np.ndarray
        ``(B,)`` Frank-Wolfe duality gap. ``value - gap`` lower-bounds ``f(S)``,
        so a small gap certifies the value.
    treated_weights : np.ndarray
        ``(B, K)`` weights on the treated units, in the order given.
    control_weights : np.ndarray
        ``(B, N - K)`` weights on the control units, in the order given.
    """

    value: np.ndarray
    gap: np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray


def solve_partition_batch(
    problem: TwoWayProblem,
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    *,
    max_iter: int = 400,
    tol: float = 1e-11,
    check_every: int = 10,
) -> PartitionSolution:
    """Solve ``f(S)`` for a batch of treated sets given as dense index arrays.

    ``treated_idx`` is ``(B, K)`` and ``control_idx`` is ``(B, N - K)``; row ``b``
    lists the units on each side of design ``b``. Keeping both blocks dense means
    the projection sorts ``K`` and ``N - K`` columns instead of masking ``N``,
    which is where the time goes.

    Iterates until every row's Frank-Wolfe gap falls under ``tol``, or for
    ``max_iter`` steps. Pass ``tol=0.0`` to run a fixed number of steps, which is
    what a cheap screening pass wants: the gaps are then loose but still valid,
    so pruning on ``value - gap`` stays sound.

    Raises
    ------
    MlsynthConfigError
        If the two index arrays disagree on batch size, if either side is empty,
        if an index is out of range, or if a unit appears on both sides.
    """
    M = problem.M
    N = problem.N
    treated_idx = np.atleast_2d(np.asarray(treated_idx, dtype=np.int64))
    control_idx = np.atleast_2d(np.asarray(control_idx, dtype=np.int64))

    if treated_idx.shape[0] != control_idx.shape[0]:
        raise MlsynthConfigError(
            f"treated and control index arrays disagree on batch size: "
            f"{treated_idx.shape[0]} vs {control_idx.shape[0]}.")
    B, K = treated_idx.shape
    n_control = control_idx.shape[1]
    if K == 0 or n_control == 0:
        raise MlsynthConfigError(
            "a two-way design needs at least one unit on each side; got "
            f"{K} treated and {n_control} control.")
    for name, block in (("treated", treated_idx), ("control", control_idx)):
        if block.min() < 0 or block.max() >= N:
            raise MlsynthConfigError(
                f"{name} indices must lie in 0..{N - 1}; got "
                f"[{block.min()}, {block.max()}].")

    rows = np.arange(B)[:, None]
    occupancy = np.zeros((B, N), dtype=np.int64)
    np.add.at(occupancy, (rows, treated_idx), 1)
    np.add.at(occupancy, (rows, control_idx), 1)
    if occupancy.max() > 1:
        raise MlsynthConfigError(
            "a unit appears on both sides of a design, or twice on one side; "
            "the treated and control blocks must partition distinct units.")

    step = problem.step
    A = np.full((B, K), 1.0 / K)
    Bw = np.full((B, n_control), 1.0 / n_control)
    A_prev, B_prev = A, Bw
    Ay, By = A, Bw
    momentum = 1.0
    contrast = np.zeros((B, N))

    def _gradients(a: np.ndarray, b: np.ndarray):
        contrast.fill(0.0)
        contrast[rows, treated_idx] = a
        contrast[rows, control_idx] = -b
        product = contrast @ M
        return product, 2.0 * product[rows, treated_idx], -2.0 * product[rows, control_idx]

    for iteration in range(max_iter):
        _, grad_a, grad_b = _gradients(Ay, By)
        A_new = project_simplex(Ay - step * grad_a)
        B_new = project_simplex(By - step * grad_b)

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        weight = (momentum - 1.0) / next_momentum
        Ay = A_new + weight * (A_new - A_prev)
        By = B_new + weight * (B_new - B_prev)
        A_prev, B_prev, momentum = A_new, B_new, next_momentum

        if tol > 0.0 and (iteration + 1) % check_every == 0:
            if _duality_gap(*_gradients(A_prev, B_prev)[1:],
                            A_prev, B_prev).max() < tol:
                break

    product, grad_a, grad_b = _gradients(A_prev, B_prev)
    value = np.einsum("bi,bi->b", contrast, product)
    gap = _duality_gap(grad_a, grad_b, A_prev, B_prev)
    return PartitionSolution(value=value, gap=gap,
                             treated_weights=A_prev, control_weights=B_prev)


def _duality_gap(grad_a, grad_b, a, b) -> np.ndarray:
    """Frank-Wolfe gap: the linear model's drop over the feasible set.

    A linear form is minimised over a simplex at one of its vertices, so the
    minimum is the smallest gradient entry within each block.
    """
    linear_drop = (
        np.einsum("bi,bi->b", grad_a, a) + np.einsum("bi,bi->b", grad_b, b)
        - grad_a.min(axis=1) - grad_b.min(axis=1)
    )
    return np.maximum(linear_drop, 0.0)

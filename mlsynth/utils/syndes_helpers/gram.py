"""Gram reduction of the SYNDES two-way global problem, and its bounds.

The two-way global problem of Doudchenko et al. (2021) chooses a treated set and
two weight vectors to minimise the mean squared error of the difference-in-
weighted-means estimator,

.. math::

   \\min_{D, w} \\; \\frac{1}{T} \\sum_t \\Big( \\sum_{i : D_i = 1} w_i Y_{it}
   - \\sum_{i : D_i = 0} w_i Y_{it} \\Big)^2 + \\sigma^2 \\sum_i w_i^2,

subject to the two normalisations
:math:`\\sum_{i : D_i = 1} w_i = \\sum_{i : D_i = 0} w_i = 1`.

``mlsynth`` hands this to SCIP as a mixed-integer program whose McCormick block
encodes :math:`q_i = w_i D_i`. Once the treated set ``S`` is named, that block
says only that ``q = w`` on ``S`` and ``0`` off it, so writing ``a`` for the
treated weights, ``b`` for the control weights and ``u = a - b``, every binary
disappears and what is left is a convex quadratic program:

.. math::

   f(S) = \\min \\{ u'(G + \\lambda I) u : u_S \\geq 0, \\textstyle\\sum_S u = 1;
   \\; u_{S^c} \\leq 0, \\textstyle\\sum_{S^c} u = -1 \\},

with ``G = Y'Y / T``. Three things follow, and this module supplies all three.

``Y`` enters only through ``G``. After one ``O(N^2 T)`` step the panel length is
gone, so a long panel costs no more per candidate design than a short one.

The two normalisations are the linear equalities ``1'u = 0`` and
``sigma'u = 2``, where ``sigma`` is the :math:`\\pm 1` indicator of ``S``.
Dropping the sign conditions -- which the paper calls "not strictly required" --
leaves a two-equality quadratic program with a closed form. With
``H = (G + lam I)^-1``, ``p = H1`` and ``alpha = 1'H1``,

    lb(S) = 4 alpha / (sigma' R sigma),      R = alpha H - p p'

lower-bounds ``f(S)`` always, and equals it whenever the minimiser it returns
satisfies the dropped sign conditions. A batch of candidate designs costs one
``(B, N) @ (N, N)`` product, so thousands of designs are scored for the price of
a single matrix multiplication.

``R 1 = 0``, because ``p = H1`` and ``p'1 = alpha``. Every feasible ``sigma``
therefore splits into a component along ``1``, which ``R`` annihilates, and an
orthogonal component of squared norm ``4K(N-K)/N``. Rayleigh's inequality then
caps the cut value and gives a bound over every size-``K`` design at once:

    f* >= alpha N / (K (N - K) lam_max(R))

from a single eigenvalue, with no relaxation to solve and no failure mode.
Because ``4 alpha`` is constant, ranking designs by ``lb`` means maximising
``sigma' R sigma`` over sign vectors with ``K`` positives -- a cardinality-
constrained max-cut, which is where the problem's hardness actually lives.
:func:`maxcut_candidates` searches it with swap moves that update ``R sigma`` by
two columns, touching no quadratic program at all.

Conditioning. ``H`` is the inverse of ``G + lam I``, which becomes singular as
``lam`` approaches zero on a panel with ``rank(G) < N``. The engine measures the
conditioning up front and refuses to report a bound it cannot compute reliably,
so a caller never receives a confidently wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)


#: Largest ``cond(M)`` at which the closed form is trusted. Chosen from measured
#: behaviour: the bound stayed sound at ``cond(M) = 1.25e9`` across a randomised
#: sweep, and a decade of headroom above that is where double precision still
#: leaves the inverse meaningful.
CONDITION_MAX = 1e10


@dataclass(frozen=True)
class TwoWayProblem:
    """A two-way global instance reduced to its Gram matrix.

    Attributes
    ----------
    G : np.ndarray
        ``Y'Y / T``, shape ``(N, N)``.
    M : np.ndarray
        ``G + lam I`` -- the matrix the objective is a quadratic form in.
    lam : float
        Penalty on the weights (``sigma^2`` in the paper, ``lam`` in mlsynth).
    N : int
        Number of units.
    """

    G: np.ndarray
    M: np.ndarray
    lam: float
    N: int

    @classmethod
    def from_panel(
        cls, Y: np.ndarray, lam: Optional[float] = None
    ) -> "TwoWayProblem":
        """Reduce a ``(T, N)`` pre-period matrix to its Gram form.

        ``lam`` defaults to the mean within-unit variance, matching
        :func:`mlsynth.utils.syndes_helpers.optimization.estimate_lambda`, so
        this path and the MIP minimise the same objective.

        Raises
        ------
        MlsynthDataError
            If ``Y`` is not a two-dimensional array.
        MlsynthConfigError
            If ``Y`` has fewer than two periods or fewer than two units, or if
            ``lam`` is negative.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
        T, N = Y.shape
        if T < 2:
            raise MlsynthConfigError("At least two pre-treatment periods are required.")
        if N < 2:
            raise MlsynthConfigError(
                "The two-way global design needs at least two units, one for each "
                f"side of the contrast; got N={N}."
            )

        lam_value = (
            float(np.mean(np.var(Y, axis=0, ddof=1))) if lam is None else float(lam)
        )
        if lam_value < 0:
            raise MlsynthConfigError("lam must be nonnegative.")

        G = (Y.T @ Y) / T
        G = 0.5 * (G + G.T)
        return cls(G=G, M=G + lam_value * np.eye(N), lam=lam_value, N=N)

    @property
    def step(self) -> float:
        """Projected-gradient step ``1 / (2 lam_max(M))``.

        Valid simultaneously for every treated set: restricting the quadratic
        form to a coordinate subset and flipping signs on part of it is a
        principal submatrix of an orthogonal similarity, so neither operation
        raises the top eigenvalue.
        """
        return 1.0 / (2.0 * float(np.linalg.eigvalsh(self.M)[-1]))

    def evaluate(self, a: np.ndarray, b: np.ndarray) -> float:
        """The objective at explicit treated and control weight vectors."""
        u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return float(u @ self.G @ u + self.lam * (a @ a + b @ b))


@dataclass(frozen=True)
class ClosedFormBounds:
    """Closed-form scores for a batch of candidate designs.

    Attributes
    ----------
    lower : np.ndarray
        ``(B,)`` values of ``lb(S)``; each lower-bounds the corresponding
        ``f(S)``.
    contrast : np.ndarray
        ``(B, N)`` minimisers of the equality-constrained relaxation.
    exact : np.ndarray
        ``(B,)`` booleans; where ``True`` the dropped sign conditions hold at the
        minimiser, so ``lower`` is ``f(S)`` itself.
    """

    lower: np.ndarray
    contrast: np.ndarray
    exact: np.ndarray


@dataclass(frozen=True)
class BoundEngine:
    """Precomputed pieces of the closed form for one instance.

    Attributes
    ----------
    H : np.ndarray
        ``M^-1``.
    p : np.ndarray
        ``H 1``.
    alpha : float
        ``1' H 1``, strictly positive because ``M`` is positive definite.
    R : np.ndarray
        ``alpha H - p p'`` -- positive semidefinite with ``R 1 = 0``. Ranking
        designs by the bound means maximising ``sigma' R sigma``.
    N : int
        Number of units.
    condition : float
        ``cond(M)``, measured when the engine was built.
    reliable : bool
        Whether ``condition`` is within ``condition_max``. When ``False`` the
        engine reports no bound instead of an unreliable one.
    """

    H: np.ndarray
    p: np.ndarray
    alpha: float
    R: np.ndarray
    N: int
    condition: float
    reliable: bool

    @classmethod
    def from_problem(
        cls, problem: TwoWayProblem, *, condition_max: float = CONDITION_MAX
    ) -> "BoundEngine":
        """Factor ``M`` and assemble the closed form's constant pieces."""
        condition = float(np.linalg.cond(problem.M))
        reliable = bool(np.isfinite(condition) and condition <= float(condition_max))
        if not reliable:
            zeros = np.zeros((problem.N, problem.N))
            return cls(H=zeros, p=np.zeros(problem.N), alpha=0.0, R=zeros,
                       N=problem.N, condition=condition, reliable=False)

        ones = np.ones(problem.N)
        # M is symmetric positive definite (a Gram matrix plus a nonnegative
        # ridge), so factor once and solve against the identity: more accurate
        # than a general inverse, and it reuses the factor for p. The batch
        # product needs H in full, so there is no avoiding the N columns.
        factor = cho_factor(problem.M, lower=True, check_finite=False)
        H = cho_solve(factor, np.eye(problem.N), check_finite=False)
        H = 0.5 * (H + H.T)
        p = cho_solve(factor, ones, check_finite=False)
        alpha = float(p @ ones)
        return cls(H=H, p=p, alpha=alpha, R=alpha * H - np.outer(p, p),
                   N=problem.N, condition=condition, reliable=True)

    def _require_reliable(self) -> None:
        if not self.reliable:
            raise MlsynthEstimationError(
                "the two-way Gram matrix is too ill-conditioned for the closed-form "
                f"bound (cond(M) = {self.condition:.3e}). This happens when the "
                "penalty approaches zero on a panel with fewer pre-periods than "
                "units. Raise lam, or add pre-treatment periods."
            )

    def bound(self, signs: np.ndarray, *, tol: float = -1e-9) -> ClosedFormBounds:
        """Score a ``(B, N)`` batch of :math:`\\pm 1` design indicators.

        One matrix product covers the whole batch. ``tol`` is the slack allowed
        on the sign conditions before a row loses its ``exact`` flag.

        Raises
        ------
        MlsynthEstimationError
            If the instance is too ill-conditioned to trust the closed form.
        """
        self._require_reliable()
        signs = np.asarray(signs, dtype=float)
        if signs.ndim != 2 or signs.shape[1] != self.N:
            raise MlsynthConfigError(
                f"signs must have shape (B, {self.N}); got {signs.shape}.")

        W = signs @ self.H                        # the one matrix product
        beta = np.einsum("bi,bi->b", W, signs)
        gamma = signs @ self.p
        # Cauchy-Schwarz in the H inner product makes this nonnegative, and
        # strictly positive whenever sigma is not parallel to the ones vector,
        # which every design with 0 < K < N satisfies.
        det = self.alpha * beta - gamma * gamma

        lower = 4.0 * self.alpha / det
        contrast = (2.0 / det)[:, None] * (
            self.alpha * W - gamma[:, None] * self.p[None, :]
        )
        exact = np.all(signs * contrast >= tol, axis=1)
        return ClosedFormBounds(lower=lower, contrast=contrast, exact=exact)

    def cut_value(self, signs: np.ndarray) -> np.ndarray:
        """``sigma' R sigma`` for a batch; the bound is ``4 alpha`` over this."""
        self._require_reliable()
        signs = np.asarray(signs, dtype=float)
        return np.einsum("bi,bi->b", signs @ self.R, signs)

    def global_lower_bound(self, K: int) -> Optional[float]:
        """Rayleigh bound on ``f(S)`` over every size-``K`` design.

        ``R 1 = 0`` removes the component of ``sigma`` along the ones vector, and
        the orthogonal component has squared norm ``4K(N-K)/N``, so
        ``sigma'R sigma <= 4K(N-K) lam_max(R)/N``. Returns ``None`` when the
        instance is too ill-conditioned for the closed form.

        Raises
        ------
        MlsynthConfigError
            If ``K`` is outside ``1 .. N-1``.
        """
        if not 1 <= int(K) <= self.N - 1:
            raise MlsynthConfigError(
                f"K must lie in 1..{self.N - 1} for a two-way design; got {K}.")
        if not self.reliable:
            return None
        top = float(np.linalg.eigvalsh(self.R)[-1])
        if not np.isfinite(top) or top <= 0.0:      # pragma: no cover - R is PSD
            return None
        return float(self.alpha * self.N / (K * (self.N - K) * top))


def signs_from_subsets(N: int, subsets: np.ndarray) -> np.ndarray:
    """Turn ``(B, K)`` treated-index rows into a ``(B, N)`` matrix of +/-1."""
    subsets = np.atleast_2d(np.asarray(subsets, dtype=np.int64))
    if subsets.size and (subsets.min() < 0 or subsets.max() >= N):
        raise MlsynthConfigError(
            f"treated indices must lie in 0..{N - 1}; got "
            f"[{subsets.min()}, {subsets.max()}].")
    signs = -np.ones((subsets.shape[0], N))
    np.put_along_axis(signs, subsets, 1.0, axis=1)
    return signs


def maxcut_candidates(
    engine: BoundEngine,
    K: int,
    *,
    n_restarts: int = 32,
    max_sweeps: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Treated sets that maximise ``sigma' R sigma``, one per restart.

    Best-improvement swaps with an incremental update: holding ``g = R sigma``,
    swapping treated ``i`` for control ``j`` changes the cut value by
    ``4 (g_j - g_i) + 4 (R_ii - 2 R_ij + R_jj)``, so the whole ``K(N-K)``
    neighbourhood is an outer sum and ``g`` refreshes from two columns of ``R``.
    No quadratic program is solved anywhere in here.

    The first restart starts from the ``K`` units with the largest diagonal of
    ``R``, which makes the result deterministic for a fixed seed; the rest are
    random. Duplicate local optima are collapsed, so the returned array may hold
    fewer rows than ``n_restarts``.

    Returns
    -------
    np.ndarray
        ``(B, K)`` array of sorted treated indices, one row per distinct optimum.
    """
    engine._require_reliable()
    if not 1 <= int(K) <= engine.N - 1:
        raise MlsynthConfigError(
            f"K must lie in 1..{engine.N - 1} for a two-way design; got {K}.")

    rng = np.random.default_rng(seed)
    N, R = engine.N, engine.R
    diagonal = np.diag(R)
    found = []

    for restart in range(max(1, int(n_restarts))):
        if restart == 0:
            treated = np.sort(np.argsort(-diagonal)[:K])
        else:
            treated = np.sort(rng.permutation(N)[:K])

        sigma = -np.ones(N)
        sigma[treated] = 1.0
        g = R @ sigma

        for _ in range(max_sweeps):
            inside = np.flatnonzero(sigma > 0)
            outside = np.flatnonzero(sigma < 0)
            delta = (
                4.0 * (g[outside][None, :] - g[inside][:, None])
                + 4.0 * (diagonal[inside][:, None] + diagonal[outside][None, :]
                         - 2.0 * R[np.ix_(inside, outside)])
            )
            flat = int(np.argmax(delta))
            row, col = divmod(flat, outside.size)
            if delta[row, col] <= 1e-12:
                break
            i, j = int(inside[row]), int(outside[col])
            sigma[i], sigma[j] = -1.0, 1.0
            g += 2.0 * (R[:, j] - R[:, i])

        found.append(np.flatnonzero(sigma > 0))

    return np.unique(np.array(found, dtype=np.int64), axis=0)

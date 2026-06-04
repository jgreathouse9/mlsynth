"""MSCMT-style backend for the bilevel SCM solver.

An alternative to the Malo et al. (2024) staged corner search of
:mod:`stages`. Instead of evaluating the ``K`` predictor corners and a local
Tykhonov descent, this backend performs a *global* search over the predictor
weights ``V`` -- log-scaled differential evolution -- with a
simplex-constrained, ``V``-weighted least-squares inner solve. This mirrors
the outer optimisation philosophy of

    Becker, M., & Kloessner, S. (2018). Fast and reliable computation of
    generalized synthetic controls. Econometrics and Statistics, 5, 1-19.
    https://doi.org/10.1016/j.ecosta.2017.08.002

whose R package MSCMT uses a conditioned global optimiser (genoud/DEoptim)
over ``log10(V)`` with a lower bound ``lb`` on the smallest predictor weight.
The two backends share the same outer objective (pre-treatment outcome MSPE)
and the same Section 3.1 global-optimum certificate; they differ only in how
the predictor weights are searched, which matters when the optimal ``V`` is
interior rather than a single corner.

The inner problem reuses the FISTA simplex-least-squares primitive
(:func:`mlsynth.utils.fscm_helpers.bilevel.simplex.simplex_lstsq`), the pure
analogue of MSCMT's WNNLS inner solver. The sunny-donor LP reduction of the
full MSCMT algorithm is not implemented here; it is a speed optimisation that
does not change the optimum.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import nnls

from .simplex import mspe
from .stages import unconstrained_feasibility, warn_on_gap
from .structure import BilevelProblem, BilevelSolution

_GAP_WARN_FACTOR = 10.0

# Big-M penalty enforcing the simplex equality 1'W = 1 inside the NNLS solve.
_BIG_M = 1e6


def _inner_weights(prob: BilevelProblem, V: np.ndarray) -> np.ndarray:
    """W*(V): simplex-constrained ``V``-weighted predictor least squares.

    Solves ``min_W ||diag(V)^{1/2} (X1 - X0 W)||^2`` over ``{W >= 0, 1'W = 1}``
    -- the MSCMT inner objective (Eq. 8'). The non-negativity constraint is
    handled by :func:`scipy.optimize.nnls` (the fast C analogue of MSCMT's
    WNNLS); the sum-to-one constraint is enforced by appending a large-penalty
    row ``M * 1' W = M``. ``nnls`` is used (rather than the pure-NumPy FISTA
    primitive) because the global outer search invokes the inner solve tens of
    thousands of times, where its speed and accuracy matter.
    """
    sq = np.sqrt(np.clip(V, 0.0, None))
    A = sq[:, None] * prob.X0
    b = sq * prob.X1
    J = prob.n_donors
    A = np.vstack([A, _BIG_M * np.ones(J)])
    b = np.concatenate([b, [_BIG_M]])
    w, _ = nnls(A, b, maxiter=10000)
    s = w.sum()
    return w / s if s > 0 else w


def solve_mscmt(
    prob: BilevelProblem,
    *,
    lb: float = 1e-8,
    maxiter: int = 300,
    popsize: int = 15,
    tol: float = 1e-10,
    seed: int = 0,
    polish: bool = True,
    feas_tol: float = 1e-8,
    canonical_v=False,
    gap_warn_factor: float = _GAP_WARN_FACTOR,
) -> BilevelSolution:
    """Solve the bilevel SCM problem by global outer search (MSCMT style).

    Parameters
    ----------
    prob : BilevelProblem
        Outcome and predictor matrices.
    lb : float
        Lower bound on the smallest predictor weight (MSCMT's conditioning
        bound, ``1e-8`` in the paper). The outer search runs over
        ``log10(V) in [log10(lb), 0]^K``; the objective is scale-free in
        ``V`` so the upper bound ``0`` (i.e. ``max V = 1``) is a free
        normalisation.
    maxiter, popsize, tol, seed, polish
        Forwarded to :func:`scipy.optimize.differential_evolution`.
    feas_tol : float
        Tolerance for the shared Section 3.1 / MSCMT Eq. 13 feasibility
        certificate (fast exact exit).
    canonical_v : bool or {"min.loss.w", "max.order"}
        If truthy, replace the raw optimiser ``V`` with an MSCMT canonical
        predictor-weight vector (see
        :func:`mlsynth.utils.fscm_helpers.bilevel.determine_v.canonical_v`).
        ``True`` selects ``"min.loss.w"`` (predictor-loss-minimising, sparse);
        ``"max.order"`` selects the leximin (balanced) vector. The predictor
        weights ``V`` are generically non-identified -- a whole polytope
        reproduces the same ``W`` -- so the raw optimiser ``V`` is not
        reproducible across runs/engines. Canonicalisation selects a unique,
        reproducible representative; it does **not** change ``W`` or the
        counterfactual. Falls back to the optimiser ``V`` if the canonical one
        fails to certify. When enabled, ``metadata["v_agreement"]`` reports the
        max gap between the two canonical choices (small = ``V`` well
        identified). Default ``False`` (historical behaviour).

    Returns
    -------
    BilevelSolution
        With ``stage="mscmt"`` (global search) or ``"mscmt-feasible"`` (the
        unconstrained outcome optimum was already bilevel-optimal).
    """
    from scipy.optimize import differential_evolution

    K = prob.n_predictors
    if K == 0:
        raise ValueError(
            "mscmt backend needs at least one predictor; for outcome-only "
            "matching use the penalized backend."
        )

    # Fast exact exit: the unconstrained outcome optimum is the global bilevel
    # solution whenever it is predictor-feasible (Malo Section 3.1 == MSCMT
    # Eq. 13). Shared by both backends.
    W_unc, V_star, lower_bound, is_optimal = unconstrained_feasibility(prob, feas_tol=feas_tol)
    if is_optimal:
        return BilevelSolution(
            V=V_star, W=W_unc,
            upper_loss=mspe(prob.y1_pre, prob.Y0_pre, W_unc),
            lower_loss=float(np.sum(V_star * (prob.X1 - prob.X0 @ W_unc) ** 2)),
            lower_bound=lower_bound, stage="mscmt-feasible", iterations=0,
            metadata={"certified": True, "backend": "mscmt"},
        )

    # Single predictor: V is fixed (up to scale), only the inner solve matters.
    if K == 1:
        V = np.array([1.0])
        W = _inner_weights(prob, V)
        upper = mspe(prob.y1_pre, prob.Y0_pre, W)
        warn_on_gap(float(upper - lower_bound), lower_bound, gap_warn_factor)
        return BilevelSolution(
            V=V, W=W, upper_loss=upper,
            lower_loss=float(np.sum(V * (prob.X1 - prob.X0 @ W) ** 2)),
            lower_bound=lower_bound, stage="mscmt", iterations=0,
            metadata={"backend": "mscmt"},
        )

    log_lb = float(np.log10(lb))

    def outer(logv: np.ndarray) -> float:
        V = np.power(10.0, logv)
        W = _inner_weights(prob, V)
        return mspe(prob.y1_pre, prob.Y0_pre, W)

    bounds = [(log_lb, 0.0)] * K

    # Seed the population with the K predictor corners (all mass on one
    # predictor, the rest at the lower bound) plus random draws, so the global
    # search starts from the same vertices the Malo corner stage would test.
    rng = np.random.default_rng(seed)
    n_init = max(popsize * K, K + 1)
    init = rng.uniform(log_lb, 0.0, size=(n_init, K))
    for k in range(min(K, n_init)):
        init[k, :] = log_lb
        init[k, k] = 0.0

    res = differential_evolution(
        outer, bounds, init=init, maxiter=maxiter, tol=tol,
        mutation=(0.3, 1.2), recombination=0.9, polish=polish, seed=seed,
    )
    if not res.success:
        warnings.warn(
            f"mscmt differential evolution did not converge (maxiter={maxiter}): "
            f"{res.message}. Returned predictor weights may be sub-optimal; "
            f"consider increasing maxiter.",
            RuntimeWarning,
            stacklevel=2,
        )

    V_raw = np.power(10.0, res.x)
    W = _inner_weights(prob, V_raw)
    V = V_raw / V_raw.sum()  # report on the simplex (objective is scale-free)
    upper = mspe(prob.y1_pre, prob.Y0_pre, W)
    warn_on_gap(float(upper - lower_bound), lower_bound, gap_warn_factor)
    v_method = "optimizer"
    meta_extra: dict = {}
    if canonical_v:
        from .determine_v import canonical_v as _canonical_v
        from .determine_v import canonical_v_diagnostics as _diag
        requested = "min.loss.w" if canonical_v is True else str(canonical_v)
        v_canon, ok = _canonical_v(prob, W, method=requested, lb=lb)
        if ok:
            V, v_method = v_canon, requested
        else:
            v_method = "optimizer-fallback"
        diag = _diag(prob, W, lb=lb)
        meta_extra["v_agreement"] = diag["agreement"]   # min.loss.w vs max.order
    return BilevelSolution(
        V=V, W=W, upper_loss=float(upper),
        lower_loss=float(np.sum((V * V_raw.sum()) * (prob.X1 - prob.X0 @ W) ** 2)),
        lower_bound=lower_bound, stage="mscmt", iterations=int(res.nit),
        metadata={
            "backend": "mscmt",
            "de_success": bool(res.success),
            "gap": float(upper - lower_bound),
            "lb": float(lb),
            "v_method": v_method,
            **meta_extra,
        },
    )

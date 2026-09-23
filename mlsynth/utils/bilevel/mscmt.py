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
interior, not a single corner.

The inner ``V``-weighted simplex least squares is solved exactly, and for the
whole outer population at once, by
:func:`~mlsynth.utils.bilevel.minnorm.solve_simplex_minnorm_batch`. The
sum-to-one constraint turns the inner objective into the homogeneous form
``W' G(V) W`` -- the minimum-norm point in the hull of the donors' predictor
discrepancies -- with ``G(V) = sum_k V_k r_k r_k'`` linear in ``V``. So the
``K`` rank-one pieces are formed once, a generation's Grams are one matrix
product against them, and the active set then certifies the whole generation in
a handful of batched linear solves. No product with the data occurs inside the
search.

Two further reductions leave the solution unchanged. The outer
differential-evolution population is evaluated *vectorised* (one objective call
per generation), and the donor pool is first reduced to its **sunny** donors
(the Becker-Kloessner LP reduction): shady donors can never carry positive inner
weight for any ``V``, so dropping them shrinks every inner solve.
"""

from __future__ import annotations

import warnings

import numpy as np

from .minnorm import solve_simplex_minnorm, solve_simplex_minnorm_batch
from .simplex import mspe
from .stages import unconstrained_feasibility, warn_on_gap
from .structure import BilevelProblem, BilevelSolution

_GAP_WARN_FACTOR = 10.0


def _predictor_discrepancies(prob: BilevelProblem) -> np.ndarray:
    """``R = X1 1' - X0``: column ``j`` is treated-minus-donor-``j``, shape ``(K, J)``.

    The inner objective is a quadratic form in these columns alone. On the
    simplex ``X1 - X0 W = R W``, so
    ``sum_k V_k (X1_k - (X0 W)_k)^2 = W' R' diag(V) R W`` and the donor weights
    are the minimum-norm point in the hull of ``R``'s columns under the metric
    ``diag(V)``.
    """
    return prob.X1[:, None] - prob.X0


def _inner_weights(prob: BilevelProblem, V: np.ndarray) -> np.ndarray:
    """W*(V): simplex-constrained ``V``-weighted predictor least squares.

    Solves ``min_W ||diag(V)^{1/2} (X1 - X0 W)||^2`` over ``{W >= 0, 1'W = 1}``
    -- the MSCMT inner objective (Eq. 8') -- exactly, by the active set of
    :func:`~mlsynth.utils.bilevel.minnorm.solve_simplex_minnorm` on the Gram
    ``R' diag(V) R``. The equality constraint is carried by the reduction to
    that form, not by a penalty, so the answer is exactly scale-free in ``V``,
    as the outer objective assumes. The hot loop in :func:`solve_mscmt` runs the
    batched form of the same solver over a whole population; this single-problem
    form serves the single-predictor path, :mod:`determine_v`, and callers that
    need one ``W*(V)``.

    Solve a *family* of ``W*(V)`` through
    :func:`~mlsynth.utils.bilevel.minnorm.solve_simplex_minnorm_batch`, not by
    calling this in a loop. The batched solver amortises its per-iteration
    numpy work over the whole batch, so at a batch of one that work is all
    overhead: this call costs about 0.4 ms on a 13-predictor, 17-donor problem
    where the penalised NNLS it replaced cost 0.02 ms, and where a loop over a
    population would cost a small fraction of either. The exchange is worth it
    at the call sites that exist -- :mod:`determine_v` and MEDSC make tens of
    calls per fit, tens of milliseconds against fits of hundreds, and MEDSC's
    Prop 99 replication runs in the same 2 seconds it did before -- and it buys
    an exact equality constraint in place of a big-M penalty. It would not be
    worth it in a loop over thousands.
    """
    R = _predictor_discrepancies(prob)
    V = np.clip(np.asarray(V, dtype=float).ravel(), 0.0, None)
    return solve_simplex_minnorm((R * V[:, None]).T @ R)


def _sunny_mask(X1: np.ndarray, X0: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of the *sunny* donors (Becker & Kloessner 2018).

    A donor ``d`` is **sunny** if it can receive positive weight in the inner
    solution for *some* predictor weighting ``V``, and **shady** if it gets zero
    weight for *every* ``V``. Geometrically (think of a light source at the
    treated unit ``X1``): sunny donors lie on the part of the donor convex hull
    that is *visible* from ``X1``; shady donors sit in its shadow, behind the
    hull. The ``V``-weighted projection of ``X1`` onto the hull can never land on
    a face containing a shady donor, so dropping shady donors leaves every inner
    ``W*(V)`` -- and therefore the whole outer objective -- unchanged.

    Donor ``d`` is visible iff there is a hyperplane through ``X0[:, d]`` with
    every donor on one side and ``X1`` on the outward side, i.e. the LP

        max_a  a . (X1 - X0_d)   s.t.   a . (X0_i - X0_d) <= 0  for all i,
                                        -1 <= a_k <= 1

    has a positive optimum. The normal ``a`` is free over ``R^K`` -- a superset
    of the ``V``-reachable normals -- so the test is **conservative**: it only
    ever classifies a donor as shady when it is provably shady (never drops a
    donor that could carry weight). Used as a pre-filter before the outer search.

    Returns a length-``J`` boolean array, ``True`` for donors to keep (sunny).
    """
    from scipy.optimize import linprog

    K, J = X0.shape
    mask = np.ones(J, dtype=bool)
    bnds = [(-1.0, 1.0)] * K
    zeros_J = np.zeros(J)
    for d in range(J):
        gap = X1 - X0[:, d]
        if float(gap @ gap) <= tol:        # donor coincides with X1 -> perfect, sunny
            continue
        A_ub = (X0 - X0[:, d : d + 1]).T   # rows a . (X0_i - X0_d) <= 0
        res = linprog(c=-gap, A_ub=A_ub, b_ub=zeros_J, bounds=bnds, method="highs")
        if res.success:
            mask[d] = (-res.fun) > tol
        # if the LP fails to solve, keep the donor (safe default)
    if not mask.any():                     # degenerate guard: never drop everything
        mask[:] = True
    return mask


def solve_mscmt(
    prob: BilevelProblem,
    *,
    lb: float = 1e-8,
    maxiter: int = 300,
    popsize: int = 15,
    tol: float = 1e-6,
    seed: int = 0,
    polish: bool = True,
    feas_tol: float = 1e-8,
    canonical_v=False,
    prune_shady: bool = True,
    inner_max_iter=None,
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
    maxiter, popsize, seed, polish
        Forwarded to :func:`scipy.optimize.differential_evolution`.
    tol : float
        Relative tolerance stopping the outer search, forwarded to
        :func:`scipy.optimize.differential_evolution`, which ends when the
        population's spread in pre-fit MSPE falls below ``tol`` times its mean
        (scipy's ``atol`` stays at its ``0`` default, so the rule is purely
        relative).

        The value is a statement about how precisely the estimate is wanted, so
        it is set from what the estimate does. On the Abadie-Gardeazabal Basque
        specification the mean energy is a pre-fit MSPE of about 0.0043; the
        donor weights reach 1e-5 of their final position by generation 93 and
        move by 1e-8 over the 120 generations after that. The default stops the
        search around generation 100, where the weights and the ATT sit within
        about 5e-6 of an exhaustive search's -- three orders finer than the four
        decimals the MSCMT replication compares to, and past the last digit the
        estimate is reported at. The previous ``1e-10`` asked 195 candidate
        predictor weightings to agree to 4.3e-13, thirteen significant figures,
        which most panels never reach: the search then spent its whole
        ``maxiter`` budget refining digits nobody reads.
    feas_tol : float
        Tolerance for the shared Section 3.1 / MSCMT Eq. 13 feasibility
        certificate (fast exact exit).
    canonical_v : bool or {"min.loss.w", "max.order"}
        If truthy, replace the raw optimiser ``V`` with an MSCMT canonical
        predictor-weight vector (see
        :func:`mlsynth.utils.bilevel.determine_v.canonical_v`).
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
    inner_max_iter : int, optional
        Cap on the inner active set's iterations per generation. ``None``
        (default) leaves the solver's own bound, which every well-posed donor
        pool reaches. A cap too small to certify is reported: the count lands in
        ``metadata["inner_unconverged"]`` and a :class:`RuntimeWarning` is
        raised once at the end of the search.
    prune_shady : bool
        If ``True`` (default), reduce the donor pool to its *sunny* donors
        (:func:`_sunny_mask`) before the outer search. Shady donors provably
        carry zero inner weight for every ``V``, so this leaves the optimum
        unchanged while shrinking the inner solve. ``metadata`` reports
        ``n_sunny`` / ``n_shady_pruned``.

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
    bounds = [(log_lb, 0.0)] * K

    # Sunny/shady reduction: drop donors that provably carry zero inner weight
    # for every V, so only the sunny pool enters the tens of thousands of inner
    # solves (the optimum is unchanged).
    sunny = _sunny_mask(prob.X1, prob.X0) if prune_shady else np.ones(prob.n_donors, bool)
    Rr = np.ascontiguousarray(
        _predictor_discrepancies(prob)[:, sunny])       # (K, Jr)
    Y0r = np.ascontiguousarray(prob.Y0_pre[:, sunny])
    y1 = prob.y1_pre
    Jr = Rr.shape[1]

    # The K rank-one pieces of the inner Gram, formed once: G(V) = sum_k V_k
    # r_k r_k' is linear in V, so a whole generation's Grams are one matrix
    # product against them and the data never enters the search again.
    rank_one = np.einsum("kj,kl->kjl", Rr, Rr).reshape(K, Jr * Jr)

    inner_state = {"unconverged": 0}

    # Every generation is solved cold, though seeding each candidate from the
    # previous generation's weights would cut the inner work by about a third.
    # The population moves slowly, so the seed would usually be optimal already
    # -- but where the inner optimum is a face and not a point, which member of
    # that face comes back would then depend on the search's history, and the
    # members differ in *outcome* fit even though they tie on predictor fit. The
    # outer objective would stop being a function of V alone. On the Lamba et al.
    # tiger reserves that showed up as a seed spread of 5e-2 ha on a 2825 ha
    # effect, against 2e-6 ha cold.
    def _inner_batch(logv: np.ndarray) -> np.ndarray:
        """Donor weights for a (K, S) block of log10 predictor weights."""
        G = (np.power(10.0, logv).T @ rank_one).reshape(-1, Jr, Jr)
        W, info = solve_simplex_minnorm_batch(
            G, max_iter=inner_max_iter, return_info=True)
        inner_state["unconverged"] += int((~info["converged"]).sum())
        return W

    def outer(logv: np.ndarray):
        # scipy passes (K,) during polish, (K, S) when vectorized=True
        single = logv.ndim == 1
        W = _inner_batch(logv[:, None] if single else logv)
        resid = y1[:, None] - Y0r @ W.T
        loss = (resid * resid).mean(axis=0)
        return float(loss[0]) if single else loss

    # Seed the population with the K predictor corners (all mass on one
    # predictor, the rest at the lower bound) plus random draws, so the global
    # search starts from the same vertices the Malo corner stage would test.
    rng = np.random.default_rng(seed)
    n_init = max(popsize * K, K + 1)
    init = rng.uniform(log_lb, 0.0, size=(n_init, K))
    for k in range(min(K, n_init)):
        init[k, :] = log_lb
        init[k, k] = 0.0

    with warnings.catch_warnings():
        # vectorized necessarily implies updating='deferred'; that's intended.
        warnings.filterwarnings("ignore", message=".*vectorized.*", category=UserWarning)
        res = differential_evolution(
            outer, bounds, init=init, maxiter=maxiter, tol=tol,
            mutation=(0.3, 1.2), recombination=0.9, polish=polish, seed=seed,
            vectorized=True,
        )
    if not res.success:
        warnings.warn(
            f"mscmt differential evolution did not converge (maxiter={maxiter}): "
            f"{res.message}. Returned predictor weights may be sub-optimal; "
            f"consider increasing maxiter.",
            RuntimeWarning,
            stacklevel=2,
        )

    if inner_state["unconverged"]:
        warnings.warn(
            f"{inner_state['unconverged']} of the inner simplex solves hit "
            f"their iteration cap during the outer search, so the predictor "
            f"weightings they scored were evaluated at a feasible but "
            f"uncertified W. The reported solution is unaffected only if the "
            f"optimiser did not settle on one of them.",
            RuntimeWarning,
            stacklevel=2,
        )

    V_raw = np.power(10.0, res.x)
    W = np.zeros(prob.n_donors)
    W[sunny] = _inner_batch(res.x[:, None])[0]   # re-expand to the full pool
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
            "n_donors": int(prob.n_donors),
            "n_sunny": int(sunny.sum()),
            "n_shady_pruned": int((~sunny).sum()),
            "inner_unconverged": int(inner_state["unconverged"]),
            **meta_extra,
        },
    )

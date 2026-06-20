"""Optional SCIP mixed-integer backend for HCW best-subset selection.

Solves the Bertsimas-King-Mazumder (2016) cardinality-constrained least squares
to provable optimality with the SCIP MIQP solver, as an alternative to the
exact Furnival-Wilson search for pools too large for the latter to certify
within its node budget. Importing this module requires ``pyscipopt``; the caller
(:func:`mlsynth.utils.pda_helpers.hcw.estimation.best_subset_select`) imports it
lazily and translates a missing dependency into an actionable error, so SCIP
stays an opt-in extra rather than a hard requirement.

For each model size ``k`` we solve, on the mean-centred donor cross-products
(the intercept is absorbed by centring),

    min  beta' Sxx beta - 2 Sxy' beta + syy
    s.t. (beta_i, 1 - z_i) : SOS-1,   sum_i z_i <= k,   z in {0, 1},

i.e. the residual sum of squares subject to at most ``k`` active donors, the
SOS-1 pair forcing ``beta_i = 0`` whenever ``z_i = 0``. The quadratic objective
is modelled as ``min t`` with ``t >= RSS(beta)`` (SCIP takes a linear objective
with a quadratic constraint). The size minimising the information criterion is
then returned, exactly as in the Furnival-Wilson two-step.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pyscipopt import Model, quicksum

from .estimation import _centered_gram, _subset_rss, info_criterion

# Box bound on the regression coefficients in the MIQP. Generous relative to any
# OLS coefficient on standardised-scale panel data; only needs to contain the
# optimum for the formulation to stay exact.
_BETA_BOUND = 1e6


def _solve_size(Sxx, Sxy, syy, N, k, time_limit):
    """Best size-``k`` subset by RSS via one MIQP solve.

    Returns ``(support, rss, gap, optimal)`` where ``support`` is the active
    donor set, ``rss`` the objective (residual sum of squares), ``gap`` SCIP's
    final optimality gap and ``optimal`` whether it certified the solution.
    """
    m = Model()
    m.hideOutput()
    if time_limit is not None:
        m.setParam("limits/time", float(time_limit))

    beta = [m.addVar(f"b{i}", lb=-_BETA_BOUND, ub=_BETA_BOUND) for i in range(N)]
    z = [m.addVar(f"z{i}", vtype="B") for i in range(N)]
    w = [m.addVar(f"w{i}", lb=0.0, ub=1.0) for i in range(N)]
    for i in range(N):
        m.addCons(w[i] == 1 - z[i])
        m.addConsSOS1([beta[i], w[i]])           # z_i = 0  =>  beta_i = 0
    m.addCons(quicksum(z) <= k)

    rss_expr = (
        quicksum(float(Sxx[i, j]) * beta[i] * beta[j]
                 for i in range(N) for j in range(N))
        - 2.0 * quicksum(float(Sxy[i]) * beta[i] for i in range(N))
        + syy
    )
    t = m.addVar("t", lb=-1e9)
    m.addCons(t >= rss_expr)
    m.setObjective(t, "minimize")
    m.optimize()

    support = sorted(i for i in range(N) if m.getVal(z[i]) > 0.5)
    rss = float(m.getObjVal())
    optimal = m.getStatus() == "optimal"
    gap = float(m.getGap())
    return support, rss, gap, optimal


def best_subset_scip(
    G: np.ndarray,
    Zty: np.ndarray,
    yty: float,
    N: int,
    n: int,
    r_max: int,
    criterion: str,
    *,
    time_limit: Optional[float] = None,
    stats: Optional[dict] = None,
) -> List[int]:
    """Best-subset donor selection by SCIP MIQP, chosen by ``criterion``.

    Solves the cardinality-constrained least squares to optimality for each size
    ``0 <= k <= r_max`` and returns the support minimising the information
    criterion. ``stats`` (if given) receives ``backend``, ``optimality_gap``
    (the worst SCIP gap over the per-size solves) and ``certified`` (whether
    every solve was proved optimal). ``time_limit`` caps each per-size solve.
    """
    Sxx, Sxy, syy = _centered_gram(G, Zty, yty, n)

    best_idx: List[int] = []
    best_ic = info_criterion(syy, n, 1, criterion)   # intercept-only (k = 0)
    worst_gap = 0.0
    all_optimal = True

    for k in range(1, r_max + 1):
        support, _rss, gap, optimal = _solve_size(Sxx, Sxy, syy, N, k, time_limit)
        worst_gap = max(worst_gap, gap)
        all_optimal = all_optimal and optimal
        # Trust SCIP for the optimal support, but recompute the RSS exactly from
        # the Gram for the criterion: the solver's objective carries a feasibility
        # tolerance that, at near-zero RSS, can skew a log-based criterion.
        rss = _subset_rss(G, Zty, yty, support)
        ic = info_criterion(rss, n, len(support) + 1, criterion)
        if ic < best_ic:
            best_ic = ic
            best_idx = support

    if stats is not None:
        stats.update(
            backend="scip",
            optimality_gap=worst_gap,
            certified=bool(all_optimal),
        )
    return sorted(best_idx)

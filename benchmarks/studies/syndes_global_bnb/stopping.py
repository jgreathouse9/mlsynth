"""The lift's stopping near the prune target (#837, candidate 1).

The library's lift (:func:`mlsynth.utils.syndes_helpers.branch.lift`) runs its
regularised dual in rounds, halving ``alpha`` from 0.1 to a floor of 5e-5, and
ends at the floor with no new triangle, or once it has run ``MIN_ROUNDS``
rounds and the smoothed value stalls. #842 found two costs in that schedule:

- where the relaxation has room, the floor leaves part of it: a floor of 1e-8
  recovered BLS at 60 units from 0.711% to 0.6465%, for eleven times the
  evaluations;
- the stall test is decided in its last digits, so whether a lower floor is
  reached depends on the machine's BLAS kernel.

And #864's flat panel at 40 units showed the opposite cost: lifts that reach
their relaxation and still cannot prune spend 299 evaluations each.

:func:`lift` is a copy of the library's round loop with three knobs, so each
can be measured on the library's own dual (``_theta``), separation and
certificate:

- ``on_stall="halve"``: a stall above the floor halves ``alpha`` and
  continues, where the shipped rule gives up; at the floor, with nothing
  left to halve, a stall ends the run;
- ``margin``: once the best bound is within ``margin`` (relative) of the
  target, the floor drops to ``deep_alpha_min`` and the tolerance to
  ``deep_tol_min``, so only nodes that nearly prune pay for the lower floor;
- ``forecast``: BiqBin's rule (Gusmeroli et al. 2022, p. 10), the run ends
  once the bound's recent progress, carried ``forecast_horizon`` rounds ahead,
  cannot reach the target; with ``forecast_from``, only once ``alpha`` is that
  small, where the bound has stopped falling with ``alpha``.

With every knob off it is the library's lift, bound and evaluations, which the
tests hold it to.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize

from mlsynth.utils.syndes_helpers import branch

#: How a run ended: the trust region was the bound, the target was reached,
#: the floor held, the rounds ran out, the smoothed value stalled, or the
#: forecast gave up.
ENDINGS = ("trust_region", "target", "floor", "rounds", "stall", "forecast")

#: The stall test's relative change, the library's 1e-6.
STALL = 1e-6


def lift(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
         target: Optional[float] = None, alpha_min: Optional[float] = None,
         on_stall: str = "give_up", margin: Optional[float] = None,
         deep_alpha_min: float = 1e-8, deep_tol_min: float = 1e-5,
         forecast: bool = False, forecast_horizon: int = 10,
         forecast_from: float = np.inf):
    """The library's lift with the module's knobs: ``(U, multipliers,
    evaluations, info)``, ``info`` holding how it ended (:data:`ENDINGS`),
    its rounds, its last ``alpha`` and whether it went deep."""
    if on_stall not in ("give_up", "halve"):
        raise ValueError(f"no stall rule named {on_stall!r}")
    if margin is not None and margin < 0:
        raise ValueError("the margin is a relative distance, at least zero")
    if alpha_min is not None and alpha_min <= 0:
        raise ValueError("the alpha floor must be positive")
    if forecast_horizon < 1:
        raise ValueError("the forecast looks at least one round ahead")
    if forecast_from <= 0:
        raise ValueError("the forecast starts at a positive alpha")
    with branch._blas().limit(limits=1, user_api="blas"):
        return _lift(c, b, A, s, target, branch.ALPHA_MIN if alpha_min is None else alpha_min,
                     on_stall, margin, deep_alpha_min, deep_tol_min, forecast,
                     forecast_horizon, forecast_from)


def _lift(c, b, A, s, target, alpha_min, on_stall, margin, deep_alpha_min,
          deep_tol_min, forecast, forecast_horizon, forecast_from):
    info = {"ended": "trust_region", "rounds": 0, "final_alpha": None, "deep": False}
    n = len(b)
    fallback, _, trm = branch.trust_region(c, b, A, s)
    if n <= 1 or (target is not None and fallback <= target):
        return fallback, trm, 0, info
    unit = max(float(np.abs(b).max()), float(np.abs(A).max()))
    if unit == 0.0:
        return fallback, trm, 0, info
    bs, As = b / unit, A / unit
    goal = -np.inf if target is None else (target - c) / unit
    floor = (fallback - c) / unit
    size = n + 1
    ny = 2 * n + 2
    y, cut, v = np.zeros(ny), branch.Cuts.empty(), np.zeros(0)
    alpha, tol = branch.ALPHA0, branch.TOL0
    alpha_floor, tol_floor = alpha_min, branch.TOL_MIN
    best = {"value": np.inf, "y": None, "cuts": None, "v": None}
    evaluations = [0]
    rounds = at_alpha = 0
    previous = None
    history = []

    while True:
        rounds += 1
        at_alpha += 1
        last: Dict[str, object] = {}

        def fun(xv, cut=cut, alpha=alpha):
            exact, smooth, gy, gv, _ = branch._theta(0.0, bs, As, s, xv[:ny], cut, xv[ny:], alpha)
            evaluations[0] += 1
            if exact < best["value"]:
                best.update(value=exact, y=xv[:ny].copy(), cuts=cut, v=xv[ny:].copy())
            last["x"], last["g"] = xv.copy(), (gy, gv)
            if min(best["value"], floor) <= goal:
                raise branch._Pruned
            return smooth, np.concatenate([gy, gv])

        def stop(xk, tol=tol):
            if "x" not in last or not np.array_equal(last["x"], xk):
                return  # pragma: no cover - as the library's: a stale gradient never decides convergence
            gy, gv = last["g"]
            if (np.abs(gy).max() < tol
                    and (not len(gv) or -min(gv.min(), 0.0) / branch.KAPPA < tol)):
                raise branch._Converged

        x0 = np.concatenate([y, v])
        bounds = [(None, None)] * ny + [(0.0, None)] * len(v)
        try:
            res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds, callback=stop,
                           options={"maxcor": branch.MAXCOR, "maxfun": branch.EVALUATIONS,
                                    "maxiter": branch.EVALUATIONS, "ftol": 0.0, "gtol": 0.0})
            xv = res.x
        except branch._Pruned:
            info["ended"] = "target"
            break
        except branch._Converged:
            xv = last["x"]
        y, v = xv[:ny], xv[ny:]
        _, smooth, gy, gv, X = branch._theta(0.0, bs, As, s, y, cut, v, alpha)
        stalled = (previous is not None and abs(previous - smooth) <= STALL * abs(smooth)
                   and alpha < 1e-4)
        previous = smooth
        if stalled and on_stall == "halve" and alpha > alpha_floor:
            stalled, at_alpha, previous = False, ROUNDS_FORCE, None
        giveup = rounds >= branch.MIN_ROUNDS and (rounds >= branch.MAX_ROUNDS or stalled)
        violation = -min(gv.min(), 0.0) / branch.KAPPA if len(gv) else 0.0
        if len(cut):
            keep = ~((v < 1e-8) & (cut.values(X) > violation / 10.0))
            cut, v = cut.take(keep), v[keep]
        found = branch.separate(X)
        fresh = ~np.isin(found.keys(size), cut.keys(size))
        added = int(fresh.sum())
        if added:
            cut = branch.Cuts.concat(cut, found.take(fresh))
            v = np.concatenate([v, np.zeros(added)])
        history.append(best["value"])
        if (margin is not None and target is not None and not info["deep"]
                and c + unit * best["value"] - target <= margin * abs(target)):
            info["deep"] = True
            alpha_floor, tol_floor = min(alpha_floor, deep_alpha_min), min(tol_floor, deep_tol_min)
        if (forecast and target is not None and alpha <= forecast_from
                and _out_of_reach(history, goal, forecast_horizon)):
            info["ended"] = "forecast"
            break
        if giveup or (alpha == alpha_floor and added == 0):
            info["ended"] = ("floor" if alpha == alpha_floor and added == 0
                             else "rounds" if rounds >= branch.MAX_ROUNDS else "stall")
            break
        if added < branch.MIN_CUTS or at_alpha > branch.ROUNDS_AT_ALPHA:
            at_alpha = 0
            alpha = max(alpha * branch.ALPHA_SCALE, alpha_floor)
            tol = max(tol * branch.TOL_SCALE, tol_floor)
    info.update(rounds=rounds, final_alpha=alpha)
    if best["value"] < floor:
        m = branch._as_multipliers(c, s, best["y"], best["cuts"], best["v"], unit)
        return c + unit * best["value"], m, evaluations[0], info
    return fallback, trm, evaluations[0], info


#: A stall under ``on_stall="halve"`` forces the next halving: the round count
#: at this ``alpha`` is set past ``ROUNDS_AT_ALPHA``.
ROUNDS_FORCE = 10 ** 9


def _out_of_reach(history: list, goal: float, horizon: int) -> bool:
    """BiqBin's forecast: the best bound's improvement over the last round,
    carried ``horizon`` rounds ahead, still short of the goal. Two rounds are
    needed to read a rate; a round that improved nothing reads as out of
    reach."""
    if len(history) < 2 or not np.isfinite(history[-2]):
        return False
    rate = history[-2] - history[-1]
    return history[-1] - horizon * max(rate, 0.0) > goal


# ----------------------------------------------------------------------
# the measurements
# ----------------------------------------------------------------------

#: The variants measured, by the knobs each sets. ``targeted`` ones read the
#: prune target; the others are measured without one, for how far they go.
VARIANTS = {
    "shipped": dict(),
    "floor": dict(alpha_min=1e-8),
    "halve": dict(alpha_min=1e-8, on_stall="halve"),
}
TARGETED = {
    "shipped": dict(),
    "margin": dict(margin=0.005),
    "forecast": dict(forecast=True),
    "forecast_late": dict(forecast=True, forecast_from=1e-3),
    "forecast_later": dict(forecast=True, forecast_from=1e-4),
    "margin_forecast_late": dict(margin=0.005, forecast=True, forecast_from=1e-3),
}

#: The node depths of ``duals.ratios``, in its order.
DEPTHS = (0, 2, 4, 6, 3, 5, 7, 1)


def ratios(Y: np.ndarray, K: int, *, depths=DEPTHS, seed: int = 0) -> list:
    """At nodes of a panel small enough to price every set: each variant's
    node bound over the subtree's minimum price, its evaluations and seconds;
    and with the prune target the optimum sets, whether each targeted variant
    prunes and what it spends deciding. A ratio of one is the subtree's
    minimum, the most any bound can reach; the node prunes once its ratio
    passes ``best / subtree``."""
    import time

    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from .duals import _node
    from .quality import price_all

    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    N = Y.shape[1]
    sets, price, _ = price_all(eng, N, K)
    best = float(price.min())
    U_target = 4.0 * eng.alpha / best
    member = np.zeros((len(sets), N), dtype=bool)
    member[np.arange(len(sets))[:, None], sets] = True
    rng = np.random.default_rng(seed)
    rows = []
    for d in depths:
        I, O, F = _node(rng, N, K, d)
        mask = np.ones(len(sets), dtype=bool)
        for u in I:
            mask &= member[:, u]
        for u in O:
            mask &= ~member[:, u]
        sub = float(price[mask].min())
        c, b, A = branch.node_parts(eng.R, I, O, F)
        s = 2 * (K - len(I)) - len(F)
        ratio = lambda U: 4.0 * eng.alpha / U / sub
        row = {"N": N, "K": K, "fixed": len(I) + len(O), "free": len(F),
               "need": best / sub, "trust_region": ratio(branch.trust_region(c, b, A, s)[0])}
        for name, knobs in VARIANTS.items():
            t0 = time.process_time()
            U, _, ev, info = lift(c, b, A, s, **knobs)
            row[name] = {"ratio": ratio(U), "evaluations": ev, "ended": info["ended"],
                         "seconds": time.process_time() - t0}
        for name, knobs in TARGETED.items():
            t0 = time.process_time()
            U, _, ev, info = lift(c, b, A, s, target=U_target, **knobs)
            row[f"{name}_targeted"] = {"pruned": bool(U <= U_target), "evaluations": ev,
                                        "ended": info["ended"], "deep": info["deep"],
                                        "seconds": time.process_time() - t0}
        rows.append(row)
    return rows


def search(Y: np.ndarray, K: int, *, work_limit: float = float("inf"),
           costs: Optional[dict] = None, **knobs) -> dict:
    """The library's branch and bound, under the library's cost model or
    ``costs``, with :func:`lift`'s knobs in place of the shipped lift, on one
    BLAS thread: its counts, value, bound and CPU seconds, and the lifts'
    evaluations and how they ended. The library's lift is put back on the way
    out, whether the search returns or raises."""
    import collections
    import time

    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.exact import SubsetFilter
    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    shipped = branch.lift
    endings = collections.Counter()
    spent = [0]

    def variant(c, b, A, s, *, target=None):
        U, m, ev, info = lift(c, b, A, s, target=target, **knobs)
        endings[info["ended"]] += 1
        spent[0] += ev
        return U, m, ev

    t0 = time.process_time()
    branch.lift = variant
    try:
        with threadpool_limits(1):
            o = branch.branch_and_bound(problem, engine, K, SubsetFilter(),
                                        work_limit=work_limit, **(costs or {}))
    finally:
        branch.lift = shipped
    return {"N": Y.shape[1], "K": K, "knobs": dict(knobs), "nodes": o.nodes,
            "lifts": o.lifts, "pruned": o.pruned, "priced": o.priced, "work": o.work,
            "closed": o.closed, "certified": o.certified, "value": o.value,
            "lower": o.lower, "evaluations": spent[0], "endings": dict(endings),
            "cpu_seconds": time.process_time() - t0}


def nodes40(Y: np.ndarray, K: int, optimum: float, *, depths=(0, 2, 4, 6, 8),
            seed: int = 0, closure: bool = True) -> list:
    """At ``duals.nodes``'s nodes of a panel too large to price: each
    variant's bound over the triangle closure's (``bnb_scale``'s
    ``root_gap.closure``, the value the lift's dual converges to), its
    evaluations and seconds; and with the prune target ``optimum`` sets,
    whether each targeted variant prunes and what it spends deciding."""
    import time

    from threadpoolctl import threadpool_limits

    from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

    from .duals import _node

    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    N = Y.shape[1]
    U_target = 4.0 * eng.alpha / optimum
    rng = np.random.default_rng(seed)
    rows = []
    with threadpool_limits(1):
        for d in depths:
            I, O, F = _node(rng, N, K, d)
            c, b, A = branch.node_parts(eng.R, I, O, F)
            s = 2 * (K - len(I)) - len(F)
            row = {"N": N, "K": K, "fixed": len(I) + len(O), "free": len(F),
                   "trust_region": branch.trust_region(c, b, A, s)[0]}
            if closure:
                from benchmarks.studies.bnb_scale.root_gap import closure as solve
                row["closure"] = solve(c, b, A, s)["U"]
            for name, knobs in VARIANTS.items():
                t0 = time.process_time()
                U, _, ev, info = lift(c, b, A, s, **knobs)
                row[name] = {"U": U, "evaluations": ev, "ended": info["ended"],
                             "seconds": time.process_time() - t0}
            for name, knobs in TARGETED.items():
                t0 = time.process_time()
                U, _, ev, info = lift(c, b, A, s, target=U_target, **knobs)
                row[f"{name}_targeted"] = {"pruned": bool(U <= U_target), "evaluations": ev,
                                            "ended": info["ended"], "deep": info["deep"],
                                            "seconds": time.process_time() - t0}
            rows.append(row)
    return rows


#: Where the runs append their rows.
RESULTS = __import__("pathlib").Path(__file__).resolve().parent / "results"

#: The search's variants by name, for the command line.
SEARCHES = {"shipped": dict(), "margin": dict(margin=0.005),
            "forecast_late": dict(forecast=True, forecast_from=1e-3),
            "margin_forecast_late": dict(margin=0.005, forecast=True, forecast_from=1e-3)}


def main(argv=None) -> int:
    """``ratios PANEL N K``, ``closures PANEL N K``, ``nodes PANEL N K OPTIMUM``
    ``search PANEL N K VARIANT`` or ``limit KIND N VARIANT``: one JSON line
    per row, appended to ``results/stopping_<mode>.jsonl``. ``closures`` is
    ``cuts.closures``; ``limit`` searches a #842 panel (``bnb_scale``'s BLS,
    noise or flat) at ``K = N/2`` under the default work limit.

        python -m benchmarks.studies.syndes_global_bnb.stopping ratios factor 24 4
        python -m benchmarks.studies.syndes_global_bnb.stopping closures factor 24 4
        python -m benchmarks.studies.syndes_global_bnb.stopping nodes factor 40 10 0.3
        python -m benchmarks.studies.syndes_global_bnb.stopping search factor 40 10 margin
        python -m benchmarks.studies.syndes_global_bnb.stopping limit bls 60 forecast_late
    """
    import sys

    from .run import PANELS

    args = list(sys.argv[1:] if argv is None else argv)
    mode, panel, N = args[0], args[1], int(args[2])
    if mode == "limit":
        from benchmarks.studies.bnb_scale import panels
        from mlsynth.utils.syndes_helpers.exact import WORK_LIMIT
        rows = [dict(search(panels.panel(panel, N), N // 2, work_limit=WORK_LIMIT,
                            **SEARCHES[args[3]]), variant=args[3])]
        return _write(mode, panel, rows)
    K = int(args[3])
    Y = PANELS[panel](N)
    if mode == "ratios":
        rows = ratios(Y, K)
    elif mode == "closures":
        from .cuts import closures
        rows = closures(Y, K)
    elif mode == "nodes":
        rows = nodes40(Y, K, float(args[4]))
    elif mode == "search":
        rows = [dict(search(Y, K, **SEARCHES[args[4]]), variant=args[4])]
    else:
        raise ValueError(f"no mode named {mode!r}")
    return _write(mode, panel, rows)


def _write(mode: str, panel: str, rows: list) -> int:
    import json

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"stopping_{mode}.jsonl", "a") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row, panel=panel), default=float) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line's entry; main is tested directly
    raise SystemExit(main())

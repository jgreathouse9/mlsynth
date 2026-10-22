"""Where the root's gap comes from (#842, for #837).

At 60 to 100 units the search's certified bound hardly moves past the root:
on 100 BLS counties, ten times the work left it at 0.372% under the design.
Two causes would look the same from outside. The relaxation may be too weak
there, which is #837's question; or the lift, which stops its regularised dual
at ``ALPHA_MIN`` and ``TOL_MIN``, may leave bound on the table, which a longer
solve recovers. :func:`root_bounds` separates them at the root, every unit
free, as node lower bounds ``4 alpha / U`` against the best design known:

- ``trust_region``: the cheap bound, the floor of every lift;
- ``lift``: the library's lift as the search runs it;
- ``lift_long``: the same dual with a finer stop, ``LONG``;
- ``interior``: Shor's relaxation with the cardinality, its products and the
  four RLT products per pair, solved by interior point and certified from its
  multipliers (``syndes_global_bnb.bounds.lift_upper``). It has no triangle
  cuts, so the lift can exceed it; the two together bracket what the
  relaxation family gives.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

#: The finer stop: ``alpha`` down to 1e-8 and gradients to 1e-5, with rounds
#: enough to get there.
LONG = {"ALPHA_MIN": 1e-8, "TOL_MIN": 1e-5, "MAX_ROUNDS": 1_000}
#: The finer stop with the stall rule off: the lift gives up only once it has
#: run ``MIN_ROUNDS`` rounds, so it ends at its floor with no new cut.
NO_STALL = {**LONG, "MIN_ROUNDS": 1_000}
_DEFAULTS = {name: getattr(branch, name) for name in NO_STALL}


@contextmanager
def schedule(**values):
    """The lift's stopping constants replaced for the length of a call, and
    every one of them put back on the way out, whether the call returns or
    raises. A name the module does not have is refused: setting it would
    change nothing, and the run would measure the shipped schedule."""
    unknown = [name for name in values if not hasattr(branch, name)]
    if unknown:
        raise ValueError(f"the lift has no stopping constant named {unknown}")
    before = {name: getattr(branch, name) for name in values}
    for name, value in values.items():
        setattr(branch, name, value)
    try:
        yield
    finally:
        for name, value in before.items():
            setattr(branch, name, value)


_ORIGINAL = {"theta": branch._theta, "separate": branch.separate,
             "concat": branch.Cuts.concat}


def trace_lift(Y: np.ndarray, K: int, **values) -> dict:
    """One root lift, every unit free, under ``schedule(**values)``, with the
    rule that ended it.

    Each round of :func:`~mlsynth.utils.syndes_helpers.branch.lift` solves the
    regularised dual at one ``alpha``, evaluates it once more at the solution,
    separates triangles and adds the new ones. The trace reads those steps by
    wrapping ``_theta``, ``separate`` and ``Cuts.concat`` for the call, and
    applies the lift's own stopping tests to each round:

    - ``at_floor``: ``alpha`` at ``ALPHA_MIN`` and no new triangle;
    - ``stalled``: the smoothed value moved by at most 1e-6 relative to the
      round before, with ``alpha`` under 1e-4;
    - ``giveup``: at least ``MIN_ROUNDS`` rounds, and ``MAX_ROUNDS`` reached
      or ``stalled``.

    The lift breaks at the first round with ``giveup`` or ``at_floor``, so a
    correct replica marks the last round and no other; the tests hold it to
    that. ``stopped`` names the rule: ``floor`` when the floor held (the run
    would have ended there whatever else held), else ``rounds`` or ``stall``.
    """
    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    N = Y.shape[1]
    parts = branch.node_parts(engine.R, (), (), tuple(range(N)))
    events: list = []

    def theta(*args, **kwargs):
        out = _ORIGINAL["theta"](*args, **kwargs)
        events.append(("theta", args[-1], out[1]))
        return out

    def separate(X, **kwargs):
        out = _ORIGINAL["separate"](X, **kwargs)
        events.append(("separate",))
        return out

    def concat(a, b):
        events.append(("concat",))
        return _ORIGINAL["concat"](a, b)

    with schedule(**values):
        alpha_min, min_rounds, max_rounds = (branch.ALPHA_MIN, branch.MIN_ROUNDS,
                                             branch.MAX_ROUNDS)
        branch._theta, branch.separate = theta, separate
        branch.Cuts.concat = staticmethod(concat)
        try:
            U, _, evaluations = branch.lift(*parts, 2 * K - N)
        finally:
            branch._theta, branch.separate = _ORIGINAL["theta"], _ORIGINAL["separate"]
            branch.Cuts.concat = staticmethod(_ORIGINAL["concat"])

    rounds, previous = [], None
    for i, event in enumerate(events):
        if event[0] != "separate":
            continue
        _, alpha, smooth = events[i - 1]              # the evaluation at the solution
        added = i + 1 < len(events) and events[i + 1][0] == "concat"
        stalled = (previous is not None and abs(previous - smooth) <= 1e-6 * abs(smooth)
                   and alpha < 1e-4)
        previous = smooth
        n = len(rounds) + 1
        giveup = n >= min_rounds and (n >= max_rounds or stalled)
        at_floor = alpha == alpha_min and not added
        rounds.append({"alpha": alpha, "smooth": smooth, "added": added,
                       "stalled": stalled, "giveup": giveup, "at_floor": at_floor,
                       "stops": giveup or at_floor})
    last = rounds[-1]
    stopped = ("floor" if last["at_floor"]
               else "rounds" if len(rounds) >= max_rounds else "stall")
    return {"N": N, "K": K, "U": U, "lower": 4.0 * engine.alpha / U,
            "evaluations": evaluations, "rounds": rounds,
            "final_alpha": last["alpha"], "stopped": stopped,
            "schedule": dict(values)}


def root_bounds(Y: np.ndarray, K: int, *, incumbent: float,
                interior: bool = True) -> dict:
    """The root's node bounds and their gaps to ``incumbent``."""
    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    N = Y.shape[1]
    F = tuple(range(N))
    parts = branch.node_parts(engine.R, (), (), F)
    s = 2 * K - N
    four_alpha = 4.0 * engine.alpha

    def lower(U: float) -> float:
        return four_alpha / U if np.isfinite(U) and U > 0 else 0.0

    out = {"N": N, "K": K, "incumbent": float(incumbent)}
    U_tr, _, _ = branch.trust_region(*parts, s)
    out["trust_region"] = lower(U_tr)
    t0 = time.process_time()
    U, _, out["evaluations"] = branch.lift(*parts, s)
    out["lift_seconds"] = time.process_time() - t0
    out["lift"] = lower(U)
    with schedule(**LONG):
        t0 = time.process_time()
        U, _, out["evaluations_long"] = branch.lift(*parts, s)
        out["lift_long_seconds"] = time.process_time() - t0
    out["lift_long"] = lower(U)
    if interior:
        from benchmarks.studies.syndes_global_bnb.bounds import lift_upper
        t0 = time.process_time()
        out["interior"] = lower(lift_upper(engine.R, (), (), F, K))
        out["interior_seconds"] = time.process_time() - t0
    for k in ("trust_region", "lift", "lift_long", "interior"):
        if k in out:
            out[f"{k}_gap"] = (out["incumbent"] - out[k]) / out["incumbent"]
    return out


def triangle_closure(Y: np.ndarray, K: int, *, gap: float = 1e-6,
                     per_round: int = 3_000, rounds: int = 8,
                     solver: str = "CLARABEL") -> dict:
    """The lift's relaxation at the root, solved to optimality by interior
    point with its triangle cuts added round by round.

    The library's lift reaches this relaxation from below through a
    regularised dual it stops early; this is the value it stops short of, for
    the triangles it would separate. The value is the solver's, not rebuilt
    from multipliers as the search's bounds are, since it settles a diagnosis
    and proves nothing: on these scaled problems the solver's relative
    accuracy is far finer than the gaps compared.
    """
    import cvxpy as cp

    from benchmarks.studies.syndes_global_bnb.bounds import _lift_problem

    problem = TwoWayProblem.from_panel(Y)
    engine = BoundEngine.from_problem(problem)
    N = Y.shape[1]
    c, b, A = branch.node_parts(engine.R, (), (), tuple(range(N)))
    s = 2 * K - N
    unit = max(abs(c), float(np.abs(b).max(initial=0.0)), float(np.abs(A).max()))
    prob, eq, rlt = _lift_problem(c / unit, b / unit, A / unit, s)
    X = prob.variables()[0]
    signs = branch._SIGNS
    cuts = []
    t0 = time.process_time()
    done = 0
    violated = 0
    value = np.nan
    for done in range(1, rounds + 1):
        value = cp.Problem(prob.objective, prob.constraints + cuts).solve(solver=solver)
        found = branch.separate(np.asarray(X.value), gap=gap, limit=per_round)
        violated = len(found)
        if not violated:
            break
        sg = signs[found.t - 1]
        cuts.append(1 + cp.multiply(sg[:, 0], X[found.i, found.j])
                    + cp.multiply(sg[:, 1], X[found.i, found.k])
                    + cp.multiply(sg[:, 2], X[found.j, found.k]) >= 0)
    U = unit * float(value)       # the last solve's: with violated > 0, before its cuts
    return {"N": N, "K": K, "lower": 4.0 * engine.alpha / U, "rounds": done,
            "violated": violated, "cuts": int(sum(c_.size for c_ in cuts)),
            "seconds": time.process_time() - t0}

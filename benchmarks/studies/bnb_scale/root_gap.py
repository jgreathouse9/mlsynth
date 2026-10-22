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
_DEFAULTS = {name: getattr(branch, name) for name in LONG}


@contextmanager
def schedule(**values):
    """The lift's stopping constants replaced for the length of a call."""
    for name, value in values.items():
        setattr(branch, name, value)
    try:
        yield
    finally:
        for name, value in _DEFAULTS.items():
            setattr(branch, name, value)


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

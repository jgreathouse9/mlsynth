"""A greedy incumbent for MAREX's base design, and its certified value.

The base design is Abadie and Zhao's program (7), MAREX's
``design="standard"``. Naming the treated set ``S`` of a cluster splits it into
two simplex fits that share nothing,

    f(S) = g(S) + g(S^c),

with ``g(U)`` the least squared distance from the cluster's target (the mean of
its markets) to the convex hull of the markets in ``U``. Without donor
exclusions or penalties the control fit is the same fit over the complement,
which is what makes the program symmetric in ``S`` and ``S^c``.

A set is admissible when its treated fit gives every market in it weight above
``WEIGHT_FLOOR`` and its complement is not empty: Abadie and Zhao count the
nonzero treated weights, so a set whose fit leaves a market at zero treats
fewer markets than it holds (:func:`~mlsynth.utils.marex_helpers.formulation.carries_weight`).

The greedy. :func:`forward` grows ``S`` one market at a time, adding the market
whose admissible extension has the least objective, up to the largest size
searched, and keeps the best admissible set it meets at a size searched.
:func:`swap` then exchanges one treated market for one control market, taking
the best improving exchange among admissible sets, until none improves. Both
evaluate ``f`` exactly with the library's simplex solver; :class:`Oracle`
caches the evaluations and counts them, two fits each.

The bound. Any admissible design bounds the optimum from above, ``f* <= f(S)``,
and :func:`bracket` evaluates ``f(S)`` exactly at the fitted weights
(:meth:`~mlsynth.utils.marex_helpers.certify.ExactData.interval`), so its upper
end is a certified upper bound on ``f*`` whatever the rounding in the fits. The
fitted weights are clipped and rescaled to the simplex exactly before they are
evaluated, so the bound needs no assumption about the solver's accuracy.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mlsynth.utils.marex_helpers.certify import ExactData
from mlsynth.utils.marex_helpers.exact import ClusterDesign, side_fit
from mlsynth.utils.marex_helpers.formulation import carries_weight


class Oracle:
    """``f(S)`` with both fits, cached by treated set."""

    def __init__(self, B: np.ndarray, A: np.ndarray):
        self.B, self.A, self.n = B, A, B.shape[1]
        self.cache: Dict[Tuple[int, ...], tuple] = {}

    def __call__(self, S: Sequence[int]) -> tuple:
        """``(f, admissible, w, control, v)`` for the treated set ``S``."""
        S = tuple(sorted(int(j) for j in S))
        if S not in self.cache:
            g, w = side_fit(self.B, self.A, list(S))
            C = tuple(j for j in range(self.n) if j not in S)
            h, v = side_fit(self.B, self.A, list(C)) if C else (np.inf, None)
            self.cache[S] = (g + h, carries_weight(w) and bool(C), w, C, v)
        return self.cache[S]

    @property
    def evaluated(self) -> int:
        """Distinct treated sets evaluated, two simplex fits each."""
        return len(self.cache)


def _key(oracle: Oracle, S: Tuple[int, ...]):
    return (oracle(S)[0], S)


def forward(oracle: Oracle, sizes: Sequence[int]
            ) -> Tuple[Optional[Tuple[int, ...]], Tuple[int, ...]]:
    """Grow ``S`` by the market whose admissible extension scores least.

    When no extension is admissible the step takes the best extension of any
    kind, so the path continues. Returns the best admissible set met at a size
    in ``sizes`` (``None`` if none was) and the last set reached.
    """
    sizes = set(int(m) for m in sizes)
    S: Tuple[int, ...] = ()
    best = None
    for k in range(1, max(sizes) + 1):
        options = [tuple(sorted(S + (j,))) for j in range(oracle.n) if j not in S]
        admissible = [T for T in options if oracle(T)[1]]
        S = min(admissible or options, key=lambda T: _key(oracle, T))
        if k in sizes and oracle(S)[1] and (
                best is None or _key(oracle, S) < _key(oracle, best)):
            best = S
    return best, S


def swap(oracle: Oracle, S: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Best-improvement exchange of a treated market for a control market.

    Moves only among admissible sets and stops when no exchange lowers ``f``.
    Starts from an inadmissible ``S`` too, and then takes the best admissible
    neighbour first. Returns ``None`` when ``S`` is inadmissible and has no
    admissible neighbour.
    """
    while True:
        current = oracle(S)[0] if oracle(S)[1] else np.inf
        moves = [tuple(sorted((set(S) - {i}) | {j}))
                 for i in S for j in range(oracle.n) if j not in S]
        admissible = [T for T in moves if oracle(T)[1]]
        if not admissible:
            return S if oracle(S)[1] else None
        T = min(admissible, key=lambda T: _key(oracle, T))
        if not oracle(T)[0] < current:
            return S
        S = T


def incumbent(B: np.ndarray, A: np.ndarray, sizes: Sequence[int]
              ) -> Tuple[Optional[ClusterDesign], float, int]:
    """The forward pass, then exchanges: the greedy incumbent of one cluster.

    Returns the design (``None`` when neither pass finds an admissible set),
    the forward pass's own value (``inf`` when it found none) and the number of
    treated sets evaluated, two simplex fits each.
    """
    oracle = Oracle(B, A)
    best, last = forward(oracle, sizes)
    forward_value = oracle(best)[0] if best is not None else np.inf
    S = swap(oracle, best if best is not None else last)
    if S is None:
        return None, forward_value, oracle.evaluated
    value, _, w, C, v = oracle(S)
    return (ClusterDesign(value=value, treated=S, control=C, w=w, v=v),
            forward_value, oracle.evaluated)


def as_design(B: np.ndarray, A: np.ndarray, S: Sequence[int]) -> ClusterDesign:
    """The design of the treated set ``S``, fitted."""
    value, _, w, C, v = Oracle(B, A)(S)
    return ClusterDesign(value=value, treated=tuple(sorted(int(j) for j in S)),
                         control=C, w=w, v=v)


def bracket(B: np.ndarray, A: np.ndarray, design: ClusterDesign
            ) -> Tuple[Fraction, Fraction]:
    """Exact lower and upper bounds on the design's objective ``f(S)``.

    The upper end is the objective at the fitted weights, exactly, so it is at
    least ``f(S)`` and hence at least the optimum ``f*``.
    """
    data = ExactData(B, A)
    lo_g, hi_g = data.interval(list(design.treated), design.w)
    lo_h, hi_h = data.interval(list(design.control), design.v)
    return lo_g + lo_h, hi_g + hi_h

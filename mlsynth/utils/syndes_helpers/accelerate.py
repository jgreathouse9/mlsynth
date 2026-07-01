"""Large-N two-way SYNDES accelerator: warm start + valid SDP objective cut.

For the two-way objective SCIP's own (McCormick) dual bound is very loose, so the
exact MIP times out even at modest N -- it keeps branching to *prove* an incumbent
that is already near-optimal. This module supplies the two levers SCIP lacks and
feeds them to :func:`mlsynth.utils.miqp_accel.solve_warm_cut`:

* a deterministic *warm start* -- the treated tuple LEXSCM's fast search
  recommends on the pre-period Gram matrix (the same primal seed MAREX uses); and
* a valid *objective lower-bound cut* -- the SDP/moment bound (Shor level-1) that
  :mod:`mlsynth.utils.syndes_helpers.certificate` validates, margined below its
  value to stay valid under the SDP solver's first-order tolerance.

The cut lifts SCIP's dual bound to the SDP bound, so the ordinary ``gap_limit``
certifies against a tight bound and SCIP stops early with a validated gap instead
of timing out. The returned design is certified-near-optimal, not proven-optimal;
this is an mlsynth addition, not part of Doudchenko et al. (2021).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .certificate import _sdp_moment_bound_two_way
from .optimization import estimate_lambda
from ..fast_scm_helpers.lexsearch import select_treated_designs


def warm_treated_vector(Y: np.ndarray, K: int, *, random_state: int = 0) -> np.ndarray:
    """Deterministic 0/1 warm start: LEXSCM's recommended treated ``K``-tuple.

    Runs LEXSCM's fast search (exact enumeration or seeded multi-start local
    search, depending on ``C(N, K)``) on the pre-period Gram matrix ``Y'Y`` and
    returns the top tuple as a length-``N`` indicator vector -- a good, fully
    deterministic initial incumbent for the two-way MIP.
    """
    N = int(Y.shape[1])
    G = np.asarray(Y, dtype=float).T @ np.asarray(Y, dtype=float)
    res = select_treated_designs(G, list(range(N)), int(K), top_K=1,
                                 random_state=random_state)
    designs = res.get("top_designs") or []
    D = np.zeros(N, dtype=float)
    if designs:
        D[np.asarray(designs[0].indices, dtype=int)] = 1.0
    return D


def two_way_accel_inputs(
    Y: np.ndarray,
    K: int,
    lam: Optional[float] = None,
    *,
    margin: float = 0.01,
    random_state: int = 0,
) -> Tuple[np.ndarray, float]:
    """``(warm_D, L_safe)`` for an accelerated two-way solve.

    ``L_safe = L * (1 - margin)`` where ``L`` is the SDP/moment lower bound; the
    margin keeps the cut strictly below the bound so it stays valid under the SDP
    solver's (first-order) tolerance. ``warm_D`` is the LEXSCM warm start.
    """
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)
    warm_D = warm_treated_vector(Y, K, random_state=random_state)
    L = _sdp_moment_bound_two_way(np.asarray(Y, dtype=float), int(K), lam_value)
    return warm_D, float(L) * (1.0 - float(margin))

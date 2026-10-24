"""Shared designs for the tests of VanillaSC's exact backend (``mlsynth.utils.bilevel.exact``).

Two sources:

- seeded factor-model panels, with the treated unit pushed outside the donor hull
  (``random_instance``) or pulled inside it (``random_interior_instance``);
- the authors' answers on designs of both kinds, captured from their code at the
  pinned commits (lychemon/synth-AS-replication 791a3865, synthx_package a10a28c5) by
  ``benchmarks/reference/vanillasc_exact/capture_designs.py`` into
  ``fixtures/exact/designs_exterior.json`` and ``designs_interior.json``, with inputs at
  full precision, so the tests that read them run offline.
"""
from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
from scipy.optimize import linprog

from mlsynth.utils.bilevel.exact.hull import outside_hull

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "exact"


def _cases(name):
    return {c["name"]: c for c in json.loads((FIXTURES / name).read_text())["cases"]}


EXTERIOR = _cases("designs_exterior.json")
INTERIOR = _cases("designs_interior.json")


def inputs(case):
    """(X1, X0, Z1, Z0) of a captured case, as float arrays."""
    return tuple(np.array(case[k], float) for k in ("X1", "X0", "Z1", "Z0"))


def factor_panel(seed, K, J, L, factors, lead):
    """X1, X0, Z1, Z0 from a factor model whose treated loadings are scaled by ``lead``."""
    rng = np.random.default_rng(seed)
    lam = rng.normal(size=(J + 1, factors))
    lam[0] *= lead
    f = rng.normal(size=(L + 6, factors)).cumsum(axis=0) * 0.3 + rng.normal(size=factors)
    Y = lam @ f.T + 0.3 * rng.normal(size=(J + 1, L + 6))
    C = lam @ rng.normal(size=(factors, K - 2)) + 0.5 * rng.normal(size=(J + 1, K - 2))
    X = np.column_stack([C, Y[:, 1], Y[:, 4]]).T
    X = X / X.std(axis=1, ddof=1)[:, None]
    Z = Y[:, 6:].T
    return X[:, 0], X[:, 1:], Z[:, 0], Z[:, 1:]


def random_instance(seed, K, J, L=12, factors=3):
    """A factor-model panel with the treated unit pushed outward; None if X1 is in the hull."""
    X1, X0, Z1, Z0 = factor_panel(seed, K, J, L, factors, 1.3)
    return (X1, X0, Z1, Z0) if outside_hull(X1, X0) else None


def random_interior_instance(seed, K, J, L=12, factors=3):
    """The same panel with the treated unit pulled toward the donors; None unless X1 is in the hull."""
    X1, X0, Z1, Z0 = factor_panel(seed, K, J, L, factors, 0.5)
    return None if outside_hull(X1, X0) else (X1, X0, Z1, Z0)


def exterior_design(seed, K, J):
    """The first exterior design from ``seed`` on."""
    while (inst := random_instance(seed, K, J)) is None:
        seed += 1
    return inst


def interior_design(seed, K, J):
    """The first interior design from ``seed`` on."""
    while (inst := random_interior_instance(seed, K, J)) is None:
        seed += 1
    return inst


def standardise(X1, X0):
    """Predictors scaled to unit standard deviation across treated and donors (ddof=1)."""
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    return X1 / sd, X0 / sd[:, None]


def admissible(X1, X0):
    """Every set of K + 1 donors whose simplex holds X1 strictly, with X1's barycentric weights."""
    K, J = X0.shape
    out = {}
    for A in itertools.combinations(range(J), K + 1):
        M = np.vstack([X0[:, A], np.ones(K + 1)])
        try:
            w = np.linalg.solve(M, np.append(X1, 1.0))
        except np.linalg.LinAlgError:          # pragma: no cover - general position holds on these designs
            continue
        if w.min() > 1e-9:
            out[A] = w
    return out


def selection(S1, S0, V):
    """The exact match the vanishing penalty selects under V: the vertex of F where
    sum_j w_j X_j'V X_j is least (paper, eq. 16), one LP."""
    J = S0.shape[1]
    res = linprog(np.einsum("kj,kl,lj->j", S0, V, S0), A_eq=np.vstack([S0, np.ones(J)]),
                  b_eq=np.append(S1, 1.0), bounds=(0, None), method="highs")
    return res.x

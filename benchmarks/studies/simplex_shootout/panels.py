"""The panel families a general-purpose simplex solver has to serve.

Every family is here because it separates the candidates differently, and the
support size is what does the separating: the cost of a seeded active set tracks
how far its guess sits from the support, and the cost of a projection-style
method tracks how many iterations the geometry needs.

    classic      Basque, German reunification, Proposition 99 -- outcome-only,
                 the shape practitioners actually run. Supports 3 to 7.
    factor       r = 3 common factors, wide pools. Supports 8 to 14.
    gaussian     18 random wide shapes, m 5 to 179, J from m+3 to 377. Supports
                 6 to 127, and the family the first seed budget was wrongly
                 tuned on alone.
    montecarlo   the AUGDID study's own cells, 10 to 40 pre-periods against 10
                 to 160 donors. Short and wide.
    ridged       SDID's two programs, with sqrt(ridge) I stacked beneath. A
                 ridge exists to spread weight, so the optima are dense -- 65 of
                 90 donors on one -- which is the case a small seed serves worst.
    degenerate   exact-fit and duplicated-donor designs, where the minimiser is
                 a face and which point comes back is a property of the method
                 and not of the data. No candidate may be scored on weights here.
"""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd

from mlsynth.utils.datautils import dataprep

BASEDATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "basedata")


class Panel(NamedTuple):
    family: str
    name: str
    B: np.ndarray          # (m, J) design
    A: np.ndarray          # (m,) target


_CLASSIC = [
    ("basque", "basque_data.csv", "regionname", "year", "gdpcap",
     "Basque Country (Pais Vasco)", 1975),
    ("germany", "german_reunification.csv", "country", "year", "gdp",
     "West Germany", 1990),
    ("prop99", "smoking_data.csv", "state", "year", "cigsale",
     "California", 1989),
]


def _classic() -> List[Panel]:
    out = []
    for name, f, unit, tcol, y, treated, t0 in _CLASSIC:
        path = os.path.join(BASEDATA, f)
        df = pd.read_csv(path)
        df["treat"] = ((df[unit] == treated) & (df[tcol] >= t0)).astype(int)
        prep = dataprep(df, unit, tcol, y, "treat")
        T0 = int(prep["pre_periods"])
        out.append(Panel("classic", name,
                         np.ascontiguousarray(
                             np.asarray(prep["donor_matrix"], float)[:T0]),
                         np.ascontiguousarray(
                             np.asarray(prep["y"], float).ravel()[:T0])))
    return out


def _factor(rng: np.random.Generator) -> List[Panel]:
    out = []
    for J in (20, 40, 80, 160, 320):
        F = rng.normal(size=(30, 3))
        L = np.abs(rng.normal(size=(3, J)))
        B = F @ L + 0.1 * rng.normal(size=(30, J))
        A = B[:, : min(7, J)].mean(axis=1) + 0.1 * rng.normal(size=30)
        out.append(Panel("factor", f"factor{J}",
                         np.ascontiguousarray(B), np.ascontiguousarray(A)))
    return out


def _gaussian(rng: np.random.Generator) -> List[Panel]:
    out = []
    for i in range(18):
        m = int(rng.integers(5, 180))
        J = int(rng.integers(m + 3, 378))
        out.append(Panel("gaussian", f"g{i}:{m}x{J}",
                         np.ascontiguousarray(rng.normal(size=(m, J))),
                         np.ascontiguousarray(rng.normal(size=m))))
    return out


def _montecarlo(rng: np.random.Generator) -> List[Panel]:
    out = []
    for T1, J in ((10, 10), (10, 40), (10, 80), (10, 160), (20, 160), (40, 160)):
        F = rng.normal(size=(T1, 3))
        L = rng.uniform(0.5, 1.5, (3, J))
        B = 1.0 + F @ L + rng.standard_normal((T1, J))
        A = 1.0 + F @ np.ones(3) + rng.standard_normal(T1)
        out.append(Panel("montecarlo", f"mc{T1}x{J}",
                         np.ascontiguousarray(B), np.ascontiguousarray(A)))
    return out


def _ridged(rng: np.random.Generator) -> List[Panel]:
    """SDID's shape: the design with ``sqrt(ridge) I`` stacked beneath it.

    The augmentation is what makes the optimum dense, and it is also why these
    designs are always at least as tall as they are wide.
    """
    out = []
    for m0, J, ridge in ((30, 90, 0.5), (60, 30, 0.25), (20, 60, 1.0)):
        F = np.cumsum(rng.standard_normal((m0, 3)) * 0.3, axis=0)
        L = rng.uniform(0.2, 1.2, (3, J))
        B0 = F @ L + 0.4 * rng.standard_normal((m0, J)) + 10.0
        A0 = B0.mean(axis=1) + 0.4 * rng.standard_normal(m0)
        root = float(np.sqrt(ridge))
        B = np.vstack([B0, root * np.eye(J)])
        A = np.concatenate([A0, np.full(J, root / J)])
        out.append(Panel("ridged", f"ridge{m0}x{J}",
                         np.ascontiguousarray(B), np.ascontiguousarray(A)))
    return out


def _degenerate(rng: np.random.Generator) -> List[Panel]:
    out = []
    # (1) the treated unit is exactly a convex combination of the donors, so the
    #     fit is exact and the argmin is a face. Becker-Kloessner's 0 in H.
    B = rng.normal(size=(8, 20))
    w = np.zeros(20); w[[2, 5, 11]] = (0.5, 0.3, 0.2)
    out.append(Panel("degenerate", "exactfit8x20",
                     np.ascontiguousarray(B), np.ascontiguousarray(B @ w)))
    # (2) duplicated donors in the support: weight trades between the twins at no
    #     cost, so the weights are not identified while the fit is.
    B = rng.normal(size=(25, 12)); B[:, 1] = B[:, 0]; B[:, 7] = B[:, 6]
    A = B[:, [0, 6, 9]].mean(axis=1) + 0.01 * rng.normal(size=25)
    out.append(Panel("degenerate", "twins25x12",
                     np.ascontiguousarray(B), np.ascontiguousarray(A)))
    # (3) one matching row against many donors: a hyperplane of minimisers.
    B = np.arange(1.0, 13.0).reshape(1, 12)
    out.append(Panel("degenerate", "onerow1x12",
                     np.ascontiguousarray(B),
                     np.ascontiguousarray(np.array([B.mean()]))))
    # (4) rank-deficient wide design, the shape the cold path meets most often.
    Bs = rng.normal(size=(7, 40))
    out.append(Panel("degenerate", "rankdef7x40",
                     np.ascontiguousarray(Bs),
                     np.ascontiguousarray(rng.normal(size=7))))
    return out


def all_panels() -> List[Panel]:
    return (_classic()
            + _factor(np.random.default_rng(0))
            + _gaussian(np.random.default_rng(1))
            + _montecarlo(np.random.default_rng(7))
            + _ridged(np.random.default_rng(11))
            + _degenerate(np.random.default_rng(23)))


def by_family() -> Dict[str, List[Panel]]:
    out: Dict[str, List[Panel]] = {}
    for p in all_panels():
        out.setdefault(p.family, []).append(p)
    return out

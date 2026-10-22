"""Graph-SC, demonstrated before it is built: Prop 99 (Path A) and the simulation (Path B).

Run::

    python -m benchmarks.studies.graphsc_replicate.run

Results are written to ``results/graphsc.json``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import oracle as G

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results" / "graphsc.json"


def prop99():
    """``(y, X, T0)`` for California and the 38 donors, through ``dataprep``."""
    from mlsynth.utils.datautils import dataprep

    df = pd.read_csv(ROOT / "basedata" / "smoking_data.csv")
    df["treat"] = df["Proposition 99"].astype(int)
    p = dataprep(df, "state", "year", "cigsale", "treat")
    return np.asarray(p["y"], float), np.asarray(p["donor_matrix"], float), int(p["pre_periods"])


#: Table 1's three arms at the paper's settings (Appendix B.2-B.3).
ARMS = {"Sparse-only": (0.05, 0.0, 50.0), "Graph-only": (0.0, 10.0, 50.0), "Graph-SC": (0.05, 10.0, 50.0)}
#: Table 1 as printed: pre-RMSPE, post-RMSPE, mean post gap, active donors, entropy.
TABLE1 = {"SC": (1.6564, 20.6056, 19.5136, 6, 1.4752),
          "Sparse-only": (0.0058, 13.0945, 12.3503, 20, 2.5791),
          "Graph-only": (0.0663, 18.4235, 17.3714, 38, 3.3940),
          "Graph-SC": (0.0893, 19.8198, 18.6245, 38, 3.4112)}


def table1_rows(k: int = 5):
    """Table 1's arms on Prop 99 in both readings of eq. (2), on the unstandardized panel.

    ``signed``: ``w`` in ``R^J`` (Definition 3.4); ``nonneg``: ``w >= 0``
    (Algorithm 2). The graph is the k = 5 Euclidean graph on the donors'
    pre-period sales (Appendix B.3).
    """
    y, X, T0 = prop99()
    L, _ = G.knn_laplacian(X[:T0].T, k)
    out = {"signed": {}, "nonneg": {}}
    for arm, (l1, lg, ls) in ARMS.items():
        for reading, solve in (("signed", G.solve_signed), ("nonneg", G.solve_exact)):
            w = solve(y[:T0], X[:T0], L, l1, lg, ls)
            out[reading][arm] = G.metrics(y, X @ w, T0, w)
    w = G.simplex_sc(y[:T0], X[:T0])
    out["SC"] = G.metrics(y, X @ w, T0, w)
    return out

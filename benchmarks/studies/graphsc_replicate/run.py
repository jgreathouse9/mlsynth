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
          "Graph-SC": (0.0893, 19.8918, 18.6245, 38, 3.4112)}


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


# ---------------------------------------------------------------- Path B


def _rep(args):
    import warnings

    from . import simulate as S

    rep, signed = args
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always", RuntimeWarning)
        out = S.one_rep(rep, signed)
    out["solver_warnings"] = sum(issubclass(w.category, RuntimeWarning) for w in log)
    return out


def _sweep(rep):
    from . import simulate as S

    return S.gate_sweep(rep)


def _ratio(num, den, rng, B: int = 2000):
    """Paired ratio of mean squared errors with a percentile bootstrap over replications."""
    n = len(num)
    idx = rng.integers(0, n, size=(B, n))
    boots = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return [float(num.mean() / den.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def path_b(reps: int = 200, workers: int = 4):
    from multiprocessing import Pool

    from . import simulate as S

    with Pool(workers) as pool:
        runs = {r: pool.map(_rep, [(i, r == "signed") for i in range(reps)]) for r in ("signed", "nonneg")}
        sweep = pool.map(_sweep, range(reps))
    rng = np.random.default_rng(20261022)
    out = {"reps": reps, "tau": S.TAU, "readings": {}}
    for reading, rows in runs.items():
        se = {a: (np.array([r["att"][a] for r in rows]) - S.TAU) ** 2 for a in S.ARM_NAMES}
        arms = {}
        for a in S.ARM_NAMES:
            att = np.array([r["att"][a] for r in rows])
            arms[a] = dict(att_mse=float(se[a].mean()), bias2=float((att.mean() - S.TAU) ** 2),
                           variance=float(att.var(ddof=1)),
                           pre_rmspe=float(np.mean([r["pre_rmspe"][a] for r in rows])),
                           post_rmspe_vs_noiseless=float(np.mean([r["post_rmspe_true"][a] for r in rows])),
                           active=float(np.mean([r["active"][a] for r in rows])))
        gate = {"learned / random graph": _ratio(se["Graph-SC"], se["Graph-SC random graph"], rng),
                "learned / ridge": _ratio(se["Graph-SC"], se["Ridge (same lambda, no graph)"], rng),
                "learned / SC": _ratio(se["Graph-SC"], se["SC"], rng),
                "oracle / learned": _ratio(se["Graph-SC oracle graph"], se["Graph-SC"], rng)}
        out["readings"][reading] = dict(arms=arms, gate=gate,
                                        solver_warnings=int(sum(r["solver_warnings"] for r in rows)))
    keys = sweep[0].keys()
    out["gate_sweep_signed"] = {k: float(np.mean([(s[k] - S.TAU) ** 2 for s in sweep])) for k in keys}
    return out


def main():
    import json
    import platform

    res = dict(paper="arXiv:2610.24113v1", python=platform.python_version(),
               path_a=dict(paper=TABLE1, ours=table1_rows()), path_b=path_b())
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


if __name__ == "__main__":
    main()

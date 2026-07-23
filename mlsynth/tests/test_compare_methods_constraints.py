"""compare_methods routes geographic constraints to every requested method.

All design methods (SYNDES / LEXSCM / MAREX) honour the same constraint
vocabulary, so ``compare_methods(constraints=...)`` applies one constraint set
uniformly and the resulting candidate designs -- whichever method produced them
-- respect it. These tests pin that routing test-first.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.design_compare import compare_methods

_ALL = ("SYNDES", "LEXSCM", "MAREX")
# keep the MIQP / lexicographic budgets small so the three-method fits stay quick
_OPTS = dict(
    syndes_options={"time_limit": 3.0, "gap_limit": 0.2},
    lexscm_options={"top_K": 4, "top_P": 3, "n_sims": 30, "max_shortlist": 4},
    marex_options={"solver": "SCIP"},
)

# 6 units; cluster/pair grouping for the no-two-from-one-cluster rule; region
# strata for the coverage quota.
_CLUSTER = {f"u{i}": f"C{i // 2}" for i in range(6)}        # C0..C2, pairs
_REGION = {f"u{i}": ("R1" if i < 3 else "R2") for i in range(6)}


def _panel(N=6, T=18, n_post=5, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)); L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = [{"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
             "cluster": _CLUSTER[f"u{j}"], "region": _REGION[f"u{j}"],
             "post": int(t >= T - n_post)}
            for j in range(N) for t in range(T)]
    return pd.DataFrame(rows), n_post


def test_constraints_unknown_key_raises():
    df, n_post = _panel()
    with pytest.raises(ValueError, match="constraint"):
        compare_methods(df, outcome="Y", unitid="unit", time="time",
                        treated_size=2, n_post=n_post, methods=("SYNDES",),
                        constraints={"not_a_real_constraint": 1})


def test_cluster_col_routed_to_all_methods():
    df, n_post = _panel()
    cmp = compare_methods(
        df, outcome="Y", unitid="unit", time="time", treated_size=2,
        horizon=5, n_post=n_post, top_K=4, methods=_ALL,
        constraints={"cluster_col": "cluster"}, **_OPTS,
    )
    assert set(cmp.table["method"]) == set(_ALL)
    for _, row in cmp.table.iterrows():
        clusters = [_CLUSTER[str(u)] for u in row["treated"]]
        assert len(clusters) == len(set(clusters)), \
            f"{row['method']} treated two from one cluster: {row['treated']}"


def test_not_to_be_treated_routed_to_all_methods():
    df, n_post = _panel()
    cmp = compare_methods(
        df, outcome="Y", unitid="unit", time="time", treated_size=2,
        horizon=5, n_post=n_post, top_K=4, methods=_ALL,
        constraints={"not_to_be_treated": ["u0"]}, **_OPTS,
    )
    for _, row in cmp.table.iterrows():
        assert "u0" not in {str(u) for u in row["treated"]}, row["method"]


def test_min_per_stratum_routed_to_all_methods():
    df, n_post = _panel()
    cmp = compare_methods(
        df, outcome="Y", unitid="unit", time="time", treated_size=2,
        horizon=5, n_post=n_post, top_K=4, methods=_ALL,
        constraints={"stratum_col": "region", "min_per_stratum": 1}, **_OPTS,
    )
    for _, row in cmp.table.iterrows():
        regions = {_REGION[str(u)] for u in row["treated"]}
        assert regions == {"R1", "R2"}, f"{row['method']} missed a region: {row['treated']}"


def test_unconstrained_runs_all_four():
    df, n_post = _panel()
    cmp = compare_methods(
        df, outcome="Y", unitid="unit", time="time", treated_size=2,
        horizon=5, n_post=n_post, top_K=4, methods=_ALL, **_OPTS,
    )
    assert set(cmp.table["method"]) == set(_ALL)
    assert np.isfinite(cmp.table["fit_rmse"]).all()

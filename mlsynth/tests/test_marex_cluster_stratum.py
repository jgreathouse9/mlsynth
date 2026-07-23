"""MAREX: cluster_col (no-two-same-cluster) and stratum coverage quotas.

MAREX already honours forced/forbidden lists, adjacency, size bands and donor
exclusion (see ``test_marex_restrictions.py``). The two remaining items in the
SYNDES constraint vocabulary are:

* ``cluster_col`` -- at most one treated market per cluster value (a no-two-from-
  one-cluster rule, distinct from MAREX's ``cluster`` design grouping), and
* ``stratum_col`` + ``min_per_stratum`` / ``max_per_stratum`` -- a coverage quota
  on the treated set.

Both reduce to linear constraints on MAREX's binary ``z``; these tests pin the
behaviour and the config validation test-first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# 8 units, one design cluster (no `cluster` grouping). Two attributes:
#   region (stratum): u0-u3 -> R1, u4-u7 -> R2
#   pair  (cluster_col): u0,u1 -> P0; u2,u3 -> P1; u4,u5 -> P2; u6,u7 -> P3
_REGION = {f"u{i}": ("R1" if i < 4 else "R2") for i in range(8)}
_PAIR = {f"u{i}": f"P{i // 2}" for i in range(8)}


def _panel(N=8, T=14, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = []
    for j in range(N):
        u = f"u{j}"
        for t in range(T):
            rows.append({"unit": u, "time": t, "Y": float(Y[t, j]),
                         "region": _REGION[u], "pair": _PAIR[u]})
    return pd.DataFrame(rows)


_BASE = dict(outcome="Y", unitid="unit", time="time", T0=10,
             inference=False, display_graph=False, verbose=False)


def _treated(res):
    """The set of treated unit labels across all clusters."""
    out = set()
    for cd in res.clusters.values():
        out |= set(cd.unit_weight_map.get("Treated", {}))
    return {str(u) for u in out}


def test_cluster_col_no_two_from_one_cluster():
    res = MAREX({"df": _panel(), **_BASE, "m_eq": 2, "cluster_col": "pair"}).fit()
    treated = _treated(res)
    pairs = [_PAIR[u] for u in treated]
    assert len(pairs) == len(set(pairs))            # never two from one pair


def test_stratum_min_per_covers_every_region():
    res = MAREX({"df": _panel(), **_BASE, "m_eq": 2,
                 "stratum_col": "region", "min_per_stratum": 1}).fit()
    treated = _treated(res)
    assert {_REGION[u] for u in treated} == {"R1", "R2"}


def test_stratum_max_per_caps_region_count():
    res = MAREX({"df": _panel(), **_BASE, "m_eq": 2,
                 "stratum_col": "region", "max_per_stratum": 1}).fit()
    treated = _treated(res)
    regions = [_REGION[u] for u in treated]
    assert all(regions.count(r) <= 1 for r in set(regions))


def test_stratum_quota_requires_stratum_col():
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="stratum"):
        MAREX({"df": _panel(), **_BASE, "m_eq": 2, "min_per_stratum": 1})


def test_stratum_min_exceeds_max_raises():
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="stratum|max"):
        MAREX({"df": _panel(), **_BASE, "m_eq": 2, "stratum_col": "region",
               "min_per_stratum": 2, "max_per_stratum": 1})


def test_unconstrained_still_works():
    """No cluster_col / stratum_col -> behaviour unchanged (smoke)."""
    res = MAREX({"df": _panel(), **_BASE, "m_eq": 2}).fit()
    assert len(_treated(res)) == 2

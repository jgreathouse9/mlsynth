"""MAREX: ``relaxed=True`` is removed, and cluster sizes count markets.

Issue #1045. Two causes, each pinned at the link where it lives.

1. ``relaxed=True`` rounded a continuous relaxation of the design, and the
   orchestration handed the relaxed solver no treated-set restrictions. The
   config refused that combination for seven restriction fields and not for
   ``cluster_col`` or the stratum quotas, so those were accepted and dropped: a
   relaxed fit put two treated markets in one pair, or none in a region it was
   told to cover, and raised nothing. The refusal list was a hand-kept copy of
   the builder's inputs, and the drop site trusted it.

   The relaxed path is removed instead of patched. The exact backend reaches
   the MIQP's optimum faster than the relaxed solve runs (0.1 s against 0.4 s
   at 100 markets), and the rounded design's objective was 28 to 52 times the
   optimum on the same panels. A setting that is slower and worse has no
   regime to keep it for, and removing it leaves no list to fall out of step.

2. The per-cluster cardinality check measured a cluster by its panel rows
   (``groupby(cluster).size()``), so a cluster of four markets observed for 14
   periods read as 56. ``m_eq`` was compared with the largest cluster, which
   cannot catch a per-cluster requirement the smallest one fails, and the
   program keeps one control in every cluster (``sum v_k = 1`` with
   ``v <= 1 - z``), so the bound is ``n_k - 1``. An infeasible ``m_eq`` or
   ``m_min`` passed the config and failed after the solve as an estimation
   error, and with no ``cluster`` there was no check at all. ``m_max`` above a
   cluster's size is an upper bound the cluster meets trivially, and is
   accepted.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

N_UNITS = 8
_REGION = {f"u{i}": ("R1" if i < 4 else "R2") for i in range(N_UNITS)}
_PAIR = {f"u{i}": f"P{i // 2}" for i in range(N_UNITS)}


def _panel(N=N_UNITS, T=14, seed=0, groups=None):
    """Factor panel; ``groups`` maps unit index -> design cluster, if given."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = []
    for j in range(N):
        u = f"u{j}"
        for t in range(T):
            row = {"unit": u, "time": t, "Y": float(Y[t, j]),
                   "region": _REGION.get(u, "R1"), "pair": _PAIR.get(u, "P0")}
            if groups is not None:
                row["g"] = int(groups[j])
            rows.append(row)
    return pd.DataFrame(rows)


_BASE = dict(outcome="Y", unitid="unit", time="time", T0=10,
             inference=False, display_graph=False, verbose=False)


def _config(df=None, **kw):
    df = _panel() if df is None else df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MAREXConfig(df=df, **_BASE, **kw)


def _treated(res):
    out = set()
    for cd in res.clusters.values():
        out |= set(cd.unit_weight_map.get("Treated", {}))
    return {str(u) for u in out}


# The issue's three constraints, and the check that each was honoured.
ISSUE_CONSTRAINTS = {
    "cluster_col": (dict(cluster_col="pair"),
                    lambda t: len({_PAIR[u] for u in t}) == len(t)),
    "max_per_stratum": (dict(stratum_col="region", max_per_stratum=1),
                        lambda t: all([_REGION[u] for u in t].count(r) <= 1 for r in ("R1", "R2"))),
    "min_per_stratum": (dict(stratum_col="region", min_per_stratum=1),
                        lambda t: {_REGION[u] for u in t} == {"R1", "R2"}),
}


# ------------------------------------------------- cause 1: relaxed is removed
class TestRelaxedIsRemoved:
    def test_relaxed_true_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="relaxed=True"):
            _config(m_eq=2, relaxed=True)

    def test_refusal_points_to_the_exact_backend(self):
        with pytest.raises(MlsynthConfigError, match="backend='exact'"):
            _config(m_eq=2, relaxed=True)

    @pytest.mark.parametrize("name", sorted(ISSUE_CONSTRAINTS))
    def test_issue_configurations_are_refused_not_dropped(self, name):
        kw, _ = ISSUE_CONSTRAINTS[name]
        with pytest.raises(MlsynthConfigError, match="relaxed=True"):
            _config(m_eq=2, relaxed=True, **kw)

    def test_relaxed_false_is_still_accepted(self):
        # The benchmark cases and older scripts pass it explicitly.
        assert _config(m_eq=2, relaxed=False).relaxed is False

    def test_the_relaxed_solver_is_gone(self):
        import mlsynth.utils.marex_helpers as mh
        from mlsynth.utils.marex_helpers import optimization, orchestration

        for name in ("solve_design_relaxed", "post_hoc_discretize"):
            assert not hasattr(optimization, name)
            assert not hasattr(mh, name)
        assert "relaxed" not in inspect.signature(orchestration.solve_marex).parameters

    @settings(max_examples=30, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(design=st.sampled_from(["standard", "penalized", "unit_penalized", "weakly_targeted"]),
           backend=st.sampled_from(["mip", "exact"]),
           extra=st.sampled_from(sorted(ISSUE_CONSTRAINTS) + [None]))
    def test_property_relaxed_true_is_refused_everywhere(self, design, backend, extra):
        kw = dict(m_eq=2, design=design, backend=backend, relaxed=True)
        if extra is not None:
            kw.update(ISSUE_CONSTRAINTS[extra][0])
        with pytest.raises(MlsynthConfigError, match="relaxed=True"):
            _config(**kw)


@pytest.mark.parametrize("name", sorted(ISSUE_CONSTRAINTS))
@pytest.mark.parametrize("seed", range(3))
def test_issue_seeds_are_honoured_by_the_default_path(name, seed):
    """The seeds that broke under relaxed=True, on the path that remains."""
    kw, honoured = ISSUE_CONSTRAINTS[name]
    res = MAREX({"df": _panel(seed=seed), **_BASE, "m_eq": 2, **kw}).fit()
    treated = _treated(res)
    assert len(treated) == 2
    assert honoured(treated)


# ------------------------------------------------------ cause 2: cluster sizes
_UNEVEN = [0, 0, 0, 1, 1, 1, 1, 1]          # clusters of 3 and 5 markets


class TestClusterCardinality:
    @pytest.mark.parametrize("kw", [dict(m_eq=3), dict(m_min=3)])
    def test_requirement_above_smallest_cluster_less_one_is_refused(self, kw):
        with pytest.raises(MlsynthDataError, match="cluster"):
            _config(df=_panel(groups=_UNEVEN), cluster="g", **kw)

    def test_refusal_reports_markets_not_rows(self):
        with pytest.raises(MlsynthDataError) as err:
            _config(df=_panel(groups=_UNEVEN), cluster="g", m_eq=3)
        assert "3 markets" in str(err.value)
        assert "42" not in str(err.value)          # 3 markets x 14 periods

    def test_without_clusters_the_panel_is_one_cluster(self):
        with pytest.raises(MlsynthDataError, match="8 markets"):
            _config(m_eq=N_UNITS)

    def test_without_clusters_m_min_is_checked_too(self):
        with pytest.raises(MlsynthDataError, match="8 markets"):
            _config(m_min=N_UNITS)

    @pytest.mark.parametrize("kw", [dict(m_min=1, m_max=4), dict(m_min=1, m_max=5),
                                    dict(m_max=50)])
    def test_m_max_above_a_cluster_size_is_accepted(self, kw):
        _config(df=_panel(groups=_UNEVEN), cluster="g", **kw)

    @pytest.mark.parametrize("kw", [dict(m_eq=2), dict(m_min=2), dict(m_min=1, m_max=4)])
    def test_largest_accepted_requirement_fits(self, kw):
        res = MAREX({"df": _panel(groups=_UNEVEN), **_BASE, "cluster": "g", **kw}).fit()
        for cd in res.clusters.values():
            assert len(cd.unit_weight_map.get("Treated", {})) >= 1

    @settings(max_examples=60, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(sizes=st.lists(st.integers(2, 5), min_size=1, max_size=3),
           m=st.integers(1, 6), T=st.integers(12, 30),
           which=st.sampled_from(["m_eq", "m_min"]))
    def test_property_refused_iff_some_cluster_cannot_keep_a_control(self, sizes, m, T, which):
        """Acceptance depends on markets per cluster, and not on the panel length."""
        groups = np.repeat(np.arange(len(sizes)), sizes)
        df = _panel(N=len(groups), T=T, groups=groups)
        if m <= min(sizes) - 1:
            _config(df=df, cluster="g", **{which: m})
        else:
            with pytest.raises(MlsynthDataError):
                _config(df=df, cluster="g", **{which: m})

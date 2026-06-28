"""Up-front, itemised feasibility audit for GeoLift market selection.

When no candidate test region satisfies the design constraints, GeoLift must say
*which* constraint bound the search, in a uniform ``have vs need -> fix`` shape --
mirroring LEXSCM's :func:`audit_feasibility`. These tests pin that diagnostic
(both the standalone helper and its end-to-end wiring) test-first per the repo
contract.
"""

import re

import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.geolift_helpers.config import GeoLiftConfig
from mlsynth.utils.geolift_helpers.marketselect.orchestration import run_design
from mlsynth.utils.geolift_helpers.marketselect.helpers.constraints import (
    build_conflict_graph,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.feasibility import (
    audit_geolift_feasibility,
)


# === audit_geolift_feasibility: the standalone, itemised helper ===

def test_audit_feasible_is_noop():
    """A satisfiable design audits clean (no exception)."""
    audit_geolift_feasibility(
        ["a", "b", "c", "d"], 2,
        stratum_map={"a": "R1", "b": "R1", "c": "R2", "d": "R2"},
        max_per_stratum=1,
    )


def test_audit_candidate_pool_too_small():
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(["a", "b"], 3)
    msg = str(ei.value)
    assert "binding constraint" in msg
    assert "candidate pool" in msg
    assert "2" in msg and "3" in msg          # have 2, need 3


def test_audit_min_per_stratum_needs_more_than_treatment_size():
    """min_per_stratum=1 over 3 required strata needs >= 3 treated, but k=2."""
    sm = {"a": "R1", "b": "R2", "c": "R3", "d": "R1"}
    required = {"R1", "R2", "R3"}
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(
            ["a", "b", "c", "d"], 2, stratum_map=sm,
            min_per_stratum=1, required_strata=required,
        )
    msg = str(ei.value)
    assert "coverage" in msg
    assert ">= 3" in msg                       # the computed need
    assert "min_per_stratum" in msg


def test_audit_min_per_stratum_quota_arithmetic():
    """min_per_stratum=2 over 4 strata needs >= 8 -- the user's actual case."""
    sm = {f"u{i}": f"R{i % 4}" for i in range(8)}
    required = {f"R{i}" for i in range(4)}
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(
            list(sm), 3, stratum_map=sm, min_per_stratum=2, required_strata=required,
        )
    assert ">= 8" in str(ei.value)             # 2 per stratum x 4 strata


def test_audit_stratum_with_too_few_eligible():
    """A required stratum that cannot meet its own min is named explicitly."""
    sm = {"a": "R1", "b": "R1", "c": "R1", "d": "R2"}   # R2 has only 1 eligible
    required = {"R1", "R2"}
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(
            ["a", "b", "c", "d"], 3, stratum_map=sm,    # k=3 clears the pool guard
            min_per_stratum=2, required_strata=required,
        )
    msg = str(ei.value)
    assert "R2" in msg                         # the short stratum is named


def test_audit_max_per_stratum_caps_below_treatment_size():
    """max_per_stratum=1 over 2 strata allows at most 2 treated, but k=3."""
    sm = {"a": "R1", "b": "R1", "c": "R2", "d": "R2"}
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(
            ["a", "b", "c", "d"], 3, stratum_map=sm, max_per_stratum=1,
        )
    msg = str(ei.value)
    assert "max_per_stratum" in msg
    assert "at most 2" in msg


def test_audit_cluster_independent_set_too_small():
    """Three clusters -> largest conflict-free set is 3 < treatment_size=4."""
    units = [f"u{i}" for i in range(6)]
    g = build_conflict_graph(units, cluster_map={f"u{i}": f"C{i // 2}" for i in range(6)})
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(units, 4, conflict=g)
    msg = str(ei.value)
    assert "spillover" in msg or "cluster" in msg
    assert "3" in msg and "4" in msg           # largest 3 < need 4


def test_audit_reports_multiple_binding_constraints_together():
    """Two independent infeasibilities are reported in one error, itemised."""
    sm = {"a": "R1", "b": "R2", "c": "R3"}
    required = {"R1", "R2", "R3"}
    g = build_conflict_graph(["a", "b", "c"],
                             cluster_map={"a": "C0", "b": "C0", "c": "C0"})
    with pytest.raises(MlsynthConfigError) as ei:
        audit_geolift_feasibility(
            ["a", "b", "c"], 3, conflict=g, stratum_map=sm,
            min_per_stratum=2, required_strata=required,
        )
    msg = str(ei.value)
    # itemised: more than one "  - " bullet
    assert len(re.findall(r"\n\s*- ", msg)) >= 2


# === end-to-end: run_design surfaces the diagnostic, not the vague message ===

_REGION_OF = {"u0": "R1", "u1": "R1", "u2": "R2", "u3": "R2", "u4": "R3", "u5": "R3"}


def _stratified_long(seed=5, T=24):
    import numpy as np
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(6):
        u = f"u{i}"
        s = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": u, "time": t, "Y": float(s[t]),
                         "region": _REGION_OF[u]})
    return pd.DataFrame(rows)


def test_run_design_min_per_stratum_infeasible_names_the_constraint():
    """The old vague 'no candidate test region' message is replaced by an
    itemised diagnostic that names coverage and the quota arithmetic."""
    with pytest.raises(MlsynthConfigError) as ei:
        run_design(GeoLiftConfig(
            df=_stratified_long(), outcome="Y", unitid="unit", time="time",
            treatment_size=2, durations=[4], effect_sizes=[0.0],
            lookback_window=1, ns=20, seed=0,
            stratum_col="region", min_per_stratum=1))
    msg = str(ei.value)
    assert "binding constraint" in msg
    assert "coverage" in msg
    assert ">= 3" in msg                       # 1 per region x 3 regions


def _clustered_long(seed=3, T=24):
    import numpy as np
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(6):
        s = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(s[t]),
                         "cluster": f"C{i // 2}"})
    return pd.DataFrame(rows)


def test_run_design_infeasible_cluster_names_independent_set():
    with pytest.raises(MlsynthConfigError) as ei:
        run_design(GeoLiftConfig(
            df=_clustered_long(), outcome="Y", unitid="unit", time="time",
            treatment_size=4, durations=[4], effect_sizes=[0.0],
            lookback_window=1, ns=20, seed=0, cluster_col="cluster"))
    msg = str(ei.value)
    assert "binding constraint" in msg
    assert "spillover" in msg or "cluster" in msg
    assert "< treatment_size=4" in msg

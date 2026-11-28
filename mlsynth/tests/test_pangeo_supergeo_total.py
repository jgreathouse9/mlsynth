"""PANGEO with the supergeo paper's objective, pairs only (roadmap phase 0b-1).

``objective="supergeo_total"`` scores a split by the squared difference of the
halves' totals over the estimation window (Chen, Doudchenko, Jiang, Stein and
Ying, "Supergeo design: generalized matching for geographic experiments",
arXiv:2301.12044, the objective of their eq. 3 with sums, no level shift, no
trajectory). With pairs (Q = 1) the design is a matching, and under this
objective the matching that pairs neighbours in sorted order of the totals is
optimal, which gives the MIP an exact oracle.

Layers per agents/agents_tests.md: the score (unit and property), the MIP
against the oracle (property and fixed instances), the backend routing, the
configuration and estimation failures, and an end-to-end smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import mip as pmip
from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs, run_pangeo
from mlsynth.utils.pangeo_helpers.parallelism import best_split, enumerate_candidate_pairs

# The cross-version CI jobs do not install highspy; its cases run where it is.
HAS_HIGHSPY = pmip._highspy_available()
BACKENDS = ["cvxpy", "highspy"] if HAS_HIGHSPY else ["cvxpy"]
needs_highspy = pytest.mark.skipif(not HAS_HIGHSPY, reason="highspy is not installed")


def _long(Y: np.ndarray) -> pd.DataFrame:
    """A one-arm long panel from a units x periods matrix."""
    n, T = Y.shape
    return pd.DataFrame({
        "unit": np.repeat([f"u{i:02d}" for i in range(n)], T),
        "time": np.tile(np.arange(T), n),
        "sales": Y.ravel(),
        "arm": "A",
    })


def _design(Y, frac_E=0.7, backend="cvxpy"):
    inp = prepare_pangeo_inputs(_long(Y), "sales", "arm", "unit", "time")
    res = run_pangeo(inp, max_supergeo_size=1, fast=False,
                     objective="supergeo_total", frac_E=frac_E,
                     compute_power=False, mip_backend=backend)
    return inp, res


def _sorted_pairing_cost(z: np.ndarray) -> float:
    """The optimal pairing's cost: neighbours in sorted order."""
    s = np.sort(z)
    return float(((s[1::2] - s[0::2]) ** 2).sum())


def _pairs_of(res, inp):
    pos = {name: i for i, name in enumerate(inp.unit_names)}
    return sorted(tuple(sorted(pos[u] for u in p.treatment + p.control))
                  for p in res.arm_designs["A"].pairs)


def _totals(Y, res):
    return Y[:, :res.metadata["n_estimation"]].sum(axis=1)


# ----------------------------------------------------------------------
# Layer 1: the score
# ----------------------------------------------------------------------

class TestScore:
    def test_a_pair_scores_the_squared_difference_of_totals(self):
        Y = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 5.0]])
        score, a, b = best_split(np.array([0, 1]), Y, max_size=1,
                                 objective="supergeo_total")
        assert score == pytest.approx((6.0 - 9.0) ** 2)
        assert sorted(a + b) == [0, 1]

    def test_the_level_is_balanced_not_removed(self):
        """Two parallel paths a level apart are a perfect DiD pair and a poor
        supergeo pair: the paper's objective balances the level."""
        base = np.array([1.0, 3.0, 2.0, 5.0])
        Y = np.vstack([base, base + 2.0])
        ss, _, _ = best_split(np.array([0, 1]), Y, max_size=1, objective="ss_res")
        tot, _, _ = best_split(np.array([0, 1]), Y, max_size=1,
                               objective="supergeo_total")
        assert ss == pytest.approx(0.0, abs=1e-12)
        assert tot == pytest.approx((2.0 * 4) ** 2)

    def test_unit_weights_do_not_enter_the_score(self):
        Y = np.array([[1.0, 2.0], [4.0, 4.0]])
        w = np.array([10.0, 0.5])
        s1, _, _ = best_split(np.array([0, 1]), Y, max_size=1,
                              objective="supergeo_total")
        s2, _, _ = best_split(np.array([0, 1]), Y, max_size=1,
                              objective="supergeo_total", unit_weights=w)
        assert s1 == s2 == pytest.approx(25.0)

    @settings(max_examples=100, deadline=None)
    @given(seed=st.integers(0, 10**9), T=st.integers(1, 8),
           c=st.floats(-50, 50), k=st.floats(0.1, 10))
    def test_score_invariants(self, seed, T, c, k):
        """A pair's score is (z_i - z_j)^2: symmetric, unchanged by a common
        shift of every period, scaled by k^2 when the data are."""
        Y = np.random.default_rng(seed).normal(100, 20, (2, T))
        f = lambda M, m: best_split(m, M, max_size=1,
                                    objective="supergeo_total")[0]
        s = f(Y, np.array([0, 1]))
        z = Y.sum(axis=1)
        assert s == pytest.approx((z[0] - z[1]) ** 2, rel=1e-9, abs=1e-9)
        assert f(Y, np.array([1, 0])) == pytest.approx(s, rel=1e-9, abs=1e-9)
        assert f(Y + c, np.array([0, 1])) == pytest.approx(s, rel=1e-6, abs=1e-6)
        assert f(k * Y, np.array([0, 1])) == pytest.approx(k * k * s, rel=1e-9,
                                                           abs=1e-9)


# ----------------------------------------------------------------------
# Layer 3: the MIP against the sorted-pairing oracle
# ----------------------------------------------------------------------

# HiGHS stops at its default relative MIP gap, 1e-4; the base runs as shipped.
HIGHS_REL_GAP = 1e-4


class TestAgainstTheOracle:
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(0, 10**9), half=st.integers(1, 7),
           T=st.integers(2, 12), backend=st.sampled_from(BACKENDS))
    def test_the_mip_reaches_the_sorted_pairing(self, seed, half, T, backend):
        """Every unit is in exactly one pair, and the design's cost is the
        sorted pairing's, within HiGHS's default gap; the sorted pairing is
        optimal, so it is also a lower bound."""
        Y = np.random.default_rng(seed).lognormal(4, 0.6, (2 * half, T))
        inp, res = _design(Y, backend=backend)
        pairs = _pairs_of(res, inp)
        assert sorted(u for p in pairs for u in p) == list(range(2 * half))
        assert all(len(p) == 2 for p in pairs)
        z = _totals(Y, res)
        cost = float(sum((z[i] - z[j]) ** 2 for i, j in pairs))
        best = _sorted_pairing_cost(z)
        assert best - 1e-9 * (1 + best) <= cost <= best * (1 + HIGHS_REL_GAP) + 1e-9
        diag = res.metadata["solver_diagnostics"]["A"]
        assert diag["objective_value"] == pytest.approx(cost, rel=1e-6, abs=1e-6)

    def test_a_fixed_instance_gives_the_sorted_pairs(self):
        z = np.array([10.0, 31.0, 12.0, 50.0, 29.0, 47.0])
        Y = np.column_stack([z / 2, z / 2])
        inp, res = _design(Y, frac_E=0.99)
        assert res.metadata["n_estimation"] == 2
        assert _pairs_of(res, inp) == [(0, 2), (1, 4), (3, 5)]

    def test_only_the_estimation_window_is_balanced(self):
        """Totals over the estimation window decide the pairs; the held-out
        tail does not enter (here it would pair the units the other way)."""
        e = np.array([[10.0], [11.0], [20.0], [21.0]])
        tail = np.array([[0.0], [100.0], [100.0], [0.0]])
        Y = np.hstack([np.repeat(e, 7, axis=1), np.repeat(tail, 3, axis=1)])
        inp, res = _design(Y, frac_E=0.7)
        assert res.metadata["n_estimation"] == 7
        assert _pairs_of(res, inp) == [(0, 1), (2, 3)]

    def test_two_units_make_one_pair(self):
        inp, res = _design(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
        assert _pairs_of(res, inp) == [(0, 1)]

    def test_equal_totals_cost_nothing(self):
        Y = np.tile(np.array([2.0, 5.0, 1.0]), (4, 1))
        inp, res = _design(Y)
        assert res.metadata["solver_diagnostics"]["A"]["objective_value"] == \
            pytest.approx(0.0, abs=1e-9)

    def test_an_odd_arm_has_no_cover(self):
        """As shipped, PANGEO covers every unit, so pairs on an odd arm are
        infeasible and the MIP reports it."""
        Y = np.random.default_rng(0).normal(100, 10, (5, 6))
        with pytest.raises(MlsynthEstimationError):
            _design(Y)


# ----------------------------------------------------------------------
# The MIP backend
# ----------------------------------------------------------------------

class TestBackend:
    @pytest.mark.parametrize("backend, used", [
        ("cvxpy", "cvxpy"),
        pytest.param("highspy", "highspy", marks=needs_highspy)])
    def test_the_backend_asked_for_is_the_one_used(self, backend, used,
                                                   monkeypatch):
        calls = []
        for name in ("_solve_partition_cvxpy", "_solve_partition_highspy"):
            real = getattr(pmip, name)
            monkeypatch.setattr(pmip, name, (lambda real, name: lambda *a, **k:
                                             calls.append(name) or real(*a, **k))(real, name))
        Y = np.random.default_rng(1).normal(100, 10, (6, 8))
        inp, res = _design(Y, backend=backend)
        assert calls == [f"_solve_partition_{used}"]
        assert res.metadata["solver_diagnostics"]["A"]["backend"] == used
        assert res.metadata["mip_backend"] == backend

    def test_the_mips_auto_backend_is_highspy_where_installed(self):
        """With pairs, PANGEO's auto path is the closed form (test_pangeo_pairs.py); the MIP's own auto rule
        is pinned here, on solve_partition, where it lives."""
        Y = np.random.default_rng(1).normal(100, 10, (6, 8))
        idx = np.arange(6)
        cands = enumerate_candidate_pairs(idx, Y, 1, objective="supergeo_total")
        _, diag = pmip.solve_partition(cands, idx, return_diagnostics=True, backend="auto")
        assert diag["backend"] == ("highspy" if HAS_HIGHSPY else "cvxpy")

    def test_auto_falls_back_to_cvxpy_without_highspy(self, monkeypatch):
        monkeypatch.setattr(pmip, "_highspy_available", lambda: False)
        Y = np.random.default_rng(2).normal(100, 10, (4, 6))
        idx = np.arange(4)
        cands = enumerate_candidate_pairs(idx, Y, 1, objective="supergeo_total")
        _, diag = pmip.solve_partition(cands, idx, return_diagnostics=True, backend="auto")
        assert diag["backend"] == "cvxpy"

    def test_highspy_asked_for_but_missing_is_reported(self, monkeypatch):
        monkeypatch.setattr(pmip, "_highspy_available", lambda: False)
        with pytest.raises(MlsynthConfigError):
            _design(np.random.default_rng(2).normal(100, 10, (4, 6)),
                    backend="highspy")

    @needs_highspy
    def test_cvxpy_and_highspy_agree(self):
        Y = np.random.default_rng(3).lognormal(4, 0.5, (8, 10))
        _, a = _design(Y, backend="cvxpy")
        _, b = _design(Y, backend="highspy")
        va = a.metadata["solver_diagnostics"]["A"]["objective_value"]
        vb = b.metadata["solver_diagnostics"]["A"]["objective_value"]
        assert va == pytest.approx(vb, rel=HIGHS_REL_GAP)


# ----------------------------------------------------------------------
# Layer 4: configuration, failures, smoke
# ----------------------------------------------------------------------

def _cfg(df, **kw):
    base = dict(df=df, outcome="sales", arm="arm", unitid="unit", time="time",
                display_graphs=False, objective="supergeo_total",
                max_supergeo_size=1, fast=False)
    base.update(kw)
    return base


class TestConfig:
    def test_defaults_are_unchanged(self):
        f = PANGEOConfig.model_fields
        assert f["objective"].default == "ss_res"
        assert f["mip_backend"].default == "auto"
        assert f["fast"].default is None

    @pytest.mark.parametrize("override", [
        {"max_supergeo_size": 2},
        {"max_supergeo_size": None},
        {"compute_q_sweep": True},
        {"mip_backend": "gurobi"},
    ])
    def test_unsupported_combinations_are_rejected(self, override):
        df = _long(np.random.default_rng(0).normal(100, 10, (4, 6)))
        with pytest.raises(MlsynthConfigError):
            PANGEO(_cfg(df, **override))

    def test_run_pangeo_rejects_groups_beyond_pairs(self):
        inp = prepare_pangeo_inputs(
            _long(np.random.default_rng(0).normal(100, 10, (4, 6))),
            "sales", "arm", "unit", "time")
        with pytest.raises(MlsynthConfigError):
            run_pangeo(inp, max_supergeo_size=2,
                       objective="supergeo_total", compute_power=False)

    def test_run_pangeo_rejects_an_unknown_backend(self):
        inp = prepare_pangeo_inputs(
            _long(np.random.default_rng(0).normal(100, 10, (4, 6))),
            "sales", "arm", "unit", "time")
        with pytest.raises(MlsynthConfigError):
            run_pangeo(inp, max_supergeo_size=1, fast=False,
                       compute_power=False, mip_backend="gurobi")


def test_smoke_end_to_end():
    Y = np.random.default_rng(5).lognormal(4, 0.5, (10, 20))
    res = PANGEO(_cfg(_long(Y), mip_backend="cvxpy")).fit()
    d = res.arm_designs["A"]
    covered = [u for p in d.pairs for u in p.treatment + p.control]
    assert len(covered) == len(set(covered)) == 10
    assert all(len(p.treatment) == len(p.control) == 1 for p in d.pairs)
    assert res.metadata["objective"] == "supergeo_total"
    assert res.metadata["solver_diagnostics"]["A"]["backend"] == "cvxpy"
    assert np.isfinite(res.metadata["solver_diagnostics"]["A"]["objective_value"])


def test_the_backend_reaches_every_design_of_the_q_sweep(monkeypatch):
    """Automatic Q designs every feasible Q; each one uses the backend asked for."""
    used = []
    real = pmip.solve_partition
    monkeypatch.setattr("mlsynth.utils.pangeo_helpers.pipeline.solve_partition",
                        lambda *a, **k: used.append(k.get("backend")) or real(*a, **k))
    Y = np.random.default_rng(6).normal(100, 10, (6, 12))
    inp = prepare_pangeo_inputs(_long(Y), "sales", "arm", "unit", "time")
    res = run_pangeo(inp, max_supergeo_size=None, fast=False, compute_power=True,
                     mip_backend="cvxpy")
    assert len(used) >= 2 and set(used) == {"cvxpy"}
    assert res.metadata["mip_backend"] == "cvxpy"

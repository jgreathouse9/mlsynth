"""Tests for the large-N two-way SYNDES accelerator (SDP-cut + warm start).

The accelerator injects a valid SDP objective cut and a deterministic LEXSCM warm
start into the two-way MIP so the existing ``gap_limit`` certifies against a tight
dual bound (the SDP/moment bound) instead of SCIP's loose McCormick relaxation.

Policy (size-gated auto): it engages only for the two-way mode with an explicit
``K``, when the treated-tuple count ``C(N, K)`` exceeds ``accel_min_tuples`` and
``N <= certify_sdp_n_max``. Small problems solve exactly, unchanged. ``gap_limit``
is the certified-gap target (reused, not a new knob). These tests pin engagement,
the gating, and that engaging never changes a small instance's optimum.
"""
import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.config_models import SYNDESConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.accelerate import (
    two_way_accel_inputs,
    warm_treated_vector,
)
from mlsynth.utils.syndes_helpers.optimization import (
    solve_synthetic_design,
    estimate_lambda,
)
from mlsynth.utils.syndes_helpers.certificate import _sdp_moment_bound_two_way


def _panel(n_units=12, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    L = rng.standard_normal((3, n_units))
    Y = F @ L + 0.3 * rng.standard_normal((T, n_units))
    rows = []
    for j in range(n_units):
        for t in range(T):
            rows.append({"unit": f"u{j}", "time": t, "y": Y[t, j],
                         "post": int(t >= T - n_post)})
    return pd.DataFrame(rows)


class TestAccelInputs:
    def test_warm_vector_is_khot(self):
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((12, 20))
        w = warm_treated_vector(Y, K=4)
        assert w.shape == (20,)
        assert set(np.unique(w)).issubset({0.0, 1.0})
        assert int(w.sum()) == 4

    def test_warm_vector_deterministic(self):
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((12, 20))
        assert np.array_equal(warm_treated_vector(Y, 4), warm_treated_vector(Y, 4))

    def test_bound_is_margined_below_sdp(self):
        rng = np.random.default_rng(3)
        Y = rng.standard_normal((10, 12))
        lam = float(estimate_lambda(Y))
        L = _sdp_moment_bound_two_way(Y, K=3, lam=lam)
        _, L_safe = two_way_accel_inputs(Y, K=3, lam=lam, margin=0.01)
        assert 0.0 < L_safe < L
        assert L_safe == pytest.approx(L * 0.99, rel=1e-9)


class TestSolveSyntheticDesignAccel:
    def test_accel_matches_plain_optimum(self):
        """Injecting warm + a valid cut yields the same optimum as a plain solve
        on a small instance, and records accel diagnostics."""
        rng = np.random.default_rng(4)
        Y = rng.standard_normal((10, 8))
        lam = float(estimate_lambda(Y))
        ref = solve_synthetic_design(Y, K=2, mode="global_2way", lam=lam)
        warm, L_safe = two_way_accel_inputs(Y, K=2, lam=lam)
        acc = solve_synthetic_design(
            Y, K=2, mode="global_2way", lam=lam, gap_limit=1e-4,
            warm_start_D=warm, objective_lower_bound=L_safe,
        )
        assert acc.objective_value == pytest.approx(ref.objective_value, abs=5e-3)
        assert "accel" in acc.raw_results
        assert acc.raw_results["accel"]["cut_applied"]

    def test_accel_params_default_none_is_plain(self):
        """Omitting the accel params leaves the solve byte-identical to before."""
        rng = np.random.default_rng(5)
        Y = rng.standard_normal((10, 8))
        d = solve_synthetic_design(Y, K=2, mode="global_2way")
        assert "accel" not in d.raw_results


class TestEstimatorPolicy:
    def test_engages_for_large_tuple_count(self):
        """With a low tuple gate the two-way estimator engages the accelerator."""
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, mode="two_way_global", post_col="post",
                           accel_min_tuples=50, run_inference=False)
        res = SYNDES(cfg).fit()
        assert res.design.raw_results.get("accel") is not None
        assert int(res.design.assignment.sum()) == 3

    def test_skips_for_small_tuple_count(self):
        """Default gate: a tiny panel solves exactly, no accel."""
        df = _panel(n_units=8)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=2, mode="two_way_global", post_col="post",
                           run_inference=False)   # C(8,2)=28 << default 2000
        res = SYNDES(cfg).fit()
        assert res.design.raw_results.get("accel") is None

    def test_accelerate_false_disables(self):
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, mode="two_way_global", post_col="post",
                           accel_min_tuples=50, accelerate=False,
                           run_inference=False)
        res = SYNDES(cfg).fit()
        assert res.design.raw_results.get("accel") is None

    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_only_two_way(self, mode):
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, mode=mode, post_col="post",
                           accel_min_tuples=50, run_inference=False)
        res = SYNDES(cfg).fit()
        assert res.design.raw_results.get("accel") is None

    def test_requires_explicit_K(self):
        """K=None two-way cannot be accelerated (no cardinality for the SDP)."""
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=None, mode="two_way_global", post_col="post",
                           accel_min_tuples=1, run_inference=False)
        res = SYNDES(cfg).fit()
        assert res.design.raw_results.get("accel") is None

    def test_accel_still_respects_restrictions(self):
        """The cut stays valid under restrictions, and the accelerated design must
        still honor them even though the (unrestricted) warm start may not."""
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, mode="two_way_global", post_col="post",
                           accel_min_tuples=50, not_to_be_treated=["u0", "u1"],
                           run_inference=False)
        res = SYNDES(cfg).fit()
        treated = set(np.asarray(res.design.selected_unit_labels).tolist())
        assert "u0" not in treated and "u1" not in treated
        assert int(res.design.assignment.sum()) == 3


class TestConfig:
    def test_defaults(self):
        df = _panel()
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, mode="two_way_global", post_col="post")
        assert cfg.accelerate is True
        assert cfg.accel_min_tuples >= 1
        assert 0.0 < cfg.accel_safety_margin < 1.0

    def test_bad_margin_rejected(self):
        df = _panel()
        with pytest.raises((MlsynthConfigError, ValueError)):
            SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                         K=3, mode="two_way_global", post_col="post",
                         accel_safety_margin=1.5)

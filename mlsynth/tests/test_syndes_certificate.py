"""Tests for the SYNDES optimality certificate (mode-aware lower bound).

Contract: the returned lower bound is always valid (LB <= the design's
objective), it is a *tight* certificate for one-way (continuous QP) and in-range
two-way (SDP moment), and it is honestly flagged ``certified=False`` for per-unit
and out-of-range two-way. ``certify=True`` on the estimator attaches it; the
default leaves the result untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.optimization import solve_synthetic_design
from mlsynth.utils.syndes_helpers.certificate import (
    SYNDESCertificate, syndes_certificate,
)

INTERNAL = {"one_way": "global_equal_weights", "two_way": "global_2way",
            "per_unit": "per_unit"}


def _Y(N=12, T=8, K=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3)); Lam = rng.standard_normal((N, 3))
    return (Lam @ F.T).T + 0.3 * rng.standard_normal((T, N))


def _incumbent(Y, K, mode):
    return float(solve_synthetic_design(Y, K=K, mode=mode).objective_value)


# ----------------------------------------------------------------------
# Validity: the bound never exceeds the incumbent (it is a lower bound)
# ----------------------------------------------------------------------
class TestValidity:
    @pytest.mark.parametrize("mode", list(INTERNAL.values()))
    def test_lower_bound_below_incumbent(self, mode):
        Y = _Y(); K = 3
        inc = _incumbent(Y, K, mode)
        c = syndes_certificate(Y, K, mode, inc)
        assert c.lower_bound <= inc + 1e-6
        assert 0.0 <= c.optimality_gap
        assert isinstance(c, SYNDESCertificate)

    def test_gap_formula(self):
        Y = _Y(); K = 3; mode = "global_2way"
        inc = _incumbent(Y, K, mode)
        c = syndes_certificate(Y, K, mode, inc)
        assert c.optimality_gap == pytest.approx(
            max(0.0, (inc - c.lower_bound) / abs(inc)), rel=1e-9)


# ----------------------------------------------------------------------
# Mode-aware certified flag and method
# ----------------------------------------------------------------------
class TestModes:
    def test_one_way_certified_continuous(self):
        Y = _Y(); c = syndes_certificate(Y, 3, "global_equal_weights",
                                         _incumbent(Y, 3, "global_equal_weights"))
        assert c.certified is True and c.method == "continuous_relaxation"

    def test_two_way_certified_sdp(self):
        Y = _Y(); c = syndes_certificate(Y, 3, "global_2way",
                                         _incumbent(Y, 3, "global_2way"))
        assert c.certified is True and c.method == "sdp_moment"
        # SDP is tight on the two-way objective
        assert c.optimality_gap < 0.15

    def test_per_unit_uncertified(self):
        Y = _Y(); c = syndes_certificate(Y, 3, "per_unit",
                                         _incumbent(Y, 3, "per_unit"))
        assert c.certified is False
        assert "per-unit" in c.note

    def test_two_way_size_gate_falls_back(self):
        # N above sdp_n_max -> continuous fallback, certified=False, note explains.
        Y = _Y(N=12); c = syndes_certificate(Y, 3, "global_2way",
                                             _incumbent(Y, 3, "global_2way"),
                                             sdp_n_max=5)
        assert c.certified is False and c.method == "continuous_relaxation"
        assert "sdp_n_max" in c.note

    def test_unknown_mode_raises(self):
        with pytest.raises(MlsynthConfigError):
            syndes_certificate(_Y(), 3, "not_a_mode", 1.0)


# ----------------------------------------------------------------------
# Estimator integration
# ----------------------------------------------------------------------
def _panel(N=10, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, N)) * 0.4 + rng.standard_normal(N)
    return pd.DataFrame([
        {"unit": f"u{i}", "time": t, "y": float(Y[t, i]), "post": int(t >= T - n_post)}
        for i in range(N) for t in range(T)])


class TestEstimator:
    def _fit(self, mode, **over):
        cfg = {"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
               "K": 3, "mode": mode, "post_col": "post", "run_inference": False}
        cfg.update(over)
        return SYNDES(cfg).fit()

    def test_certify_attaches_valid_certificate(self):
        res = self._fit("two_way_global", certify=True)
        c = res.certificate
        assert c is not None
        assert c.lower_bound <= float(res.design.objective_value) + 1e-6
        assert c.method == "sdp_moment" and c.certified is True

    def test_default_off_no_certificate(self):
        assert self._fit("two_way_global").certificate is None

    def test_per_unit_estimator_uncertified(self):
        c = self._fit("per_unit", certify=True).certificate
        assert c is not None and c.certified is False

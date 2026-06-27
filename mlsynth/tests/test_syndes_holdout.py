"""TDD for SYNDES holdout (train/validate) design selection.

The vanilla SYNDES MIP selects the treated set that minimises the *in-sample*
pre-period contrast, which can overfit transient co-movement. The holdout
selector instead learns each candidate design on the first ``1 - holdout_frac``
of the pre-period and ranks the candidate pool by *out-of-sample* contrast error
on the held-out tail; the candidate with the smallest OOS error wins. Power and
inference are then computed exactly as in the in-sample path.

These tests pin:

* Layer 1 -- the pure split / OOS / ranking helpers in
  :mod:`mlsynth.utils.syndes_helpers.holdout`;
* Layer 3 -- the estimator wiring (``holdout_frac`` config -> OOS-ranked pool,
  winner = min-OOS design, power curve still attached);
* config / failure semantics (translated ``MlsynthConfigError``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.holdout import (
    oos_contrast_rmse,
    select_by_holdout,
    split_pre,
)


def _panel(N=8, T=18, n_post=5, seed=0):
    """Factor-model long panel with a 0/1 ``post`` column (last ``n_post``)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = [
        {"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
         "post": int(t >= T - n_post)}
        for j in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Layer 1 -- split_pre
# ----------------------------------------------------------------------

class TestSplitPre:
    def test_basic_split_shapes(self):
        Y = np.zeros((10, 3))
        Y_tr, Y_va, n_train = split_pre(Y, 0.3)
        assert n_train == 7
        assert Y_tr.shape == (7, 3) and Y_va.shape == (3, 3)

    def test_train_plus_val_equals_pre(self):
        Y = np.arange(13 * 4, dtype=float).reshape(13, 4)
        Y_tr, Y_va, n_train = split_pre(Y, 0.3)
        assert Y_tr.shape[0] + Y_va.shape[0] == 13
        # val is the contiguous tail
        assert np.allclose(Y_va, Y[n_train:])

    def test_clamps_to_at_least_one_each_side(self):
        Y = np.zeros((2, 2))
        Y_tr, Y_va, n_train = split_pre(Y, 0.9)      # would round to 2 val
        assert Y_tr.shape[0] >= 1 and Y_va.shape[0] >= 1
        assert n_train == 1

    def test_raises_when_too_short(self):
        with pytest.raises(MlsynthConfigError):
            split_pre(np.zeros((1, 3)), 0.3)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_raises_on_bad_frac(self, bad):
        with pytest.raises(MlsynthConfigError):
            split_pre(np.zeros((10, 3)), bad)


# ----------------------------------------------------------------------
# Layer 1 -- oos_contrast_rmse
# ----------------------------------------------------------------------

class TestOOSContrastRMSE:
    def test_finite_nonnegative(self):
        rng = np.random.default_rng(0)
        Y_val = rng.normal(size=(4, 5))

        class _D:
            contrast_weights = np.array([1.0, -0.5, -0.5, 0.0, 0.0])

        v = oos_contrast_rmse(_D(), Y_val)
        assert np.isfinite(v) and v >= 0.0

    def test_zero_contrast_is_zero(self):
        Y_val = np.ones((3, 4))

        class _D:
            contrast_weights = np.zeros(4)

        assert oos_contrast_rmse(_D(), Y_val) == pytest.approx(0.0)

    def test_missing_contrast_returns_inf(self):
        class _D:
            contrast_weights = None

        assert oos_contrast_rmse(_D(), np.ones((3, 2))) == float("inf")


# ----------------------------------------------------------------------
# Layer 1 -- select_by_holdout ranking
# ----------------------------------------------------------------------

class TestSelectByHoldout:
    def _Ypre(self):
        df = _panel()
        Ywide = df[df["post"] == 0].pivot(index="time", columns="unit",
                                          values="Y").sort_index()
        return np.asarray(Ywide.values, dtype=float)

    def test_pool_ranked_by_oos_ascending(self):
        Y_pre = self._Ypre()
        ranked, oos = select_by_holdout(
            Y_pre, holdout_frac=0.3, top_K=4, K=2, mode="global_2way",
            solver="SCIP", unit_index=None, gap_limit=0.2, time_limit=5.0,
        )
        assert len(ranked) == len(oos) >= 1
        assert all(oos[i] <= oos[i + 1] + 1e-12 for i in range(len(oos) - 1))

    def test_winner_is_argmin_oos(self):
        Y_pre = self._Ypre()
        ranked, oos = select_by_holdout(
            Y_pre, holdout_frac=0.3, top_K=4, K=2, mode="global_2way",
            solver="SCIP", unit_index=None, gap_limit=0.2, time_limit=5.0,
        )
        # Recompute OOS independently on the same split and confirm rank-1 is min.
        _, Y_val, _ = split_pre(Y_pre, 0.3)
        recomputed = [oos_contrast_rmse(d, Y_val) for d in ranked]
        assert recomputed[0] == pytest.approx(min(recomputed))
        assert recomputed == pytest.approx(sorted(recomputed))


# ----------------------------------------------------------------------
# Layer 3 -- estimator integration
# ----------------------------------------------------------------------

class TestSYNDESHoldoutIntegration:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 K=2, mode="two_way_global", run_inference=False,
                 solver="SCIP", gap_limit=0.2, time_limit=5.0)

    def _fit(self, **over):
        df = _panel()
        cfg = {"df": df, **self._BASE, **over}
        return SYNDES(cfg).fit()

    def test_smoke_runs_and_keeps_power_curve(self):
        res = self._fit(holdout_frac=0.3, top_K=4)
        assert res is not None
        assert res.pool is not None and len(res.pool) >= 1
        assert res.power_curve is not None

    def test_pool_entries_carry_oos_rmse(self):
        res = self._fit(holdout_frac=0.3, top_K=4)
        assert all("oos_rmse" in e for e in res.pool)
        assert all(e["oos_rmse"] is not None
                   and np.isfinite(e["oos_rmse"]) for e in res.pool)

    def test_pool_ordered_by_oos_and_winner_is_min(self):
        res = self._fit(holdout_frac=0.3, top_K=4)
        oos = [e["oos_rmse"] for e in res.pool]
        assert oos == pytest.approx(sorted(oos))
        # winner design == rank-1 pool entry
        assert (list(np.asarray(res.selected_unit_labels))
                == list(res.pool[0]["markets"]))

    def test_reported_oos_matches_independent_recompute(self):
        res = self._fit(holdout_frac=0.3, top_K=4)
        Y_pre = np.asarray(res.inputs.Y_pre, dtype=float)
        _, Y_val, _ = split_pre(Y_pre, 0.3)
        for e in res.pool:
            assert e["oos_rmse"] == pytest.approx(
                oos_contrast_rmse(e["design"], Y_val))

    def test_pool_defaults_to_holdout(self):
        # A pool (top_K>1) now holdout-validates by default: oos_rmse is populated.
        res = self._fit(top_K=4)
        assert res.pool is not None
        assert all(e.get("oos_rmse") is not None
                   and np.isfinite(e["oos_rmse"]) for e in res.pool)

    def test_explicit_in_sample_opts_out_of_holdout(self):
        # selection='in_sample' keeps the paper-faithful in-sample pool (no oos).
        res = self._fit(top_K=4, selection="in_sample")
        assert res.pool is not None
        assert all(e.get("oos_rmse") is None for e in res.pool)

    def test_backward_compat_single_design(self):
        res = self._fit()                 # top_K=1, no holdout
        assert not getattr(res, "pool", None)
        assert res.design is not None


# ----------------------------------------------------------------------
# Config / failure semantics
# ----------------------------------------------------------------------

class TestSYNDESHoldoutConfig:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 K=2, run_inference=False)

    def test_requires_top_K_ge_2(self):
        df = _panel()
        with pytest.raises(MlsynthConfigError):
            SYNDES({"df": df, **self._BASE, "mode": "two_way_global",
                    "holdout_frac": 0.3, "top_K": 1})

    def test_rejects_annealed_mode(self):
        df = _panel()
        with pytest.raises(MlsynthConfigError):
            SYNDES({"df": df, **self._BASE, "mode": "two_way_global_annealed",
                    "holdout_frac": 0.3, "top_K": 4})

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
    def test_rejects_out_of_range_frac(self, bad):
        df = _panel()
        with pytest.raises(MlsynthConfigError):
            SYNDES({"df": df, **self._BASE, "mode": "two_way_global",
                    "holdout_frac": bad, "top_K": 4})

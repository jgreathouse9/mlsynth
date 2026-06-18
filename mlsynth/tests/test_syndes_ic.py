"""TDD for SYNDES information-criterion (IC) design selection.

The holdout selector spends data on a validation tail, which is noisy when the
pre-period is short -- exactly the regime SYNDES targets (few aggregate units).
The IC selector instead uses the *whole* pre-window: it ranks the candidate pool
by an information criterion

    IC(d) = SSR_pre(d) + 2 * sigma^2 * df(d),

where ``SSR_pre`` is the in-sample contrast sum of squares, ``df`` is the
synthetic-control degrees of freedom (active control donors minus one, after
Pouliot-Xie-Liu's ``df = |A| - 1``), and ``sigma^2`` is a Mallows-style noise
estimate (the best-fitting candidate's per-period contrast variance). The
candidate with the smallest IC wins -- penalising designs that buy fit by
activating more donors.

These tests pin the pure IC helpers, the estimator wiring (the new ``selection``
field and its back-compat inference from ``holdout_frac``), and config/failure
semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.infocriterion import design_df, select_by_ic


def _panel(N=8, T=18, n_post=5, seed=0):
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


class _FakeDesign:
    """Duck-typed stand-in for SYNDESDesign (only the fields the IC reads)."""

    def __init__(self, contrast, control=None, treated2d=None):
        self.contrast_weights = None if contrast is None else np.asarray(contrast, float)
        self.control_weights = None if control is None else np.asarray(control, float)
        self.treated_weights = None if treated2d is None else np.asarray(treated2d, float)


# ----------------------------------------------------------------------
# Layer 1 -- design_df
# ----------------------------------------------------------------------

class TestDesignDF:
    def test_active_control_minus_one(self):
        d = _FakeDesign([1, -.5, -.5, 0, 0], control=[0, .5, .5, 0, 0])
        assert design_df(d) == 1                       # 2 active controls - 1

    def test_denser_control_higher_df(self):
        d = _FakeDesign([1, -.25, -.25, -.25, -.25],
                        control=[0, .25, .25, .25, .25])
        assert design_df(d) == 3                       # 4 active controls - 1

    def test_floor_at_zero(self):
        d = _FakeDesign([1, -1, 0], control=[0, 1, 0])
        assert design_df(d) == 0                        # single active control

    def test_per_unit_uses_column_sum(self):
        # control_weights None, treated_weights is the (K, N) per-unit SC matrix
        d = _FakeDesign(None, control=None,
                        treated2d=[[0.5, 0, 0.5, 0, 0], [0, 0.5, 0.5, 0, 0]])
        assert design_df(d) == 2                        # cols {0,1,2} active - 1

    def test_no_control_is_zero(self):
        d = _FakeDesign([1, -1], control=None, treated2d=None)
        assert design_df(d) == 0


# ----------------------------------------------------------------------
# Layer 1 -- select_by_ic
# ----------------------------------------------------------------------

class TestSelectByIC:
    def test_penalty_prefers_sparser_at_equal_fit(self):
        # Same contrast (=> same SSR) but B activates more control donors,
        # so its df is larger and the IC must rank the sparser A first.
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(12, 5)) + 10.0
        c = np.array([1.0, -0.5, -0.5, 0.0, 0.0])
        A = _FakeDesign(c, control=[0, .5, .5, 0, 0])             # df 1
        B = _FakeDesign(c.copy(), control=[0, .25, .25, .25, .25])  # df 3
        ranked, ic, dfs, sigma2 = select_by_ic([B, A], Y)
        assert dfs[0] < dfs[1]
        assert ranked[0] is A
        assert ic[0] <= ic[1] + 1e-12

    def test_ranks_ascending_and_sigma2_is_min_mse(self):
        rng = np.random.default_rng(1)
        Y = rng.normal(size=(15, 6)) + 5.0
        designs = [
            _FakeDesign([1, -1, 0, 0, 0, 0], control=[0, 1, 0, 0, 0, 0]),
            _FakeDesign([1, -.5, -.5, 0, 0, 0], control=[0, .5, .5, 0, 0, 0]),
            _FakeDesign([1, -.34, -.33, -.33, 0, 0],
                        control=[0, .34, .33, .33, 0, 0]),
        ]
        ranked, ic, dfs, sigma2 = select_by_ic(designs, Y)
        assert ic == sorted(ic)
        ssr = [float(np.sum((Y @ d.contrast_weights) ** 2)) for d in designs]
        assert sigma2 == pytest.approx(min(ssr) / Y.shape[0])

    def test_empty_pool(self):
        ranked, ic, dfs, sigma2 = select_by_ic([], np.zeros((4, 3)))
        assert ranked == [] and ic == [] and dfs == []

    def test_missing_contrast_never_wins(self):
        # A design with no contrast vector gets infinite SSR -> ranked last.
        rng = np.random.default_rng(2)
        Y = rng.normal(size=(10, 4)) + 7.0
        good = _FakeDesign([1, -1, 0, 0], control=[0, 1, 0, 0])
        bad = _FakeDesign(None, control=[0, .5, .5, 0])
        ranked, ic, dfs, sigma2 = select_by_ic([bad, good], Y)
        assert ranked[0] is good
        assert ic[-1] == float("inf")


# ----------------------------------------------------------------------
# Layer 3 -- estimator integration
# ----------------------------------------------------------------------

class TestSYNDESICIntegration:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 K=2, mode="two_way_global", run_inference=False,
                 solver="SCIP", gap_limit=0.2, time_limit=5.0)

    def _fit(self, **over):
        return SYNDES({"df": _panel(), **self._BASE, **over}).fit()

    def test_smoke_runs_and_keeps_power_curve(self):
        res = self._fit(selection="ic", top_K=4)
        assert res is not None
        assert res.pool is not None and len(res.pool) >= 1
        assert res.power_curve is not None

    def test_pool_entries_carry_ic_and_df(self):
        res = self._fit(selection="ic", top_K=4)
        assert all("ic" in e and "df" in e for e in res.pool)
        assert all(e["ic"] is not None and np.isfinite(e["ic"]) for e in res.pool)
        assert all(isinstance(e["df"], int) and e["df"] >= 0 for e in res.pool)

    def test_pool_ordered_by_ic_and_winner_is_min(self):
        res = self._fit(selection="ic", top_K=4)
        ic = [e["ic"] for e in res.pool]
        assert ic == sorted(ic)
        assert (list(np.asarray(res.selected_unit_labels))
                == list(res.pool[0]["markets"]))

    def test_reported_df_matches_recompute(self):
        res = self._fit(selection="ic", top_K=4)
        for e in res.pool:
            assert e["df"] == design_df(e["design"])

    def test_default_none_is_in_sample(self):
        res = self._fit(top_K=4)                # no selection, no holdout
        assert res.pool is not None
        assert all(e.get("ic") is None for e in res.pool)

    def test_holdout_still_works_via_frac(self):
        # back-compat: holdout_frac alone still routes to holdout selection.
        res = self._fit(holdout_frac=0.3, top_K=4)
        assert all(e.get("oos_rmse") is not None for e in res.pool)
        assert all(e.get("ic") is None for e in res.pool)


# ----------------------------------------------------------------------
# Config / failure semantics
# ----------------------------------------------------------------------

class TestSYNDESICConfig:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 K=2, run_inference=False)

    def _make(self, **over):
        return SYNDES({"df": _panel(), **self._BASE, "mode": "two_way_global",
                       **over})

    def test_ic_requires_top_K_ge_2(self):
        with pytest.raises(MlsynthConfigError):
            self._make(selection="ic", top_K=1)

    def test_ic_rejects_annealed(self):
        with pytest.raises(MlsynthConfigError):
            SYNDES({"df": _panel(), **self._BASE,
                    "mode": "two_way_global_annealed",
                    "selection": "ic", "top_K": 4})

    def test_ic_conflicts_with_holdout_frac(self):
        with pytest.raises(MlsynthConfigError):
            self._make(selection="ic", holdout_frac=0.3, top_K=4)

    def test_in_sample_conflicts_with_holdout_frac(self):
        with pytest.raises(MlsynthConfigError):
            self._make(selection="in_sample", holdout_frac=0.3, top_K=4)

    def test_holdout_selection_requires_frac(self):
        with pytest.raises(MlsynthConfigError):
            self._make(selection="holdout", top_K=4)     # no holdout_frac

    def test_invalid_selection_string(self):
        with pytest.raises(MlsynthConfigError):
            self._make(selection="bogus", top_K=4)

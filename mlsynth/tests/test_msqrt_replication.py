"""Coverage tests for the Shen-Song-Abadie (2025) MSC replication helpers.

Exercises :mod:`mlsynth.utils.msqrt_helpers.replication`: both DGP settings,
the long-panel reshaper, a single estimator run, and the Monte-Carlo driver
with both the CV-selected and overridden penalty paths -- all at the smallest
valid sizes with a fixed seed.
"""

from __future__ import annotations

import matplotlib  # noqa: E402  (must precede any pyplot import)
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.msqrt_helpers.replication import (
    DEMO,
    PAPER,
    SimConfig,
    _one_run,
    run_msqrt_simulation,
    simulate_shen2025,
    to_long_df,
)


@pytest.fixture(autouse=True)
def _no_artifacts(tmp_path, monkeypatch):
    """Run inside a temp dir and close figures so nothing lands in the repo."""
    import matplotlib.pyplot as plt
    monkeypatch.chdir(tmp_path)
    yield
    plt.close("all")


# A tiny config: small donor pool, short panels, minimal reps.
_TINY = SimConfig(n=8, T0=12, T1=4, s=10, sigma=0.3,
                  m_grid=[2, 3], n_reps=1, lambda_grid=[0.1, 1.0])


def _draw(setting, m=3, seed=0):
    rng = np.random.default_rng(seed)
    return simulate_shen2025(setting, m, n=8, T0=12, T1=4, s=10,
                             sigma=0.3, rng=rng)


# ----------------------------------------------------------------------
# DGP: both settings + error branch
# ----------------------------------------------------------------------

class TestSimulateShen2025:
    def test_setting1_shape(self):
        X, Y = _draw(1, m=3)
        assert X.shape == (16, 8)        # (T0+T1, n)
        assert Y.shape == (16, 3)        # (T, m)

    def test_setting2_shape(self):
        X, Y = _draw(2, m=2)
        assert X.shape == (16, 8)
        assert Y.shape == (16, 2)

    @pytest.mark.parametrize("setting", [1, 2])
    def test_determinism(self, setting):
        Xa, Ya = _draw(setting, seed=1)
        Xb, Yb = _draw(setting, seed=1)
        np.testing.assert_array_equal(Xa, Xb)
        np.testing.assert_array_equal(Ya, Yb)

    def test_invalid_setting_raises(self):
        with pytest.raises(ValueError, match="setting must be 1 or 2"):
            _draw(3)


# ----------------------------------------------------------------------
# Long-panel reshape
# ----------------------------------------------------------------------

class TestToLongDf:
    def test_columns_and_block_structure(self):
        X, Y = _draw(2, m=3)
        T0 = 12
        df = to_long_df(X, Y, T0)
        assert set(df.columns) == {"unit", "time", "Y", "treated"}
        n, m = X.shape[1], Y.shape[1]
        T = X.shape[0]
        assert len(df) == (n + m) * T
        # control units (prefix c) never treated
        ctrl = df[df["unit"].str.startswith("c")]
        assert (ctrl["treated"] == 0).all()
        # treated units (prefix t) switch on at T0, off before
        trt = df[df["unit"].str.startswith("t")]
        assert (trt["treated"] == (trt["time"] >= T0).astype(int)).all()
        assert ctrl["unit"].nunique() == n
        assert trt["unit"].nunique() == m

    def test_values_roundtrip(self):
        X, Y = _draw(2, m=2)
        df = to_long_df(X, Y, 12)
        # first control column matches the long-form values for c0
        cname = sorted(df[df["unit"].str.startswith("c")]["unit"].unique())[0]
        c0 = df[df["unit"] == cname].sort_values("time")["Y"].to_numpy()
        np.testing.assert_allclose(c0, X[:, 0])


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

class TestPresets:
    def test_paper_and_demo_frozen(self):
        assert isinstance(PAPER, SimConfig)
        assert isinstance(DEMO, SimConfig)
        # lambda_grid default factory yields a 6-point log grid.
        cfg = SimConfig(n=2, T0=2, T1=1, s=1, sigma=0.1,
                        m_grid=[1], n_reps=1)
        assert len(cfg.lambda_grid) == 6


# ----------------------------------------------------------------------
# Single estimator run
# ----------------------------------------------------------------------

class TestOneRun:
    def test_returns_bias_and_rmse(self):
        X, Y = _draw(2, m=3, seed=2)
        bias, rmse = _one_run(X, Y, T0=12, lambd=0.5)
        assert np.isfinite(bias)
        assert np.isfinite(rmse) and rmse >= 0.0


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

class TestDriver:
    def test_override_lambda_silent(self):
        # lambda_override branch + verbose=False (skips all printing).
        out = run_msqrt_simulation(
            _TINY, settings=(2,), seed=0,
            lambda_override=0.5, verbose=False,
        )
        assert set(out) == {2}
        rows = out[2]
        assert [r["m"] for r in rows] == _TINY.m_grid
        for r in rows:
            assert set(r) == {"m", "rmse_mean", "rmse_std",
                              "bias_mean", "bias_std"}
            assert np.isfinite(r["rmse_mean"])

    def test_cv_path_verbose_both_settings(self, capsys):
        # No lambda_override -> CV pre-selection branch; verbose=True -> prints.
        out = run_msqrt_simulation(
            _TINY, settings=(1, 2), seed=0, verbose=True,
        )
        assert set(out) == {1, 2}
        captured = capsys.readouterr().out
        assert "pre-selected lambda" in captured

"""Coverage tests for mlsynth.utils.rmsi_helpers.replication."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

import mlsynth.utils.rmsi_helpers.replication as rep
from mlsynth.utils.rmsi_helpers.replication import (
    DEMO,
    RMSISimConfig,
    run_rmsi_simulation,
    simulate_rmsi_dgp,
    replicate_prop99,
    _poly_features,
)

BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
P99 = BASEDATA / "P99data.csv"


def test_poly_features_shape_and_determinism():
    C = np.random.default_rng(0).standard_normal((6, 2))
    a = _poly_features(C, 4, np.random.default_rng(1))
    b = _poly_features(C, 4, np.random.default_rng(1))
    assert a.shape == (6, 4)
    np.testing.assert_array_equal(a, b)


def test_simulate_rmsi_dgp_shapes_and_determinism():
    Y, M, X, Z = simulate_rmsi_dgp(20, 15, seed=0)
    assert Y.shape == (20, 15)
    assert M.shape == (20, 15)
    assert X.shape == (20, 4)
    assert Z.shape == (15, 4)
    Y2, M2, X2, Z2 = simulate_rmsi_dgp(20, 15, seed=0)
    np.testing.assert_array_equal(Y, Y2)
    np.testing.assert_array_equal(M, M2)


def test_run_rmsi_simulation_verbose(capsys):
    cfg = RMSISimConfig(N=24, T=24, N0=12, T0=12, J=3, n_reps=2)
    out = run_rmsi_simulation(cfg, seed=0, verbose=True)
    assert set(out) == {"rmsi", "no_side_info", "rel_improvement"}
    assert isinstance(out["rmsi"], float)
    captured = capsys.readouterr()
    assert "RMSI synthetic MNAR" in captured.out


def test_run_rmsi_simulation_quiet():
    cfg = RMSISimConfig(N=24, T=24, N0=12, T0=12, J=3, n_reps=1)
    out = run_rmsi_simulation(cfg, seed=1, verbose=False)
    assert "rmsi" in out


@pytest.mark.skipif(not P99.exists(), reason="P99data not present")
def test_replicate_prop99_with_path_verbose(capsys):
    res = replicate_prop99(str(P99), rank=3, verbose=True)
    assert hasattr(res, "att")
    captured = capsys.readouterr()
    assert "Proposition 99" in captured.out


@pytest.mark.skipif(not P99.exists(), reason="P99data not present")
def test_replicate_prop99_data_none_branch(monkeypatch):
    # data=None branch: redirect the download URL to the local CSV path.
    monkeypatch.setattr(rep, "PROP99_URL", str(P99))
    res = replicate_prop99(data=None, rank=3, verbose=False)
    assert hasattr(res, "att")

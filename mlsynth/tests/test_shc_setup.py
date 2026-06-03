"""Coverage tests for mlsynth.utils.shc_helpers.setup.prepare_shc_inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.shc_helpers import setup as setup_mod
from mlsynth.utils.shc_helpers.setup import prepare_shc_inputs
from mlsynth.utils.shc_helpers.structures import SHCInputs


def _panel(T0: int = 8, n_post: int = 2, seed: int = 0) -> pd.DataFrame:
    T = T0 + n_post
    t = np.arange(1, T + 1)
    rng = np.random.RandomState(seed)
    y = np.linspace(1.0, 2.0, T) + rng.normal(0.0, 0.01, T)
    return pd.DataFrame(
        {
            "unit": np.ones(T, dtype=int),
            "time": t,
            "y": y,
            "treated": (t > T0).astype(int),
        }
    )


def test_prepare_inputs_happy_path_structure():
    df = _panel(T0=8, n_post=2)
    inp = prepare_shc_inputs(df, "y", "treated", "unit", "time", m=4)
    assert isinstance(inp, SHCInputs)
    assert inp.T0 == 8
    assert inp.n == 2
    assert inp.m == 4
    # N = T0 - m - n + 1 = 8 - 4 - 2 + 1 = 3
    assert inp.N == 3
    assert inp.metadata["n_historical_blocks"] == 3
    assert inp.y.shape == (10,)
    assert np.all(np.isfinite(inp.y))
    assert inp.metadata["Ywide"] is not None
    # time index labels cover all periods
    assert len(inp.time_index.labels) == 10


def test_prepare_inputs_m_nonpositive_raises():
    df = _panel(T0=8, n_post=2)
    with pytest.raises(MlsynthDataError, match="m > 0"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=0)
    with pytest.raises(MlsynthDataError, match="m > 0"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=-3)


def test_prepare_inputs_insufficient_pre_period_raises():
    # m too large so N = T0 - m - n + 1 <= 0.
    df = _panel(T0=8, n_post=2)
    with pytest.raises(MlsynthDataError, match="Insufficient pre-treatment"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=8)


def test_prepare_inputs_missing_period_counts_raises(monkeypatch):
    """Defensive branch: dataprep returns no pre/post counts."""
    df = _panel(T0=8, n_post=2)

    def fake_dataprep(*args, **kwargs):
        return {"pre_periods": None, "post_periods": None}

    monkeypatch.setattr(setup_mod, "dataprep", fake_dataprep)
    with pytest.raises(MlsynthDataError, match="did not return pre/post"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=4)


def test_prepare_inputs_no_post_period_raises(monkeypatch):
    """Defensive branch: dataprep reports zero post periods."""
    df = _panel(T0=8, n_post=2)

    def fake_dataprep(*args, **kwargs):
        return {
            "pre_periods": 8,
            "post_periods": 0,
            "y": np.linspace(1.0, 2.0, 10),
            "time_labels": np.arange(1, 11),
            "treated_unit_name": "1",
            "Ywide": None,
        }

    monkeypatch.setattr(setup_mod, "dataprep", fake_dataprep)
    with pytest.raises(MlsynthDataError, match="at least one post-treatment"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=4)

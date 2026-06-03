"""Additional coverage tests for shc_helpers.setup.prepare_shc_inputs.

A pre-existing ``test_shc_setup.py`` already exercises the error branches;
this file adds independent structural / wiring assertions and re-covers the
happy path and validation raises so setup.py reaches 100% on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.shc_helpers import setup as setup_mod
from mlsynth.utils.shc_helpers.setup import prepare_shc_inputs
from mlsynth.utils.shc_helpers.structures import SHCInputs


def _panel(T0: int = 10, n_post: int = 3, seed: int = 1) -> pd.DataFrame:
    T = T0 + n_post
    t = np.arange(1, T + 1)
    rng = np.random.RandomState(seed)
    y = np.linspace(0.0, 5.0, T) + rng.normal(0.0, 0.05, T)
    return pd.DataFrame(
        {
            "unit": np.full(T, 7, dtype=int),
            "time": t,
            "y": y,
            "treated": (t > T0).astype(int),
        }
    )


def test_inputs_wiring_and_indexset():
    df = _panel(T0=10, n_post=3)
    inp = prepare_shc_inputs(df, "y", "treated", "unit", "time", m=5)
    assert isinstance(inp, SHCInputs)
    assert inp.T == 13
    assert inp.T0 == 10
    assert inp.n == 3
    assert inp.m == 5
    # N = T0 - m - n + 1 = 10 - 5 - 3 + 1 = 3
    assert inp.N == 3
    assert inp.metadata["n_historical_blocks"] == 3
    # time index has one label per period and is monotone increasing
    labels = np.asarray(inp.time_index.labels)
    assert labels.shape == (13,)
    np.testing.assert_array_equal(labels, np.arange(1, 14))
    # treated label is identified
    assert inp.treated_label is not None


def test_inputs_deterministic():
    df = _panel()
    a = prepare_shc_inputs(df, "y", "treated", "unit", "time", m=5)
    b = prepare_shc_inputs(df, "y", "treated", "unit", "time", m=5)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.T0 == b.T0 and a.N == b.N


def test_m_nonpositive_raises():
    df = _panel()
    with pytest.raises(MlsynthDataError, match="m > 0"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=0)


def test_insufficient_pre_period_raises():
    df = _panel(T0=10, n_post=3)
    with pytest.raises(MlsynthDataError, match="Insufficient pre-treatment"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=10)


def test_missing_period_counts_raises(monkeypatch):
    df = _panel()
    monkeypatch.setattr(
        setup_mod, "dataprep",
        lambda *a, **k: {"pre_periods": None, "post_periods": None},
    )
    with pytest.raises(MlsynthDataError, match="did not return pre/post"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=5)


def test_zero_post_period_raises(monkeypatch):
    df = _panel()

    def fake_dataprep(*a, **k):
        return {
            "pre_periods": 10,
            "post_periods": 0,
            "y": np.zeros(13),
            "time_labels": np.arange(1, 14),
            "treated_unit_name": "7",
            "Ywide": None,
        }

    monkeypatch.setattr(setup_mod, "dataprep", fake_dataprep)
    with pytest.raises(MlsynthDataError, match="at least one post-treatment"):
        prepare_shc_inputs(df, "y", "treated", "unit", "time", m=5)

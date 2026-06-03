"""Coverage tests for mlsynth.utils.ssc_helpers.replication."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

import mlsynth.utils.ssc_helpers.replication as rep
from mlsynth.utils.ssc_helpers.replication import (
    DEMO,
    SSCSimConfig,
    run_ssc_simulation,
    replicate_guanajuato,
)

BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
CRIME = BASEDATA / "guanajuato_crime_ssc.csv"
CARTEL = BASEDATA / "guanajuato_cartel_ssc.csv"


def test_run_ssc_simulation_verbose(capsys):
    cfg = SSCSimConfig(n_units=20, n_never=4, S=6, n_factors=2,
                       T0_grid=[42], n_reps=2)
    out = run_ssc_simulation(cfg, seed=0, verbose=True)
    assert len(out) == 1
    (key,) = out.keys()
    assert key == (2, 42)
    captured = capsys.readouterr()
    assert "SSC event-time RMSE" in captured.out
    assert "true ATT" in captured.out


def test_run_ssc_simulation_n_factors_override_quiet():
    cfg = SSCSimConfig(n_units=20, n_never=4, S=6, n_factors=2,
                       T0_grid=[42], n_reps=1)
    out = run_ssc_simulation(cfg, n_factors=3, seed=0, verbose=False)
    (key,) = out.keys()
    assert key[0] == 3  # override applied


@pytest.mark.skipif(not (CRIME.exists() and CARTEL.exists()),
                    reason="Guanajuato basedata not present")
def test_replicate_guanajuato_verbose_window_and_none_branches(capsys):
    # Pick one windowed crime outcome (window-query branch) and one cartel
    # outcome (window=None branch) to exercise both paths plus verbose=True.
    crime = pd.read_csv(CRIME)
    cartel = pd.read_csv(CARTEL)
    out = replicate_guanajuato(
        crime=crime, cartel=cartel,
        outcomes=["hom_all_rate", "presence_strength"],
        verbose=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert set(out["outcome"]) == {"hom_all_rate", "presence_strength"}
    assert {"event_time", "att", "ci_lower", "ci_upper", "T0", "S"} <= set(out.columns)
    captured = capsys.readouterr()
    assert "ATT^e_1" in captured.out


@pytest.mark.skipif(not (CRIME.exists() and CARTEL.exists()),
                    reason="Guanajuato basedata not present")
def test_replicate_guanajuato_data_none_branch(monkeypatch):
    # crime=None / cartel=None branches: redirect URL prefix to local files.
    monkeypatch.setattr(rep, "GUANAJUATO_URL", str(BASEDATA) + "/")
    out = replicate_guanajuato(outcomes=["presence_strength"], verbose=False)
    assert list(out["outcome"].unique()) == ["presence_strength"]

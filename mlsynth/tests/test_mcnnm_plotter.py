"""Coverage-focused tests for mlsynth.utils.mcnnm_helpers.plotter."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from mlsynth.utils.mcnnm_helpers.plotter import (
    plot_mcnnm,
    _plot_event_study,
    _is_staggered,
)
from mlsynth.config_models import EffectsResults
from mlsynth.utils.mcnnm_helpers.structures import MCNNMInputs, MCNNMResults


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _make_inputs(N=3, T=6, T0=4, treated_idx=(0,), adopt_times=None):
    """Build an MCNNMInputs with a treatment design.

    adopt_times: optional dict {unit_idx: adoption_period}; if None, all
    treated units adopt at T0.
    """
    Y = np.arange(N * T, dtype=float).reshape(N, T) + 1.0
    D = np.zeros((N, T), dtype=int)
    treated_idx = np.array(treated_idx, dtype=int)
    if adopt_times is None:
        adopt_times = {int(i): T0 for i in treated_idx}
    for i, a in adopt_times.items():
        D[i, a:] = 1
    mask = (D == 0).astype(int)
    unit_names = [f"u{i}" for i in range(N)]
    time_labels = np.arange(T)
    return MCNNMInputs(
        Y=Y, mask=mask, D=D, treated_idx=treated_idx, T0=T0,
        unit_names=unit_names, time_labels=time_labels,
    )


def _make_results(inputs, event_study=None):
    N, T = inputs.Y.shape
    counterfactual = inputs.Y - 0.5
    effects = np.full((N, T), np.nan)
    return MCNNMResults(
        inputs=inputs,
        effects=EffectsResults(att=1.234),
        counterfactual_matrix=counterfactual,
        effects_matrix=effects,
        att_by_period={},
        cohort_att={},
        event_study=event_study or {},
        L=np.zeros((N, T)),
        gamma=np.zeros(N),
        delta=np.zeros(T),
        best_lambda=0.1,
        rank=1,
    )


def test_single_treated_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(treated_idx=(0,))
    results = _make_results(inputs)
    # save=False -> plt.show path; not staggered, single treated
    plot_mcnnm(results)
    assert not list(tmp_path.iterdir())


def test_single_treated_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(treated_idx=(0,))
    results = _make_results(inputs)
    plot_mcnnm(results, save={"directory": str(tmp_path), "filename": "mc", "extension": "png"})
    assert (tmp_path / "mc.png").exists()


def test_multi_treated_same_adopt(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # two treated, same adoption time -> not staggered -> mean branch
    inputs = _make_inputs(N=4, treated_idx=(0, 1))
    results = _make_results(inputs)
    plot_mcnnm(results, counterfactual_color=["red", "blue"])
    assert not list(tmp_path.iterdir())


def test_counterfactual_color_list_single_treated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(treated_idx=(0,))
    results = _make_results(inputs)
    # list cf color -> cf_color_single picks first element
    plot_mcnnm(results, counterfactual_color=["green"])
    assert not list(tmp_path.iterdir())


def test_counterfactual_color_empty_list(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(treated_idx=(0,))
    results = _make_results(inputs)
    # empty list -> fall back to "red"
    plot_mcnnm(results, counterfactual_color=[])
    assert not list(tmp_path.iterdir())


def test_staggered_event_study_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # two treated, different adoption times -> staggered -> event study
    inputs = _make_inputs(N=4, T=8, T0=3, treated_idx=(0, 1),
                          adopt_times={0: 3, 1: 5})
    assert _is_staggered(_make_results(inputs))
    es = {-2: 0.0, -1: 0.1, 0: 0.5, 1: 0.7}
    results = _make_results(inputs, event_study=es)
    plot_mcnnm(results)
    assert not list(tmp_path.iterdir())


def test_staggered_event_study_save_str_no_ext(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(N=4, T=8, T0=3, treated_idx=(0, 1),
                          adopt_times={0: 3, 1: 5})
    es = {-1: 0.1, 0: 0.5, 1: 0.7}
    results = _make_results(inputs, event_study=es)
    # str save without extension -> ".png" appended; saved to cwd (tmp)
    plot_mcnnm(results, save="myevent")
    assert (tmp_path / "myevent.png").exists()


def test_staggered_event_study_save_bool_default_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(N=4, T=8, T0=3, treated_idx=(0, 1),
                          adopt_times={0: 3, 1: 5})
    es = {0: 0.5, 1: 0.7}
    results = _make_results(inputs, event_study=es)
    # save truthy non-str (True) -> default filename branch
    plot_mcnnm(results, save=True)
    assert (tmp_path / "MC-NNM_event_study.png").exists()


def test_event_study_empty_returns_early(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    inputs = _make_inputs(N=4, T=8, T0=3, treated_idx=(0, 1),
                          adopt_times={0: 3, 1: 5})
    results = _make_results(inputs, event_study={})
    # staggered but empty event_study -> _plot_event_study returns immediately
    _plot_event_study(results, "black", "red", False, "Outcome")
    assert not list(tmp_path.iterdir())

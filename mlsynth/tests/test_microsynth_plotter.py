"""Line-coverage tests for ``mlsynth.utils.microsynth_helpers.plotter``.

Constructs ``MicroSynthResults`` directly so every branch (single-panel
T_post==1 love-plot-only, two-panel T_post>1 lift trajectory, string-color
vs list-color counterfactual, and the three ``save`` modes) is exercised
without leaving any artifact in the repo tree.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from mlsynth.utils.microsynth_helpers.plotter import plot_microsynth
from mlsynth.utils.microsynth_helpers.structures import (
    MicroSynthDesign,
    MicroSynthInference,
    MicroSynthInputs,
    MicroSynthResults,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_results(*, t_post: int) -> MicroSynthResults:
    rng = np.random.default_rng(0)
    d = 3
    n_C = 8
    cov_names = [f"x{k}" for k in range(d)]
    if t_post == 1:
        Y_T = rng.standard_normal(5)
        Y_C = rng.standard_normal(n_C)
        cf = rng.standard_normal(())
        gap = np.array(0.2)
    else:
        Y_T = rng.standard_normal((5, t_post))
        Y_C = rng.standard_normal((n_C, t_post))
        cf = rng.standard_normal(t_post)
        gap = rng.standard_normal(t_post)
    inputs = MicroSynthInputs(
        X_T=rng.standard_normal((5, d)),
        X_C=rng.standard_normal((n_C, d)),
        Y_T=Y_T,
        Y_C=Y_C,
        treated_unit_names=[f"t{i}" for i in range(5)],
        control_unit_names=[f"c{i}" for i in range(n_C)],
        covariate_names=cov_names,
        cohort_time=1,
        covariate_sd=np.ones(d),
        outcome="converted",
    )
    design = MicroSynthDesign(
        w=np.full(n_C, 1.0 / n_C),
        dual_lambda=np.zeros(d),
        dual_nu=0.0,
        smd_before=rng.standard_normal(d) * 0.3,
        smd_after=rng.standard_normal(d) * 0.02,
        ess=float(n_C),
        max_weight=1.0 / n_C,
        feasible=True,
        feasibility_message="ok",
        n_iterations=3,
        converged=True,
    )
    inference = MicroSynthInference(
        method="none", att=0.1, se=0.0,
        ci=np.array([0.0, 0.2]), n_bootstrap=0,
        bootstrap_atts=np.array([]),
    )
    gap_traj = np.atleast_1d(gap).astype(float)
    return MicroSynthResults(
        inputs=inputs, design=design, inference_detail=inference,
        counterfactual_post=cf, gap_post=gap, gap_trajectory=gap_traj,
        att_value=float(gap_traj.mean()), donor_weights_map={"c0": 1.0 / n_C},
    )


def test_single_panel_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=1)
    plot_microsynth(res)  # save=False -> plt.show()
    assert list(tmp_path.iterdir()) == []


def test_two_panel_lift_trajectory_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=4)
    plot_microsynth(res, counterfactual_color="green")
    assert list(tmp_path.iterdir()) == []


def test_counterfactual_color_as_list(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=3)
    plot_microsynth(res, counterfactual_color=["purple", "orange"])
    assert list(tmp_path.iterdir()) == []


def test_save_str(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=4)
    out = tmp_path / "ms.png"
    plot_microsynth(res, save=str(out))
    assert out.exists()


def test_save_dict_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=1)
    plot_microsynth(res, save={})  # uses default path "microsynth.png" + dpi
    assert (tmp_path / "microsynth.png").exists()


def test_save_dict_explicit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _make_results(t_post=2)
    out = tmp_path / "custom.png"
    plot_microsynth(res, save={"path": str(out), "dpi": 80})
    assert out.exists()

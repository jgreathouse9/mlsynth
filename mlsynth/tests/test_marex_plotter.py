"""Line-coverage tests for ``mlsynth.utils.marex_helpers.plotter``.

MAREX needs the SCIP solver (pyscipopt, installed). We drive the plotter
both from real ``MAREX.fit()`` results and from hand-built ``MAREXResults``
to hit every branch: default-cluster selection, the lone-"0"-cluster skip,
the ``n == 0`` early return, single- vs multi-panel layout, treatment vs
prediction plot types, the CI ``fill_between`` branch, and the
blank-period ``axvspan`` shading.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.plotter import plot_marex
from mlsynth.utils.marex_helpers.structures import (
    MAREXClusterDesign,
    MAREXGlobalDesign,
    MAREXInference,
    MAREXResults,
    MAREXStudy,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(J=8, T=14, clusters=False, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (J, 2))
    Y = lam @ F.T + 0.2 * rng.standard_normal((J, T))
    grp = np.repeat([0, 1], J // 2)
    rows = []
    for j in range(J):
        for t in range(T):
            row = {"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t])}
            if clusters:
                row["grp"] = int(grp[j])
            rows.append(row)
    return pd.DataFrame(rows)


def _inference(T, T1=4, Tb=2, with_ci=True):
    ci = np.full((T, 2), np.nan)
    if with_ci:
        ci[:, 0] = -1.0
        ci[:, 1] = 1.0
    return MAREXInference(
        treated_effects=np.zeros(T1),
        placebo_effects=np.zeros(Tb),
        fulltreated_effects=np.zeros(T),
        s_obs=0.1, global_p_value=0.5,
        per_period_pvals=np.full(T1, 0.5),
        ci=ci,
    )


def _cluster(label, T, with_ci, members):
    rng = np.random.default_rng(abs(hash(label)) % 2**32)
    N = 8
    return MAREXClusterDesign(
        label=label, members=members, cardinality=len(members),
        treated_weights=np.zeros(N), control_weights=np.zeros(N),
        selection_indicators=np.zeros(len(members)),
        synthetic_treated=rng.standard_normal(T),
        synthetic_control=rng.standard_normal(T),
        pre_treatment_means=rng.standard_normal(3),
        rmse=0.1, unit_weight_map={"Treated": {}, "Control": {}},
        inference=_inference(T, with_ci=with_ci) if with_ci is not None else None,
    )


def _results(*, cluster_labels, T=14, blank_periods=2, with_ci=True):
    clusters = {
        lab: _cluster(lab, T, with_ci, [f"u{lab}_{i}" for i in range(2)])
        for lab in cluster_labels
    }
    rng = np.random.default_rng(99)
    glob = MAREXGlobalDesign(
        Y_full=rng.standard_normal((8, T)),
        Y_fit=rng.standard_normal((8, T - blank_periods)),
        Y_blank=None,
        treated_weights_agg=np.zeros(8),
        control_weights_agg=np.zeros(8),
        synthetic_treated=rng.standard_normal(T),
        synthetic_control=rng.standard_normal(T),
        inference=_inference(T, with_ci=with_ci) if with_ci is not None else None,
    )
    study = MAREXStudy(design="standard", T0=10, blank_periods=blank_periods)
    return MAREXResults(clusters=clusters, study=study, globres=glob)


# ---------------------------------------------------------------------------
# Real fitted results
# ---------------------------------------------------------------------------

def test_plot_from_fit_treatment_and_prediction():
    res = MAREX({"df": _panel(), "outcome": "y", "unitid": "unit",
                 "time": "time", "T0": 10, "m_eq": 2}).fit()
    plot_marex(res, plot_type="treatment")
    plot_marex(res, plot_type="prediction")


# ---------------------------------------------------------------------------
# Branch coverage on hand-built results
# ---------------------------------------------------------------------------

def test_default_clusters_none_uses_all_keys():
    # clusters=None -> list(results.clusters.keys()); two real clusters + global
    res = _results(cluster_labels=["1", "2"])
    plot_marex(res)  # default plot_type="treatment", global_result=True


def test_lone_zero_cluster_is_skipped_global_only():
    # the single "0" cluster is dropped, leaving just the global panel (n==1)
    res = _results(cluster_labels=["0"])
    plot_marex(res)


def test_no_panels_early_return():
    # lone "0" cluster dropped AND global disabled -> n == 0 -> return
    res = _results(cluster_labels=["0"])
    plot_marex(res, global_result=False)


def test_prediction_type_multipanel():
    res = _results(cluster_labels=["1", "2"])
    plot_marex(res, plot_type="prediction")


def test_inference_without_ci():
    res = _results(cluster_labels=["1"], with_ci=False)
    plot_marex(res, plot_type="treatment")


def test_no_inference_object():
    res = _results(cluster_labels=["1"], with_ci=None)
    plot_marex(res, plot_type="treatment")


def test_no_blank_period_shading():
    res = _results(cluster_labels=["1"], blank_periods=0)
    plot_marex(res)


def test_explicit_cluster_subset_no_global():
    res = _results(cluster_labels=["1", "2"])
    plot_marex(res, clusters=["1"], global_result=False)


# ---------------------------------------------------------------------------
# donor_cloud: overlay the individual unit trajectories (the "Figure 4" cloud)
# ---------------------------------------------------------------------------

def test_donor_cloud_overlays_units_on_global_prediction():
    # On the global prediction panel, donor_cloud=True draws one faint line per
    # unit (rows of globres.Y_full) behind the two synthetic series.
    res = _results(cluster_labels=["0"])               # lone "0" -> global panel only
    plot_marex(res, plot_type="prediction", donor_cloud=False)
    base = len(plt.gcf().axes[0].lines)
    plt.close("all")
    plot_marex(res, plot_type="prediction", donor_cloud=True)
    ax = plt.gcf().axes[0]
    J = res.globres.Y_full.shape[0]
    assert len(ax.lines) == base + J
    # the cloud is drawn first, faint gray, and behind the synthetic series
    assert ax.lines[0].get_color() == "0.8"
    assert ax.lines[0].get_zorder() < ax.lines[-1].get_zorder()


def test_donor_cloud_defaults_off():
    res = _results(cluster_labels=["0"])
    plot_marex(res, plot_type="prediction")            # default donor_cloud=False
    assert len(plt.gcf().axes[0].lines) == 2           # synthetic treated + control only


def test_donor_cloud_ignored_in_treatment_mode():
    # the cloud is an outcome overlay; on the effect (treatment) plot it is a no-op
    res = _results(cluster_labels=["0"])
    plot_marex(res, plot_type="treatment", donor_cloud=False)
    base = len(plt.gcf().axes[0].lines)
    plt.close("all")
    plot_marex(res, plot_type="treatment", donor_cloud=True)
    assert len(plt.gcf().axes[0].lines) == base


def test_donor_cloud_empty_matrix_no_error():
    # a design with no units to overlay adds no cloud lines and does not raise
    res = _results(cluster_labels=["0"])
    T = res.globres.synthetic_treated.shape[0]
    glob = replace(res.globres, Y_full=np.empty((0, T)))
    res2 = MAREXResults(clusters=res.clusters, study=res.study, globres=glob)
    plot_marex(res2, plot_type="prediction", donor_cloud=True)
    assert len(plt.gcf().axes[0].lines) == 2


def test_donor_cloud_from_fit_prediction():
    res = MAREX({"df": _panel(), "outcome": "y", "unitid": "unit",
                 "time": "time", "T0": 10, "m_eq": 2}).fit()
    plot_marex(res, plot_type="prediction", donor_cloud=True)
    assert len(plt.gcf().axes[0].lines) > 2            # cloud + the two synthetics


# ---------------------------------------------------------------------------
# House style: the plot must use the mlsynth house style (Samsung-blue grid),
# not a plotter-local override.
# ---------------------------------------------------------------------------

def test_plot_uses_house_grid_style():
    from matplotlib.colors import to_hex
    res = _results(cluster_labels=["0"])
    plot_marex(res, plot_type="prediction")
    ax = plt.gcf().axes[0]
    gridlines = ax.get_ygridlines()
    assert gridlines                                   # grid is on
    assert to_hex(gridlines[0].get_color()).lower() == "#1428a0"   # house blue, not gray

"""Branch coverage for mlsynth.utils.shc_helpers.plotter.plot_shc."""

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlsynth import SHC
from mlsynth.utils.shc_helpers.simulation import simulate_shc_panel
from mlsynth.utils.shc_helpers.plotter import plot_shc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _fit(m=12, use_augmented=False, seed=0):
    df, _ = simulate_shc_panel(m=m, h=4, n=8, seed=seed)
    return SHC({"df": df, "outcome": "y", "treat": "treated",
                "unitid": "unit", "time": "time", "m": m,
                "use_augmented": use_augmented,
                "display_graphs": False}).fit()


def test_full_with_inference_band():
    # inference present, m < len(time), conformal lo/hi present & length-matched,
    # treatment axvline drawn, default SHC title.
    res = _fit()
    assert res.inference_detail is not None and res.inputs.m < len(res.time_labels)
    plot_shc(res)


def test_augmented_variant_title():
    res = _fit(use_augmented=True)
    assert res.design.use_augmented
    plot_shc(res)


def test_explicit_title():
    res = _fit()
    plot_shc(res, title="Custom Title", treated_color="navy",
             counterfactual_color="green")


def test_no_inference_branch():
    res = _fit()
    res2 = res.model_copy(update={"inference_detail": None})
    plot_shc(res2)


def test_conformal_bounds_length_mismatch():
    # lo/hi present but length != post_time: fill_between skipped.
    res = _fit()
    bad = dataclasses.replace(
        res.inference_detail,
        conformal_lower=np.array([0.0]),
        conformal_upper=np.array([1.0]),
    )
    res2 = res.model_copy(update={"inference_detail": bad})
    plot_shc(res2)


def test_conformal_bounds_none():
    # lo/hi are None: fill_between skipped even though inference present.
    res = _fit()
    bad = dataclasses.replace(
        res.inference_detail, conformal_lower=None, conformal_upper=None,
    )
    res2 = res.model_copy(update={"inference_detail": bad})
    plot_shc(res2)


def test_m_not_less_than_len_skips_band_and_axvline():
    # m >= len(time): both the conformal band and the treatment-start
    # axvline are skipped.
    res = _fit()
    big_m = dataclasses.replace(res.inputs, m=len(res.time_labels) + 5)
    res2 = res.model_copy(update={"inputs": big_m, "inference_detail": None})
    plot_shc(res2)

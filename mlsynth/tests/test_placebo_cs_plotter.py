"""Drawing the placebo-inverted confidence set.

The inversion returns two effect paths and, optionally, a sweep of bounds over
the assignment-probability tilt. Both are shapes, so both want a picture. These
tests pin what the picture is made of -- the band is the confidence paths, the
line is the gap, the pre-period band has zero width -- and the two things the
library promises about any plot helper: it returns its figure, and it never
displays or saves on the caller's behalf.

The refusal tests matter as much as the drawing ones. A result whose inference
was something else, or whose search failed, has nothing to draw; asking for the
picture anyway must say which, not raise an ``AttributeError`` three frames
down or return an empty axis.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlsynth.config_models import (  # noqa: E402
    InferenceResults, TimeSeriesResults,
)
from mlsynth.exceptions import MlsynthPlottingError  # noqa: E402
from mlsynth.utils.vanillasc_helpers.placebo_cs import (  # noqa: E402
    PlaceboConfidenceSet, SensitivityRow, confidence_set, effect_path,
    sensitivity_sweep,
)
from mlsynth.utils.vanillasc_helpers.placebo_cs_plotter import (  # noqa: E402
    plot_confidence_set, plot_sensitivity,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _panel(n_periods=12, n_units=5, pre=7, seed=1):
    """A small panel with donor weights that sum to one."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(50.0, 3.0, size=(n_periods, n_units))
    W = rng.random((n_units - 1, n_units))
    W /= W.sum(axis=0, keepdims=True)
    return Y, W, 0, pre


@pytest.fixture(scope="module")
def toy_cs():
    Y, W, treated, pre = _panel()
    cs = confidence_set(Y, W, treated, pre, kind="linear", alpha=2 / 5,
                        precision=10)
    gap = Y[:, treated] - Y[:, [k for k in range(Y.shape[1]) if k != treated]] @ W[:, treated]
    return cs, gap, pre


@pytest.fixture(scope="module")
def toy_rows():
    Y, W, treated, pre = _panel()
    v = np.zeros(Y.shape[1]); v[treated] = 1.0
    return sensitivity_sweep(Y, W, treated, pre, phis=(0.0, 0.5, 1.0), v=v,
                             kind="linear", alpha=2 / 5, precision=10)


def _result(cs, gap, pre, *, method="placebo-inverted confidence set (FP 2018)",
            details=None):
    """The parts of a fitted result the plotter reads, and nothing else."""
    class _Stub:
        pass

    if details is None:
        details = {"effect_class": cs.kind, "point_estimate": cs.point_estimate,
                   "contains_zero": cs.contains_zero, "phi": cs.phi,
                   "lower_path": cs.lower_path.tolist(),
                   "upper_path": cs.upper_path.tolist()}
    res = _Stub()
    res.inference = InferenceResults(
        method=method, confidence_level=0.6,
        ci_lower=None if cs is None else cs.lower,
        ci_upper=None if cs is None else cs.upper,
        details=details)
    res.time_series = TimeSeriesResults(
        estimated_gap=np.asarray(gap, dtype=float),
        time_periods=np.arange(1970, 1970 + len(gap)),
        intervention_time=1970 + pre)
    return res


# =========================================================================== #
# the confidence-set band
# =========================================================================== #
class TestConfidenceSetPlot:

    def test_returns_a_figure(self, toy_cs):
        cs, gap, _ = toy_cs
        fig = plot_confidence_set(cs, gap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_it_never_displays(self, toy_cs, monkeypatch):
        """Presentation is the caller's; the helper hands back the figure."""
        cs, gap, _ = toy_cs
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
        fig = plot_confidence_set(cs, gap)
        assert shown == []
        plt.close(fig)

    def test_the_line_is_the_gap(self, toy_cs):
        cs, gap, _ = toy_cs
        fig = plot_confidence_set(cs, gap)
        ax = fig.axes[0]
        drawn = [ln for ln in ax.lines
                 if len(ln.get_ydata()) == len(gap)]
        assert any(np.allclose(ln.get_ydata(), gap) for ln in drawn)
        plt.close(fig)

    def test_the_band_is_the_confidence_paths(self, toy_cs):
        """Shaded region spans lower_path to upper_path at every period."""
        cs, gap, _ = toy_cs
        fig = plot_confidence_set(cs, gap)
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        verts = ax.collections[0].get_paths()[0].vertices
        ys = verts[:, 1]
        assert ys.min() == pytest.approx(min(cs.lower_path.min(), cs.upper_path.min()))
        assert ys.max() == pytest.approx(max(cs.lower_path.max(), cs.upper_path.max()))
        plt.close(fig)

    def test_the_pre_period_band_has_no_width(self, toy_cs):
        """The effect path is zero before treatment, so the band closes there."""
        cs, gap, pre = toy_cs
        assert not cs.lower_path[:pre].any()
        assert not cs.upper_path[:pre].any()
        fig = plot_confidence_set(cs, gap)
        plt.close(fig)

    def test_it_draws_on_a_supplied_axis(self, toy_cs):
        cs, gap, _ = toy_cs
        fig, ax = plt.subplots()
        out = plot_confidence_set(cs, gap, ax=ax)
        assert out is fig
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_it_reads_a_fitted_result(self, toy_cs):
        """The usual entry point: hand it what ``fit()`` returned."""
        cs, gap, pre = toy_cs
        fig = plot_confidence_set(_result(cs, gap, pre))
        ax = fig.axes[0]
        verts = ax.collections[0].get_paths()[0].vertices
        assert verts[:, 1].min() == pytest.approx(cs.lower_path.min())
        plt.close(fig)

    def test_the_time_axis_comes_from_the_result(self, toy_cs):
        cs, gap, pre = toy_cs
        fig = plot_confidence_set(_result(cs, gap, pre))
        xs = fig.axes[0].lines[0].get_xdata()
        assert xs[0] == 1970 and xs[-1] == 1970 + len(gap) - 1
        plt.close(fig)


# =========================================================================== #
# the sensitivity sweep
# =========================================================================== #
class TestSensitivityPlot:

    def test_returns_a_figure(self, toy_rows):
        fig = plot_sensitivity(toy_rows)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_interval_per_resolved_tilt(self, toy_rows):
        fig = plot_sensitivity(toy_rows)
        ax = fig.axes[0]
        resolved = [r for r in toy_rows if r.confidence_set is not None]
        bars = [ln for ln in ax.lines
                if len(ln.get_xdata()) == 2
                and ln.get_xdata()[0] == ln.get_xdata()[1]
                and ln.get_linestyle() == "-"]
        assert len(bars) == len(resolved)
        xs = sorted(round(float(ln.get_xdata()[0]), 12) for ln in bars)
        assert xs == pytest.approx(sorted(r.phi for r in resolved))
        for ln in bars:
            phi = float(ln.get_xdata()[0])
            row = next(r for r in resolved if r.phi == pytest.approx(phi))
            lo, hi = sorted(ln.get_ydata())
            assert lo == pytest.approx(row.confidence_set.lower)
            assert hi == pytest.approx(row.confidence_set.upper)
        plt.close(fig)

    def test_an_unresolved_tilt_is_marked_not_dropped(self):
        """A tilt whose search failed is still on the axis, labelled."""
        rows = [SensitivityRow(phi=0.0, confidence_set=None, reason="unbounded")]
        fig = plot_sensitivity(rows)
        ax = fig.axes[0]
        assert "unbounded" in " ".join(t.get_text() for t in ax.texts)
        plt.close(fig)

    def test_a_long_reason_does_not_stretch_the_figure(self):
        """The label is an annotation on the axis, not a caption around it."""
        short = [SensitivityRow(phi=0.0, confidence_set=None, reason="empty")]
        long = [SensitivityRow(phi=0.0, confidence_set=None,
                               reason="the upper bound was not found " * 12)]
        boxes = []
        for rows in (short, long):
            fig = plot_sensitivity(rows)
            fig.canvas.draw()
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            boxes.append((bb.width, bb.height))
            plt.close(fig)
        assert boxes[0] == pytest.approx(boxes[1], rel=1e-6)

    def test_it_marks_the_breakdown_point(self, toy_rows):
        fig = plot_sensitivity(toy_rows)
        ax = fig.axes[0]
        assert any(abs(ln.get_ydata()[0]) < 1e-12 for ln in ax.lines
                   if len(ln.get_ydata()))
        plt.close(fig)

    def test_it_draws_on_a_supplied_axis(self, toy_rows):
        fig, ax = plt.subplots()
        assert plot_sensitivity(toy_rows, ax=ax) is fig
        plt.close(fig)

    def test_a_list_of_something_else_is_refused(self):
        with pytest.raises(MlsynthPlottingError, match="sensitivity_sweep"):
            plot_sensitivity([{"phi": 0.0, "lower": -1.0, "upper": 1.0}])

    def test_it_reads_a_fitted_result(self, toy_cs, toy_rows):
        cs, gap, pre = toy_cs
        details = {"effect_class": cs.kind, "phi": cs.phi,
                   "lower_path": cs.lower_path.tolist(),
                   "upper_path": cs.upper_path.tolist(),
                   "sensitivity": [
                       {"phi": r.phi,
                        "lower": None if r.confidence_set is None else r.confidence_set.lower,
                        "upper": None if r.confidence_set is None else r.confidence_set.upper,
                        "contains_zero": r.contains_zero,
                        "reason": r.reason}
                       for r in toy_rows],
                   "breakdown_phi": None}
        fig = plot_sensitivity(_result(cs, gap, pre, details=details))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# refusals
# =========================================================================== #
class TestRefusals:

    def test_a_result_from_another_inference_mode_is_refused(self, toy_cs):
        cs, gap, pre = toy_cs
        res = _result(cs, gap, pre, method="in-space placebo",
                      details={"p_value": 0.1})
        with pytest.raises(MlsynthPlottingError, match="placebo_cs"):
            plot_confidence_set(res)

    def test_an_unavailable_set_reports_its_reason(self, toy_cs):
        cs, gap, pre = toy_cs
        res = _result(cs, gap, pre,
                      details={"effect_class": "linear", "phi": 0.0,
                               "unavailable_reason": "the set is unbounded"})
        with pytest.raises(MlsynthPlottingError, match="unbounded"):
            plot_confidence_set(res)

    def test_a_bare_confidence_set_needs_its_gap(self, toy_cs):
        cs, _, _ = toy_cs
        with pytest.raises(MlsynthPlottingError, match="gap"):
            plot_confidence_set(cs)

    def test_a_gap_of_the_wrong_length_is_refused(self, toy_cs):
        cs, gap, _ = toy_cs
        with pytest.raises(MlsynthPlottingError, match="length"):
            plot_confidence_set(cs, gap[:-2])

    def test_a_result_carrying_no_effect_paths_is_refused(self, toy_cs):
        cs, gap, pre = toy_cs
        res = _result(cs, gap, pre, details={"effect_class": "linear"})
        with pytest.raises(MlsynthPlottingError, match="effect paths"):
            plot_confidence_set(res)

    def test_a_result_carrying_no_gap_is_refused(self, toy_cs):
        cs, gap, pre = toy_cs
        res = _result(cs, gap, pre)
        res.time_series = TimeSeriesResults(estimated_gap=None)
        with pytest.raises(MlsynthPlottingError, match="no estimated gap"):
            plot_confidence_set(res)

    def test_a_result_without_a_sweep_is_refused(self, toy_cs):
        cs, gap, pre = toy_cs
        with pytest.raises(MlsynthPlottingError, match="sweep|sensitivity"):
            plot_sensitivity(_result(cs, gap, pre))

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(MlsynthPlottingError, match="empty|no tilt"):
            plot_sensitivity([])

"""SHC's conformal inference, reached through the standard result contract.

The test of Chen, Yang & Yang (2024, footnote 21) and the Andrews-Genton band
are computed by :func:`run_conformal_inference` and were reachable only through
``res.inference_detail``, a per-estimator dataclass. A caller reading the
contract found ``res.inference`` carrying a method name and a p-value, with
``ci_lower``/``ci_upper`` empty, ``details`` holding a dataclass instead of a
mapping, and the band absent from ``res.time_series`` -- so
``has_prediction_interval`` was False on a fit that had computed one.

These tests pin the inference onto the contract: the scalar summaries on
``InferenceResults``, the per-period band on ``TimeSeriesResults`` where every
other band in the library lives, and ``details`` as a mapping a caller can
subscript.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth import SHC
from mlsynth.utils.shc_helpers.simulation import simulate_shc_panel

_M, _N = 25, 4


@pytest.fixture(scope="module")
def fitted():
    """One SHC fit with a real effect, so a rejection means something."""
    df, info = simulate_shc_panel(m=_M, h=4, n=_N, seed=3)
    df = df.copy()
    df.loc[df.time > info["T_o"], "y"] += -1.5
    return SHC({"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
                "time": "time", "m": _M, "display_graphs": False}).fit()


def _fit(**extra):
    df, info = simulate_shc_panel(m=_M, h=4, n=_N, seed=3)
    df = df.copy()
    df.loc[df.time > info["T_o"], "y"] += -1.5
    cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
           "time": "time", "m": _M, "display_graphs": False}
    cfg.update(extra)
    return SHC(cfg).fit()


# =========================================================================== #
# InferenceResults
# =========================================================================== #
class TestInferenceResults:

    def test_the_test_is_reported_on_the_contract(self, fitted):
        inf = fitted.inference
        assert inf is not None
        assert "conformal" in inf.method
        assert 0.0 <= inf.p_value <= 1.0

    def test_details_is_a_mapping_a_caller_can_subscript(self, fitted):
        """The contract's ``details`` is read with ``[]``, not ``getattr``."""
        d = fitted.inference.details
        assert isinstance(d, dict)
        for key in ("test_statistic", "critical_values", "reject",
                    "num_resamples", "levels"):
            assert key in d

    def test_the_statistic_and_its_critical_values_travel_together(self, fitted):
        """A statistic without its critical values cannot be acted on."""
        d = fitted.inference.details
        stat = float(d["test_statistic"])
        crit = d["critical_values"]
        assert set(crit) == {0.01, 0.05, 0.10}
        # a laxer level has a lower bar
        assert crit[0.10] <= crit[0.05] <= crit[0.01]
        for level, rejected in d["reject"].items():
            assert rejected is (stat > crit[level])

    def test_the_statistic_is_the_papers_scaled_absolute_sum(self, fitted):
        r"""``S = n^{-1/2} sum |eps_t|`` over the post window (footnote 21)."""
        gap = np.asarray(fitted.time_series.estimated_gap, dtype=float)[_M:]
        expected = float(np.sum(np.abs(gap)) / np.sqrt(gap.size))
        assert float(fitted.inference.details["test_statistic"]) == pytest.approx(
            expected, rel=1e-12)

    def test_the_confidence_level_is_reported(self, fitted):
        assert fitted.inference.confidence_level == pytest.approx(0.90)

    def test_a_real_effect_is_rejected_at_every_level(self, fitted):
        d = fitted.inference.details
        assert d["reject"][0.10] and d["reject"][0.01]


# =========================================================================== #
# the band, where every other band lives
# =========================================================================== #
class TestPredictionInterval:

    def test_the_band_is_on_the_time_series_contract(self, fitted):
        ts = fitted.time_series
        assert ts.has_prediction_interval
        assert ts.prediction_interval_level == pytest.approx(0.90)
        assert "andrews-genton" in ts.prediction_interval_kind

    def test_the_band_is_aligned_to_the_time_axis(self, fitted):
        """Aligned to ``time_periods``, NaN over the block's pre-window."""
        ts = fitted.time_series
        T = len(ts.time_periods)
        lo = np.asarray(ts.counterfactual_lower, dtype=float)
        hi = np.asarray(ts.counterfactual_upper, dtype=float)
        assert lo.shape == hi.shape == (T,)
        assert np.isnan(lo[:_M]).all()
        assert np.isfinite(lo[_M:]).all()
        assert np.isfinite(hi[_M:]).all()

    def test_the_band_brackets_the_counterfactual(self, fitted):
        ts = fitted.time_series
        cf = np.asarray(ts.counterfactual_outcome, dtype=float)[_M:]
        lo = np.asarray(ts.counterfactual_lower, dtype=float)[_M:]
        hi = np.asarray(ts.counterfactual_upper, dtype=float)[_M:]
        assert (lo <= cf + 1e-9).all()
        assert (cf <= hi + 1e-9).all()

    def test_it_is_the_same_band_the_helper_computed(self, fitted):
        """The contract carries the fit's band, not a recomputed one."""
        detail = fitted.inference_detail
        lo = np.asarray(fitted.time_series.counterfactual_lower, dtype=float)
        hi = np.asarray(fitted.time_series.counterfactual_upper, dtype=float)
        assert lo[_M:] == pytest.approx(np.asarray(detail.conformal_lower, float))
        assert hi[_M:] == pytest.approx(np.asarray(detail.conformal_upper, float))


# =========================================================================== #
# the variants stay distinguishable on the contract
# =========================================================================== #
class TestVariants:

    def test_the_exact_test_names_its_permutation_scheme(self):
        for scheme in ("moving_block", "iid"):
            extra = {"inference_method": "exact", "permutation_scheme": scheme}
            if scheme == "iid":
                extra["num_permutations"] = 200
            inf = _fit(**extra).inference
            assert scheme in inf.method
            assert "scheme" in inf.details
            assert inf.details["scheme"] == scheme

    def test_the_bootstrap_default_names_itself(self, fitted):
        assert fitted.inference.method == "conformal_permutation"
        assert fitted.inference.details["num_resamples"] == 1000

    def test_the_two_methods_report_the_same_statistic(self):
        """Only the null distribution differs; the observed statistic does not."""
        a = _fit().inference.details["test_statistic"]
        b = _fit(inference_method="exact").inference.details["test_statistic"]
        assert float(a) == pytest.approx(float(b), rel=1e-12)


# =========================================================================== #
# the guard on a band that does not fit the axis
# =========================================================================== #
class TestBandAlignmentGuard:

    def test_a_band_that_does_not_span_the_window_is_not_published(self, fitted):
        """Publishing a misaligned band would put bounds on the wrong periods.

        The contract's band is indexed by ``time_periods``, so a band whose
        length does not complete the window cannot be placed. The fields stay
        empty and ``has_prediction_interval`` stays False; the numbers are still
        on ``inference_detail``, which is indexed by the post window alone.
        """
        import dataclasses

        from mlsynth.utils.shc_helpers.structures import SHCResults

        detail = dataclasses.replace(
            fitted.inference_detail,
            conformal_lower=np.array([]), conformal_upper=np.array([]))
        rebuilt = SHCResults(
            inputs=fitted.inputs, design=fitted.design,
            att_value=fitted.att_value, att_percent=fitted.att_percent,
            observed=fitted.observed, cf_window=fitted.cf_window,
            gap_window=fitted.gap_window, time_labels=fitted.time_labels,
            fit_diagnostics_detail=fitted.fit_diagnostics_detail,
            inference_detail=detail, metadata=dict(fitted.metadata))
        assert rebuilt.time_series.counterfactual_lower is None
        assert rebuilt.time_series.has_prediction_interval is False
        # the test itself is unaffected: only the band could not be placed
        assert rebuilt.inference.details["test_statistic"] == pytest.approx(
            fitted.inference.details["test_statistic"])


# =========================================================================== #
# backward compatibility
# =========================================================================== #
class TestInferenceDetailStillWorks:

    def test_the_dataclass_accessor_is_unchanged(self, fitted):
        from mlsynth.utils.shc_helpers.structures import SHCInference
        detail = fitted.inference_detail
        assert isinstance(detail, SHCInference)
        assert detail.p_value == fitted.inference.p_value
        assert detail.null_distribution.size == 1000

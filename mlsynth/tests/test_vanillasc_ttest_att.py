"""The reported ATT and the reported interval must describe one estimator.

``VanillaSC(inference="ttest")`` runs the Chernozhukov, Wuthrich & Zhu debiased
SC t-test. That test is about the debiased estimator
:math:`\\widehat{\\tau} = K^{-1}\\sum_k \\widehat{\\tau}_k`, not about the plain
SC ATT -- debiasing is the whole point of the method, because the plain SC ATT
is biased when the weights are estimated in high dimension relative to
:math:`T_0`.

The two were reported side by side. ``effects.att`` came from the full-sample
SC counterfactual and ``inference.ci_lower/ci_upper`` came from the debiased
estimator, so the interval was not an interval for the number printed next to
it. On the carbon-tax replication the paper's Table 5(a) reports an ATT of
-0.27; the debiased estimate is -0.2739 and is what
``benchmarks/cases/cwz_ttest.py`` validates, while ``effects.att`` read -0.2837,
three times further from the paper. On California Proposition 99 with
``ttest_K=2`` the reported 90% interval was [-18.13, -16.56] against a reported
ATT of -19.51: the interval excluded its own point estimate.

The fix keeps one estimator per result. Under ``inference="ttest"`` the
post-period counterfactual becomes the fold average of
:math:`\\mathbf{x}_t'\\widehat{\\mathbf{w}}_{(k)} + b_k`, where :math:`b_k` is
fold ``k``'s held-out pre-period gap. Its post-period mean gap is
:math:`\\widehat{\\tau}` identically,

.. math::

   \\operatorname{mean}_{t>T_0}\\bigl(y_t - K^{-1}\\textstyle\\sum_k
   (\\mathbf{x}_t'\\widehat{\\mathbf{w}}_{(k)} + b_k)\\bigr)
   = K^{-1}\\sum_k (\\text{post gap}_k - b_k) = \\widehat{\\tau},

so the ATT, the gap series, the interval and the p-value all describe the
estimator the method actually delivers. The pre-period counterfactual is left
as the SC fit, so pre-period fit diagnostics still describe the SC match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"


def _basque():
    f = _BASEDATA / "basque_jasa.csv"
    if not f.exists():
        pytest.skip("basque_jasa.csv not available")
    df = pd.read_csv(f)
    df = df[df.regionname != "Spain (Espana)"].copy()
    df["treated"] = ((df.regionname == "Basque Country (Pais Vasco)")
                     & (df.year >= 1975)).astype(int)
    return dict(df=df, outcome="gdpcap", treat="treated",
                unitid="regionname", time="year")


def _prop99():
    f = _BASEDATA / "augmented_cali_long.csv"
    if not f.exists():
        pytest.skip("augmented_cali_long.csv not available")
    return dict(df=pd.read_csv(f), outcome="cigsale", treat="Proposition 99",
                unitid="state", time="year")


def _carbontax():
    f = _BASEDATA / "carbontax_data.dta"
    if not f.exists():
        pytest.skip("carbontax_data.dta not available")
    ct = pd.read_stata(f)
    ct["treated"] = ((ct.country == "Sweden") & (ct.year >= 1990)).astype(int)
    return dict(df=ct, outcome="CO2_transport_capita", treat="treated",
                unitid="country", time="year")


def _n_post_from(spec) -> int:
    """Post-period count, read off the treat indicator in the input panel.

    ``time_series.intervention_time`` is ``None`` here, so the split comes from
    the data, not the result.
    """
    df, treat, time = spec["df"], spec["treat"], spec["time"]
    return int(df.loc[df[treat] == 1, time].nunique())


def _fit(spec, **over):
    cfg = dict(backend="outcome-only", alpha=0.1, display_graphs=False)
    cfg.update(spec)
    cfg.update(over)
    return VanillaSC(cfg).fit()


# =========================================================================
# 1. One estimator per result
# =========================================================================

class TestTheAttAndTheIntervalAgree:

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_reported_att_is_the_debiased_estimator(self, K):
        res = _fit(_prop99(), inference="ttest", ttest_K=K)
        assert res.effects.att == pytest.approx(
            res.inference.details["att_debiased"], rel=1e-9, abs=1e-10)

    @pytest.mark.parametrize("K", [2, 3, 5])
    def test_the_interval_is_centred_on_the_reported_att(self, K):
        res = _fit(_prop99(), inference="ttest", ttest_K=K)
        inf = res.inference
        midpoint = 0.5 * (inf.ci_lower + inf.ci_upper)
        assert midpoint == pytest.approx(res.effects.att, rel=1e-9, abs=1e-10)

    def test_prop99_k2_interval_contains_its_own_point_estimate(self):
        """The case that made the mismatch visible instead of merely latent.

        Before the fix: ATT -19.51 with a 90% interval of [-18.13, -16.56].
        """
        res = _fit(_prop99(), inference="ttest", ttest_K=2)
        inf = res.inference
        assert inf.ci_lower <= res.effects.att <= inf.ci_upper, (
            f"90% interval [{inf.ci_lower:.4f}, {inf.ci_upper:.4f}] excludes "
            f"the reported ATT {res.effects.att:.4f}")

    def test_the_p_value_tests_the_reported_att(self):
        """``tstat`` is the reported ATT over the reported standard error."""
        res = _fit(_prop99(), inference="ttest", ttest_K=3)
        d = res.inference.details
        assert d["tstat"] == pytest.approx(res.effects.att / d["se"], rel=1e-9)

    def test_the_naive_sc_att_is_still_reported(self):
        """Keeping it is how a reader sees the size of the bias correction."""
        res = _fit(_prop99(), inference="ttest", ttest_K=3)
        plain = _fit(_prop99(), inference=False)
        assert res.inference.details["att_naive"] == pytest.approx(
            plain.effects.att, rel=1e-9)

    def test_standard_error_reaches_the_contract_field(self):
        """``InferenceResults.standard_error`` exists; it was left empty."""
        res = _fit(_basque(), inference="ttest", ttest_K=3)
        assert res.inference.standard_error == pytest.approx(
            res.inference.details["se"], rel=1e-12)


# =========================================================================
# 2. The counterfactual the ATT comes from
# =========================================================================

class TestTheCounterfactualIsCoherent:

    def test_post_period_gap_series_averages_to_the_reported_att(self):
        spec = _basque()
        res = _fit(spec, inference="ttest", ttest_K=3)
        ts = res.time_series
        obs = np.asarray(ts.observed_outcome, dtype=float)
        cf = np.asarray(ts.counterfactual_outcome, dtype=float)
        gap = np.asarray(ts.estimated_gap, dtype=float)
        n_pre = len(obs) - _n_post_from(spec)
        assert np.mean((obs - cf)[n_pre:]) == pytest.approx(
            res.effects.att, rel=1e-8, abs=1e-10)
        assert np.mean(gap[n_pre:]) == pytest.approx(
            res.effects.att, rel=1e-8, abs=1e-10)

    def test_the_pre_period_counterfactual_is_the_sc_fit(self):
        """Pre-period fit diagnostics still describe the SC match."""
        spec = _basque()
        tt = _fit(spec, inference="ttest", ttest_K=3)
        plain = _fit(spec, inference=False)
        n_pre = len(np.asarray(tt.time_series.observed_outcome)) - _n_post_from(spec)
        a = np.asarray(tt.time_series.counterfactual_outcome, dtype=float)[:n_pre]
        b = np.asarray(plain.time_series.counterfactual_outcome, dtype=float)[:n_pre]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)

    def test_the_post_period_counterfactual_moves(self):
        """It has to, or the ATT could not have changed."""
        spec = _prop99()
        tt = _fit(spec, inference="ttest", ttest_K=2)
        plain = _fit(spec, inference=False)
        n_pre = len(np.asarray(tt.time_series.observed_outcome)) - _n_post_from(spec)
        a = np.asarray(tt.time_series.counterfactual_outcome, dtype=float)[n_pre:]
        b = np.asarray(plain.time_series.counterfactual_outcome, dtype=float)[n_pre:]
        assert not np.allclose(a, b, rtol=1e-6, atol=1e-8)


# =========================================================================
# 3. Nothing else moves
# =========================================================================

class TestOtherModesAreUntouched:

    @pytest.mark.parametrize("mode", [False, "conformal"])
    def test_other_inference_modes_keep_the_sc_att(self, mode):
        res = _fit(_basque(), inference=mode)
        plain = _fit(_basque(), inference=False)
        assert res.effects.att == pytest.approx(plain.effects.att, rel=1e-9)

    def test_carbontax_headline_now_matches_the_paper(self):
        """Table 5(a) reports -0.27; that is the debiased estimate."""
        res = _fit(_carbontax(), inference="ttest", ttest_K=3)
        assert res.effects.att == pytest.approx(-0.27, abs=0.01)
        assert res.inference.ci_lower == pytest.approx(-0.41, abs=0.01)
        assert res.inference.ci_upper == pytest.approx(-0.14, abs=0.01)

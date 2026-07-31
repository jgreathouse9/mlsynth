"""Choosing a single TSSC variant, and running it without inference.

``TSSC`` fits all four SC-class variants (SC, MSCa, MSCb, MSCc), runs Step 1 --
a subsampling test of the SC pretrends restriction -- to recommend one, and
attaches a subsampling confidence interval to each variant. That is the
estimator, and it stays the default.

Two situations want less than that. A caller who wants the demeaned synthetic
control specifically wants ``MSCa`` and nothing else (Ferman & Pinto 2021;
``benchmarks/cases/ferman_demeaned_basque.py`` already reaches into
``variants["MSCa"]`` for exactly this). And a Monte Carlo wants a point estimate
many thousands of times and never reads a confidence interval. At the default
500 draws a single fit costs about 12 s, so one 1000-replication arm is over
three hours of subsampling that nothing consumes.

Hence two knobs:

``method``
    Fit one named variant instead of all four. Step 1 is *skipped*, not
    overridden -- there is no recommendation to make once the variant is
    chosen -- so ``selection`` comes back as ``None``.
``inference``
    Skip the subsampling confidence intervals.

The load-bearing property is that neither knob may move a number. Selecting
``MSCa`` must give bit-identical estimates to reading ``variants["MSCa"]`` off a
full run, and turning inference off must not perturb the point estimate. Both
are asserted below rather than assumed, because a change that silently altered
the estimate while making it faster would look like a success.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

METHODS = ("SC", "MSCa", "MSCb", "MSCc")


@pytest.fixture(scope="module")
def panel():
    """A small balanced panel with one treated unit.

    Deliberately modest (16 donors, 40 periods) so the full four-variant run
    with inference stays affordable inside the suite -- several tests below
    need a full run to compare against.
    """
    rng = np.random.default_rng(11)
    rows = []
    for j in range(17):
        level = rng.normal(0.0, 1.0)
        slope = rng.normal(0.05, 0.02)
        for t in range(40):
            rows.append({"u": f"u{j}", "t": t,
                         "y": level + slope * t + rng.normal(0.0, 0.25)})
    d = pd.DataFrame(rows)
    d["treat"] = ((d.u == "u0") & (d.t >= 30)).astype(int)
    return d


def _fit(panel, **kw):
    from mlsynth import TSSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TSSC({"df": panel, "outcome": "y", "treat": "treat", "unitid": "u",
                     "time": "t", "display_graphs": False, "seed": 7,
                     "draws": 40, **kw}).fit()


@pytest.fixture(scope="module")
def full(panel):
    """One full default run, reused by the equivalence tests."""
    return _fit(panel)


class TestTheDefaultIsUnchanged:
    """Adding the options must not move any existing result."""

    def test_all_four_variants_are_still_fit(self, full):
        assert set(full.variants) == set(METHODS)

    def test_step_one_still_runs_and_recommends(self, full):
        assert full.selection is not None
        assert full.selection.recommended in METHODS

    def test_inference_is_still_populated(self, full):
        ci = full.variants["MSCa"].att_ci
        assert len(ci) == 2 and np.all(np.isfinite(ci))

    def test_omitting_the_new_options_is_identical_to_passing_the_defaults(self, panel):
        a = _fit(panel)
        b = _fit(panel, method=None, inference=True)
        assert a.summary.effects.att == b.summary.effects.att
        for m in METHODS:
            assert a.variants[m].att == b.variants[m].att


class TestSelectingOneVariant:
    """``method="MSCa"`` fits that variant and nothing else."""

    @pytest.mark.parametrize("m", METHODS)
    def test_only_the_requested_variant_is_fit(self, panel, m):
        res = _fit(panel, method=m)
        assert set(res.variants) == {m}

    @pytest.mark.parametrize("m", METHODS)
    def test_the_estimate_is_bit_identical_to_the_full_run(self, panel, full, m):
        """The whole point: this is a cheaper route to the same number.

        Bit-identical, not approximately equal -- the variant fit does not
        depend on which other variants were fit alongside it, and if that ever
        stops being true it is a bug worth failing on.
        """
        res = _fit(panel, method=m)
        assert res.variants[m].att == full.variants[m].att
        np.testing.assert_array_equal(
            np.asarray(res.variants[m].counterfactual, dtype=float),
            np.asarray(full.variants[m].counterfactual, dtype=float))

    def test_the_summary_is_the_requested_variant(self, panel, full):
        res = _fit(panel, method="MSCb")
        assert res.summary.effects.att == full.variants["MSCb"].att

    def test_step_one_is_skipped_not_overridden(self, panel):
        """``selection`` is absent rather than pointing at the forced variant.

        Stated as behaviour because fabricating a recommendation equal to the
        forced method would be the natural shortcut, and it would misreport
        Step 1 as having endorsed a choice the user made.
        """
        res = _fit(panel, method="MSCa")
        assert res.selection is None

    def test_the_forced_variant_is_recorded_somewhere_visible(self, panel):
        """A reader of the result must be able to tell which variant produced
        it, now that ``selection`` no longer says."""
        res = _fit(panel, method="MSCa")
        details = res.summary.method_details
        assert details is not None
        assert "MSCa" in str(details.method_name) or "MSCa" in str(details)


class TestTheResultSurfaceStillWorks:
    """The flat accessors must not break when Step 1 was skipped.

    ``TSSCResults.att`` / ``att_ci`` / ``recommended`` all resolve through
    ``selection.recommended`` on the default path. With a forced variant there
    is no selection, and the obvious implementation raises ``AttributeError``
    on ``None`` -- which would make ``method=`` unusable for anyone who reads
    the result the normal way, or who leaves plotting on.
    """

    def test_the_flat_accessors_resolve_to_the_forced_variant(self, panel, full):
        res = _fit(panel, method="MSCa")
        assert res.att == full.variants["MSCa"].att
        assert res.recommended.att == full.variants["MSCa"].att
        assert len(res.att_ci) == 2

    def test_recommended_method_names_the_forced_variant(self, panel):
        res = _fit(panel, method="MSCb")
        assert res.recommended_method == "MSCb"

    def test_the_standard_submodels_are_populated(self, panel):
        """A forced-variant result is still an EffectResult like any other."""
        res = _fit(panel, method="MSCa")
        for field in ("effects", "time_series", "weights", "method_details"):
            assert getattr(res, field) is not None

    def test_plotting_works_with_a_forced_variant(self, panel):
        """Left broken, anyone using the default display_graphs=True with
        method= would hit the plotter's `results.recommended`."""
        import matplotlib
        matplotlib.use("Agg")
        from mlsynth.utils.tssc_helpers.plotter import plot_tssc
        import matplotlib.pyplot as plt

        res = _fit(panel, method="MSCa")
        plot_tssc(res)
        plt.close("all")


class TestTurningInferenceOff:
    """``inference=False`` drops the subsampling, and nothing else."""

    def test_the_point_estimate_is_unchanged(self, panel, full):
        res = _fit(panel, method="MSCa", inference=False)
        assert res.variants["MSCa"].att == full.variants["MSCa"].att

    def test_no_confidence_interval_is_reported(self, panel):
        res = _fit(panel, method="MSCa", inference=False)
        ci = res.variants["MSCa"].att_ci
        assert ci is None or not np.any(np.isfinite(np.asarray(ci, dtype=float)))

    def test_a_nan_interval_is_not_passed_off_as_an_interval(self, panel):
        """If the CI comes back as NaN rather than None, the standardized
        ``inference`` sub-model must not present it as a real interval."""
        res = _fit(panel, method="MSCa", inference=False)
        inf = res.summary.inference
        if inf is not None and getattr(inf, "ci", None) is not None:
            assert not np.all(np.isfinite(np.asarray(inf.ci, dtype=float)))

    def test_it_is_materially_faster(self, panel):
        """The reason the option exists, asserted so it cannot regress into a
        no-op that merely hides the interval."""
        import time

        t0 = time.perf_counter(); _fit(panel, method="MSCa", inference=False)
        fast = time.perf_counter() - t0
        t0 = time.perf_counter(); _fit(panel, method="MSCa", inference=True, draws=400)
        slow = time.perf_counter() - t0
        assert fast < slow / 3.0

    def test_the_seed_is_irrelevant_when_inference_is_off(self, panel):
        """Nothing stochastic should remain, so two seeds must agree exactly."""
        a = _fit(panel, method="MSCa", inference=False, seed=1)
        b = _fit(panel, method="MSCa", inference=False, seed=999)
        assert a.variants["MSCa"].att == b.variants["MSCa"].att


class TestTheCombinationsAreCoherent:
    """Which pairings are allowed, and what they mean."""

    def test_inference_off_requires_a_method(self, panel):
        """Step 1 is itself a subsampling procedure, so with inference off there
        is no way to recommend a variant and no way to know which variant the
        summary should describe. Refused at construction rather than resolved
        by guessing.
        """
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="method"):
            _fit(panel, inference=False)

    def test_a_method_with_inference_on_still_gives_an_interval(self, panel):
        res = _fit(panel, method="MSCa", inference=True)
        ci = res.variants["MSCa"].att_ci
        assert len(ci) == 2 and np.all(np.isfinite(ci))
        assert res.selection is None      # still no Step 1


class TestFailuresAreReported:
    """Invalid input raises a translated error naming what was wrong."""

    def test_unknown_variant_name(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError) as exc:
            _fit(panel, method="MSCd")
        assert "MSCd" in str(exc.value) or "MSCa" in str(exc.value)

    def test_the_error_lists_the_valid_variants(self, panel):
        """A caller who mistypes should not have to read the source."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError) as exc:
            _fit(panel, method="msca")          # right variant, wrong case
        assert all(m in str(exc.value) for m in METHODS)

    def test_lowercase_is_not_silently_accepted(self, panel):
        """Deciding this explicitly: the variant names are proper nouns from
        Li & Shankar's Table 1, and quietly upcasing would make ``MSCa`` and
        ``msca`` two spellings of one thing in some places and not others."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError):
            _fit(panel, method="msca")

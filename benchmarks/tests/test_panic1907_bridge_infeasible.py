"""The Panic-of-1907 panel does not identify the bridge-based proximal methods.

``proximal_panic1907`` pinned ``dr_att`` and ``pipw_att`` alongside the PI /
PI-S / PI-P cells. Both of those methods route through
``fit_treatment_bridge``, which solves the square moment system

    E_pre[exp((1,Z) beta) (1,W)] = E_post[(1,W)]

and on this panel it does not solve: the largest absolute residual is 0.401
against a tolerance of 1e-06. Until #420 the solver returned its stalled
``sol.x`` regardless, so the case passed on two numbers built from a vector that
satisfied nothing, and those numbers were then pinned as regression guards.

The design is what makes it unsolvable, so it is reported rather than retried.
The system is 49-dimensional -- an intercept and 48 donor-trust proxies -- and
asks for the post-period mean of ``(1,W)`` to be reproducible by an exponential
tilt of the pre-period. With T0 = 229 against T1 = 182 that tilt does not exist
in this sample.

These tests pin the diagnosis, so that the two cells cannot be restored without
the panel or the estimator having changed: DR and PIPW raise here, the four
bridge-free methods do not, and the case asks for exactly the latter.
"""
from __future__ import annotations

import warnings

import pytest

from mlsynth.exceptions import MlsynthEstimationError

pytest.importorskip("pandas")

BRIDGE_FREE = ["PI", "PIS", "PIPost", "SPSC"]
BRIDGE_BASED = ["DR", "PIPW"]


@pytest.fixture(scope="module")
def panel():
    from benchmarks.cases.proximal_panic1907 import _panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _panel()


def _att(panel, methods):
    from benchmarks.cases.proximal_panic1907 import _att as att
    df, donors, surrogates = panel
    return att(df, donors, surrogates, methods)


class TestTheBridgeIsInfeasibleOnThisPanel:
    @pytest.mark.parametrize("method", BRIDGE_BASED)
    def test_the_bridge_methods_raise(self, panel, method):
        with pytest.raises(MlsynthEstimationError):
            _att(panel, [method])

    @pytest.mark.parametrize("method", BRIDGE_BASED)
    def test_the_raise_names_the_moment_condition(self, panel, method):
        with pytest.raises(MlsynthEstimationError, match="moment condition"):
            _att(panel, [method])

    def test_the_residual_is_far_above_tolerance(self, panel):
        """Not a near miss a tighter solve would close."""
        with pytest.raises(MlsynthEstimationError) as exc:
            _att(panel, ["DR"])
        assert "0.4" in str(exc.value)


class TestTheBridgeFreeMethodsAreUnaffected:
    @pytest.mark.parametrize("method", BRIDGE_FREE)
    def test_they_still_estimate(self, panel, method):
        got = _att(panel, [method])[method]
        assert got == pytest.approx(got)   # finite, not nan

    def test_they_match_the_pinned_table3_values(self, panel):
        got = _att(panel, ["PI", "PIS", "PIPost"])
        assert got["PI"] == pytest.approx(-1.148, abs=0.05)
        assert got["PIS"] == pytest.approx(-1.148, abs=0.05)
        assert got["PIPost"] == pytest.approx(-1.220, abs=0.05)


class TestTheCaseAsksOnlyForWhatIsIdentified:
    def test_it_does_not_pin_the_bridge_methods(self):
        from benchmarks.cases import proximal_panic1907 as case
        assert not {"dr_att", "pipw_att"} & set(case.EXPECTED)

    def test_it_still_pins_the_bridge_free_methods(self):
        from benchmarks.cases import proximal_panic1907 as case
        assert {"pi_att", "pis_att", "pipost_att", "spsc_att"} <= set(case.EXPECTED)

    def test_the_case_runs_without_raising(self):
        """The whole point: the suite gets an answer from this case again."""
        from benchmarks.cases import proximal_panic1907 as case
        from benchmarks.compare import BenchmarkSkipped
        try:
            got = case.run()
        except BenchmarkSkipped:
            pytest.skip("reference clone unavailable")
        assert set(case.EXPECTED) <= set(got)

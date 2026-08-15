"""What the guard costs a placebo fit, asserted rather than left to a profiler.

A default ``vce="placebo"`` fit poses 1000 simplex programs and asked
``gram_reduction_is_safe`` about every one of them, each answer a full singular
spectrum. On a 101x120 panel all 1000 came back ``True``, decided by a ratio
that clears the 1e-8 tolerance by four to seven orders of magnitude.

The cause was not that the guard is wrong. It is that nothing anywhere said a
guard must cost less than the solve it protects, so the cost was free to grow
past it: 4.25 s of a 16.6 s three-fit profile, against the batched solver's own
4.33 s. These tests write that contract down.

Two families, and only one of them can be certified:

* the unit-weight program is ridge augmented, so ``lambda_min >= r`` gives a
  free lower bound on its conditioning and its SVD goes away entirely;
* the time-weight program carries no ridge, so the same bound is vacuous and it
  keeps paying. Asserted here as a fact about the fix's reach, so that half is
  visible rather than assumed away.

The binding assertions are counts, not clocks -- wall time carries enough
run-to-run noise at these sizes to hide a regression this size.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.utils.bilevel import minnorm
from mlsynth.utils.sdid_helpers import weights as sdid_weights


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(n_donors=24, n_periods=20, n_pre=14, effect=-2.0, seed=5):
    """Small enough to fit repeatedly, wide enough that both programs run."""
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    factors = np.cumsum(rng.standard_normal((n_periods, 3)) * 0.3, axis=0)
    loadings = rng.uniform(0.2, 1.2, (n_units, 3))
    outcome = (loadings @ factors.T
               + rng.standard_normal((n_units, n_periods)) * 0.4 + 10.0)
    treat = np.zeros((n_units, n_periods), dtype=int)
    treat[0, n_pre:] = 1
    outcome[0, n_pre:] += effect
    return pd.DataFrame(
        [{"id": f"u{i:03d}", "time": t + 1, "y": outcome[i, t], "d": treat[i, t]}
         for i in range(n_units) for t in range(n_periods)]
    )


def _fit(df, draws=40, **overrides):
    config = {"df": df, "outcome": "y", "treat": "d", "unitid": "id",
              "time": "time", "display_graphs": False, "vce": "placebo",
              "B": draws}
    config.update(overrides)
    return SDID(config).fit()


class _SvdSpy:
    def __init__(self, monkeypatch):
        self.shapes = []
        real = np.linalg.svd

        def counting(a, *args, **kwargs):
            self.shapes.append(np.asarray(a).shape)
            return real(a, *args, **kwargs)

        monkeypatch.setattr(np.linalg, "svd", counting)

    def __len__(self):
        return len(self.shapes)


def _force_slow_guard(monkeypatch):
    """Restore the pre-change behaviour: every problem gets its own spectrum."""
    monkeypatch.setattr(
        sdid_weights, "ridged_gram_reduction_is_safe",
        lambda design, ridge, tol=1e-8: minnorm.gram_reduction_is_safe(
            np.vstack([design, np.sqrt(ridge) * np.eye(design.shape[1])])
            if ridge > 0.0 else design, tol))


# --------------------------------------------------------------------------- #
# 1. smoke
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_placebo_fit_runs(self):
        result = _fit(_panel())
        assert np.isfinite(result.effects.att)
        assert np.isfinite(result.inference.standard_error)


# --------------------------------------------------------------------------- #
# 2. the numbers do not move
# --------------------------------------------------------------------------- #
class TestAnswerUnchanged:
    def test_att_and_standard_error_identical(self, monkeypatch):
        df = _panel()
        fast = _fit(df)
        _force_slow_guard(monkeypatch)
        slow = _fit(df)
        assert fast.effects.att == pytest.approx(slow.effects.att, rel=0, abs=1e-12)
        assert (fast.inference.standard_error
                == pytest.approx(slow.inference.standard_error, rel=0, abs=1e-12))

    def test_weights_identical(self, monkeypatch):
        df = _panel()
        fast = _fit(df)
        _force_slow_guard(monkeypatch)
        slow = _fit(df)
        for name in ("donor_weights", "time_weights", "unit_weights"):
            got, want = getattr(fast.weights, name), getattr(slow.weights, name)
            if want is None:
                assert got is None
                continue
            assert set(got) == set(want)
            np.testing.assert_allclose(
                [got[k] for k in want], [want[k] for k in want], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. the cost contract
# --------------------------------------------------------------------------- #
class TestCostContract:
    def test_the_ridged_family_stops_paying_for_a_spectrum(self, monkeypatch):
        """The unit-weight program is ridge augmented, so its conditioning is
        bounded below for free and its SVD goes away."""
        df = _panel()
        spy_fast = _SvdSpy(monkeypatch)
        _fit(df)
        fast = len(spy_fast)

        monkeypatch.undo()
        _force_slow_guard(monkeypatch)
        spy_slow = _SvdSpy(monkeypatch)
        _fit(df)
        slow = len(spy_slow)

        assert slow > fast, "the certificate removed no work"
        assert fast <= slow * 0.6, (
            f"expected the ridged half of the guard's SVDs to go; {fast} of {slow}")

    def test_a_guard_costs_less_than_what_it_guards(self, monkeypatch):
        """The invariant nobody had written down. Counted in SVDs per placebo
        draw, not in seconds."""
        draws = 40
        spy = _SvdSpy(monkeypatch)
        _fit(_panel(), draws=draws)
        assert len(spy) <= draws, (
            f"{len(spy)} spectra for {draws} placebo draws -- the guard is "
            "again costing more than one factorisation per draw")


# --------------------------------------------------------------------------- #
# 4. the half that is not fixed, recorded
# --------------------------------------------------------------------------- #
class TestReachOfTheFix:
    def test_the_unridged_program_still_factorises(self, monkeypatch):
        """``lambda_min >= r`` is vacuous at ``r = 0``, so the time-weight
        program keeps its SVD. Asserted so the limit of the fix is visible."""
        spy = _SvdSpy(monkeypatch)
        _fit(_panel(), draws=12)
        assert len(spy) > 0, (
            "no spectra at all would mean the unridged family found a "
            "certificate this test does not know about")

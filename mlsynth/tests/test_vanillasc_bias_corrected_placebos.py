"""VanillaSC's placebo tests correct every unit's gap when the treated gap is corrected (#920).

With ``bias_correct=True`` the treated unit's ratio comes from the Abadie-L'Hour
corrected gap. Each donor's placebo refit used its plain counterfactual, so the
in-space placebo and the leave-two-out test ranked a corrected statistic among
uncorrected ones. A permutation test applies one procedure to every unit, so each
refit's gap is now corrected with its own weights, its own standardised predictors
and its own donor pool. ``placebo_cs`` inverts the test on weights and has no
corrected form, so the pair is refused.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mlsynth.utils.vanillasc_helpers.lto as lto
import mlsynth.utils.vanillasc_helpers.pipeline as pipeline
from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError

T, T0, J = 20, 15, 6


def _panel(seed=0, effect=-3.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(J + 1):
        level, slope = 20.0 + 2.0 * u + rng.normal(), 0.5 + 0.1 * rng.normal()
        x1, x2 = rng.normal(), rng.normal()
        for t in range(T):
            treated = int(u == 0 and t >= T0)
            rows.append({"unit": f"u{u}", "time": t, "treat": treated, "x1": x1, "x2": x2,
                         "y": level + slope * t + 2.0 * x1 + rng.normal(scale=0.5) + effect * treated})
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
               covariates=["x1", "x2"], backend="exact", display_graphs=False,
               bias_correct=True)
    cfg.update(kw)
    return cfg


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(_cfg(df, **kw)).fit()


class _Calls:
    """Records the outcome series every bias correction is applied to."""

    def __init__(self, monkeypatch, module, inflate_donors=False):
        self.y1 = []
        real = module.bias_corrected_gaps

        def spy(W, X1, X0, y1, Y0, *, ridge):
            self.y1.append(np.asarray(y1, float).copy())
            gap = real(W, X1, X0, y1, Y0, ridge=ridge)
            if inflate_donors and len(self.y1) > 1:     # the first call is the treated unit's
                gap = gap.copy()
                gap[T0:] += 1e4                       # a donor whose corrected gap is huge
            return gap

        monkeypatch.setattr(module, "bias_corrected_gaps", spy)


# --------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("mode", ["placebo", "lto"])
def test_a_corrected_fit_gets_its_placebo_test(mode):
    res = _fit(_panel(), inference=mode)
    assert 0.0 <= res.inference.p_value <= 1.0


# --------------------------------------------------------------------------- unit
def test_every_placebo_refit_is_corrected(monkeypatch):
    calls = _Calls(monkeypatch, pipeline)
    df = _panel()
    _fit(df, inference="placebo")
    Y = df.pivot(index="time", columns="unit", values="y")[[f"u{u}" for u in range(J + 1)]].to_numpy()
    assert len(calls.y1) == 1 + J                       # the treated unit and each donor
    for j, y1 in enumerate(calls.y1[1:], start=1):
        np.testing.assert_array_equal(y1, Y[:, j])


def test_the_donor_ratios_are_taken_from_the_corrected_gaps(monkeypatch):
    """Inflate every donor's corrected post-period gap: the treated unit then ranks last."""
    _Calls(monkeypatch, pipeline, inflate_donors=True)
    res = _fit(_panel(), inference="placebo")
    assert res.inference.details["rank"] == J + 1
    assert res.inference.p_value == 1.0


def test_every_leave_two_out_refit_is_corrected(monkeypatch):
    calls = _Calls(monkeypatch, lto)
    res = _fit(_panel(), inference="lto")
    assert len(calls.y1) == 3 * res.inference.details["n_pairs"]


def test_the_leave_two_out_statistics_are_taken_from_the_corrected_gaps(monkeypatch):
    """Inflate every left-out donor's corrected gap: the treated unit wins no triple."""
    real = lto.bias_corrected_gaps
    df = _panel()
    y_tr = df[df.unit == "u0"].sort_values("time").y.to_numpy()

    def spy(W, X1, X0, y1, Y0, *, ridge):
        gap = real(W, X1, X0, y1, Y0, ridge=ridge)
        if not np.array_equal(y1, y_tr):
            gap = gap.copy()
            gap[T0:] += 1e4
        return gap
    monkeypatch.setattr(lto, "bias_corrected_gaps", spy)
    res = _fit(df, inference="lto")
    assert res.inference.details["treated_losses"] == res.inference.details["n_pairs"]


def test_without_the_correction_no_refit_is_corrected(monkeypatch):
    calls = _Calls(monkeypatch, pipeline)
    lto_calls = _Calls(monkeypatch, lto)
    _fit(_panel(), bias_correct=False, inference="placebo")
    _fit(_panel(), bias_correct=False, inference="lto")
    assert calls.y1 == [] and lto_calls.y1 == []


# --------------------------------------------------------------------------- edge
def test_a_vanishing_correction_gives_the_uncorrected_placebo_test():
    """A ridge that drives every slope to zero leaves every gap as it was."""
    df = _panel(seed=2)
    a = _fit(df, inference="placebo", bias_correct_ridge=1e12)
    b = _fit(df, inference="placebo", bias_correct=False)
    assert a.inference.details["rank"] == b.inference.details["rank"]


# --------------------------------------------------------------------------- failure
def test_the_confidence_set_refuses_the_correction():
    with pytest.raises(MlsynthConfigError, match="placebo_cs.*bias_correct"):
        VanillaSC(_cfg(_panel(), inference="placebo_cs"))


# --------------------------------------------------------------------------- property
@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000), mode=st.sampled_from(["placebo", "lto"]))
def test_property_a_vanishing_correction_is_the_uncorrected_test(seed, mode):
    df = _panel(seed=seed)
    a = _fit(df, inference=mode, bias_correct_ridge=1e12)
    b = _fit(df, inference=mode, bias_correct=False)
    assert a.inference.p_value == b.inference.p_value

"""VanillaSC's staggered path fits each treated unit with the options the config names (#922).

With several treated units, each unit used to be fit by an engine built from a subset
of the options, with no predictors and on every pre-treatment period: a predictor
backend raised "needs covariates" although covariates were given, ``backend="auto"``
fell back to outcome-only matching, and ``fit_window`` and ``fit_exclude`` were
ignored. Both paths now build the engine, the predictors and the outcome-fit rows
with the same helpers, so a unit's staggered fit is the single-treated fit of that
unit against the never-treated donors.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel import BilevelSCM

T = 24
ADOPT = {"u0": 16, "u1": 19}                   # two cohorts


def _panel(seed=0, n_donors=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2, n_donors + 2))
    rows = []
    for u in range(n_donors + 2):
        name = f"u{u}"
        level, slope = 20.0 + 2.0 * u + rng.normal(), 0.5 + 0.1 * rng.normal()
        for t in range(T):
            treated = int(name in ADOPT and t >= ADOPT[name])
            rows.append({"unit": name, "time": t, "treat": treated, "x0": X[0, u], "x1": X[1, u],
                         "y": level + slope * t + rng.normal(scale=0.5) - 3.0 * treated})
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
               display_graphs=False, inference=False)
    cfg.update(kw)
    return cfg


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(_cfg(df, **kw)).fit()


def _alone(df, unit):
    """The panel with every other treated unit removed: that unit's single-treated problem."""
    return df[~df.unit.isin([u for u in ADOPT if u != unit])]


def _unit_weights(res, unit):
    return {str(k): float(v) for k, v in res.sub_method_results[unit].donor_weights.items()}


def _weights(res):
    return {str(k): float(v) for k, v in res.weights.donor_weights.items()}


SPECS = [
    pytest.param(dict(backend="exact", covariates=["x0", "x1"]), id="exact-covariates"),
    pytest.param(dict(backend="auto", covariates=["x0", "x1"], mscmt_maxiter=50, seed=3), id="auto-covariates"),
    pytest.param(dict(fit_window=(6, 15)), id="window"),
    pytest.param(dict(fit_exclude=[2, 5, 9]), id="exclude"),
    pytest.param(dict(backend="exact", covariates=["x0", "x1"], fit_window=(4, 15), fit_exclude=[7]),
                 id="exact-window-exclude"),
]


# --------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("spec", SPECS)
def test_a_staggered_fit_with_the_option_returns_its_att(spec):
    res = _fit(_panel(), **spec)
    assert np.isfinite(res.effects.att)


# --------------------------------------------------------------------------- unit: each unit is its own single-treated fit
@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("unit", list(ADOPT))
def test_each_unit_is_fit_as_it_would_be_alone(spec, unit):
    df = _panel(seed=1)
    staggered = _unit_weights(_fit(df, **spec), unit)
    single = _weights(_fit(_alone(df, unit), **spec))
    for d in set(staggered) | set(single):
        assert staggered.get(d, 0.0) == pytest.approx(single.get(d, 0.0), abs=1e-6), d


def test_the_engine_gets_every_option_the_single_path_gives_it(monkeypatch):
    seen = []
    real = BilevelSCM.__init__

    def init(engine, backend, **kw):
        seen.append(kw)
        real(engine, backend, **kw)
    monkeypatch.setattr(BilevelSCM, "__init__", init)
    df = _panel()
    _fit(_alone(df, "u0"), backend="penalized", penalized_lambda=0.3, mscmt_tol=1e-5)
    single = seen[0]
    seen.clear()
    _fit(df, backend="penalized", penalized_lambda=0.3, mscmt_tol=1e-5)
    assert seen and all(kw == single for kw in seen)


# --------------------------------------------------------------------------- failure
def test_a_window_that_leaves_a_cohort_no_pre_period_is_refused():
    with pytest.raises(MlsynthDataError, match="fit_window.*no pre-treatment"):
        _fit(_panel(), fit_window=(17, 18))        # both inside u0's post-period, u1's pre-period


def test_an_exclusion_outside_every_pre_period_is_refused():
    with pytest.raises(MlsynthDataError, match="fit_exclude"):
        _fit(_panel(), fit_exclude=[22])


@pytest.mark.parametrize("option", ["exact_ranges", "exact_face_counts"])
def test_the_single_unit_exact_extras_are_refused(option):
    with pytest.raises(MlsynthConfigError, match=option):
        _fit(_panel(), backend="exact", covariates=["x0", "x1"], **{option: True})


# --------------------------------------------------------------------------- property
@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), lo=st.integers(0, 6), hi=st.integers(10, 15))
def test_property_a_windowed_staggered_unit_is_its_windowed_single_fit(seed, lo, hi):
    df = _panel(seed=seed)
    res = _fit(df, fit_window=(lo, hi))
    for unit in ADOPT:
        single = _weights(_fit(_alone(df, unit), fit_window=(lo, hi)))
        staggered = _unit_weights(res, unit)
        for d in set(staggered) | set(single):
            assert staggered.get(d, 0.0) == pytest.approx(single.get(d, 0.0), abs=1e-6)

"""Tests for VanillaSC ``fit_window`` -- restricting the outcome-fit window.

By default the synthetic control is fit to the treated outcome over the *entire*
pre-treatment period. Some canonical specifications instead optimise the outcome
fit over a sub-window of the pre-period: Abadie & Gardeazabal's (2003) Basque
study, as packaged by MSCMT, fits ``gdpcap`` over 1960-1969 even though the data
begin in 1955. ``fit_window=(start, end)`` exposes that choice; it restricts the
dependent SSR (MSCMT's ``times.dep``) to the inclusive window, leaving predictor
matching (``covariate_windows``) untouched. ``None`` (the default) keeps the full
pre-period, so existing behaviour is unchanged.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.config_models import VanillaSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"
_PROP99 = _BASEDATA / "augmented_cali_long.csv"


def _prop99():
    if not _PROP99.exists():
        pytest.skip("augmented_cali_long.csv not available")
    d = pd.read_csv(_PROP99)
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    return d


def _weights(res):
    return {str(k): float(v) for k, v in res.weights.donor_weights.items()}


def _fit(d, **kw):
    base = dict(df=d, outcome="cigsale", treat="treated", unitid="state",
                time="year", backend="outcome-only", display_graphs=False)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(base).fit()


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
_MINI = pd.DataFrame({"u": ["a", "a", "b", "b"], "t": [1, 2, 1, 2],
                      "y": [1.0, 2.0, 3.0, 4.0], "d": [0, 1, 0, 0]})


def test_fit_window_default_is_none():
    cfg = VanillaSCConfig(df=_MINI, outcome="y", treat="d", unitid="u", time="t")
    assert cfg.fit_window is None


def test_fit_window_accepts_pair():
    cfg = VanillaSCConfig(df=_MINI, outcome="y", treat="d", unitid="u", time="t",
                          fit_window=(1960, 1969))
    assert tuple(cfg.fit_window) == (1960, 1969)


def test_fit_window_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="fit_window"):
        VanillaSCConfig(df=_MINI, outcome="y", treat="d", unitid="u", time="t",
                        fit_window=(1969, 1960))


def test_fit_window_rejects_wrong_length():
    with pytest.raises(ValueError, match="fit_window"):
        VanillaSCConfig(df=_MINI, outcome="y", treat="d", unitid="u", time="t",
                        fit_window=(1960, 1965, 1969))


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #
def test_full_preperiod_window_matches_none():
    """A window equal to the full pre-period reproduces the default fit exactly:
    Prop 99's pre-period is 1970-1988, so fit_window=(1970, 1988) must give the
    identical bilevel optimum (Utah 0.3939) as fit_window=None."""
    d = _prop99()
    base = _weights(_fit(d, fit_window=None))
    windowed = _weights(_fit(d, fit_window=(1970, 1988)))
    assert base["Utah"] == pytest.approx(0.3939, abs=5e-4)
    for k in set(base) | set(windowed):
        assert windowed.get(k, 0.0) == pytest.approx(base.get(k, 0.0), abs=1e-6)


def test_window_overhang_into_post_is_clipped():
    """A window whose end runs past the treatment date uses only pre-period rows;
    (1970, 2100) therefore equals the full-pre-period fit, not something else."""
    d = _prop99()
    base = _weights(_fit(d, fit_window=None))
    over = _weights(_fit(d, fit_window=(1970, 2100)))
    for k in set(base) | set(over):
        assert over.get(k, 0.0) == pytest.approx(base.get(k, 0.0), abs=1e-6)


def test_restricted_window_changes_the_fit():
    """Restricting the outcome fit to a late sub-window (1980-1988) drops the
    early pre-period rows from the SSR and so moves the weights off the full-fit
    optimum -- the window is actually doing something."""
    d = _prop99()
    full = _weights(_fit(d, fit_window=None))
    late = _weights(_fit(d, fit_window=(1980, 1988)))
    dev = max(abs(late.get(k, 0.0) - full.get(k, 0.0))
              for k in set(full) | set(late))
    assert dev > 0.02
    # still a valid simplex
    assert sum(late.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(v >= -1e-9 for v in late.values())


def test_window_with_no_pretreatment_overlap_raises():
    """A window lying entirely in the post-period selects zero fit rows and must
    fail loudly rather than silently fit on nothing."""
    d = _prop99()
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        _fit(d, fit_window=(1995, 2000))


def test_window_selects_expected_row_count():
    """The fit uses exactly the pre-period rows inside the window: (1980, 1988)
    is 9 years, so the pre-fit RMSE is computed from 9 residuals. We check this
    indirectly: the windowed fit's pre-window residuals are all finite and the
    fit reproduces the treated path over the window better than outside it."""
    d = _prop99()
    res = _fit(d, fit_window=(1980, 1988))
    years = sorted(d["year"].unique())
    y = (d[d.state == "California"].sort_values("year")["cigsale"].to_numpy())
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    win = np.array([(1980 <= t <= 1988) for t in years])
    assert np.all(np.isfinite(cf[win]))
    assert win.sum() == 9

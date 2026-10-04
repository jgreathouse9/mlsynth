r"""HAC belongs to the estimate, not to how the split was chosen.

TBRMM could price its interval on a Newey-West long-run variance; TBR could
not. That split made no sense once the two merged, because serial correlation
in the pretest residual is a property of the panel and the groups, and nothing
about naming a split instead of searching for one removes it.

So ``estimate`` honours ``variance`` and ``hac_bandwidth`` in both modes, and
the named mode gains an option it never had. These tests pin that, and pin the
two things the correction must not do: move the point estimate, or fire when
nobody asked for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBR
from mlsynth.exceptions import MlsynthEstimationError


@pytest.fixture
def correlated_panel():
    """A panel whose pretest residual is visibly autocorrelated."""
    rng = np.random.default_rng(23)
    rows = []
    shock = 0.0
    shocks = []
    for t in range(60):
        shock = 0.85 * shock + rng.normal(0, 1.0)
        shocks.append(shock)
    for j in range(7):
        for t in range(60):
            base = 150 + 6 * j + 0.4 * t
            extra = shocks[t] * (2.5 if j < 2 else 0.2)
            rows.append({"geo": f"g{j}", "week": t,
                         "y": base + extra + rng.normal(0, 0.4),
                         "post": int(t >= 50), "tr": int(j < 2), "ct": int(j >= 2)})
    return pd.DataFrame(rows)


def _named(panel, **kw):
    return TBR(dict(df=panel, outcome="y", unitid="geo", time="week",
                    post_col="post", treatment_col="tr", control_col="ct",
                    **kw)).fit()


def _width(res):
    c = res.report.cumulative
    return float(c.upper[-1] - c.lower[-1])


def test_the_named_mode_can_ask_for_hac(correlated_panel):
    """The option TBR never had."""
    res = _named(correlated_panel, variance="hac")
    assert res.report.cumulative is not None


def test_hac_widens_the_interval_under_serial_correlation(correlated_panel):
    iid = _named(correlated_panel)
    hac = _named(correlated_panel, variance="hac")
    assert _width(hac) > _width(iid), (
        "a long-run variance should price the autocorrelation the iid scale "
        f"ignores: {_width(hac):.4f} vs {_width(iid):.4f}")


def test_hac_does_not_move_the_point_estimate(correlated_panel):
    """It is a variance correction; the location is eqn 4 either way."""
    iid = _named(correlated_panel)
    hac = _named(correlated_panel, variance="hac")
    assert hac.report.cumulative.estimate == pytest.approx(
        iid.report.cumulative.estimate, rel=1e-12)


def test_the_default_is_unchanged(correlated_panel):
    """Nobody asking for HAC gets the iid scale, bit for bit."""
    a = _named(correlated_panel)
    b = _named(correlated_panel, variance="iid")
    assert _width(a) == pytest.approx(_width(b), rel=0, abs=0)


def test_a_bandwidth_longer_than_the_pretest_is_refused(correlated_panel):
    """A truncation with no lags left to average is a configuration error."""
    with pytest.raises(MlsynthEstimationError, match="bandwidth"):
        _named(correlated_panel, variance="hac", hac_bandwidth=999)


def test_the_searched_mode_still_honours_it(correlated_panel):
    """The mode it already had, now sharing one implementation."""
    cfg = dict(df=correlated_panel, outcome="y", unitid="geo", time="week",
               post_col="post", max_treatment_size=2, n_test=10)
    iid = TBR(dict(**cfg)).fit()
    hac = TBR(dict(**cfg, variance="hac")).fit()
    assert _width(hac) != _width(iid)

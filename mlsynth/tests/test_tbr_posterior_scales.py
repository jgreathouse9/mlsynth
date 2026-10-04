r"""The three scales a TBR posterior reports, and how they relate.

A cumulative effect on a treatment group can be quoted three ways, and the
merged result carries all three because readers want different ones:

* ``total_*`` -- the cumulative effect on the group over the whole test window,
  which is TBR's own estimand and what ``report.cumulative`` shows;
* ``group_*`` -- that total averaged over the test periods, the scale
  ``report.inference`` is on, so an interval and the ``att`` beside it agree;
* ``att_*`` -- averaged over periods and over treated geos, the standard
  average effect on the treated, and the scale the design search ranks on.

They are one number divided by different things, and the identity is

    total = group * n_periods = att * n_periods * n_treated_units

Nothing recorded that before, so a reader comparing ``report.inference`` with
``recommended.effect.posterior.att_lower`` found figures differing by the
treated count with nothing to say why. The relation is pinned here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBR


def _panel(n_geo=10, n_post=8, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_geo):
        for t in range(36 + n_post):
            rows.append({"geo": f"g{j}", "week": t,
                         "y": 200 + 9 * j + 0.8 * t + rng.normal(0, 2),
                         "post": int(t >= 36)})
    return pd.DataFrame(rows)


def _fit(n_geo=10, n_post=8, m=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBR(dict(df=_panel(n_geo, n_post), outcome="y", unitid="geo",
                        time="week", post_col="post",
                        max_treatment_size=m, n_test=n_post)).fit()


@pytest.mark.parametrize("n_geo,n_post,m", [(10, 8, 3), (10, 4, 3),
                                            (12, 6, 2), (14, 10, 4)])
def test_the_identity_between_the_three_scales(n_geo, n_post, m):
    res = _fit(n_geo, n_post, m)
    p = res.recommended.effect.posterior
    n_treated = len(res.recommended.treatment_units)
    for total, group, att in ((p.total_lower, p.group_lower, p.att_lower),
                              (p.total_upper, p.group_upper, p.att_upper)):
        assert group == pytest.approx(total / n_post, rel=1e-12)
        assert att == pytest.approx(total / (n_post * n_treated), rel=1e-12)


def test_the_report_interval_is_on_the_group_scale(res=None):
    """Which is what makes report.inference and report.effects.att agree."""
    res = _fit()
    p = res.recommended.effect.posterior
    inf = res.report.inference
    assert inf.ci_lower == pytest.approx(p.group_lower, rel=1e-9)
    assert inf.ci_upper == pytest.approx(p.group_upper, rel=1e-9)


def test_the_report_att_sits_inside_its_own_interval():
    res = _fit()
    inf, att = res.report.inference, res.report.effects.att
    assert inf.ci_lower < att < inf.ci_upper


def test_a_single_treated_geo_collapses_two_of_the_scales():
    """With one geo there is nothing to average across, so group == att."""
    res = _fit(n_geo=12, n_post=8, m=1)
    p = res.recommended.effect.posterior
    assert len(res.recommended.treatment_units) == 1
    assert p.group_lower == pytest.approx(p.att_lower, rel=1e-12)
    assert p.group_upper == pytest.approx(p.att_upper, rel=1e-12)

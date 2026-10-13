r"""One class, one result type, two ways in.

``TBR.fit()`` returns a :class:`~mlsynth.config_models.DesignResult` in both
modes, with the estimate on ``report``. That follows LEXSCM and MAREX, the two
other estimators here that design and realise in one call, and it means a
caller never has to ask which type came back.

The point of these tests is that the two modes agree on what ``report`` holds.
Before the merge, TBRMM's report carried the group posterior's bounds and
nothing else, while TBR's carried the pretest fit, the cumulative posterior,
the cooldown split and iROAS. Both now come from the same builder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBR
from mlsynth.config_models import DesignResult, TBRConfig


@pytest.fixture
def panel():
    rng = np.random.default_rng(17)
    rows = []
    for j in range(8):
        for t in range(44):
            rows.append({"geo": f"g{j}", "week": t,
                         "y": 200 + 9 * j + 0.8 * t + rng.normal(0, 2.0),
                         "spend": 40 + rng.normal(0, 1.0),
                         "post": int(t >= 36), "tr": int(j < 3), "ct": int(j >= 3)})
    return pd.DataFrame(rows)


_BASE = dict(outcome="y", unitid="geo", time="week", post_col="post")


def _named(panel, **kw):
    return TBR(dict(df=panel, treatment_col="tr", control_col="ct",
                    **_BASE, **kw)).fit()


def _searched(panel, **kw):
    return TBR(dict(df=panel, max_treatment_size=3, n_test=8,
                    **_BASE, **kw)).fit()


def test_both_modes_return_a_design_result(panel):
    for res in (_named(panel), _searched(panel)):
        assert isinstance(res, DesignResult)


def test_both_modes_put_the_same_shape_on_report(panel):
    """The trap the merge exists to remove."""
    for res in (_named(panel), _searched(panel)):
        r = res.report
        assert r is not None
        assert r.cumulative is not None, "the cumulative posterior must be there"
        assert r.tbr_fit is not None, "the pretest fit must be there"
        assert r.treated_units and r.control_units


def test_the_named_mode_searches_nothing(panel):
    res = _named(panel)
    assert not res.designs
    assert res.recommended is None
    assert sorted(res.selected_units) == ["g0", "g1", "g2"]


def test_the_searched_mode_returns_a_menu(panel):
    res = _searched(panel)
    assert res.designs, "one design per treatment size"
    assert res.recommended is not None
    assert set(res.recommended.treatment_units) == set(res.selected_units)


def test_the_searched_report_is_the_recommendation(panel):
    """The report estimates the design the search actually recommends."""
    res = _searched(panel)
    assert set(res.report.treated_units) == set(res.recommended.treatment_units)
    assert set(res.report.control_units) == set(res.recommended.control_units)


def test_cost_reaches_the_report_in_both_modes(panel):
    for res in (_named(panel, cost_col="spend"), _searched(panel, cost_col="spend")):
        assert res.report.iroas is not None
        assert res.report.cumulative_cost is not None

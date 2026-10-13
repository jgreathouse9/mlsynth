r"""One estimate path for both of the merged TBR's modes.

The named mode reads the split off ``treatment_col`` and ``control_col``. The
searched mode produces a split by hill climb, and once it has one there is
nothing left that makes it a different estimation problem -- it is TBR on the
geos the search chose. So ``build_inputs`` takes the split directly, and the
searched mode's report is built by the same code that builds the named mode's.

Without this the two modes report different things: TBRMM assembled a generic
effect result while TBR assembled one carrying the pretest fit, the cumulative
posterior, the cooldown split and iROAS. A caller reading ``res.report`` would
have found ``cumulative`` present in one mode and absent in the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.setup import build_inputs


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    rows = []
    for j in range(6):
        for t in range(40):
            rows.append({"geo": f"g{j}", "week": t,
                         "y": 100 + 4 * j + 0.5 * t + rng.normal(0, 1.5),
                         "post": int(t >= 32), "tr": int(j < 2), "ct": int(j >= 2)})
    return pd.DataFrame(rows)


def _named(panel, **kw):
    return TBRConfig(df=panel, outcome="y", unitid="geo", time="week",
                     post_col="post", treatment_col="tr", control_col="ct", **kw)


def test_an_explicit_split_overrides_the_columns(panel):
    cfg = _named(panel)
    from_cols = build_inputs(cfg)
    assert sorted(from_cols.treated_units) == ["g0", "g1"]

    override = build_inputs(cfg, treated=["g4"], controls=["g0", "g1"])
    assert sorted(override.treated_units) == ["g4"]
    assert sorted(override.control_units) == ["g0", "g1"]


def test_the_override_changes_the_aggregates_it_hands_back(panel):
    """Not just the rosters: the regression's two series follow the split."""
    cfg = _named(panel)
    a = build_inputs(cfg)
    b = build_inputs(cfg, treated=["g4"], controls=["g0", "g1"])
    assert not np.allclose(a.y, b.y), "the treatment aggregate should differ"
    assert not np.allclose(a.x, b.x), "the control aggregate should differ"


def test_a_unit_in_both_halves_of_the_override_is_refused(panel):
    with pytest.raises(MlsynthDataError, match="one group|both"):
        build_inputs(_named(panel), treated=["g0"], controls=["g0", "g1"])


@pytest.mark.parametrize("treated,controls,match", [
    ([], ["g2"], "treatment group is empty"),
    (["g0"], [], "control group"),
])
def test_an_empty_half_is_refused(panel, treated, controls, match):
    with pytest.raises(MlsynthDataError, match=match):
        build_inputs(_named(panel), treated=treated, controls=controls)


def test_a_unit_not_in_the_panel_is_refused(panel):
    with pytest.raises(MlsynthDataError, match="not in the panel|unknown"):
        build_inputs(_named(panel), treated=["nope"], controls=["g2"])


def test_the_override_needs_both_halves(panel):
    """Half an override would silently mix a chosen group with a column one."""
    with pytest.raises(MlsynthDataError, match="both"):
        build_inputs(_named(panel), treated=["g4"])

r"""The merged TBR config: one class, two ways of naming the groups.

TBR and TBRMM were two estimators over one method. TBR took a split and
estimated with it; TBRMM searched for a split and scored it on TBR's own
posterior, importing ``tbr_helpers.posterior`` to do so. They are the design
half and the analysis half of Au (2018), and the merged class holds both.

The config therefore accepts the groups in exactly one of two ways, and these
tests pin which:

* named -- ``treatment_col`` and ``control_col`` mark the geos, so the split is
  already made and there is nothing to search;
* searched -- ``max_treatment_size`` and ``n_test`` describe an experiment to
  be designed, and the hill climb picks the split. The ``*_eligible_col`` flags
  narrow the pools it may draw from, and are optional: omitted, every geo is
  eligible, which is why they cannot be what tells the two modes apart.

Naming both is a contradiction and raises. Naming neither leaves the method
with no groups and raises. The mode is readable off the config so the pipeline
dispatches on a value and not on a chain of ``is None`` checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError


@pytest.fixture
def panel():
    rng = np.random.default_rng(3)
    units = [f"g{j:02d}" for j in range(8)]
    rows = []
    for j, u in enumerate(units):
        for t in range(30):
            rows.append({"geo": u, "week": t, "y": 100 + 5 * j + rng.normal(0, 2),
                         "post": int(t >= 24),
                         "tr": int(j < 2), "ct": int(j >= 2),
                         "tr_ok": int(j < 4), "ct_ok": int(j >= 2)})
    return pd.DataFrame(rows)


_BASE = dict(outcome="y", unitid="geo", time="week", post_col="post")


def test_named_groups_give_the_named_mode(panel):
    cfg = TBRConfig(df=panel, treatment_col="tr", control_col="ct", **_BASE)
    assert cfg.mode == "named"


def test_an_experiment_to_design_gives_the_searched_mode(panel):
    cfg = TBRConfig(df=panel, max_treatment_size=3, n_test=6, **_BASE)
    assert cfg.mode == "searched"


def test_eligibility_flags_are_optional_in_the_searched_mode(panel):
    """Omitted, every geo is eligible -- so they cannot discriminate the mode."""
    cfg = TBRConfig(df=panel, max_treatment_size=3, n_test=6,
                    treatment_eligible_col="tr_ok", control_eligible_col="ct_ok",
                    **_BASE)
    assert cfg.mode == "searched"


def test_naming_both_ways_is_refused(panel):
    with pytest.raises(MlsynthConfigError, match="at the same time"):
        TBRConfig(df=panel, treatment_col="tr", control_col="ct",
                  max_treatment_size=3, n_test=6, **_BASE)


def test_naming_neither_way_is_refused(panel):
    with pytest.raises(MlsynthConfigError, match="no groups|neither"):
        TBRConfig(df=panel, **_BASE)


def test_a_treatment_column_without_a_control_column_is_refused(panel):
    """Half a split is not a split: TBR regresses one aggregate on the other."""
    with pytest.raises(MlsynthConfigError, match="control_col"):
        TBRConfig(df=panel, treatment_col="tr", **_BASE)


def test_searching_needs_the_experiment_length(panel):
    """The objective's power term is a function of how long the test runs."""
    with pytest.raises(MlsynthConfigError, match="n_test"):
        TBRConfig(df=panel, max_treatment_size=3, **_BASE)


def test_the_design_knobs_are_rejected_in_the_named_mode(panel):
    """``max_treatment_size`` has nothing to size when the split is given."""
    with pytest.raises(MlsynthConfigError, match="named|search"):
        TBRConfig(df=panel, treatment_col="tr", control_col="ct",
                  max_treatment_size=3, **_BASE)


def test_cost_and_cooldown_survive_both_modes(panel):
    """They belong to the estimate, which both modes produce."""
    panel = panel.assign(spend=1.0, cool=0)
    named = TBRConfig(df=panel, treatment_col="tr", control_col="ct",
                      cost_col="spend", cooldown_col="cool", **_BASE)
    searched = TBRConfig(df=panel, max_treatment_size=3, n_test=6,
                         cost_col="spend", cooldown_col="cool", **_BASE)
    assert named.cost_col == searched.cost_col == "spend"
    assert named.cooldown_col == searched.cooldown_col == "cool"


def test_a_column_that_is_not_in_the_frame_is_refused(panel):
    with pytest.raises((MlsynthConfigError, Exception), match="nope|not in|missing"):
        TBRConfig(df=panel, treatment_col="nope", control_col="ct", **_BASE)

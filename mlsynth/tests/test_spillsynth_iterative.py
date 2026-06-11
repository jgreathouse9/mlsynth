"""Tests for SPILLSYNTH method='iterative' (Melnychuk 2024 waterfall SCM)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.utils.spillsynth_helpers.structures import IterativeFit

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "repgermany.dta"


@pytest.fixture(scope="module")
def german_panel():
    if not _DATA.exists():
        pytest.skip("repgermany.dta not available")
    d = pd.read_stata(_DATA)
    d = d[["country", "year", "gdp", "trade", "infrate",
           "industry", "schooling", "invest80"]].copy()
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def _cfg(d, **kw):
    base = dict(df=d, outcome="gdp", treat="treat", unitid="country",
                time="year", method="iterative", affected_units=["Austria"],
                display_graphs=False)
    base.update(kw)
    return base


def test_iterative_outcome_only_structure(german_panel):
    # demeaned outcome-only backend -> deterministic and fast
    res = SPILLSYNTH(_cfg(german_panel, iscm_intercept=True)).fit()
    f = res.iterative
    assert isinstance(f, IterativeFit)
    assert res.method == "iterative"
    assert f.bilevel_solver == "intercept"
    # Austria is the one cleaned donor; 15 clean controls remain.
    assert f.cleaned_units == ["Austria"]
    assert f.n_clean == 15
    assert "Austria" in f.spillover_panel and "Austria" in f.spillover_att
    # Reunification depressed West German GDP -> negative effect both ways.
    assert res.att < 0 and res.att_scm < 0
    # The cleaning moves the estimate away from the naive gap.
    assert abs(res.att - res.att_scm) > 1.0
    # Shapes line up with the post-period (1990-2003 -> 14 periods).
    T1 = int(german_panel.year.ge(1990).groupby(german_panel.country).sum().max())
    assert f.gap.shape == (T1,)
    assert f.gap_scm.shape == (T1,)
    assert res.counterfactual.shape == (T1,)
    assert f.spillover_panel["Austria"].shape == (T1,)


def test_iterative_cleans_only_post_period(german_panel):
    # The waterfall replaces affected donors' POST outcomes only; the treated
    # unit's pre-fit is therefore the naive pre-fit (weights unchanged).
    res = SPILLSYNTH(_cfg(german_panel, iscm_intercept=True)).fit()
    f = res.iterative
    # naive and iterative counterfactuals share the pre-period weights, so they
    # differ only because Austria's post outcomes were cleaned -> the gaps
    # diverge in the post-period.
    assert not np.allclose(f.gap, f.gap_scm)
    assert np.isfinite(f.pre_rmspe)


def test_iterative_dict_and_config_equivalent(german_panel):
    r1 = SPILLSYNTH(_cfg(german_panel, iscm_intercept=True)).fit()
    r2 = SPILLSYNTH(SPILLSYNTHConfig(**_cfg(german_panel, iscm_intercept=True))).fit()
    assert r1.att == pytest.approx(r2.att)
    assert r1.att_scm == pytest.approx(r2.att_scm)


def test_iterative_recovers_donor_spillover_synthetically():
    """On a controlled DGP the waterfall removes a planted donor spillover.

    Two clean donors drive the treated unit (true effect -5). One affected donor
    carries a positive post-period spillover that contaminates the naive SC; the
    iterative clean of that donor moves the estimate back toward the truth.
    """
    rng = np.random.default_rng(0)
    T, T0 = 20, 14
    f = np.cumsum(rng.normal(0, 1, T)) + 50.0            # common factor
    treated = f + rng.normal(0, 0.1, T)
    clean1 = f + rng.normal(0, 0.1, T)
    clean2 = 0.5 * f + rng.normal(0, 0.1, T) + 10
    affected = f + rng.normal(0, 0.1, T)
    treated[T0:] += -5.0                                  # true effect
    affected[T0:] += 6.0                                  # planted spillover
    units = ["T", "A", "c1", "c2"]
    series = {"T": treated, "A": affected, "c1": clean1, "c2": clean2}
    rows = [{"unit": u, "time": t, "y": series[u][t],
             "d": int(u == "T" and t >= T0)} for u in units for t in range(T)]
    df = pd.DataFrame(rows)
    res = SPILLSYNTH(dict(df=df, outcome="y", treat="d", unitid="unit",
                          time="time", method="iterative", affected_units=["A"],
                          iscm_intercept=True, display_graphs=False)).fit()
    # cleaning the contaminated donor pulls the estimate toward the true -5,
    # away from the naive (spillover-biased) gap.
    assert abs(res.att - (-5.0)) < abs(res.att_scm - (-5.0))
    assert res.iterative.spillover_att["A"] != 0.0

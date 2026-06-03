"""Tests for the Grossi et al. (2025) partial-interference method of SPILLSYNTH."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.spillsynth_helpers import GrossiFit, run_grossi
from mlsynth.utils.spillsynth_helpers.setup import prepare_spillsynth_inputs

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
                time="year", method="grossi", affected_units=["Austria"],
                display_graphs=False)
    base.update(kw)
    return base


def test_grossi_runs_and_excludes_affected(german_panel):
    res = SPILLSYNTH(_cfg(german_panel)).fit()
    f = res.grossi
    assert isinstance(f, GrossiFit)
    assert res.method == "grossi"
    # Reunification depressed West German GDP -> negative direct effect.
    assert res.att < 0
    # Austria is the dropped cluster-mate: it must NOT be a donor for WG.
    assert "Austria" not in f.donor_weights
    # ... but it IS given a spillover effect.
    assert "Austria" in f.spillover_att
    # Donor pool is the far/clean controls only.
    assert f.n_clean == len(german_panel.country.unique()) - 2  # drop WG + Austria
    T1 = int(german_panel.year.ge(1990).groupby(german_panel.country).sum().max())
    assert f.gap.shape == (T1,)


def test_grossi_direct_differs_from_naive(german_panel):
    res = SPILLSYNTH(_cfg(german_panel)).fit()
    # Dropping the affected neighbour changes the estimate vs the naive
    # all-controls SCM that keeps it.
    assert abs(res.grossi.direct_att - res.grossi.att_scm) > 1e-6


def test_grossi_requires_affected_unit(german_panel):
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTH(_cfg(german_panel, affected_units=None)).fit()


def test_grossi_dict_and_config_equivalent(german_panel):
    cfg = _cfg(german_panel)
    r1 = SPILLSYNTH(cfg).fit()
    r2 = SPILLSYNTH(SPILLSYNTHConfig(**cfg)).fit()
    assert r1.att == pytest.approx(r2.att)


def test_run_grossi_direct_matches_estimator(german_panel):
    inputs = prepare_spillsynth_inputs(
        df=german_panel, outcome="gdp", treat="treat", unitid="country",
        time="year", affected_units=["Austria"])
    fit = run_grossi(inputs, bilevel_solver="penalized")
    res = SPILLSYNTH(_cfg(german_panel)).fit()
    assert fit.direct_att == pytest.approx(res.att)


def test_grossi_inference_brackets_point_estimate(german_panel):
    # Outcome-only with a small bootstrap for speed.
    res = SPILLSYNTH(_cfg(german_panel, n_boot=30, ci_level=0.90, seed=1)).fit()
    f = res.grossi
    assert f.direct_ci is not None and f.direct_ci.shape == (f.gap.shape[0], 2)
    assert f.avg_spillover_ci is not None
    # Pivotal bias-corrected interval must bracket the per-period point.
    lo, hi = f.direct_ci[:, 0], f.direct_ci[:, 1]
    assert np.all(lo <= f.gap + 1e-6) and np.all(f.gap - 1e-6 <= hi)
    assert np.all(lo <= hi)


def test_grossi_inference_rejected_for_non_penalized_covariates(german_panel):
    cov = ["trade", "infrate", "industry"]
    with pytest.raises(MlsynthConfigError):
        SPILLSYNTH(_cfg(german_panel, covariates=cov, bilevel_solver="mscmt",
                        n_boot=10)).fit()

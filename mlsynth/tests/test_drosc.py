"""Tests for :class:`mlsynth.DROSC` (Distributionally Robust Synthetic Control,
Koo & Guo 2026).

The deterministic point estimand is pinned value-for-value against the authors'
R ``DRoSC`` (helpers.R, ``limSolve::lsei``) on the Basque study: at
``robustness_lambda`` = 0 / 0.03 / 0.06 the ATT is -0.7424 / -0.2557 / 0.0, and
the classical-SC reference is -0.8946. The perturbation union CI is stochastic
and seed-dependent, so it is checked for sanity (finite, contains the ATT), not
exact endpoints.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from mlsynth import DROSC
from mlsynth.config_models import DROSCConfig
from mlsynth.exceptions import MlsynthConfigError

_BASQUE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                       "basque_jasa.csv")


def _basque_config(**overrides):
    if not os.path.exists(_BASQUE):
        pytest.skip("basque_jasa.csv not available")
    d = pd.read_csv(_BASQUE)
    d = d[d.regionname != "Spain (Espana)"].copy()
    treat_year = sorted(d.year.unique())[15]           # T0 = 15 (Basque.R)
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= treat_year)).astype(int)
    cfg = dict(df=d, outcome="gdpcap", treat="treat", unitid="regionname",
               time="year", display_graphs=False)
    cfg.update(overrides)
    return cfg


def _fit(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DROSC(_basque_config(**overrides)).fit()


# --------------------------------------------------------------------------- #
# reproduction (deterministic point estimand vs the authors' R)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam,expected", [(0.0, -0.742368), (0.03, -0.255693),
                                          (0.06, 0.0)])
def test_att_matches_reference(lam, expected):
    res = _fit(robustness_lambda=lam)
    assert res.effects.att == pytest.approx(expected, abs=0.02)


def test_classical_sc_reference():
    res = _fit(robustness_lambda=0.0)
    assert res.additional_outputs["classical_sc_att"] == pytest.approx(-0.8946, abs=0.02)


def test_att_shrinks_toward_zero_with_lambda():
    atts = [abs(_fit(robustness_lambda=l).effects.att)
            for l in (0.0, 0.03, 0.06)]
    assert atts[0] > atts[1] > atts[2] - 1e-9        # monotone toward 0
    assert atts[2] == pytest.approx(0.0, abs=0.02)


def test_weights_are_a_simplex_concentrated_on_madrid():
    res = _fit(robustness_lambda=0.0)
    w = res.weights.donor_weights
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-9 for v in w.values())
    top = max(w, key=w.get)
    assert "Madrid" in top and w[top] == pytest.approx(0.388, abs=0.03)


# --------------------------------------------------------------------------- #
# result contract
# --------------------------------------------------------------------------- #
def test_result_contract_shapes():
    res = _fit(robustness_lambda=0.02)
    ts = res.time_series
    T = len(res.additional_outputs["donor_names"]) * 0 + len(ts.observed_outcome)
    assert len(ts.counterfactual_outcome) == T == 43
    assert len(ts.estimated_gap) == T
    assert res.method_details.method_name == "DROSC"
    assert res.additional_outputs["robustness_lambda"] == 0.02
    # ATT is the mean post-period gap of the reported counterfactual
    pre = res.additional_outputs["pre_periods"]
    post_gap = np.asarray(ts.observed_outcome)[pre:] - np.asarray(ts.counterfactual_outcome)[pre:]
    assert float(post_gap.mean()) == pytest.approx(res.effects.att, abs=1e-6)


def test_flat_accessors():
    res = _fit(robustness_lambda=0.0)
    assert np.isfinite(res.att)
    assert res.donor_weights is not None
    assert res.counterfactual is not None


# --------------------------------------------------------------------------- #
# perturbation union CI (stochastic; sanity only)
# --------------------------------------------------------------------------- #
def test_inference_union_ci_is_sane():
    res = _fit(robustness_lambda=0.0, inference=True, seed=1, n_perturbations=60)
    inf = res.inference
    assert inf is not None
    assert inf.ci_lower < inf.ci_upper
    assert inf.ci_lower <= res.effects.att <= inf.ci_upper
    intervals = inf.details["ci_intervals"]
    assert isinstance(intervals, list) and len(intervals) >= 1
    # the union's hull equals the reported (ci_lower, ci_upper)
    assert min(iv[0] for iv in intervals) == pytest.approx(inf.ci_lower)
    assert max(iv[1] for iv in intervals) == pytest.approx(inf.ci_upper)
    # Basque effect is not robustly distinguishable from zero (the paper's point)
    assert inf.ci_lower <= 0 <= inf.ci_upper
    assert inf.confidence_level == pytest.approx(0.95)


def test_inference_off_by_default():
    assert _fit(robustness_lambda=0.0).inference is None


def test_inference_conditional_runs():
    res = _fit(robustness_lambda=0.0, inference=True, conditional=True, seed=1,
               n_perturbations=60)
    assert res.inference is not None
    assert res.inference.details["conditional"] is True
    assert res.inference.ci_lower < res.inference.ci_upper


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_alpha0_must_be_below_alpha():
    with pytest.raises(MlsynthConfigError):
        DROSCConfig(**_basque_config(alpha=0.05, alpha0=0.10))


def test_config_rejects_negative_lambda():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        DROSCConfig(**_basque_config(robustness_lambda=-1.0))


def test_config_rejects_extra_field():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        DROSCConfig(**_basque_config(nonsense_field=1))


def test_dict_and_config_object_equivalent():
    cfg = DROSCConfig(**_basque_config(robustness_lambda=0.03))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = DROSC(cfg).fit().effects.att
        b = DROSC(_basque_config(robustness_lambda=0.03)).fit().effects.att
    assert a == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------- #
# options and plotting
# --------------------------------------------------------------------------- #
def test_dependent_hac_runs():
    res = _fit(robustness_lambda=0.0, dependent=True)
    assert np.isfinite(res.effects.att)


def test_plotting_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = tmp_path / "drosc.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DROSC(_basque_config(robustness_lambda=0.03, display_graphs=True,
                             save=str(out))).fit()
    plt.close("all")
    assert out.exists()


def test_single_treated_unit_required():
    # a panel with no treated unit dataprep-flags is a data error
    d = pd.read_csv(_BASQUE) if os.path.exists(_BASQUE) else None
    if d is None:
        pytest.skip("basque_jasa.csv not available")
    d = d[d.regionname != "Spain (Espana)"].copy()
    d["treat"] = 0
    from mlsynth.exceptions import MlsynthDataError
    with pytest.raises((MlsynthDataError, Exception)):
        DROSC(dict(df=d, outcome="gdpcap", treat="treat", unitid="regionname",
                   time="year", display_graphs=False)).fit()

"""Tests for the MEDSC estimator (mediation-analysis synthetic control).

Path A replicates Mellace & Pasquini (2022) Table 1 on Proposition 99 (the
cross-world direct effect and the negative, growing indirect price channel);
the remaining tests pin the decomposition identity, the dual-pool machinery,
the covariate (mscmt) path, placebo inference, and the config/failure contract.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlsynth import MEDSC
from mlsynth.config_models import MEDSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.medsc_helpers.plotter import plot_medsc
from mlsynth.utils.medsc_helpers.structures import MEDSCResults

_HERE = os.path.dirname(__file__)
_PROP99 = os.path.abspath(os.path.join(_HERE, "..", "..", "basedata",
                                       "prop99_mediation.csv"))

PROGRAM = ["Massachusetts", "Arizona", "Oregon", "Florida", "District of Columbia"]
TAX = ["Alaska", "Hawaii", "Maryland", "Michigan", "New Jersey", "New York",
       "Washington"]
TREATED = "California"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _prop99_frame():
    """Prop 99 mediation panel restricted to CA + the direct donor pool."""
    d = pd.read_csv(_PROP99)
    allst = sorted(d.state.unique())
    direct_pool = [s for s in allst if s not in [TREATED] + PROGRAM]
    total_pool = [s for s in direct_pool if s not in TAX]
    d = d[d.state.isin([TREATED] + direct_pool)].copy()
    d["treated"] = ((d.state == TREATED) & (d.year >= 1989)).astype(int)
    return d, total_pool, direct_pool


@pytest.fixture(scope="module")
def prop99():
    return _prop99_frame()


def _synth_panel(n_units=8, T=24, T0=16, seed=0, med_coef=2.0, direct_shift=3.0):
    """Synthetic panel: mediator carries part of the treatment effect.

    The treated unit's mediator is bumped post-treatment (an indirect channel)
    and its outcome additionally shifted (a direct channel), so both effects are
    nonzero and separable.
    """
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = 0.6 * common[t - 1] + rng.standard_normal()
    rows = []
    for i in range(n_units):
        load = rng.standard_normal()
        med = 1.0 + 0.04 * i + 0.03 * np.arange(T) + 0.03 * rng.standard_normal(T)
        treated = i == 0
        y = 20.0 + load * common - med_coef * med + rng.standard_normal(T) * 0.2
        for t in range(T):
            m = med[t]
            yy = y[t]
            if treated and t >= T0:
                m = med[t] + 0.5 * (t - T0 + 1)          # mediator moves -> indirect
                yy = 20.0 + load * common[t] - med_coef * m - direct_shift \
                    + rng.standard_normal() * 0.2         # + direct shift
            rows.append({"state": f"u{i:02d}", "year": 2000 + t,
                         "cigsale": float(yy), "price": float(m),
                         "lninc": float(5.0 + 0.1 * i),
                         "treated": int(treated and t >= T0)})
    return pd.DataFrame(rows)


def _base(df, **kw):
    cfg = {"df": df, "outcome": "cigsale", "mediator": "price",
           "treat": "treated", "unitid": "state", "time": "year",
           "display_graphs": False}
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# Path A: Prop 99 replication
# --------------------------------------------------------------------------- #
def test_path_a_direct_effect_reproduces_paper(prop99):
    """The cross-world direct effect matches Mellace-Pasquini Table 1."""
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    dec = res.decomposition
    yr = list(res.inputs.time_labels)
    d1995 = dec.direct[yr.index(1995)]
    d2000 = dec.direct[yr.index(2000)]
    # paper: Direct 1995 = -16.77, 2000 = -17.28
    assert d1995 == pytest.approx(-16.8, abs=1.0)
    assert d2000 == pytest.approx(-18.0, abs=1.5)
    # total tracks a canonical Abadie SC (well-fit pre-period)
    assert dec.pre_rmse_total < 3.0


def test_path_a_indirect_channel_starts_at_zero_and_grows(prop99):
    """The indirect (price) channel is ~0 at intervention and grows negative."""
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    dec = res.decomposition
    yr = list(res.inputs.time_labels)
    assert abs(dec.indirect[yr.index(1989)]) < 1.0        # starts near zero
    assert dec.indirect[yr.index(2000)] < -3.0            # negative by 2000
    # monotone-ish growth in magnitude across the late post-period
    assert dec.indirect[yr.index(2000)] < dec.indirect[yr.index(1992)]


# --------------------------------------------------------------------------- #
# Smoke + core contract
# --------------------------------------------------------------------------- #
def test_returns_effect_result_with_finite_effects(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    assert isinstance(res, MEDSCResults)
    assert np.isfinite(res.att)
    assert np.isfinite(res.att_direct)
    assert np.isfinite(res.att_indirect)
    assert res.inputs.n_post == res.inputs.T - res.inputs.T0
    assert res.inputs.L == 0


def test_total_weights_on_simplex(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    w = np.array(list(res.decomposition.total_weights.values()))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()


def test_decomposition_identity_holds(prop99):
    """indirect == total - direct in every post period."""
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    dec = res.decomposition
    post = np.arange(res.inputs.T0, res.inputs.T)
    assert np.allclose(dec.indirect[post], dec.total[post] - dec.direct[post])
    assert res.att_indirect == pytest.approx(res.att - res.att_direct, abs=1e-9)


def test_direct_defined_only_post_treatment(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    dec = res.decomposition
    T0 = res.inputs.T0
    assert np.all(np.isnan(dec.direct[:T0]))
    assert np.all(np.isfinite(dec.direct[T0:]))
    assert np.all(np.isfinite(dec.total))          # total defined everywhere


def test_att_matches_post_mean_total(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    T0 = res.inputs.T0
    assert res.att == pytest.approx(np.mean(res.decomposition.total[T0:]))


# --------------------------------------------------------------------------- #
# Dual pool
# --------------------------------------------------------------------------- #
def test_direct_donors_default_to_total(prop99):
    """direct_donors=None uses the total pool for the direct fit."""
    df, total_pool, _ = prop99
    res = MEDSC(_base(df, total_donors=total_pool)).fit()
    assert res.metadata["n_direct_donors"] == len(total_pool)
    assert res.metadata["n_total_donors"] == len(total_pool)


def test_wider_direct_pool_recorded(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool)).fit()
    assert res.metadata["n_direct_donors"] > res.metadata["n_total_donors"]


def test_all_donors_when_pools_unspecified():
    """No pool lists -> every non-treated unit is a donor."""
    df = _synth_panel()
    res = MEDSC(_base(df)).fit()
    assert res.metadata["n_total_donors"] == df.state.nunique() - 1


# --------------------------------------------------------------------------- #
# Covariate (mscmt) path
# --------------------------------------------------------------------------- #
def test_covariate_backend_runs_and_identity_holds():
    df = _synth_panel()
    res = MEDSC(_base(df, covariates=["lninc"], inference=False,
                      mscmt_maxiter=30, mscmt_popsize=8)).fit()
    assert res.metadata["backend"] == "mscmt"
    assert res.metadata["n_covariates"] == 1
    dec = res.decomposition
    post = np.arange(res.inputs.T0, res.inputs.T)
    assert np.allclose(dec.indirect[post], dec.total[post] - dec.direct[post])


def test_backend_outcome_only_explicit():
    df = _synth_panel()
    res = MEDSC(_base(df, backend="outcome-only")).fit()
    assert res.metadata["backend"] == "outcome-only"


def test_predictor_lags_option_runs():
    df = _synth_panel()
    lags = [2001, 2004, 2008]
    res = MEDSC(_base(df, predictor_lags=lags, inference=False)).fit()
    assert np.isfinite(res.att)
    assert np.isfinite(res.att_direct)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_placebo_inference_reports_pvalue(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool,
                      inference=True)).fit()
    p = res.inference.p_value
    assert p is not None and 0.0 < p <= 1.0
    assert res.inference.method == "placebo"
    assert res.inference.details["n_placebos"] > 0


def test_inference_disabled_yields_no_pvalue(prop99):
    df, total_pool, direct_pool = prop99
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool,
                      inference=False)).fit()
    assert res.inference.p_value is None
    assert res.inference.method is None


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def test_plot_smoke(prop99, monkeypatch):
    df, total_pool, direct_pool = prop99
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    res = MEDSC(_base(df, total_donors=total_pool, direct_donors=direct_pool,
                      display_graphs=True)).fit()
    assert isinstance(res, MEDSCResults)
    plot_medsc(res)          # direct call, too
    plt.close("all")


# --------------------------------------------------------------------------- #
# Config / construction contract
# --------------------------------------------------------------------------- #
def test_accepts_config_object():
    df = _synth_panel()
    cfg = MEDSCConfig(**_base(df))
    res = MEDSC(cfg).fit()
    assert isinstance(res, MEDSCResults)


def test_rejects_non_config():
    with pytest.raises(MlsynthConfigError):
        MEDSC(["not", "a", "config"])


def test_mediator_column_must_exist():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSCConfig(**_base(df, mediator="ghost"))


def test_mediator_must_differ_from_outcome():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSCConfig(**_base(df, mediator="cigsale"))


def test_empty_mediator_name_rejected():
    df = _synth_panel()
    with pytest.raises((MlsynthConfigError, ValueError)):
        MEDSCConfig(**_base(df, mediator="   "))


def test_missing_covariate_rejected():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSCConfig(**_base(df, covariates=["ghost"]))


def test_empty_donor_pool_rejected():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSCConfig(**_base(df, total_donors=[]))


def test_unknown_donor_raises_data_error():
    df = _synth_panel()
    with pytest.raises(MlsynthDataError):
        MEDSC(_base(df, total_donors=["ghost"])).fit()


def test_mscmt_backend_without_covariates_rejected():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSC(_base(df, backend="mscmt")).fit()


def test_predictor_lag_in_post_period_rejected():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSC(_base(df, predictor_lags=[2023])).fit()      # post-treatment year


def test_predictor_lag_unknown_label_rejected():
    df = _synth_panel()
    with pytest.raises(MlsynthConfigError):
        MEDSC(_base(df, predictor_lags=[1900])).fit()


def test_mediator_missing_for_treated_raises():
    df = _synth_panel()
    df.loc[(df.state == "u00") & (df.year == 2000), "price"] = np.nan
    with pytest.raises(MlsynthDataError):
        MEDSC(_base(df)).fit()


def test_nonfinite_donor_outcome_raises():
    df = _synth_panel()
    df.loc[(df.state == "u03") & (df.year == 2001), "cigsale"] = np.nan
    with pytest.raises(MlsynthDataError):
        MEDSC(_base(df)).fit()

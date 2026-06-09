"""Tests for the NumPy-first RESCM estimator and its laxscm_helpers subpackage."""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.laxscm import RESCM
from mlsynth.config_models import RESCMConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.laxscm_helpers import (
    ELASTIC,
    METHOD_SPECS,
    RELAXED,
    RESCMInputs,
    RESCMResults,
    ate_inference,
    derive_treatment,
    normalize_method,
    prepare_rescm_inputs,
    resolve_specs,
    run_rescm,
)
from mlsynth.utils.fast_scm_helpers.structure import IndexSet


@pytest.fixture
def panel() -> pd.DataFrame:
    """Factor-model panel: unit u0 treated from t>=30 with a -3 effect."""
    rng = np.random.default_rng(0)
    N, T, T0, r, EFFECT = 16, 40, 30, 3, -3.0
    f = np.cumsum(rng.normal(size=(T, r)), axis=0)
    rows = []
    for i in range(N):
        load = rng.normal(size=r)
        a = rng.normal()
        for t in range(T):
            y = a + load @ f[t] + rng.normal(scale=0.3)
            treat = int(i == 0 and t >= T0)
            if treat:
                y += EFFECT
            rows.append({"unit": f"u{i}", "time": t, "y": y, "treat": treat})
    return pd.DataFrame(rows)


@pytest.fixture
def inputs(panel) -> RESCMInputs:
    return prepare_rescm_inputs(panel, unitid="unit", time="time", outcome="y", treat="treat")


# --- specs / method registry ------------------------------------------------

def test_method_registry_branches():
    assert METHOD_SPECS["SC"].branch == ELASTIC
    assert METHOD_SPECS["RELAX_L2"].branch == RELAXED
    assert {s.branch for s in METHOD_SPECS.values()} == {ELASTIC, RELAXED}


def test_normalize_aliases():
    assert normalize_method("l2") == "RELAX_L2"
    assert normalize_method("linf") == "LINF"
    assert normalize_method("SC") == "SC"
    with pytest.raises(ValueError):
        normalize_method("not_a_method")


def test_resolve_specs_order():
    specs = resolve_specs(["RELAX_L2", "SC"])
    assert [s.name for s in specs] == ["RELAX_L2", "SC"]


# --- setup boundary ---------------------------------------------------------

def test_prepare_inputs_shapes(inputs):
    assert inputs.y.shape == (40,)
    assert inputs.X.shape == (40, 15)
    assert inputs.T0 == 30 and inputs.T2 == 10
    assert inputs.treated_label == "u0"
    assert len(inputs.unit_index) == 15


def test_setup_rejects_multiple_treated(panel):
    bad = panel.copy()
    bad.loc[(bad["unit"] == "u1") & (bad["time"] >= 35), "treat"] = 1
    with pytest.raises(MlsynthDataError):
        prepare_rescm_inputs(bad, unitid="unit", time="time", outcome="y", treat="treat")


def test_derive_treatment(panel):
    unit, t = derive_treatment(panel, "unit", "time", "treat")
    assert unit == "u0" and t == 30


# --- inference --------------------------------------------------------------

def test_ate_inference_recovers_constant_effect():
    T0, T2 = 20, 20
    gap = np.concatenate([np.zeros(T0), np.full(T2, -3.0)])
    att, se, ci, p = ate_inference(gap, T0)
    assert att == pytest.approx(-3.0)
    assert se == 0.0 or np.isnan(se)  # zero variance => degenerate


def test_ate_inference_noisy():
    rng = np.random.default_rng(2)
    gap = np.concatenate([np.zeros(30), -3.0 + rng.normal(scale=0.5, size=20)])
    att, se, ci, p = ate_inference(gap, 30)
    assert att == pytest.approx(-3.0, abs=0.4)
    assert se > 0 and ci[0] < att < ci[1] and p < 0.05


# --- engine orchestration ---------------------------------------------------

def test_run_rescm_recovers_effect(inputs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = run_rescm(inputs, ["SC", "LINF", "RELAX_L2"])
    assert set(fits) == {"SC", "LINF", "RELAX_L2"}
    for fit in fits.values():
        assert fit.counterfactual.shape == (40,)
        assert fit.gap.shape == (40,)
        assert fit.att == pytest.approx(-3.0, abs=1.5)
        assert fit.weights.shape == (15,)


# --- estimator class --------------------------------------------------------

def test_fit_returns_results_with_aliases(panel):
    cfg = RESCMConfig(df=panel, outcome="y", treat="treat", unitid="unit",
                      time="time", methods=["SC", "RELAX_L2"], display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RESCM(cfg).fit()
    assert isinstance(res, RESCMResults)
    assert res.selected_variant == "SC"
    assert res.att == res.fits["SC"].att
    assert set(res.att_by_method()) == {"SC", "RELAX_L2"}


def test_standardize_toggle(panel):
    """The relaxed branch's ``standardize`` flag is wired and changes the fit.

    The relaxed-balance L-infinity FOC is scale-sensitive, so toggling
    standardization yields a different relaxation weight vector (``False`` matches
    the authors' ``scmrelax`` reference, which solves on the raw series). Both
    settings must run and return a valid simplex-weighted relaxation.
    """
    def _fit(std):
        cfg = RESCMConfig(df=panel, outcome="y", treat="treat", unitid="unit",
                          time="time", methods=["RELAX_L2"], standardize=std,
                          n_taus=20, display_graphs=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return RESCM(cfg).fit().fits["RELAX_L2"]

    fit_std, fit_raw = _fit(True), _fit(False)
    # Both produce non-empty donor weights summing to ~1 (simplex).
    for fit in (fit_std, fit_raw):
        assert fit.donor_weights
        assert abs(sum(fit.donor_weights.values()) - 1.0) < 1e-2
    # The two regimes give materially different weight vectors.
    keys = set(fit_std.donor_weights) | set(fit_raw.donor_weights)
    diff = sum(abs(fit_std.donor_weights.get(k, 0.0) - fit_raw.donor_weights.get(k, 0.0))
               for k in keys)
    assert diff > 1e-3


def test_linf_penalty_is_inf_norm_not_l2():
    """Guard the alpha==0 shortcut bug: pure L-infinity (``L1_INF``) must differ
    from pure ridge (``L1_L2``). ``build_objective`` used to short-circuit
    ``alpha == 0`` to a squared-L2 penalty regardless of ``second_norm``, so
    ``LINF`` silently became ridge; a regression would make the two identical.
    """
    from mlsynth.utils.laxscm_helpers.crossval import fit_en_scm

    rng = np.random.default_rng(0)
    T0, J = 40, 12
    X = rng.normal(size=(T0 + 5, J))
    y = X @ rng.normal(size=J) + rng.normal(scale=0.5, size=T0 + 5)
    common = dict(
        X_pre=X[:T0], y_pre=y[:T0], X_post=X[T0:], y=y,
        donor_names=[f"d{j}" for j in range(J)], fit_intercept=True,
        constraint_type="unconstrained", standardize=False, alpha=[0.0], lam=[2.0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_inf = np.array(list(fit_en_scm(**common, second_norm="L1_INF")["donor_weights"].values()))
        w_l2 = np.array(list(fit_en_scm(**common, second_norm="L1_L2")["donor_weights"].values()))
    assert np.abs(w_inf - w_l2).max() > 1e-2


def test_linf_is_faithful_dense_wxy(panel):
    """LINF/L1LINF must realise the Wang-Xing-Ye dense L-infinity SC: weights
    that leave the simplex (negative and/or not summing to one) and are denser
    than classic SC -- not the sparse, simplex-bound near-SC they used to be.
    """
    cfg = RESCMConfig(df=panel, outcome="y", treat="treat", unitid="unit",
                      time="time", methods=["SC", "LINF"], display_graphs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RESCM(cfg).fit()
    sc = np.array(list(res.fits["SC"].donor_weights.values()))
    linf = np.array(list(res.fits["LINF"].donor_weights.values()))
    # LINF leaves the simplex: negative weights or a sum away from one.
    assert (linf < -1e-3).any() or abs(linf.sum() - 1.0) > 1e-2
    # LINF is denser than the sparse classic SC.
    assert np.sum(np.abs(linf) > 1e-3) > np.sum(np.abs(sc) > 1e-3)


def test_config_rejects_unknown_method(panel):
    with pytest.raises((MlsynthConfigError, ValueError)):
        RESCMConfig(df=panel, outcome="y", treat="treat", unitid="unit",
                    time="time", methods=["bogus"])


def test_config_default_methods(panel):
    cfg = RESCMConfig(df=panel, outcome="y", treat="treat", unitid="unit", time="time")
    assert cfg.methods == ["SC", "LINF", "RELAX_L2"]

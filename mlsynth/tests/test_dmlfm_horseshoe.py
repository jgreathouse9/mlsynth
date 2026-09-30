"""Tests for the horseshoe prior on DMLFM.

Ma, X., Gao, Q., Wang, J., Wang, M., & Zhu, C. (2026). "A Bayesian synthetic
control method via horseshoe priors." Economic Modelling 157:107502.

H-BSCM is Pang, Liu & Xu (2022) with the Bayesian lasso on each shrinkage block
replaced by a horseshoe, drawn through the Makalic & Schmidt (2015)
inverse-gamma auxiliary scheme. Appendix A of the paper states the change
exactly: steps (2)-(3) and (7)-(12) of the Pang et al. sampler, which are the
scale updates and nothing else.

Written before the implementation. The oracle is ``function-code.R`` from the
authors' replication archive, whose four ``xhorseshoe``/``zhorseshoe``/
``ahorseshoe``/``fhorseshoe`` blocks this mirrors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import DMLFM
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.tests.test_dmlfm import base_config, make_panel
from mlsynth.utils.dmlfm_helpers.config import DMLFMConfig
from mlsynth.utils.dmlfm_helpers.sampler import _horseshoe_update

SEED = 20260930


# --------------------------------------------------------------------------
# unit: the shrinkage block itself
# --------------------------------------------------------------------------
def _fresh(k: int):
    """Block state as the sampler initialises it: unit scales, unit auxiliaries."""
    return np.ones(k), np.ones(k), 1.0, 1.0      # prior_var, nu, tau2, xi


def test_horseshoe_update_returns_positive_scales():
    rng = np.random.default_rng(SEED)
    coef = rng.normal(size=6)
    prior_var, nu, tau2, xi = _fresh(6)
    tau2, xi = _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
    assert tau2 > 0.0, "the global scale must stay positive"
    assert xi > 0.0, "the global auxiliary must stay positive"
    assert np.all(prior_var > 0.0), "every local prior variance must be positive"
    assert np.all(nu > 0.0), "every local auxiliary must be positive"


def test_horseshoe_update_writes_prior_var_in_place():
    """The sampler passes ``var_beta`` and friends by reference, as the lasso does."""
    rng = np.random.default_rng(SEED)
    coef = rng.normal(size=4)
    prior_var, nu, tau2, xi = _fresh(4)
    before = prior_var.copy()
    _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
    assert prior_var.shape == (4,)
    assert not np.allclose(prior_var, before), "prior_var was not updated"


def test_horseshoe_update_mutates_the_auxiliary_in_place():
    rng = np.random.default_rng(SEED)
    coef = rng.normal(size=4)
    prior_var, nu, tau2, xi = _fresh(4)
    before = nu.copy()
    _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
    assert not np.allclose(nu, before), "the local auxiliary nu was not updated"


def test_horseshoe_leaves_a_strong_signal_less_shrunk_than_noise():
    """The global-local property: one large coefficient beside near-zero ones.

    The horseshoe is defended over the lasso precisely because the local scale
    adapts per coefficient, so a genuine signal keeps a wide prior while the
    noise coefficients are pulled to a narrow one. Averaged over sweeps to take
    the property off a single draw.
    """
    rng = np.random.default_rng(SEED)
    coef = np.array([8.0, 0.01, -0.02, 0.005, 0.0])
    prior_var, nu, tau2, xi = _fresh(coef.size)
    signal, noise = [], []
    for _ in range(200):
        tau2, xi = _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
        signal.append(prior_var[0])
        noise.append(prior_var[1:].mean())
    assert np.median(signal) > np.median(noise), (
        f"signal prior variance {np.median(signal):.4g} should exceed noise "
        f"{np.median(noise):.4g}")


def test_horseshoe_update_is_deterministic_under_a_seed():
    def run():
        rng = np.random.default_rng(SEED)
        coef = np.array([1.0, -0.5, 0.25])
        prior_var, nu, tau2, xi = _fresh(3)
        tau2, xi = _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
        return tau2, xi, prior_var.copy(), nu.copy()

    a, b = run(), run()
    assert a[0] == b[0] and a[1] == b[1]
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])


# --------------------------------------------------------------------------
# edge: degenerate coefficient blocks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("coef", [
    np.zeros(3),                       # every coefficient exactly zero
    np.array([1e-300, 1e-300]),        # underflowing magnitudes
    np.array([1e6, -1e6]),             # very large magnitudes
    np.array([2.0]),                   # a single coefficient
])
def test_horseshoe_update_survives_degenerate_blocks(coef):
    rng = np.random.default_rng(SEED)
    prior_var, nu, tau2, xi = _fresh(coef.size)
    tau2, xi = _horseshoe_update(coef, tau2, prior_var, nu, xi, rng)
    assert np.isfinite(tau2) and tau2 > 0.0
    assert np.isfinite(xi) and xi > 0.0
    assert np.all(np.isfinite(prior_var)) and np.all(prior_var > 0.0)
    assert np.all(np.isfinite(nu)) and np.all(nu > 0.0)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_prior_defaults_to_lasso():
    cfg = DMLFMConfig(**base_config(make_panel()))
    assert cfg.prior == "lasso", "the default must not move; every pin assumes it"


def test_prior_accepts_horseshoe():
    cfg = DMLFMConfig(**base_config(make_panel(), prior="horseshoe"))
    assert cfg.prior == "horseshoe"


def test_unknown_prior_is_refused():
    with pytest.raises((MlsynthConfigError, ValueError)):
        DMLFMConfig(**base_config(make_panel(), prior="ridge"))


# --------------------------------------------------------------------------
# smoke and integration
# --------------------------------------------------------------------------
def test_horseshoe_fit_returns_a_finite_att():
    res = DMLFM(base_config(make_panel(), prior="horseshoe")).fit()
    att = float(res.effects.att)
    assert np.isfinite(att)


def test_horseshoe_recovers_the_planted_effect():
    """``make_panel`` plants an additive effect of 2.0 on the treated unit."""
    res = DMLFM(base_config(make_panel(), prior="horseshoe", niter=1200,
                            burn=400)).fit()
    assert abs(float(res.effects.att) - 2.0) < 1.0


def test_horseshoe_fills_the_same_result_contract_as_lasso():
    hs = DMLFM(base_config(make_panel(), prior="horseshoe")).fit()
    la = DMLFM(base_config(make_panel(), prior="lasso")).fit()
    assert type(hs) is type(la)
    assert set(hs.model_dump()) == set(la.model_dump())
    for side in ("ci_lower", "ci_upper"):
        assert np.isfinite(float(getattr(hs.inference, side)))
    assert float(hs.inference.ci_lower) <= float(hs.effects.att) <= float(
        hs.inference.ci_upper)


def test_horseshoe_counterfactual_spans_the_whole_panel():
    res = DMLFM(base_config(make_panel(), prior="horseshoe")).fit()
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    assert cf.shape[0] == 30
    assert np.all(np.isfinite(cf))


def test_horseshoe_reports_its_prior_in_method_details():
    res = DMLFM(base_config(make_panel(), prior="horseshoe")).fit()
    blob = str(res.method_details.model_dump())
    assert "horseshoe" in blob, "the fitted prior must be recoverable from the result"


# --------------------------------------------------------------------------
# regression: the lasso path must not move
# --------------------------------------------------------------------------
def test_lasso_path_is_bit_identical_to_the_default():
    """Adding the switch must not perturb the validated lasso draw sequence."""
    explicit = DMLFM(base_config(make_panel(), prior="lasso")).fit()
    default = DMLFM(base_config(make_panel())).fit()
    assert float(explicit.effects.att) == float(default.effects.att)
    np.testing.assert_array_equal(
        np.asarray(explicit.time_series.counterfactual_outcome, dtype=float),
        np.asarray(default.time_series.counterfactual_outcome, dtype=float))


def test_horseshoe_and_lasso_disagree():
    """A prior that changed nothing would be a switch that is not wired up."""
    hs = DMLFM(base_config(make_panel(), prior="horseshoe")).fit()
    la = DMLFM(base_config(make_panel(), prior="lasso")).fit()
    assert float(hs.effects.att) != float(la.effects.att)

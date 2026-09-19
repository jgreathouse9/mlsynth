"""The Bayesian GEOX engine: MVBBSC in the augsynth slot.

The engine seam is contract-first, and
``mlsynth/tests/test_geox_engine_properties.py`` parametrizes its metamorphic
relations over ``ENGINE_NAMES``, so registering this engine subjects it to that
suite by construction. What is here is the part that suite cannot reach: the
things specific to a sampler-backed engine.

Two of those need saying.

A posterior is not a calibrated predictive interval. MVBBSC's own counterfactual
carries an iid shock, and the ATT averages a whole post window, so under iid the
variance of that mean falls as ``sigma^2 / h`` while under positive
autocorrelation it falls more slowly. Measured on Meta's GeoLift panel the
shipped form covers 65% of a nominal 90% and the corrected form 87.5%. The
engine therefore carries the pre-period AR(1) into the predictive shock, and
``test_autocorrelated_shock_widens_the_interval`` pins that it does.

NUTS is chaotic, so donor column order is information to a sampler even though
it is not information to the estimand: permuting the donors moved the posterior
weights by 7e-3 against the suite's 1e-6 tolerance. ``run_mvbbsc`` canonicalises
its own columns, so the engine inherits the invariance instead of re-imposing
it. ``test_donor_order_is_invariant`` asserts it holds through the seam, which
is the thing the scoring loop depends on when nomination hands candidates over
in whatever order it produced them, and it is a separate claim from the
estimator-level property in ``test_mvbbsc_donor_order.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.geox_helpers.engines import ENGINE_NAMES, resolve_engine

numpyro = pytest.importorskip("numpyro", reason="the mvbbsc engine needs mlsynth[bayes]")


def _panel(n_periods=24, n_donors=5, n_pre=19, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    donors = (rng.uniform(50.0, 500.0, n_donors)[None, :]
              + (10.0 * np.sin(2 * np.pi * t / 7.0))[:, None]
              + rng.normal(0.0, 5.0, (n_periods, n_donors)))
    y = donors @ rng.dirichlet(np.ones(n_donors)) + rng.normal(0.0, 3.0, n_periods)
    return y, donors, n_pre, n_periods - 1


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------
def test_engine_is_registered():
    assert "mvbbsc" in ENGINE_NAMES
    assert resolve_engine("mvbbsc").name == "mvbbsc"


def test_fit_once_returns_the_contract_shapes():
    y, Y0, n_pre, end = _panel()
    fit = resolve_engine("mvbbsc").fit_once(y, Y0, n_pre, n_pre, end, 1)
    assert fit.counterfactual.shape == y.shape
    assert fit.donor_weights.shape == (Y0.shape[1],)
    assert np.all(np.isfinite(fit.counterfactual))
    assert np.isfinite(fit.pre_rmspe) and fit.pre_rmspe >= 0.0


def test_donor_weights_are_a_simplex():
    # MVBBSC's prior is a Dirichlet, so the posterior mean stays on the simplex.
    y, Y0, n_pre, end = _panel()
    fit = resolve_engine("mvbbsc").fit_once(y, Y0, n_pre, n_pre, end, 1)
    assert np.all(fit.donor_weights >= -1e-9)
    assert fit.donor_weights.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# the two mechanisms the property suite cannot reach
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", [0, 1, 2])
def test_donor_order_is_invariant(key):
    """Relabelling donors permutes the weights identically, exactly.

    Inherited from ``run_mvbbsc``, and asserted here because the engine is where
    the scoring loop consumes it. Without the canonicalisation the sampler sees
    a different input and the weights move by about 7e-3, which the engine
    property suite rejects at 1e-6.
    """
    y, Y0, n_pre, end = _panel()
    eng = resolve_engine("mvbbsc")
    perm = np.random.default_rng(key).permutation(Y0.shape[1])
    base = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
    shuffled = eng.fit_once(y, Y0[:, perm], n_pre, n_pre, end, 1)
    np.testing.assert_allclose(shuffled.donor_weights, base.donor_weights[perm],
                               atol=1e-12)
    np.testing.assert_allclose(shuffled.counterfactual, base.counterfactual,
                               atol=1e-12)


def test_autocorrelated_shock_widens_the_interval():
    """The AR(1) shock is what makes the interval cover an averaged estimand.

    An iid shock understates the variance of a post-window mean whenever the
    residuals are positively autocorrelated, so the corrected interval is the
    wider of the two on such a panel.
    """
    rng = np.random.default_rng(3)
    n_periods, n_donors, n_pre = 40, 4, 30
    donors = rng.normal(100.0, 5.0, (n_periods, n_donors))
    resid = np.zeros(n_periods)
    for t in range(1, n_periods):           # strongly autocorrelated gap
        resid[t] = 0.85 * resid[t - 1] + rng.normal(0.0, 2.0)
    y = donors @ rng.dirichlet(np.ones(n_donors)) + resid
    eng = resolve_engine("mvbbsc")
    fit = eng.fit_once(y, donors, n_pre, n_pre, n_periods - 1, 1)
    assert abs(fit.extras["rho"]) > 0.2, "the fixture must have autocorrelation to detect"
    _, ar = eng.point_inference(fit, y, donors, n_pre, n_pre, n_periods - 1,
                                alpha=0.1, autocorr=True)
    _, iid = eng.point_inference(fit, y, donors, n_pre, n_pre, n_periods - 1,
                                 alpha=0.1, autocorr=False)
    assert (ar["ci_upper"] - ar["ci_lower"]) > (iid["ci_upper"] - iid["ci_lower"])


def test_point_inference_reports_a_credible_interval_around_the_att():
    y, Y0, n_pre, end = _panel()
    eng = resolve_engine("mvbbsc")
    fit = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
    p, details = eng.point_inference(fit, y, Y0, n_pre, n_pre, end, alpha=0.1)
    assert 0.0 <= p <= 1.0
    assert details["ci_lower"] < details["ci_upper"]
    assert details["ci_lower"] <= details["att"] <= details["ci_upper"]
    assert np.isfinite(details["max_rhat"])


def test_sweep_matches_the_analytic_shift():
    """The fit uses pre-period data alone, so injecting an effect moves the ATT
    by exactly ``effect * mean(treated post)`` and nothing else."""
    y, Y0, n_pre, end = _panel()
    eng = resolve_engine("mvbbsc")
    fit = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
    effects = [-0.2, 0.0, 0.15]
    out = eng.sweep_p_values(fit, y, Y0, n_pre, n_pre, end, effects, alpha=0.1)
    baseline = float(np.mean(y[n_pre:end + 1]))
    for e, tau in zip(effects, out["tau"]):
        assert tau == pytest.approx(out["tau0"] + e * baseline, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------
def test_single_donor():
    y, Y0, n_pre, end = _panel(n_donors=1)
    fit = resolve_engine("mvbbsc").fit_once(y, Y0, n_pre, n_pre, end, 1)
    assert fit.donor_weights.shape == (1,)
    assert fit.donor_weights[0] == pytest.approx(1.0, abs=1e-6)


def test_constant_treated_series():
    # A flat target has zero pre-period variance; the standardisation must not
    # divide by it and the fit must stay finite.
    _, Y0, n_pre, end = _panel()
    y = np.full(Y0.shape[0], 250.0)
    fit = resolve_engine("mvbbsc").fit_once(y, Y0, n_pre, n_pre, end, 1)
    assert np.all(np.isfinite(fit.counterfactual))


def test_single_post_period():
    y, Y0, _, _ = _panel()
    n_pre = Y0.shape[0] - 1
    end = Y0.shape[0] - 1
    eng = resolve_engine("mvbbsc")
    fit = eng.fit_once(y, Y0, n_pre, n_pre, end, 1)
    _, details = eng.point_inference(fit, y, Y0, n_pre, n_pre, end, alpha=0.1)
    assert details["ci_lower"] < details["ci_upper"]


# --------------------------------------------------------------------------
# configuration contract
# --------------------------------------------------------------------------
def _cfg(**over):
    import pandas as pd

    from mlsynth.config_models import GEOXConfig

    rng = np.random.default_rng(0)
    rows = [{"location": f"g{g}", "date": t, "Y": 100.0 + g + rng.normal()}
            for g in range(6) for t in range(20)]
    base = dict(df=pd.DataFrame(rows), outcome="Y", unitid="location",
                time="date", treatment_size=[2], durations=[3],
                effect_sizes=[0.0, 0.1], n_backtests=1, seed=0)
    base.update(over)
    return GEOXConfig(**base)


def test_bayes_is_the_default_inference_for_the_bayesian_engine():
    assert _cfg(engine="mvbbsc").inference == "bayes"


def test_bayes_inference_is_rejected_for_a_frequentist_engine():
    with pytest.raises(MlsynthConfigError, match="bayes"):
        _cfg(engine="sdid", inference="bayes")


def test_bayesian_engine_rejects_a_frequentist_null():
    with pytest.raises(MlsynthConfigError, match="mvbbsc"):
        _cfg(engine="mvbbsc", inference="conformal")


def test_existing_engine_defaults_are_unchanged():
    # The new value must not move what the other engines resolve to.
    assert _cfg(engine="augsynth").inference == "conformal"
    assert _cfg(engine="sdid").inference == "placebo"

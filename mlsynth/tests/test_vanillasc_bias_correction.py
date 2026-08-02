"""Abadie-L'Hour bias correction on a general synthetic control.

The correction removes the part of the gap attributable to residual predictor
imbalance ``X1 - X0 W``::

    tau_bc(t) = (y1_t - Y0_t W) - (X1 - X0 W)' beta_t

with ``beta_t`` the (ridge) slope of the donor outcome on the donor predictors.
``bias_corrected_gaps`` has always been general -- it takes ``W`` from any
backend -- but only ``SPILLSYNTH`` could reach it, and only on its predictors
branch.

Whether this matters is not a matter of taste. On Wiltshire's (2023) Walmart
panel the correction roughly doubles the headline: aggregate employment goes
from -1.77% to -3.14%, and the corrected figure is the one the paper quotes.

Two properties make it testable without a reference implementation. The
correction is exactly zero when the predictors balance, because the imbalance
vector it multiplies is zero; and it is bounded by the ridge, which is what
stops a weakly-explanatory predictor set from injecting noise.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _panel(n_units=8, T=12, pre=9, seed=0, shift=0.0):
    """Donor pool plus a treated unit. ``shift`` moves the treated unit's
    predictors away from anything the donors can match, forcing imbalance."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        base = float(rng.normal()) * 3.0
        for t in range(T):
            treated = u == 0
            rows.append({
                "unit": f"u{u}", "t": t,
                "y": float(base + 0.4 * t + rng.normal() * 0.2),
                "x1": float(rng.normal()) + (shift if treated else 0.0),
                "x2": float(rng.normal()) * 50.0,
                "treat": int(treated and t >= pre),
            })
    return pd.DataFrame(rows)


def _fit(df=None, **over):
    from mlsynth import VanillaSC

    cfg = {"df": _panel() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "t",
           "covariates": ["x1", "x2"], "backend": "mscmt",
           "inference": False, "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


# ---------------------------------------------------------------- smoke
class TestSmoke:
    def test_it_fits_with_the_correction_on(self):
        res = _fit(bias_correct=True)
        assert np.isfinite(res.effects.att)

    def test_the_correction_is_recorded_in_the_spec(self):
        res = _fit(bias_correct=True)
        params = res.method_details.parameters_used or {}
        assert params.get("bias_correct") is True
        assert params.get("bias_correct_ridge") is not None


# ------------------------------------------------- the default is unchanged
class TestTheDefaultIsUnchanged:
    def test_the_correction_is_off_by_default(self):
        from mlsynth.utils.vanillasc_helpers.config import VanillaSCConfig

        assert VanillaSCConfig.model_fields["bias_correct"].default is False

    def test_omitting_it_reproduces_the_previous_estimate_exactly(self):
        """Adding a knob must not move the number anybody already has."""
        plain = _fit()
        explicit_off = _fit(bias_correct=False)
        assert plain.effects.att == pytest.approx(explicit_off.effects.att,
                                                  rel=1e-12)
        np.testing.assert_allclose(
            np.asarray(plain.time_series.estimated_gap, float),
            np.asarray(explicit_off.time_series.estimated_gap, float),
            atol=1e-12)


# ---------------------------------------------------------------- the rule
class TestTheCorrectionDoesWhatItClaims:
    def test_it_vanishes_when_the_predictors_balance(self):
        """``tau_bc - tau = -(X1 - X0 W)' beta``. With the treated unit's
        predictors inside the donor hull the simplex fit can balance them
        exactly, the imbalance vector is zero, and the correction must be too.
        """
        from mlsynth.utils.bilevel import bias_corrected_gaps

        rng = np.random.default_rng(1)
        J, K, T = 6, 3, 8
        X0 = rng.normal(size=(K, J))
        W = rng.dirichlet(np.ones(J))
        X1 = X0 @ W                                   # exactly balanced
        Y0 = rng.normal(size=(T, J))
        y1 = Y0 @ W + rng.normal(size=T) * 0.1
        bc = bias_corrected_gaps(W, X1, X0, y1, Y0)
        raw = y1 - Y0 @ W
        np.testing.assert_allclose(bc, raw, atol=1e-10)

    def test_it_moves_the_gap_when_the_predictors_do_not_balance(self):
        from mlsynth.utils.bilevel import bias_corrected_gaps

        rng = np.random.default_rng(2)
        J, K, T = 6, 3, 8
        X0 = rng.normal(size=(K, J))
        W = rng.dirichlet(np.ones(J))
        X1 = X0 @ W + np.array([1.0, -0.5, 0.25])     # deliberate imbalance
        Y0 = rng.normal(size=(T, J))
        y1 = Y0 @ W
        bc = bias_corrected_gaps(W, X1, X0, y1, Y0)
        raw = y1 - Y0 @ W
        assert not np.allclose(bc, raw, atol=1e-6)

    def test_a_large_ridge_shrinks_the_correction_toward_zero(self):
        """The ridge is why a weakly-explanatory predictor set cannot inject
        noise: it bounds the slope, hence the correction."""
        from mlsynth.utils.bilevel import bias_corrected_gaps

        rng = np.random.default_rng(3)
        J, K, T = 6, 3, 8
        X0 = rng.normal(size=(K, J))
        W = rng.dirichlet(np.ones(J))
        X1 = X0 @ W + np.array([1.0, -0.5, 0.25])
        Y0 = rng.normal(size=(T, J))
        y1 = Y0 @ W
        raw = y1 - Y0 @ W
        small = np.abs(bias_corrected_gaps(W, X1, X0, y1, Y0, ridge=1e-8) - raw)
        large = np.abs(bias_corrected_gaps(W, X1, X0, y1, Y0, ridge=1e6) - raw)
        assert large.max() < small.max()
        assert large.max() < 1e-3

    def test_the_ridge_setting_reaches_the_estimator(self):
        loose = _fit(bias_correct=True, bias_correct_ridge=1e-8)
        tight = _fit(bias_correct=True, bias_correct_ridge=1e6)
        assert loose.effects.att != pytest.approx(tight.effects.att, rel=1e-9)

    def test_a_huge_ridge_collapses_onto_the_uncorrected_fit(self):
        """Consistency between the two paths: shrink the correction to nothing
        and the corrected estimator must agree with the uncorrected one."""
        off = _fit(bias_correct=False)
        shrunk = _fit(bias_correct=True, bias_correct_ridge=1e12)
        assert shrunk.effects.att == pytest.approx(off.effects.att, abs=1e-6)


# ---------------------------------------------------------------- failures
class TestFailuresAreReported:
    def test_asking_for_it_without_covariates_is_an_error(self):
        """The SPILLSYNTH path silently returns the raw gap when predictors are
        absent. A request that cannot be honoured should say so."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises((MlsynthConfigError, ValueError), match="covariat"):
            _fit(covariates=None, bias_correct=True)

    def test_a_negative_ridge_is_rejected(self):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises((MlsynthConfigError, ValueError)):
            _fit(bias_correct=True, bias_correct_ridge=-1.0)

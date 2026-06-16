"""TDD for the Orthogonalized Synthetic Control (OSC) helpers.

OSC (Fry, "Orthogonalized Synthetic Controls") is an IV synthetic control whose
ATT estimate is Neyman-orthogonalized with respect to the (partially identified,
high-dimensional, simplex-constrained) control weights, with a fixed-smoothing
Series-HAC variance and a Sun (2013) bandwidth giving a t-test that controls
size without a consistent variance.

Layer 1 (numerical helpers) is tested by invariants here -- feasibility,
normalization, dimensional correctness, finiteness -- not brittle floats. The
exact value-for-value anchor is the integration test against the live R
reference on Andersson (2019)'s carbon-tax panel: with control pool = Andersson's
14 donors and instruments = the 7 carbon/fuel-tax countries he excluded, the R
package returns ATT = -0.29013, p = 0.000183, smoothing K = 4 -- which the
pipeline must reproduce.

Array conventions (clean; the helpers absorb the reference's transposes):
    pre_y0   (T0,)        treated unit, pre-period
    pre_yj   (J, T0)      control units x pre-period
    Z        (Q, T0)      instrument units x pre-period (no constant; helpers add it)
    post_y0  (T1,)        treated unit, post-period
    post_yj  (J, T1)      control units x post-period
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.osc_helpers.serieshac import (
    orthonormal_basis,
    series_hac_variance,
    cpe_optimal_h,
    ttest_pvalue,
    ttest_ci,
)
from mlsynth.utils.osc_helpers.regularized import estimate_delta, estimate_eta
from mlsynth.utils.osc_helpers.orthogonal import orthogonalized_att
from mlsynth.utils.osc_helpers.pipeline import orthogonalized_sce


# ---------------------------------------------------------------- fixtures ----

def exact_sc_panel(J=5, T0=30, T1=16, tau=-0.3, seed=0):
    """A panel where the treated unit is an exact convex combination of controls
    pre-treatment, plus a constant additive effect ``tau`` post-treatment.

    Returns the OSC arrays plus the planted ``delta_true`` and ``tau``. Instruments
    are extra untreated series correlated with the controls (the exclusion
    restriction holds by construction in the pre-period up to noise).
    """
    rng = np.random.default_rng(seed)
    T = T0 + T1
    # latent factors drive everyone (linear factor model)
    F = rng.normal(size=(T, 2))
    load_ctrl = rng.uniform(0.5, 1.5, size=(2, J))
    YJ = (F @ load_ctrl).T + 0.01 * rng.normal(size=(J, T))      # (J, T)
    delta_true = rng.dirichlet(np.ones(J))                       # simplex weights
    treated = delta_true @ YJ                                    # exact pre-fit
    Y0 = treated.copy()
    Y0[T0:] += tau                                               # additive effect post
    # instruments: 3 more untreated units sharing the factors
    load_z = rng.uniform(0.5, 1.5, size=(2, 3))
    Z_full = (F @ load_z).T + 0.01 * rng.normal(size=(3, T))     # (3, T)
    return dict(
        pre_y0=Y0[:T0], pre_yj=YJ[:, :T0], Z=Z_full[:, :T0],
        post_y0=Y0[T0:], post_yj=YJ[:, T0:],
        delta_true=delta_true, tau=tau,
    )


_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "carbontax_fullsample_data.dta.txt")
_CONTROLS = ["Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
             "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
             "Switzerland", "United States"]                     # Andersson's 14
_INSTRS = ["Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
           "United Kingdom"]                                     # his excluded 7


def carbontax_arrays():
    """Andersson (2019) carbon-tax panel as the OSC arrays (Sweden treated, 1990
    break, T0=30 / T1=16); skip if the .dta is unreadable."""
    df = pd.read_stata(os.path.abspath(_DATA))
    wide = df.pivot(index="year", columns="country", values="CO2_transport_capita")
    pre = wide.index < 1990
    post = wide.index >= 1990
    return dict(
        pre_y0=wide.loc[pre, "Sweden"].to_numpy(),
        pre_yj=wide.loc[pre, _CONTROLS].to_numpy().T,
        Z=wide.loc[pre, _INSTRS].to_numpy().T,
        post_y0=wide.loc[post, "Sweden"].to_numpy(),
        post_yj=wide.loc[post, _CONTROLS].to_numpy().T,
    )


# ------------------------------------------------ Layer 1: Series-HAC core ----

class TestSeriesHAC:
    def test_orthonormal_basis_is_orthonormal(self):
        # <phi_i, phi_j> ~ delta_ij on a fine grid of [0, 1]
        x = (np.arange(1, 2001) - 0.5) / 2000.0
        B = np.column_stack([orthonormal_basis(x, j) for j in range(1, 6)])
        G = (B.T @ B) / len(x)
        assert np.allclose(G, np.eye(5), atol=1e-2)

    def test_series_hac_variance_finite_positive_scalar(self):
        rng = np.random.default_rng(1)
        preg = rng.normal(size=(3, 30)); postg = rng.normal(size=16)
        eta = rng.normal(size=4); eta[-1] = 1.0
        V = series_hac_variance(preg, postg, eta, h=4)
        assert np.isscalar(V) or np.ndim(V) == 0
        assert np.isfinite(V) and V > 0

    def test_cpe_optimal_h_returns_positive_even_int_in_range(self):
        rng = np.random.default_rng(2)
        # AR(1) pre-residuals
        e = np.zeros((1, 60))
        for t in range(1, 60):
            e[0, t] = 0.5 * e[0, t - 1] + rng.normal()
        k = cpe_optimal_h(e, p=1, sig=0.05)
        assert isinstance(k, (int, np.integer))
        assert 0 < k <= 60 and k % 2 == 0

    def test_ttest_pvalue_in_unit_interval_and_monotone(self):
        p_near = ttest_pvalue(beta_hat=0.01, V=1.0, h=4, n=16, beta0=0.0)
        p_far = ttest_pvalue(beta_hat=5.0, V=1.0, h=4, n=16, beta0=0.0)
        assert 0.0 <= p_far <= p_near <= 1.0
        assert ttest_pvalue(beta_hat=0.0, V=1.0, h=4, n=16, beta0=0.0) == pytest.approx(1.0)

    def test_ttest_ci_brackets_estimate_and_widens_with_variance(self):
        lo1, hi1 = ttest_ci(beta_hat=-0.3, V=0.01, h=4, alpha=0.05)
        lo2, hi2 = ttest_ci(beta_hat=-0.3, V=0.04, h=4, alpha=0.05)
        assert lo1 < -0.3 < hi1
        assert (hi2 - lo2) > (hi1 - lo1)              # larger V -> wider CI


# --------------------------------------- Layer 1: regularized nuisance fit ----

class TestRegularizedWeights:
    def test_estimate_delta_on_simplex(self):
        d = exact_sc_panel()
        out = estimate_delta(d["pre_y0"], d["pre_yj"], d["Z"], scaled=True)
        delta = np.asarray(out["delta"])
        assert delta.shape == (d["pre_yj"].shape[0],)
        assert np.all(delta >= -1e-8)
        assert np.isclose(delta.sum(), 1.0, atol=1e-6)
        assert np.all(np.isfinite(delta))

    def test_estimate_delta_fits_exact_combination(self):
        # exact pre-fit: the synthetic control reproduces the treated pre-path
        d = exact_sc_panel()
        out = estimate_delta(d["pre_y0"], d["pre_yj"], d["Z"], scaled=True)
        fit = np.asarray(out["delta"]) @ d["pre_yj"]
        assert np.allclose(fit, d["pre_y0"], atol=1e-2)

    def test_estimate_eta_finite_1d(self):
        d = exact_sc_panel()
        out = estimate_eta(d["pre_y0"], d["pre_yj"], d["post_y0"], d["post_yj"],
                           d["Z"], scaled=True)
        eta = np.asarray(out["eta"])
        assert eta.ndim == 1 and np.all(np.isfinite(eta))


# ----------------------------------------- Layer 1: orthogonalized ATT --------

class TestOrthogonalizedATT:
    def test_orthogonalized_att_shapes_and_finite(self):
        d = exact_sc_panel()
        J, T0 = d["pre_yj"].shape
        delta = np.full(J, 1.0 / J)
        eta = np.ones(d["Z"].shape[0] + 2)            # instruments + constant + post
        out = orthogonalized_att(d["pre_y0"], d["pre_yj"], d["Z"],
                                 d["post_y0"], d["post_yj"], delta, eta,
                                 include_constant=True)
        assert np.isfinite(out["beta"])
        assert np.asarray(out["preg"]).shape[1] == T0
        assert np.asarray(out["postg"]).shape[-1] == d["post_yj"].shape[1]


# --------------------------------- Layer 3: pipeline + carbon-tax anchor ------

class TestPipeline:
    def test_pipeline_recovers_planted_effect(self):
        d = exact_sc_panel(tau=-0.3, seed=3)
        res = orthogonalized_sce(d["pre_y0"], d["pre_yj"], d["Z"],
                                 d["post_y0"], d["post_yj"])
        assert res["beta"] == pytest.approx(-0.3, abs=0.05)
        assert np.isfinite(res["pvalue"]) and 0.0 <= res["pvalue"] <= 1.0
        lo, hi = res["ci"]
        assert lo < res["beta"] < hi

    def test_pipeline_matches_R_on_carbontax(self):
        """Value-for-value cross-validation against the GeoLift-style R reference
        (live R run): ATT -0.29013, p 0.000183, smoothing K 4."""
        a = carbontax_arrays()
        res = orthogonalized_sce(a["pre_y0"], a["pre_yj"], a["Z"],
                                 a["post_y0"], a["post_yj"])
        assert res["beta"] == pytest.approx(-0.29013, abs=0.01)
        assert res["pvalue"] < 0.001
        assert int(res["df"]) == 4


# ----------------------------------------------------- failure semantics ------

class TestFailures:
    def test_mismatched_pre_periods_raise(self):
        d = exact_sc_panel()
        with pytest.raises((MlsynthEstimationError, MlsynthConfigError)):
            estimate_delta(d["pre_y0"][:-1], d["pre_yj"], d["Z"])   # T0 mismatch

    def test_empty_instruments_raise(self):
        d = exact_sc_panel()
        with pytest.raises((MlsynthEstimationError, MlsynthConfigError)):
            estimate_delta(d["pre_y0"], d["pre_yj"], d["Z"][:0])    # no instruments

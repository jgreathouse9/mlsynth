"""Staggered/multi-treated scpi engine: the weight-constraint family (inference).

Task 93 generalized the *fit* (``b_est``) to scpi's ols / simplex / lasso / ridge
/ L1-L2 family. This validates the *inference* deterministic layer that feeds the
prediction-interval conic program:

* ``df_EST`` — effective degrees of freedom per constraint;
* ``local_geom`` + ``localgeom2step`` — the localised weight-set budget
  (``Q_star`` / ``Q2_star``) and per-donor lower bounds (``lb``) used by the
  in-sample conic optimisation.

The reference values are captured live from ``scpi_pkg`` (Cattaneo, Feng, Palomba
& Titiunik 2025) on the two-treated Germany panel (West Germany 1991, Italy 1992)
via ``scdataMulti`` -> ``scest`` -> ``local_geom`` / ``localgeom2step`` /
``df_EST``. Because mlsynth's clean-room ``scdata_multi`` + ``scest`` reproduce
scpi's weights value-for-value (see ``test_staggered_constraints.py``), the
residuals ``res`` and donor design ``B`` feeding these functions match too, so the
deterministic inference pieces must match scpi cell-for-cell.

The simultaneous conic bands themselves are exercised by the staggered
benchmarks; here we pin the deterministic budgets that make those bands coherent
for the non-simplex constraints.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.vanillasc_helpers.staggered_engine import (
    scdata_multi, scest, _normalize_spec, local_geom, localgeom2step, df_EST,
)

_GERMANY = pathlib.Path(__file__).resolve().parents[2] / "basedata" / "scpi_germany.csv"
pytestmark = pytest.mark.skipif(not _GERMANY.exists(), reason="Germany data absent")

_TREATED = ["Italy", "West Germany"]

# --- scpi_pkg reference (captured live; see module docstring) ---------------
_REF = {
    "simplex": {"Q_star": {"Italy": 1.0, "West Germany": 1.0}, "df": 14.0,
                "lb_all_neg_inf": False, "lb_sum": 0.1923},
    "ridge":   {"Q_star": {"Italy": 0.6975, "West Germany": 0.7278}, "df": 23.2549,
                "lb_all_neg_inf": True},
    "lasso":   {"Q_star": {"Italy": 1.0605, "West Germany": 1.0643}, "df": 13.0,
                "lb_all_neg_inf": True},
    "ols":     {"Q_star": {"Italy": None, "West Germany": None}, "df": 30.0,
                "lb_all_neg_inf": True},
}


def _md():
    d = pd.read_csv(_GERMANY)[["country", "year", "gdp"]].dropna()
    d["status"] = (((d.country == "West Germany") & (d.year >= 1991))
                   | ((d.country == "Italy") & (d.year >= 1992))).astype(int)
    feats, cadj, const, coint = _normalize_spec(
        None, None, False, False, "gdp", _TREATED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scdata_multi(d, "gdp", feats, cadj, constant=const,
                            cointegrated_data=coint, effect="unit")


@pytest.fixture(scope="module")
def md():
    return _md()


def _inference_pieces(md, name):
    """Run the generalized deterministic inference layer for one constraint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = scest(md, name)
    w = est["w"]
    res = est["res"]
    B = est["B"]
    blocks = est["blocks"]
    wcinf = est["w_constr_inf"]

    from copy import deepcopy
    from mlsynth.utils.vanillasc_helpers.staggered_engine import mat2dict

    res_d = {tr: res.loc[pd.IndexSlice[tr, :, :]] for tr in _TREATED}
    B_d = mat2dict(B, _TREATED, cols=True)
    w_d = {tr: w.loc[pd.IndexSlice[tr, :]] for tr in _TREATED}
    J = {tr: blocks[tr]["J"] for tr in _TREATED}
    KM = {tr: blocks[tr]["KM"] for tr in _TREATED}
    T0_M = {tr: sum(blocks[tr]["T0_features"].values()) for tr in _TREATED}

    rho_dict, Qpre, Q2pre, wc_upd = {}, {}, {}, {}
    wc_aux = {tr: deepcopy(wcinf[tr]) for tr in _TREATED}
    for tr in _TREATED:
        wc_upd[tr], _iw, rho_dict[tr], Qpre[tr], Q2pre[tr] = local_geom(
            wc_aux[tr], "type-2", 0.2, res_d[tr], B_d[tr],
            T0_M[tr], J[tr], KM[tr], w_d[tr])
    Q = {tr: wcinf[tr]["Q"] for tr in _TREATED}
    Q_star, lb = localgeom2step(w, rho_dict, wcinf, Q, _TREATED, J, rho_max=0.2)

    Jtot = sum(J.values())
    KMI = sum(KM.values())
    df = df_EST(wcinf[_TREATED[-1]], w, np.asarray(B, float), Jtot, KMI)
    return Q_star, lb, df


@pytest.mark.parametrize("name", ["simplex", "ridge", "lasso", "ols"])
def test_df_matches_scpi(md, name):
    _, _, df = _inference_pieces(md, name)
    assert df == pytest.approx(_REF[name]["df"], abs=1e-2)


@pytest.mark.parametrize("name", ["simplex", "ridge", "lasso"])
def test_Q_star_matches_scpi(md, name):
    Q_star, _, _ = _inference_pieces(md, name)
    for tr in _TREATED:
        assert float(Q_star[tr]) == pytest.approx(_REF[name]["Q_star"][tr], abs=1e-3)


@pytest.mark.parametrize("name", ["ridge", "lasso", "ols"])
def test_signed_constraints_have_unbounded_lb(md, name):
    _, lb, _ = _inference_pieces(md, name)
    assert all(np.isneginf(x) for x in lb)


def test_simplex_lb_pins_small_weights(md):
    _, lb, _ = _inference_pieces(md, "simplex")
    assert all(x >= 0.0 for x in lb)          # simplex lb floors at zero
    assert float(np.sum(lb)) == pytest.approx(_REF["simplex"]["lb_sum"], abs=1e-3)


# ----------------------------------------------------------------------
# family conic bands: coherence invariants. scpi_pkg's band-simulation ECOS
# is broken under numpy 2.x, so the bands cannot be cross-validated directly;
# the deterministic budgets they consume are (tests above). Here we pin that
# every constraint yields a finite, correctly-ordered band that brackets the
# fit, and that the width ordering tracks the constraint's freedom.
# ----------------------------------------------------------------------
def _bands(md, name, sims=60):
    from mlsynth.utils.vanillasc_helpers.staggered_engine import scpi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = scest(md, name)
        out = scpi(est, sims=sims, seed=8894)
    lo = np.asarray(out["sc_l_1"], float)
    hi = np.asarray(out["sc_r_1"], float)
    yfit = np.asarray(out["Y_post_fit"], float)
    return lo, hi, yfit


@pytest.mark.parametrize("name", ["simplex", "ridge", "lasso", "ols"])
def test_family_bands_are_finite_and_bracket_fit(md, name):
    lo, hi, yfit = _bands(md, name)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi >= lo)
    assert np.all(lo <= yfit + 1e-6) and np.all(hi >= yfit - 1e-6)


def test_unconstrained_band_is_wider_than_simplex(md):
    lo_s, hi_s, _ = _bands(md, "simplex")
    lo_o, hi_o, _ = _bands(md, "ols")
    # ols freely extrapolates; its in-sample compatible set (hence band) is at
    # least as wide as the simplex-constrained one at every predictand.
    assert np.mean(hi_o - lo_o) > np.mean(hi_s - lo_s)

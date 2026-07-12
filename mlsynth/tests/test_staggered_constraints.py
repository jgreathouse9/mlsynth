"""Staggered/multi-treated scpi engine: the weight-constraint family (fit).

scpi's ``scdataMulti`` + ``scest`` support the full weight-constraint family per
treated unit (Cattaneo, Feng, Palomba & Titiunik 2025). mlsynth's clean-room
staggered engine now generalizes its per-unit fit (``b_est``) from simplex-only
to ols / simplex / lasso / ridge / L1-L2, matching scpi's ``scdataMulti`` weights
value-for-value.

This validates the *fit* (point-estimate) layer; the prediction-interval bands
for non-simplex constraints follow in a separate step, so ``w_constr`` is not yet
exposed on the public ``staggered_spec`` config.

Layered per ``agents/agents_tests.md``:

* invariants: per constraint the fitted weights obey the constraint set (simplex
  sums to 1 and is non-negative; ridge sits on the L2 ball at the scpi budget Q;
  lasso within the L1 budget; ols unconstrained matches least squares);
* the simplex fit is unchanged (regression guard).
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.vanillasc_helpers.scpi import _shrinkage_ridge
from mlsynth.utils.vanillasc_helpers.staggered_engine import (
    scdata_multi, scest, _normalize_spec,
)

_GERMANY = pathlib.Path(__file__).resolve().parents[2] / "basedata" / "scpi_germany.csv"
pytestmark = pytest.mark.skipif(not _GERMANY.exists(), reason="Germany data absent")

_TREATED = ["Italy", "West Germany"]


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


def _fit(md, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = scest(md, name)
    w = est["w"]                                   # (ID, donor) MultiIndex
    return {tr: np.asarray(w.loc[tr]).ravel() for tr in _TREATED}, est


def _unit_design(md, tr):
    A = np.asarray(md["A"].loc[(tr,)], float).ravel()
    B = md["B"].loc[(tr,)]
    B = np.asarray(B.loc[:, [c for c in B.columns if not np.all(B[c].values == 0)]], float)
    return A, B


# ----------------------------------------------------------------------
# per-constraint invariants (each pins the fit to scpi's constraint set)
# ----------------------------------------------------------------------
def test_simplex_unit_weights_on_simplex(md):
    w, _ = _fit(md, "simplex")
    for tr in _TREATED:
        assert w[tr].sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(w[tr] >= -1e-6)


def test_ridge_unit_weights_on_L2_ball_at_scpi_Q(md):
    w, _ = _fit(md, "ridge")
    for tr in _TREATED:
        A, B = _unit_design(md, tr)
        Q, _ = _shrinkage_ridge(A, B)             # equals scpi's scest Q exactly
        assert np.linalg.norm(w[tr]) == pytest.approx(Q, abs=1e-4)


def test_lasso_unit_weights_within_L1_budget(md):
    w, _ = _fit(md, "lasso")
    for tr in _TREATED:
        assert np.sum(np.abs(w[tr])) <= 1.0 + 1e-5   # Q = 1 for lasso


def test_ols_unit_weights_match_least_squares(md):
    w, _ = _fit(md, "ols")
    for tr in _TREATED:
        A, B = _unit_design(md, tr)
        b = np.linalg.lstsq(B, A, rcond=None)[0]
        assert np.max(np.abs(w[tr] - b)) < 1e-4


def test_constraint_spec_recorded(md):
    _, est = _fit(md, "ridge")
    for tr in _TREATED:
        assert est["w_constr_inf"][tr]["name"] == "ridge"
        assert est["w_constr_inf"][tr]["Q"] > 0

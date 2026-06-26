"""Over-identified DR proximal estimator (Qiu et al. Kansas configuration).

These tests pin the *converged* GMM optimum on the detrended Kansas panel. The
authors' published table is under-converged (R's ``optim(BFGS)`` stops early);
mlsynth's decoupled trust-region solver reaches the genuine optimum, which R
only approaches as its tolerance tightens. See
``benchmarks/cases/dr_proximal_kansas.py`` and the replication docs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.proximal_helpers.dr import estimate_dr_overid, DROveridResult
from mlsynth.utils.proximal_helpers.dr.overid import _alpha_closed
from mlsynth.utils.proximal_helpers.bridges import augment

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "kansas_ascm.csv"
_DONORS = ["North Dakota", "South Carolina", "Texas", "Washington"]


@pytest.fixture(scope="module")
def kansas():
    """Detrended Kansas panel: residualize lngdpcapita on a quadratic time
    trend fit on the non-Kansas states (the Rmd's ``analysis_de_time_trend``)."""
    d = pd.read_csv(_DATA)
    d["t"] = ((d["year_qtr"] - 1990) * 4 + 1).round().astype(int)
    non = d[d.state != "Kansas"]
    X = np.c_[np.ones(len(non)), non.t, non.t ** 2]
    c = np.linalg.lstsq(X, non.lngdpcapita.values, rcond=None)[0]
    d["r"] = d.lngdpcapita - (c[0] + c[1] * d.t + c[2] * d.t ** 2)
    wide = d.pivot(index="t", columns="state", values="r").sort_index()
    T0 = (2012 - 1990) * 4 + 1
    return {
        "Y": wide["Kansas"].to_numpy(),
        "W": wide[_DONORS].to_numpy(),
        "Zfull": wide.drop(columns=["Kansas"] + _DONORS).to_numpy(),
        "names": [s for s in wide.columns if s not in ["Kansas"] + _DONORS],
        "T0": T0,
        "wide": wide,
    }


def _sub(k, states):
    idx = {n: i for i, n in enumerate(k["names"])}
    return k["Zfull"][:, [idx[s] for s in states]]


# --- smoke -----------------------------------------------------------------
def test_smoke_returns_result(kansas):
    r = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"],
                           _sub(kansas, ["Iowa"]), kansas["T0"], 2)
    assert isinstance(r, DROveridResult)
    assert np.isfinite(r.att) and np.isfinite(r.se)
    assert r.counterfactual.shape == kansas["Y"].shape


# --- converged optimum (the binding numbers) -------------------------------
@pytest.mark.parametrize("states,expected", [
    (["Iowa"], -0.10643),
    (["South Dakota", "Iowa"], -0.07349),
    (["South Dakota", "Iowa", "Oklahoma"], -0.07349),
])
def test_converged_att(kansas, states, expected):
    r = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"],
                           _sub(kansas, states), kansas["T0"], 2, n_starts=10)
    assert r.att == pytest.approx(expected, abs=2e-3)
    assert r.converged and r.n_basins == 1


def test_outcome_bridge_is_exact_closed_form(kansas):
    """alpha decouples to the exact LS minimiser of the linear g1 block; the
    bridge-only ATT equals the converged h value (-0.1073), not R's printed
    -0.1062 (which is under-converged)."""
    Wc = augment(kansas["W"]); GH = augment(kansas["Zfull"])
    pre = np.arange(len(kansas["Y"])) < kansas["T0"]
    a = _alpha_closed(kansas["Y"], Wc, GH, pre)
    h = Wc @ a
    phi_h = float((kansas["Y"] - h)[~pre].mean())
    assert phi_h == pytest.approx(-0.10730, abs=1e-3)


def test_dr_collapses_to_bridge_for_single_instrument(kansas):
    """With one instrument the q-correction nearly vanishes (h fits the
    pre-period), so DR[Iowa] ~ outcome bridge."""
    r = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"],
                           _sub(kansas, ["Iowa"]), kansas["T0"], 2)
    assert abs(r.att - (-0.1073)) < 0.005


# --- invariants ------------------------------------------------------------
def test_deterministic(kansas):
    z = _sub(kansas, ["South Dakota", "Iowa"])
    a = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"], z, kansas["T0"], 2, seed=0)
    b = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"], z, kansas["T0"], 2, seed=0)
    assert a.att == b.att and a.se == b.se


def test_ridge_regularizes_beta(kansas):
    """A positive ridge shrinks the (huge) flat-valley beta toward 0, which
    removes the fragile q-correction and pulls the ATT back to the outcome
    bridge (~-0.107) -- the conditioning made visible."""
    z = _sub(kansas, ["South Dakota", "Iowa"])
    r0 = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"], z, kansas["T0"], 2, ridge=0.0)
    rr = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"], z, kansas["T0"], 2, ridge=1e-3)
    assert np.linalg.norm(rr.beta[1:]) < np.linalg.norm(r0.beta[1:])
    assert abs(rr.att - (-0.107)) < abs(r0.att - (-0.107))   # ridge -> bridge


def test_se_is_finite_and_positive(kansas):
    r = estimate_dr_overid(kansas["Y"], kansas["W"], kansas["Zfull"],
                           _sub(kansas, ["Iowa"]), kansas["T0"], 2)
    assert r.se > 0 and np.isfinite(r.se)

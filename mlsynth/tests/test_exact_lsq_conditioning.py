"""The exact backend's face fits are certified optimal, not refused by a condition-number cap.

The least distance reduction in ``lsq._lsi`` used to refuse every face whose validation
block had a condition number above ``COND_MAX = 1e4``. On Abadie and Gardeazabal's
Basque specification (``school.higher`` dropped, outcome fit over 1955-1969) the
search makes 88,631 face fits, 429 of them above the cap and none above 2.62e4.
Against a 50-digit reference, KKT-verified, the reduction's answers on those faces
were right: weights within 9.1e-9, fits within 1.3e-11, losses within 1.2e-11. The
weights' error grows like ``kappa^2 eps`` and the fit's does not, as Björck's
condition number of the least squares problem predicts (Numerical Methods in Matrix
Computations, Theorem 2.2.6), so the cap refused answers the problem itself allowed.

The cap is replaced by a certificate on the face's own data (``agents_solvers.md``,
rule 2): the KKT conditions' backward error, ``lsq.kkt_backward_error``. On 4,192
captured answers it was at most 1.3e-14 at every condition number; an answer moved
by 1e-6 along a free direction scored at least 1.6e-10. ``lsq.CERTIFY = 1e-12`` sits
between them. A face whose answer misses it raises; a rank-deficient face is still
refused.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from _exact_support import FIXTURES
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.bilevel.exact import lsq

BASQUE = json.loads((FIXTURES / "conditioning_basque.json").read_text())["cases"]
EPS = np.finfo(float).eps


def _case(c):
    Gin = None if c["Gin"] is None else np.array(c["Gin"], float)
    hin = None if c["hin"] is None else np.array(c["hin"], float)
    return np.array(c["z"], float), np.array(c["Z"], float), Gin, hin


def _reduced(z, Z, Gin, hin):
    """The face's problem in the simplex's coordinates, as ``simplex_ls`` hands it to ``_lsi``."""
    b0, N = lsq._simplex_basis(Z.shape[1])
    G, h = N, -b0
    if Gin is not None and len(Gin):
        G, h = np.vstack([G, Gin @ N]), np.concatenate([h, hin - Gin @ b0])
    return Z @ N, z - Z @ b0, G, h, b0, N


def _free_direction(G, h, y, rng):
    """A unit direction along which y's active rows stay active."""
    slack = G @ y - h
    act = slack <= lsq.VERIFY * (np.linalg.norm(G, axis=1) * np.linalg.norm(y) + np.abs(h))
    Ga = G[act]
    N = np.linalg.svd(Ga, full_matrices=True)[2][len(Ga):].T if len(Ga) else np.eye(len(y))
    d = N @ rng.normal(size=N.shape[1])
    return d / np.linalg.norm(d)


# --------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("case", BASQUE, ids=lambda c: f"cond={c['cond']:.2e}")
def test_a_face_the_old_cap_refused_is_solved(case):
    loss, b = lsq.simplex_ls(*_case(case))
    assert b is not None and np.isfinite(loss)
    assert case["cond"] > 1e4


# --------------------------------------------------------------------------- unit: against the 50-digit reference
@pytest.mark.parametrize("case", BASQUE, ids=lambda c: f"cond={c['cond']:.2e}")
def test_the_answer_is_the_reference_optimum_to_its_own_condition(case):
    """Weights to 10 kappa^2 eps (the weights' condition, Björck Theorem 2.2.6); the loss to 100 kappa eps."""
    loss, b = lsq.simplex_ls(*_case(case))
    b_ref = np.array([float(v) for v in case["b_ref"]])
    kappa = case["cond"]
    assert np.linalg.norm(b - b_ref) <= 10 * kappa**2 * EPS * np.linalg.norm(b_ref)
    assert abs(loss - float(case["loss_ref"])) <= 100 * kappa * EPS * float(case["loss_ref"])


@pytest.mark.parametrize("case", BASQUE, ids=lambda c: f"cond={c['cond']:.2e}")
def test_the_certificate_accepts_the_optimum_and_rejects_a_point_off_it(case):
    z, Z, Gin, hin = _case(case)
    E, f, G, h, b0, N = _reduced(z, Z, Gin, hin)
    b_ref = np.array([float(v) for v in case["b_ref"]])
    y = N.T @ (b_ref - b0)
    assert lsq.kkt_backward_error(E, f, G, h, y) <= lsq.CERTIFY
    d = _free_direction(G, h, y, np.random.default_rng(0))
    y_off = y + 1e-6 * max(1.0, np.linalg.norm(y)) * d
    assert lsq.kkt_backward_error(E, f, G, h, y_off) > 100 * lsq.CERTIFY


def test_a_negative_multiplier_is_not_optimal():
    """min (y - 1)^2 subject to y >= 2 is at y = 2 with multiplier 1. Subject to y >= 0 the
    optimum is y = 1; at y = 0 the active row's multiplier is -1, though stationarity holds."""
    E, f = np.array([[1.0]]), np.array([1.0])
    assert lsq.kkt_backward_error(E, f, np.array([[1.0]]), np.array([2.0]), np.array([2.0])) <= lsq.CERTIFY
    assert lsq.kkt_backward_error(E, f, np.array([[1.0]]), np.array([0.0]), np.array([0.0])) >= 0.5


# --------------------------------------------------------------------------- edge
def test_a_rank_deficient_face_is_still_refused():
    """Fewer validation periods than free weights: the weights are not identified (B Theorem 2.7.1)."""
    rng = np.random.default_rng(3)
    Z, z = rng.normal(size=(2, 5)), rng.normal(size=2)
    with pytest.raises(MlsynthDataError, match="rank-deficient"):
        lsq.simplex_ls(z, Z)


def test_a_numerically_rank_deficient_face_is_refused():
    """Two donors identical to rounding: rank is decided at max(L, n) eps times the largest singular value."""
    rng = np.random.default_rng(4)
    Z, z = rng.normal(size=(8, 4)), rng.normal(size=8)
    Z[:, -1] = Z[:, 0] * (1.0 + 1e-17)
    with pytest.raises(MlsynthDataError, match="rank-deficient"):
        lsq.simplex_ls(z, Z)


# --------------------------------------------------------------------------- failure: reported, not swallowed
def test_an_uncertified_answer_raises_and_names_the_condition(monkeypatch):
    monkeypatch.setattr(lsq, "kkt_backward_error", lambda *a, **k: 1e-6)
    with pytest.raises(MlsynthEstimationError, match=r"condition number 2\.6e\+04.*1\.0e-06"):
        lsq.simplex_ls(*_case(BASQUE[0]))


def test_the_basque_specification_is_fit_end_to_end():
    """The positive definite class on the twelve-predictor Basque specification, refused before."""
    import benchmarks.cases.mscmt_solver as spec
    from mlsynth import VanillaSC

    covs = [c for c in spec._COVS if c != "school.higher"]
    win = {k: v for k, v in spec._WINDOWS.items() if k != "school.higher"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC(dict(df=spec._panel(), outcome="gdpcap", treat="treat", unitid="regionname",
                             time="year", covariates=covs, covariate_windows=win,
                             fit_window=(1955, 1969), backend="exact", exact_v_class="pd",
                             inference=False, display_graphs=False)).fit()
    w = np.array(list(res.weights.donor_weights.values()), float)
    assert np.isfinite(res.effects.att) and w.min() >= 0.0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- property
@settings(max_examples=120, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), m=st.integers(3, 6), extra=st.integers(1, 8), k=st.integers(0, 5),
       log_eta=st.floats(-9.0, 0.0))
def test_property_every_returned_face_fit_is_certified(seed, m, extra, k, log_eta):
    """Two donors 10^log_eta apart: whatever the condition, an answer comes back certified or the call raises."""
    rng = np.random.default_rng(seed)
    L = m + extra
    Z, z = rng.normal(size=(L, m)), rng.normal(size=L)
    Z[:, -1] = Z[:, 0] + 10.0 ** log_eta * rng.normal(size=L)
    Gin = rng.normal(size=(k, m)) if k else None
    hin = 0.3 * rng.normal(size=k) - 1.0 if k else None
    try:
        _, b = lsq.simplex_ls(z, Z, Gin=Gin, hin=hin)
    except (MlsynthDataError, MlsynthEstimationError):
        return
    assume(b is not None)
    E, f, G, h, b0, N = _reduced(z, Z, Gin, hin)
    assert lsq.kkt_backward_error(E, f, G, h, N.T @ (b - b0)) <= 10 * lsq.CERTIFY


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), m=st.integers(2, 6), extra=st.integers(1, 8),
       log_c=st.floats(-8.0, 8.0))
def test_property_the_certificate_does_not_depend_on_the_outcomes_units(seed, m, extra, log_c):
    rng = np.random.default_rng(seed)
    Z, z = rng.normal(size=(m + extra, m)), rng.normal(size=m + extra)
    E, f, G, h, b0, N = _reduced(z, Z, None, None)
    _, b = lsq.simplex_ls(z, Z)
    y = N.T @ (b - b0)
    c = 10.0 ** log_c
    a, s = lsq.kkt_backward_error(E, f, G, h, y), lsq.kkt_backward_error(c * E, c * f, G, h, y)
    assert abs(a - s) <= 64 * EPS

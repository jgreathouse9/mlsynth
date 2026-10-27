"""VanillaSC inference refits a ``w_constr`` fit with its own constrained program (#919).

A ``w_constr`` fit solves scpi's constrained weight program, and it used to leave the
pipeline with no engine, so every inference mode that refits the estimator failed:
``ttest`` and ``conformal_cumulative`` raised an internal ``AttributeError``, and the
placebo-type tests were reported as not computed. The constrained program now has an
engine, and every refit is the configured family. The batched outcome-only placebo
path solves the simplex QP, so it is kept off these fits. ``conformal`` refits the
simplex program under each null, so it is offered for ``w_constr="simplex"`` and
refused for the other families.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import mlsynth.utils.solvers.minnorm as minnorm
import mlsynth.utils.vanillasc_helpers.pipeline as pipeline
from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError

T, T0 = 24, 18


def _panel(n_donors=6, seed=0, effect=-4.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_donors + 1):
        level, slope = 20.0 + 2.0 * u + rng.normal(), 0.5 + 0.1 * rng.normal()
        for t in range(T):
            treated = int(u == 0 and t >= T0)
            rows.append({"unit": f"u{u}", "time": t, "treat": treated,
                         "y": level + slope * t + rng.normal(scale=0.5) + effect * treated})
    return pd.DataFrame(rows)


def _fit(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time", display_graphs=False)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


class _Spy:
    """Records the family of every constrained fit the pipeline makes."""

    def __init__(self, monkeypatch):
        self.families = []
        real = pipeline._ConstrainedFit.__init__

        def init(fit, y_pre, Y0_pre, donor_names, w_constr):
            self.families.append(w_constr if isinstance(w_constr, str) else w_constr.get("name"))
            real(fit, y_pre, Y0_pre, donor_names, w_constr)

        monkeypatch.setattr(pipeline._ConstrainedFit, "__init__", init)


# --------------------------------------------------------------------------- smoke
@pytest.mark.parametrize("mode", ["placebo", "lto", "placebo_cs", "ttest", "conformal_cumulative"])
def test_a_ridge_family_fit_gets_its_inference(mode):
    res = _fit(_panel(), w_constr="ridge", inference=mode, alpha=0.2, placebo_cs_precision=8,
               conformal_horizon=2)
    inf = res.inference
    assert "requested but not computed" not in inf.method
    if mode in ("placebo", "lto", "ttest"):
        assert 0.0 <= inf.p_value <= 1.0
    elif mode == "conformal_cumulative":
        assert inf.details["cumulative_lower"] <= inf.details["cumulative_upper"]
    else:                                           # bounds, or the inversion's own reason for none
        assert inf.method.startswith("placebo-inverted confidence set")
        assert "refit" not in inf.details.get("unavailable_reason", "")


# --------------------------------------------------------------------------- unit: which program each refit solves
@pytest.mark.parametrize("family", ["ridge", "lasso", "simplex"])
def test_every_placebo_refit_is_the_configured_family(family, monkeypatch):
    spy = _Spy(monkeypatch)
    batched = []
    monkeypatch.setattr(minnorm, "solve_simplex_loo_exact", lambda *a, **k: batched.append(1) or None)
    res = _fit(_panel(), w_constr=family, inference="placebo")
    assert spy.families == [family] * 7            # the treated unit's fit and one refit per donor
    assert res.inference.details["n_placebos"] == 6
    assert not batched                              # the simplex family path is not taken


def test_each_placebo_refit_is_the_constrained_program_on_that_donor():
    """The refit's ratio is the one the constrained program gives the donor, computed directly."""
    df = _panel(seed=3)
    res = _fit(df, w_constr="ridge", inference="placebo")
    Y = df.pivot(index="time", columns="unit", values="y")[[f"u{u}" for u in range(7)]].to_numpy()
    ratios = []
    for u in range(1, 7):
        pool = [v for v in range(1, 7) if v != u]
        fit = pipeline._ConstrainedFit(Y[:T0, u], Y[:T0][:, pool], [str(v) for v in pool], "ridge")
        g = Y[:, u] - Y[:, pool] @ fit.W
        ratios.append(np.sqrt(np.mean(g[T0:] ** 2)) / np.sqrt(np.mean(g[:T0] ** 2)))
    r_tr = res.inference.details["treated_rmspe_ratio"]
    assert res.inference.details["rank"] == 1 + sum(r >= r_tr for r in ratios)


def test_the_ttest_cross_fit_refits_the_family(monkeypatch):
    spy = _Spy(monkeypatch)
    _fit(_panel(), w_constr="lasso", inference="ttest", ttest_K=3)
    assert len(spy.families) == 1 + 3 and set(spy.families) == {"lasso"}


def test_conformal_with_the_simplex_family_is_the_simplex_conformal_test():
    df = _panel(seed=5)
    a = _fit(df, w_constr="simplex", inference="conformal", conformal_n_perm=50)
    b = _fit(df, inference="conformal", conformal_n_perm=50)
    assert a.inference.p_value == pytest.approx(b.inference.p_value)


# --------------------------------------------------------------------------- failure
@pytest.mark.parametrize("family", ["ridge", "lasso", "ols", "L1-L2", {"name": "ridge"}])
def test_conformal_refuses_a_family_its_null_refit_cannot_solve(family):
    with pytest.raises(MlsynthConfigError, match="conformal.*simplex"):
        VanillaSC(dict(df=_panel(), outcome="y", treat="treat", unitid="unit", time="time",
                       display_graphs=False, w_constr=family, inference="conformal"))


# --------------------------------------------------------------------------- property
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10_000), n_donors=st.integers(3, 6))
def test_property_the_simplex_family_placebo_is_the_simplex_placebo(seed, n_donors):
    """scpi's simplex family is the simplex QP, so its placebo test is the outcome-only one."""
    df = _panel(n_donors=n_donors, seed=seed)
    a = _fit(df, w_constr="simplex", inference="placebo")
    b = _fit(df, inference="placebo")
    ra, rb = a.inference.details["treated_rmspe_ratio"], b.inference.details["treated_rmspe_ratio"]
    assume(abs(ra - rb) < 1e-5 * rb)                # the two solvers agree on the treated fit
    assert a.inference.details["n_placebos"] == b.inference.details["n_placebos"] == n_donors
    assert a.inference.p_value == b.inference.p_value

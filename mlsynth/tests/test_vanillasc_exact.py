"""VanillaSC's ``backend="exact"``: configuration, the fit, and the outcome-fit exclusions.

The backend is ``mlsynth.utils.bilevel.exact.solve_exact`` behind the bilevel engine; its
search and predictor weights are tested in ``test_exact_*.py`` and its agreement with the
authors' run in ``test_vanillasc_exact_replication.py``. Here: the configuration refuses
what it cannot honour, ``VanillaSC.fit()`` hands the backend the matrices it would build
by hand and returns its typed details, ``fit_exclude`` drops periods from the outcome fit
for every backend, and the refits of inference run without the options only the main
fit reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _exact_support import exterior_design, interior_design
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.bilevel import BilevelProblem
from mlsynth.utils.bilevel.exact import ExactDetails, solve_exact
from mlsynth.utils.solvers.ridge_augment import simplex_qp
from mlsynth.utils.vanillasc_helpers.structures import VanillaSCResults


def _panel(X1, X0, Z1, Z0, post=3, seed=0):
    """A long panel whose covariates are constant in time, so their window means are X; the
    pre-period outcomes are Z and the treated unit is u0, treated after period L."""
    rng = np.random.default_rng(seed)
    K, J = X0.shape
    L = len(Z1)
    X = np.column_stack([X1, X0])
    Y = np.vstack([np.column_stack([Z1, Z0]), rng.normal(size=(post, J + 1)) + np.r_[Z1[-1], Z0[-1]]])
    rows = []
    for i in range(J + 1):
        for t in range(L + post):
            rows.append({"unit": f"u{i}", "time": t, "y": Y[t, i], "treat": int(i == 0 and t >= L),
                         **{f"x{k}": X[k, i] for k in range(K)}})
    return pd.DataFrame(rows)


def _cfg(df, K, **kw):
    return {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
            "backend": "exact", "covariates": [f"x{k}" for k in range(K)], "inference": False,
            "display_graphs": False, **kw}


def _standardised(X1, X0):
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    return X1 / sd, X0 / sd[:, None]


@pytest.fixture(scope="module")
def design():
    return exterior_design(1010, 4, 12)


# --------------------------------------------------------------------------- smoke
def test_an_exact_fit_returns_vanillasc_results_with_typed_details(design):
    X1, X0, Z1, Z0 = design
    res = VanillaSC(_cfg(_panel(*design), 4)).fit()
    assert isinstance(res, VanillaSCResults) and isinstance(res.exact, ExactDetails)
    assert res.exact.v_class == "diagonal" and res.exact.regime == "exterior"
    assert res.method_details.method_name == "VanillaSC[exact]"
    assert res.method_details.parameters_used["exact_v_class"] == "diagonal"
    w = np.array(list(res.weights.donor_weights.values()))
    assert w.min() > 0.0 and abs(w.sum() - 1.0) <= 1e-12
    v = res.weights.summary_stats["predictor_weights"]
    assert list(v) == ["x0", "x1", "x2", "x3"] and abs(sum(v.values()) - 1.0) <= 1e-9


# --------------------------------------------------------------------------- unit: the pipeline
def _weights(res, J):
    d = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    return np.array([d.get(f"u{j + 1}", 0.0) for j in range(J)])


@pytest.mark.parametrize("v_class", ["diagonal", "pd"])
def test_the_fit_is_the_backend_s_answer_on_the_panel_s_matrices(design, v_class):
    X1, X0, Z1, Z0 = design
    res = VanillaSC(_cfg(_panel(*design), 4, exact_v_class=v_class)).fit()
    S1, S0 = _standardised(X1, X0)
    sol = solve_exact(BilevelProblem(y1_pre=Z1, Y0_pre=Z0, X1=S1, X0=S0), v_class=v_class)
    assert np.abs(_weights(res, 12) - sol.W).max() <= 1e-10
    assert np.isclose(res.exact.validation_loss, sol.upper_loss, rtol=1e-12)
    np.testing.assert_allclose(list(res.weights.summary_stats["predictor_weights"].values()), sol.V, atol=1e-9)
    gap = res.time_series.observed_outcome - res.time_series.counterfactual_outcome
    assert np.isclose(res.effects.att, np.mean(np.asarray(gap)[len(Z1):]), rtol=1e-12)


def test_options_reach_the_backend(design):
    res = VanillaSC(_cfg(_panel(*design), 4, exact_ranges=True, exact_face_counts=True)).fit()
    p, fc = res.exact.pd, res.exact.face_counts
    assert p.range_lo is not None and np.all(p.range_lo <= p.logdet_shares + 1e-9)
    assert np.all(p.logdet_shares <= p.range_hi + 1e-9)
    assert fc is not None and len(fc.total) == 4 and sum(fc.attainable) >= 1


def test_an_interior_treated_unit_is_fit_by_the_interior_search():
    inst = interior_design(4020, 3, 12)
    res = VanillaSC(_cfg(_panel(*inst), 3)).fit()
    assert res.exact.regime == "interior" and res.exact.face_counts is None
    w = _weights(res, 12)
    assert np.count_nonzero(w) == 4 and np.allclose(inst[1] @ w, inst[0], atol=1e-9)


def test_other_backends_carry_no_exact_details(design):
    res = VanillaSC(_cfg(_panel(*design), 4, backend="malo")).fit()
    assert isinstance(res, VanillaSCResults) and res.exact is None


# --------------------------------------------------------------------------- unit: fit_exclude
def test_excluded_periods_leave_the_outcome_fit(design):
    """Dropping periods 2 and 5 from validation is the backend run on the other rows."""
    X1, X0, Z1, Z0 = design
    res = VanillaSC(_cfg(_panel(*design), 4, fit_exclude=[2, 5])).fit()
    keep = [t for t in range(len(Z1)) if t not in (2, 5)]
    S1, S0 = _standardised(X1, X0)
    sol = solve_exact(BilevelProblem(y1_pre=Z1[keep], Y0_pre=Z0[keep], X1=S1, X0=S0))
    assert np.abs(_weights(res, 12) - sol.W).max() <= 1e-10
    assert res.method_details.parameters_used["fit_exclude"] == [2, 5]


def test_exclusions_apply_to_every_backend(design):
    X1, X0, Z1, Z0 = design
    df = _panel(*design)
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
                     "inference": False, "display_graphs": False, "fit_exclude": [0, 1, 2]}).fit()
    w = simplex_qp(Z0[3:], Z1[3:])
    assert np.abs(_weights(res, 12) - w).max() <= 1e-8


def test_exclusions_compose_with_the_fit_window(design):
    X1, X0, Z1, Z0 = design
    res = VanillaSC(_cfg(_panel(*design), 4, fit_window=(3, 11), fit_exclude=[7])).fit()
    keep = [t for t in range(3, 12) if t != 7]
    S1, S0 = _standardised(X1, X0)
    sol = solve_exact(BilevelProblem(y1_pre=Z1[keep], Y0_pre=Z0[keep], X1=S1, X0=S0))
    assert np.abs(_weights(res, 12) - sol.W).max() <= 1e-10


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_property_an_empty_exclusion_changes_nothing(seed):
    inst = exterior_design(seed, 3, 9)
    df = _panel(*inst, seed=seed)
    a = VanillaSC(_cfg(df, 3)).fit()
    b = VanillaSC(_cfg(df, 3, fit_exclude=[])).fit()
    assert a.weights.donor_weights == b.weights.donor_weights


# --------------------------------------------------------------------------- inference refits
def test_placebo_refits_skip_the_main_fit_s_extras(design, monkeypatch):
    """Ranges and face counts are reported for the treated unit; each placebo refit needs only weights."""
    import mlsynth.utils.bilevel.solver as solver
    calls = []
    real = solver.solve_exact

    def spy(prob, **kw):
        calls.append(kw)
        return real(prob, **kw)
    monkeypatch.setattr(solver, "solve_exact", spy)
    res = VanillaSC(_cfg(_panel(*design), 4, exact_ranges=True, exact_face_counts=True,
                         inference="placebo")).fit()
    assert 0.0 < res.inference.p_value <= 1.0
    assert calls[0]["ranges"] and calls[0]["face_counts"]
    assert len(calls) > 1 and not any(c["ranges"] or c["face_counts"] for c in calls[1:])


# --------------------------------------------------------------------------- failure
def _base(design, **kw):
    return _cfg(_panel(*design), 4, **kw)


@pytest.mark.parametrize("kw,match", [
    ({"covariates": None}, "covariates"),
    ({"exact_v_class": "full"}, "exact_v_class"),
    ({"backend": "mscmt", "exact_v_class": "pd"}, "backend='exact'"),
    ({"backend": "auto", "exact_ranges": True}, "backend='exact'"),
    ({"backend": "malo", "exact_face_counts": True}, "backend='exact'"),
    ({"augment": "ridge"}, "augment"),
])
def test_configurations_the_backend_cannot_honour_are_refused(design, kw, match):
    with pytest.raises(MlsynthConfigError, match=match):
        VanillaSC(_base(design, **kw))


@pytest.mark.parametrize("exclude,match", [
    ([99], "not a pre-treatment period"),
    (list(range(12)), "no pre-treatment periods"),
])
def test_exclusions_that_leave_nothing_or_miss_the_pre_period_are_refused(design, exclude, match):
    with pytest.raises(MlsynthDataError, match=match):
        VanillaSC(_base(design, fit_exclude=exclude)).fit()


def test_the_engine_refuses_the_exact_backend_without_predictors(design):
    """Called directly, the engine must not fall back to outcome-only matching when the exact
    backend is asked for and no predictors are given."""
    from mlsynth.utils.bilevel import BilevelSCM
    _, _, Z1, Z0 = design
    with pytest.raises(ValueError, match="needs covariates"):
        BilevelSCM("exact").fit(Z1, Z0)


def test_a_backend_failure_is_reported_not_swallowed(design):
    """A design the search refuses (predictors that span fewer than K dimensions) surfaces as the
    backend's error, not as some other backend's answer."""
    X1, X0, Z1, Z0 = design
    X0 = X0.copy()
    X0[3] = X0[2]
    X1 = X1.copy()
    X1[3] = X1[2]
    with pytest.raises(MlsynthDataError, match="fewer than K dimensions"):
        VanillaSC(_cfg(_panel(X1, X0, Z1, Z0), 4)).fit()


# --------------------------------------------------------------------------- properties
@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 4), extra=st.integers(4, 8), interior=st.booleans(),
       v_class=st.sampled_from(["diagonal", "pd"]))
def test_property_the_fit_is_the_backend_s_answer(seed, K, extra, interior, v_class):
    """Over designs on both sides of the hull and both classes, VanillaSC returns what the backend
    returns on the matrices it would build by hand, and the ATT is the mean post-period gap."""
    if K == 2 and not interior:
        K = 3                                               # the seeded exterior designs need K >= 3
    X1, X0, Z1, Z0 = (interior_design if interior else exterior_design)(seed, K, K + extra)
    res = VanillaSC(_cfg(_panel(X1, X0, Z1, Z0, seed=seed), K, exact_v_class=v_class)).fit()
    S1, S0 = _standardised(X1, X0)
    sol = solve_exact(BilevelProblem(y1_pre=Z1, Y0_pre=Z0, X1=S1, X0=S0), v_class=v_class)
    assert np.abs(_weights(res, X0.shape[1]) - sol.W).max() <= 1e-10
    assert res.exact.regime == ("interior" if interior else "exterior")
    gap = np.asarray(res.time_series.observed_outcome) - np.asarray(res.time_series.counterfactual_outcome)
    assert np.isclose(res.effects.att, gap[len(Z1):].mean(), rtol=1e-12, atol=1e-12)


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000), drop=st.sets(st.integers(0, 11), max_size=7))
def test_property_exclusions_are_the_backend_on_the_kept_rows(seed, drop):
    X1, X0, Z1, Z0 = exterior_design(seed, 3, 9)
    res = VanillaSC(_cfg(_panel(X1, X0, Z1, Z0, seed=seed), 3, fit_exclude=sorted(drop))).fit()
    keep = [t for t in range(len(Z1)) if t not in drop]
    S1, S0 = _standardised(X1, X0)
    sol = solve_exact(BilevelProblem(y1_pre=Z1[keep], Y0_pre=Z0[keep], X1=S1, X0=S0))
    assert np.abs(_weights(res, 9) - sol.W).max() <= 1e-10


@settings(max_examples=20, deadline=None)
@given(backend=st.sampled_from(["auto", "outcome-only", "malo", "mscmt", "regression", "penalized"]),
       option=st.sampled_from([("exact_v_class", "pd"), ("exact_ranges", True), ("exact_face_counts", True)]))
def test_property_exact_options_need_the_exact_backend(design, backend, option):
    """Any exact option set away from its default beside any other backend is refused; set to its
    default, it is accepted."""
    name, value = option
    with pytest.raises(MlsynthConfigError, match="backend='exact'"):
        VanillaSC(_base(design, backend=backend, **{name: value}))
    default = {"exact_v_class": "diagonal", "exact_ranges": False, "exact_face_counts": False}[name]
    VanillaSC(_base(design, backend=backend, **{name: default}))

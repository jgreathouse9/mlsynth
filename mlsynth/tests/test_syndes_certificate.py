"""The SYNDES optimality certificate reads the exact searches' bounds (#822).

Contract: the lower bound is the bound the mode's exact search reports, so it is
always valid (``lower_bound <= optimum <= incumbent``). Where the search closes,
one-way's and per-unit's bound is the optimum itself and the gap is zero to the
searches' tolerance; two-way's is the Rayleigh bound over every size-``K`` design.
``certified`` says the search closed. ``certify=True`` on the estimator attaches
it, reusing the estimator's own search where it ran one on the whole pre-period;
the default leaves the result untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.certificate import (
    SYNDESCertificate,
    design_objective,
    syndes_certificate,
)
from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
from mlsynth.utils.syndes_helpers.one_way_exact import solve_one_way_exact
from mlsynth.utils.syndes_helpers.per_unit_exact import solve_per_unit_exact
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions

MODES = ("global_equal_weights", "global_2way", "per_unit")
SOLVE = {"global_equal_weights": solve_one_way_exact,
         "global_2way": solve_synthetic_design_exact,
         "per_unit": solve_per_unit_exact}


def _Y(N=10, T=8, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3)); Lam = rng.standard_normal((N, 3))
    return (Lam @ F.T).T + 0.3 * rng.standard_normal((T, N))


def _incumbent(Y, K, mode, **kw):
    d = SOLVE[mode](Y, K, **kw)
    return d, design_objective(Y, mode, d, d.lambda_value)


# ----------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_is_certified_by_its_search(mode):
    Y = _Y()
    d, inc = _incumbent(Y, 3, mode)
    c = syndes_certificate(Y, 3, mode, inc, lam=d.lambda_value)
    assert isinstance(c, SYNDESCertificate)
    assert c.method == "search" and c.certified is True and c.note == ""
    assert c.lower_bound <= inc * (1 + 1e-8)
    assert 0.0 <= c.optimality_gap < 1.0


# ----------------------------------------------------------------------
# the bound is the search's
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["global_equal_weights", "per_unit"])
def test_where_the_search_closes_the_gap_is_zero(mode):
    """One-way's and per-unit's searches report the optimum as their bound."""
    Y = _Y(seed=1)
    d, inc = _incumbent(Y, 3, mode)
    c = syndes_certificate(Y, 3, mode, inc, lam=d.lambda_value)
    assert c.lower_bound == pytest.approx(d.solve_outcome.dual_bound, rel=1e-12)
    assert c.optimality_gap <= 1e-8


def test_the_two_way_bound_is_the_searchs_rayleigh_bound():
    Y = _Y(seed=2)
    d, inc = _incumbent(Y, 3, "global_2way")
    c = syndes_certificate(Y, 3, "global_2way", inc, lam=d.lambda_value)
    assert c.lower_bound == pytest.approx(d.raw_results["lower_bound"], rel=1e-12)
    assert 0.8 * inc <= c.lower_bound <= inc


@pytest.mark.parametrize("mode", MODES)
def test_a_passed_search_is_read_not_repeated(mode, monkeypatch):
    import mlsynth.utils.syndes_helpers.certificate as cert
    Y = _Y(seed=3)
    d, inc = _incumbent(Y, 3, mode)
    fresh = syndes_certificate(Y, 3, mode, inc, lam=d.lambda_value)

    def refuse(*a, **k):
        raise AssertionError("the certificate searched again")
    monkeypatch.setattr(cert, "_search", refuse)
    reused = syndes_certificate(Y, 3, mode, inc, lam=d.lambda_value, search=d)
    assert reused.lower_bound == pytest.approx(fresh.lower_bound, rel=1e-12)
    assert reused.certified == fresh.certified


def test_restrictions_give_the_restricted_problems_bound():
    """Forbidding the unrestricted optimum's units raises the optimum; the
    restricted bound sees it, and the unrestricted one bounds it too."""
    Y = _Y(seed=4)
    free, _ = _incumbent(Y, 3, "per_unit")
    rules = DesignRestrictions(forbidden=list(free.selected_unit_indices))
    d, inc = _incumbent(Y, 3, "per_unit", restrictions=rules)
    tight = syndes_certificate(Y, 3, "per_unit", inc, lam=d.lambda_value,
                               restrictions=rules)
    loose = syndes_certificate(Y, 3, "per_unit", inc, lam=d.lambda_value)
    assert tight.lower_bound == pytest.approx(inc, rel=1e-9)
    assert loose.lower_bound < tight.lower_bound
    assert loose.lower_bound == pytest.approx(free.objective_value, rel=1e-9)


def test_costs_give_the_budgeted_problems_bound():
    Y = _Y(seed=5)
    costs = np.arange(1.0, 11.0)
    d, inc = _incumbent(Y, 3, "global_equal_weights", costs=costs, budget=8.0)
    c = syndes_certificate(Y, 3, "global_equal_weights", inc, lam=d.lambda_value,
                           costs=costs, budget=8.0)
    assert c.lower_bound == pytest.approx(inc, rel=1e-9)


# ----------------------------------------------------------------------
# failures
# ----------------------------------------------------------------------


def test_unknown_mode_raises():
    with pytest.raises(MlsynthConfigError):
        syndes_certificate(_Y(), 3, "not_a_mode", 1.0)


def test_a_design_without_a_search_bound_is_refused():
    class Bare:
        raw_results = {"solver": "SCIP"}
        solve_outcome = None
    with pytest.raises(MlsynthConfigError, match="exact backends"):
        syndes_certificate(_Y(), 3, "global_2way", 1.0, search=Bare())


# ----------------------------------------------------------------------
# the estimator
# ----------------------------------------------------------------------


def _panel(N=10, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, N)) * 0.4 + rng.standard_normal(N)
    return pd.DataFrame([
        {"unit": f"u{i}", "time": t, "y": float(Y[t, i]), "post": int(t >= T - n_post)}
        for i in range(N) for t in range(T)])


def _fit(mode, **over):
    cfg = {"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
           "K": 3, "mode": mode, "post_col": "post", "run_inference": False}
    cfg.update(over)
    return SYNDES(cfg).fit()


@pytest.mark.parametrize("mode", ["two_way_global", "one_way_global", "per_unit"])
def test_certify_attaches_a_certificate_from_the_search(mode):
    res = _fit(mode, certify=True)
    c = res.certificate
    assert c is not None and c.method == "search" and c.certified is True
    assert c.lower_bound <= float(res.design.objective_value) * (1 + 1e-8)


def test_default_off_no_certificate():
    assert _fit("two_way_global").certificate is None


@pytest.mark.parametrize("over", [{}, {"top_K": 3, "selection": "in_sample"},
                                  {"top_K": 3, "selection": "ic"}])
def test_the_estimator_reuses_its_whole_pre_period_search(over, monkeypatch):
    import mlsynth.utils.syndes_helpers.certificate as cert

    def refuse(*a, **k):
        raise AssertionError("the certificate searched again")
    monkeypatch.setattr(cert, "_search", refuse)
    c = _fit("per_unit", certify=True, **over).certificate
    assert c.certified is True


def test_under_holdout_the_certificate_searches_the_whole_pre_period(monkeypatch):
    import mlsynth.utils.syndes_helpers.certificate as cert
    calls = []
    original = cert._search

    def spy(Y, *a, **k):
        calls.append(Y.shape)
        return original(Y, *a, **k)
    monkeypatch.setattr(cert, "_search", spy)
    res = _fit("per_unit", certify=True, top_K=3, selection="holdout",
               holdout_frac=0.3)
    assert calls == [res.inputs.Y_pre.shape]
    assert res.certificate.lower_bound <= res.certificate.incumbent * (1 + 1e-8)


def test_the_estimators_restrictions_reach_the_certificate():
    res = _fit("one_way_global", certify=True, not_to_be_treated=["u0", "u1", "u2"])
    c = res.certificate
    assert c.lower_bound == pytest.approx(c.incumbent, rel=1e-9)


def test_under_holdout_the_estimators_restrictions_reach_the_certificate():
    """Under holdout the certificate searches the whole pre-period itself, so it
    needs the design's restrictions: banning the unrestricted optimum's units
    raises the bound to the restricted optimum."""
    holdout = dict(certify=True, top_K=3, selection="holdout", holdout_frac=0.3)
    free = _fit("per_unit", **holdout)
    Y_pre, lam = free.inputs.Y_pre, free.design.lambda_value
    best = solve_per_unit_exact(Y_pre, 3, lam=lam)
    idx = [int(i) for i in best.selected_unit_indices]
    banned = [free.inputs.unit_index.labels[i] for i in idx]
    res = _fit("per_unit", not_to_be_treated=banned, **holdout)
    restricted = solve_per_unit_exact(
        Y_pre, 3, lam=res.design.lambda_value,
        restrictions=DesignRestrictions(forbidden=idx))
    assert res.certificate.lower_bound == pytest.approx(
        restricted.objective_value, rel=1e-9)
    assert res.certificate.lower_bound > best.objective_value * (1 + 1e-6)

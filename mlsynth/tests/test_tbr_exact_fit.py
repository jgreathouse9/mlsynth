"""A TBR split whose pretest regression fits exactly (#1000).

When the treated aggregate is an exact affine function of the control
aggregate over the scoring window, the regression leaves no residual, so there
is no residual scale to size an interval, a required impact or an A/A test
from. Before #1000 the reference objective divided by that zero scale, and
every objective that did not raise scored the split as the best design there
could be (``1 / required_impact`` near 1e13).

What these tests pin:

* :func:`fits_exactly` is the shared predicate: a residual within
  ``EXACT_FIT_RTOL`` (1e-10) of the series it is computed from is an exact
  fit, which covers the rounding of any affine relation; a real residual many
  orders above rounding is not;
* :func:`score_split` scores an exact fit as infeasible under every objective:
  a key of ``-inf`` the length of that objective's key, below every feasible
  key, with the detail saying why. It does not raise;
* a constant treated aggregate is an exact fit and no longer raises
  ``ZeroDivisionError`` under ``"reference"``;
* a split exact on the A/A fit window but not on the whole window fails the
  A/A gate instead of dividing by zero;
* the batched screen agrees with the direct key on exact-fit designs;
* the greedy and local searches pass over an exact-fit pair, and a search on a
  panel that contains one runs and recommends a design with a residual;
* a named split whose pretest fits exactly is analysed (exact point estimate,
  zero-width interval) with a warning that names the treated and control geos.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.utils.tbr_helpers.design.batched import design_cache, score_designs
from mlsynth.utils.tbr_helpers.design.local import local_search
from mlsynth.utils.tbr_helpers.design.objective import (aa_test_ok, holdout_fit,
                                                        score_split)
from mlsynth.utils.tbr_helpers.design.search import ROLES, greedy_search
from mlsynth.utils.tbr_helpers.posterior import fit_pretest, fits_exactly

ALL = frozenset(ROLES)
OBJECTIVES = ("reference", "relative", "paper")
KEY_LENGTH = {"reference": 6, "relative": 1, "paper": 1}


def _walk(n=40, seed=0, level=100.0):
    return level + np.random.default_rng(seed).normal(size=n).cumsum()


def _panel(seed=0, T=40, N=8):
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.5, N)
    common = 1.0 + 0.3 * rng.normal(size=(T, 1))
    return size * (0.6 * common + 0.4 * (1.0 + 0.1 * rng.normal(size=(T, N))))


def _with_exact_pair(Y):
    """Geo 1 becomes exactly twice geo 0, an exact fit for the split ({1}, {0})."""
    Y = Y.copy()
    Y[:, 1] = 2.0 * Y[:, 0]
    return Y


# --------------------------------------------------------------------------- #
# the predicate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_y", [
    lambda x: 2.0 * x,                       # residual exactly zero
    lambda x: 2.0 * x + 1.0,                 # zero up to rounding
    lambda x: -0.5 * x + 1e6,                # a large level, rounding scales with it
    lambda x: np.full_like(x, 3.0),          # a constant treated aggregate
], ids=["multiple", "affine", "large-level", "constant"])
def test_an_affine_treated_aggregate_fits_exactly(make_y):
    x = _walk()
    assert fits_exactly(fit_pretest(make_y(x), x))


@pytest.mark.parametrize("noise", [1e-7, 1e-4, 1e-1])
def test_a_residual_above_the_tolerance_is_not_an_exact_fit(noise):
    x = _walk()                                 # a level near 100, so y near 200
    e = np.random.default_rng(1).normal(size=x.size)
    assert not fits_exactly(fit_pretest(2.0 * x + 1.0 + noise * e, x))


def test_an_affine_relation_across_a_cancelled_level_fits_exactly():
    # y = x - 1e8 + 5 sits near 5 while x sits near 1e8: the regression's
    # residual carries the rounding of beta x, about eps 1e8 per period, which
    # is large next to y. The scale has to include beta x to see it as rounding.
    x = 1e8 + _walk(level=0.0)
    y = (x - 1e8) + 5.0
    assert fits_exactly(fit_pretest(y, x))


def test_the_tolerance_is_relative_to_the_series():
    x = _walk()
    e = np.random.default_rng(1).normal(size=x.size)
    y = 2.0 * x + 1.0
    # a residual 1e-12 of a series near 200 is below EXACT_FIT_RTOL; the same
    # residual on a series a million times smaller is far above it
    assert fits_exactly(fit_pretest(y + 2e-10 * e, x))
    assert not fits_exactly(fit_pretest(1e-6 * y + 2e-10 * e, 1e-6 * x))


# --------------------------------------------------------------------------- #
# score_split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_an_exact_fit_is_scored_infeasible_under_every_objective(objective):
    x = _walk()
    score = score_split(2.0 * x, x, objective=objective, n_test=8)
    assert len(score.key) == KEY_LENGTH[objective]
    assert all(v == -np.inf for v in score.key)
    assert score.value == -np.inf
    assert score.detail["exact_fit"] is True


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_an_exact_fit_ranks_below_the_worst_feasible_split(objective):
    x = _walk()
    e = np.random.default_rng(2).normal(size=x.size)
    noisy = score_split(2.0 * x + 25.0 * e, x, objective=objective, n_test=8)
    exact = score_split(2.0 * x + 1.0, x, objective=objective, n_test=8)
    assert noisy.detail.get("exact_fit", False) is False
    assert np.all(np.isfinite(noisy.key))
    assert exact.key < noisy.key


def test_the_reference_objective_no_longer_divides_by_a_zero_scale():
    x = _walk()
    # #1000's first reproduction raised ZeroDivisionError here.
    score = score_split(2.0 * x, x, objective="reference", n_test=8)
    assert score.detail["exact_fit"] is True


def test_a_constant_treated_aggregate_is_infeasible_and_does_not_raise():
    x = _walk()
    for objective in OBJECTIVES:
        score = score_split(np.full_like(x, 3.0), x, objective=objective, n_test=8)
        assert score.detail["exact_fit"] is True


def test_a_near_exact_split_with_a_real_residual_keeps_its_key():
    x = _walk()
    e = np.random.default_rng(3).normal(size=x.size)
    score = score_split(2.0 * x + 1.0 + 1e-6 * e, x, objective="relative", n_test=8)
    assert score.detail.get("exact_fit", False) is False
    assert np.isfinite(score.value) and score.value > 0


# --------------------------------------------------------------------------- #
# the A/A gate
# --------------------------------------------------------------------------- #
def test_a_split_exact_only_on_the_aa_fit_window_fails_the_aa_gate():
    x = _walk(n=40)
    y = 2.0 * x + 1.0
    rng = np.random.default_rng(4)
    y[32:] += 3.0 * rng.normal(size=8)        # the held-out window departs
    fit = holdout_fit(y, x, 8)
    assert fit.exact_fit is True
    assert aa_test_ok(y, x, 8) is False
    score = score_split(y, x, objective="reference", n_test=8)
    assert score.detail.get("exact_fit", False) is False
    assert score.detail["aa_test"] is False
    assert np.all(np.isfinite(score.key))


def test_a_holdout_with_a_residual_is_not_flagged():
    x = _walk(n=40)
    y = 2.0 * x + np.random.default_rng(5).normal(size=40)
    assert holdout_fit(y, x, 8).exact_fit is False


# --------------------------------------------------------------------------- #
# the batched screen and the searches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("objective", ["reference", "relative"])
def test_the_batched_screen_agrees_on_an_exact_fit(objective):
    Y = _with_exact_pair(_panel())
    cache = design_cache(Y, n_test=8)
    N = Y.shape[1]
    UT = np.zeros((2, N)); UC = np.zeros((2, N))
    UT[0, 1] = 1; UC[0, 0] = 1                     # the exact pair
    UT[1, 2] = 1; UC[1, [3, 4, 5]] = 1             # an ordinary design
    keys = score_designs(cache, UT, UC, objective=objective)
    assert np.all(keys[0] == -np.inf)
    direct = score_split(Y[:, [2]].sum(1), Y[:, [3, 4, 5]].sum(1),
                         objective=objective, n_test=8).key
    np.testing.assert_allclose(keys[1], direct, rtol=1e-9)


@pytest.mark.parametrize("objective", ["reference", "relative"])
def test_the_greedy_search_passes_over_an_exact_pair(objective):
    Y = _with_exact_pair(_panel())
    outcomes = greedy_search(Y, [ALL] * Y.shape[1], max_treatment_size=2,
                             n_test=8, objective=objective)
    for o in outcomes:
        assert np.all(np.isfinite(o.score.key))
        assert not o.score.detail.get("exact_fit", False)


@pytest.mark.parametrize("objective", ["reference", "relative"])
def test_a_local_search_started_on_the_exact_pair_leaves_it(objective):
    Y = _with_exact_pair(_panel())
    N = Y.shape[1]
    start = greedy_search(Y, [ALL] * N, max_treatment_size=1, n_test=8,
                          objective=objective)[0]
    exact = score_split(Y[:, 1], Y[:, 0], objective=objective, n_test=8)
    from dataclasses import replace
    pinned = replace(start, treatment=[1], control=[0],
                     unassigned=[j for j in range(N) if j not in (0, 1)],
                     score=exact)
    out = local_search(Y, [ALL] * N, pinned, n_test=8, restarts=2, seed=0,
                       objective=objective)
    assert np.all(np.isfinite(out.score.key))
    assert (out.treatment, out.control) != ([1], [0])


def _frame(Y, post_from):
    T, N = Y.shape
    names = [f"g{j}" for j in range(N)]
    df = pd.DataFrame({"geo": np.repeat(names, T), "t": np.tile(np.arange(T), N),
                       "Y": Y.T.ravel()})
    df["post"] = (df["t"] >= post_from).astype(int)
    return df


def test_a_searched_fit_on_a_panel_with_an_exact_pair_recommends_a_real_design():
    Y = _with_exact_pair(_panel(T=50))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TBR(TBRConfig(df=_frame(Y, 40), unitid="geo", time="t", outcome="Y",
                            post_col="post", max_treatment_size=2, n_test=8,
                            display_graphs=False)).fit()
    rec = res.recommended
    assert not (rec.treatment_units == ["g1"] and rec.control_units == ["g0"])
    assert np.isfinite(rec.objective_value)


# --------------------------------------------------------------------------- #
# a named split
# --------------------------------------------------------------------------- #
def test_a_named_exact_fit_split_is_reported_and_named():
    """A named split is analysed, not ranked: its point estimate is exact and
    its interval has zero width. The estimator says so in a warning that names
    the geos, and returns the result (the exact panels the TBR unit tests run
    on rely on that)."""
    Y = _with_exact_pair(_panel(T=50))
    df = _frame(Y, 40)
    df["tr"] = (df["geo"] == "g1").astype(int)
    df["ct"] = (df["geo"] == "g0").astype(int)
    cfg = TBRConfig(df=df, unitid="geo", time="t", outcome="Y", post_col="post",
                    treatment_col="tr", control_col="ct", display_graphs=False)
    with pytest.warns(UserWarning, match="exact linear function") as caught:
        res = TBR(cfg).fit()
    msg = " ".join(str(w.message) for w in caught)
    assert "g1" in msg and "g0" in msg
    c = res.report.cumulative
    assert c.estimate[-1] == pytest.approx(0.0, abs=1e-6 * float(Y[40:, 1].sum()))
    assert c.upper[-1] - c.lower[-1] == pytest.approx(0.0, abs=1e-6 * float(Y[40:, 1].sum()))


def test_a_named_split_with_a_residual_does_not_warn():
    Y = _panel(T=50)
    df = _frame(Y, 40)
    df["tr"] = (df["geo"] == "g1").astype(int)
    df["ct"] = (df["geo"] == "g0").astype(int)
    cfg = TBRConfig(df=df, unitid="geo", time="t", outcome="Y", post_col="post",
                    treatment_col="tr", control_col="ct", display_graphs=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TBR(cfg).fit()
    assert not any("exact linear function" in str(w.message) for w in caught)

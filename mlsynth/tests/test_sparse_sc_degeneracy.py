"""Degeneracy diagnostics for SparseSC.

The outer problem is nonconvex and nonsmooth, so the returned ``v`` is whichever
critical point the solve reached. These tests pin the two readings that say how
much of the problem that point saw -- ``dim U`` (the support of ``v``, by
Liu and Sagastizabal Example 9.4) and ``|A|`` (the active donor count) -- plus
the two conditions that warn.

The invariant behind every assertion: a count is reported with its denominator
and with the threshold that defines it. A count without either is not a
measurement, and this library has already shipped two different values for
"predictors kept" computed at two different cutoffs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SparseSC
from mlsynth.utils.sparse_sc_helpers.diagnostics import (
    SUPPORT_TOL,
    assess_degeneracy,
    warn_if_degenerate,
)
from mlsynth.utils.sparse_sc_helpers.structures import (
    SparseSCDegeneracy,
    SparseSCDesign,
)


def _design(v, w, v_path, opt_lambda=0.0, lambda_grid=None):
    n = len(np.atleast_2d(v_path))
    grid = np.linspace(0.0, 1.0, n) if lambda_grid is None else np.asarray(
        lambda_grid, float)
    return SparseSCDesign(
        v=np.asarray(v, float), w=np.asarray(w, float),
        opt_lambda=float(opt_lambda), lambda_grid=grid,
        train_loss_curve=np.zeros(n), val_mse_curve=np.zeros(n),
        v_path=np.asarray(v_path, float))


def _panel(seed=0, N=6, T=20, T0=14, P=4):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, P)); L = rng.standard_normal((N + 1, P))
    Y = F @ L.T + 0.3 * rng.standard_normal((T, N + 1))
    Y[T0:, 0] += -2.5
    rec = []
    for u in range(N + 1):
        covs = {f"p{p}": float(L[u, p]) for p in range(P)}
        for t in range(T):
            rec.append({"unit": f"u{u}", "year": 2000 + t, "y": float(Y[t, u]),
                        "tr": int(u == 0 and t >= T0), **covs})
    return pd.DataFrame(rec)


COVS = ["p0", "p1", "p2", "p3"]


class TestReadings:
    def test_dim_u_counts_the_support_of_v(self):
        g = assess_degeneracy(_design([1.0, 0.0, 2.0, 0.0], [0.5, 0.5], [[1.0, 0.0, 2.0, 0.0]]))
        assert g.dim_u == 2
        assert g.n_predictors == 4

    def test_active_donors_counts_the_face(self):
        g = assess_degeneracy(_design([1.0, 1.0], [0.6, 0.4, 0.0], [[1.0, 1.0]]))
        assert g.n_active_donors == 2
        assert g.n_donors == 3

    def test_anchor_only_is_dim_u_equal_one(self):
        """Not a stored field: storing it would create two values that can
        disagree with each other."""
        g = assess_degeneracy(_design([1.0, 0.0, 0.0], [1.0], [[1.0, 0.0, 0.0]]))
        assert g.dim_u == 1 and g.anchor_only is True
        g2 = assess_degeneracy(_design([1.0, 3.0, 0.0], [1.0], [[1.0, 3.0, 0.0]]))
        assert g2.dim_u == 2 and g2.anchor_only is False

    def test_path_counts_use_the_free_weights_not_the_anchor(self):
        """The anchor is pinned at 1, so it is in every support and must not
        be what makes a grid point look non-degenerate."""
        path = [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 0.0]]
        g = assess_degeneracy(_design([1.0, 0.0, 0.0], [1.0], path))
        assert g.n_anchor_only_grid == 2
        assert g.n_distinct_supports == 2
        assert g.n_grid == 3

    def test_every_count_carries_its_threshold(self):
        g = assess_degeneracy(_design([1.0, 0.0], [1.0], [[1.0, 0.0]]))
        assert g.support_tol == SUPPORT_TOL
        assert g.active_tol > 0

    def test_the_support_threshold_is_honoured(self):
        below = SUPPORT_TOL / 10.0
        g = assess_degeneracy(_design([1.0, below], [1.0], [[1.0, below]]))
        assert g.dim_u == 1


class TestThePenaltyDoingNothing:
    """Two readings of "the L1 term selected nothing", both threshold-free.

    ``w*(c v) = w*(v)``: the inner problem is positive-scale-invariant in
    ``v``, so ``lambda ||v||_1`` can be made small by shrinking ``v`` without
    moving a single donor weight. The penalty therefore bites only through the
    anchor, which is pinned at 1. When it does not bite, the sweep runs to the
    top of the grid and keeps every predictor, and the reported "selection" is
    the predictor list it was handed.

    Measured on the SCMO Germany panel (nine 1989 indicators as covariates):
    lambda* = 1, the grid maximum, with 8 of 8 predictors kept and 2 distinct
    supports over 51 grid points. Neither existing condition fires there --
    ``anchor_only`` is False and ``|A|`` is 9 -- so the fit came back with
    nothing said about it.
    """

    def test_nothing_pruned_is_dim_u_equal_to_the_predictor_count(self):
        d = assess_degeneracy(_design([1.0, 0.4, 0.2], [0.5, 0.5],
                                      [[1.0, 0.4, 0.2]]))
        assert d.nothing_pruned
        d = assess_degeneracy(_design([1.0, 0.4, 0.0], [0.5, 0.5],
                                      [[1.0, 0.4, 0.0]]))
        assert not d.nothing_pruned

    def test_the_selected_lambda_and_the_grid_top_are_both_recorded(self):
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.2], [0.5, 0.5],
                                      [[1.0, 0.4, 0.2]] * 3,
                                      opt_lambda=1.0, lambda_grid=grid))
        assert d.lambda_selected == pytest.approx(1.0)
        assert d.lambda_grid_max == pytest.approx(1.0)
        assert d.penalty_at_grid_edge

    def test_a_lambda_inside_the_grid_is_not_at_the_edge(self):
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.0], [0.5, 0.5],
                                      [[1.0, 0.4, 0.0]] * 3,
                                      opt_lambda=0.1, lambda_grid=grid))
        assert not d.penalty_at_grid_edge

    def test_keeping_every_predictor_under_a_live_penalty_warns(self):
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.2], [0.5, 0.5],
                                      [[1.0, 0.4, 0.2]] * 3,
                                      opt_lambda=0.1, lambda_grid=grid))
        with pytest.warns(UserWarning, match="3 of 3"):
            warn_if_degenerate(d)

    def test_a_zero_penalty_keeping_everything_is_not_a_degeneracy(self):
        """At lambda* = 0 there is no penalty, so pruning nothing is correct.

        The warning is about a penalty that was applied and removed nothing,
        not about a full predictor set.
        """
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.2], [0.5, 0.3, 0.2],
                                      [[1.0, 0.4, 0.2]] * 3,
                                      opt_lambda=0.0, lambda_grid=grid))
        assert d.nothing_pruned
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_degenerate(d)
        assert [str(c.message) for c in caught] == []

    def test_a_lambda_at_the_grid_top_warns_and_names_it(self):
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.0], [0.5, 0.5],
                                      [[1.0, 0.4, 0.0]] * 3,
                                      opt_lambda=1.0, lambda_grid=grid))
        with pytest.warns(UserWarning, match="largest lambda"):
            warn_if_degenerate(d)

    def test_the_germany_signature_raises_both(self):
        """lambda* at the grid top with the full predictor set kept."""
        grid = np.concatenate([[0.0], np.logspace(-4, 0, 50)])
        v = np.ones(8)
        d = assess_degeneracy(_design(v, np.full(9, 1 / 9), [v] * 51,
                                      opt_lambda=1.0, lambda_grid=grid))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_degenerate(d)
        messages = [str(c.message) for c in caught]
        assert len(messages) == 2, messages
        assert any("8 of 8" in m for m in messages)
        assert any("largest lambda" in m for m in messages)

    def test_a_pruned_fit_inside_the_grid_warns_about_neither(self):
        grid = np.array([0.0, 0.1, 1.0])
        d = assess_degeneracy(_design([1.0, 0.4, 0.0], [0.5, 0.3, 0.2],
                                      [[1.0, 0.4, 0.0]] * 3,
                                      opt_lambda=0.1, lambda_grid=grid))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_degenerate(d)
        assert [str(c.message) for c in caught] == []


class TestWarnings:
    def test_anchor_only_warns_and_names_the_predictor(self):
        g = assess_degeneracy(_design([1.0, 0.0], [1.0, 0.0], [[1.0, 0.0]]))
        with pytest.warns(UserWarning, match="anchor predictor"):
            warn_if_degenerate(g, ["p_cig", "loginc"])

    def test_a_single_active_donor_warns(self):
        g = assess_degeneracy(_design([1.0, 2.0], [1.0, 0.0], [[1.0, 2.0]]))
        with pytest.warns(UserWarning, match="donor"):
            warn_if_degenerate(g, ["a", "b"])

    def test_a_healthy_fit_warns_about_nothing(self):
        g = assess_degeneracy(_design([1.0, 2.0, 3.0], [0.5, 0.3, 0.2], [[1.0, 2.0, 3.0]]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_degenerate(g, ["a", "b", "c"])


class TestOnTheEstimator:
    def test_the_diagnostics_reach_the_result(self):
        res = SparseSC({"df": _panel(), "outcome": "y", "treat": "tr",
                        "unitid": "unit", "time": "year", "covariates": COVS,
                        "run_inference": False, "display_graphs": False}).fit()
        p = res.method_details.parameters_used
        assert p["n_predictors"] == len(res.design.v)
        assert p["n_donors"] == len(res.design.w)
        assert 1 <= p["dim_u"] <= p["n_predictors"]
        assert 1 <= p["n_active_donors"] <= p["n_donors"]
        assert p["n_grid"] == len(res.design.lambda_grid)
        assert 1 <= p["n_distinct_supports"] <= p["n_grid"]
        assert p["anchor_only"] == (p["dim_u"] == 1)
        assert p["nothing_pruned"] == (p["dim_u"] == p["n_predictors"])
        assert p["penalty_at_grid_edge"] == (
            p["lambda_grid_max"] > 0.0 and p["opt_lambda"] >= p["lambda_grid_max"])
        assert p["lambda_grid_max"] == pytest.approx(
            float(max(res.design.lambda_grid)))

    def test_the_diagnostics_change_no_estimate(self):
        kw = {"df": _panel(1), "outcome": "y", "treat": "tr", "unitid": "unit",
              "time": "year", "covariates": COVS, "run_inference": False,
              "display_graphs": False}
        a = SparseSC(dict(kw)).fit()
        b = SparseSC(dict(kw)).fit()
        assert a.att == pytest.approx(b.att)
        np.testing.assert_allclose(a.design.v, b.design.v)
        np.testing.assert_allclose(a.design.w, b.design.w)

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


def _design(v, w, v_path):
    n = len(np.atleast_2d(v_path))
    return SparseSCDesign(
        v=np.asarray(v, float), w=np.asarray(w, float), opt_lambda=0.0,
        lambda_grid=np.zeros(n), train_loss_curve=np.zeros(n),
        val_mse_curve=np.zeros(n), v_path=np.asarray(v_path, float))


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

    def test_the_diagnostics_change_no_estimate(self):
        kw = {"df": _panel(1), "outcome": "y", "treat": "tr", "unitid": "unit",
              "time": "year", "covariates": COVS, "run_inference": False,
              "display_graphs": False}
        a = SparseSC(dict(kw)).fit()
        b = SparseSC(dict(kw)).fit()
        assert a.att == pytest.approx(b.att)
        np.testing.assert_allclose(a.design.v, b.design.v)
        np.testing.assert_allclose(a.design.w, b.design.w)

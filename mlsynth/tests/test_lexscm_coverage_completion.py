"""Targeted tests closing the last coverage gaps in the LEXSCM modules.

Each test names the exact branch it exercises so the intent is auditable.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlsynth import LEXSCM
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fast_scm_helpers.conflict import max_independent_set_size_at_least
from mlsynth.utils.fast_scm_helpers.lexpower import _as_series_list
from mlsynth.utils.fast_scm_helpers.lexselect import DesignMetrics, select_design
from mlsynth.utils.fast_scm_helpers import fast_scm_control_helpers as ch


def test_max_independent_set_too_few_candidates():
    # conflict.py: fewer candidates than m -> cannot host a size-m independent set.
    A = np.zeros((4, 4), bool)
    assert max_independent_set_size_at_least(A, [0, 1], m=3) is False
    assert max_independent_set_size_at_least(None, [0, 1], m=3) is True
    assert max_independent_set_size_at_least(A, [0, 1, 2, 3], m=2) is True


def test_as_series_list_list_of_series():
    # lexpower.py: a list/tuple of 1-D series is returned as-is (dropping empties).
    out = _as_series_list([np.array([1.0, 2.0]), np.array([]), [3.0, 4.0]])
    assert len(out) == 2 and out[0].tolist() == [1.0, 2.0]
    # the scalar-pool fallback path
    assert _as_series_list([5.0, 6.0])[0].tolist() == [5.0, 6.0]
    assert _as_series_list([]) == []


def test_select_design_empty():
    # lexselect.py: no candidate designs -> EMPTY recommendation.
    rec = select_design([])
    assert rec.status == "EMPTY" and rec.winner is None


def test_select_design_power_not_established():
    # lexselect.py: every gated design has an infeasible MDE -> POWER_NOT_ESTABLISHED,
    # and the winner's absolute MDE is not finite -> the "(not reached)" tail.
    designs = [
        DesignMetrics(design_id="d1", indices=[0, 1], imbalance=1.0),   # mde_sd=inf
        DesignMetrics(design_id="d2", indices=[2, 3], imbalance=1.1),
    ]
    rec = select_design(designs)
    assert rec.status == "POWER_NOT_ESTABLISHED"
    assert rec.winner.design_id == "d1"                 # best balance
    assert "not reached" in rec.explanation


def _panel(n=16, T=40, T_post=10, seed=0, n_cand=12, n_regions=4):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 2)); nu = rng.normal(size=(T, 2))
    Y = 100 + nu @ g.T + 0.1 * rng.normal(size=(T, n))
    return pd.DataFrame([
        {"unitid": i, "time": t, "y": Y[t, i], "post": int(t >= T - T_post),
         "candidate": int(i < n_cand), "region": f"R{i % n_regions}"}
        for i in range(n) for t in range(T)
    ])


def test_spillover_empties_all_donor_pools_raises():
    # fast_scm_control.py: with cluster == region and one treated per region
    # (m == n_regions), the Stage-2 exclusion (treated + same-region neighbours)
    # empties every donor pool -> each candidate is skipped, then the all-empty
    # guard raises.
    df = _panel(n_regions=4)
    with pytest.raises(MlsynthConfigError, match="donor pool emptied"):
        LEXSCM({"df": df, "outcome": "y", "unitid": "unitid", "time": "time",
                "candidate_col": "candidate", "post_col": "post", "m": 4,
                "cluster_col": "region", "top_K": 5, "verbose": False}).fit()


def test_solve_control_qp_returns_none_on_solver_failure():
    # fast_scm_control_helpers.py: when the inner QP solver returns no solution,
    # solve_control_qp propagates None rather than crashing.
    X_E = np.random.default_rng(0).normal(size=(5, 5))
    treated_vec = X_E[:, 0]
    with patch.object(ch, "_solve_qp_problem", return_value=None):
        out = ch.solve_control_qp(X_E, treated_vec, treated_idx=[0], lambda_penalty=0.1)
    assert out is None

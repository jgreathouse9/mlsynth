"""Tests for the ``backend`` switch on the SYNDES estimator.

Contract. ``backend`` selects how ``mode="two_way_global"`` is solved and
nothing else. It defaults to ``"exact"``, the treated-set search; ``"mip"``
keeps the branch-and-bound path. Both minimise the same objective, so they must
reach the same design whenever the MIP is asked to prove optimality.

The switch is refused where the reformulation does not apply: one-way pins the
treated weights and per-unit carries an ``(N, N)`` weight matrix, so neither
reduces to a search over treated sets, and donor-side restrictions tie the
control weights to the assignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units=9, T=16, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    L = rng.standard_normal((3, n_units))
    Y = F @ L + 0.3 * rng.standard_normal((T, n_units))
    return pd.DataFrame([
        {"unit": f"u{j}", "time": t, "y": float(Y[t, j]),
         "post": int(t >= T - n_post)}
        for j in range(n_units) for t in range(T)
    ])


def _fit(**over):
    cfg = {"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
           "K": 3, "mode": "two_way_global", "post_col": "post",
           "run_inference": False}
    cfg.update(over)
    return SYNDES(cfg).fit()


class TestDefault:
    def test_backend_defaults_to_the_search(self):
        assert _fit().design.raw_results["solver"] == "exact"

    def test_explicit_and_implicit_defaults_agree(self):
        explicit = _fit(backend="exact")
        implicit = _fit()
        assert (tuple(explicit.design.selected_unit_indices)
                == tuple(implicit.design.selected_unit_indices))
        assert explicit.design.objective_value == pytest.approx(
            implicit.design.objective_value, rel=1e-12)


class TestAgreement:
    """The MIP path stops early by default, so parity is one-sided.

    ``gap_limit`` defaults to 0.05, which lets SCIP return a design up to 5
    percent above the optimum, and ``time_limit`` can cut it shorter still. The
    exact backend does not stop early, so the claim that holds at every ``K`` is
    that it is never worse. Strict set equality is asserted separately, with the
    MIP told to prove optimality.
    """

    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_exact_is_never_worse(self, K):
        mip, exact = _fit(backend="mip", K=K), _fit(backend="exact", K=K)
        assert (float(exact.design.objective_value)
                <= float(mip.design.objective_value) + 1e-8)

    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_same_design_once_the_mip_must_prove_optimality(self, K):
        mip = _fit(backend="mip", K=K, gap_limit=0.0, time_limit=600.0)
        exact = _fit(backend="exact", K=K)
        assert (tuple(mip.design.selected_unit_indices)
                == tuple(exact.design.selected_unit_indices))
        assert exact.design.objective_value == pytest.approx(
            float(mip.design.objective_value), rel=1e-6)

    def test_exact_reports_its_provenance(self):
        design = _fit(backend="exact").design
        assert design.raw_results["solver"] == "exact"
        assert design.raw_results["certified"] is True

    def test_mode_label_is_the_paper_vocabulary(self):
        assert _fit(backend="exact").design.mode == "two_way_global"

    def test_exact_beats_the_early_stopping_default_somewhere(self):
        """Pin the reason parity is one-sided, so the gap_limit default is visible."""
        beaten = []
        for K in (2, 3, 4, 5):
            mip, exact = _fit(backend="mip", K=K), _fit(backend="exact", K=K)
            if float(exact.design.objective_value) < float(
                    mip.design.objective_value) - 1e-9:
                beaten.append(K)
        assert beaten, (
            "the MIP's 5 percent gap_limit never bound on this panel; if that "
            "becomes true generally, the one-sided parity above can tighten")

    def test_results_object_is_the_same_shape(self):
        mip, exact = _fit(backend="mip"), _fit(backend="exact")
        assert type(exact) is type(mip)
        assert exact.design.treated_weights.shape == mip.design.treated_weights.shape


class TestPoolAndSelection:
    def test_pool_agrees_with_the_mip(self):
        mip = _fit(backend="mip", top_K=3, selection="in_sample")
        exact = _fit(backend="exact", top_K=3, selection="in_sample")
        assert (tuple(mip.design.selected_unit_indices)
                == tuple(exact.design.selected_unit_indices))

    def test_holdout_selection_runs_on_the_exact_backend(self):
        res = _fit(backend="exact", top_K=3, selection="holdout", holdout_frac=0.3)
        assert int(res.design.assignment.sum()) == 3

    def test_ic_selection_runs_on_the_exact_backend(self):
        res = _fit(backend="exact", top_K=3, selection="ic")
        assert int(res.design.assignment.sum()) == 3


class TestRefusals:
    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_rejected_for_other_modes(self, mode):
        with pytest.raises(MlsynthConfigError):
            _fit(backend="exact", mode=mode)

    def test_rejected_with_donor_exclusions(self):
        adjacency = pd.DataFrame(
            np.eye(9), index=[f"u{i}" for i in range(9)],
            columns=[f"u{i}" for i in range(9)])
        with pytest.raises(MlsynthConfigError):
            _fit(backend="exact", donor_exclusion=adjacency)

    def test_rejects_an_unknown_backend(self):
        with pytest.raises((MlsynthConfigError, ValueError)):
            _fit(backend="quantum")


class TestRestrictionsThroughTheEstimator:
    def test_forced_units_reach_the_search(self):
        res = _fit(backend="exact", to_be_treated=["u0", "u4"])
        chosen = set(res.design.selected_unit_labels)
        assert {"u0", "u4"} <= chosen

    def test_forbidden_units_are_excluded(self):
        res = _fit(backend="exact", not_to_be_treated=["u1", "u2"])
        assert not ({"u1", "u2"} & set(res.design.selected_unit_labels))

    def test_budget_is_enforced(self):
        costs = list(np.arange(1.0, 10.0))
        res = _fit(backend="exact", costs=costs, budget=9.0)
        spent = sum(costs[i] for i in res.design.selected_unit_indices)
        assert spent <= 9.0 + 1e-9

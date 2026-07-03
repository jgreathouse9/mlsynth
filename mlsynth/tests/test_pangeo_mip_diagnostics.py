"""TDD for Gurobi-style MIP diagnostics on the exact PANGEO cover solver.

Beyond the basic ``solve_partition`` diagnostics (status, objective, solver,
time, selected count), scientists debugging a design want the solver's own
optimality accounting -- how far from proven-optimal it stopped -- and, when no
exact cover exists, *why*. This adds:

* success-path fields ``mip_gap`` (optimality gap), ``dual_bound`` (best proven
  bound), ``node_count`` (branch-and-bound nodes), ``simplex_iterations``,
  pulled from HiGHS' ``extra_stats`` when available (``None`` otherwise);
* an IIS-style structural infeasibility message: when the cover MIP is
  infeasible the raised ``MlsynthEstimationError`` names the obstruction --
  units that appear in no admissible pair, or an odd-arm/even-pair parity clash.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers.mip import solve_partition
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


def _wide(n, T, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, T)), axis=1)


class TestMIPSolveStats:
    def test_success_diagnostics_include_gap_and_bound(self):
        Y = _wide(8, 16, seed=0)
        idx = np.arange(8)
        cands = enumerate_candidate_pairs(idx, Y, 2)
        chosen, diag = solve_partition(cands, idx, return_diagnostics=True)
        assert diag["status"] == "optimal"
        # The gap/bound/node fields come from the backing MILP solver's
        # extra_stats; HiGHS reports them, but a fallback solver (or a cvxpy
        # build that does not surface HighsInfo) may leave them None. Assert
        # their *values* only when the solver populated them -- the schema
        # (keys always present) is pinned separately -- so this stays green
        # across solver/wheel differences between Python versions.
        if diag["mip_gap"] is not None:
            assert diag["mip_gap"] == pytest.approx(0.0, abs=1e-6)
        if diag["dual_bound"] is not None:
            # at a proven optimum the dual bound meets the objective (min)
            assert diag["dual_bound"] <= diag["objective_value"] + 1e-6
        if diag["node_count"] is not None:
            assert diag["node_count"] >= 0

    def test_stats_keys_present_even_if_solver_omits_them(self):
        # The keys must always exist (value may be None on solvers that do not
        # report them), so downstream code can rely on the schema.
        Y = _wide(6, 14, seed=1)
        idx = np.arange(6)
        cands = enumerate_candidate_pairs(idx, Y, 2)
        _, diag = solve_partition(cands, idx, return_diagnostics=True)
        for k in ("mip_gap", "dual_bound", "node_count", "simplex_iterations"):
            assert k in diag


class TestInfeasibilityReason:
    def test_odd_arm_q1_parity_message(self):
        # 5 units, Q=1 -> only size-2 pairs -> no exact cover of an odd arm.
        Y = _wide(5, 12, seed=0)
        idx = np.arange(5)
        cands = enumerate_candidate_pairs(idx, Y, 1)
        with pytest.raises(MlsynthEstimationError) as exc:
            solve_partition(cands, idx)
        msg = str(exc.value).lower()
        assert "odd" in msg
        assert "exact cover" in msg

    def test_uncoverable_unit_message(self):
        # unit 3 appears in no candidate pair -> structurally uncoverable.
        cands = [
            {"members": np.array([0, 1]), "score": 1.0,
             "side_a": np.array([0]), "side_b": np.array([1])},
            {"members": np.array([1, 2]), "score": 1.0,
             "side_a": np.array([1]), "side_b": np.array([2])},
            {"members": np.array([0, 2]), "score": 1.0,
             "side_a": np.array([0]), "side_b": np.array([2])},
        ]
        with pytest.raises(MlsynthEstimationError) as exc:
            solve_partition(cands, np.array([0, 1, 2, 3]))
        assert "no admissible supergeo pair" in str(exc.value)

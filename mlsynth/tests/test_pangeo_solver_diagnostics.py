"""TDD for PANGEO solver diagnostics surfaced in ``results.metadata``.

The design solver -- either the OSD-style fast partition or the exact-cover
set-partitioning MIP -- previously discarded its own accounting: how many
candidate groupings/pairs it considered, which one won, the objective it
achieved, the MIP status, and the solve time. A user inspecting a design could
not tell how many candidate partitions were tried or whether the MIP solved
cleanly.

These tests pin that:

1. ``fast_partition(..., return_diagnostics=True)`` returns the design *and* a
   diagnostics dict (candidates requested/feasible, winner index + linkage,
   winning total score, per-candidate scores), while the default call stays
   backward-compatible (returns just the design list).
2. ``solve_partition(..., return_diagnostics=True)`` returns the chosen pairs
   *and* a diagnostics dict (candidate-pair count = |F|, objective value, MIP
   status, solver name, solve seconds, selected-pair count), default unchanged.
3. ``PANGEO.fit()`` surfaces these per arm in
   ``results.metadata["solver_diagnostics"]`` and echoes ``fast_candidates``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PANGEO
from mlsynth.config_models import PANGEOConfig
from mlsynth.utils.pangeo_helpers.fast_partition import fast_partition
from mlsynth.utils.pangeo_helpers.mip import solve_partition
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


def _panel(n=12, T=24, seed=0):
    """Balanced parallel-factor panel (units move in parallel up to a level)."""
    rng = np.random.default_rng(seed)
    F = np.cumsum(rng.standard_normal((T, 2)), axis=0)
    rows = []
    for i in range(n):
        b = rng.standard_normal(2)
        y = 10.0 + i + F @ b + rng.standard_normal(T) * 0.1
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "sales": float(y[t]), "arm": "A"})
    return pd.DataFrame(rows)


def _wide(n, T, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, T)), axis=1)


class TestFastPartitionDiagnostics:
    def test_returns_diagnostics_when_requested(self):
        Y = _wide(12, 20, seed=0)
        design, diag = fast_partition(
            np.arange(12), Y, 2, n_candidates=5, return_diagnostics=True
        )
        assert isinstance(design, list) and len(design) > 0
        assert diag["path"] == "fast"
        assert diag["n_candidates_requested"] == 5
        assert 1 <= diag["n_candidates_feasible"] <= 5
        assert 0 <= diag["winning_candidate"] < 5
        assert diag["winning_linkage"] in {"ward", "average", "complete"}
        assert np.isfinite(diag["winning_total_score"])
        assert len(diag["candidate_scores"]) == 5
        # the winner is the feasible candidate of minimum total score
        feasible = [s for s in diag["candidate_scores"] if s is not None]
        assert diag["winning_total_score"] == pytest.approx(min(feasible))

    def test_backward_compatible_default_returns_list(self):
        Y = _wide(10, 18, seed=1)
        out = fast_partition(np.arange(10), Y, 2, n_candidates=3)
        assert isinstance(out, list)  # not a (design, diag) tuple


class TestExactMIPDiagnostics:
    def test_returns_diagnostics_when_requested(self):
        Y = _wide(8, 16, seed=0)
        idx = np.arange(8)
        cands = enumerate_candidate_pairs(idx, Y, 2)
        chosen, diag = solve_partition(cands, idx, return_diagnostics=True)
        assert isinstance(chosen, list) and len(chosen) > 0
        assert diag["path"] == "exact_mip"
        assert diag["n_candidate_pairs"] == len(cands)
        assert diag["status"] in {"optimal", "optimal_inaccurate"}
        assert np.isfinite(diag["objective_value"])
        assert diag["solve_seconds"] >= 0.0
        assert diag["n_selected_pairs"] == len(chosen)

    def test_backward_compatible_default_returns_list(self):
        Y = _wide(6, 14, seed=2)
        idx = np.arange(6)
        cands = enumerate_candidate_pairs(idx, Y, 2)
        out = solve_partition(cands, idx)
        assert isinstance(out, list)


class TestPipelineSurfacesDiagnostics:
    def test_fast_path_metadata(self):
        df = _panel(12, 24, seed=3)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=2, fast=True, fast_candidates=4,
            compute_power=False, display_graphs=False)).fit()
        assert "solver_diagnostics" in res.metadata
        assert res.metadata["fast_candidates"] == 4
        d = res.metadata["solver_diagnostics"]["A"]
        assert d["path"] == "fast"
        assert d["n_candidates_requested"] == 4
        assert 0 <= d["winning_candidate"] < 4

    def test_exact_path_metadata(self):
        df = _panel(8, 20, seed=4)
        res = PANGEO(PANGEOConfig(
            df=df, outcome="sales", arm="arm", unitid="unit", time="time",
            max_supergeo_size=2, fast=False,
            compute_power=False, display_graphs=False)).fit()
        d = res.metadata["solver_diagnostics"]["A"]
        assert d["path"] == "exact_mip"
        assert d["n_candidate_pairs"] > 0
        assert d["status"] in {"optimal", "optimal_inaccurate"}
        assert d["n_selected_pairs"] == len(res.arm_designs["A"].pairs)

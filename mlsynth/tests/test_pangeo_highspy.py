"""Differential tests: the highspy MIP backend must match the cvxpy oracle.

PANGEO's exact-cover partition MIP is solved by highspy directly when available
(dropping the cvxpy middleman), falling back to cvxpy otherwise. The two must
return the same optimum -- the same total score over a valid exact cover -- on
every instance. Ties in score can yield a different but equally optimal cover,
so the contract is on the objective value and cover validity, not the exact
selection.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers.mip import (
    solve_partition,
    _solve_partition_highspy,
    _solve_partition_cvxpy,
    _highspy_available,
)

pytestmark = pytest.mark.skipif(
    not _highspy_available(), reason="highspy not installed")


def _pair(members, score):
    m = [int(u) for u in members]
    half = len(m) // 2
    return {"members": m, "score": float(score),
            "side_a": m[:half], "side_b": m[half:]}


def _rand_instance(seed, n_units, extra=10, allow_supergeo=True):
    """Random FEASIBLE candidate set: a base perfect matching + random extras."""
    rng = np.random.default_rng(seed)
    pairs = [_pair([i, i + 1], rng.uniform(0.1, 3.0))
             for i in range(0, n_units, 2)]
    for _ in range(extra):
        k = int(rng.choice([2, 4])) if allow_supergeo else 2
        k = min(k, n_units - (n_units % 2))
        if k < 2:
            continue
        mem = sorted(rng.choice(n_units, size=k, replace=False).tolist())
        pairs.append(_pair(mem, rng.uniform(0.1, 3.0)))
    return np.arange(n_units), pairs


def _cover_ok(chosen, units):
    covered = [int(u) for p in chosen for u in p["members"]]
    return sorted(covered) == sorted(int(u) for u in units)


def _obj(chosen):
    return sum(p["score"] for p in chosen)


class TestHighspyMatchesCvxpy:
    @pytest.mark.parametrize("seed", range(30))
    def test_same_objective_and_valid_cover(self, seed):
        units, pairs = _rand_instance(seed, n_units=6)
        ch_h = _solve_partition_highspy(pairs, units, min_pairs=1)
        ch_c = _solve_partition_cvxpy(pairs, units, min_pairs=1)
        assert _cover_ok(ch_h, units), "highspy returned an invalid cover"
        assert _cover_ok(ch_c, units), "cvxpy returned an invalid cover"
        assert _obj(ch_h) == pytest.approx(_obj(ch_c), abs=1e-6)

    @pytest.mark.parametrize("n_units", [4, 8, 10])
    def test_larger_arms(self, n_units):
        units, pairs = _rand_instance(7, n_units=n_units, extra=16)
        ch_h = _solve_partition_highspy(pairs, units, 1)
        ch_c = _solve_partition_cvxpy(pairs, units, 1)
        assert _cover_ok(ch_h, units) and _cover_ok(ch_c, units)
        assert _obj(ch_h) == pytest.approx(_obj(ch_c), abs=1e-6)

    @pytest.mark.parametrize("min_pairs", [1, 2, 3])
    def test_min_pairs_constraint(self, min_pairs):
        units, pairs = _rand_instance(9, n_units=8, extra=14)
        ch_h = _solve_partition_highspy(pairs, units, min_pairs)
        ch_c = _solve_partition_cvxpy(pairs, units, min_pairs)
        assert len(ch_h) >= min_pairs and len(ch_c) >= min_pairs
        assert _cover_ok(ch_h, units) and _cover_ok(ch_c, units)
        assert _obj(ch_h) == pytest.approx(_obj(ch_c), abs=1e-6)


class TestHighspyContract:
    def test_no_candidates_raises(self):
        with pytest.raises(MlsynthEstimationError):
            _solve_partition_highspy([], np.arange(4), 1)

    def test_uncoverable_unit_raises_with_reason(self):
        pairs = [_pair([0, 1], 1.0), _pair([0, 2], 1.0)]   # unit 3 in no pair
        with pytest.raises(MlsynthEstimationError, match="uncoverable"):
            _solve_partition_highspy(pairs, np.arange(4), 1)

    def test_diagnostics_schema(self):
        units, pairs = _rand_instance(1, 6)
        chosen, diag = _solve_partition_highspy(
            pairs, units, 1, return_diagnostics=True)
        assert diag["path"] == "exact_mip"
        assert diag["status"] == "optimal"
        assert diag["n_selected_pairs"] == len(chosen)
        for k in ("mip_gap", "dual_bound", "node_count", "simplex_iterations",
                  "objective_value", "n_candidate_pairs", "solve_seconds",
                  "solver_name"):
            assert k in diag
        assert diag["objective_value"] == pytest.approx(_obj(chosen), abs=1e-6)


class TestDispatch:
    def test_solve_partition_prefers_highspy(self):
        units, pairs = _rand_instance(3, 6)
        chosen = solve_partition(pairs, units, 1)
        assert _cover_ok(chosen, units)
        _, diag = solve_partition(pairs, units, 1, return_diagnostics=True)
        assert diag["path"] == "exact_mip"

    def test_falls_back_to_cvxpy_when_highspy_absent(self, monkeypatch):
        import mlsynth.utils.pangeo_helpers.mip as mip
        monkeypatch.setattr(mip, "_highspy_available", lambda: False)
        units, pairs = _rand_instance(5, 6)
        chosen = mip.solve_partition(pairs, units, 1)   # routes to the cvxpy MIP
        assert _cover_ok(chosen, units)

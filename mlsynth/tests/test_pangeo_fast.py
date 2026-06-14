"""TDD for the OSD-style fast partition path for PANGEO.

Background -- the exact PANGEO design problem
---------------------------------------------
For one treatment arm with units ``U`` and pre-period outcomes ``Ypre`` (rows =
units, cols = time), PANGEO partitions ``U`` into *supergeo pairs*. Each pair is
a subset ``G`` of units split into a treatment half ``A`` and control half ``B``
(each of size ``<= Q = max_supergeo_size``), scored by the level-removed
("difference-in-differences") gap variance of their mean trajectories

    score(G) = min over splits (A,B) of  sum_t [ (Ybar_A - Ybar_B)_t - delta ]^2,
    delta = mean_t (Ybar_A - Ybar_B)_t                      (free DiD level).

The exact estimator (``enumerate_candidate_pairs`` + the set-partitioning MIP)
considers *every* admissible subset of size ``2..2Q`` -- ``O(n^{2Q})`` candidates
-- then solves an NP-hard exact-cover MIP. That is the bottleneck.

The OSD-style fast path (this module)
-------------------------------------
Borrowing Shaw (2025) "Optimized Supergeo Design": instead of enumerating all
subsets, *cluster* the units into size-bounded groups and reuse the exact
per-group split. Setup: level-remove each unit's trajectory (subtract its own
temporal mean -> shape), embed (PCA), hierarchically order, and chunk into groups
of size ``2..2Q`` so that parallel-moving units land together. Optimization: per
group, ``best_split`` (unchanged) returns the optimal level-removed split. A few
candidate groupings are generated (varying linkage / perturbation) and the one
with the smallest total ``score`` is kept -- producing the same output contract
as the exact partition (a list of ``{members, score, side_a, side_b}``) without
the enumeration or the MIP.

These tests pin the setup (size-bounded exact-cover grouping that recovers
parallel clusters) and the optimization (per-group split parity with the exact
path, near-optimal total score on small instances where the exact solve is
feasible).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.pangeo_helpers.fast_partition import fast_partition, group_units
from mlsynth.utils.pangeo_helpers.parallelism import (
    best_split,
    enumerate_candidate_pairs,
)


# ---------------------------------------------------------------------------
# DGP: latent parallel "shapes"; units = shape + level offset + small noise
# ---------------------------------------------------------------------------

def _parallel_panel(n_shapes=4, per_shape=2, T=20, noise=0.05, seed=0):
    """Build Ypre where ``per_shape`` units share each latent trajectory shape
    (up to a level offset). Units of the same shape are perfectly parallel, so
    the ideal design pairs them together."""
    rng = np.random.default_rng(seed)
    shapes = rng.standard_normal((n_shapes, T))
    rows, truth = [], []
    for s in range(n_shapes):
        for _ in range(per_shape):
            level = rng.standard_normal() * 5.0           # arbitrary level
            rows.append(shapes[s] + level + rng.standard_normal(T) * noise)
            truth.append(s)
    Y = np.array(rows)
    return Y, np.array(truth)


# ---------------------------------------------------------------------------
# Setup: size-bounded exact-cover grouping
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_is_exact_cover_with_size_bounds(self):
        Y, _ = _parallel_panel(n_shapes=5, per_shape=2)   # n=10
        idx = np.arange(Y.shape[0])
        groups = group_units(Y, idx, max_size=2, seed=0)
        # every unit covered exactly once
        covered = np.sort(np.concatenate(groups))
        np.testing.assert_array_equal(covered, idx)
        # each group splittable into two halves each <= 2  => size in [2, 4]
        assert all(2 <= len(g) <= 4 for g in groups)

    def test_groups_recover_parallel_shapes(self):
        # With pairs of identical-shape units and max_size=1 (pairs of 2),
        # grouping should put same-shape units together.
        Y, truth = _parallel_panel(n_shapes=6, per_shape=2, noise=0.02, seed=1)
        idx = np.arange(Y.shape[0])
        groups = group_units(Y, idx, max_size=1, seed=0)   # size-2 groups
        assert all(len(g) == 2 for g in groups)
        pure = sum(1 for g in groups if truth[g[0]] == truth[g[1]])
        assert pure >= 5                                   # >=5 of 6 pairs pure

    def test_odd_remainder_handled_when_q_ge_2(self):
        Y, _ = _parallel_panel(n_shapes=3, per_shape=3)   # n=9 (odd)
        idx = np.arange(Y.shape[0])
        groups = group_units(Y, idx, max_size=2, seed=0)
        covered = np.sort(np.concatenate(groups))
        np.testing.assert_array_equal(covered, idx)
        assert all(2 <= len(g) <= 4 for g in groups)

    def test_reproducible(self):
        Y, _ = _parallel_panel()
        idx = np.arange(Y.shape[0])
        a = group_units(Y, idx, max_size=2, seed=3)
        b = group_units(Y, idx, max_size=2, seed=3)
        assert [g.tolist() for g in a] == [g.tolist() for g in b]


# ---------------------------------------------------------------------------
# Optimization: per-group split parity + near-optimal partition
# ---------------------------------------------------------------------------

class TestFastPartition:
    def test_output_contract(self):
        Y, _ = _parallel_panel(n_shapes=5, per_shape=2)
        idx = np.arange(Y.shape[0])
        pairs = fast_partition(idx, Y, max_size=2, n_candidates=3, seed=0)
        # exact cover by the selected pairs
        covered = np.sort(np.concatenate([p["members"] for p in pairs]))
        np.testing.assert_array_equal(covered, idx)
        for p in pairs:
            assert set(p) >= {"members", "score", "side_a", "side_b"}
            assert len(p["side_a"]) <= 2 and len(p["side_b"]) <= 2
            assert np.isfinite(p["score"])

    def test_per_group_score_matches_best_split(self):
        # The score attached to each selected pair must equal best_split's score
        # on that pair's members (the optimization is unchanged per group).
        Y, _ = _parallel_panel()
        idx = np.arange(Y.shape[0])
        pairs = fast_partition(idx, Y, max_size=2, n_candidates=2, seed=0)
        for p in pairs:
            s, _, _ = best_split(np.asarray(p["members"]), Y, max_size=2)
            assert p["score"] == pytest.approx(s, abs=1e-9)

    def test_near_optimal_vs_exact_small(self):
        # On a small panel the exact enumerate+greedy-cover is feasible; the
        # fast path's total score must be close (it is a heuristic, so >= exact,
        # but tight when the parallel structure is clear).
        from mlsynth.utils.pangeo_helpers.mip import solve_partition

        Y, _ = _parallel_panel(n_shapes=4, per_shape=2, noise=0.02, seed=2)  # n=8
        idx = np.arange(Y.shape[0])
        cands = enumerate_candidate_pairs(idx, Y, max_size=1)
        exact = solve_partition(cands, idx, min_pairs=1)
        exact_total = sum(p["score"] for p in exact)

        fast = fast_partition(idx, Y, max_size=1, n_candidates=5, seed=0)
        fast_total = sum(p["score"] for p in fast)
        assert fast_total >= exact_total - 1e-9
        assert fast_total <= exact_total + 1e-6   # clean clusters -> matches exact

    def test_recovers_parallel_pairs(self):
        Y, truth = _parallel_panel(n_shapes=6, per_shape=2, noise=0.02, seed=4)
        idx = np.arange(Y.shape[0])
        pairs = fast_partition(idx, Y, max_size=1, n_candidates=5, seed=0)
        pure = sum(1 for p in pairs
                   if truth[p["members"][0]] == truth[p["members"][1]])
        assert pure >= 5


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_too_few_units(self):
        Y, _ = _parallel_panel(n_shapes=1, per_shape=1)   # n=1
        with pytest.raises(Exception):
            fast_partition(np.array([0]), Y, max_size=1)

    def test_odd_n_with_q1_rejected(self):
        Y, _ = _parallel_panel(n_shapes=3, per_shape=3)   # n=9, Q=1 -> infeasible
        with pytest.raises(Exception):
            group_units(Y, np.arange(9), max_size=1)

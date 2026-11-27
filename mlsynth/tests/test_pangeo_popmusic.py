"""PANGEO's POPMUSIC finder (:mod:`.popmusic`): the incumbent improved by exact re-covers of clusters of its groups.

After a proof stops at its cap, :func:`columns.solve_exact` hands the best design to :func:`popmusic.improve` before
the certify rounds. Each pass splits the design's groups into disjoint clusters, a seed group and its nearest groups
by the centroid of their markets' centred paths up to ``dim`` markets, and re-covers each cluster's markets exactly
with the volume tree over every group inside them, keeping the design's group floor ``k``. A cover is taken only when
its rigorous upper bound is below the cluster's. ``dim`` grows by ``delta`` after a pass that improves nothing or
``repeat`` passes at one size, while it is below the arm's size, and the passes share one node budget (Bijiga,
Warwicker and Rebennack, Networks 2026, Algorithm 1, p. 12). The oracle is the plain cover search over every group,
run to its proof.
"""
from __future__ import annotations

import itertools
import warnings
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover, popmusic, vtree
from mlsynth.utils.pangeo_helpers.structures import PangeoCertifiedGapWarning


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _arm(seed, n, Q, min_pairs=1):
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    root = columns.root_lp(pool, k)
    return Y, idx, pool, k, root


def _paths(Y):
    P = np.asarray(Y, dtype=float)
    return P - P.mean(axis=1, keepdims=True)


def _optimum(pool, root, k, Q):
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return sum((Fraction(float(pool.cost_up[j])) for j in res.rows), Fraction(0))


def _upper(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _seed_design(pool, n):
    return np.array([columns.column_index(g, n) for g in columns.seed_partition(n)], dtype=np.int64)


def _units(pool, j):
    return [int(u) for u in pool.members[j] if u >= 0]


def _is_cover(pool, rows, n, k):
    got = sorted(u for j in rows for u in _units(pool, j))
    return got == list(range(n)) and len(rows) >= k


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_improve_returns_a_cover_no_worse_than_the_seed_design():
    """Smoke: from the seed partition of a 14-unit Q = 2 arm, POPMUSIC returns an exact cover with at least k groups,
    its upper bound at most the start's, with its counts."""
    n, Q = 14, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    rows = _seed_design(pool, n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=4, delta=2)
    assert _is_cover(pool, out["rows"], n, k)
    assert out["upper"] == _upper(pool, out["rows"]) and out["upper"] <= _upper(pool, rows)
    assert out["start_upper"] == _upper(pool, rows)
    assert out["passes"] >= 1 and out["subproblems"] >= 1 and out["nodes"] >= 0 and out["seconds"] >= 0
    assert isinstance(out["rows"], np.ndarray) and out["rows"].dtype == np.int64


def test_smoke_improve_lowers_the_seed_design_and_can_stop_above_the_optimum():
    """Smoke: from the seed partition of a 12-unit arm, POPMUSIC lowers the design and stays at or above the optimum.
    It re-covers only clusters of nearest groups below the arm's size, so here it stops above the optimum."""
    n, Q = 12, 2
    Y, _, pool, k, root = _arm(5, n, Q)
    rows = _seed_design(pool, n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**7, alpha=4, delta=2)
    assert _optimum(pool, root, k, Q) <= out["upper"] < out["start_upper"] and out["improved"] >= 1


# --- unit: the groups inside a set of markets ---------------------------------------------------------------------

@pytest.mark.parametrize("n,Q,U", [(10, 2, [0, 3, 4, 7, 9]), (12, 3, [1, 2, 5, 6, 8, 10, 11]), (8, 2, [2, 6]),
                                   (9, 1, [0, 4, 8])])
def test_rows_within_lists_every_group_of_the_markets_at_its_complete_pool_row(n, Q, U):
    """Every group of 2 to min(2Q, |U|) of the markets U, ascending by size and then lexicographically, at its row
    of a complete pool (:func:`columns.column_index`), with its members as positions into U."""
    rows, members = popmusic.rows_within(np.array(U), n, Q)
    want = [g for m in range(2, min(2 * Q, len(U)) + 1) for g in itertools.combinations(U, m)]
    assert rows.tolist() == [columns.column_index(g, n) for g in want]
    width = min(2 * Q, len(U))
    assert members.shape == (len(want), width)
    for g, mem in zip(want, members):
        assert [U[p] for p in mem if p >= 0] == list(g) and (mem[len(g):] == -1).all()


def test_rows_within_is_empty_with_fewer_than_two_markets():
    rows, members = popmusic.rows_within(np.array([3]), 6, 2)
    assert rows.size == 0 and members.shape[0] == 0


def test_the_subproblem_pool_holds_the_pool_rows_costs_and_splits_on_local_positions():
    """The subproblem's groups carry the arm pool's costs, bounds and splits; its units are positions into U."""
    n, Q = 10, 2
    _, _, pool, _, _ = _arm(1, n, Q)
    U = np.array([0, 2, 3, 6, 7, 9])
    sub, rows = popmusic.subproblem(pool, U, Q)
    assert np.array_equal(sub.unit_indices, np.arange(U.size)) and sub.complete
    assert np.array_equal(sub.cost, pool.cost[rows]) and np.array_equal(sub.cost_up, pool.cost_up[rows])
    assert np.array_equal(sub.cost_lo, pool.cost_lo[rows]) and np.array_equal(sub.side_a, pool.side_a[rows])
    for i, j in enumerate(rows):
        assert [int(U[p]) for p in sub.members[i] if p >= 0] == _units(pool, j)


# --- unit: the clusters -------------------------------------------------------------------------------------------

def test_clusters_split_the_design_into_disjoint_parts_within_dim():
    """A pass's clusters are disjoint, cover every group of the design once, and each holds at most dim markets
    (a single group larger than dim stands alone)."""
    n, Q = 16, 2
    Y, _, pool, _, _ = _arm(2, n, Q)
    rows = _seed_design(pool, n)
    sizes = [len(_units(pool, j)) for j in rows]
    cent = np.array([_paths(Y)[_units(pool, j)].mean(axis=0) for j in rows])
    parts = popmusic.clusters(cent, sizes, 6, np.random.default_rng(0))
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(rows)))
    for p in parts:
        assert sum(sizes[i] for i in p) <= 6 or len(p) == 1


def test_a_cluster_is_its_seed_and_the_groups_nearest_it_by_centroid():
    """With groups on two far-apart lines and room for three groups, every cluster is one line, whichever seed the
    generator draws, and starts at its seed."""
    cent = np.array([[0.0], [10.0], [1.0], [11.0], [2.0], [12.0]])
    for g in range(5):
        parts = popmusic.clusters(cent, [2] * 6, 6, np.random.default_rng(g))
        assert sorted(sorted(p) for p in parts) == [[0, 2, 4], [1, 3, 5]]


def test_a_group_that_would_overflow_is_passed_over_for_a_farther_one_that_fits():
    """A nearer group too large for the room left is skipped and a farther one that fits joins the cluster."""
    cent = np.array([[0.0], [1.0], [5.0]])
    parts = popmusic.clusters(cent, [2, 4, 2], 4, np.random.default_rng(0))
    assert sorted(sorted(p) for p in parts) in ([[0, 2], [1]],)


def test_clusters_are_reproducible_from_the_generator():
    cent = np.random.default_rng(0).normal(size=(9, 3))
    a = popmusic.clusters(cent, [2] * 9, 6, np.random.default_rng(7))
    b = popmusic.clusters(cent, [2] * 9, 6, np.random.default_rng(7))
    assert a == b


# --- unit: the passes ---------------------------------------------------------------------------------------------

def test_a_cover_is_taken_only_when_its_upper_bound_is_lower(monkeypatch):
    """From the optimum nothing improves: every subproblem's best cover is the cluster's own, so the design comes
    back unchanged and no subproblem counts as improved."""
    n, Q = 12, 2
    Y, _, pool, k, root = _arm(5, n, Q)
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9)
    out = popmusic.improve(pool, root, k, Q, res.rows, _paths(Y), 10**6, alpha=4, delta=2)
    assert sorted(out["rows"].tolist()) == sorted(res.rows.tolist()) and out["improved"] == 0


def test_every_subproblem_searches_below_its_clusters_upper_bound_with_the_floor_kept(monkeypatch):
    """Each subproblem's tree searches below the least double at or above its cluster's rigorous upper bound, over
    every group inside the cluster's markets, with the group floor the rest of the design leaves it, and with the
    arm root's duals on those markets."""
    n, Q = 14, 2
    Y, _, pool, k, root = _arm(3, n, Q, min_pairs=6)
    rows = _seed_design(pool, n)
    calls, real, uppers = [], vtree.search, []

    def tree(sub, r, keep, kk, best_up, node_limit, **kw):
        calls.append(dict(sub=sub, pi=np.array(r.pi), sigma=r.sigma, keep=np.array(keep), k=kk, best_up=best_up,
                          cluster=uppers[-1]))
        return real(sub, r, keep, kk, best_up, node_limit, **kw)
    monkeypatch.setattr(vtree, "search", tree)
    seen = []
    real_sub, real_up = popmusic.subproblem, popmusic._upper
    monkeypatch.setattr(popmusic, "subproblem", lambda p, U, q: seen.append(np.array(U)) or real_sub(p, U, q))

    def up(p, groups):
        v = real_up(p, groups)
        uppers.append((sorted(u for j in groups for u in _units(p, j)), v))   # the last before a tree is its cluster's
        return v
    monkeypatch.setattr(popmusic, "_upper", up)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=6, delta=2, repeat=1)
    assert calls and len(calls) == len(seen) == out["subproblems"]
    for c, U in zip(calls, seen):
        assert np.array_equal(c["keep"], np.arange(len(c["sub"]))) and c["sigma"] == 0.0
        assert np.array_equal(c["pi"], np.asarray(root.pi, dtype=float)[U])
        assert c["cluster"][0] == U.tolist() and c["best_up"] == columns._up(c["cluster"][1]) and c["k"] >= 1
    assert _is_cover(pool, out["rows"], n, k)


def test_the_floor_is_the_groups_the_rest_of_the_design_leaves_short_of_k(monkeypatch):
    """With every group a pair forced (k = n / 2), a subproblem of r groups must keep r groups, so the design keeps
    n / 2 groups."""
    n, Q = 12, 2
    Y, _, pool, k, root = _arm(4, n, Q, min_pairs=6)
    assert k == 6
    rows = _seed_design(pool, n)
    ks, real = [], vtree.search
    monkeypatch.setattr(vtree, "search", lambda sub, r, keep, kk, *a, **kw: ks.append((len(sub.unit_indices), kk))
                        or real(sub, r, keep, kk, *a, **kw))
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=4, delta=2)
    assert all(kk == m // 2 for m, kk in ks)
    assert len(out["rows"]) == 6 and _is_cover(pool, out["rows"], n, k)


def test_the_passes_share_one_node_budget_and_each_tree_gets_what_is_left(monkeypatch):
    """Every tree's limit is the lesser of the subproblem cap and what the earlier trees left; the counted nodes
    are the trees' and never pass the budget."""
    n, Q = 16, 2
    Y, _, pool, k, root = _arm(6, n, Q)
    rows = _seed_design(pool, n)
    limits, used, real = [], [], vtree.search

    def tree(sub, r, keep, kk, best_up, node_limit, **kw):
        limits.append(int(node_limit))
        res = real(sub, r, keep, kk, best_up, node_limit, **kw)
        used.append(int(res.nodes))
        return res
    monkeypatch.setattr(vtree, "search", tree)
    budget = 40
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), budget, alpha=6, delta=2, sub_limit=25)
    left = budget
    for lim, u in zip(limits, used):
        assert lim == min(25, left) and lim > 0
        left -= u
    assert out["nodes"] == sum(used) <= budget


def test_a_zero_budget_returns_the_start_unchanged_with_no_pass():
    n, Q = 12, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    rows = _seed_design(pool, n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 0, alpha=4, delta=2)
    assert out["rows"].tolist() == rows.tolist() and out["passes"] == 0 and out["nodes"] == 0


@pytest.mark.parametrize("seed,repeat", [(3, 1), (3, 2), (6, 3), (9, 2)])
def test_dim_grows_after_a_pass_that_improves_nothing_or_repeat_passes_and_stops_below_the_arm(seed, repeat):
    """The cluster size starts at alpha and stays after a pass that improves while fewer than ``repeat`` passes ran
    at it; it grows by delta otherwise. No pass runs at a size of the whole arm or more, and with budget left the last
    pass is the last size below the arm."""
    n, Q, alpha, delta = 16, 2, 4, 3
    Y, _, pool, k, root = _arm(seed, n, Q)
    rows = _seed_design(pool, n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**7, alpha=alpha, delta=delta, repeat=repeat)
    tr = out["trace"]
    assert tr and tr[0][0] == alpha and all(d < n for d, *_ in tr) and len(tr) == out["passes"]
    at = 0
    for (d, _, improved, _), nxt in zip(tr, tr[1:] + [None]):
        at += 1
        grows = improved == 0 or at >= repeat
        if nxt is not None:
            assert nxt[0] == (d + delta if grows else d)
        if grows:
            at = 0
    assert tr[-1][0] + delta >= n and out["nodes"] < 10**7
    assert sum(t[1] for t in tr) == out["subproblems"] and sum(t[2] for t in tr) == out["improved"]
    assert sum(t[3] for t in tr) == out["nodes"]


def test_with_alpha_at_the_arms_size_no_pass_runs():
    n, Q = 10, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    rows = _seed_design(pool, n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=n, delta=2)
    assert out["passes"] == 0 and out["rows"].tolist() == rows.tolist()


def test_improve_is_deterministic():
    n, Q = 14, 2
    Y, _, pool, k, root = _arm(8, n, Q)
    rows = _seed_design(pool, n)
    a = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**5, alpha=4, delta=2)
    b = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**5, alpha=4, delta=2)
    assert a["rows"].tolist() == b["rows"].tolist() and a["nodes"] == b["nodes"]


# --- unit: the wiring in solve_exact ------------------------------------------------------------------------------

def _capped(monkeypatch):
    """A 14-unit Q = 3 arm whose quick search and proof stop at their caps."""
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)


def test_a_capped_proof_runs_popmusic_and_certifies_from_its_design(monkeypatch):
    """After a capped proof, POPMUSIC improves the proof's design with the arm's centred paths and its budget, and
    the certify rounds start from POPMUSIC's design and bound; the diagnostics report its counts."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    _capped(monkeypatch)
    seen, certs = [], []
    real_imp, real_cert = popmusic.improve, columns._certify

    def imp(pl, rt, kk, q, rows, paths, limit, **kw):
        seen.append(dict(rows=np.array(rows), paths=np.array(paths), limit=limit))
        return real_imp(pl, rt, kk, q, rows, paths, limit, **kw)

    def cert(pl, rt, kk, q, z_up, rows, limit, **kw):
        certs.append(dict(z_up=z_up, rows=np.array(rows)))
        return real_cert(pl, rt, kk, q, z_up, rows, limit, **kw)
    monkeypatch.setattr(popmusic, "improve", imp)
    monkeypatch.setattr(columns, "_certify", cert)
    _, d = columns.solve_exact(idx, Y, Q, popmusic_node_limit=777)
    assert len(seen) == 1 and seen[0]["limit"] == 777
    assert np.allclose(seen[0]["paths"], _paths(Y)[idx])
    out = real_imp(pool, root, k, Q, seen[0]["rows"], _paths(Y), 777)
    assert len(certs) == 1 and certs[0]["z_up"] == out["upper"]
    assert sorted(certs[0]["rows"].tolist()) == sorted(out["rows"].tolist())
    assert d["popmusic_nodes"] == out["nodes"] and d["popmusic_passes"] == out["passes"]
    assert d["popmusic_improved"] == out["improved"] and d["popmusic_start"] == float(out["start_upper"])
    assert d["search_nodes"] >= d["popmusic_nodes"] + d["certify_nodes"]


def test_popmusics_trees_count_in_the_search_nodes_and_the_progress_total(monkeypatch):
    """POPMUSIC's nodes are part of ``search_nodes``, which ends equal to the progress array's total."""
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    _capped(monkeypatch)
    monkeypatch.setattr(popmusic, "ALPHA", 4)
    monkeypatch.setattr(popmusic, "DELTA", 2)
    prog = cover.new_progress()
    _, d = columns.solve_exact(idx, Y, Q, progress=prog)
    assert d["popmusic_nodes"] > 0 and d["search_nodes"] == int(prog[cover.P_TOTAL])


def test_a_better_popmusic_design_is_the_one_returned_with_its_certified_gap(monkeypatch):
    """When the certify budget runs out, the design returned is POPMUSIC's and the gap is measured from it."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    _capped(monkeypatch)
    opt = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9).rows
    monkeypatch.setattr(popmusic, "improve", lambda pl, rt, kk, q, rows, paths, limit, **kw: dict(
        rows=np.asarray(opt, dtype=np.int64), upper=_upper(pool, opt), start_upper=_upper(pool, rows), passes=1,
        subproblems=1, improved=1, nodes=5, seconds=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PangeoCertifiedGapWarning)
        chosen, d = columns.solve_exact(idx, Y, Q, certify_node_limit=1)
    assert d["objective_value"] == pytest.approx(float(pool.cost[opt].sum()), rel=1e-12)
    assert d["popmusic_nodes"] == 5


def test_the_rounds_start_from_popmusics_design_not_the_proofs(monkeypatch):
    """The certify rounds receive POPMUSIC's design and bound, whatever the proof held: with POPMUSIC stubbed to
    return the seed partition (a different exact cover), the rounds start from exactly that design and bound."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    _capped(monkeypatch)
    seed_rows = _seed_design(pool, n)
    starts, real_cert = [], columns._certify
    monkeypatch.setattr(popmusic, "improve", lambda pl, rt, kk, q, rows, paths, limit, **kw: starts.append(
        np.array(rows)) or dict(rows=seed_rows, upper=_upper(pool, seed_rows), start_upper=_upper(pool, rows),
                                passes=1, subproblems=1, improved=1, nodes=0, seconds=0.0))
    certs = []
    monkeypatch.setattr(columns, "_certify", lambda pl, rt, kk, q, z_up, rows, limit, **kw: certs.append(
        (z_up, sorted(np.asarray(rows).tolist()))) or real_cert(pl, rt, kk, q, z_up, rows, limit, **kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PangeoCertifiedGapWarning)
        columns.solve_exact(idx, Y, Q, certify_node_limit=1)
    assert starts and sorted(starts[0].tolist()) != sorted(seed_rows.tolist())
    assert certs == [(_upper(pool, seed_rows), sorted(seed_rows.tolist()))]


def test_a_zero_popmusic_budget_skips_it(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    _capped(monkeypatch)
    called = []
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: called.append(1))
    _, d = columns.solve_exact(idx, Y, Q, popmusic_node_limit=0)
    assert called == [] and "popmusic_nodes" not in d and d["certify_rounds"] >= 1


def test_an_arm_that_proves_never_runs_popmusic(monkeypatch):
    n, Q = 12, 2
    Y, idx, *_ = _arm(3, n, Q)
    called = []
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: called.append(1))
    _, d = columns.solve_exact(idx, Y, Q)
    assert d["search"] == "proved" and called == [] and "popmusic_nodes" not in d


def test_a_zero_certify_budget_leaves_the_mip_loop_and_skips_popmusic(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    _capped(monkeypatch)
    called = []
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: called.append(1))
    _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=0)
    assert called == [] and d["mip_solves"] >= 1


def test_the_default_popmusic_budget_is_the_module_constant(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    _capped(monkeypatch)
    seen, real = [], popmusic.improve
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: seen.append(a[6]) or real(*a, **kw))
    columns.solve_exact(idx, Y, Q)
    assert seen == [cover.POPMUSIC_NODE_LIMIT]


def test_a_capped_proof_from_the_first_mips_design_runs_popmusic_too(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(13, n, Q)
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "WINDOW_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    seen, real = [], popmusic.improve
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: seen.append(1) or real(*a, **kw))
    _, d = columns.solve_exact(idx, Y, Q)
    assert seen == [1] and d["search_start"] == "first_mip" and d["popmusic_passes"] >= 0


# --- failures -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 2.5, True, "10"])
def test_an_invalid_popmusic_budget_is_refused_by_name(bad):
    n, Q = 12, 2
    Y, idx, *_ = _arm(3, n, Q)
    with pytest.raises(MlsynthConfigError, match="popmusic_node_limit"):
        columns.solve_exact(idx, Y, Q, popmusic_node_limit=bad)


@pytest.mark.parametrize("kw,name", [(dict(node_limit=-1), "node_limit"), (dict(node_limit=True), "node_limit"),
                                     (dict(alpha=1), "alpha"), (dict(alpha=2.0), "alpha"), (dict(delta=0), "delta"),
                                     (dict(repeat=0), "repeat"), (dict(sub_limit=0), "sub_limit")])
def test_improve_refuses_invalid_settings_by_name(kw, name):
    n, Q = 10, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    args = dict(node_limit=100, alpha=4, delta=2, repeat=1, sub_limit=10)
    args.update(kw)
    limit = args.pop("node_limit")
    with pytest.raises(MlsynthConfigError, match=name):
        popmusic.improve(pool, root, k, Q, _seed_design(pool, n), _paths(Y), limit, **args)


def test_the_config_refuses_a_negative_popmusic_budget():
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel
    from mlsynth.utils.pangeo_helpers.config import PANGEOConfig
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A",), T=20, seed=0)
    with pytest.raises(Exception, match="popmusic_node_limit"):
        PANGEOConfig(df=df, outcome="sales", arm="arm", unitid="unit", time="time", popmusic_node_limit=-1)
    cfg = PANGEOConfig(df=df, outcome="sales", arm="arm", unitid="unit", time="time")
    assert cfg.popmusic_node_limit == cover.POPMUSIC_NODE_LIMIT


def test_the_estimator_passes_its_popmusic_budget_to_the_exact_path(monkeypatch):
    from mlsynth import PANGEO
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, pipeline
    df = make_seasonal_sales_panel(units_per_arm=8, arms=("A",), T=24, seed=0)
    seen, real = [], columns.solve_exact
    monkeypatch.setattr(pipeline, "solve_exact", lambda *a, **kw: seen.append(kw.get("popmusic_node_limit"))
                        or real(*a, **kw))
    PANGEO(dict(df=df, outcome="sales", arm="arm", unitid="unit", time="time", fast=False, max_supergeo_size=2,
                popmusic_node_limit=1234)).fit()
    assert seen and set(seen) == {1234}


# --- properties ---------------------------------------------------------------------------------------------------

@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 14), Q=st.integers(1, 3), budget=st.integers(0, 3000),
       alpha=st.integers(2, 8), delta=st.integers(1, 4), repeat=st.integers(1, 3), sub=st.integers(1, 500),
       pairs=st.integers(1, 7))
def test_property_improve_returns_a_cover_between_the_optimum_and_its_start(seed, n, Q, budget, alpha, delta, repeat,
                                                                            sub, pairs):
    """Property: over arms, sizes, floors and every setting, the design returned is an exact cover with at least k
    groups, its upper bound at least the optimum's and at most the start's, and its nodes within the budget."""
    if not columns.feasible(n, Q, pairs):
        return
    Y, _, pool, k, root = _arm(seed, n, Q, min_pairs=pairs)
    start = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 1).rows
    if start.size == 0:
        start = _seed_design(pool, n)
    if not _is_cover(pool, start, n, k):
        return
    out = popmusic.improve(pool, root, k, Q, start, _paths(Y), budget, alpha=alpha, delta=delta, repeat=repeat,
                           sub_limit=sub)
    assert _is_cover(pool, out["rows"], n, k)
    assert _optimum(pool, root, k, Q) <= out["upper"] <= _upper(pool, start) and out["nodes"] <= budget


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(U=st.lists(st.integers(0, 15), min_size=0, max_size=9, unique=True), Q=st.integers(1, 3))
def test_property_rows_within_matches_the_combinatorial_number_system(U, Q):
    n = 16
    U = sorted(U)
    rows, members = popmusic.rows_within(np.array(U, dtype=np.int64), n, Q)
    want = [g for m in range(2, min(2 * Q, len(U)) + 1) for g in itertools.combinations(U, m)]
    assert rows.tolist() == [columns.column_index(g, n) for g in want]
    assert all([U[p] for p in mem if p >= 0] == list(g) for g, mem in zip(want, members))


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(8, 13), quick=st.integers(1, 60),
       pm=st.sampled_from([0, 1, 50, 10**6]), cert=st.sampled_from([1, 50, 10**7]))
def test_property_solve_exact_brackets_the_optimum_whatever_the_popmusic_budget(seed, n, quick, pm, cert):
    """Property: with POPMUSIC between the capped proof and the rounds, the dual bound stays at or below the optimum
    and the design at or above it, and a proof returns the optimum."""
    Q = 3
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    z = float(_optimum(pool, columns.root_lp(pool, k), k, Q))
    saved = (cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT)
    cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = quick, 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PangeoCertifiedGapWarning)
            _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=cert, popmusic_node_limit=pm)
    finally:
        cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = saved
    assert d["objective_value"] >= z * (1 - 1e-9)
    if d.get("dual_bound") is not None:
        assert d["dual_bound"] <= z * (1 + 1e-9) + 1e-12
    if d["search"] == "proved":
        assert d["objective_value"] == pytest.approx(z, rel=1e-9)

"""The branch and bound refuses a bound that will not fit in memory, before launching it.

Each node bound is a semidefinite program with one block of order ``k + 1`` per
side, ``k`` the markets that side can use, solved by an interior-point method. The
method carries, for a block of order ``s``, a dense scaling block with
``(s (s + 1) / 2)^2`` entries (Goulart and Chen, Clarabel, sec. 2.5), so a bound's
memory grows as the fourth power of the markets. A national search over 150 DMAs
launched a root bound that needed about 15 GB and was killed by the operating
system at 14 GB, after half an hour, with no message: the work limit charges a
bound only once it returns, so it could not stop one that never did.

When a node's two blocks will not fit, the search bounds it on the treated block
alone, plus the certified control floor, about half the memory; only a node whose
one block will not fit is refused.

``lift_bytes`` is that law, calibrated on one root bound measured in fresh
processes at 40 to 80 markets, on one block and on two (the tables below). The search computes it before
every bound and raises, naming the estimate and the remedies, when it exceeds
``MEMORY_SHARE`` of the memory the process may use (``available_memory``: the
tightest of the cgroup limit and the machine's physical memory). Children use
fewer markets than their parent, so in practice the root is the only bound that
can fail, and the search raises before it has done any work.

Levels: smoke and failure through MAREX, unit and edge on the two helpers and
the check, and properties on the scaling law and the decision. The mutants are
under ``marex-branch-memory`` in ``tools/mutation/targets.toml``.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.tests.test_marex_exact_backend import _cfg, _panel
from mlsynth.utils.marex_helpers import branch as branch_mod
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.branch import (MEMORY_SHARE, available_memory,
                                                check_memory, lift_bytes)
from mlsynth.utils.marex_helpers.config import MAREXConfig

MB = 2 ** 20
GB = 2 ** 30

#: Peak memory of one root bound above the process's baseline, Clarabel through
#: CVXPY, measured in a fresh process per size (markets: MB): both blocks, and the
#: treated block alone.
TWO_BLOCKS_MB = {40: 112.0, 50: 219.5, 60: 482.5, 70: 859.2}
ONE_BLOCK_MB = {40: 61.2, 50: 128.5, 60: 219.0, 70: 392.5, 80: 642.3}


def _low_limit(monkeypatch, limit=10):
    monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", limit)


def _costs(monkeypatch, price_seconds, lift_seconds):
    """Run MAREX's branch and bound under these costs: they decide whether a node
    is bounded or priced, never what the search proves."""
    real = branch_mod.branch_and_bound

    def with_costs(*args, **kwargs):
        return real(*args, price_seconds=price_seconds, lift_seconds=lift_seconds, **kwargs)

    monkeypatch.setattr(branch_mod, "branch_and_bound", with_costs)


def _always_bound(monkeypatch):
    _costs(monkeypatch, 1.0, lambda k: 0.0)


def _spy_bounds(monkeypatch):
    calls = []
    real = branch_mod.node_bound

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(branch_mod, "node_bound", spy)
    return calls


# ------------------------------------------------------------- through MAREX
def test_smoke_a_bound_that_fits_runs_as_before(monkeypatch):
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    calls = _spy_bounds(monkeypatch)
    got = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert calls, "the fixture no longer reaches the branch and bound's bounds"
    assert set(map(str, got.assignment["treated"])) == set(map(str, want.assignment["treated"]))


def test_a_bound_that_will_not_fit_is_refused_before_it_is_launched(monkeypatch):
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    calls = _spy_bounds(monkeypatch)
    monkeypatch.setattr(branch_mod, "available_memory", lambda: 1 * MB)
    with pytest.raises(MlsynthConfigError) as err:
        MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert calls == []
    msg = str(err.value)
    for word in ("9 markets", "GB", "cluster"):
        assert word in msg


def test_a_root_whose_two_blocks_will_not_fit_is_searched_on_one(monkeypatch):
    """Memory for one block of nine markets but not two: the search runs, on the
    treated block plus the control floor, and returns the same design."""
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    calls = _spy_bounds(monkeypatch)
    between = (lift_bytes(9) + lift_bytes(9, 9)) / 2 / MEMORY_SHARE
    monkeypatch.setattr(branch_mod, "available_memory", lambda: between)
    got = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert calls, "the search refused instead of bounding on one block"
    assert set(map(str, got.assignment["treated"])) == set(map(str, want.assignment["treated"]))


def _two_forbidden(monkeypatch, memory):
    """A nine-market panel with two markets forbidden, so the root's treated block
    uses 7 markets and its control side 9, searched with ``memory``."""
    df, T0 = _panel(J=9)
    units = sorted(df["unit"].unique(), key=str)
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    monkeypatch.setattr(branch_mod, "available_memory", lambda: memory)
    return MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, not_to_be_treated=units[:2]))


def test_the_one_block_is_estimated_on_the_treated_sides_markets(monkeypatch):
    """Memory between one block of 7 markets and one of 9: the treated side's 7
    fit, so the search runs. Read on the control side's 9, it would refuse."""
    cfg = _two_forbidden(monkeypatch, (lift_bytes(7) + lift_bytes(9)) / 2 / MEMORY_SHARE)
    calls = _spy_bounds(monkeypatch)
    res = MAREX(cfg).fit()
    assert calls and len(res.assignment["treated"]) == 3


def test_the_one_block_counts_the_free_markets(monkeypatch):
    """Memory under one block of 7 markets: refused, though the root forces no
    market in and a count of the forced markets alone would let it run."""
    cfg = _two_forbidden(monkeypatch, (lift_bytes(0) + lift_bytes(7)) / 2 / MEMORY_SHARE)
    calls = _spy_bounds(monkeypatch)
    with pytest.raises(MlsynthConfigError, match="9 markets"):
        MAREX(cfg).fit()
    assert calls == []


def test_a_root_priced_set_by_set_needs_no_memory_for_a_bound(monkeypatch):
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    _low_limit(monkeypatch)
    monkeypatch.setattr(branch_mod, "available_memory", lambda: 1 * MB)
    _costs(monkeypatch, 0.0, lambda k: 1.0)                 # pricing is always cheaper
    calls = _spy_bounds(monkeypatch)
    got = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert calls == []
    assert set(map(str, got.assignment["treated"])) == set(map(str, want.assignment["treated"]))


def test_without_a_known_memory_limit_the_search_runs(monkeypatch):
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    calls = _spy_bounds(monkeypatch)
    monkeypatch.setattr(branch_mod, "available_memory", lambda: None)
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert len(res.assignment["treated"]) == 3 and calls


def test_the_root_is_estimated_on_the_markets_each_side_can_use(monkeypatch):
    """Two markets forbidden and one forced: the treated side can use 7 of 9
    markets and the control side 8."""
    df, T0 = _panel(J=9)
    units = sorted(df["unit"].unique(), key=str)
    _low_limit(monkeypatch)
    _always_bound(monkeypatch)
    seen = []
    real = branch_mod.lift_bytes

    def spy(kt, kc=None):
        seen.append((kt, kc))
        return real(kt, kc)

    monkeypatch.setattr(branch_mod, "lift_bytes", spy)
    MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, not_to_be_treated=units[:2],
                             to_be_treated=units[2:3]))).fit()
    assert seen and seen[0] == (7, 8)
    assert all(kt <= 7 and (kc is None or kc <= 8) for kt, kc in seen)


# ------------------------------------------------------------- the estimate
@pytest.mark.parametrize("k, mb", sorted(TWO_BLOCKS_MB.items()))
def test_the_estimate_matches_the_measured_bounds_on_two_blocks(k, mb):
    assert lift_bytes(k, k) / MB == pytest.approx(mb, rel=0.15)


@pytest.mark.parametrize("k, mb", sorted(ONE_BLOCK_MB.items()))
def test_the_estimate_matches_the_measured_bounds_on_one_block(k, mb):
    assert lift_bytes(k) / MB == pytest.approx(mb, rel=0.15)


def test_the_estimate_covers_the_bound_that_was_killed():
    """150 markets on two blocks: killed at 14 GB in a 15 GB container. On one
    block it needs about half, under that container's share."""
    assert lift_bytes(150, 150) > 14 * GB
    assert 6 * GB < lift_bytes(150) < 0.75 * 15 * GB


def test_the_estimate_reads_both_sides():
    assert lift_bytes(60, 20) == lift_bytes(20, 60)
    assert lift_bytes(60, 20) > lift_bytes(60) > lift_bytes(0)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 400), st.integers(0, 400), st.integers(1, 50))
def test_the_estimate_grows_with_either_side(kt, kc, d):
    assert lift_bytes(kt + d, kc) > lift_bytes(kt, kc)
    assert lift_bytes(kt, kc + d) > lift_bytes(kt, kc)
    assert lift_bytes(kt + d) > lift_bytes(kt)


@settings(max_examples=40, deadline=None)
@given(st.integers(80, 2000))
def test_the_estimate_grows_as_the_fourth_power(k):
    """Doubling the markets multiplies the estimate by close to 16 once the
    fixed part is small next to the block, from about 80 markets."""
    ratio = lift_bytes(2 * k) / lift_bytes(k)
    assert 12 < ratio < 16.5


# ------------------------------------------------------------- the memory available
def _limits(monkeypatch, tmp_path, v2=None, v1=None, physical=64 * GB):
    files = []
    for name, text in (("v2", v2), ("v1", v1)):
        p = tmp_path / name
        if text is not None:
            p.write_text(text + "\n")
        files.append(str(p))
    monkeypatch.setattr(branch_mod, "CGROUP_LIMIT_FILES", tuple(files))
    monkeypatch.setattr(branch_mod, "_physical_memory", lambda: physical)


def test_a_cgroup_v2_limit_is_read(monkeypatch, tmp_path):
    _limits(monkeypatch, tmp_path, v2=str(15 * GB))
    assert available_memory() == 15 * GB


def test_an_unlimited_cgroup_v2_leaves_the_physical_memory(monkeypatch, tmp_path):
    _limits(monkeypatch, tmp_path, v2="max")
    assert available_memory() == 64 * GB


def test_a_cgroup_v1_limit_is_read_and_its_unlimited_value_ignored(monkeypatch, tmp_path):
    _limits(monkeypatch, tmp_path, v1=str(8 * GB))
    assert available_memory() == 8 * GB
    _limits(monkeypatch, tmp_path, v1="9223372036854771712")
    assert available_memory() == 64 * GB


def test_the_tightest_limit_wins(monkeypatch, tmp_path):
    _limits(monkeypatch, tmp_path, v2=str(20 * GB), v1=str(12 * GB), physical=16 * GB)
    assert available_memory() == 12 * GB


def test_without_any_limit_the_memory_is_unknown(monkeypatch, tmp_path):
    _limits(monkeypatch, tmp_path, physical=None)
    assert available_memory() is None


def test_the_physical_memory_is_read_from_the_system():
    phys = branch_mod._physical_memory()
    assert phys is None or phys > 0


# ------------------------------------------------------------- the decision
def test_a_bound_within_the_share_passes_and_one_above_it_is_refused():
    avail = 10 * GB
    check_memory(MEMORY_SHARE * avail, 80, available=avail)
    with pytest.raises(MlsynthConfigError, match="80 markets"):
        check_memory(MEMORY_SHARE * avail + 1, 80, available=avail)


def test_an_unknown_limit_passes_every_bound():
    check_memory(1e30, 80, available=None)


@settings(max_examples=80, deadline=None)
@given(st.floats(1e3, 1e14), st.floats(1e6, 1e13))
def test_a_bound_is_refused_exactly_when_it_exceeds_the_share(need, avail):
    refused = need > MEMORY_SHARE * avail
    try:
        check_memory(need, 100, available=avail)
        assert not refused
    except MlsynthConfigError:
        assert refused

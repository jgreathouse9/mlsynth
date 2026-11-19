"""The treatment floors over the groups as a stream.

certify_treatment built every k-subset of the geos as a Python tuple before the
floors: at 40 geos and 8 treated that is 76,904,685 tuples, about 8.6 GB, before the
chunks were copied to the workers. The floors now take rank ranges of the subsets in
lexicographic order (the order of itertools.combinations); each worker builds its
range a block at a time from the first subset of the range (_unrank), screens the
block, and keeps only the rows that may compete. The rows, their order and the
screened count are the ones the list gave."""
import tracemalloc
from itertools import combinations, islice
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import fastfloor, treatment
from .brute import raw_panel
from .treatment import gram_floor

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _panel(seed, n=40, p=7):
    """A raw panel and its centred copy: the search floors the centred one and reads
    the levels off the raw one."""
    R = raw_panel(seed, n=n, p=p)
    return R, R - R.mean(axis=0)


def _class(R, k, required=True):
    """The screen terms (bar, levels, margin) of the start's design on the raw panel
    R, as the search builds them."""
    Mc = R - R.mean(axis=0)
    T0, C0 = treatment._start(R, k, N_TEST, 5)
    d = treatment._design(Mc, R, T0, C0, N_TEST)
    assert d is not None or not required
    sums = treatment.geo_sums(R)
    bar = None if d is None else treatment.relative_bar(d.rss, treatment.group_level(sums, T0, R.shape[0]), 0)
    return treatment._screen_terms(bar, treatment.level_doubles(sums, R.shape[0]), k)


def _listed(M, k, cls, pool=None, n_jobs=1):
    groups = list(combinations(range(M.shape[1]), k))
    return treatment._screened_floor_rows(M, groups, gram_floor(M), pool, n_jobs, cls)


def _same(a, b):
    (ra, da), (rb, db) = a, b
    assert da == db and len(ra) == len(rb)
    for (fa, ga, Ta), (fb, gb, Tb) in zip(ra, rb):
        assert type(Ta) is tuple and all(type(v) is int for v in Ta)
        assert Ta == Tb and fa == fb and ga == gb


@st.composite
def _pkr(draw, p_max=12):
    p = draw(st.integers(1, p_max))
    k = draw(st.integers(0, p))
    r = draw(st.integers(0, comb(p, k) - 1))
    return p, k, r


# ---------------------------------------------------------------- smoke


def test_smoke_the_streamed_floors_are_the_listed_floors():
    R, M = _panel(5)
    cls = _class(R, 3)
    got = treatment._streamed_floor_rows(M, M.shape[1], 3, gram_floor(M), None, 1, cls)
    _same(got, _listed(M, 3, cls))
    assert got[1] > 0


# ---------------------------------------------------------------- unit: ranks and blocks


@SETTINGS
@given(_pkr())
def test_property_unrank_is_the_rth_combination(pkr):
    p, k, r = pkr
    assert treatment._unrank(r, p, k) == next(islice(combinations(range(p), k), r, None))


@SETTINGS
@given(_pkr(), st.integers(1, 40))
def test_property_a_block_is_the_next_combinations_in_order(pkr, n):
    p, k, r = pkr
    first = np.array(treatment._unrank(r, p, k), dtype=np.int64)
    got = fastfloor.combinations_from(first, n, p)
    want = list(islice(combinations(range(p), k), r, r + n))
    assert got.dtype == np.int64 and got.shape == (len(want), k)
    assert [tuple(int(v) for v in row) for row in got] == want


def test_the_last_block_stops_at_the_last_combination():
    got = fastfloor.combinations_from(np.array([3, 4], dtype=np.int64), 10, 5)
    assert got.tolist() == [[3, 4]]


@SETTINGS
@given(seed=st.integers(0, 50), k=st.integers(2, 3), block=st.integers(1, 30), cut=st.data())
def test_property_a_rank_range_in_blocks_is_its_slice_of_the_list(seed, k, block, cut):
    """Any range [lo, hi) of ranks, any block size: the rows and the count of the
    listed groups in that range, screened as one chunk."""
    R, M = _panel(seed, p=7)
    total = comb(7, k)
    lo = cut.draw(st.integers(0, total))
    hi = cut.draw(st.integers(lo, total))
    cls = None if seed % 5 == 0 else _class(R, k, required=False)
    groups = list(combinations(range(7), k))[lo:hi]
    want = treatment._floor_chunk_screened(M, gram_floor(M), groups, cls)
    got = treatment._floor_range_screened(M, gram_floor(M), 7, k, lo, hi, cls, block=block)
    _same(got, want)


# ---------------------------------------------------------------- unit: the floors


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [2, 3])
def test_the_streamed_floors_over_a_pool_are_the_listed_floors(k, n_jobs):
    R, M = _panel(5)
    cls = _class(R, k)
    with treatment._pool(n_jobs) as pool:
        got = treatment._streamed_floor_rows(M, M.shape[1], k, gram_floor(M), pool, n_jobs, cls)
        want = _listed(M, k, cls, pool, n_jobs)
    _same(got, want)
    _same(got, _listed(M, k, cls))


def test_without_the_kernel_the_streamed_floors_are_the_listed_floors(monkeypatch):
    R, M = _panel(5, p=7)
    cls = _class(R, 2)
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    _same(treatment._streamed_floor_rows(M, 7, 2, gram_floor(M), None, 1, cls), _listed(M, 2, cls))


def test_without_a_gram_floor_the_streamed_floors_are_the_listed_floors():
    R, M = _panel(5, p=7)
    cls = _class(R, 2)
    groups = list(combinations(range(7), 2))
    _same(treatment._streamed_floor_rows(M, 7, 2, None, None, 1, cls),
          treatment._screened_floor_rows(M, groups, None, None, 1, cls))


def test_without_a_screen_every_row_comes_back():
    R, M = _panel(5, p=7)
    rows, dropped = treatment._streamed_floor_rows(M, 7, 3, gram_floor(M), None, 1, None)
    assert dropped == 0 and [T for _, _, T in rows] == list(combinations(range(7), 3))


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_search_over_the_stream_is_the_search_over_the_list(monkeypatch, k, n_jobs):
    R, M = _panel(5)
    streamed = treatment.certify_treatment(R, k, n_test=N_TEST, restarts=5, pass_restarts=3, n_jobs=n_jobs)
    monkeypatch.setattr(treatment, "_streamed_floor_rows", lambda M, p, k, lam, pool, n_jobs, cls:
                        treatment._screened_floor_rows(M, list(combinations(range(p), k)), lam, pool, n_jobs, cls))
    listed = treatment.certify_treatment(R, k, n_test=N_TEST, restarts=5, pass_restarts=3, n_jobs=n_jobs)
    for attr in ("treatment", "controls", "rss", "key", "slope", "certified"):
        assert getattr(streamed, attr) == getattr(listed, attr), attr


def test_the_search_streams_its_floors(monkeypatch):
    """certify_treatment reads its floors from the stream, with the start's screen."""
    R, M = _panel(5)
    seen, real = [], treatment._streamed_floor_rows

    def spy(M_, p, k, lam, pool, n_jobs, cls):
        out = real(M_, p, k, lam, pool, n_jobs, cls)
        seen.append((p, k, cls, out[1]))
        return out

    monkeypatch.setattr(treatment, "_streamed_floor_rows", spy)
    treatment.certify_treatment(R, 3, n_test=N_TEST, restarts=5, pass_restarts=3, n_jobs=1)
    (p, k, cls, dropped), = seen
    want = _class(R, 3)
    assert (p, k) == (M.shape[1], 3) and dropped > 0
    assert cls[0] == want[0] and cls[2] == want[2] and np.array_equal(cls[1], want[1])


def test_the_stream_holds_a_block_not_the_groups(monkeypatch):
    """The groups are never all held at once: with the start's screen most rows drop,
    and with blocks of 512 of the 10,626 groups the peak allocation of the stream is a
    small part of the list's."""
    monkeypatch.setattr(treatment, "FLOOR_BLOCK", 512)
    R, M = _panel(3, n=60, p=24)
    cls, lam = _class(R, 4), gram_floor(M)
    treatment._streamed_floor_rows(M, 24, 4, lam, None, 1, cls)          # compiled before measuring
    tracemalloc.start()
    groups = list(combinations(range(24), 4))
    treatment._screened_floor_rows(M, groups, lam, None, 1, cls)
    listed = tracemalloc.get_traced_memory()[1]
    del groups
    tracemalloc.stop()
    tracemalloc.start()
    rows, dropped = treatment._streamed_floor_rows(M, 24, 4, lam, None, 1, cls)
    streamed = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert dropped > 0.9 * comb(24, 4)
    assert streamed < listed / 4


# ---------------------------------------------------------------- edges


def test_an_empty_rank_range_has_no_rows():
    R, M = _panel(5, p=6)
    assert treatment._floor_range_screened(M, gram_floor(M), 6, 2, 4, 4, None) == ([], 0)


def test_one_group_per_control_count():
    """k = p - 1: as many groups as geos, the smallest pool the search allows."""
    R, M = _panel(5, p=5)
    _same(treatment._streamed_floor_rows(M, 5, 4, gram_floor(M), None, 1, None),
          _listed(M, 4, None))


def test_a_block_larger_than_the_range():
    R, M = _panel(5, p=6)
    _same(treatment._floor_range_screened(M, gram_floor(M), 6, 2, 3, 9, None, block=1000),
          treatment._floor_chunk_screened(M, gram_floor(M), list(combinations(range(6), 2))[3:9], None))


def test_a_chunk_as_an_array_gives_tuple_groups():
    R, M = _panel(5, p=6)
    arr = np.array(list(combinations(range(6), 2)), dtype=np.int64)
    rows = treatment._floor_chunk(M, gram_floor(M), arr)
    assert [T for _, _, T in rows] == list(combinations(range(6), 2))
    assert all(type(v) is int for _, _, T in rows for v in T)


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("r", [-1, comb(6, 2)])
def test_a_rank_outside_the_subsets_is_refused(r):
    with pytest.raises(ValueError, match="rank"):
        treatment._unrank(r, 6, 2)


def test_a_worker_error_in_the_floors_is_reported(monkeypatch):
    import os

    parent, real = os.getpid(), treatment._floor_chunk_screened

    def broken(*a, **k):
        if os.getpid() == parent:
            return real(*a, **k)
        raise ValueError("a floor failed in a worker")

    monkeypatch.setattr(treatment, "_floor_chunk_screened", broken)
    R, M = _panel(5)
    with treatment._pool(2) as pool, pytest.raises(ValueError, match="in a worker"):
        treatment._streamed_floor_rows(M, M.shape[1], 3, gram_floor(M), pool, 2, _class(R, 3))

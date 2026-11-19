"""A group's sizes over workers the treatment search has idle (#965).

On the 100-geo search one group, (4, 27, 33, 58), certified its 47 sizes one after
another on one worker while the other three had nothing left to do. A size is
certified at the incumbent's ratio and nothing else, so its sizes can run at once:
certify_treatment lends its idle workers (slopefree.SPARE), and the size loop runs as
many sizes on threads as it holds, one plus those borrowed. Each runs on a copy of the
search's state and the results are applied in ascending size, as the loop applies
them. A size that started before an earlier size raised the incumbent is applied as it
is when it closed and adopted nothing (a size certified at a ratio stays certified at
a larger one) and is run again at the committed state otherwise, so the certified
design is the sequential loop's. With nothing lent the loop is the sequential one."""
import threading
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import slopefree
from .brute import geo_panel, panel, raw_panel
from .slopefree import certify_slope_free
from .treatment import _split

SETTINGS = settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FIELDS = ("incumbent", "controls", "ratio", "certified", "ceiling_held")


class _Tokens:
    """A lender of ``n`` workers that records what it lent and got back."""

    def __init__(self, n):
        self.n, self.out, self.most, self.lock = n, 0, 0, threading.Lock()

    def take(self, want):
        with self.lock:
            got = max(0, min(want, self.n - self.out))
            self.out += got
            self.most = max(self.most, self.out)
            return got

    def give(self, n):
        with self.lock:
            self.out -= n
            assert self.out >= 0


def _lend(monkeypatch, n):
    t = _Tokens(n)
    monkeypatch.setattr(slopefree, "SPARE", (t.take, t.give))
    return t


def _group(seed=4, p=14, T=(0, 3)):
    y, X = _split(geo_panel(seed, n=24, p=p), list(T))[:2]
    return y, X


def _same(a, b):
    for f in FIELDS:
        assert getattr(a, f) == getattr(b, f), f
    assert not a.open and not b.open


def _overlap(monkeypatch):
    """Record, around every size's branch and bound, how many run at once."""
    now, most, lock, real = [0], [0], threading.Lock(), slopefree._branch

    def branch(c, G, out, m, cuts, fixes, *a, **k):
        if not fixes:
            with lock:
                now[0] += 1
                most[0] = max(most[0], now[0])
            time.sleep(0.02)
        try:
            return real(c, G, out, m, cuts, fixes, *a, **k)
        finally:
            if not fixes:
                with lock:
                    now[0] -= 1

    monkeypatch.setattr(slopefree, "_branch", branch)
    return most


# ---------------------------------------------------------------- smoke


def test_smoke_lent_workers_certify_the_sequential_design(monkeypatch):
    y, X = _group()
    one = certify_slope_free(y, X, restarts=5)
    t = _lend(monkeypatch, 3)
    many = certify_slope_free(y, X, restarts=5)
    _same(one, many)
    assert t.out == 0 and t.most >= 1


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 400), n=st.integers(1, 3))
def test_property_any_number_of_lent_workers_certifies_the_sequential_design(seed, n):
    y, X = _group(seed)
    one = certify_slope_free(y, X, restarts=5)
    with pytest.MonkeyPatch.context() as mp:
        t = _lend(mp, n)
        many = certify_slope_free(y, X, restarts=5)
    _same(one, many)
    assert t.out == 0


# ---------------------------------------------------------------- the window


def test_sizes_run_at_once_when_workers_are_lent(monkeypatch):
    y, X = _group()
    _lend(monkeypatch, 3)
    most = _overlap(monkeypatch)
    certify_slope_free(y, X, restarts=5, hull=False)
    assert most[0] >= 2


def test_nothing_lent_runs_one_size_at_a_time(monkeypatch):
    y, X = _group()
    most = _overlap(monkeypatch)
    certify_slope_free(y, X, restarts=5, hull=False)
    assert most[0] == 1


def test_a_lender_with_nothing_free_runs_one_size_at_a_time(monkeypatch):
    y, X = _group()
    _lend(monkeypatch, 0)
    most = _overlap(monkeypatch)
    certify_slope_free(y, X, restarts=5, hull=False)
    assert most[0] == 1


def test_at_most_one_plus_the_lent_workers_run(monkeypatch):
    y, X = _group()
    t = _lend(monkeypatch, 2)
    most = _overlap(monkeypatch)
    certify_slope_free(y, X, restarts=5, hull=False)
    assert 2 <= most[0] <= 3 and t.most <= 2


def test_sizes_are_applied_in_ascending_order(monkeypatch):
    y, X = _group()
    _lend(monkeypatch, 3)
    out = certify_slope_free(y, X, restarts=5, hull=False)
    assert list(out.sizes) == sorted(out.sizes) and len(out.sizes) == X.shape[1]


def _poor_start(monkeypatch, y, X):
    """Start the search from the worst positive-slope single control, so sizes adopt."""
    from . import search
    from .certificate import ExactData
    D = ExactData(y, X)
    ratios = [(slopefree._ratio(D.c, D.G, {j})[1], j) for j in range(X.shape[1])
              if slopefree._ratio(D.c, D.G, {j})[0] > 0]
    poor = min(ratios)[1]
    monkeypatch.setattr(search, "incumbent", lambda y_, X_, restarts=0, seed=0, **kw: (None, [poor]))
    return poor


@pytest.mark.parametrize("seed", [4, 7, 11])
def test_sizes_that_adopt_certify_the_sequential_design(monkeypatch, seed):
    """From the worst single control, sizes adopt better designs as they run; the
    adoptions are committed in order and the certificate is the sequential one."""
    y, X = _group(seed)
    poor = _poor_start(monkeypatch, y, X)
    seq = certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False)
    assert seq.controls != [poor]                    # the sequential loop adopted
    _lend(monkeypatch, 3)
    par = certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False)
    _same(seq, par)


@pytest.mark.parametrize("seed", [6, 9])
def test_stale_sizes_that_leave_nodes_open_are_run_again(monkeypatch, seed):
    """Twenty-four controls, the depth limit at the root, the worst single control as
    the start and size 1 made slow, so the sizes after it start at the poor ratio. Some
    leave their root open there that the ratio size 1 commits closes. They are run
    again at the committed state, so the group certifies as the sequential loop does;
    applied as they ran they left one root open on each of these seeds (and on 7 of the
    first 25 seeds of this panel)."""
    y, X = _split(geo_panel(seed, n=24, p=26), [0, 5])[:2]
    _poor_start(monkeypatch, y, X)
    kw = dict(restarts=1, deep_restarts=0, hull=False, max_depth=0)
    seq = certify_slope_free(y, X, **kw)
    assert seq.certified and not seq.open
    real = slopefree._branch

    def branch(c, G, out, m, cuts, fixes, *a, **k):
        if not fixes and m == 1:
            time.sleep(0.3)
        return real(c, G, out, m, cuts, fixes, *a, **k)

    monkeypatch.setattr(slopefree, "_branch", branch)
    _lend(monkeypatch, 3)
    par = certify_slope_free(y, X, **kw)
    _same(seq, par)


def test_each_size_gets_its_own_workspaces(monkeypatch):
    """A size's thread solves on workspaces of its own: the search's one would have
    its spaces and pool rewritten by the other threads."""
    from .workspace import Workspaces
    made, real = [], Workspaces.__init__
    monkeypatch.setattr(Workspaces, "__init__", lambda self: made.append(1) or real(self))
    y, X = panel(5, n=20, p=10)
    _lend(monkeypatch, 2)
    out = certify_slope_free(y, X, restarts=5, solver="workspace", hull=False)
    assert len(made) >= 1 + len(out.sizes)


@SETTINGS
@given(n=st.integers(0, 4), k=st.integers(0, 12), seed=st.integers(0, 1000))
def test_property_the_window_commits_in_order_and_gives_every_worker_back(n, k, seed):
    """_size_window on its own: every item committed once, in order, with its own
    result, never more than one plus the lent workers at once, and every lent worker
    given back."""
    import random
    rng = random.Random(seed)
    t = _Tokens(n)
    now, most, lock, got = [0], [0], threading.Lock(), []

    def start(m):
        with lock:
            now[0] += 1
            most[0] = max(most[0], now[0])
        time.sleep(rng.choice([0.0, 0.002, 0.005]))
        with lock:
            now[0] -= 1
        return m * m

    slopefree._size_window(list(range(k)), start, lambda m, r: got.append((m, r)), t.take, t.give)
    assert got == [(m, m * m) for m in range(k)]
    assert most[0] <= 1 + n and t.out == 0


# ---------------------------------------------------------------- edges


def test_one_size():
    y, X = _group(p=3, T=(0, 1))
    with pytest.MonkeyPatch.context() as mp:
        _lend(mp, 3)
        many = certify_slope_free(y, X, restarts=5)
    _same(certify_slope_free(y, X, restarts=5), many)


def test_a_workspace_search_with_lent_workers(monkeypatch):
    y, X = panel(5, n=20, p=10)
    one = certify_slope_free(y, X, restarts=5, solver="workspace")
    _lend(monkeypatch, 2)
    _same(one, certify_slope_free(y, X, restarts=5, solver="workspace"))


# ---------------------------------------------------------------- failures


def test_an_error_in_a_size_is_reported_and_the_workers_given_back(monkeypatch):
    y, X = _group()
    t = _lend(monkeypatch, 3)
    real = slopefree._branch

    def branch(c, G, out, m, cuts, fixes, *a, **k):
        if m == 3 and not fixes:
            raise RuntimeError("size 3 failed")
        return real(c, G, out, m, cuts, fixes, *a, **k)

    monkeypatch.setattr(slopefree, "_branch", branch)
    with pytest.raises(RuntimeError, match="size 3"):
        certify_slope_free(y, X, restarts=5, hull=False)
    assert t.out == 0


# ---------------------------------------------------------------- the treatment search lends


def test_the_lender_lends_only_what_is_free():
    from .treatment import _Lender
    lend = _Lender()
    assert lend.take(3) == 0                         # nothing free yet
    lend.set_free(2)
    assert lend.take(3) == 2 and lend.take(1) == 0
    lend.give(1)
    assert lend.take(1) == 1
    lend.set_free(0)                                 # the window filled up again
    assert lend.take(1) == 0
    lend.give(2)
    assert lend.lent.value == 0


def test_the_window_reports_how_many_run():
    from concurrent.futures import ThreadPoolExecutor
    from . import treatment
    seen = []
    with ThreadPoolExecutor(4) as pool:
        treatment._window(pool, 2, list(range(5)), lambda r: True, lambda r: (time.sleep, (0.005,), None),
                          lambda r, ctx, res: None, occupancy=seen.append)
    assert seen and max(seen) <= 2 and seen[-1] == 0


def test_the_treatment_search_lends_its_workers_and_takes_them_back(monkeypatch, tmp_path):
    """Inside the workers the size loop holds the search's lender; after the search the
    default is back."""
    from . import treatment
    real = treatment._optimum

    def spy(Mc, M, T, *a, **k):
        (tmp_path / f"{'-'.join(map(str, T))}").write_text(str(slopefree.SPARE[0] is not slopefree._lend_none))
        return real(Mc, M, T, *a, **k)

    monkeypatch.setattr(treatment, "_optimum", spy)
    M = raw_panel(5, p=6)
    one = treatment.certify_treatment(M, 2, n_test=8, restarts=5, blank_periods=0)
    two = treatment.certify_treatment(M, 2, n_test=8, restarts=5, n_jobs=2, blank_periods=0)
    assert {f.read_text() for f in tmp_path.iterdir()} >= {"True"}
    assert slopefree.SPARE[0] is slopefree._lend_none
    assert (one.treatment, one.controls, one.key) == (two.treatment, two.controls, two.key)


def test_the_search_tells_the_lender_what_is_free(monkeypatch):
    """The window reports its idle workers to the lender, every one of them once the
    groups are done."""
    from . import treatment
    seen, real = [], treatment._Lender.set_free
    monkeypatch.setattr(treatment._Lender, "set_free", lambda self, n: seen.append(n) or real(self, n))
    treatment.certify_treatment(raw_panel(5, p=6), 2, n_test=8, restarts=5, n_jobs=3, blank_periods=0)
    assert all(0 <= n <= 3 for n in seen)
    assert seen[-2:] == [3, 2]                       # every worker free at the window's end; two lent to the retries

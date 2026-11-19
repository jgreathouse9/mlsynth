"""The certifications over a sliding window (#1069): _window and certify_treatment(n_jobs=).

Up to n_jobs groups run at once and the next admitted group is submitted as soon as
a worker is free; the screen and the ceiling are read at submission, against every
result committed by then, and results are committed in queue order. The waves this
replaces filled the next n_jobs only once the slowest of the last had finished, so
one slow group held every worker (on an 80-geo panel at n_jobs = 4, 54% busy).

The window's own invariants are asserted on a thread pool with more workers than
n_jobs, so that the bound on what is in flight is the window's and not the pool's."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import treatment
from .brute import raw_panel
from .treatment import certify_treatment

N_TEST = 8
SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class _Run:
    """A window over rows 0..n-1: row r sleeps durations[r] in a worker and returns
    r * r; admit(r) is a function of r and of the rows committed so far. Records what
    each call saw, the most jobs in flight at once and when each job ran."""

    def __init__(self, durations, rule):
        self.durations, self.rule = durations, rule
        self.committed, self.refused, self.seen_at_admit = [], [], {}
        self.lock, self.now, self.most = threading.Lock(), 0, 0
        self.span = {}

    def work(self, r):
        with self.lock:
            self.now += 1
            self.most = max(self.most, self.now)
        t0 = time.perf_counter()
        time.sleep(self.durations[r])
        with self.lock:
            self.now -= 1
            self.span[r] = (t0, time.perf_counter())
        return r * r

    def admit(self, r):
        self.seen_at_admit[r] = tuple(self.committed)
        ok = self.rule(r, self.committed)
        if not ok:
            self.refused.append(r)
        return ok

    def job(self, r):
        return self.work, (r,), len(self.committed)

    def commit(self, r, ctx, result):
        assert result == r * r
        assert ctx <= len(self.committed)          # the context was read at submission
        self.committed.append(r)

    def run(self, pool, n_jobs):
        treatment._window(pool, n_jobs, list(range(len(self.durations))), self.admit, self.job, self.commit)
        return self


def _sequential(durations, rule):
    """The loop the window has to agree with when nothing runs in parallel."""
    committed = []
    for r in range(len(durations)):
        if rule(r, committed):
            committed.append(r)
    return committed


def _parity(r, committed):
    """A screen that reads the committed state: refuse r when it has the parity of
    how many rows are committed and r is a multiple of three."""
    return not (r % 3 == 0 and r % 2 == len(committed) % 2)


# ---------------------------------------------------------------- smoke


def test_smoke_the_window_commits_every_admitted_row_in_order():
    with ThreadPoolExecutor(max_workers=8) as pool:
        run = _Run([0.01, 0.0, 0.02, 0.0, 0.01], lambda r, c: True).run(pool, 2)
    assert run.committed == [0, 1, 2, 3, 4] and run.refused == []


def test_smoke_two_processes_certify_the_sequential_design_over_the_window():
    M = raw_panel(5, p=6)
    one = certify_treatment(M, 2, n_test=N_TEST, restarts=5, blank_periods=0)
    two = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2, blank_periods=0)
    assert two.certified and (two.treatment, two.controls, two.key) == (one.treatment, one.controls, one.key)


# ---------------------------------------------------------------- the barrier is gone


def test_a_slow_head_does_not_hold_the_other_workers():
    """Row 0 takes 0.4 s and the rest 0.01 s. With two jobs, every later row runs on
    the second worker while row 0 is still running; under waves only row 1 would."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        run = _Run([0.4] + [0.01] * 6, lambda r, c: True).run(pool, 2)
    end0 = run.span[0][1]
    assert all(run.span[r][0] < end0 for r in range(1, 7))
    assert run.committed == list(range(7))


def test_a_slow_group_does_not_hold_the_search(monkeypatch, tmp_path):
    """The first group certified is made to take a second. Under waves of two only its
    partner in the wave starts before it ends; over the window the second worker goes
    on to the groups after it. Each worker is its own process, so starts and ends are
    written to files."""
    order = treatment._bound_order               # worst bound first: many groups survive, whatever the timing
    monkeypatch.setattr(treatment, "_bound_order", lambda r, ell: -order(r, ell))
    M = raw_panel(0, p=7)
    kw = dict(n_test=N_TEST, restarts=5, pass_restarts=0, start=[0, 1])
    plain = certify_treatment(M, 2, **kw, blank_periods=0)
    assert len(plain.survivors) >= 3
    slow, real = plain.survivors[0]["treatment"], treatment._optimum

    def timed(Mc, M_, T, *a, **k):
        t0 = time.time()
        if sorted(T) == slow:
            time.sleep(1.0)
        res = real(Mc, M_, T, *a, **k)
        (tmp_path / "-".join(map(str, sorted(T)))).write_text(f"{t0} {time.time()}")
        return res

    monkeypatch.setattr(treatment, "_optimum", timed)       # read inside the forked workers
    out = certify_treatment(M, 2, n_jobs=2, **kw, blank_periods=0)
    spans = {f.name: tuple(map(float, f.read_text().split())) for f in tmp_path.iterdir()}
    end = spans.pop("-".join(map(str, slow)))[1]
    assert sum(t0 < end for t0, _ in spans.values()) >= 2
    assert (out.treatment, out.controls, out.key) == (plain.treatment, plain.controls, plain.key)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(durations=st.lists(st.sampled_from([0.0, 0.001, 0.004]), min_size=0, max_size=14),
       n_jobs=st.integers(1, 4))
def test_property_the_window_keeps_its_invariants(durations, n_jobs):
    with ThreadPoolExecutor(max_workers=16) as pool:
        run = _Run(durations, _parity).run(pool, n_jobs)
    rows = set(range(len(durations)))
    assert run.committed == sorted(run.committed)                    # in queue order
    assert set(run.committed) | set(run.refused) == rows              # each row decided
    assert not set(run.committed) & set(run.refused) and len(run.committed) + len(run.refused) == len(rows)
    assert run.most <= n_jobs                                         # never more in flight
    for r, seen in run.seen_at_admit.items():                        # admit reads a committed prefix
        assert list(seen) == run.committed[:len(seen)] and all(s < r for s in seen)


@SETTINGS
@given(durations=st.lists(st.sampled_from([0.0, 0.001, 0.003]), min_size=0, max_size=14),
       n_jobs=st.integers(1, 4))
def test_property_a_stateless_screen_commits_what_the_sequential_loop_does(durations, n_jobs):
    rule = lambda r, c: r % 4 != 1                                    # noqa: E731
    with ThreadPoolExecutor(max_workers=16) as pool:
        run = _Run(durations, rule).run(pool, n_jobs)
    assert run.committed == _sequential(durations, rule)


@SETTINGS
@given(durations=st.lists(st.just(0.0), min_size=0, max_size=14))
def test_property_without_a_pool_the_window_is_the_sequential_loop(durations):
    run = _Run(durations, _parity).run(None, 3)
    assert run.committed == _sequential(durations, _parity)
    assert run.most <= 1


# ---------------------------------------------------------------- edges


def test_no_rows():
    with ThreadPoolExecutor(max_workers=2) as pool:
        run = _Run([], lambda r, c: True).run(pool, 2)
    assert run.committed == [] and run.refused == []


def test_every_row_refused_submits_nothing():
    with ThreadPoolExecutor(max_workers=2) as pool:
        run = _Run([0.0] * 5, lambda r, c: False).run(pool, 2)
    assert run.committed == [] and run.refused == list(range(5)) and run.span == {}


def test_more_jobs_than_rows():
    with ThreadPoolExecutor(max_workers=16) as pool:
        run = _Run([0.002, 0.0, 0.001], lambda r, c: True).run(pool, 8)
    assert run.committed == [0, 1, 2] and run.most <= 3


def test_one_job_runs_one_at_a_time():
    with ThreadPoolExecutor(max_workers=8) as pool:
        run = _Run([0.003, 0.0, 0.002, 0.0], _parity).run(pool, 1)
    assert run.most == 1 and run.committed == _sequential([0] * 4, _parity)


# ---------------------------------------------------------------- failures


def test_a_worker_error_is_reported_and_not_committed():
    """Row 2 raises. The error reaches the caller; rows 0 and 1 are committed, and no
    row after 2 is, though they may have finished: a result waits for the rows ahead."""
    run = _Run([0.0, 0.0, 0.0, 0.0, 0.0], lambda r, c: True)
    real = run.work

    def work(r):
        if r == 2:
            time.sleep(0.02)
            raise ValueError("row 2 failed in a worker")
        return real(r)

    run.work = work
    with ThreadPoolExecutor(max_workers=8) as pool, pytest.raises(ValueError, match="row 2"):
        run.run(pool, 3)
    assert run.committed == [0, 1]


def test_an_error_in_the_screen_is_reported():
    def rule(r, c):
        if r == 3:
            raise ValueError("the screen failed")
        return True

    with ThreadPoolExecutor(max_workers=4) as pool, pytest.raises(ValueError, match="screen failed"):
        _Run([0.0] * 5, rule).run(pool, 2)

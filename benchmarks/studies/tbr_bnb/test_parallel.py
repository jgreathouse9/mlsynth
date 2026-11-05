"""The treatment search over several processes (#965): certify_treatment(n_jobs=).

The search runs in waves of n_jobs groups. A wave's groups are screened and given
their ceilings from the incumbent as the wave starts, and their results are applied
in queue order. So the certified design is the same at every n_jobs, ties included,
and n_jobs = 1 is the sequential search itself."""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import treatment
from .brute import geo_panel, treatment_optimum
from .treatment import certify_treatment

N_TEST = 8
SETTINGS = settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _answer(out):
    return (out.treatment, out.controls, out.key, out.rss, out.passing, out.certified)


# ---------------------------------------------------------------- smoke


def test_smoke_two_processes_certify_the_sequential_design():
    M = geo_panel(5, p=6)
    one = certify_treatment(M, 2, n_test=N_TEST, restarts=5)
    two = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2)
    assert two.certified and _answer(two) == _answer(one)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 300), k=st.integers(1, 2), jobs=st.integers(2, 3))
def test_property_every_n_jobs_certifies_the_same_design(seed, k, jobs):
    M = geo_panel(seed, p=6)
    one = certify_treatment(M, k, n_test=N_TEST, restarts=5)
    many = certify_treatment(M, k, n_test=N_TEST, restarts=5, n_jobs=jobs)
    assert _answer(many) == _answer(one)


@pytest.mark.parametrize("seed", [1, 7])
def test_two_processes_reach_the_key_optimum(seed):
    M = geo_panel(seed, p=6)
    key, T, C = treatment_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2)
    assert out.certified and out.key == key and set(out.treatment) == set(T)


# ---------------------------------------------------------------- the waves


def test_every_ceiling_holds_against_the_final_incumbent():
    """A survivor certified in a wave holds no design below its ceiling, and its
    ceiling is at least the final incumbent's for it: the incumbent only improves."""
    M = geo_panel(5, p=6)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=3, pass_restarts=0)
    final = (None, None) if out.key is None else (treatment.Q(out.key[4]), out.rss)
    for s in out.survivors:
        if s.get("ceiling") is not None:
            yy = treatment._yy(M, s["treatment"])
            assert s["ceiling"] >= treatment.ceiling(yy, *final)


def test_every_group_is_screened_or_certified_in_parallel():
    M = geo_panel(4, p=6)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, n_jobs=2)
    assert out.screened_out + len(out.survivors) == out.groups


def test_a_tie_in_a_wave_keeps_the_first_design(monkeypatch):
    """A group in a wave whose optimum has exactly the incumbent's key does not
    replace it: ties go to the first design in queue order. Exact ties between
    groups need identical data, so one group's result is given the incumbent's key."""
    M = geo_panel(5, p=6)
    plain = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0)
    assert plain.passing and len(plain.survivors) >= 2
    target, real = plain.survivors[1]["treatment"], treatment._optimum

    def tied(M_, T, *a, **k):
        res, C = real(M_, T, *a, **k)
        if sorted(T) == target:
            res.passing, res.key = True, plain.key
            C = [j for j in range(M_.shape[1]) if j not in T][:1]
        return res, C

    monkeypatch.setattr(treatment, "_optimum", tied)       # read inside the forked workers
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0, n_jobs=2)
    assert out.replaced == 0
    assert (out.treatment, out.controls, out.key) == (plain.treatment, plain.controls, plain.key)


def test_the_floors_are_the_same_in_chunks():
    M = geo_panel(2, p=7)
    lam = treatment.gram_floor(M)
    groups = [T for T in combinations(range(7), 3) if list(T) != [0, 1, 2]]
    one = treatment._floor_rows(M, groups, lam, None, 1)
    with treatment._pool(3) as pool:
        three = treatment._floor_rows(M, groups, lam, pool, 3)
    assert three == one and len(one) == len(groups)


# ---------------------------------------------------------------- edges and failures


def test_one_job_starts_no_pool(monkeypatch):
    monkeypatch.setattr(treatment, "_pool", lambda n: pytest.fail("a pool was started for one job"))
    out = certify_treatment(geo_panel(3, p=5), 1, n_test=N_TEST, restarts=5, n_jobs=1)
    assert out.certified


def test_more_jobs_than_groups():
    M = geo_panel(3, p=4)
    one = certify_treatment(M, 1, n_test=N_TEST, restarts=5)
    out = certify_treatment(M, 1, n_test=N_TEST, restarts=5, n_jobs=8)
    assert _answer(out) == _answer(one)


@pytest.mark.parametrize("bad", [0, -2])
def test_a_non_positive_n_jobs_is_refused(bad):
    with pytest.raises(MlsynthDataError, match="n_jobs"):
        certify_treatment(geo_panel(3, p=5), 1, n_test=N_TEST, n_jobs=bad)


def test_a_worker_failure_is_reported(monkeypatch):
    """An error in a worker reaches the caller; it is not read as a group that
    closed or a group left open."""
    import os

    parent, real = os.getpid(), treatment._optimum

    def broken(*a, **k):
        if os.getpid() == parent:                          # the start group is certified here
            return real(*a, **k)
        raise ValueError("a certification failed in a worker")

    monkeypatch.setattr(treatment, "_optimum", broken)     # read inside the forked worker
    with pytest.raises(ValueError, match="in a worker"):
        certify_treatment(geo_panel(5, p=6), 2, n_test=N_TEST, restarts=5, n_jobs=2)


def test_a_group_left_open_in_a_wave_is_retried(monkeypatch, tmp_path):
    """A survivor whose certificate does not close in its wave is certified again
    against the final ceiling, as in the sequential search. The first survivor after
    a poor start is cut short once; a file marks it, since each worker is its own
    process."""
    from .test_treatment import _worst_passing

    M = geo_panel(5)
    start = _worst_passing(M)
    plain = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    target = plain.survivors[1]["treatment"]
    real, mark = treatment._optimum, tmp_path / "cut"

    def flaky(M_, T, *a, **k):
        res, C = real(M_, T, *a, **k)
        if sorted(T) == target and not mark.exists():
            mark.write_text("cut")
            res.certified, res.passing = False, False
        return res, C

    monkeypatch.setattr(treatment, "_optimum", flaky)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0, n_jobs=2)
    assert mark.exists() and out.retried >= 1
    assert _answer(out) == _answer(plain)


# ---------------------------------------------------------------- Clarabel after a fork


def _recorded_size():
    """A relaxation of the size the four-geo GeoLift search solves (36 controls)."""
    from . import slopefree as sf
    from .certificate import ExactData

    M = geo_panel(11, n=90, p=37)
    D = ExactData(M[:, 0], M[:, 1:])
    lam = sf.dyadic_below(D.yy / 2)
    return D.c, D.G, lam, sf.Rows(D.c, 10)


def _solve_status(c, G, lam, R):
    from . import slopefree as sf
    return sf.solve(c, G, lam, R).status


def test_the_relaxation_is_solved_on_one_thread(monkeypatch):
    """Clarabel's automatic threading (max_threads = 0) starts a native thread pool
    on a solve of this size, and a worker forked afterwards inherits the pool's
    state but not its threads: its next solve never returns (the four-geo GeoLift
    search at n_jobs = 4 hung in its first certification wave that reached
    Clarabel). The study's parallelism is its processes, so Clarabel runs on one."""
    import cvxpy as cp

    seen = []
    real = cp.Problem.solve
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: seen.append(k) or real(self, *a, **k))
    _solve_status(*_recorded_size())
    assert seen and all(k.get("max_threads") == 1 for k in seen)


def test_a_forked_worker_solves_after_the_parent_has():
    import os
    from concurrent.futures import TimeoutError as Timeout

    args = _recorded_size()
    _solve_status(*args)                     # the parent solves first, as the start group does
    pool = treatment._pool(1)
    try:
        assert pool.submit(_solve_status, *args).result(timeout=120) is not None
    except Timeout:                          # pragma: no cover - reached only while the hang is back
        for p in list(pool._processes):
            os.kill(p, 9)
        pytest.fail("a worker forked after the parent's solve never returned")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

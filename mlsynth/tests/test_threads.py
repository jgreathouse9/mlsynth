"""One BLAS and OpenMP thread per pytest-xdist worker (mlsynth/tests/_threads.py).

Under ``pytest -n auto`` every worker process starts its own BLAS thread pool the
size of the machine, so four workers on four cores run sixteen threads. Small
eigendecompositions and products then spend their time contending: one TBR test
took 117 s with the default pools and 8 s with one thread. The suite's root
conftest calls ``limit_threads`` once a worker starts."""
import os

import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from mlsynth.tests._threads import ENV_VARS, limit_threads


def _blas():
    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


@pytest.fixture
def restore():
    before = dict(os.environ)
    with threadpool_limits(None):          # restores the pools' sizes on exit
        yield
    os.environ.clear(); os.environ.update(before)


# ---------------------------------------------------------------- smoke / unit


def test_a_worker_gets_one_thread(restore):
    env = {"PYTEST_XDIST_WORKER": "gw0"}
    assert limit_threads(env) is True
    assert _blas() and set(_blas()) == {1}
    assert all(env[v] == "1" for v in ENV_VARS)


def test_outside_xdist_nothing_changes(restore):
    with threadpool_limits(2, user_api="blas"):
        before, env = _blas(), {}
        assert limit_threads(env) is False
        assert _blas() == before and env == {}


def test_a_value_already_set_is_kept(restore):
    """An explicit OMP_NUM_THREADS from the caller is theirs: only unset ones are filled."""
    env = {"PYTEST_XDIST_WORKER": "gw1", "OMP_NUM_THREADS": "3"}
    limit_threads(env)
    assert env["OMP_NUM_THREADS"] == "3" and env["OPENBLAS_NUM_THREADS"] == "1"


def test_the_environment_variables_cover_the_common_runtimes():
    assert {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"} <= set(ENV_VARS)


# ---------------------------------------------------------------- the suite itself


def test_this_worker_runs_one_blas_thread():
    """The root conftest applied it: in a worker every BLAS pool has one thread."""
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("only meaningful inside a pytest-xdist worker")
    assert set(_blas()) == {1}

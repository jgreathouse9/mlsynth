"""One BLAS and OpenMP thread in each pytest-xdist worker.

``pytest -n auto`` starts one worker per core, and each worker's BLAS and OpenMP
runtimes start a pool the size of the machine, so four workers on four cores run
sixteen threads. The suite's linear algebra is mostly small (eigendecompositions
of 30 x 30 to 100 x 100 matrices, sparse products), where a pool's threads spend
their time contending: one TBR test took 117 s with the default pools and 8 s
with one thread. The benchmarks workflow already sets these variables; this does
the same for every xdist run, in CI and locally.

``limit_threads`` sets the limit in the worker itself through threadpoolctl (the
runtimes are already loaded when the conftest runs, so environment variables
alone come too late) and fills the environment variables that are unset, so a
process the tests start inherits them. A variable the caller set is kept. Outside
xdist it does nothing: a single pytest process may use the whole machine."""
from __future__ import annotations

import os
from typing import MutableMapping, Optional

#: The thread-count variables of the runtimes numpy, scipy and scikit-learn load.
ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads(env: Optional[MutableMapping[str, str]] = None) -> bool:
    """In an xdist worker (``PYTEST_XDIST_WORKER`` set in ``env``), one thread per
    runtime; returns whether it applied."""
    env = os.environ if env is None else env
    if not env.get("PYTEST_XDIST_WORKER"):
        return False
    for var in ENV_VARS:
        env.setdefault(var, "1")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    return True

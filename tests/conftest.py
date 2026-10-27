import sys
from pathlib import Path as _P

import numpy as np
import pytest
from hypothesis import strategies as st

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from gppmta import GPPParams, Path  # noqa: E402


def random_params(rng, p, q, density=0.6, scale=0.6):
    alpha = rng.uniform(0, scale, (p, q)) * (rng.random((p, q)) < density)
    mu = rng.uniform(0.0, 0.3, q)
    decay = rng.uniform(0.2, 3.0, p)
    return GPPParams(mu=mu, alpha=alpha, decay=decay)


def random_path(rng, p, m, T=10.0, end_with_conversion=True):
    times = np.sort(rng.uniform(0, T, m))
    times = np.unique(times)
    types = rng.integers(1, p, times.size)
    if end_with_conversion:
        types[-1] = 0
    return Path(times, types, T)


@st.composite
def model_and_path(draw):
    """Random parameters and a path ending in a conversion."""
    seed = draw(st.integers(0, 2**32 - 1))
    p = draw(st.integers(3, 6))
    q = draw(st.integers(1, p - 1))
    m = draw(st.integers(2, 14))
    rng = np.random.default_rng(seed)
    params = random_params(rng, p, q)
    path = random_path(rng, p, m)
    # Customer-initiated types with zero intensity cannot occur; keep mu > 0.
    params = GPPParams(np.maximum(params.mu, 1e-3), params.alpha, params.decay)
    return params, path, rng


@pytest.fixture
def rng():
    return np.random.default_rng(12345)

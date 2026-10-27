import numpy as np
import pytest
from scipy.integrate import quad

from gppmta import Path, sufficient_stats


def features(path, p, beta, t):
    """X(t) = (1, X_1..X_p) by direct summation over earlier events."""
    x = np.zeros(p + 1)
    x[0] = 1.0
    for tl, el in zip(path.times, path.types):
        if tl < t:
            x[el + 1] += beta[el] * np.exp(-beta[el] * (t - tl))
    return x


def brute_stats(paths, p, beta):
    G = np.zeros((p + 1, p + 1))
    B = np.zeros((p, p + 1))
    for path in paths:
        brk = [0.0, *path.times, path.T]
        for a in range(p + 1):
            for b in range(a, p + 1):
                val = sum(
                    quad(lambda t: features(path, p, beta, t)[a] * features(path, p, beta, t)[b],
                         lo, hi, epsabs=1e-13, epsrel=1e-12)[0]
                    for lo, hi in zip(brk[:-1], brk[1:]) if hi > lo
                )
                G[a, b] = G[b, a] = G[a, b] + val
        for t, e in zip(path.times, path.types):
            B[e] += features(path, p, beta, t)
    return G, B


def test_matches_quadrature(rng):
    p = 3
    beta = np.array([0.5, 1.3, 2.0])
    paths = [
        Path(np.sort(rng.uniform(0, 5, 6)), rng.integers(0, p, 6), 6.0),
        Path(np.array([0.2, 0.9]), np.array([2, 0]), 3.0),
        Path(np.array([]), np.array([], dtype=int), 2.5),  # empty path
    ]
    stats = sufficient_stats(paths, p, beta)
    G, B = brute_stats(paths, p, beta)
    np.testing.assert_allclose(stats.G, G, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(stats.B, B, rtol=1e-10, atol=1e-12)
    assert stats.n == 3


def test_additive_over_paths(rng):
    p = 4
    paths = [Path(np.sort(rng.uniform(0, 8, k)), rng.integers(0, p, k), 9.0) for k in (1, 5, 9, 3)]
    whole = sufficient_stats(paths, p, 0.7)
    parts = sufficient_stats(paths[:2], p, 0.7) + sufficient_stats(paths[2:], p, 0.7)
    np.testing.assert_allclose(whole.G, parts.G)
    np.testing.assert_allclose(whole.B, parts.B)
    back = whole - sufficient_stats(paths[2:], p, 0.7)
    np.testing.assert_allclose(back.G, sufficient_stats(paths[:2], p, 0.7).G, atol=1e-12)


def test_V_is_psd_and_contrast_unbiased_at_truth():
    """E[b - V theta*] = 0 is the martingale identity behind the estimator."""
    from gppmta.paper import ALPHA, DECAY, MU_ALL, TRUE_PARAMS
    from gppmta import simulate

    sim = simulate(MU_ALL, ALPHA, DECAY, 365.0, 4000, seed=3)
    stats = sufficient_stats(sim.paths, 5, DECAY)
    assert np.linalg.eigvalsh(stats.V).min() > 0
    for e in range(4):
        theta = np.r_[TRUE_PARAMS.mu[e], TRUE_PARAMS.alpha[:, e]]
        score = stats.b(e) - stats.V @ theta
        # Each coordinate is a mean of martingale terms: small relative to b.
        assert np.all(np.abs(score) < 0.08 * np.abs(stats.b(e)).max() + 1e-6), (e, score)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        sufficient_stats([Path(np.array([1.0]), np.array([3]), 2.0)], 3, 1.0)
    with pytest.raises(ValueError):
        sufficient_stats([Path(np.array([1.0]), np.array([0]), 2.0)], 3, 0.0)
    with pytest.raises(ValueError):
        sufficient_stats([], 3, 1.0)
    with pytest.raises(ValueError):
        Path(np.array([1.0, 1.0]), np.array([0, 1]), 2.0)  # tie
    with pytest.raises(ValueError):
        Path(np.array([3.0]), np.array([0]), 2.0)  # outside window

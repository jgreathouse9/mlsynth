"""The treatment bridge solves the moment it says it solves, or says it did not.

Rung 3 of the ladder in ``agents/agents_tests.md``, for the root solve that both
proximal estimators depend on.

``fit_treatment_bridge`` returns the ``beta`` satisfying

    E_pre[exp((1,Z) beta) (1,W)] = E_post[(1,W)]

which is a square system, and it finds it with a Newton/hybr solve. A root
finder can fail, and when it does there is a returned vector either way. The
failure is not visible in the shape or the finiteness of that vector, and the
downstream estimate built from it is an ordinary-looking number: on the authors'
``short_post`` and ``small_T`` designs a non-converged bridge produced ATTs
within 1e-2 of theirs and standard errors of 8.5 to 12.4, which are not
estimates of anything.

So the invariant is the defining property of the answer, checked on the answer:
the moment residual at the returned ``beta`` is zero. It needs no reference
implementation and no known truth, and it is exactly what the solver's own
``success`` flag was reporting before that flag was discarded.

The short post-period is what makes the system hard. ``psi = E_post[(1, W)]`` is
a mean over the post-treatment periods, so with twenty of them it is noisy, and
the exponential tilt that has to reproduce it may sit far out in ``beta`` or not
exist at all in finite sample. That is a real property of the design rather than
a numerical accident, which is why it is reported instead of retried away.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.proximal_helpers.bridges import augment, fit_treatment_bridge


def _panel(seed=0, T=600, T_post=None, nU=2):
    """AR(1) confounders; ``T_post`` short enough to strain the bridge if given."""
    rng = np.random.default_rng(seed)
    T0 = T - (T_post if T_post is not None else T // 2)
    U = np.zeros((T, nU))
    U[0] = rng.standard_normal(nU)
    for t in range(1, T):
        U[t] = 0.1 * U[t - 1] + 0.9 * rng.standard_normal(nU)
    post = np.arange(T) >= T0
    Y = 2.0 * post + 2 * U.sum(1) + rng.standard_normal(T)
    W = 2 * U + rng.standard_normal((T, nU))
    Z = 2 * U + rng.standard_normal((T, nU))
    return Y, W, Z, T0


def _residual(beta, Zc_pre, Wc_pre, psi):
    q = np.exp(Zc_pre @ beta)
    return float(np.max(np.abs((q[:, None] * Wc_pre).mean(0) - psi)))


class TestTheReturnedBridgeSolvesItsMoment:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_the_moment_residual_is_zero(self, seed):
        Y, W, Z, T0 = _panel(seed=seed)
        pre = np.arange(len(Y)) < T0
        Wc, Zc = augment(W), augment(Z)
        psi = Wc[~pre].mean(0)
        beta = fit_treatment_bridge(Zc[pre], Wc[pre], psi)
        assert _residual(beta, Zc[pre], Wc[pre], psi) < 1e-6

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_supplied_start_reaches_the_same_root(self, seed):
        """The root is a property of the moment, not of where the search began."""
        Y, W, Z, T0 = _panel(seed=seed)
        pre = np.arange(len(Y)) < T0
        Wc, Zc = augment(W), augment(Z)
        psi = Wc[~pre].mean(0)
        a = fit_treatment_bridge(Zc[pre], Wc[pre], psi)
        b = fit_treatment_bridge(Zc[pre], Wc[pre], psi, beta_init=a + 0.05)
        assert np.allclose(a, b, atol=1e-6)


class TestAFailedSolveIsReported:
    """What the discarded ``success`` flag was for."""

    def test_an_unreachable_moment_raises(self):
        """``psi`` outside the range of the exponential tilt has no solution.

        A negative target for a moment that is a positive-weight average of
        positive quantities cannot be met, so no ``beta`` exists.
        """
        rng = np.random.default_rng(0)
        Zc = augment(rng.standard_normal((200, 2)))
        Wc = augment(np.abs(rng.standard_normal((200, 2))) + 1.0)
        psi = np.array([-5.0, -5.0, -5.0])
        with pytest.raises(MlsynthEstimationError, match="treatment bridge"):
            fit_treatment_bridge(Zc, Wc, psi)

    def test_the_error_reports_the_residual(self):
        rng = np.random.default_rng(1)
        Zc = augment(rng.standard_normal((150, 2)))
        Wc = augment(np.abs(rng.standard_normal((150, 2))) + 1.0)
        with pytest.raises(MlsynthEstimationError, match=r"residual"):
            fit_treatment_bridge(Zc, Wc, np.array([-3.0, -3.0, -3.0]))

    def test_a_reachable_moment_does_not_raise(self):
        Y, W, Z, T0 = _panel(seed=5)
        pre = np.arange(len(Y)) < T0
        Wc, Zc = augment(W), augment(Z)
        beta = fit_treatment_bridge(Zc[pre], Wc[pre], Wc[~pre].mean(0))
        assert np.all(np.isfinite(beta))


class TestTheEstimatorsSurfaceIt:
    """The translated error reaches the caller instead of a wrong number."""

    def test_pipw_raises_on_an_unsolvable_bridge(self):
        from mlsynth.utils.proximal_helpers.pipw.estimation import estimate_pipw
        rng = np.random.default_rng(3)
        T, T0 = 120, 100
        Y = rng.standard_normal(T)
        W = np.abs(rng.standard_normal((T, 2))) + 1.0
        W[T0:] *= -1.0                      # post-period mean outside the tilt's range
        Z = rng.standard_normal((T, 2))
        with pytest.raises(MlsynthEstimationError):
            estimate_pipw(Y, W, Z, T0, hac_bandwidth=2)

    def test_dr_raises_on_an_unsolvable_bridge(self):
        from mlsynth.utils.proximal_helpers.dr.estimation import estimate_dr
        rng = np.random.default_rng(4)
        T, T0 = 120, 100
        Y = rng.standard_normal(T)
        W = np.abs(rng.standard_normal((T, 2))) + 1.0
        W[T0:] *= -1.0
        Z = rng.standard_normal((T, 2))
        with pytest.raises(MlsynthEstimationError):
            estimate_dr(Y, W, Z, T0, hac_bandwidth=2)

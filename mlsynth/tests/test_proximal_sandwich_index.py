"""The GMM sandwich reports the standard error of the parameter it was asked for.

Rung 3 of the ladder in ``agents/agents_tests.md``, for the proximal estimators
that stack their estimand into a just-identified GMM and read one entry out of
the sandwich.

The invariant. ``gmm_sandwich_se(theta, moments, phi_index, ...)`` returns the
square root of one diagonal entry of a covariance matrix over ``theta``. Which
entry is chosen by an integer, and the moment function unpacks ``theta`` by its
own hard-coded offsets. Nothing ties the two together: they are two independent
statements about the same layout, and if they disagree the function returns a
perfectly well-formed standard error belonging to a different parameter.

So the property asserted here is structural, not numerical -- ``theta`` at the
index handed to the sandwich must *be* the estimate the routine returns. It needs
no reference implementation and no known truth, which is what makes it the right
instrument: the failure it catches produces a finite, positive, plausibly-sized
number, and every assertion of the form ``se > 0`` passes straight through it.

The second class below is the same invariant reached from the estimator side.
In the just-identified design the treatment bridge solves
``E_pre[q (1, W)] = E_post[(1, W)]`` exactly, so the DR correction term vanishes
and DR and PIPW are the same estimator applied to the same moments. Their point
estimates coincide, and so must their standard errors. Any drift between the two
is a disagreement about the layout, not about the econometrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError

from mlsynth.utils.proximal_helpers.dr.estimation import estimate_dr
from mlsynth.utils.proximal_helpers.pipw.estimation import estimate_pipw


def _panel(seed=0, T=600, nU=2, effect=2.0):
    """AR(1) latent confounders with independent W and Z proxy blocks."""
    rng = np.random.default_rng(seed)
    T0 = T // 2
    U = np.zeros((T, nU))
    U[0] = rng.standard_normal(nU)
    for t in range(1, T):
        U[t] = 0.1 * U[t - 1] + 0.9 * rng.standard_normal(nU)
    post = np.arange(T) >= T0
    Y = effect * post + 2 * U.sum(1) + rng.standard_normal(T)
    W = 2 * U + rng.standard_normal((T, nU))
    Z = 2 * U + rng.standard_normal((T, nU))
    return Y, W, Z, T0


class TestTheIndexPointsAtTheEstimand:
    """The contract ``gmm_sandwich_se`` now enforces, exercised directly.

    A caller that hands the helper an index not holding the estimate gets an
    error instead of another parameter's standard error.
    """

    def test_a_mismatched_index_raises(self):
        from mlsynth.utils.proximal_helpers.bridges import gmm_sandwich_se
        theta = np.array([1.0, 2.0, 3.0])
        moments = lambda th: np.column_stack([
            np.full(50, th[0]), np.full(50, th[1]), np.full(50, th[2])])
        with pytest.raises(MlsynthEstimationError, match="index 2"):
            gmm_sandwich_se(theta, moments, 2, 50, 2, expected_value=2.0)

    def test_a_matching_index_is_accepted(self):
        from mlsynth.utils.proximal_helpers.bridges import gmm_sandwich_se
        rng = np.random.default_rng(0)
        theta = np.array([1.0, 2.0])
        U = rng.standard_normal((80, 2))
        moments = lambda th: U + th[None, :]
        assert np.isfinite(gmm_sandwich_se(theta, moments, 1, 80, 2,
                                           expected_value=2.0))

    def test_the_check_is_opt_in(self):
        """Omitting the estimate keeps the old signature working."""
        from mlsynth.utils.proximal_helpers.bridges import gmm_sandwich_se
        rng = np.random.default_rng(1)
        U = rng.standard_normal((60, 2))
        assert np.isfinite(gmm_sandwich_se(np.array([1.0, 2.0]),
                                           lambda th: U + th[None, :], 0, 60, 2))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pipw_asks_for_its_own_att(self, seed):
        """End to end: the estimator wires its own estimate into the check, so a
        layout drift fails at the call instead of returning a wrong number."""
        Y, W, Z, T0 = _panel(seed=seed)
        _, att, se = estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)
        assert np.isfinite(se) and se > 0 and np.isfinite(att)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_dr_asks_for_its_own_att(self, seed):
        Y, W, Z, T0 = _panel(seed=seed)
        *_, att, se = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
        assert np.isfinite(se) and se > 0 and np.isfinite(att)


class TestDrAndPipwAgreeWhenJustIdentified:
    """The estimator-side reading of the same invariant."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_the_point_estimates_coincide(self, seed):
        Y, W, Z, T0 = _panel(seed=seed)
        _, _, _, dr_att, _ = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
        _, pipw_att, _ = estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)
        assert dr_att == pytest.approx(pipw_att, rel=1e-8)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_the_standard_errors_coincide(self, seed):
        """Same estimator, same moments, same sandwich: the standard errors
        cannot differ. A gap here means the two are reading different entries
        out of it."""
        Y, W, Z, T0 = _panel(seed=seed)
        _, _, _, _, dr_se = estimate_dr(Y, W, Z, T0, hac_bandwidth=4)
        _, _, pipw_se = estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)
        assert dr_se == pytest.approx(pipw_se, rel=1e-6), (
            f"DR reports {dr_se:.6f} and PIPW {pipw_se:.6f} for the same estimate")


class TestTheStandardErrorIsSized:
    """What ``se > 0`` cannot see.

    A standard error belonging to the wrong parameter is finite and positive.
    What separates it from the right one is its size relative to the estimate's
    own sampling variability, so that is what gets checked.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_wald_interval_covers_the_truth(self, seed):
        Y, W, Z, T0 = _panel(seed=seed, T=2000, effect=2.0)
        _, att, se = estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)
        assert abs(att - 2.0) <= 1.96 * se

    def test_the_standard_error_shrinks_with_the_sample(self):
        """Root-n: quadrupling T should roughly halve it. A standard error
        attached to a different parameter need not scale this way."""
        ses = []
        for T in (500, 2000):
            Y, W, Z, T0 = _panel(seed=7, T=T)
            ses.append(estimate_pipw(Y, W, Z, T0, hac_bandwidth=4)[2])
        assert 0.3 < ses[1] / ses[0] < 0.8, f"se ratio {ses[1] / ses[0]:.3f}"

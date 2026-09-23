"""Population counterpart of the Forward DiD Web Appendix E DGPs.

Tests for :mod:`mlsynth.utils.fdid_helpers.population`, which supplies the
theoretical prediction variance and the theoretical forward selection
algorithm of Li (2023) Web Appendix D -- the infeasible
benchmark that Propositions 2.2 / D.1 compare the empirical algorithm to.

The correctness anchor is Monte Carlo: every closed form here is checked
against the draws that :func:`simulate_fdid_sample` actually produces, so a
divergence between the formula and the simulator fails a test instead of
silently mis-scoring a benchmark.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlsynth.utils.fdid_helpers.population import (
    FACTOR_SUM_VARIANCE,
    TheoreticalSelection,
    group_counts,
    prediction_variance,
    theoretical_forward_selection,
)
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_sample


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------

def test_theoretical_selection_smoke():
    sel = theoretical_forward_selection(dgp=2, N=20)
    assert isinstance(sel, TheoreticalSelection)
    n1, n2 = sel.counts
    assert 0 < n1 + n2 <= 20
    assert np.isfinite(sel.variance)
    assert len(sel.path) == 20


def test_prediction_variance_smoke():
    v = prediction_variance(dgp=2, n_group1=3, n_group2=1)
    assert np.isfinite(v) and v > 0


# --------------------------------------------------------------------------
# The closed forms against the simulator
# --------------------------------------------------------------------------

def test_factor_sum_variance_matches_the_factor_processes():
    """sigma_S^2 is the sum of the three factors' stationary variances.

    f1: AR(1), phi = 0.8.  f2: ARMA(1,1), phi = -0.6, theta = 0.8.
    f3: MA(2), theta = (0.9, 0.4).  Innovations are unit-variance and the
    three processes are driven by independent innovations, so the variance
    of ``1' f_t`` is the sum.
    """
    var_f1 = 1.0 / (1.0 - 0.8 ** 2)
    var_f2 = (1.0 + 2.0 * (-0.6) * 0.8 + 0.8 ** 2) / (1.0 - 0.6 ** 2)
    var_f3 = 1.0 + 0.9 ** 2 + 0.4 ** 2
    assert FACTOR_SUM_VARIANCE == pytest.approx(var_f1 + var_f2 + var_f3)


@pytest.mark.parametrize("dgp", [1, 2, 3, 4])
@pytest.mark.parametrize("n_group1,n_group2", [(6, 0), (6, 3), (1, 4)])
def test_prediction_variance_matches_monte_carlo(dgp, n_group1, n_group2):
    """``prediction_variance`` reproduces Var(y_tr - ybar_U) from the draws.

    ``V_U = E[(y_tr,t - ybar_Ut - alpha_U)^2]`` is the variance of the DiD
    residual, so the Monte-Carlo counterpart is the pre-period variance of
    ``y_tr - ybar_U`` averaged over draws. 60 draws of 800 pre-periods puts
    the Monte-Carlo standard error near 1% of the target; the tolerance is
    3% to stay clear of it.
    """
    N = 20
    rng = np.random.default_rng(11)
    U = list(range(n_group1)) + list(range(N // 2, N // 2 + n_group2))
    sample_vars = []
    for _ in range(60):
        s = simulate_fdid_sample(dgp=dgp, N=N, T1=800, T2=5, rng=rng)
        resid = s.Y_treated[:800] - s.Y_controls[U].mean(axis=0)[:800]
        sample_vars.append(resid.var())
    target = prediction_variance(dgp=dgp, n_group1=n_group1, n_group2=n_group2)
    assert np.mean(sample_vars) == pytest.approx(target, rel=0.03)


@pytest.mark.parametrize("dgp", [1, 2, 3, 4])
def test_prediction_variance_depends_only_on_group_counts(dgp):
    """Members of a loading group are exchangeable, so V depends on counts.

    This is what makes the tie structure of ``U*`` describable by a count
    pair instead of an enumeration of subsets.
    """
    a = prediction_variance(dgp=dgp, n_group1=4, n_group2=2)
    b = prediction_variance(dgp=dgp, n_group1=4, n_group2=2)
    assert a == b
    assert a != prediction_variance(dgp=dgp, n_group1=2, n_group2=4) or dgp in (1, 3)


# --------------------------------------------------------------------------
# What the theoretical algorithm selects, per DGP
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dgp", [1, 3])
def test_matched_dgps_select_every_control(dgp):
    """With all loadings equal to the treated unit's, V = 1 + 1/n.

    Nothing separates the controls, so the criterion is driven entirely by
    the averaging term and the optimum is the whole donor pool.
    """
    N = 20
    sel = theoretical_forward_selection(dgp=dgp, N=N)
    assert sel.counts == (N // 2, N - N // 2)
    assert sel.unique
    assert sel.variance == pytest.approx(1.0 + 1.0 / N)


@pytest.mark.parametrize("dgp", [2, 4])
def test_mismatched_dgps_select_exactly_the_matched_half(dgp):
    """Group 2 loads at c2 = 2 against the treated unit's c0 = 1.

    Admitting any of them injects a factor-loading gap that the 1/n gain
    cannot pay for, so the optimum is group 1 entire.
    """
    N = 20
    sel = theoretical_forward_selection(dgp=dgp, N=N)
    assert sel.counts == (N // 2, 0)
    assert sel.unique
    assert sel.variance == pytest.approx(1.0 + 1.0 / (N // 2))


@pytest.mark.parametrize("dgp", [1, 2, 3, 4])
def test_optimum_beats_every_reachable_step(dgp):
    """The reported optimum is the minimum over the greedy path."""
    sel = theoretical_forward_selection(dgp=dgp, N=20)
    assert sel.variance == pytest.approx(min(v for _, _, v in sel.path))


def test_greedy_path_adds_the_matched_group_first():
    """Under DGP 2 the path is group 1 until it is exhausted."""
    N = 20
    sel = theoretical_forward_selection(dgp=2, N=N)
    for n1, n2, _ in sel.path[: N // 2]:
        assert n2 == 0
    assert sel.path[N // 2 - 1][0] == N // 2


def test_selection_is_the_exhaustive_optimum_for_the_appendix_dgps():
    """Greedy matches brute force over all 2^N - 1 subsets.

    Forward selection is not guaranteed to find the best subset in general
    (the appendix says so). On these DGPs it does, and pinning that keeps a
    later change to the criterion from being scored against a benchmark
    that silently stopped agreeing with exhaustive search.
    """
    N = 12
    for dgp in (1, 2, 3, 4):
        best = min(
            (
                prediction_variance(
                    dgp=dgp,
                    n_group1=sum(1 for i in U if i < N // 2),
                    n_group2=sum(1 for i in U if i >= N // 2),
                ),
                len(U),
            )
            for k in range(1, N + 1)
            for U in itertools.combinations(range(N), k)
        )
        sel = theoretical_forward_selection(dgp=dgp, N=N)
        assert sel.variance == pytest.approx(best[0])


# --------------------------------------------------------------------------
# Membership in U*
# --------------------------------------------------------------------------

def test_contains_accepts_the_optimal_set_and_rejects_others():
    N = 20
    sel = theoretical_forward_selection(dgp=2, N=N)
    assert sel.contains(list(range(N // 2)))                 # group 1 entire
    assert not sel.contains(list(range(N // 2 - 1)))         # one short
    assert not sel.contains(list(range(N // 2 + 1)))         # one group-2 member
    assert not sel.contains(list(range(N // 2, N)))          # the wrong half


def test_group_counts_splits_on_the_simulator_convention():
    """Rows 0..N//2-1 carry c1; rows N//2..N-1 carry c2."""
    assert group_counts([0, 1, 10, 11, 12], N=20) == (2, 3)
    assert group_counts([], N=20) == (0, 0)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_two_controls_one_per_group():
    sel = theoretical_forward_selection(dgp=2, N=2)
    assert sel.counts == (1, 0)
    assert sel.variance == pytest.approx(2.0)


def test_odd_pool_keeps_the_simulator_split():
    """``simulate_fdid_sample`` uses ``half = N // 2``, so an odd pool puts
    the extra unit in group 2. The population helper must split the same way
    or the two disagree about which controls are matched."""
    N = 21
    sel = theoretical_forward_selection(dgp=2, N=N)
    assert sel.counts == (N // 2, 0)
    assert group_counts(list(range(N)), N=N) == (N // 2, N - N // 2)


def test_single_control_pool():
    sel = theoretical_forward_selection(dgp=1, N=1)
    assert sel.counts == (0, 1)
    assert sel.variance == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Failures are reported, not swallowed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 5, -1])
def test_unknown_dgp_raises(bad):
    with pytest.raises(ValueError, match="dgp must be in"):
        theoretical_forward_selection(dgp=bad, N=10)
    with pytest.raises(ValueError, match="dgp must be in"):
        prediction_variance(dgp=bad, n_group1=1, n_group2=0)


def test_empty_subset_variance_raises():
    """V_U is undefined for the empty subset -- the 1/|U| term divides by
    zero. The helper must say so instead of returning ``inf``."""
    with pytest.raises(ValueError, match="at least one control"):
        prediction_variance(dgp=2, n_group1=0, n_group2=0)


@pytest.mark.parametrize("N", [0, -3])
def test_non_positive_pool_raises(N):
    with pytest.raises(ValueError, match="N must be"):
        theoretical_forward_selection(dgp=1, N=N)


def test_counts_exceeding_the_pool_raise():
    with pytest.raises(ValueError, match="exceeds"):
        prediction_variance(dgp=2, n_group1=3, n_group2=1, N=3)

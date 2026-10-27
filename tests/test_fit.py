import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.optimize import minimize

from gppmta import GraphicalPointProcess, admm, cv_gamma, simulate, solve_node, sufficient_stats
from gppmta.fit import gamma_max, objective
from gppmta.paper import ALPHA, DECAY, MU_ALL, TRUE_PARAMS


@pytest.fixture(scope="module")
def sim_stats():
    sim = simulate(MU_ALL, ALPHA, DECAY, 365.0, 6000, seed=21)
    return sim, sufficient_stats(sim.paths, 5, DECAY)


def kkt_violation(V, b, gamma, theta):
    g = V @ theta - b + gamma * np.r_[0.0, np.ones(len(b) - 1)]
    pos = theta > 1e-12
    return max(np.abs(g[pos]).max(initial=0), (-g[~pos]).max(initial=0))


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(2, 7), st.floats(0, 2))
def test_nnls_solution_is_optimal(seed, k, gamma):
    """KKT holds and no general-purpose solver finds a lower objective."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(k + 3, k))
    V = A.T @ A + 0.05 * np.eye(k)
    b = rng.normal(size=k) * 2
    theta = solve_node(V, b, gamma)
    assert np.all(theta >= 0)
    assert kkt_violation(V, b, gamma, theta) < 1e-7 * (1 + np.abs(b).max())
    ref = minimize(lambda th: objective(V, b, gamma, th), np.ones(k),
                   jac=lambda th: V @ th - b + gamma * np.r_[0.0, np.ones(k - 1)],
                   bounds=[(0, None)] * k, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
    assert objective(V, b, gamma, theta) <= ref.fun + 1e-9


def test_admm_lands_near_but_not_on_the_optimum_on_the_paper_design(sim_stats):
    """Algorithm 3 stops at a fixed point of its clipped theta-step, which is
    within 1% of the exact solution here (objective gap ~1e-8) but not equal
    to it: its residuals are exactly zero there, so no tolerance helps."""
    _, stats = sim_stats
    worst = 0.0
    for e in range(4):
        for gamma in (0.0, 1e-4, 1e-3):
            exact = solve_node(stats.V, stats.b(e), gamma)
            th, _ = admm(stats.V, stats.b(e), gamma, eta=1.0, tol=1e-15, max_iter=200_000)
            gap = objective(stats.V, stats.b(e), gamma, th) - objective(stats.V, stats.b(e), gamma, exact)
            assert gap >= -1e-12  # NNLS is never beaten
            rel = np.abs(th - exact).max() / np.abs(exact).max()
            assert rel < 0.01, (e, gamma, rel)
            worst = max(worst, rel)
    assert worst > 1e-4  # ...and the gap is real, not round-off


def test_admm_clipping_step_is_inexact_when_mu_wants_to_be_negative():
    """Algorithm 3 clips the unconstrained theta-step; with a binding mu bound
    and correlated features the fixed point is not the constrained optimum."""
    V = np.array([[1.0, 0.9], [0.9, 1.0]])
    b = np.array([-0.5, 1.0])  # unconstrained mu < 0, alpha > 0
    exact = solve_node(V, b, 0.0)
    th, _ = admm(V, b, 0.0)
    assert exact[0] == 0 and exact[1] == pytest.approx(1.0)
    f = lambda t: objective(V, b, 0.0, t)
    assert f(th) > f(exact) + 1e-3  # ADMM stalls at a worse point


def test_recovers_table1_graph(sim_stats):
    _, stats = sim_stats
    from gppmta import fit_params

    est = fit_params(stats, 4, DECAY, gamma=2e-5)
    truth = TRUE_PARAMS.alpha
    np.testing.assert_allclose(est.alpha, truth, atol=0.025)
    np.testing.assert_allclose(est.mu, TRUE_PARAMS.mu, rtol=0.15, atol=2e-4)
    # Every true edge is found.
    assert set(TRUE_PARAMS.edges) <= set(est.edges)


def test_gamma_max_zeroes_every_edge(sim_stats):
    _, stats = sim_stats
    for e in range(4):
        g = gamma_max(stats.V, stats.b(e))
        th = solve_node(stats.V, stats.b(e), g * 1.0001)
        assert np.all(th[1:] == 0) and th[0] > 0
        th = solve_node(stats.V, stats.b(e), g * 0.9)
        assert np.any(th[1:] > 0)


def test_refit_removes_shrinkage_on_the_selected_support(sim_stats):
    _, stats = sim_stats
    for e in range(4):
        g = 0.3 * gamma_max(stats.V, stats.b(e))
        lasso = solve_node(stats.V, stats.b(e), g)
        relaxed = solve_node(stats.V, stats.b(e), g, refit=True)
        on = lasso > 0
        on[0] = True
        assert np.all(relaxed[~on] == 0)
        assert relaxed[1:][on[1:]].sum() >= lasso[1:][on[1:]].sum() - 1e-12


def test_cv_picks_a_small_gamma_with_plenty_of_data(sim_stats):
    sim, _ = sim_stats
    cv = cv_gamma(sim.paths, 5, 4, DECAY, n_folds=4)
    assert cv.best.shape == (4,)
    for e in range(4):
        assert cv.best[e] < 0.5 * cv.grid[e, 0]  # never the null model
    est = GraphicalPointProcess(decay=DECAY, gamma="cv").fit(sim.paths, 5, 4).params_
    assert set(TRUE_PARAMS.edges) <= set(est.edges)


def test_cv_decay_recovers_the_true_kernel(sim_stats):
    from gppmta.fit import cv_decay

    sim, _ = sim_stats
    best, losses = cv_decay(sim.paths, 5, 4, [0.025, 0.05, 0.1, 0.2, 0.4])
    assert best == DECAY
    assert losses.shape == (5,)


def test_fit_rejects_bad_arguments(sim_stats):
    sim, _ = sim_stats
    with pytest.raises(ValueError):
        GraphicalPointProcess(decay=DECAY, gamma="auto").fit(sim.paths, 5, 4)
    with pytest.raises(ValueError):
        GraphicalPointProcess(decay=DECAY, gamma=0.1).fit(sim.paths, 5, 6)
    with pytest.raises(ValueError):
        solve_node(np.eye(2), np.ones(2), -1.0)

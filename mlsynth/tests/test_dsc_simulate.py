"""Tests for the DSC Monte-Carlo simulator (Zhang, Zhang & Zhang 2024, Sec. 5.1).

Covers the four levels the repository requires of a new unit of behaviour:
a smoke run, the invariants of each piece (rank construction, closed-form
population risk, oracle weights), degenerate inputs, and the translated
failures.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.dsc_helpers.simulate import (
    donor_scales,
    draw_model_free_design,
    model_free_replication,
    model_free_risk_matrices,
    oracle_simplex_weights,
    paired_uniform_ranks,
    population_risk,
    simulate_model_free_period,
)


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_model_free_replication_smoke():
    """One replication on a minimal design returns two finite, signed-right numbers."""
    ratio, norm = model_free_replication(J=4, M=60, T0=3, seed=0)
    assert np.isfinite(ratio) and np.isfinite(norm)
    # The oracle minimises the population risk over the simplex, so any feasible
    # weight vector scores at least as badly: the ratio cannot fall below one.
    assert ratio >= 1.0 - 1e-9
    assert norm >= 0.0


# --------------------------------------------------------------------------
# paired_uniform_ranks
# --------------------------------------------------------------------------

def test_paired_ranks_shape_and_support():
    V = paired_uniform_ranks(200, np.random.default_rng(0))
    assert V.shape == (200,)
    assert np.all((V > 0.0) & (V < 1.0))


def test_paired_ranks_pairing_rule():
    """Second half is the first shifted by delta, toward the centre of (0, 1)."""
    delta = 0.01
    V = paired_uniform_ranks(400, np.random.default_rng(1), delta=delta)
    first, second = V[:200], V[200:]
    expected = np.where(first < 0.5, first + delta, first - delta)
    np.testing.assert_allclose(second, expected)
    assert np.any(first < 0.5) and np.any(first >= 0.5)   # both branches exercised


def test_paired_ranks_are_dependent_not_iid():
    """The paired draws sit delta apart -- the dependence the paper introduces."""
    V = paired_uniform_ranks(1000, np.random.default_rng(2), delta=0.01)
    gaps = np.abs(V[500:] - V[:500])
    np.testing.assert_allclose(gaps, 0.01)


def test_paired_ranks_odd_M_raises():
    with pytest.raises(MlsynthEstimationError, match="even"):
        paired_uniform_ranks(51, np.random.default_rng(0))


@pytest.mark.parametrize("delta", [0.0, -0.01, 0.5, 0.9])
def test_paired_ranks_bad_delta_raises(delta):
    with pytest.raises(MlsynthEstimationError, match="delta"):
        paired_uniform_ranks(50, np.random.default_rng(0), delta=delta)


def test_paired_ranks_small_M_raises():
    with pytest.raises(MlsynthEstimationError, match="M must be"):
        paired_uniform_ranks(0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# donor_scales / draw_model_free_design
# --------------------------------------------------------------------------

def test_donor_scales_alternate_by_unit_label():
    """sigma_j = 3 for odd j, 2.5 for even j, over the donor labels j = 2..J+1."""
    np.testing.assert_allclose(donor_scales(5), [2.5, 3.0, 2.5, 3.0, 2.5])


def test_donor_scales_length():
    assert donor_scales(50).shape == (50,)


def test_donor_scales_requires_positive_J():
    with pytest.raises(MlsynthEstimationError, match="J must be"):
        donor_scales(0)


def test_draw_model_free_design_requires_positive_J():
    with pytest.raises(MlsynthEstimationError, match="J must be"):
        draw_model_free_design(0, np.random.default_rng(0))


def test_draw_model_free_design_support():
    mu, sigma = draw_model_free_design(200, np.random.default_rng(3))
    assert np.all((mu >= 3.0) & (mu <= 10.0))
    assert set(np.unique(sigma)) == {2.5, 3.0}


# --------------------------------------------------------------------------
# closed-form population risk
# --------------------------------------------------------------------------

def test_risk_matrices_match_the_paper_moment_matrix():
    """G = mu mu' + diag(sigma^2), b = mu_1 mu = 2 mu, const = E[Y_1^2] = 8."""
    mu = np.array([4.0, 7.0, 9.5])
    sigma = np.array([2.5, 3.0, 2.5])
    G, b, const = model_free_risk_matrices(mu, sigma)
    np.testing.assert_allclose(G, np.outer(mu, mu) + np.diag(sigma ** 2))
    np.testing.assert_allclose(b, 2.0 * mu)
    assert const == pytest.approx(8.0)          # chi2(2): var 4 + mean^2 4


def test_population_risk_matches_brute_force_monte_carlo():
    """The closed form equals a direct simulation of E[(sum_j w_j Y_j - Y_1)^2]."""
    rng = np.random.default_rng(4)
    mu = np.array([4.0, 7.0, 9.5])
    sigma = np.array([2.5, 3.0, 2.5])
    w = np.array([0.5, 0.3, 0.2])
    G, b, const = model_free_risk_matrices(mu, sigma)

    n = 400_000
    donors = rng.normal(mu, sigma, size=(n, 3))
    treated = rng.chisquare(2, size=n)
    brute = float(np.mean((donors @ w - treated) ** 2))
    assert population_risk(w, G, b, const) == pytest.approx(brute, rel=2e-2)


def test_population_risk_is_nonnegative_on_the_simplex():
    rng = np.random.default_rng(5)
    mu, sigma = draw_model_free_design(8, rng)
    G, b, const = model_free_risk_matrices(mu, sigma)
    for _ in range(20):
        w = rng.dirichlet(np.ones(8))
        assert population_risk(w, G, b, const) >= 0.0


def test_risk_matrices_reject_mismatched_lengths():
    with pytest.raises(MlsynthEstimationError, match="same length"):
        model_free_risk_matrices(np.array([4.0, 7.0]), np.array([2.5]))


# --------------------------------------------------------------------------
# oracle weights
# --------------------------------------------------------------------------

def test_oracle_weights_are_feasible_and_optimal():
    rng = np.random.default_rng(6)
    mu, sigma = draw_model_free_design(12, rng)
    G, b, const = model_free_risk_matrices(mu, sigma)
    w_opt = oracle_simplex_weights(G, b)

    assert np.all(w_opt >= -1e-9)
    assert w_opt.sum() == pytest.approx(1.0)
    best = population_risk(w_opt, G, b, const)
    for _ in range(50):
        w = rng.dirichlet(np.ones(12))
        assert population_risk(w, G, b, const) >= best - 1e-8


def test_oracle_singles_out_the_only_matching_donor():
    """With one donor whose mean sits at the treated mean, the oracle loads it."""
    mu = np.array([2.0, 9.0, 9.5])          # treated chi2(2) has mean 2
    sigma = np.array([0.05, 3.0, 2.5])
    G, b, _ = model_free_risk_matrices(mu, sigma)
    w_opt = oracle_simplex_weights(G, b)
    assert w_opt[0] > 0.95


def test_oracle_single_donor_is_degenerate():
    G, b, _ = model_free_risk_matrices(np.array([5.0]), np.array([3.0]))
    np.testing.assert_allclose(oracle_simplex_weights(G, b), [1.0])


# --------------------------------------------------------------------------
# one simulated period
# --------------------------------------------------------------------------

def test_simulate_period_shapes_and_marginals():
    rng = np.random.default_rng(7)
    mu = np.array([4.0, 8.0])
    sigma = np.array([2.5, 3.0])
    donors, treated = simulate_model_free_period(mu, sigma, 40_000, rng)
    assert donors.shape == (40_000, 2)
    assert treated.shape == (40_000,)
    np.testing.assert_allclose(donors.mean(axis=0), mu, atol=0.1)
    assert np.all(treated > 0.0)


def test_paired_ranks_compress_the_donor_marginal():
    """The pairing pulls the tails in, so the realised spread is 0.9633 sigma.

    Shifting a rank by delta = 0.01 barely moves a central draw and moves an
    extreme one a long way, so half the sample loses tail mass. The second
    moment of the mixture is E[Z^2] = 0.928 against a nominal 1, and the
    donors' realised standard deviation is sqrt(0.928) = 0.9633 times sigma.
    The paper writes its risk in terms of the nominal N(mu, sigma^2), so this
    gap is a property of its design, pinned here so it stays visible.
    """
    rng = np.random.default_rng(21)
    sigma = np.array([2.5, 3.0])
    donors, _ = simulate_model_free_period(np.array([4.0, 8.0]), sigma,
                                           400_000, rng)
    np.testing.assert_allclose(donors.std(axis=0), np.sqrt(0.928) * sigma,
                               rtol=6e-3)


def test_paired_ranks_compress_the_treated_marginal():
    """Same effect on the treated unit: mean 1.9578 against a nominal 2."""
    rng = np.random.default_rng(22)
    _, treated = simulate_model_free_period(np.array([4.0]), np.array([2.5]),
                                            400_000, rng)
    assert treated.mean() == pytest.approx(1.9578, abs=0.02)


def test_simulate_period_units_are_drawn_independently():
    """Each unit carries its own rank sequence, so donor columns are uncorrelated.

    This is the reading of the design that reproduces the paper's figures; the
    alternative (one rank sequence shared by every unit) makes the columns
    comonotonic and is checked against here so the choice cannot drift.
    """
    rng = np.random.default_rng(8)
    mu = np.array([4.0, 8.0, 6.0])
    sigma = np.array([2.5, 3.0, 2.5])
    donors, treated = simulate_model_free_period(mu, sigma, 40_000, rng)
    corr = np.corrcoef(np.column_stack([donors, treated]), rowvar=False)
    off = corr[np.triu_indices_from(corr, k=1)]
    assert np.max(np.abs(off)) < 0.05


def test_simulate_period_rejects_bad_M():
    with pytest.raises(MlsynthEstimationError):
        simulate_model_free_period(np.array([4.0]), np.array([2.5]), 3,
                                   np.random.default_rng(0))


def test_simulate_period_rejects_mismatched_lengths():
    with pytest.raises(MlsynthEstimationError, match="same length"):
        simulate_model_free_period(np.array([4.0, 7.0]), np.array([2.5]), 50,
                                   np.random.default_rng(0))


# --------------------------------------------------------------------------
# the replication driver
# --------------------------------------------------------------------------

def test_replication_is_deterministic_in_the_seed():
    a = model_free_replication(J=6, M=80, T0=3, seed=11)
    b = model_free_replication(J=6, M=80, T0=3, seed=11)
    assert a == b


def test_replication_varies_with_the_seed():
    a = model_free_replication(J=6, M=80, T0=3, seed=11)
    b = model_free_replication(J=6, M=80, T0=3, seed=12)
    assert a != b


def test_replication_single_donor_is_exact():
    """J = 1 forces w = 1, so the estimate coincides with the oracle."""
    ratio, norm = model_free_replication(J=1, M=60, T0=2, seed=0)
    assert ratio == pytest.approx(1.0)
    assert norm == pytest.approx(0.0)


def test_replication_single_pre_period_runs():
    ratio, norm = model_free_replication(J=3, M=60, T0=1, seed=0)
    assert np.isfinite(ratio) and np.isfinite(norm)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(J=0, M=60, T0=2), "J must be"),
        (dict(J=3, M=1, T0=2), "M must be"),
        (dict(J=3, M=61, T0=2), "even"),
        (dict(J=3, M=60, T0=0), "T0 must be"),
    ],
)
def test_replication_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(MlsynthEstimationError, match=match):
        model_free_replication(seed=0, **kwargs)


# --------------------------------------------------------------------------
# the theorems the design exists to illustrate
# --------------------------------------------------------------------------

def test_optimality_gap_shrinks_in_M():
    """Theorem 1: the risk ratio falls toward one as the draw count grows."""
    small = np.mean([model_free_replication(J=5, M=50, T0=3, seed=s)[0]
                     for s in range(12)])
    large = np.mean([model_free_replication(J=5, M=800, T0=3, seed=s)[0]
                     for s in range(12)])
    assert 1.0 <= large < small


def test_weight_error_shrinks_in_M():
    """Theorem 2: the DSC weight converges to the risk-minimising weight."""
    small = np.mean([model_free_replication(J=5, M=50, T0=3, seed=s)[1]
                     for s in range(12)])
    large = np.mean([model_free_replication(J=5, M=800, T0=3, seed=s)[1]
                     for s in range(12)])
    assert 0.0 <= large < small

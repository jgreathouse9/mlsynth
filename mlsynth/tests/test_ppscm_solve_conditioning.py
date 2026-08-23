"""The partially-pooled program is solved at the scale its tolerances assume.

``solve_cohort_qp`` asks OSQP for ``eps_abs=1e-8``. That is an absolute
demand, so what it costs depends on the units the caller's outcome happens to
be measured in. Synthetic control is scale-equivariant -- multiplying every
series by a constant leaves the weights alone and scales the effect with them
-- so any dependence of the answer on the panel's magnitude is an artefact of
the solver, and these tests pin it out.

The separate fit is the one that breaks. It normalizes by ``1.0`` (it is the
fit that produces the norms the pooled fit uses), so its objective grows with
the square of the panel's magnitude, and past about 1e3 OSQP returns nothing
at all and the SCS fallback answers a different question. The pooled fit
divides by norms carrying the same units, so its objective was already
dimensionless.
"""

import numpy as np
import pytest

from mlsynth.utils.ppscm_helpers.engine import (
    run_multisynth,
    solve_cohort_qp,
    solve_scale,
)


def staggered_panel(seed=7, n_donor=40, n_treat=6, T0=100, H=8, effect=0.5):
    """Factor-model panel with one adoption date: donors, then treated."""
    rng = np.random.default_rng(seed)
    n, T = n_donor + n_treat, T0 + H
    F = rng.normal(size=(T, 3)).cumsum(axis=0) / np.sqrt(T)
    L = rng.normal(size=(n, 3))
    Y = L @ F.T + rng.normal(scale=0.3, size=(n, T)) + 5.0
    trt = np.full(n, np.inf)
    trt[n_donor:] = T0
    Y[n_donor:, T0:] += effect
    return Y, trt, H


def fit(Y, trt, H, **kw):
    return run_multisynth(Y, trt, d=20, n_leads=H, n_lags=20, **kw)


# --- solve_scale: the summary itself -----------------------------------------

def test_a_unit_scale_residual_is_left_alone():
    rng = np.random.default_rng(0)
    res = {5: rng.normal(size=(20, 30))}
    assert solve_scale(res) == 1.0


def test_the_scale_is_a_power_of_two():
    rng = np.random.default_rng(0)
    for level in (1e3, 1e5, 1e-6, 2.0 ** 40):
        s = solve_scale({5: rng.normal(size=(20, 30)) * level})
        assert s > 0.0
        assert np.log2(s) == int(np.log2(s)), (level, s)


def test_dividing_by_the_scale_and_multiplying_back_is_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 30)) * 1e5
    s = solve_scale({5: x})
    assert s != 1.0
    assert np.array_equal((x / s) * s, x)


def test_the_scale_lands_within_a_factor_of_two_of_the_typical_residual():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 30)) * 2.0 ** 20
    assert solve_scale({5: x}) == pytest.approx(float(np.median(np.abs(x))), rel=0.5)


def test_one_very_large_market_does_not_set_the_scale():
    """A panel of markets spans an order of magnitude in size, so the divisor
    is the typical residual and not the biggest one: reading it off the
    largest would push every other market's residuals far below the tolerance
    the solver is working to."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 30))
    x[0, 0] = 1e6
    assert solve_scale({5: x}) == 1.0


def test_a_degenerate_residual_gets_unit_scale():
    assert solve_scale({}) == 1.0
    assert solve_scale({5: np.zeros((4, 5))}) == 1.0
    assert solve_scale({5: np.full((4, 5), np.nan)}) == 1.0


# --- equivariance of the fit --------------------------------------------------

@pytest.mark.parametrize("level", [1e3, 1e5])
def test_the_effect_scales_with_the_panel(level):
    Y, trt, H = staggered_panel()
    base = fit(Y, trt, H)["att"]
    scaled = fit(Y * level, trt, H)["att"] / level
    assert scaled == pytest.approx(base, rel=1e-8)


def test_the_weights_do_not_move_with_the_panels_magnitude():
    Y, trt, H = staggered_panel()
    base = fit(Y, trt, H)["weights"]
    scaled = fit(Y * 1e5, trt, H)["weights"]
    assert set(base) == set(scaled)
    for g in base:
        assert np.allclose(base[g], scaled[g], atol=1e-7)


def test_the_event_study_scales_with_the_panel():
    Y, trt, H = staggered_panel()
    base = np.asarray(fit(Y, trt, H)["per_time"], dtype=float)
    scaled = np.asarray(fit(Y * 1e5, trt, H)["per_time"], dtype=float) / 1e5
    assert np.allclose(scaled, base, rtol=1e-8, atol=1e-10)


def _tiny_program(seed=11, n=6, T=12, tj=8):
    """One cohort of two treated units and four donors, in residual space."""
    rng = np.random.default_rng(seed)
    res = {tj: rng.normal(size=(n, T))}
    groups = [tj]
    return dict(
        res=res, groups=groups, adopt_of={tj: tj},
        members={tj: [0, 1]}, donors={tj: np.array([2, 3, 4, 5])},
        n1=np.array([2.0]), d=tj, n=n, n_lags=tj,
    )


def _solve(p, nu, norm_pool, norm_sep, lam, divisor=1.0):
    res = {k: v / divisor for k, v in p["res"].items()}
    return solve_cohort_qp(res, p["groups"], p["adopt_of"], p["members"],
                           p["donors"], p["n1"], p["d"], p["n"], p["n_lags"],
                           nu, norm_pool, norm_sep, lam, None)


def test_the_separate_fits_ridge_is_divided_by_the_square_of_the_scale():
    """Its norms are 1.0, so dividing the residuals by ``s`` divides the
    imbalance by ``s ** 2``, and the ridge has to follow to leave the minimizer
    where it was."""
    p, s, lam = _tiny_program(), 16.0, 0.3
    base = _solve(p, 0.0, 1.0, 1.0, lam)
    rescaled = _solve(p, 0.0, 1.0, 1.0, lam / s ** 2, divisor=s)
    for g in base:
        assert np.allclose(base[g], rescaled[g], atol=1e-7)


def test_the_pooled_fits_ridge_is_left_alone():
    """Its norms carry the residuals' units, so dividing both by ``s`` leaves
    the imbalance term exactly as it was and the ridge with it."""
    p, s, lam = _tiny_program(), 16.0, 0.3
    base = _solve(p, 0.4, 2.5, 1.7, lam)
    rescaled = _solve(p, 0.4, 2.5 / s ** 2, 1.7 / s ** 2, lam, divisor=s)
    for g in base:
        assert np.allclose(base[g], rescaled[g], atol=1e-7)


def test_the_pooling_level_is_read_off_the_same_separate_fit_at_any_magnitude():
    """``nu_used`` comes entirely from the separate fit's imbalance matrix, so
    it is the run's window onto that fit. A ridge stated in the caller's units
    has to reach it the same way at any magnitude, which it does only because
    the ridge is divided by the same square as the imbalance it trades
    against."""
    Y, trt, H = staggered_panel()
    base = fit(Y, trt, H, lam=0.5)["nu_used"]
    scaled = fit(Y * 1e5, trt, H, lam=0.5 * 1e5 ** 2)["nu_used"]
    assert base == pytest.approx(0.244, abs=0.01)      # the ridge is biting
    assert scaled == pytest.approx(base, rel=1e-6)


def test_a_ridge_penalty_survives_the_rescale_end_to_end():
    Y, trt, H = staggered_panel()
    out = fit(Y * 1e5, trt, H, lam=0.05)
    assert np.isfinite(out["att"])
    for g in out["weights"]:
        assert np.all(np.isfinite(out["weights"][g]))


def test_covariates_are_carried_through_the_rescale():
    Y, trt, H = staggered_panel()
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(Y.shape[0], 2))
    base = fit(Y, trt, H, Z=Z)["att"]
    scaled = fit(Y * 1e5, trt, H, Z=Z)["att"] / 1e5
    assert scaled == pytest.approx(base, rel=1e-7)


# --- the untouched path -------------------------------------------------------

def test_a_near_unit_panel_is_solved_without_any_rescaling():
    """No pinned fit moves: the panels mlsynth already ships sit at unit scale.

    Asserted on the residuals the program actually sees, because that is what
    ``solve_scale`` reads -- a panel with a large mean and small variation has
    small residuals, and dividing it by its raw level would push the objective
    under the solver's absolute tolerance instead of onto it.
    """
    Y, trt, H = staggered_panel()
    assert solve_scale(fit(Y, trt, H)["res"]) == 1.0


def test_a_uniform_weight_fit_poses_no_program_and_is_unaffected():
    from mlsynth.utils.ppscm_helpers.engine import Conventions
    Y, trt, H = staggered_panel()
    conv = Conventions(donor_weights="uniform")
    base = fit(Y, trt, H, conventions=conv)["att"]
    scaled = fit(Y * 1e5, trt, H, conventions=conv)["att"] / 1e5
    assert scaled == pytest.approx(base, rel=1e-10)


def test_a_single_donor_panel_still_solves_when_rescaled():
    Y, trt, H = staggered_panel(n_donor=1, n_treat=2, H=4)
    out = fit(Y * 1e5, trt, H)
    assert np.isfinite(out["att"])

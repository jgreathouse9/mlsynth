"""ATEL: the sieve basis, the diversified projection, and the localized estimand.

ATEL (Lee 2026) estimates a kernel-weighted average effect localized at the
adoption date,

.. math::

   \\alpha = \\frac{1}{T_1} \\sum_{t > T_0}
            \\bigl( Y^I_{1t} - Y^N_{1t} \\bigr)\\,
            K_h\\!\\left( \\frac{t - T_0}{T_1} \\right),

so later post-periods count for less than early ones. The counterfactual comes
from a low-rank time-varying factor model: factors by diversified projection
(Fan and Liao 2022) -- a cross-sectional weighted average of donor outcomes,
with no eigendecomposition -- and a loading that moves with time, fit by local
linear regression on the pre-period and extrapolated forward.

Numbers marked as reference pins were produced by the author's MATLAB toolbox
(github.com/rueichilee/ATEL) run under Octave 8.4, on the two fixtures built
here. They are cross-validation targets, not this implementation's own output.

The B-spline basis is checked against scipy's ``BSpline.design_matrix``, which
is the same object the toolbox's ``spcol`` computes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

from mlsynth import ATEL
from mlsynth.config_models import ATELConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.atel_helpers.factors import diversified_factors
from mlsynth.utils.atel_helpers.inference import atel_variance
from mlsynth.utils.atel_helpers.loadings import (
    cv_bandwidth,
    half_epanechnikov,
    local_linear_loadings,
)
from mlsynth.utils.atel_helpers.sieve import (
    bspline_weights,
    construct_weights,
    poly_weights,
    trig_weights,
)

BASEDATA = "basedata/fdi_oecd_brexit.csv"


# --------------------------------------------------------------- fixtures
def _synthetic_panel():
    """A deterministic panel with a planted post-period shift of 4.0.

    No RNG, so the arrays are bit-identical to the ones handed to the toolbox.
    The outcome is ``10 + 3 x1 f1 + 2 x2 f2 + small wobble`` with the loadings
    driven by the covariates, which is the model ATEL assumes.
    """
    n_units, T, T0 = 9, 14, 9
    i = np.arange(n_units)[:, None]
    t = np.arange(T)[None, :]
    x1 = 0.5 + 0.3 * np.sin(0.7 * t + 0.4 * i)
    x2 = 0.2 + 0.4 * np.cos(0.5 * t + 0.9 * i)
    Y = (
        10.0
        + 3.0 * x1 * np.sin(0.9 * t)
        + 2.0 * x2 * np.cos(0.6 * t)
        + 0.10 * np.cos(2.1 * t + 1.3 * i)
    )
    Y[0, T0:] += 4.0
    return Y, np.stack([x1, x2], axis=2), T0


def _synthetic_long():
    """The same panel as a long frame, for the config path."""
    Y, X, T0 = _synthetic_panel()
    n_units, T = Y.shape
    rows = []
    for u in range(n_units):
        for p in range(T):
            rows.append(
                {
                    "unit": f"u{u}",
                    "period": p,
                    "y": Y[u, p],
                    "x1": X[u, p, 0],
                    "x2": X[u, p, 1],
                    "treat": int(u == 0 and p >= T0),
                }
            )
    return pd.DataFrame(rows), T0


@pytest.fixture
def synth():
    return _synthetic_panel()


@pytest.fixture
def synth_long():
    return _synthetic_long()


@pytest.fixture
def brexit():
    return pd.read_csv(BASEDATA)


def _cfg(df, **kw):
    base = dict(
        df=df,
        outcome="y",
        treat="treat",
        unitid="unit",
        time="period",
        covariates=["x1", "x2"],
        n_factors=2,
        display_graphs=False,
    )
    base.update(kw)
    return ATELConfig(**base)


# ------------------------------------------------------------ sieve basis
@pytest.mark.parametrize("J,order,n_basis", [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 4, 5)])
def test_bspline_basis_has_the_order_and_width_J_asks_for(J, order, n_basis):
    x = np.linspace(-1.3, 2.7, 40)
    B = bspline_weights(x, J)
    assert B.shape == (40, n_basis)


@pytest.mark.parametrize("J", [2, 3, 4, 5])
def test_bspline_basis_is_a_partition_of_unity(J):
    """Every B-spline basis of full width sums to one on its support."""
    B = bspline_weights(np.linspace(0.0, 1.0, 37), J)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("J", [2, 3, 4, 5])
def test_bspline_basis_matches_scipy(J):
    """The toolbox's augknt + spcol is scipy's design matrix at degree J-1."""
    x = np.linspace(-1.3, 2.7, 23)
    if J == 2:
        k, n_breaks = 2, 2
    elif J == 3:
        k, n_breaks = 3, 2
    else:
        k, n_breaks = 4, J - 2
    breaks = (
        np.array([x.min(), x.max()])
        if n_breaks == 2
        else np.linspace(x.min(), x.max(), n_breaks)
    )
    knots = np.concatenate(
        [np.repeat(breaks[0], k), breaks[1:-1], np.repeat(breaks[-1], k)]
    )
    want = BSpline.design_matrix(x, knots, k - 1, extrapolate=False).toarray()
    assert np.allclose(bspline_weights(x, J), want, atol=1e-13)


def test_bspline_basis_refuses_a_width_below_two():
    with pytest.raises(MlsynthConfigError):
        bspline_weights(np.linspace(0, 1, 10), 1)


def test_a_constant_covariate_is_refused():
    """Zero-width support gives a degenerate knot vector, not a basis."""
    with pytest.raises(MlsynthDataError):
        bspline_weights(np.full(12, 0.4), 3)


@pytest.mark.parametrize("J", [2, 3, 5])
def test_trig_basis_starts_with_an_intercept_and_pairs_up(J):
    W = trig_weights(np.linspace(0.0, 1.0, 19), J)
    assert W.shape == (19, J)
    assert np.allclose(W[:, 0], 1.0)


@pytest.mark.parametrize("J", [2, 4])
def test_poly_basis_is_the_monomials(J):
    x = np.linspace(0.2, 1.4, 11)
    W = poly_weights(x, J)
    assert W.shape == (11, J)
    for j in range(J):
        assert np.allclose(W[:, j], x ** (j + 1))


def test_construct_weights_lays_blocks_out_basis_major(synth):
    """``(basis j, covariate p)`` with p inner, T columns per block."""
    _Y, X, _T0 = synth
    n_units, T, P = X.shape
    J = 2
    W = construct_weights(X, J, "bspline")
    assert W.shape == (n_units, T * J * P)
    per_cov = [bspline_weights(X[:, :, p].ravel(order="F"), J) for p in range(P)]
    for j in range(J):
        for p in range(P):
            block = W[:, (j * P + p) * T : (j * P + p + 1) * T]
            want = per_cov[p][:, j].reshape(n_units, T, order="F")
            assert np.allclose(block, want)


# --------------------------------------------------- diversified projection
def test_a_factor_is_a_donor_average_of_weighted_outcomes(synth):
    _Y, X, _T0 = synth
    Y, _X, _ = synth
    W = construct_weights(X, 2, "bspline")
    F = diversified_factors(Y[1:], W, 2)
    T = Y.shape[1]
    for j in range(2):
        want = (Y[1:] * W[1:, j * T : (j + 1) * T]).mean(axis=0)
        assert np.allclose(F[:, j], want)


def test_the_projection_is_linear_in_the_outcomes(synth):
    """No eigendecomposition anywhere: doubling Y doubles every factor."""
    Y, X, _T0 = synth
    W = construct_weights(X, 2, "bspline")
    F1 = diversified_factors(Y[1:], W, 2)
    F2 = diversified_factors(2.0 * Y[1:], W, 2)
    assert np.allclose(F2, 2.0 * F1, atol=1e-12)


def test_the_projection_ignores_donor_order(synth):
    """A mean over donors cannot depend on how the donors are listed."""
    Y, X, _T0 = synth
    J = 2
    W = construct_weights(X, J, "bspline")
    F = diversified_factors(Y[1:], W, J)
    order = np.array([5, 0, 3, 7, 1, 6, 2, 4])
    Xs = np.concatenate([X[:1], X[1:][order]], axis=0)
    Ws = construct_weights(Xs, J, "bspline")
    Fs = diversified_factors(Y[1:][order], Ws, J)
    assert np.allclose(F, Fs, atol=1e-12)


def test_the_projection_consumes_the_first_J_of_the_J_times_P_blocks(synth):
    """DP takes ``1+(j-1)T : jT`` for j = 1..J out of a J*P-block layout.

    With two covariates and J = 3 that is basis 1 of both covariates and basis 2
    of only the first, which is why the config refuses such a J.
    """
    Y, X, _T0 = synth
    J, P = 3, 2
    W = construct_weights(X, J, "bspline")
    F = diversified_factors(Y[1:], W, J)
    T = Y.shape[1]
    consumed = [(b // P, b % P) for b in range(J)]
    assert consumed == [(0, 0), (0, 1), (1, 0)]
    for j in range(J):
        assert np.allclose(F[:, j], (Y[1:] * W[1:, j * T : (j + 1) * T]).mean(axis=0))


# ------------------------------------------------------------- the kernel
def test_the_half_kernel_doubles_the_epanechnikov_mass():
    u = np.linspace(-1.0, 0.0, 9)
    assert np.allclose(half_epanechnikov(u, "left"), 2 * 0.75 * (1 - u**2))


@pytest.mark.parametrize(
    "side,outside", [("left", np.array([-1.4, -2.0])), ("right", np.array([1.4, 2.0]))]
)
def test_the_half_kernel_is_zero_off_its_side(side, outside):
    assert np.allclose(half_epanechnikov(outside, side), 0.0)


def test_the_kernel_downweights_later_post_periods():
    """The localization: weight falls monotonically away from the adoption date."""
    u = np.linspace(0.05, 0.95, 10)
    w = half_epanechnikov(u, "right")
    assert np.all(np.diff(w) < 0)


# ------------------------------------------------------------ bandwidth CV
def test_cv_returns_a_point_of_its_own_grid(synth):
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    h = cv_bandwidth(Y[0, :T0], F[:T0])
    assert h in set(np.round(np.linspace(0.30, 0.95, 14), 10))


def test_cv_picks_an_interior_bandwidth_on_this_panel(synth):
    """Reference pin: the toolbox selects h = 0.90 here, not the grid edge."""
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    assert cv_bandwidth(Y[0, :T0], F[:T0]) == pytest.approx(0.90)


# ------------------------------------------------------ time-varying loading
def test_the_loading_has_one_column_per_post_period(synth):
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    beta = local_linear_loadings(Y[0], F[:T0], 0.8)
    assert beta.shape == (2, Y.shape[1] - T0)


def test_a_loading_constant_in_time_is_recovered_exactly(synth):
    """With Y_1t exactly F_t'b for a fixed b, the local linear fit returns b.

    The slope term must come out at zero, so every post-period column equals b.
    """
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    b = np.array([1.7, -0.6])
    y = np.concatenate([F[:T0] @ b, F[T0:] @ b])
    beta = local_linear_loadings(y, F[:T0], 0.9)
    assert np.allclose(beta, b[:, None], atol=1e-8)


# -------------------------------------------------------- reference pins
@pytest.mark.parametrize(
    "J,atel,se,p",
    [
        (2, 4.202070821323, 0.232805116266, 0.000370919765),
        (4, 4.724570275491, 0.018144927928, 0.000000124918),
    ],
)
def test_synthetic_panel_matches_the_matlab_toolbox(synth_long, J, atel, se, p):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, n_factors=J)).fit()
    assert res.atel == pytest.approx(atel, rel=1e-9)
    assert res.inference.standard_error == pytest.approx(se, rel=1e-9)
    assert res.inference.p_value == pytest.approx(p, rel=1e-7)


def test_a_fixed_bandwidth_matches_the_toolbox(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, n_factors=2, bandwidth=0.7)).fit()
    assert res.atel == pytest.approx(2.447256625552, rel=1e-9)
    assert res.inference.standard_error == pytest.approx(0.259627402394, rel=1e-9)
    assert res.bandwidth == pytest.approx(0.7)


@pytest.mark.parametrize(
    "J,atel,se",
    [(2, -1.092280148169, 1.493840763116), (4, -2.004180386448, 1.542584609115)],
)
def test_brexit_panel_matches_the_matlab_toolbox(brexit, J, atel, se):
    res = ATEL(
        ATELConfig(
            df=brexit,
            outcome="fdi",
            treat="treated",
            unitid="country",
            time="year",
            covariates=["log_gdp", "log_gdp_percap"],
            n_factors=J,
            display_graphs=False,
        )
    ).fit()
    assert res.atel == pytest.approx(atel, rel=1e-8)
    assert res.inference.standard_error == pytest.approx(se, rel=1e-8)


# ------------------------------------------------------------- recovery
def test_a_planted_constant_shift_is_recovered_exactly(synth_long):
    """When the counterfactual is exactly recoverable, the estimate is exact.

    Put the treated unit's untreated path exactly in the span of the factors with
    a loading constant in time, so the local linear fit reproduces it and the
    post-period gap is the planted shift on the nose. ATEL is then the shift
    times the kernel mass -- the weights do not sum to one, so the target is not
    the shift itself.
    """
    df, T0 = synth_long
    Y, X, _ = _synthetic_panel()
    T = Y.shape[1]

    # The factors depend only on the donors and the covariates, so the treated
    # unit's outcome can be rebuilt on top of them without changing them.
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    loading = np.array([1.3, -0.45])
    shift = 4.0
    exact = F @ loading
    exact[T0:] += shift

    rebuilt = df.copy()
    treated_rows = rebuilt["unit"] == "u0"
    rebuilt.loc[treated_rows, "y"] = exact[
        rebuilt.loc[treated_rows, "period"].to_numpy()
    ]

    res = ATEL(_cfg(rebuilt, n_factors=2)).fit()
    T1h = int(np.floor((T - T0) * res.bandwidth))
    mass = float(
        half_epanechnikov((np.arange(T0 + 1, T + 1) - T0) / T1h, "right").sum() / T1h
    )
    assert res.atel == pytest.approx(shift * mass, rel=1e-8)
    assert res.kernel_mass == pytest.approx(mass, rel=1e-12)
    # the pre-period fit is exact, so the residuals carry no signal
    assert np.abs(res.time_series.observed_outcome[:T0]
                  - res.time_series.counterfactual_outcome[:T0]).max() < 1e-8


# ----------------------------------------------------------- invariance
def test_the_estimate_ignores_covariate_order_when_J_is_a_multiple_of_P(synth_long):
    """The reason the config constrains J: only then is the block slice complete.

    With J a multiple of P the first J blocks are a full set of (basis,
    covariate) pairs, so relabelling the covariates permutes the factors and
    leaves their span, hence the fit, alone.
    """
    df, _T0 = synth_long
    a = ATEL(_cfg(df, n_factors=2, covariates=["x1", "x2"])).fit().atel
    b = ATEL(_cfg(df, n_factors=2, covariates=["x2", "x1"])).fit().atel
    assert a == pytest.approx(b, rel=1e-10)


def test_a_scaled_outcome_scales_the_estimate(synth_long):
    df, _T0 = synth_long
    base = ATEL(_cfg(df, n_factors=2)).fit().atel
    scaled = df.assign(y=df["y"] * 3.0)
    assert ATEL(_cfg(scaled, n_factors=2)).fit().atel == pytest.approx(3.0 * base, rel=1e-9)


# ----------------------------------------------------------- the contract
def test_fit_populates_the_standardized_submodels(synth_long):
    df, T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    T = int(df["period"].nunique())
    assert res.effects.att is not None and np.isfinite(res.effects.att)
    assert res.inference.ci_lower < res.atel < res.inference.ci_upper
    assert res.method_details.method_name == "ATEL"
    assert res.time_series.counterfactual_outcome is not None
    assert len(res.time_series.counterfactual_outcome) == T
    assert res.fit_diagnostics is not None
    assert res.n_factors == 2
    assert res.factors.shape == (T, 2)
    assert res.loadings.shape == (2, T - T0)
    assert res.kernel_weights.shape == (T - T0,)
    assert res.pointwise_standard_errors.shape == (T - T0,)


def test_atel_is_reported_separately_from_the_plain_att(synth_long):
    """They are different estimands: att is the unweighted post-period mean."""
    df, _T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    assert res.effects.additional_effects["atel"] == pytest.approx(res.atel)
    assert res.effects.att != pytest.approx(res.atel)


# ------------------------------------------------------------ edge cases
def test_a_single_covariate_needs_no_multiple_of_P(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, covariates=["x1"], n_factors=3)).fit()
    assert np.isfinite(res.atel)


def test_a_single_post_period_is_refused(synth_long):
    """The localization window is ``floor(T1 * h)``, which is 0 when T1 = 1.

    Every bandwidth on the grid is below 1, so one post-period leaves the kernel
    no window to average over and there is no localized estimand to report.
    """
    df, T0 = synth_long
    trimmed = df[df["period"] <= T0].copy()
    with pytest.raises(MlsynthDataError, match="window"):
        ATEL(_cfg(trimmed)).fit()


def test_two_post_periods_are_enough(synth_long):
    df, T0 = synth_long
    trimmed = df[df["period"] <= T0 + 1].copy()
    res = ATEL(_cfg(trimmed, bandwidth=0.95)).fit()
    assert np.isfinite(res.atel)
    assert res.loadings.shape[1] == 2


def test_a_near_collinear_covariate_pair_does_not_break_the_fit(synth_long):
    df, _T0 = synth_long
    df = df.assign(x2=lambda d: d["x1"] + 1e-9 * d["x2"])
    res = ATEL(_cfg(df)).fit()
    assert np.isfinite(res.atel)


def test_the_minimum_donor_pool_is_one(synth_long):
    df, _T0 = synth_long
    keep = {"u0", "u1"}
    res = ATEL(_cfg(df[df["unit"].isin(keep)].copy())).fit()
    assert np.isfinite(res.atel)


# -------------------------------------------------------------- failures
def test_a_factor_count_not_a_multiple_of_the_covariate_count_is_refused(synth_long):
    """The order-dependent configurations are refused at construction.

    Measured on the paper's own panel: J = 3 moves the estimate by 22.6 and
    J = 5 by 71.4 when the two covariates are passed in the other order.
    """
    df, _T0 = synth_long
    with pytest.raises(MlsynthConfigError, match="multiple"):
        _cfg(df, n_factors=3, covariates=["x1", "x2"])


def test_a_consumed_weight_block_of_zeros_is_refused(synth_long):
    """A factor identically zero is refused instead of returned.

    The trigonometric basis fills an intercept plus ``floor((J-1)/2)`` cos/sin
    pairs, so it leaves trailing columns at zero for some J. Whether one of
    those reaches the projection depends on J and the covariate count together,
    which is checked where the blocks are consumed and not by case analysis on
    the config.
    """
    df, _T0 = synth_long
    with pytest.raises(MlsynthEstimationError, match="zero"):
        ATEL(_cfg(df, basis="trigonometric", n_factors=2, covariates=["x1"])).fit()


def test_the_trigonometric_basis_is_usable_where_the_blocks_are_filled(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, basis="trigonometric", n_factors=3, covariates=["x1"])).fit()
    assert np.isfinite(res.atel)


def test_the_polynomial_basis_runs_end_to_end(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df, basis="polynomial", n_factors=2)).fit()
    assert np.isfinite(res.atel)


def test_a_factor_count_below_two_is_refused(synth_long):
    df, _T0 = synth_long
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, n_factors=1)


def test_no_covariates_is_refused(synth_long):
    """The diversified weights are built from covariates; there is no fallback."""
    df, _T0 = synth_long
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, covariates=[])


def test_an_unknown_basis_is_refused(synth_long):
    df, _T0 = synth_long
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, basis="wavelet")


def test_a_bandwidth_outside_the_unit_interval_is_refused(synth_long):
    df, _T0 = synth_long
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, bandwidth=1.5)
    with pytest.raises((MlsynthConfigError, ValueError)):
        _cfg(df, bandwidth=0.0)


def test_a_missing_covariate_column_is_reported(synth_long):
    df, _T0 = synth_long
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        ATEL(_cfg(df, covariates=["x1", "nope"])).fit()


def test_an_unbalanced_panel_is_reported(synth_long):
    df, _T0 = synth_long
    with pytest.raises(MlsynthDataError):
        ATEL(_cfg(df.drop(index=df.index[5]).copy())).fit()


def test_a_covariate_with_a_missing_cell_is_reported(synth_long):
    df, _T0 = synth_long
    df = df.copy()
    df.loc[df.index[7], "x1"] = np.nan
    with pytest.raises(MlsynthDataError):
        ATEL(_cfg(df)).fit()


def test_too_few_pre_periods_for_the_local_linear_fit_is_reported(synth_long):
    """The design has 2J columns, so the pre-period must be at least that long."""
    df, T0 = synth_long
    df = df[(df["period"] >= T0 - 2)].copy()
    df["period"] = df["period"] - (T0 - 2)
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        ATEL(_cfg(df, n_factors=4)).fit()


# ----------------------------------------------------------------- plots
def test_plotting_is_off_by_default_in_these_tests_and_returns_a_figure(synth_long):
    """Computation and presentation stay separate: the helper returns a Figure."""
    import matplotlib
    matplotlib.use("Agg")
    from mlsynth.utils.atel_helpers.plotter import plot_atel

    df, _T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    fig = plot_atel(res)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_the_bandwidth_at_the_grid_edge_is_surfaced(brexit):
    """A corner solution is a diagnostic the caller can act on, so it warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ATEL(
            ATELConfig(
                df=brexit,
                outcome="fdi",
                treat="treated",
                unitid="country",
                time="year",
                covariates=["log_gdp", "log_gdp_percap"],
                n_factors=2,
                display_graphs=False,
            )
        ).fit()
    assert any("grid" in str(w.message).lower() for w in caught)


# --------------------------------------------- helper-level guards and edges
def test_the_estimator_accepts_a_config_dict(synth_long):
    df, _T0 = synth_long
    res = ATEL(
        dict(
            df=df, outcome="y", treat="treat", unitid="unit", time="period",
            covariates=["x1", "x2"], n_factors=2, display_graphs=False,
        )
    ).fit()
    assert np.isfinite(res.atel)


def test_the_estimator_refuses_a_foreign_config():
    with pytest.raises(MlsynthConfigError, match="ATELConfig"):
        ATEL(object())


def test_the_projection_refuses_malformed_arguments(synth):
    Y, X, _T0 = synth
    W = construct_weights(X, 2, "bspline")
    with pytest.raises(MlsynthEstimationError, match="2-D"):
        diversified_factors(Y[1:, 0], W, 2)
    with pytest.raises(MlsynthEstimationError, match="treated unit"):
        diversified_factors(Y, W, 2)
    with pytest.raises(MlsynthEstimationError, match="blocks"):
        diversified_factors(Y[1:], W[:, :3], 2)


def test_the_variance_refuses_an_empty_window(synth):
    from mlsynth.utils.atel_helpers.inference import pre_period_residuals

    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    with pytest.raises(MlsynthDataError, match="window"):
        pre_period_residuals(Y[0], F[:T0], 1e-6)
    with pytest.raises(MlsynthDataError, match="window"):
        atel_variance(Y[0], F[:T0], F[T0:], 1e-6)


def test_the_bandwidth_search_refuses_mismatched_factors():
    with pytest.raises(MlsynthDataError, match="pre-period factors"):
        cv_bandwidth(np.arange(6.0), np.zeros((4, 2)))


def test_the_bandwidth_search_skips_grid_points_with_no_window():
    """With three pre-periods the low end of the grid has an empty window."""
    y = np.array([1.0, 2.0, 1.5])
    F = np.array([[1.0], [0.5], [1.2]])
    h = cv_bandwidth(y, F)
    assert h in set(np.round(np.linspace(0.30, 0.95, 14), 10))
    assert np.floor(3 * h) >= 1


def test_the_loading_fit_refuses_a_panel_with_no_post_period(synth):
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    with pytest.raises(MlsynthDataError, match="post-treatment"):
        local_linear_loadings(Y[0, :T0], F[:T0], 0.9)


def test_the_loading_fit_refuses_an_empty_window(synth):
    Y, X, T0 = synth
    F = diversified_factors(Y[1:], construct_weights(X, 2, "bspline"), 2)
    with pytest.raises(MlsynthDataError, match="window"):
        local_linear_loadings(Y[0], F[:T0], 1e-6)


def test_ingestion_refuses_an_empty_covariate_list(synth_long):
    from mlsynth.utils.atel_helpers.setup import prepare_atel_inputs

    df, _T0 = synth_long
    with pytest.raises(MlsynthDataError, match="at least one covariate"):
        prepare_atel_inputs(df, "y", "treat", "unit", "period", [], 2)


def test_ingestion_names_a_column_it_cannot_find(synth_long):
    from mlsynth.utils.atel_helpers.setup import prepare_atel_inputs

    df, _T0 = synth_long
    with pytest.raises(MlsynthDataError, match="not found"):
        prepare_atel_inputs(df, "y", "treat", "unit", "period", ["absent"], 2)


def test_ingestion_refuses_a_non_finite_covariate(synth_long):
    from mlsynth.utils.atel_helpers.setup import prepare_atel_inputs

    df, _T0 = synth_long
    df = df.copy()
    df.loc[df.index[3], "x1"] = np.inf
    with pytest.raises(MlsynthDataError, match="non-finite"):
        prepare_atel_inputs(df, "y", "treat", "unit", "period", ["x1", "x2"], 2)


def test_the_basis_refuses_non_finite_values():
    x = np.array([0.1, 0.4, np.nan, 0.8])
    with pytest.raises(MlsynthDataError):
        bspline_weights(x, 3)


def test_an_unknown_basis_name_is_named_in_the_error():
    from mlsynth.utils.atel_helpers.sieve import basis_values

    with pytest.raises(MlsynthConfigError, match="Unknown basis"):
        basis_values(np.linspace(0, 1, 8), 2, "wavelet")


def test_a_two_dimensional_cube_is_read_as_one_covariate(synth):
    _Y, X, _T0 = synth
    flat = construct_weights(X[:, :, 0], 2, "bspline")
    cubed = construct_weights(X[:, :, :1], 2, "bspline")
    assert np.allclose(flat, cubed)


def test_the_weights_refuse_a_cube_of_the_wrong_rank():
    with pytest.raises(MlsynthDataError, match="2-D or 3-D"):
        construct_weights(np.arange(6.0), 2, "bspline")


def test_the_weights_refuse_a_cube_with_no_covariates():
    with pytest.raises(MlsynthDataError, match="at least"):
        construct_weights(np.zeros((4, 5, 0)), 2, "bspline")


def test_the_weights_refuse_a_non_finite_cube():
    X = np.zeros((4, 5, 1))
    X[0, 0, 0] = np.nan
    with pytest.raises(MlsynthDataError, match="non-finite"):
        construct_weights(X, 2, "bspline")


def test_the_inputs_expose_the_panel_shape(synth_long):
    df, T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    inputs = res.inputs
    assert inputs.n_donors == 8
    assert inputs.n_periods == 14
    assert inputs.n_post == 14 - T0
    assert inputs.treated_unit_name == "u0"
    assert inputs.covariate_names == ("x1", "x2")


def test_the_display_path_saves_a_figure(synth_long, tmp_path):
    """The display branch renders and saves without leaving a window open."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, _T0 = synth_long
    target = tmp_path / "atel_plot.png"
    ATEL(_cfg(df, display_graphs=True, save=str(target))).fit()
    assert target.exists() and target.stat().st_size > 0
    plt.close("all")


# ------------------------------------------------- implied donor weights
def test_the_counterfactual_is_a_weighted_sum_of_donor_outcomes(synth_long):
    """ATEL is a donor-weighting estimator, with weights that move over time.

    Substituting the projection into the counterfactual,

        Yhat_1t = sum_j beta_jt F_tj
                = sum_i Y_it * [ (1/N) sum_j beta_jt W_it^(j) ],

    so each donor carries an implied weight at each post-period. The identity is
    exact by construction, which makes it a check on the reported weights and
    not on the arithmetic.
    """
    df, T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    Y = res.inputs.outcomes
    donors = Y[1:]
    omega = res.implied_donor_weights
    assert omega.shape == (donors.shape[0], res.kernel_weights.size)
    rebuilt = (donors[:, T0:] * omega).sum(axis=0)
    assert np.allclose(
        rebuilt, res.time_series.counterfactual_outcome[T0:], atol=1e-9
    )


def test_the_weights_container_carries_each_donor(synth_long):
    df, _T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    assert res.weights is not None
    assert not res.weights.is_empty
    names = list(res.inputs.unit_labels[1:])
    assert set(res.donor_weights) == set(names)


def test_the_reported_donor_weight_is_the_localized_average(synth_long):
    """The scalar per donor is its implied weight averaged against the kernel."""
    df, _T0 = synth_long
    res = ATEL(_cfg(df)).fit()
    window = res.diagnostics["post_window"]
    want = (res.implied_donor_weights * res.kernel_weights).sum(axis=1) / window
    got = np.array([res.donor_weights[n] for n in res.inputs.unit_labels[1:]])
    assert np.allclose(got, want, atol=1e-12)

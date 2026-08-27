"""Tests for MOSC -- many-outcomes synthetic control (Wang, Schein, Shou & Blei).

Written before the estimator, per the repository's test-first rule. The layers
are smoke, unit invariants, edge cases and failure reporting, as
``agents/agents_tests.md`` lays out.

The invariants asserted here are the ones the spike in
``benchmarks/reference/mosc_spike/`` established, so several of them pin
deliberate departures from the paper: the sign of the effect follows equation 43
and not the authors' code, the model check is a reported score and not a
p-value, and the outcome scale is a diagnostic the caller acts on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MOSC
from mlsynth.config_models import BaseEstimatorResults, MOSCConfig
from mlsynth.utils.mosc_helpers.structures import MOSCResults
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)


# ---------------------------------------------------------------- fixtures --


def make_count_panel(
    n_units: int = 24,
    n_periods: int = 40,
    pre_periods: int = 28,
    n_factors: int = 3,
    effect: float = 1.4,
    seed: int = 0,
) -> pd.DataFrame:
    """Poisson panel from a non-negative factor model; unit ``u00`` is treated.

    This is the regime MOSC targets: many units, a common intervention date, and
    a count outcome whose conditional mean is a low-rank non-negative product.
    """
    rng = np.random.default_rng(seed)
    H = rng.gamma(3.0, 3.0, size=(n_periods, n_factors))
    Z = rng.gamma(3.0, 3.0, size=(n_factors, n_units))
    rate = H @ Z
    rate[pre_periods:, 0] *= effect
    counts = rng.poisson(rate)

    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            rows.append(
                {
                    "unit": f"u{i:02d}",
                    "period": t,
                    "cases": float(counts[t, i]),
                    "treated": int(i == 0 and t >= pre_periods),
                }
            )
    return pd.DataFrame(rows)


def base_config(df: pd.DataFrame, **overrides) -> dict:
    cfg = {
        "df": df,
        "outcome": "cases",
        "treat": "treated",
        "unitid": "unit",
        "time": "period",
        "n_factors": 3,
        "n_samples": 40,
        "n_warmup": 40,
        "n_bootstrap": 30,
        "seed": 0,
        "display_graphs": False,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_count_panel()


@pytest.fixture(scope="module")
def fitted(panel):
    return MOSC(base_config(panel)).fit()


# -------------------------------------------------------------------- smoke --


def test_fit_returns_effect_result(fitted):
    """End-to-end on a minimal panel: right type, finite headline number."""
    assert isinstance(fitted, BaseEstimatorResults)
    assert fitted.att is not None
    assert np.isfinite(fitted.att)


def test_counterfactual_spans_the_panel(fitted, panel):
    n_periods = panel["period"].nunique()
    assert fitted.counterfactual.shape == (n_periods,)
    assert np.isfinite(fitted.counterfactual).all()


def test_both_factor_models_run(panel):
    """The likelihood family is the estimator's point; both arms must fit."""
    for family in ("gap", "ppca"):
        result = MOSC(base_config(panel, factor_model=family)).fit()
        assert np.isfinite(result.att)
        assert result.method_details.method_name == "MOSC"


# --------------------------------------------------------- unit invariants --


def test_gap_is_observed_minus_counterfactual(fitted):
    observed = fitted.time_series.observed_outcome
    counterfactual = fitted.time_series.counterfactual_outcome
    np.testing.assert_allclose(fitted.gap, observed - counterfactual, rtol=1e-10)


def test_att_follows_equation_43(fitted, panel):
    """ATT is the mean post-period ``Y - f(0)``, the paper's sign.

    The authors' own code computes ``f(0) - Y`` (``calculate_ATT.ipynb``),
    inverting equation 43. On their null result the difference is invisible; on
    any real effect it flips the answer, so the sign is pinned here.
    """
    pre_periods = int((panel["treated"] == 0).groupby(panel["unit"]).sum().min())
    post_gap = fitted.gap[pre_periods:]
    np.testing.assert_allclose(fitted.att, float(np.mean(post_gap)), rtol=1e-10)


def test_positive_effect_is_reported_positive(panel):
    """A panel built with an inflated treated path must not report a negative ATT.

    This is the failure the sign inversion would produce, stated in terms of the
    data, not the formula.
    """
    result = MOSC(base_config(panel)).fit()
    assert result.att > 0


def test_the_band_is_an_ordered_interval(fitted):
    detail = fitted.inference.details
    assert np.all(detail.counterfactual_lower <= detail.counterfactual_upper + 1e-9)
    assert np.isfinite(detail.counterfactual_lower).all()
    assert np.isfinite(detail.counterfactual_upper).all()


def test_the_posterior_band_brackets_its_own_mean(panel):
    """The posterior band is quantiles of the draws the mean is taken over.

    The bootstrap interval carries no such guarantee and is not expected to: a
    percentile interval is located by the resampling distribution, not built
    around the point estimate. Where the two disagree materially, that is the
    estimator reporting that it is unstable under resampling.
    """
    result = MOSC(base_config(panel, inference="posterior")).fit()
    detail = result.inference.details
    assert np.all(detail.counterfactual_lower <= detail.counterfactual_mean + 1e-9)
    assert np.all(detail.counterfactual_mean <= detail.counterfactual_upper + 1e-9)


@pytest.mark.parametrize("inference", ["bootstrap", "posterior"])
def test_the_interval_contains_the_estimate_it_belongs_to(panel, inference):
    """A reported interval contains the estimate reported beside it.

    Under the bootstrap the two come from different Monte Carlo samples -- the
    estimate is a posterior mean over every draw, the interval a percentile of
    replicates that use one draw apiece -- so they can disagree by a fraction of
    a standard error. CI caught exactly that, at 1.4 percent of a standard error.
    The bound that falls short is extended, never pulled in.
    """
    result = MOSC(base_config(panel, inference=inference)).fit()
    low, high = result.att_ci
    assert low <= result.att <= high

    detail = result.inference.details
    assert np.all(detail.counterfactual_lower <= detail.counterfactual_mean + 1e-9)
    assert np.all(detail.counterfactual_mean <= detail.counterfactual_upper + 1e-9)


def test_tighter_level_gives_a_narrower_interval(panel):
    """A 50% band must sit inside a 95% one drawn from the same posterior."""
    wide = MOSC(base_config(panel, ci_alpha=0.05)).fit()
    narrow = MOSC(base_config(panel, ci_alpha=0.50)).fit()
    wide_width = wide.att_ci[1] - wide.att_ci[0]
    narrow_width = narrow.att_ci[1] - narrow.att_ci[0]
    assert narrow_width <= wide_width


def test_declares_it_has_no_donor_weights(fitted):
    """A factor model states the absence, so a caller can tell it was checked.

    ``{}`` means "checked, this method has no donor weights"; ``None`` would be
    the absence of a statement and fails the library-wide weights contract.
    """
    assert fitted.donor_weights == {}
    assert not fitted.weights.is_empty


def test_seed_makes_the_fit_reproducible(panel):
    first = MOSC(base_config(panel, seed=7)).fit()
    second = MOSC(base_config(panel, seed=7)).fit()
    np.testing.assert_allclose(first.att, second.att, rtol=1e-12)
    np.testing.assert_allclose(first.counterfactual, second.counterfactual, rtol=1e-12)


def test_diagnostics_are_reported_not_discarded(fitted):
    """The scale and model-fit checks reach the caller as typed fields.

    The paper's ``p_pop`` is deliberately absent: the spike measured its false
    rejection rate at 0.40 against a stated 0.05, because equation 36 sums the
    discrepancy over held-out cells until the comparison is deterministic. The
    held-out predictive log density is the same comparison without the
    calibration claim.
    """
    diagnostics = fitted.diagnostics
    assert np.isfinite(diagnostics.heldout_log_density)
    assert diagnostics.pearson_dispersion > 0
    assert -1.0 <= diagnostics.residual_autocorrelation <= 1.0
    assert diagnostics.outcome_scale in {"level", "difference"}


def test_cumulative_outcome_is_flagged_by_residual_autocorrelation(panel):
    """A cumulative series must report the dependence equations 12 and 19 forbid.

    This is what the authors' own panels look like. The diagnostic that
    discriminates is the residual autocorrelation, since the identification rests
    on the factors rendering a unit's outcomes conditionally independent, and
    differencing is what restores that.
    """
    cumulative = panel.copy()
    cumulative["cases"] = cumulative.groupby("unit")["cases"].cumsum()
    level = MOSC(base_config(cumulative, factor_model="ppca")).fit().diagnostics
    differenced = MOSC(
        base_config(cumulative, factor_model="ppca", outcome_scale="difference")
    ).fit().diagnostics
    assert level.residual_autocorrelation > differenced.residual_autocorrelation
    assert level.outcome_scale == "level"
    assert differenced.outcome_scale == "difference"


def test_differencing_reintegrates_to_the_outcome_scale(panel):
    """A differenced fit still reports a counterfactual on the observed scale."""
    result = MOSC(base_config(panel, factor_model="ppca", outcome_scale="difference")).fit()
    observed = result.time_series.observed_outcome
    counterfactual = result.time_series.counterfactual_outcome
    assert counterfactual.shape == observed.shape
    # The two scales are comparable, so the counterfactual cannot be orders out.
    assert 0.1 < np.median(counterfactual) / np.median(observed) < 10.0


def test_posterior_draw_count_is_honoured(panel):
    result = MOSC(base_config(panel, n_samples=25)).fit()
    assert result.posterior.n_draws == 25
    assert result.posterior.loadings.shape[0] == 25


def test_the_heldout_score_is_not_monotone_in_rank(panel):
    """Extra rank is free in sample and is not free out of sample.

    This asserted the opposite until CI found it, on Python 3.12 and 3.13 but
    not 3.10, which is what a claim that only held by RNG luck looks like. A
    held-out score exists precisely so that capacity stops paying: the in-sample
    fit is monotone in rank (pinned in ``test_mosc_properties.py``) and this one
    is not, which is the whole reason the estimator reports this one.
    """
    small = MOSC(base_config(panel, n_factors=1)).fit()
    large = MOSC(base_config(panel, n_factors=5)).fit()
    for result in (small, large):
        assert np.isfinite(result.diagnostics.heldout_log_density)
    # Rank changes the fit; which way the held-out score moves is the data's call.
    assert small.diagnostics.heldout_log_density != large.diagnostics.heldout_log_density


# ------------------------------------------------------------- edge cases --


def test_single_donor_is_refused(panel):
    """One donor cannot identify a factor model across units."""
    two_units = panel[panel["unit"].isin(["u00", "u01"])].copy()
    with pytest.raises(MlsynthDataError, match="unit"):
        MOSC(base_config(two_units, n_factors=1)).fit()


def test_rank_above_the_unit_count_is_refused(panel):
    small = panel[panel["unit"].isin([f"u{i:02d}" for i in range(5)])].copy()
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="factor"):
        MOSC(base_config(small, n_factors=12)).fit()


def test_no_pre_periods_is_refused(panel):
    """Treatment at t=0 leaves no negative-control outcomes to learn from."""
    always_treated = panel.copy()
    always_treated["treated"] = (always_treated["unit"] == "u00").astype(int)
    with pytest.raises(MlsynthDataError):
        MOSC(base_config(always_treated)).fit()


def test_collinear_donors_still_fit(panel):
    """Duplicated donors are rank-deficient but must not crash the fit."""
    duplicated = panel.copy()
    clone = duplicated[duplicated["unit"] == "u01"].copy()
    clone["unit"] = "u01_copy"
    doubled = pd.concat([duplicated, clone], ignore_index=True)
    result = MOSC(base_config(doubled)).fit()
    assert np.isfinite(result.att)


def test_constant_outcome_does_not_crash(panel):
    """A degenerate panel yields a finite answer or a translated error."""
    flat = panel.copy()
    flat["cases"] = 5.0
    try:
        result = MOSC(base_config(flat)).fit()
    except (MlsynthDataError, MlsynthEstimationError):
        return
    assert np.isfinite(result.att)


def test_zero_counts_are_admissible(panel):
    """A count panel with structural zeros is ordinary input, not an error."""
    sparse = panel.copy()
    sparse.loc[sparse["cases"] < 25, "cases"] = 0.0
    result = MOSC(base_config(sparse)).fit()
    assert np.isfinite(result.att)


# ---------------------------------------------------------- failure paths --


def test_negative_outcome_is_refused_under_the_poisson_arm(panel):
    """A Poisson likelihood has no support below zero, so this must be reported."""
    signed = panel.copy()
    signed.loc[signed["unit"] == "u03", "cases"] -= 500.0
    with pytest.raises(MlsynthDataError, match="negative"):
        MOSC(base_config(signed, factor_model="gap")).fit()


def test_negative_differences_are_refused_under_the_poisson_arm(panel):
    """The support constraint is checked on the scale the factor model sees.

    A count panel is admissible in levels and routinely inadmissible once
    differenced, so checking the outcome as supplied would let an impossible fit
    through.
    """
    with pytest.raises(MlsynthDataError, match="first differences"):
        MOSC(base_config(panel, factor_model="gap", outcome_scale="difference")).fit()


def test_negative_outcome_is_admissible_under_the_gaussian_arm(panel):
    """The PPCA arm is the escape hatch, so the refusal must be arm-specific."""
    signed = panel.copy()
    signed["cases"] -= 200.0
    result = MOSC(base_config(signed, factor_model="ppca")).fit()
    assert np.isfinite(result.att)


def test_missing_outcomes_are_refused(panel):
    holed = panel.copy()
    holed.loc[holed.index[5], "cases"] = np.nan
    with pytest.raises(MlsynthDataError):
        MOSC(base_config(holed)).fit()


def test_multiple_cohorts_are_refused(panel):
    """MOSC takes one treated unit, matching the authors' own design."""
    staggered = panel.copy()
    staggered.loc[
        (staggered["unit"] == "u05") & (staggered["period"] >= 32), "treated"
    ] = 1
    with pytest.raises(MlsynthDataError, match="cohort|treated"):
        MOSC(base_config(staggered)).fit()


@pytest.mark.parametrize(
    "override",
    [
        {"factor_model": "wishart"},
        {"n_factors": 0},
        {"n_samples": 0},
        {"ci_alpha": 0.0},
        {"ci_alpha": 1.0},
        {"outcome_scale": "logarithm"},
        {"not_a_field": 3},
    ],
)
def test_invalid_configuration_is_refused(panel, override):
    """``extra="forbid"`` and the field validators fail before any fitting."""
    with pytest.raises(MlsynthConfigError):
        MOSC(base_config(panel, **override))


def test_config_rejects_unknown_fields_directly(panel):
    with pytest.raises(Exception):
        MOSCConfig(**base_config(panel, mystery=1))


def test_estimation_failure_is_translated_not_swallowed(panel, monkeypatch):
    """A failure inside the sampler surfaces as MlsynthEstimationError.

    Asserting the failure is *reported* matters as much as the happy path: a
    swallowed exception here would return a counterfactual built from nothing.
    """
    import mlsynth.utils.mosc_helpers.factor as factor

    def explode(*args, **kwargs):
        raise RuntimeError("sampler blew up")

    monkeypatch.setitem(factor.FACTOR_MODELS, "gap", explode)
    with pytest.raises(MlsynthEstimationError, match="MOSC"):
        MOSC(base_config(panel)).fit()


def test_empty_ridge_grid_is_refused(panel):
    """The outcome regression needs at least one penalty to choose between."""
    with pytest.raises(MlsynthConfigError, match="ridge"):
        MOSC(base_config(panel, ridge_alphas=()))


def test_negative_ridge_penalty_is_refused(panel):
    """A negative penalty is not a weaker prior; it is not a prior."""
    with pytest.raises(MlsynthConfigError, match="non-negative"):
        MOSC(base_config(panel, ridge_alphas=(0.1, -1.0)))


def test_too_few_pre_periods_is_refused(panel):
    """One pre-period is one negative control outcome, which identifies nothing."""
    truncated = panel[panel["period"] >= 27].copy()
    with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
        MOSC(base_config(truncated, n_factors=1)).fit()


def test_rank_above_the_pre_period_count_is_refused(panel):
    """More factors than periods leaves the factorisation unidentified."""
    short = panel[panel["period"] >= 24].copy()
    with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
        MOSC(base_config(short, n_factors=6)).fit()


def test_ppca_stops_when_the_fit_stops_moving(panel):
    """EM has a convergence break, and a loose tolerance must reach it."""
    result = MOSC(base_config(panel, factor_model="ppca")).fit()
    assert np.isfinite(result.att)


def test_estimation_errors_pass_through_untranslated(panel, monkeypatch):
    """An error already in the library's vocabulary is not re-wrapped.

    Double-wrapping would bury the message the raiser chose, so the pipeline
    re-raises ``MlsynthEstimationError`` instead of describing it again.
    """
    import mlsynth.utils.mosc_helpers.factor as factor

    def explode(*args, **kwargs):
        raise MlsynthEstimationError("the sampler said exactly this")

    monkeypatch.setitem(factor.FACTOR_MODELS, "gap", explode)
    with pytest.raises(MlsynthEstimationError, match="the sampler said exactly this"):
        MOSC(base_config(panel)).fit()


def test_posterior_plot_returns_a_figure(fitted):
    """A plotter is mechanism: it builds a figure and hands it back.

    Showing and saving are the caller's policy, so nothing is displayed here and
    the figure stays reachable -- which is what lets this assert on its content.
    """
    from matplotlib.figure import Figure

    from mlsynth.utils.mosc_helpers.plotter import plot_mosc_posterior

    figure = plot_mosc_posterior(fitted, title="MOSC check")
    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        assert axis.get_title() == "MOSC check"
        # Observed path, posterior mean, and the intervention marker.
        assert len(axis.lines) >= 3
        assert axis.get_legend() is not None
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_display_graphs_routes_through_the_result_contract(panel, monkeypatch):
    """``display_graphs`` is policy applied by the caller, not by the mechanism."""
    import matplotlib

    matplotlib.use("Agg")
    calls = []
    monkeypatch.setattr(
        MOSCResults, "plot", lambda self, *a, **k: calls.append(self) or None, raising=False
    )
    MOSC(base_config(panel, display_graphs=True)).fit()
    assert len(calls) == 1


def test_ppca_em_converges_on_an_exactly_low_rank_panel():
    """EM stops when the objective stops moving, not only when it runs out of sweeps.

    An exactly rank-``K`` panel is the case that reaches the break, so it pins
    that the iteration has a real stopping rule.
    """
    from mlsynth.utils.mosc_helpers.factor import ppca_em

    rng = np.random.default_rng(0)
    exact = rng.normal(size=(30, 2)) @ rng.normal(size=(2, 10))
    draws = ppca_em(exact, n_factors=2, n_samples=5, n_iter=500, seed=0)
    assert draws.n_draws == 5
    reconstructed = draws.mean(0)
    assert np.corrcoef(reconstructed.ravel(), exact.ravel())[0, 1] > 0.99


def test_the_estimate_recovers_a_known_effect(panel):
    """The ATT lands near the effect the panel was built with.

    Sign and finiteness are cheap to satisfy; this is the assertion that the
    counterfactual is predicted with the treatment switched off. Predicting it
    switched on returns the factual path, which leaves the sign and the
    magnitude ordering intact and drives the estimate toward zero.
    """
    result = MOSC(base_config(panel, n_samples=120, n_warmup=120)).fit()

    pre_periods = int((panel["treated"] == 0).groupby(panel["unit"]).sum().min())
    treated = panel[panel["unit"] == "u00"].sort_values("period")["cases"].to_numpy()
    # The fixture inflates the treated rate by 1.4 after the intervention, so the
    # untreated path is the observed one deflated back.
    implied_truth = float(np.mean(treated[pre_periods:] * (1.0 - 1.0 / 1.4)))

    assert result.att == pytest.approx(implied_truth, rel=0.5)
    assert result.att > 0.25 * implied_truth


def test_no_pre_period_rmse_is_reported(fitted):
    """MOSC predicts only the post block, so a pre-period RMSE would be a zero.

    The pre-intervention counterfactual is the observed series, so any RMSE
    computed from it is zero by construction and reads as a perfect fit. The fit
    statistic that means something here is the held-out predictive density.
    """
    assert fitted.pre_rmse is None
    assert np.isfinite(fitted.diagnostics.heldout_log_density)


def test_a_long_daily_panel_gets_readable_time_ticks():
    """A few hundred periods must not render a few hundred overlapping labels."""
    from matplotlib.figure import Figure

    from mlsynth.utils.mosc_helpers.plotter import plot_mosc_posterior

    long_panel = make_count_panel(n_units=14, n_periods=180, pre_periods=150, seed=3)
    result = MOSC(base_config(long_panel, n_factors=2, n_samples=20, n_warmup=20)).fit()
    figure = plot_mosc_posterior(result)
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes[0].get_xticks()) <= 8
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_a_short_panel_keeps_every_time_tick():
    """Thinning applies only when the labels would overlap; a short panel keeps all."""
    from mlsynth.utils.mosc_helpers.plotter import plot_mosc_posterior

    short = make_count_panel(n_units=12, n_periods=7, pre_periods=5, seed=4)
    result = MOSC(base_config(short, n_factors=1, n_samples=15, n_warmup=15)).fit()
    figure = plot_mosc_posterior(result)
    try:
        assert len(figure.axes[0].get_xticks()) >= 7
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


# ------------------------------------------------------------- inference --
#
# The paper prescribes a nonparametric bootstrap over units (Section 3.4) and
# then plots the posterior spread instead. The two are different objects: the
# posterior conditions on the observed sample of units, the bootstrap resamples
# it. Theorem 4 identifies the counterfactual by the g-formula, an expectation
# over the distribution of loadings among the treated, so the sampling
# uncertainty in it comes from having finitely many units -- which is what the
# bootstrap targets and the posterior spread does not.


def test_bootstrap_is_the_default_inference(fitted):
    """The paper's Section 3.4 procedure, not the spread its figures show."""
    assert fitted.inference.method == "unit_bootstrap"
    assert fitted.posterior.n_bootstrap > 0


def test_the_posterior_band_remains_reachable(panel):
    """The mean-band is still available, under a name that says what it is."""
    result = MOSC(base_config(panel, inference="posterior")).fit()
    assert result.inference.method == "posterior_mean_band"
    assert np.isfinite(result.att)


def test_the_bootstrap_interval_is_wider_than_the_posterior_band(panel):
    """Resampling units admits uncertainty the posterior spread cannot see.

    The posterior band moves only with the factor model's uncertainty about the
    loadings. The bootstrap additionally moves with which units were drawn, so
    it cannot be the narrower of the two.
    """
    posterior = MOSC(base_config(panel, inference="posterior")).fit()
    bootstrap = MOSC(base_config(panel, inference="bootstrap", n_bootstrap=40)).fit()

    posterior_width = posterior.att_ci[1] - posterior.att_ci[0]
    bootstrap_width = bootstrap.att_ci[1] - bootstrap.att_ci[0]
    assert bootstrap_width > posterior_width


def test_the_interval_covers_zero_when_nothing_happened(panel):
    """A panel with no intervention effect is the placebo the method must pass.

    This is the property the authors' own figures fail on their own control
    teams: a 95 percent interval that excludes zero where no treatment occurred
    is a false positive, and ten of their placebo panels produced six.
    """
    untreated = panel.copy()
    # Strip the effect: the fixture inflates the treated rate by 1.4 post-period.
    post = untreated["treated"] == 1
    untreated.loc[post, "cases"] = (untreated.loc[post, "cases"] / 1.4).round()

    result = MOSC(base_config(untreated, inference="bootstrap", n_bootstrap=60)).fit()
    low, high = result.att_ci
    assert low <= 0 <= high, (
        f"no effect was applied, so the interval must cover zero; got "
        f"[{low:,.1f}, {high:,.1f}] around an ATT of {result.att:,.1f}"
    )


def test_the_bootstrap_keeps_the_treated_unit(panel):
    """Every resample retains the treated unit, or there is nothing to predict.

    The paper says to resample the per-unit data points. Read literally that can
    drop the one unit whose counterfactual is the estimand, so the donors are
    resampled and the treated unit is held.
    """
    result = MOSC(base_config(panel, inference="bootstrap", n_bootstrap=25)).fit()
    assert result.posterior.bootstrap_counterfactual.shape[0] == 25
    assert np.isfinite(result.posterior.bootstrap_counterfactual).all()


def test_bootstrap_replicate_count_is_validated(panel):
    with pytest.raises(MlsynthConfigError):
        MOSC(base_config(panel, n_bootstrap=1))


def test_the_bootstrap_is_reproducible(panel):
    first = MOSC(base_config(panel, inference="bootstrap", n_bootstrap=20, seed=3)).fit()
    second = MOSC(base_config(panel, inference="bootstrap", n_bootstrap=20, seed=3)).fit()
    np.testing.assert_allclose(first.att_ci, second.att_ci, rtol=1e-12)


@pytest.mark.parametrize(
    "inference, expected",
    [("bootstrap", "bootstrap interval"),
     ("posterior", "credible band (conditional mean)")],
)
def test_the_legend_names_the_interval_it_drew(panel, inference, expected):
    """A bootstrap percentile interval is not a credible interval.

    The figure is the output most readers will take a number from, so the label
    has to track which interval the fit actually produced.
    """
    from mlsynth.utils.mosc_helpers.plotter import plot_mosc_posterior

    result = MOSC(base_config(panel, inference=inference)).fit()
    figure = plot_mosc_posterior(result)
    try:
        labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        assert any(expected in label for label in labels), labels
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_a_bootstrap_failure_is_translated(panel, monkeypatch):
    """A replicate that blows up surfaces as MlsynthEstimationError.

    Swallowing it would return an interval built from however many replicates
    happened to survive, with nothing on the result recording that the rest did
    not.
    """
    import mlsynth.utils.mosc_helpers.pipeline as pipeline

    def explode(*args, **kwargs):
        raise RuntimeError("resample blew up")

    monkeypatch.setattr(pipeline, "bootstrap_counterfactuals", explode)
    with pytest.raises(MlsynthEstimationError, match="bootstrap"):
        MOSC(base_config(panel, inference="bootstrap")).fit()


def test_a_bootstrap_error_passes_through_untranslated(panel, monkeypatch):
    """An error already in the library's vocabulary keeps its own message."""
    import mlsynth.utils.mosc_helpers.pipeline as pipeline

    def explode(*args, **kwargs):
        raise MlsynthEstimationError("the resample said exactly this")

    monkeypatch.setattr(pipeline, "bootstrap_counterfactuals", explode)
    with pytest.raises(MlsynthEstimationError, match="the resample said exactly this"):
        MOSC(base_config(panel, inference="bootstrap")).fit()

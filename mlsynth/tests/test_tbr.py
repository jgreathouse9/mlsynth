"""TBR: Time-Based Regression (Kerman, Wang and Vaver 2017).

Test-first, per ``agents/agents_tests.md``: every test here is written before the
estimator exists and is RED until it lands.

The paper is the authority. Equation numbers below are its own, and the closed
forms are restated here independently of the implementation so that a test
fails when the implementation drifts from the paper and not merely when it
changes:

    pretest           y_t = alpha + beta x_t + eps_t                  eqn 1
    cumulative        Delta(T) = T (ybar_T - alpha - xbar_T beta)     eqn 4
    scale             T s (v_a + 2 xbar_T v_ab + v_b xbar_T^2 + 1/T)^(1/2)
                                                                      eqn 6
    posterior         shifted, scaled t on n - 2 degrees of freedom

``v_a``, ``v_b``, ``v_ab`` are entries of the unscaled ``V = (X'X)^-1``, ``s``
the classical residual standard deviation, and ``n`` the number of pretest time
points.

The replication behind these numbers is ``benchmarks/studies/tbr_geo``, which
cross-validates the same formulas against ``google/matched_markets`` on the
reference's own panel and reproduces the paper's coverage grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mlsynth import TBR
from mlsynth.config_models import TBRConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest


# --------------------------------------------------------------------------- #
# panel builders
# --------------------------------------------------------------------------- #
def geo_panel(n_control=4, n_treat=3, n_unassigned=0, T=20, T0=14,
              alpha=5.0, beta=1.5, noise=0.0, lift=0.0, cooldown_from=None,
              cost_in_test=0.0, cost_in_pre=0.0, seed=0):
    """A geo panel whose group aggregates satisfy eqn 1 by construction.

    The control geos are drawn first; the treatment geos are then defined so
    that their sum is exactly ``alpha + beta * (control sum)``, split evenly.
    With ``noise=0`` the pretest relation is exact, so ``Delta(T)`` is zero
    without a lift and exactly ``T * lift`` with one -- which is what lets the
    unit tests assert equalities instead of tolerances.

    ``cooldown_from`` is a period index (0-based) at which the cooldown flag
    turns 1 and stays 1. ``cost_in_pre`` puts spend in the pretest, which is
    what takes the cost arm out of section 3.4's fixed-cost case.
    """
    rng = np.random.default_rng(seed)
    ctl = rng.uniform(10.0, 30.0, size=(T, n_control))
    X = ctl.sum(axis=1)
    target = alpha + beta * X + noise * rng.normal(size=T)
    target = target + lift * (np.arange(T) >= T0)
    trt = np.tile((target / n_treat)[:, None], (1, n_treat))
    una = rng.uniform(100.0, 900.0, size=(T, n_unassigned))

    rows = []
    for block, tag, arr in (("c", "control", ctl), ("t", "treat", trt),
                            ("u", "unassigned", una)):
        for j in range(arr.shape[1]):
            for t in range(T):
                rows.append(dict(
                    geo=f"{block}{j}", date=t, sales=float(arr[t, j]),
                    is_control=int(tag == "control"),
                    D=int(tag == "treat" and t >= T0),
                    cooldown=0 if cooldown_from is None
                             else int(t >= cooldown_from),
                    cost=(cost_in_test if (tag == "treat" and t >= T0)
                          else (cost_in_pre if t < T0 else 0.0)),
                ))
    return pd.DataFrame(rows)


def base_config(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="sales", treat="D",
              control_col="is_control", display_graphs=False)
    kw.update(over)
    return TBRConfig(**kw)


def _control_aggregate(df):
    """The control-group series the pretest relation is fitted on."""
    wide = df.pivot_table(index="date", columns="geo", values="sales",
                          aggfunc="sum")
    ctl = sorted(df[df.is_control == 1].geo.unique())
    return wide[ctl].sum(axis=1).to_numpy()


def reference_posterior(df, T0, level=0.9, with_cooldown=False):
    """Eqns 4 and 6 computed straight from the frame, independent of mlsynth."""
    wide = df.pivot_table(index="date", columns="geo", values="sales",
                          aggfunc="sum")
    trt = [g for g in wide.columns if g.startswith("t")]
    ctl = sorted(df[df.is_control == 1].geo.unique())
    Y, X = wide[trt].sum(axis=1).to_numpy(), wide[ctl].sum(axis=1).to_numpy()
    if with_cooldown:
        end = len(Y)
    else:
        cd = df.groupby("date").cooldown.max()
        end = int((cd == 0).sum()) if cd.max() else len(Y)
    Yp, Xp = Y[:T0], X[:T0]
    A = np.column_stack([np.ones(T0), Xp])
    coef, *_ = np.linalg.lstsq(A, Yp, rcond=None)
    resid = Yp - A @ coef
    df_resid = T0 - 2
    s2 = float(resid @ resid) / df_resid
    V = np.linalg.pinv(A.T @ A)
    Yt, Xt = Y[T0:end], X[T0:end]
    Tn = np.arange(1, Yt.size + 1, dtype=float)
    ybar, xbar = np.cumsum(Yt) / Tn, np.cumsum(Xt) / Tn
    loc = Tn * (ybar - coef[0] - xbar * coef[1])
    scale = Tn * np.sqrt(s2) * np.sqrt(
        V[0, 0] + 2 * xbar * V[0, 1] + V[1, 1] * xbar ** 2 + 1.0 / Tn)
    q = stats.t.ppf(0.5 * (1 + level), df_resid)
    return dict(alpha=coef[0], beta=coef[1], s2=s2, df=df_resid, loc=loc,
                scale=scale, lower=loc - q * scale, upper=loc + q * scale)


# --------------------------------------------------------------------------- #
# Layer 4: smoke
# --------------------------------------------------------------------------- #
def test_fits_and_returns_an_effect_result():
    from mlsynth.config_models import EffectResult
    res = TBR(base_config(geo_panel())).fit()
    assert isinstance(res, EffectResult)
    assert np.isfinite(res.report.att)
    assert np.all(np.isfinite(np.asarray(res.counterfactual, dtype=float)))


def test_the_counterfactual_spans_the_whole_panel():
    T, T0 = 20, 14
    res = TBR(base_config(geo_panel(T=T, T0=T0))).fit()
    assert len(np.asarray(res.counterfactual, dtype=float)) == T


# --------------------------------------------------------------------------- #
# Layer 1: the paper's closed forms
# --------------------------------------------------------------------------- #
def test_pretest_coefficients_match_equation_1():
    df = geo_panel(noise=2.0, seed=3)
    want = reference_posterior(df, 14)
    got = TBR(base_config(df)).fit().tbr_fit
    assert got.alpha == pytest.approx(want["alpha"], rel=1e-12)
    assert got.beta == pytest.approx(want["beta"], rel=1e-12)
    assert got.sigma_sq == pytest.approx(want["s2"], rel=1e-12)
    assert got.df == want["df"] == got.n_pretest - 2


def test_cumulative_estimate_matches_equation_4():
    df = geo_panel(noise=2.0, seed=4)
    want = reference_posterior(df, 14)
    got = TBR(base_config(df)).fit().cumulative
    assert np.allclose(np.asarray(got.estimate, float), want["loc"], rtol=1e-12)


def test_cumulative_scale_matches_equation_6():
    df = geo_panel(noise=2.0, seed=5)
    want = reference_posterior(df, 14)
    got = TBR(base_config(df)).fit().cumulative
    assert np.allclose(np.asarray(got.scale, float), want["scale"], rtol=1e-12)


def test_interval_is_the_t_quantile_of_that_scale():
    df = geo_panel(noise=2.0, seed=6)
    want = reference_posterior(df, 14, level=0.9)
    got = TBR(base_config(df, level=0.9)).fit().cumulative
    assert np.allclose(np.asarray(got.lower, float), want["lower"], rtol=1e-12)
    assert np.allclose(np.asarray(got.upper, float), want["upper"], rtol=1e-12)


def test_cumulative_is_the_running_sum_of_the_per_period_gap():
    """Section 3.2: Delta(t) is the partial sum of phi_t = y_t - y*_t."""
    df = geo_panel(noise=2.0, seed=7)
    res = TBR(base_config(df)).fit()
    gap = np.asarray(res.report.gap, dtype=float)[-len(res.report.cumulative.estimate):]
    assert np.allclose(np.cumsum(gap),
                       np.asarray(res.report.cumulative.estimate, float), rtol=1e-10)


# --------------------------------------------------------------------------- #
# Layer 2: invariants
# --------------------------------------------------------------------------- #
def test_an_exact_relation_with_no_lift_reports_no_effect():
    res = TBR(base_config(geo_panel(noise=0.0, lift=0.0))).fit()
    assert np.allclose(np.asarray(res.report.cumulative.estimate, float), 0.0,
                       atol=1e-8)


def test_a_known_constant_lift_is_recovered_exactly():
    T, T0, lift = 20, 14, 7.0
    res = TBR(base_config(geo_panel(T=T, T0=T0, noise=0.0, lift=lift))).fit()
    got = np.asarray(res.report.cumulative.estimate, float)
    assert np.allclose(got, lift * np.arange(1, T - T0 + 1), atol=1e-7)


def test_unassigned_units_enter_neither_aggregate():
    """A unit that is neither treated nor flagged control changes nothing."""
    a = TBR(base_config(geo_panel(n_unassigned=0, seed=8))).fit()
    b = TBR(base_config(geo_panel(n_unassigned=5, seed=8))).fit()
    assert np.allclose(np.asarray(a.report.cumulative.estimate, float),
                       np.asarray(b.report.cumulative.estimate, float), rtol=1e-12)
    assert a.report.tbr_fit.beta == pytest.approx(b.report.tbr_fit.beta, rel=1e-12)


def test_scaling_the_outcome_scales_the_effect_and_its_scale():
    df = geo_panel(noise=2.0, seed=9)
    big = df.copy()
    big["sales"] = big["sales"] * 1000.0
    a = TBR(base_config(df)).fit().cumulative
    b = TBR(base_config(big)).fit().cumulative
    assert np.allclose(np.asarray(b.estimate, float),
                       1000.0 * np.asarray(a.estimate, float), rtol=1e-10)
    assert np.allclose(np.asarray(b.scale, float),
                       1000.0 * np.asarray(a.scale, float), rtol=1e-10)


def test_relabelling_units_within_a_group_changes_nothing():
    df = geo_panel(noise=2.0, seed=10)
    shuffled = df.copy()
    order = {g: f"z{i}" for i, g in
             enumerate(sorted(df[df.is_control == 1].geo.unique())[::-1])}
    shuffled["geo"] = shuffled.geo.map(lambda g: order.get(g, g))
    a = TBR(base_config(df)).fit().cumulative
    b = TBR(base_config(shuffled)).fit().cumulative
    assert np.allclose(np.asarray(a.estimate, float),
                       np.asarray(b.estimate, float), rtol=1e-12)


def test_the_weights_slot_says_there_are_none():
    """TBR carries no donor weights; the container must still not be empty."""
    res = TBR(base_config(geo_panel())).fit()
    assert not res.report.weights.is_empty


# --------------------------------------------------------------------------- #
# the cooldown flag
# --------------------------------------------------------------------------- #
def test_the_window_always_covers_the_whole_post_period():
    """Section 3.2 credits the cooldown to the effect, which is the reference's
    ``use_cooldown=True`` default, so the flag splits the window it does not
    shorten it. Told nothing about a cooldown, TBR cannot invent one: the two
    fits must agree on every number.
    """
    T, T0 = 24, 14
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=19, seed=11)
    with_cd = TBR(base_config(df, cooldown_col="cooldown")).fit()
    without = TBR(base_config(df)).fit()
    assert len(with_cd.report.cumulative.estimate) == T - T0
    assert len(without.report.cumulative.estimate) == T - T0
    assert np.allclose(np.asarray(with_cd.report.cumulative.estimate, float),
                       np.asarray(without.report.cumulative.estimate, float),
                       rtol=1e-12)


def test_the_flag_splits_the_window_into_intervention_and_cooldown():
    T, T0, cd = 24, 14, 19
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=cd, seed=11)
    res = TBR(base_config(df, cooldown_col="cooldown")).fit()
    assert res.report.cooldown_periods == T - cd
    assert res.report.intervention_periods == cd - T0
    assert res.report.cooldown_periods + res.report.intervention_periods == T - T0


def test_the_effect_at_the_end_of_the_intervention_is_reported_separately():
    """Section 3.5 reads the two against each other to decide whether the
    cooldown was needed at all, so both have to be available."""
    T, T0, cd = 24, 14, 19
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=cd, seed=11)
    res = TBR(base_config(df, cooldown_col="cooldown")).fit()
    full = np.asarray(res.report.cumulative.estimate, float)
    assert res.report.effect_at_intervention_end == pytest.approx(full[cd - T0 - 1],
                                                           rel=1e-12)
    assert res.report.effect_at_cooldown_end == pytest.approx(full[-1], rel=1e-12)


def test_no_cooldown_column_reports_no_cooldown_periods():
    T, T0 = 20, 14
    res = TBR(base_config(geo_panel(T=T, T0=T0))).fit()
    assert res.report.cooldown_periods == 0
    assert res.report.intervention_periods == T - T0
    assert res.report.effect_at_intervention_end == pytest.approx(
        res.report.effect_at_cooldown_end, rel=1e-12)


# --------------------------------------------------------------------------- #
# iROAS
# --------------------------------------------------------------------------- #
def test_no_cost_column_means_no_iroas():
    assert TBR(base_config(geo_panel())).fit().iroas is None


def test_zero_pretest_cost_is_the_fixed_cost_case():
    """Section 3.4: with no pretest spend the cost counterfactual is zero with
    certainty, so Delta_cost(T) is the total test spend and iROAS is a scaled t.
    """
    T, T0, per = 20, 14, 100.0
    df = geo_panel(T=T, T0=T0, noise=2.0, cost_in_test=per, seed=12)
    res = TBR(base_config(df, cost_col="cost")).fit()
    n_treat = df[df.D == 1].geo.nunique()
    total = per * n_treat * (T - T0)
    assert res.report.iroas.fixed_cost is True
    assert res.report.iroas.total_incremental_cost == pytest.approx(total, rel=1e-12)
    assert res.report.iroas.estimate == pytest.approx(
        float(np.asarray(res.report.cumulative.estimate, float)[-1]) / total,
        rel=1e-12)
    assert res.report.iroas.lower == pytest.approx(
        float(np.asarray(res.report.cumulative.lower, float)[-1]) / total, rel=1e-12)


def test_pretest_cost_leaves_the_fixed_cost_case():
    df = geo_panel(noise=2.0, cost_in_test=100.0, cost_in_pre=20.0, seed=13)
    res = TBR(base_config(df, cost_col="cost")).fit()
    assert res.report.iroas.fixed_cost is False
    assert np.isfinite(res.report.iroas.estimate)


def test_the_rank_deficient_cost_fit_is_reported_not_hidden():
    """The branch exists by name, so a caller can see which route ran.

    These diagnostics live on ``TBRResults`` and not on the shared
    ``MethodDetailsResults``: a cooldown count and a cost-design rank are
    TBR's, where the standardized model is every estimator's.
    """
    df = geo_panel(noise=2.0, cost_in_test=100.0, seed=14)
    res = TBR(base_config(df, cost_col="cost")).fit()
    assert res.report.cost_fit.rank_deficient is True
    assert res.report.cost_fit.df == res.report.tbr_fit.df


# --------------------------------------------------------------------------- #
# the zero fill
# --------------------------------------------------------------------------- #
def test_an_absent_cell_is_filled_to_zero():
    """Dropping a geo-day must equal recording it as zero, which is what the
    reference's own aggregation does."""
    df = geo_panel(noise=1.0, seed=15)
    holed = df.drop(df.index[(df.geo == "c1") & (df.date == 3)])
    zeroed = df.copy()
    zeroed.loc[(zeroed.geo == "c1") & (zeroed.date == 3), "sales"] = 0.0
    a = TBR(base_config(holed)).fit().cumulative
    b = TBR(base_config(zeroed)).fit().cumulative
    assert np.allclose(np.asarray(a.estimate, float),
                       np.asarray(b.estimate, float), rtol=1e-12)


def test_the_fill_is_reported():
    df = geo_panel(noise=1.0, seed=16)
    holed = df.drop(df.index[(df.geo == "c1") & (df.date.isin([3, 4]))])
    res = TBR(base_config(holed)).fit()
    assert res.report.filled_cells == 2


# --------------------------------------------------------------------------- #
# Layer 3: edge cases
# --------------------------------------------------------------------------- #
def test_a_single_control_unit():
    res = TBR(base_config(geo_panel(n_control=1, noise=1.0))).fit()
    assert np.isfinite(res.report.att)


def test_a_single_treated_unit():
    res = TBR(base_config(geo_panel(n_treat=1, noise=1.0))).fit()
    assert np.isfinite(res.report.att)


def test_three_pretest_periods_is_the_shortest_usable_panel():
    """df = n - 2, so three pretest points leave one degree of freedom."""
    res = TBR(base_config(geo_panel(T=6, T0=3, noise=0.5))).fit()
    assert res.report.tbr_fit.df == 1


def test_one_test_period():
    res = TBR(base_config(geo_panel(T=15, T0=14, noise=1.0))).fit()
    assert len(res.report.cumulative.estimate) == 1


def test_a_constant_control_aggregate_is_rank_deficient_and_says_so():
    df = geo_panel(noise=0.0, seed=17)
    df.loc[df.is_control == 1, "sales"] = 5.0
    res = TBR(base_config(df)).fit()
    assert res.report.tbr_fit.rank_deficient is True
    assert np.all(np.isfinite(np.asarray(res.report.cumulative.estimate, float)))


# --------------------------------------------------------------------------- #
# failure tests: each raises the translated error, and is reported
# --------------------------------------------------------------------------- #
def test_missing_control_column_raises():
    df = geo_panel().drop(columns=["is_control"])
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="is_control"):
        TBR(base_config(df)).fit()


def test_control_column_not_constant_within_unit_raises():
    df = geo_panel()
    df.loc[(df.geo == "c0") & (df.date > 5), "is_control"] = 0
    with pytest.raises(MlsynthDataError, match="constant"):
        TBR(base_config(df)).fit()


def test_control_column_with_a_third_value_raises():
    df = geo_panel()
    df.loc[df.geo == "c0", "is_control"] = 2
    with pytest.raises(MlsynthDataError):
        TBR(base_config(df)).fit()


def test_no_control_units_raises():
    df = geo_panel()
    df["is_control"] = 0
    with pytest.raises(MlsynthDataError, match="control"):
        TBR(base_config(df)).fit()


def test_a_unit_both_treated_and_flagged_control_raises():
    df = geo_panel()
    df.loc[df.geo == "t0", "is_control"] = 1
    with pytest.raises(MlsynthDataError):
        TBR(base_config(df)).fit()


def test_cooldown_flag_that_varies_across_units_raises():
    df = geo_panel(T=24, T0=14, cooldown_from=19)
    df.loc[(df.geo == "c0") & (df.date == 20), "cooldown"] = 0
    with pytest.raises(MlsynthDataError, match="block"):
        TBR(base_config(df, cooldown_col="cooldown")).fit()


def test_cooldown_flag_that_turns_back_off_raises():
    df = geo_panel(T=24, T0=14, cooldown_from=19)
    df.loc[df.date == 22, "cooldown"] = 0
    with pytest.raises(MlsynthDataError, match="sustained|monotone|block"):
        TBR(base_config(df, cooldown_col="cooldown")).fit()


def test_cooldown_beginning_before_treatment_raises():
    df = geo_panel(T=24, T0=14, cooldown_from=10)
    with pytest.raises(MlsynthDataError, match="cooldown"):
        TBR(base_config(df, cooldown_col="cooldown")).fit()


def test_cooldown_column_with_a_third_value_raises():
    df = geo_panel(T=24, T0=14, cooldown_from=19)
    df.loc[df.date == 21, "cooldown"] = 5
    with pytest.raises(MlsynthDataError):
        TBR(base_config(df, cooldown_col="cooldown")).fit()


def test_fewer_than_three_pretest_periods_raises():
    with pytest.raises(MlsynthDataError):
        TBR(base_config(geo_panel(T=10, T0=2))).fit()


def test_no_pretest_periods_raises():
    with pytest.raises(MlsynthDataError):
        TBR(base_config(geo_panel(T=10, T0=0))).fit()


def _config_kwargs(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="sales", treat="D",
              control_col="is_control", display_graphs=False)
    kw.update(over)
    return kw


def test_a_level_outside_the_unit_interval_is_refused():
    """Directly constructing the config raises Pydantic's error, which is the
    repository's convention; the dict path through the estimator translates it
    to ``MlsynthConfigError``. Both are pinned so neither can drift."""
    from pydantic import ValidationError
    df = geo_panel()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            base_config(df, level=bad)
        with pytest.raises(MlsynthConfigError, match="level"):
            TBR(_config_kwargs(df, level=bad))


def test_an_unknown_config_field_is_refused():
    from pydantic import ValidationError
    df = geo_panel()
    with pytest.raises(ValidationError):
        base_config(df, tbr_mode="fast")
    with pytest.raises(MlsynthConfigError, match="tbr_mode"):
        TBR(_config_kwargs(df, tbr_mode="fast"))


def test_a_blank_column_name_is_refused():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        base_config(geo_panel(), control_col="   ")


def test_a_non_config_input_raises_a_config_error():
    with pytest.raises(MlsynthConfigError, match="TBRConfig"):
        TBR(42)


def test_a_missing_cost_column_raises():
    with pytest.raises((MlsynthConfigError, MlsynthDataError), match="spend|cost"):
        TBR(base_config(geo_panel(), cost_col="spend")).fit()


# --------------------------------------------------------------------------- #
# the remaining branches, each reachable, so tested and not excused
# --------------------------------------------------------------------------- #
def test_a_cooldown_column_that_never_turns_on_is_no_cooldown():
    """Supplying the column is not the same as having a cooldown."""
    T, T0 = 20, 14
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=None, seed=18)
    res = TBR(base_config(df, cooldown_col="cooldown")).fit()
    assert res.report.cooldown_periods == 0
    assert res.report.intervention_periods == T - T0


def test_a_repeated_unit_period_cell_raises():
    """Summing across units makes a repeated cell ambiguous: two observations to
    add, or a duplicated row. The two give different totals, so neither is
    guessed."""
    df = geo_panel(noise=1.0, seed=19)
    doubled = pd.concat([df, df[(df.geo == "c0") & (df.date == 2)]],
                        ignore_index=True)
    with pytest.raises(MlsynthDataError, match="repeats"):
        TBR(base_config(doubled)).fit()


def test_a_panel_with_no_treated_unit_raises():
    df = geo_panel(noise=1.0, seed=20)
    df["D"] = 0
    with pytest.raises(MlsynthDataError, match="treatment group is empty"):
        TBR(base_config(df)).fit()


def test_a_cost_column_that_nets_to_zero_raises():
    """iROAS is a ratio, and a zero denominator is not an estimate."""
    df = geo_panel(noise=1.0, cost_in_test=0.0, seed=21)
    with pytest.raises(MlsynthDataError, match="denominator|zero"):
        TBR(base_config(df, cost_col="cost")).fit()


# --------------------------------------------------------------------------- #
# the plotter: returns its Figure, displays nothing (invariant 7)
# --------------------------------------------------------------------------- #
def test_the_plotter_returns_a_figure_and_shows_nothing(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlsynth.utils.tbr_helpers.plotter import plot_tbr

    shown = {"n": 0}
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.__setitem__("n", 1))
    res = TBR(base_config(geo_panel(noise=1.0, seed=22))).fit()
    fig = plot_tbr(res)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    assert shown["n"] == 0
    plt.close(fig)


def test_the_plotter_marks_where_the_intervention_stopped():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlsynth.utils.tbr_helpers.plotter import plot_tbr

    df = geo_panel(T=24, T0=14, noise=1.0, cooldown_from=19, seed=23)
    with_cd = plot_tbr(TBR(base_config(df, cooldown_col="cooldown")).fit())
    without = plot_tbr(TBR(base_config(df)).fit())
    # one vertical rule per panel marks the cooldown boundary; without a
    # cooldown flag there is no boundary to mark
    assert sum(len(ax.lines) for ax in with_cd.axes) > \
           sum(len(ax.lines) for ax in without.axes)
    plt.close(with_cd)
    plt.close(without)


def test_a_custom_title_reaches_the_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlsynth.utils.tbr_helpers.plotter import plot_tbr

    fig = plot_tbr(TBR(base_config(geo_panel(noise=1.0))).fit(),
                   title="Geo lift, Q3")
    assert fig.axes[0].get_title() == "Geo lift, Q3"
    plt.close(fig)


# --------------------------------------------------------------------------- #
# design mode: no treatment in the panel, the window named by a post column
#
# LEXSCM and SYNDES separate designing from estimating, and TBR has to as well:
# Au (2018) scores a candidate split by fitting TBR on history where nothing was
# treated, and his Example 3 is an A/A test built that way -- take the control
# geos of a finished experiment, split them, and check what TBR reports when the
# truth is zero. Neither is expressible through ``treat``, which asserts that a
# treatment happened.
# --------------------------------------------------------------------------- #
def design_panel(n_control=4, n_treat=3, T=20, T0=14, noise=1.0, seed=0,
                 **kw):
    """The same panel, with the group split and the window named directly."""
    df = geo_panel(n_control=n_control, n_treat=n_treat, T=T, T0=T0,
                   noise=noise, seed=seed, **kw)
    df["is_treatment"] = (df.geo.str.startswith("t")).astype(int)
    df["post"] = (df.date >= T0).astype(int)
    return df


def design_config(df, **over):
    kw = dict(df=df, unitid="geo", time="date", outcome="sales",
              control_col="is_control", treatment_col="is_treatment",
              post_col="post", display_graphs=False)
    kw.update(over)
    return TBRConfig(**kw)


def test_design_mode_fits_without_a_treatment_indicator():
    res = TBR(design_config(design_panel())).fit()
    assert np.isfinite(res.report.att)
    assert len(res.report.cumulative.estimate) == 6


def test_design_mode_and_estimation_mode_agree_on_the_same_panel():
    """The two routes differ in how the window and the groups are named, not in
    what is computed, so every number must coincide."""
    df = design_panel(noise=2.0, seed=24)
    estimated = TBR(base_config(df)).fit()
    designed = TBR(design_config(df)).fit()
    assert designed.report.tbr_fit.alpha == pytest.approx(estimated.report.tbr_fit.alpha,
                                                   rel=1e-12)
    assert designed.report.tbr_fit.beta == pytest.approx(estimated.report.tbr_fit.beta,
                                                  rel=1e-12)
    assert np.allclose(np.asarray(designed.report.cumulative.estimate, float),
                       np.asarray(estimated.report.cumulative.estimate, float),
                       rtol=1e-12)
    assert np.allclose(np.asarray(designed.report.cumulative.scale, float),
                       np.asarray(estimated.report.cumulative.scale, float),
                       rtol=1e-12)


def test_an_a_a_split_of_untreated_geos_reports_no_effect():
    """Au's Example 3 shape: nothing was treated, so the truth is zero and the
    interval has to cover it."""
    T, T0 = 30, 20
    rng = np.random.default_rng(25)
    rows = []
    for j in range(10):
        base = rng.uniform(50.0, 150.0)
        series = base * (1.0 + 0.02 * np.arange(T)) + rng.normal(0, 2.0, T)
        for t in range(T):
            rows.append(dict(geo=f"g{j}", date=t, sales=float(series[t]),
                             is_treatment=int(j < 5), is_control=int(j >= 5),
                             post=int(t >= T0)))
    res = TBR(design_config(pd.DataFrame(rows))).fit()
    final = len(res.report.cumulative.estimate) - 1
    assert res.report.cumulative.lower[final] <= 0.0 <= res.report.cumulative.upper[final]


def test_design_mode_reports_the_groups_it_used():
    res = TBR(design_config(design_panel())).fit()
    assert sorted(res.report.treated_units) == ["t0", "t1", "t2"]
    assert sorted(res.report.control_units) == ["c0", "c1", "c2", "c3"]


def test_neither_treat_nor_post_col_is_refused():
    from pydantic import ValidationError
    df = design_panel()
    with pytest.raises(MlsynthConfigError, match="post-treatment window"):
        TBRConfig(df=df, unitid="geo", time="date", outcome="sales",
                  control_col="is_control", treatment_col="is_treatment",
                  display_graphs=False)


def test_post_col_without_a_treatment_group_is_refused():
    from pydantic import ValidationError
    df = design_panel()
    with pytest.raises(MlsynthConfigError, match="treatment_col"):
        TBRConfig(df=df, unitid="geo", time="date", outcome="sales",
                  control_col="is_control", post_col="post",
                  display_graphs=False)


def test_a_post_column_that_is_never_one_raises():
    df = design_panel()
    df["post"] = 0
    with pytest.raises(MlsynthDataError, match="post"):
        TBR(design_config(df)).fit()


def test_a_post_column_that_is_always_one_raises():
    df = design_panel()
    df["post"] = 1
    with pytest.raises(MlsynthDataError):
        TBR(design_config(df)).fit()


def test_a_post_column_that_varies_across_units_raises():
    df = design_panel()
    df.loc[(df.geo == "c0") & (df.date == 15), "post"] = 0
    with pytest.raises(MlsynthDataError, match="block"):
        TBR(design_config(df)).fit()


def test_a_post_column_that_turns_back_off_raises():
    df = design_panel()
    df.loc[df.date == 17, "post"] = 0
    with pytest.raises(MlsynthDataError, match="sustained|block"):
        TBR(design_config(df)).fit()


def test_the_treatment_column_must_be_constant_within_unit():
    df = design_panel()
    df.loc[(df.geo == "t0") & (df.date > 5), "is_treatment"] = 0
    with pytest.raises(MlsynthDataError, match="constant"):
        TBR(design_config(df)).fit()


def test_a_unit_in_both_groups_is_refused_in_design_mode():
    df = design_panel()
    df.loc[df.geo == "t0", "is_control"] = 1
    with pytest.raises(MlsynthDataError):
        TBR(design_config(df)).fit()


def test_design_mode_with_an_empty_treatment_group_raises():
    df = design_panel()
    df["is_treatment"] = 0
    with pytest.raises(MlsynthDataError, match="treatment"):
        TBR(design_config(df)).fit()


def test_cooldown_works_in_design_mode_too():
    T, T0, cd = 24, 14, 19
    df = design_panel(T=T, T0=T0, cooldown_from=cd, seed=26)
    res = TBR(design_config(df, cooldown_col="cooldown")).fit()
    assert res.report.cooldown_periods == T - cd
    assert res.report.intervention_periods == cd - T0


def test_an_empty_panel_is_refused():
    with pytest.raises(MlsynthDataError, match="empty"):
        base_config(geo_panel().iloc[0:0])


def test_rows_without_a_unit_or_a_period_are_refused():
    df = geo_panel()
    df.loc[df.index[0], "date"] = np.nan
    with pytest.raises(MlsynthDataError, match="not observations"):
        base_config(df)


# --------------------------------------------------------------------------- #
# the fill and the design columns interact
#
# Filling an absent cell has to reconstruct the design columns from what they
# are, not from a neighbouring row. A period flag is a property of the period
# and a group flag a property of the unit, so carrying either from the wrong
# axis invents a value: a geo absent on the first cooldown day would inherit
# the previous day's 0 while every other geo reads 1, and a treated geo absent
# on the first post day would inherit a pretest 0 and stop looking treated.
# The reference's own panel is unbalanced, so both cases are reachable there.
# --------------------------------------------------------------------------- #
def test_a_hole_on_the_first_cooldown_day_does_not_break_the_flag():
    T, T0, cd = 24, 14, 19
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=cd, seed=27)
    holed = df.drop(df.index[(df.geo == "c1") & (df.date == cd)])
    res = TBR(base_config(holed, cooldown_col="cooldown")).fit()
    assert res.report.filled_cells == 1
    assert res.report.cooldown_periods == T - cd
    assert res.report.intervention_periods == cd - T0


def test_a_hole_on_the_first_post_day_does_not_unmark_treatment():
    T, T0 = 20, 14
    df = geo_panel(T=T, T0=T0, noise=1.0, seed=28)
    holed = df.drop(df.index[(df.geo == "t0") & (df.date == T0)])
    res = TBR(base_config(holed)).fit()
    assert res.report.filled_cells == 1
    assert len(res.report.cumulative.estimate) == T - T0
    assert sorted(res.report.treated_units) == ["t0", "t1", "t2"]


def test_a_hole_anywhere_equals_recording_that_cell_as_zero():
    """Across every column kind at once, and at the boundaries that matter."""
    T, T0, cd = 24, 14, 19
    df = geo_panel(T=T, T0=T0, noise=1.0, cooldown_from=cd,
                   cost_in_test=50.0, seed=29)
    holes = [("c1", cd), ("t0", T0), ("c0", 0), ("t1", T - 1), ("c2", cd - 1)]
    mask = np.zeros(len(df), dtype=bool)
    for geo, date in holes:
        mask |= ((df.geo == geo) & (df.date == date)).to_numpy()
    holed = df[~mask]
    zeroed = df.copy()
    zeroed.loc[mask, ["sales", "cost"]] = 0.0

    a = TBR(base_config(holed, cooldown_col="cooldown", cost_col="cost")).fit()
    b = TBR(base_config(zeroed, cooldown_col="cooldown", cost_col="cost")).fit()
    assert a.report.filled_cells == len(holes)
    assert b.report.filled_cells == 0
    assert np.allclose(np.asarray(a.report.cumulative.estimate, float),
                       np.asarray(b.report.cumulative.estimate, float), rtol=1e-12)
    assert np.allclose(np.asarray(a.report.cumulative.scale, float),
                       np.asarray(b.report.cumulative.scale, float), rtol=1e-12)
    assert a.report.iroas.estimate == pytest.approx(b.report.iroas.estimate, rel=1e-12)
    assert a.report.cooldown_periods == b.report.cooldown_periods


def test_a_hole_in_design_mode_keeps_the_post_flag_block_assigned():
    T, T0 = 20, 14
    df = design_panel(T=T, T0=T0, noise=1.0, seed=30)
    holed = df.drop(df.index[(df.geo == "c0") & (df.date == T0)])
    res = TBR(design_config(holed)).fit()
    assert res.report.filled_cells == 1
    assert len(res.report.cumulative.estimate) == T - T0


# --------------------------------------------------------------------------- #
# the pretest fit has to be visible
#
# TBR's validity rests on the pretest relation holding, and only its pretest half
# is checkable at all, so the standardized fit slot is the one a caller reaches
# for to judge it. Setting the counterfactual's pretest half to the observed
# series makes that slot report a perfect fit by construction -- a tautology
# where the one checkable assumption should be.
# --------------------------------------------------------------------------- #
def test_the_pretest_counterfactual_is_the_fitted_relation():
    """Not the observed series, which would make the residual zero by fiat."""
    df = geo_panel(noise=3.0, seed=31)
    res = TBR(base_config(df)).fit()
    n = res.report.tbr_fit.n_pretest
    obs = np.asarray(res.report.time_series.observed_outcome, float).ravel()
    cf = np.asarray(res.report.time_series.counterfactual_outcome, float).ravel()
    want = reference_posterior(df, n)
    x = _control_aggregate(df)
    assert np.allclose(cf[:n], want["alpha"] + want["beta"] * x[:n], rtol=1e-10)
    assert not np.allclose(cf[:n], obs[:n], atol=1e-6)


def test_the_reported_pretest_fit_is_the_real_one():
    df = geo_panel(noise=3.0, seed=32)
    res = TBR(base_config(df)).fit()
    n = res.report.tbr_fit.n_pretest
    obs = np.asarray(res.report.time_series.observed_outcome, float).ravel()[:n]
    cf = np.asarray(res.report.time_series.counterfactual_outcome, float).ravel()[:n]
    rmse = float(np.sqrt(np.mean((obs - cf) ** 2)))
    assert res.report.fit_diagnostics.rmse_pre == pytest.approx(rmse, rel=1e-10)
    assert res.report.fit_diagnostics.rmse_pre > 0.0
    assert res.report.fit_diagnostics.r_squared_pre < 1.0
    # the residual variance is sigma^2 up to the degrees-of-freedom correction
    assert rmse ** 2 * n / res.report.tbr_fit.df == pytest.approx(res.report.tbr_fit.sigma_sq,
                                                           rel=1e-10)


def test_a_noisier_panel_reports_a_worse_pretest_fit():
    """The diagnostic has to move with the thing it measures."""
    a = TBR(base_config(geo_panel(noise=1.0, seed=33))).fit()
    b = TBR(base_config(geo_panel(noise=8.0, seed=33))).fit()
    assert b.report.fit_diagnostics.rmse_pre > 4.0 * a.report.fit_diagnostics.rmse_pre
    assert b.report.fit_diagnostics.r_squared_pre < a.report.fit_diagnostics.r_squared_pre


def test_the_post_period_counterfactual_is_unchanged_by_this():
    """Delta(T) comes from eqn 4 and must not move when the pretest half of the
    plotted counterfactual changes."""
    df = geo_panel(noise=3.0, seed=34)
    res = TBR(base_config(df)).fit()
    want = reference_posterior(df, 14)
    assert np.allclose(np.asarray(res.report.cumulative.estimate, float), want["loc"],
                       rtol=1e-12)


# --------------------------------------------------------------------------- #
# the unscaled covariance of eqn 6, in closed form
# --------------------------------------------------------------------------- #
def _centred_reference(x):
    """``V = (X'X)^-1`` for ``X = [1, x]``, in float128 about the mean.

    Restated from the paper's ``V`` independently of the implementation, and
    carried at extended precision so it can referee a float64 route on a design
    whose Gram matrix is badly conditioned.
    """
    xl = np.asarray(x, dtype=np.longdouble)
    n = xl.size
    xbar = xl.mean()
    s_xx = (xl - xbar) @ (xl - xbar)
    return np.array([[1 / np.longdouble(n) + xbar * xbar / s_xx, -xbar / s_xx],
                     [-xbar / s_xx, 1 / s_xx]], dtype=np.longdouble)


def test_the_unscaled_covariance_inverts_the_gram():
    """On a design nothing strains, ``V`` is the inverse and not an approximation."""
    x = np.linspace(1.0, 4.0, 40)
    design = np.column_stack([np.ones(x.size), x])
    fit = fit_pretest(2.0 + 3.0 * x, x)
    assert np.allclose(fit.unscaled_cov @ (design.T @ design), np.eye(2),
                       atol=1e-12)
    assert np.allclose(fit.unscaled_cov, np.linalg.inv(design.T @ design),
                       rtol=1e-12)


def test_the_unscaled_covariance_holds_at_the_scale_a_geo_aggregate_has():
    """A control aggregate is a large mean with a small spread.

    Summing forty markets gives a series near 44,000 varying by a few hundred,
    so ``X'X`` is conditioned around 1e13 and the two routes that go through the
    determinant or a decomposition both lose digits there. The closed form about
    the mean is asserted against float128, which the SVD route misses by 2.4e-07.
    """
    rng = np.random.default_rng(11)
    for _ in range(25):
        x = 44000.0 + 500.0 * rng.standard_normal(90)
        want = _centred_reference(x)
        got = fit_pretest(3.0 + 1.1 * x, x).unscaled_cov.astype(np.longdouble)
        err = float(np.abs(got - want).max() / np.abs(want).max())
        assert err < 1e-14, err


def test_a_near_constant_regressor_is_not_called_rank_deficient():
    """The flag means section 3.4's constant regressor, so it means exactly that.

    Here the regressor varies in the eleventh significant figure. The design's
    smaller singular value falls under ``np.linalg.matrix_rank``'s tolerance, so
    a rank test calls this the zero-cost case and it is not one: the slope is
    identified, and its variance is ``1 / S_xx``, a large finite number.
    """
    rng = np.random.default_rng(3)
    x = 1e6 + 1e-5 * rng.standard_normal(90)
    design = np.column_stack([np.ones(x.size), x])
    assert np.linalg.matrix_rank(design) < 2      # what the rank test sees
    assert np.ptp(x) > 0.0                        # what the paper's case is

    fit = fit_pretest(2.0 + 0.5 * x, x)
    assert fit.rank_deficient is False
    s_xx = float((x - x.mean()) @ (x - x.mean()))
    assert fit.unscaled_cov[1, 1] == pytest.approx(1.0 / s_xx, rel=1e-12)
    assert np.isfinite(fit.unscaled_cov).all()


def test_equation_6_still_responds_to_the_test_period_mean_there():
    """``v_b xbar_T^2`` is the term that grows when the test period drifts.

    A route that discards the slope's variance sets ``v_b`` to zero, and eqn 6's
    scale then reads the same whether the test period sits on the pretest mean or
    a hundred standard deviations away. It has to widen.

    The response is drawn with noise because the scale carries the factor ``s``.
    An exactly linear ``y`` has ``s = 0``, so both scales are zero and their ratio
    is undefined; the first version of this test used one and passed only on the
    rounding error of the least-squares route it was written against, which the
    closed form does not produce.
    """
    rng = np.random.default_rng(4)
    x_pre = 1e6 + 1e-5 * rng.standard_normal(90)
    y_pre = 2.0 + 0.5 * x_pre + rng.standard_normal(90)
    fit = fit_pretest(y_pre, x_pre)
    sd = float(x_pre.std())

    on_mean = cumulative_posterior(fit, np.full(14, y_pre.mean()),
                                   np.full(14, x_pre.mean()))[1][-1]
    drifted = cumulative_posterior(fit, np.full(14, y_pre.mean()),
                                   np.full(14, x_pre.mean() + 100.0 * sd))[1][-1]
    assert drifted > 10.0 * on_mean


def test_a_constant_regressor_takes_the_pseudoinverse_branch():
    """Section 3.4's own case: ``S_xx`` is zero, so ``V`` is the pseudoinverse."""
    x = np.full(40, 7.0)
    fit = fit_pretest(np.full(40, 3.0), x)
    assert fit.rank_deficient is True
    design = np.column_stack([np.ones(x.size), x])
    assert np.allclose(fit.unscaled_cov, np.linalg.pinv(design.T @ design),
                       rtol=1e-12)
    assert np.isfinite(fit.unscaled_cov).all()


# --------------------------------------------------------------------------- #
# the fit from centred sums
# --------------------------------------------------------------------------- #
def test_the_fit_is_still_the_least_squares_solution():
    """Closed form or solver, eqn 1's coefficients are the same two numbers.

    The referee is ``np.linalg.lstsq`` on the explicit design, which is what
    this replaced, so the test fails if the closed form drifts from it.
    """
    rng = np.random.default_rng(7)
    for _ in range(30):
        x = 40000.0 + 9000.0 * rng.standard_normal(60)
        y = 12.0 + 0.8 * x + 300.0 * rng.standard_normal(60)
        fit = fit_pretest(y, x)
        design = np.column_stack([np.ones(x.size), x])
        want, *_ = np.linalg.lstsq(design, y, rcond=None)
        assert fit.alpha == pytest.approx(float(want[0]), rel=1e-10, abs=1e-9)
        assert fit.beta == pytest.approx(float(want[1]), rel=1e-12)


def test_the_fit_carries_the_centred_sums_it_was_built_from():
    """``S_xx``, ``S_xy`` and ``S_yy`` are eqn 1's sufficient statistics.

    They are on the result because every consumer needs them and recomputing
    them is a second pass over the window: ``R^2`` is ``1 - RSS / S_yy`` and the
    correlation is ``S_xy / sqrt(S_xx S_yy)``.
    """
    rng = np.random.default_rng(8)
    x = 500.0 + 30.0 * rng.standard_normal(45)
    y = 3.0 + 1.7 * x + 10.0 * rng.standard_normal(45)
    s = fit_pretest(y, x).sums
    assert s.n == 45
    assert s.s_xx == pytest.approx(float((x - x.mean()) @ (x - x.mean())), rel=1e-12)
    assert s.s_xy == pytest.approx(float((x - x.mean()) @ (y - y.mean())), rel=1e-12)
    assert s.s_yy == pytest.approx(float((y - y.mean()) @ (y - y.mean())), rel=1e-12)


def test_the_residual_series_is_on_the_fit_and_is_the_real_residual():
    """One residual pass, shared. The scoring gates read this series."""
    rng = np.random.default_rng(9)
    x = 100.0 + 5.0 * rng.standard_normal(50)
    y = 2.0 + 0.4 * x + rng.standard_normal(50)
    fit = fit_pretest(y, x)
    assert np.allclose(fit.resid, y - (fit.alpha + fit.beta * x), rtol=0, atol=1e-12)
    assert fit.sigma_sq == pytest.approx(float(fit.resid @ fit.resid) / fit.df,
                                         rel=1e-15)


def test_a_constant_regressor_keeps_the_least_squares_route():
    """Section 3.4's zero-cost case is unchanged.

    ``S_xx`` is zero there, so the closed form has no slope to compute and the
    solver's minimum-norm answer is kept. Asserted against ``lstsq`` so the
    branch cannot drift.
    """
    x = np.full(30, 6.0)
    y = np.linspace(1.0, 4.0, 30)
    fit = fit_pretest(y, x)
    design = np.column_stack([np.ones(30), x])
    want, *_ = np.linalg.lstsq(design, y, rcond=None)
    assert fit.rank_deficient is True
    assert fit.alpha == pytest.approx(float(want[0]), rel=1e-12)
    assert fit.beta == pytest.approx(float(want[1]), rel=1e-12)
    assert np.isfinite(fit.sigma_sq)

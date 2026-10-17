"""MAREX on several outcomes at once (Vives-i-Bastida 2022; Trejo et al. 2021).

A pilot that will measure several outcomes can treat only one set of units,
so the design has to fit them together. Following Vives-i-Bastida (2022),
the predictor vector stacks every outcome's fitting-window series ahead of the
covariates,

    x_j = (y^1_{j,E}, ..., y^K_{j,E}, z_j),

each outcome period a column of weight one, standardised with the rest when
``standardize=True``. The result then reports, for every outcome, the
synthetic paths, the normalised fit against the population and against the
synthetic control in the fitting and blank windows (his Tables 1 and 2), the
average post-period effect, and the placebo inference.

The invariants pinned here:

* the design is the one obtained by appending the extra outcomes' fitting
  windows to the predictors at weight one;
* listing an outcome twice changes nothing (every predictor's weight doubles,
  so the objective doubles and its minimiser stays);
* the order of the extra outcomes, and under standardisation their units, do
  not change the design;
* an extra outcome's blank-window and post-period values do not change the
  design;
* each outcome's report is computed from the design's weights, and the
  primary outcome's report agrees with the global design;
* with no extra outcomes nothing changes and no report is attached.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import solve_design_exact

warnings.filterwarnings("ignore")

N, T, T0 = 10, 20, 16


def _arrays(seed):
    """Two outcomes driven by different factors, so they disagree about which
    units look like the population."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=N), rng.normal(size=N)
    trend = np.linspace(0, 1, T)
    Y1 = f1[:, None] + np.outer(rng.normal(size=N), trend) \
        + rng.normal(scale=0.3, size=(N, T))
    Y2 = 50 * f2[:, None] + 20 * np.outer(rng.normal(size=N), trend) \
        + rng.normal(scale=5, size=(N, T))
    Y3 = rng.normal(size=(N, T)).cumsum(axis=1)
    return Y1, Y2, Y3


def _frame(Y1, Y2, Y3=None, Z=None):
    rows = []
    for j in range(N):
        for t in range(T):
            r = dict(unit=f"u{j}", t=t, y=Y1[j, t], y2=Y2[j, t])
            if Y3 is not None:
                r["y3"] = Y3[j, t]
            if Z is not None:
                r["z"] = Z[j]
            rows.append(r)
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    cfg = dict(df=df, outcome="y", unitid="unit", time="t", T0=T0,
               blank_periods=4, m_eq=2, backend="exact", inference=True,
               standardize=True)
    cfg.update(kw)
    return cfg


def _fit(df, **kw):
    return MAREX(_cfg(df, **kw)).fit()


def _summary(res):
    w = res.design_weights.donor_weights
    v = res.design_weights.summary_stats["control_weights_agg"]
    return (sorted(map(str, res.selected_units)),
            {k: round(x, 8) for k, x in w.items()},
            {k: round(x, 8) for k, x in v.items()})


# --------------------------------------------------------------------- smoke
def test_a_two_outcome_design_reports_both_outcomes():
    Y1, Y2, _ = _arrays(0)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"])
    assert list(res.outcomes) == ["y", "y2"]
    for name, rep in res.outcomes.items():
        assert rep.outcome == name
        assert rep.synthetic_treated.shape == (T,)
        assert rep.synthetic_control.shape == (T,)
        assert np.isfinite(rep.att)
        assert rep.nmse_fit_population >= 0 and rep.nmse_fit_control >= 0
        assert 0.0 <= rep.inference.global_p_value <= 1.0


# ----------------------------------------------------------- the design used
def test_the_design_appends_the_extra_fitting_windows():
    """The same design from the solver directly, with y2's fitting window
    passed as predictor columns of weight one."""
    Y1, Y2, _ = _arrays(1)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"], inference=False)
    T_fit = T0 - 4
    raw = solve_design_exact(
        Y_full=pd.DataFrame(Y1, index=[f"u{j}" for j in range(N)]), T0=T0,
        clusters=np.zeros(N, dtype=int), blank_periods=4, m_eq=2,
        covariates=Y2[:, :T_fit], covariate_weight=np.ones(T_fit),
        standardize=True)
    np.testing.assert_allclose(res.globres.treated_weights_agg,
                               raw["w_agg"], atol=1e-9)
    np.testing.assert_allclose(res.globres.objective, raw["objective"],
                               rtol=1e-10)


def test_an_extra_outcome_changes_the_design_somewhere():
    moved = False
    for seed in range(8):
        Y1, Y2, _ = _arrays(seed)
        df = _frame(Y1, Y2)
        if _summary(_fit(df, inference=False))[0] != \
                _summary(_fit(df, extra_outcomes=["y2"], inference=False))[0]:
            moved = True
            break
    assert moved


@pytest.mark.parametrize("backend,design", [("exact", "standard"),
                                            ("exact", "weakly_targeted"),
                                            ("mip", "standard")])
def test_listing_the_primary_outcome_again_changes_nothing(backend, design):
    Y1, Y2, _ = _arrays(2)
    df = _frame(Y1, Y2).assign(ycopy=lambda d: d["y"])
    a = _fit(df, backend=backend, design=design, inference=False)
    b = _fit(df, backend=backend, design=design, inference=False,
             extra_outcomes=["ycopy"])
    assert _summary(a)[0] == _summary(b)[0]
    np.testing.assert_allclose(a.globres.treated_weights_agg,
                               b.globres.treated_weights_agg, atol=1e-6)
    np.testing.assert_allclose(b.globres.objective, 2 * a.globres.objective,
                               rtol=1e-6)


def test_the_order_of_the_extra_outcomes_does_not_matter():
    Y1, Y2, Y3 = _arrays(3)
    df = _frame(Y1, Y2, Y3)
    assert (_summary(_fit(df, extra_outcomes=["y2", "y3"]))
            == _summary(_fit(df, extra_outcomes=["y3", "y2"])))


def test_standardised_outcomes_do_not_depend_on_their_units():
    Y1, Y2, _ = _arrays(4)
    df = _frame(Y1, Y2)
    a = _fit(df, extra_outcomes=["y2"])
    b = _fit(df.assign(y2=1000.0 * df["y2"]), extra_outcomes=["y2"])
    assert _summary(a) == _summary(b)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), scale=st.floats(0.1, 100.0))
def test_extra_outcomes_outside_the_fitting_window_do_not_move_the_design(
        seed, scale):
    Y1, Y2, _ = _arrays(seed)
    Y2b = Y2.copy()
    rng = np.random.default_rng(seed + 1)
    Y2b[:, T0 - 4:] = scale * rng.normal(size=(N, T - T0 + 4))
    a = _fit(_frame(Y1, Y2), extra_outcomes=["y2"], inference=False)
    b = _fit(_frame(Y1, Y2b), extra_outcomes=["y2"], inference=False)
    assert _summary(a) == _summary(b)


# ------------------------------------------------------------- the reports
def test_each_report_is_the_design_applied_to_that_outcome():
    Y1, Y2, _ = _arrays(5)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"])
    w = res.globres.treated_weights_agg
    v = res.globres.control_weights_agg
    for name, Y in (("y", Y1), ("y2", Y2)):
        rep = res.outcomes[name]
        np.testing.assert_allclose(rep.synthetic_treated, w @ Y)
        np.testing.assert_allclose(rep.synthetic_control, v @ Y)
        np.testing.assert_allclose(rep.population, Y.mean(axis=0))
        gap = w @ Y - v @ Y
        assert rep.att == pytest.approx(gap[T0:].mean())


def test_the_normalised_fit_is_vives_i_bastidas():
    """NMSE over a window: the mean over its periods of
    ((target_t - synthetic treated_t) / sd_t)^2, sd_t the cross-unit standard
    deviation of the outcome in period t."""
    Y1, Y2, _ = _arrays(6)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"])
    rep = res.outcomes["y2"]
    st_, sc, pop = rep.synthetic_treated, rep.synthetic_control, Y2.mean(axis=0)
    sd = Y2.std(axis=0)
    fit, blank = slice(0, T0 - 4), slice(T0 - 4, T0)
    nmse = lambda a, b, s: float(np.mean(((a[s] - b[s]) / sd[s]) ** 2))
    assert rep.nmse_fit_population == pytest.approx(nmse(pop, st_, fit))
    assert rep.nmse_fit_control == pytest.approx(nmse(sc, st_, fit))
    assert rep.nmse_blank_population == pytest.approx(nmse(pop, st_, blank))
    assert rep.nmse_blank_control == pytest.approx(nmse(sc, st_, blank))


def test_the_primary_report_agrees_with_the_global_design():
    Y1, Y2, _ = _arrays(7)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"])
    rep = res.outcomes["y"]
    np.testing.assert_allclose(rep.synthetic_treated,
                               res.globres.synthetic_treated)
    np.testing.assert_allclose(rep.synthetic_control,
                               res.globres.synthetic_control)
    assert rep.inference.global_p_value == res.globres.inference.global_p_value


def test_without_a_blank_window_there_is_no_blank_fit_or_inference():
    Y1, Y2, _ = _arrays(8)
    res = _fit(_frame(Y1, Y2), extra_outcomes=["y2"], blank_periods=0,
               inference=False)
    rep = res.outcomes["y2"]
    assert rep.nmse_blank_population is None and rep.nmse_blank_control is None
    assert rep.inference is None


def test_the_solution_pool_uses_the_extra_outcomes():
    """``top_K > 1`` re-solves through its own path. Its menu equals the
    exact pool solved directly on the stacked predictors, and differs from
    the menu without the extra outcome, so the check has power."""
    from mlsynth.utils.marex_helpers.exact import solve_design_exact_pool
    Y1, Y2, _ = _arrays(9)
    df = _frame(Y1, Y2)

    def pool(**kw):
        res = _fit(df, top_K=3, inference=False, **kw)
        return sorted(tuple(sorted(map(str, d["markets"]))) for d in res.pool)

    raws = solve_design_exact_pool(
        Y_full=pd.DataFrame(Y1, index=[f"u{j}" for j in range(N)]), T0=T0,
        clusters=np.zeros(N, dtype=int), top_K=3, blank_periods=4, m_eq=2,
        covariates=Y2[:, :T0 - 4], covariate_weight=np.ones(T0 - 4),
        standardize=True)
    direct = sorted(tuple(f"u{j}" for j in np.where(r["z_opt"][:, 0] > 0.5)[0])
                    for r in raws)
    with_extra = pool(extra_outcomes=["y2"])
    assert with_extra == direct
    assert with_extra != pool()


def test_extra_outcomes_and_covariates_together():
    """The extra outcome's periods go in at weight one ahead of the
    covariates, which keep their own weight; the balance diagnostics see the
    covariates only."""
    rng = np.random.default_rng(10)
    Y1, Y2, _ = _arrays(10)
    z = rng.normal(size=N)
    res = _fit(_frame(Y1, Y2, Z=z), extra_outcomes=["y2"],
               covariates=["z"], covariate_weight=0.1, inference=False)
    assert list(res.outcomes) == ["y", "y2"]
    assert res.post_fit.covariate_names == ("z",)
    T_fit = T0 - 4
    raw = solve_design_exact(
        Y_full=pd.DataFrame(Y1, index=[f"u{j}" for j in range(N)]), T0=T0,
        clusters=np.zeros(N, dtype=int), blank_periods=4, m_eq=2,
        covariates=np.hstack([Y2[:, :T_fit], z[:, None]]),
        covariate_weight=np.r_[np.ones(T_fit), 0.1], standardize=True)
    np.testing.assert_allclose(res.globres.objective, raw["objective"],
                               rtol=1e-10)


# -------------------------------------------------------------- the default
def test_no_extra_outcomes_changes_nothing():
    Y1, Y2, _ = _arrays(11)
    df = _frame(Y1, Y2)
    a = _fit(df)
    b = _fit(df, extra_outcomes=None)
    assert a.outcomes is None and b.outcomes is None
    assert _summary(a) == _summary(b)


# ------------------------------------------------------------------ failure
@pytest.mark.parametrize("extra,match", [
    (["y"], "primary"),
    (["y2", "y2"], "more than once"),
    (["nope"], "nope"),
])
def test_bad_extra_outcomes_are_refused(extra, match):
    Y1, Y2, _ = _arrays(12)
    with pytest.raises(MlsynthConfigError, match=match):
        MAREXConfig(**_cfg(_frame(Y1, Y2), extra_outcomes=extra))


def test_a_non_numeric_extra_outcome_is_refused():
    Y1, Y2, _ = _arrays(13)
    df = _frame(Y1, Y2).assign(label="a")
    with pytest.raises(MlsynthConfigError, match="numeric"):
        MAREXConfig(**_cfg(df, extra_outcomes=["label"]))


def test_a_missing_value_in_an_extra_outcome_is_reported():
    Y1, Y2, _ = _arrays(14)
    df = _frame(Y1, Y2)
    df.loc[3, "y2"] = np.nan
    with pytest.raises((MlsynthDataError, MlsynthConfigError), match="y2"):
        _fit(df, extra_outcomes=["y2"])


# ------------------------------------------------------------- the stacking
class TestStacking:

    def test_no_extra_outcomes_returns_the_inputs(self):
        from mlsynth.utils.marex_helpers.optimization import stack_extra_outcomes
        Z = np.ones((4, 2))
        cov, w = stack_extra_outcomes([], Z, 0.3, 5)
        assert cov is Z and w == 0.3

    def test_outcome_periods_precede_the_covariates_at_weight_one(self):
        from mlsynth.utils.marex_helpers.optimization import stack_extra_outcomes
        rng = np.random.default_rng(0)
        Y2, Y3, z = rng.normal(size=(4, 9)), rng.normal(size=(4, 9)), rng.normal(size=4)
        cov, w = stack_extra_outcomes([Y2, Y3], z, 0.25, 5)
        np.testing.assert_array_equal(cov, np.hstack([Y2[:, :5], Y3[:, :5], z[:, None]]))
        np.testing.assert_array_equal(w, np.r_[np.ones(10), 0.25])

    def test_without_covariates_only_the_outcomes_are_stacked(self):
        from mlsynth.utils.marex_helpers.optimization import stack_extra_outcomes
        Y2 = np.arange(12.0).reshape(3, 4)
        cov, w = stack_extra_outcomes([Y2], None, 1.0, 2)
        np.testing.assert_array_equal(cov, Y2[:, :2])
        np.testing.assert_array_equal(w, np.ones(2))

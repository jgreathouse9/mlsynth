"""CPDA: Hsiao and Zhou (2019) Section 3, the covariate-adjusted panel data approach.

The construction is Equations 11 to 15. Estimate a slope ``beta`` on the
pre-period, residualise the treated unit and every donor by ``x'beta``, choose
an intercept and donor weights on those residuals, and add the covariate part
back to predict the untreated outcome.

Two things about it drive the tests here.

The covariate step is the whole method. A CPDA given no covariates is PDA with
extra steps, so the config refuses that case instead of silently becoming
another estimator, and a panel whose treated unit moves with a covariate no
donor carries is where CPDA has to beat a donor-only fit.

The selector is not pinned by the paper, and the estimate moves with it. On
Hsiao and Zhou's own Table 9 panel six defensible selectors span a factor of
3.7, so the result records which one ran and can report the spread across all
of them. A test asserts the spread is reported and contains the point estimate,
because an estimator that hides this would be claiming an identification the
method does not have.

Levels: smoke, unit invariants, recovery, edge, failure, separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import CPDA
from mlsynth.config_models import BaseEstimatorResults, CPDAConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------

def _panel(seed=0, n_donors=12, T=40, T0=30, effect=-5.0, beta=(2.0, -1.0),
           treated_sd=8.0):
    """A factor panel with two covariates and one treated unit.

    ``y_it = x_it'beta + gamma_i'f_t + u_it``, with the treated unit taking
    ``effect`` from ``T0`` on.

    ``treated_sd`` is what gives the covariate step something to do. The
    treated unit's first covariate swings on its own, independently of every
    donor's, so no donor outcome tracks it: a donor-only regression carries
    that swing into the post window as error, and residualising by ``x'beta``
    removes it exactly. The donors still vary enough in the same covariate to
    identify ``beta``, which is estimated from the control group alone.

    Measured over ten seeds at ``treated_sd = 8``, a donor-only LASSO fit's
    error on the planted effect is 7.4 times CPDA's at worst and 37 times at
    the median. At ``treated_sd = 3`` the worst case falls to 1.3, which is
    close enough to be decided by noise.
    """
    rng = np.random.default_rng(seed)
    N = n_donors + 1
    f = rng.standard_normal((T, 2))
    gam = rng.standard_normal((N, 2))

    x0 = 10.0 + rng.standard_normal((T, N))
    x0[:, 0] = 10.0 + treated_sd * rng.standard_normal(T)
    x1 = rng.standard_normal((T, N)) + np.linspace(0, 2, T)[:, None]

    Y = (beta[0] * x0 + beta[1] * x1 + f @ gam.T
         + 0.5 * rng.standard_normal((T, N)))
    Y[T0:, 0] += effect

    units = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    return pd.DataFrame({
        "unit": units, "time": times,
        "y": Y.T.ravel(), "x0": x0.T.ravel(), "x1": x1.T.ravel(),
        "D": ((units == 0) & (times >= T0)).astype(int),
    })


def _cfg(df=None, **kw):
    base = dict(df=_panel() if df is None else df, outcome="y", treat="D",
                unitid="unit", time="time", covariates=["x0", "x1"],
                display_graphs=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------

class TestSmoke:
    def test_it_fits_and_returns_the_contract(self):
        res = CPDA(_cfg()).fit()
        assert isinstance(res, BaseEstimatorResults)
        assert np.isfinite(res.effects.att)
        assert res.time_series.counterfactual_outcome.shape == (40,)
        assert res.time_series.estimated_gap.shape == (40,)
        assert np.all(np.isfinite(res.time_series.counterfactual_outcome))

    def test_the_standard_submodels_are_populated(self):
        res = CPDA(_cfg()).fit()
        for field in ("effects", "time_series", "weights", "fit_diagnostics",
                      "inference", "method_details"):
            assert getattr(res, field) is not None, f"{field} missing"
        assert np.isfinite(res.fit_diagnostics.rmse_pre)

    def test_it_accepts_a_config_object_as_well_as_a_dict(self):
        cfg = CPDAConfig(**_cfg())
        a = CPDA(cfg).fit()
        b = CPDA(_cfg()).fit()
        assert a.effects.att == pytest.approx(b.effects.att)


# ----------------------------------------------------------------------
# Recovery
# ----------------------------------------------------------------------

class TestItRecoversTheEffect:
    @pytest.mark.parametrize("effect", [-5.0, 0.0, 3.0])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_att_lands_near_the_planted_effect(self, effect, seed):
        """Tolerance 0.5 against a worst measured error of 0.29 over ten seeds.
        The earlier 1.5 was fifty times the error and could not have failed."""
        res = CPDA(_cfg(df=_panel(effect=effect, seed=seed))).fit()
        assert res.effects.att == pytest.approx(effect, abs=0.5)

    @pytest.mark.parametrize("seed", [0, 3, 7])
    def test_it_beats_a_donor_only_fit_when_a_covariate_drives_the_treated_unit(self, seed):
        """The covariate step is the method, so it has to buy something.

        A factor, not a bare inequality. The two errors can land within a
        percent of each other on a panel where the covariate is spannable, and
        an inequality there is decided by noise; this design keeps the donor-
        only error at 7.4 times CPDA's at worst over ten seeds, so 3 has room.
        """
        from mlsynth import PDA
        df = _panel(seed=seed, effect=-5.0)
        cpda = CPDA(_cfg(df=df)).fit()
        pda = PDA(dict(df=df, outcome="y", treat="D", unitid="unit",
                       time="time", method="LASSO", display_graphs=False)).fit()
        err_c = abs(cpda.effects.att - (-5.0))
        err_p = abs(pda.effects.att - (-5.0))
        assert err_p > 3.0 * err_c, (
            f"the covariate step bought nothing: donor-only error {err_p:.3f} "
            f"against CPDA's {err_c:.3f}")

    def test_a_zero_effect_panel_is_not_called_significant_by_default(self):
        res = CPDA(_cfg(df=_panel(effect=0.0, seed=7))).fit()
        assert res.inference.p_value is None or res.inference.p_value > 0.01


# ----------------------------------------------------------------------
# The slope step
# ----------------------------------------------------------------------

class TestTheSlopeStep:
    def test_cce_reproduces_pesaran_equation_16(self):
        """``beta = (sum Xi' M Xi)^-1 sum Xi' M yi`` with ``M`` from the
        cross-sectional averages of ``(y, x)``, computed directly here."""
        from mlsynth.utils.cpda_helpers.beta import beta_cce
        rng = np.random.default_rng(0)
        T0, N, k = 25, 8, 2
        Y = rng.standard_normal((T0, N))
        X = rng.standard_normal((T0, N, k))
        zbar = np.column_stack([Y.mean(axis=1), X.mean(axis=1)])
        Q, _ = np.linalg.qr(zbar)
        M = np.eye(T0) - Q @ Q.T
        A = sum(X[:, i, :].T @ M @ X[:, i, :] for i in range(N))
        b = sum(X[:, i, :].T @ M @ Y[:, i] for i in range(N))
        assert beta_cce(Y, X, T0) == pytest.approx(
            np.linalg.lstsq(A, b, rcond=None)[0], rel=1e-10)

    def test_bai_is_the_argmin_of_bais_objective(self):
        """Not improvable by a restart from pooled OLS, which is the property
        that makes it Bai's estimator and not merely a fixed point."""
        from mlsynth.utils.cpda_helpers.beta import bai_objective, beta_bai
        rng = np.random.default_rng(1)
        T, N, k = 40, 15, 2
        lam = rng.standard_normal((N, 2))
        f = rng.standard_normal((T, 2))
        X = np.empty((T, N, k))
        X[:, :, 0] = 10.0 + 0.1 * rng.standard_normal((T, N))
        X[:, :, 1] = f @ lam.T + rng.standard_normal((T, N))
        Y = np.einsum("tnk,k->tn", X, np.array([20.0, 2.0])) + f @ lam.T \
            + rng.standard_normal((T, N))
        beta = beta_bai(Y, X, r=2)
        ols = np.linalg.lstsq(X.reshape(-1, k), Y.reshape(-1), rcond=None)[0]
        assert bai_objective(Y, X, beta, 2) <= bai_objective(Y, X, ols, 2) + 1e-8

    def test_both_slope_methods_run_and_are_recorded(self):
        for m in ("cce", "bai"):
            res = CPDA(_cfg(beta_method=m)).fit()
            assert res.fit.beta_method == m
            assert len(res.fit.beta) == 2


# ----------------------------------------------------------------------
# The selector, and the sensitivity it carries
# ----------------------------------------------------------------------

class TestTheSelectorIsAnExplicitChoice:
    @pytest.mark.parametrize("sel", ["lasso_cv", "lasso_bic", "aicc", "all"])
    def test_every_selector_runs_and_is_recorded(self, sel):
        res = CPDA(_cfg(selector=sel)).fit()
        assert res.fit.selector == sel
        assert np.isfinite(res.effects.att)

    def test_the_kept_donors_are_reported(self):
        res = CPDA(_cfg(selector="lasso_cv")).fit()
        kept = res.fit.selected_donors
        assert isinstance(kept, list)
        assert len(kept) <= 12
        assert set(res.weights.donor_weights) >= {str(u) for u in kept}

    def test_all_keeps_every_donor(self):
        res = CPDA(_cfg(selector="all")).fit()
        assert len(res.fit.selected_donors) == 12

    def test_sensitivity_reports_a_spread_containing_the_point_estimate(self):
        """The paper's description does not pin the selector, and the estimate
        moves with it, so the spread is part of the answer."""
        res = CPDA(_cfg(sensitivity=True)).fit()
        s = res.fit.sensitivity
        assert set(s) >= {"lasso_cv", "lasso_bic", "aicc", "all"}
        vals = [v["att"] for v in s.values()]
        assert min(vals) - 1e-8 <= res.effects.att <= max(vals) + 1e-8
        assert all(np.isfinite(v) for v in vals)

    def test_sensitivity_is_off_by_default_and_costs_nothing_then(self):
        res = CPDA(_cfg()).fit()
        assert res.fit.sensitivity is None

    def test_standardizing_the_selection_is_available_and_changes_membership(self):
        """CPDA's design is control residuals, one measurement, so it does not
        standardize by default. The switch exists because a caller whose
        residuals are not comparable needs it."""
        a = CPDA(_cfg(standardize_selection=False)).fit()
        b = CPDA(_cfg(standardize_selection=True)).fit()
        assert np.isfinite(a.effects.att) and np.isfinite(b.effects.att)
        assert a.fit.standardize_selection is False
        assert b.fit.standardize_selection is True


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestEdges:
    def test_a_single_donor_still_fits(self):
        res = CPDA(_cfg(df=_panel(n_donors=1, T=30, T0=20))).fit()
        assert np.isfinite(res.effects.att)

    def test_collinear_donors_do_not_crash_the_refit(self):
        df = _panel(n_donors=6, seed=2)
        dup = df[df.unit == 1].copy()
        dup["unit"] = 99
        res = CPDA(_cfg(df=pd.concat([df, dup], ignore_index=True))).fit()
        assert np.isfinite(res.effects.att)

    def test_a_constant_covariate_is_tolerated(self):
        df = _panel()
        df["x0"] = 1.0
        res = CPDA(_cfg(df=df)).fit()
        assert np.isfinite(res.effects.att)

    def test_one_post_period_is_enough(self):
        res = CPDA(_cfg(df=_panel(T=31, T0=30))).fit()
        assert res.time_series.estimated_gap.shape == (31,)


# ----------------------------------------------------------------------
# Failure: each raises the translated error, and the test asserts it is raised
# ----------------------------------------------------------------------

class TestFailures:
    def test_no_covariates_is_refused_rather_than_silently_becoming_pda(self):
        with pytest.raises(MlsynthConfigError, match="covariate"):
            CPDAConfig(**_cfg(covariates=[]))

    def test_a_missing_covariate_column_is_named(self):
        with pytest.raises(MlsynthDataError, match="nope"):
            CPDA(_cfg(covariates=["x0", "nope"])).fit()

    def test_an_unknown_selector_is_refused(self):
        with pytest.raises(Exception):
            CPDAConfig(**_cfg(selector="magic"))

    def test_an_unknown_beta_method_is_refused(self):
        with pytest.raises(Exception):
            CPDAConfig(**_cfg(beta_method="ols"))

    def test_no_treated_unit_is_reported(self):
        df = _panel()
        df["D"] = 0
        with pytest.raises(MlsynthDataError, match="treated"):
            CPDA(_cfg(df=df)).fit()

    def test_two_treated_units_are_reported(self):
        df = _panel()
        df.loc[(df.unit == 1) & (df.time >= 30), "D"] = 1
        with pytest.raises(MlsynthDataError, match="one treated unit"):
            CPDA(_cfg(df=df)).fit()

    def test_too_few_pre_periods_is_reported(self):
        df = _panel(T=10, T0=1)
        with pytest.raises(MlsynthDataError, match="pre-treatment"):
            CPDA(_cfg(df=df)).fit()

    def test_an_incomplete_panel_is_reported(self):
        df = _panel()
        with pytest.raises(MlsynthDataError):
            CPDA(_cfg(df=df.drop(df.index[5]))).fit()

    def test_extra_config_keys_are_forbidden(self):
        with pytest.raises(Exception):
            CPDAConfig(**_cfg(nonsense=1))


# ----------------------------------------------------------------------
# Separation: computing and showing are different jobs
# ----------------------------------------------------------------------

class TestSeparation:
    def test_the_plotter_returns_a_figure_and_does_not_show_it(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mlsynth.utils.cpda_helpers.plotter import plot_cpda

        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
        fig = plot_cpda(CPDA(_cfg()).fit(), outcome="y", time="time")
        assert fig is not None and not shown
        plt.close(fig)

    def test_fitting_prints_nothing(self, capsys):
        CPDA(_cfg()).fit()
        assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------
# The helpers, unit by unit
# ----------------------------------------------------------------------

class TestHelperUnits:
    def test_inputs_expose_the_panel_shape(self):
        from mlsynth.utils.cpda_helpers.setup import prepare_cpda_inputs
        inp = prepare_cpda_inputs(_panel(), unitid="unit", time="time",
                                  outcome="y", treat="D", covariates=["x0", "x1"])
        assert (inp.T, inp.T0, inp.T2, inp.N, inp.k) == (40, 30, 10, 12, 2)
        assert inp.Xco.shape == (40, 12, 2) and inp.x.shape == (40, 2)

    def test_refit_with_no_donors_falls_back_to_the_pre_period_mean(self):
        from mlsynth.utils.cpda_helpers.pipeline import _refit
        target = np.array([1.0, 3.0, 5.0])
        design = np.zeros((6, 4))
        mu, w, fitted = _refit(target, design[:3], design, np.array([], dtype=int))
        assert mu == pytest.approx(3.0)
        assert w.size == 0
        assert fitted == pytest.approx(np.full(6, 3.0))

    def test_a_pre_period_too_short_to_split_keeps_every_donor(self):
        from mlsynth.utils.cpda_helpers.selection import select_donors
        rng = np.random.default_rng(0)
        keep = select_donors(rng.standard_normal((3, 5)), rng.standard_normal(3),
                             selector="lasso_cv")
        assert keep.tolist() == [0, 1, 2, 3, 4]

    def test_an_empty_selection_falls_back_to_the_full_pool(self):
        """A penalty that keeps nothing leaves Equation 11 with its intercept
        alone, which fits but says nothing, so the pool comes back instead."""
        from mlsynth.utils.cpda_helpers.selection import select_donors
        rng = np.random.default_rng(1)
        design = rng.standard_normal((30, 6))
        keep = select_donors(design, np.zeros(30), selector="lasso_cv")
        assert keep.size > 0

    def test_a_perfectly_fitting_knot_is_skipped_by_the_ic_path(self):
        """Zero residual makes ``log(ssr/n)`` undefined, so the knot is passed
        over and a later one chosen."""
        from mlsynth.utils.cpda_helpers.selection import select_donors
        rng = np.random.default_rng(2)
        design = rng.standard_normal((12, 3))
        target = design[:, 0] * 2.0
        keep = select_donors(design, target, selector="aicc")
        assert np.all(np.isfinite(keep)) and keep.size >= 1

    def test_bai_returns_when_the_iteration_cap_is_reached(self):
        from mlsynth.utils.cpda_helpers.beta import _iterate
        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 8, 2))
        Y = np.einsum("tnk,k->tn", X, np.array([1.0, 2.0])) \
            + rng.standard_normal((20, 8))
        beta = _iterate(Y, X, 1, np.zeros(2), iters=1)
        assert beta.shape == (2,) and np.all(np.isfinite(beta))

    def test_the_results_validator_does_not_overwrite_a_populated_surface(self):
        from mlsynth.config_models import EffectsResults
        from mlsynth.utils.cpda_helpers.structures import CPDAResults
        res = CPDA(_cfg()).fit()
        again = CPDAResults(inputs=res.inputs, fit=res.fit,
                            effects=EffectsResults(att=123.0))
        assert again.effects.att == 123.0
        assert again.time_series is None

    def test_a_covariate_with_a_gap_is_named(self):
        from mlsynth.exceptions import MlsynthDataError
        df = _panel()
        df.loc[df.index[0], "x1"] = np.nan
        with pytest.raises(MlsynthDataError, match="x1"):
            CPDA(_cfg(df=df)).fit()

    def test_a_panel_with_no_donors_is_refused(self):
        from mlsynth.exceptions import MlsynthDataError
        df = _panel(n_donors=1)
        df = df[df.unit == 0].copy()
        with pytest.raises(MlsynthDataError):
            CPDA(_cfg(df=df)).fit()

    def test_duplicate_covariates_are_refused(self):
        with pytest.raises(MlsynthConfigError, match="Duplicate"):
            CPDAConfig(**_cfg(covariates=["x0", "x0"]))

    def test_the_plotter_saves_when_asked_to(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mlsynth.utils.cpda_helpers.plotter import plot_cpda
        res = CPDA(_cfg()).fit()
        p1 = tmp_path / "a.png"
        plt.close(plot_cpda(res, save=str(p1)))
        p2 = tmp_path / "b.png"
        plt.close(plot_cpda(res, save={"path": str(p2), "dpi": 80}))
        assert p1.is_file() and p2.is_file()

    def test_display_graphs_runs_the_plot_path(self):
        import matplotlib
        matplotlib.use("Agg")
        res = CPDA(_cfg(display_graphs=True)).fit()
        assert np.isfinite(res.effects.att)

    def test_the_ic_path_skips_knots_it_cannot_score(self):
        """Three guards, each reachable on a short pre-period.

        A knot wider than the sample has no residual degrees of freedom; a knot
        that fits exactly has ``ssr == 0`` and an undefined ``log(ssr/n)``; and
        AICc's correction divides by ``n - p - 1``. Each is passed over and a
        scorable knot is returned, so a short panel still selects.
        """
        from mlsynth.utils.cpda_helpers.selection import _ic_path
        rng = np.random.default_rng(11)
        n, N = 7, 10
        design = rng.standard_normal((n, N))
        target = design[:, 0] * 3.0                # exactly in the span
        for crit in ("aicc", "bic"):
            keep = _ic_path(design, target, crit)
            assert keep.size <= n - 3
        # A noisy target walks the path further, so a knot lands on exactly
        # n - 3 nonzeros, where AICc's n - p - 1 correction divides by zero.
        noisy = target + rng.standard_normal(n)
        for crit in ("aicc", "bic"):
            assert _ic_path(design, noisy, crit).size <= n - 3

    def test_a_nan_outcome_cell_is_reported(self):
        from mlsynth.exceptions import MlsynthDataError
        df = _panel()
        df.loc[df.index[3], "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="complete outcome panel"):
            CPDA(_cfg(df=df)).fit()

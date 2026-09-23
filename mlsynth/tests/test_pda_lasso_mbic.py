"""Shi and Huang's modified BIC as a LASSO penalty rule for PDA.

``fsPDA``'s ``lasso.BIC`` picks the penalty by minimising

    IC(lambda) = log(sigma^2(lambda))
                 + H * log(log N) * log(T1) / T1 * k(lambda)

over a grid, where ``k`` counts the selected donors and ``H`` is the constant the
paper says it tunes ("we tune the constants in the modified BIC to allow Lasso to
take in more variables", Remark 4 cont.). mlsynth selects the penalty by
cross-validation instead, so its LASSO cells have never been comparable to the
paper's cell by cell.

Three things separate the two rules. The criterion is one. The second is the
intercept: ``lasso.BIC`` fits ``glmnet(..., intercept = FALSE)`` while mlsynth's
cross-validated path fits one. The third is ``standardize = TRUE``, which divides
each donor by its standard deviation without centring it. A port that changed
only the criterion would still not be their estimator, and neither would one that
centred.

Most of what is asserted here is structural: the selection is the argmin it
claims to be, the criterion is its own formula, the constant moves sparsity in
the documented direction, and the default is untouched. Sweeping agreement with
``glmnet`` across panels belongs to a benchmark case, but one value is pinned
here -- the coefficient 0.51838043 that ``glmnet`` 4.1.8 returns on the panel
below -- because the conventions are exactly what no structural invariant can
see. Centring the donors leaves every invariant in this file satisfied and moves
that coefficient to 0.49909512.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from mlsynth.config_models import PDAConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pda_helpers.lasso import lasso_mbic_alpha, mbic_criterion

_HK = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                   "HongKong.csv")


def _panel(T=60, T0=40, N=20, seed=0, noise=0.5):
    """A sparse donor panel: three relevant controls, the rest noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    beta = np.zeros(N)
    beta[:3] = (0.8, -0.5, 0.3)[:N]
    y = X @ beta + noise * rng.standard_normal(T)
    return y, X, T0


# ----------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------


class TestSmoke:
    def test_returns_a_penalty_from_the_grid(self):
        y, X, T0 = _panel()
        grid = np.arange(0.01, 1.001, 0.01)
        alpha = lasso_mbic_alpha(y, X, T0, grid=grid)
        assert isinstance(alpha, float)
        assert np.isclose(grid, alpha).any()

    def test_selects_only_from_the_true_support(self):
        """At their default constant the rule is aggressive.

        The panel has three relevant donors, and ``H = 2`` keeps only the
        strongest -- which is what the paper is describing when it says it tunes
        the constant "to allow Lasso to take in more variables". Checked against
        their ``lasso.BIC`` on this exact panel: both return a penalty of 0.32,
        the single donor 1, and a coefficient of 0.51838043. So the invariant is
        that nothing outside the true support is selected and that the strongest
        donor survives, not that all three do.
        """
        y, X, T0 = _panel()
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        alpha = lasso_mbic_alpha(y, X, T0)
        _, _, _, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                     standardize=True)
        chosen = set(np.flatnonzero(support))
        assert chosen <= {0, 1, 2}
        assert 0 in chosen

    def test_a_smaller_constant_takes_in_more_donors(self):
        """The paper's own tuning direction, on the panel above."""
        y, X, T0 = _panel()
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        counts = []
        for const in (2.0, 0.25):
            alpha = lasso_mbic_alpha(y, X, T0, const=const)
            _, _, _, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                         standardize=True)
            counts.append(int(support.sum()))
        assert counts[1] > counts[0]


# ----------------------------------------------------------------------
# the rule is the argmin it claims to be
# ----------------------------------------------------------------------


class TestSelectsTheArgmin:
    def test_alpha_minimises_the_criterion_over_the_grid(self):
        """Recomputed independently: no grid point beats the one returned."""
        y, X, T0 = _panel()
        grid = np.arange(0.01, 1.001, 0.01)
        alpha = lasso_mbic_alpha(y, X, T0, grid=grid)
        ic = np.array([mbic_criterion(y, X, T0, a) for a in grid])
        assert np.isclose(grid[int(np.argmin(ic))], alpha)

    def test_the_fit_is_glmnets_and_not_a_centred_one(self):
        """One pinned coefficient, because the conventions are the whole point.

        ``glmnet(standardize = TRUE, intercept = FALSE)`` divides each column by
        its population standard deviation and does not centre. Fitting the
        centred columns instead is the plausible reading, and it moves this
        coefficient in the second decimal -- to 0.49909512 -- so an invariant
        test cannot separate the two. glmnet 4.1.8 on this panel returns the
        value below at the penalty the rule selects, and that agreement to 5e-09
        is the only evidence that the port is their estimator.
        """
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        y, X, T0 = _panel()
        alpha = lasso_mbic_alpha(y, X, T0)
        beta, _, _, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                        standardize=True)
        assert alpha == pytest.approx(0.32, abs=1e-12)
        assert list(np.flatnonzero(support)) == [0]
        assert beta[0] == pytest.approx(0.51838043, abs=1e-8)

    def test_criterion_is_log_mse_plus_the_penalty_term(self):
        """The formula, checked against its own pieces at one grid point."""
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        y, X, T0 = _panel()
        alpha, const = 0.2, 2.0
        beta, _, _, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False, standardize=True)
        mse = float(np.mean((y[:T0] - X[:T0] @ beta) ** 2))
        k = int(support.sum())
        expected = (np.log(mse)
                    + const * np.log(np.log(X.shape[1])) * np.log(T0) / T0 * k)
        assert mbic_criterion(y, X, T0, alpha, const=const) == pytest.approx(
            expected, rel=1e-9)


# ----------------------------------------------------------------------
# the constant is the knob the paper says it is
# ----------------------------------------------------------------------


class TestConstantMovesSparsity:
    def test_a_larger_constant_never_selects_more_donors(self):
        """H scales the per-donor charge, so raising it cannot add donors."""
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        y, X, T0 = _panel()
        sizes = []
        for const in (0.5, 2.0, 8.0, 32.0):
            alpha = lasso_mbic_alpha(y, X, T0, const=const)
            _, _, _, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False, standardize=True)
            sizes.append(int(support.sum()))
        assert sizes == sorted(sizes, reverse=True)

    @pytest.mark.parametrize("n_donors", [2, 3, 20])
    def test_a_larger_constant_never_lowers_the_criterion(self, n_donors):
        """The charge per donor is a charge at every pool size.

        ``log(log N)`` is negative below ``N = e``, so at two or three donors the
        unfloored term would pay for each one selected and the criterion would
        fall as the constant rises. Two donors is an ordinary pool, so the
        criterion is floored at zero there and the ordering below has to hold at
        every size.
        """
        y, X, T0 = _panel(N=n_donors)
        scores = [mbic_criterion(y, X, T0, 0.05, const=c)
                  for c in (0.5, 2.0, 50.0)]
        assert scores == sorted(scores)

    def test_a_vanishing_constant_chases_fit_alone(self):
        """With no charge per donor the criterion is log MSE, minimised densest."""
        y, X, T0 = _panel()
        grid = np.arange(0.01, 1.001, 0.01)
        alpha = lasso_mbic_alpha(y, X, T0, const=1e-12, grid=grid)
        mses = np.array([
            mbic_criterion(y, X, T0, a, const=1e-12) for a in grid])
        assert np.isclose(grid[int(np.argmin(mses))], alpha)
        assert alpha == pytest.approx(grid.min(), abs=1e-12)


# ----------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------


class TestEdges:
    def test_single_donor(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 1))
        y = (X[:, 0] * 0.7 + 0.3 * rng.standard_normal(40))
        assert isinstance(lasso_mbic_alpha(y, X, 25), float)

    def test_a_constant_donor_does_not_divide_by_zero(self):
        """glmnet's scaling divides by each column's spread; a flat donor has none."""
        y, X, T0 = _panel()
        X = X.copy()
        X[:, 5] = 3.0
        alpha = lasso_mbic_alpha(y, X, T0)
        assert np.isfinite(alpha)

    def test_two_donors_and_a_short_pre_period(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((12, 2))
        y = X[:, 0] + 0.1 * rng.standard_normal(12)
        assert np.isfinite(lasso_mbic_alpha(y, X, 8))

    def test_a_perfectly_fitting_donor_is_finite(self):
        """Zero residual sends log MSE to -inf; the rule must still return."""
        X = np.arange(1.0, 41.0).reshape(-1, 1)
        y = 2.0 * X[:, 0]
        assert np.isfinite(lasso_mbic_alpha(y, X, 25))


# ----------------------------------------------------------------------
# failure cases
# ----------------------------------------------------------------------


class TestFailures:
    @pytest.mark.parametrize("const", [0.0, -1.0])
    def test_a_nonpositive_constant_raises(self, const):
        y, X, T0 = _panel()
        with pytest.raises(MlsynthConfigError):
            lasso_mbic_alpha(y, X, T0, const=const)

    def test_an_empty_grid_raises(self):
        y, X, T0 = _panel()
        with pytest.raises(MlsynthConfigError):
            lasso_mbic_alpha(y, X, T0, grid=np.array([]))

    def test_a_nonpositive_penalty_in_the_grid_raises(self):
        """A zero penalty is ordinary least squares, not a LASSO path point."""
        y, X, T0 = _panel()
        with pytest.raises(MlsynthConfigError):
            lasso_mbic_alpha(y, X, T0, grid=np.array([0.0, 0.1]))

    @pytest.mark.parametrize("T0", [0, -3])
    def test_a_nonpositive_pre_period_raises(self, T0):
        y, X, _ = _panel()
        with pytest.raises(MlsynthConfigError):
            lasso_mbic_alpha(y, X, T0)


# ----------------------------------------------------------------------
# the config field, and the promise that the default is untouched
# ----------------------------------------------------------------------


def _hk_config(**kwargs):
    return PDAConfig(df=pd.read_csv(os.path.abspath(_HK)), outcome="GDP",
                     treat="Integration", unitid="Country", time="Time",
                     method="LASSO", **kwargs)


class TestConfig:
    def test_default_criterion_is_cross_validation(self):
        assert _hk_config().lasso_criterion == "cv"

    def test_default_constant_is_theirs(self):
        assert _hk_config().lasso_mbic_const == 2.0

    def test_mbic_is_accepted(self):
        assert _hk_config(lasso_criterion="mbic").lasso_criterion == "mbic"

    def test_an_unknown_criterion_is_rejected(self):
        """Named, so the test cannot pass off ``extra="forbid"`` as validation."""
        with pytest.raises(Exception, match="lasso_criterion"):
            _hk_config(lasso_criterion="aic")

    @pytest.mark.parametrize("const", [0.0, -1.0])
    def test_a_nonpositive_constant_is_rejected(self, const):
        with pytest.raises(Exception, match="lasso_mbic_const"):
            _hk_config(lasso_mbic_const=const)


class TestEstimator:
    def test_the_default_fit_is_unchanged(self):
        """The cross-validated path is the default and nothing about it moves."""
        base = PDA(_hk_config()).fit()
        again = PDA(_hk_config(lasso_criterion="cv")).fit()
        assert base.effects.att == pytest.approx(again.effects.att, rel=1e-12)

    def test_mbic_fits_and_reports_the_rule(self):
        res = PDA(_hk_config(lasso_criterion="mbic")).fit()
        assert np.isfinite(res.effects.att)
        meta = res.fits["lasso"].metadata
        assert meta["criterion"] == "mbic"
        assert meta["alpha"] > 0

    def test_the_default_reports_cross_validation(self):
        meta = PDA(_hk_config()).fit().fits["lasso"].metadata
        assert meta["criterion"] == "cv"

    def test_mbic_moves_the_donor_set(self):
        """Their rule fits without an intercept and charges per donor.

        On Hong Kong that is a different model from the cross-validated one, so
        the two paths must not return the same answer -- if they did, the field
        would not be wired through.
        """
        cv = PDA(_hk_config()).fit()
        mbic = PDA(_hk_config(lasso_criterion="mbic")).fit()
        assert cv.effects.att != pytest.approx(mbic.effects.att, rel=1e-6)

    def test_the_fit_is_the_one_the_criterion_scored(self):
        """The estimate has to be the model the selected penalty was chosen for.

        This is what makes the wiring more than a recorded label: the estimator's
        coefficients must reproduce, to the last digit, a direct call at the
        penalty the rule returned under the conventions the rule assumes.
        """
        from mlsynth.utils.pda_helpers.lasso import fit_lasso

        res = PDA(_hk_config(lasso_criterion="mbic")).fit()
        y, X, T0 = res.inputs.y, res.inputs.X, res.inputs.T0
        alpha = lasso_mbic_alpha(y, X, T0)
        beta, _, _, _ = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                  standardize=True)
        assert res.fits["lasso"].beta == pytest.approx(beta, abs=1e-12)
        assert res.fits["lasso"].intercept == pytest.approx(0.0, abs=1e-12)

    def test_a_larger_constant_takes_in_fewer_donors_end_to_end(self):
        loose = PDA(_hk_config(lasso_criterion="mbic")).fit()
        tight = PDA(_hk_config(lasso_criterion="mbic",
                               lasso_mbic_const=8.0)).fit()
        assert len(loose.weights.donor_weights) > len(tight.weights.donor_weights)

    @pytest.mark.parametrize("criterion,tol", [("cv", 1e-4), ("mbic", 1e-12)])
    def test_the_bootstrap_refit_reproduces_the_fit_it_resamples(self, criterion, tol):
        """Fed the original outcome, the refit callback must return the original
        counterfactual.

        Jiang et al. (2025) bootstrap the sampling distribution of a *particular*
        estimator, so the callback has to carry the penalty and the fit
        conventions of the fit it is standing in for. Handing it the unresampled
        data is the one input whose answer is known.

        The two tolerances are not cosmetic. Under the modified BIC the penalty
        is fixed before either fit, so both are the same coordinate descent to
        ``tol = 1e-12`` and the agreement is exact. Under cross-validation the
        point estimate comes from ``LassoCV``, whose refit at the selected
        penalty runs to scikit-learn's default ``tol = 1e-4``, so it sits about
        6e-06 from the bootstrap's tighter refit of the same problem. That gap
        predates this rule and is left alone here; closing it would move every
        cross-validated LASSO result in the library.
        """
        from mlsynth.utils.pda_helpers.orchestration import _build_refit

        res = PDA(_hk_config(lasso_criterion=criterion,
                             prediction_intervals=True, pi_n_boot=5)).fit()
        fit = res.fits["lasso"]
        y, X, T0 = res.inputs.y, res.inputs.X, res.inputs.T0
        refit = _build_refit("lasso", X, T0, tau=None, l2_standardize=True,
                             fs_intercept=False,
                             lasso_alpha=fit.metadata["alpha"],
                             lasso_criterion=criterion)
        cf, support_idx = refit(y)
        assert cf == pytest.approx(fit.counterfactual, abs=tol)
        assert list(support_idx) == list(np.flatnonzero(np.abs(fit.beta) > 1e-10))

    def test_prediction_intervals_reuse_the_mbic_penalty(self):
        """The bootstrap refits at a fixed penalty; under mBIC it is theirs.

        The refit has to use the same penalty *and* the same glmnet conventions
        as the fit it is bootstrapping, so the penalty recorded in the metadata
        is the one the criterion selected, not a cross-validated one.
        """
        res = PDA(_hk_config(lasso_criterion="mbic", prediction_intervals=True,
                             pi_n_boot=5)).fit()
        fit = res.fits["lasso"]
        assert fit.prediction_intervals is not None
        y, X, T0 = res.inputs.y, res.inputs.X, res.inputs.T0
        assert fit.metadata["alpha"] == pytest.approx(
            lasso_mbic_alpha(y, X, T0), rel=1e-12)

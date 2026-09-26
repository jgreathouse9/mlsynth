"""Tests for SL, the synthetic learner of Viviano and Bradic (2023).

Layered as ``agents/agents_tests.md`` asks: the pure pieces first, where an
invariant can be asserted exactly, then the assembled estimator, then the edges
and the failures.

Two of these tests exist because the authors' replication code gets them wrong,
and the measurements are in issue #651:

* ``test_bootstrap_uses_the_configured_eta`` -- their ``library.R:214`` refits
  the ensemble with ``eta=1`` inside the bootstrap while the point estimate uses
  51.43, which inflates the critical values by 19 to 61 percent.
* ``test_lasso_expert_is_seed_independent`` -- their ``cv.glmnet(nfolds=5)``
  draws the penalty's folds from the RNG, and on a 30-period window that has
  three attractors, one of which keeps no donors, swinging the effect 40 percent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import SL
from mlsynth.config_models import EffectsResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sl_helpers.config import SLConfig
from mlsynth.utils.sl_helpers.diagnostics import (
    error_correlation,
    error_matrix,
    flag_degenerate,
    participation_ratio,
)
from mlsynth.utils.sl_helpers.experts import EXPERTS, build_experts
from mlsynth.utils.sl_helpers.inference import (
    bias_adjusted_att,
    block_bootstrap_test,
    test_statistic as sl_statistic,
)
from mlsynth.utils.sl_helpers.setup import prepare_sl_inputs
from mlsynth.utils.sl_helpers.structures import SLResults
from mlsynth.utils.sl_helpers.weights import (
    effective_k,
    exponential_weights,
    paper_eta,
)

TAU = -3.0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _panel(seed: int = 0, n_donors: int = 6, T: int = 60, T0: int = 40,
           tau: float = TAU, covariate: bool = True) -> pd.DataFrame:
    """A two-factor panel with a planted constant effect on the treated unit."""
    rng = np.random.default_rng(seed)
    N = n_donors + 1
    f = rng.standard_normal((T, 2))
    lam = rng.standard_normal((N, 2))
    Y = 10.0 + f @ lam.T + 0.5 * rng.standard_normal((T, N))
    Y[T0:, 0] += tau
    z = rng.standard_normal((T, N)) + 0.3 * f[:, [0]]
    units = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    out = pd.DataFrame({
        "unit": units, "time": times, "y": Y.T.ravel(),
        "D": ((units == 0) & (times >= T0)).astype(int),
    })
    if covariate:
        out["z"] = z.T.ravel()
    return out


def _cfg(df: pd.DataFrame, **kw) -> dict:
    base = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                display_graphs=False, n_boot=200, seed=0)
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def fitted():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SL(_cfg(_panel())).fit()


# --------------------------------------------------------------------------- #
# Layer 1: the exponential weights, Equations 11 and 12
# --------------------------------------------------------------------------- #

class TestWeights:
    def test_simplex(self):
        w = exponential_weights(np.array([1.0, 2.0, 3.0]), eta=2.0)
        assert w.shape == (3,)
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()

    def test_lower_loss_gets_more_weight(self):
        w = exponential_weights(np.array([1.0, 2.0, 5.0]), eta=1.0)
        assert w[0] > w[1] > w[2]

    def test_shift_invariance(self):
        """A softmax is invariant to a common shift in the losses."""
        ssr = np.array([0.4, 0.9, 1.3])
        a = exponential_weights(ssr, eta=3.0)
        b = exponential_weights(ssr + 17.0, eta=3.0)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_eta_zero_is_the_simple_average(self):
        w = exponential_weights(np.array([1.0, 2.0, 9.0]), eta=0.0)
        np.testing.assert_allclose(w, np.full(3, 1 / 3), atol=1e-12)

    def test_large_eta_concentrates_on_the_best(self):
        w = exponential_weights(np.array([1.0, 2.0, 9.0]), eta=1e4)
        assert w[0] > 1 - 1e-6

    def test_permutation_equivariance(self):
        ssr = np.array([1.0, 4.0, 2.0])
        p = np.array([2, 0, 1])
        np.testing.assert_allclose(
            exponential_weights(ssr, eta=1.5)[p],
            exponential_weights(ssr[p], eta=1.5), atol=1e-12)

    def test_no_overflow_on_huge_losses(self):
        """exp(-eta * ssr) underflows to zero for every expert unless shifted."""
        w = exponential_weights(np.array([1e6, 1e6 + 1.0]), eta=1e3)
        assert np.isfinite(w).all() and np.isclose(w.sum(), 1.0)
        assert w[0] > w[1]

    @pytest.mark.parametrize("eta,expected", [(0.0, 4.0), (1e4, 1.0)])
    def test_effective_k_spans_one_to_k(self, eta, expected):
        w = exponential_weights(np.array([1.0, 2.0, 3.0, 4.0]), eta=eta)
        assert effective_k(w) == pytest.approx(expected, abs=1e-4)

    def test_effective_k_is_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = exponential_weights(rng.random(5) * 10, eta=rng.random() * 50)
            assert 1.0 - 1e-9 <= effective_k(w) <= 5.0 + 1e-9

    def test_paper_eta_matches_its_formula(self):
        y = np.arange(1.0, 26.0)
        assert paper_eta(y, 88) == pytest.approx(
            1.0 / (np.sqrt(88) * np.var(y, ddof=1)))

    def test_paper_eta_refuses_a_constant_series(self):
        """var(y) = 0 makes the formula a division by zero, not a large eta."""
        with pytest.raises(MlsynthDataError, match="constant"):
            paper_eta(np.ones(10), 10)


# --------------------------------------------------------------------------- #
# Layer 1: the test statistic and the bias adjustment
# --------------------------------------------------------------------------- #

class TestStatistic:
    def test_non_negative_and_zero_only_when_exact(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.arange(4)
        assert sl_statistic(y.copy(), y, w) == pytest.approx(0.0)
        assert sl_statistic(y + 0.5, y, w) > 0

    def test_scales_quadratically(self):
        y = np.zeros(9)
        r = np.array([0.3, -0.7, 1.1, 0.2, -0.4, 0.9, -1.2, 0.6, 0.1])
        w = np.arange(9)
        a = sl_statistic(r, y, w)
        b = sl_statistic(3.0 * r, y, w)
        assert b == pytest.approx(9.0 * a)

    def test_divides_by_sqrt_n(self):
        """Equations 7 and 8 normalise by sqrt of the window length."""
        y = np.zeros(16)
        pred = np.full(16, 2.0)
        got = sl_statistic(pred, y, np.arange(16))
        assert got == pytest.approx(16 * 4.0 / np.sqrt(16))

    def test_bias_adjustment_is_a_difference_of_residual_means(self):
        y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        pred = np.array([1.5, 2.5, 3.5, 10.0, 11.0, 12.0])
        W, P = np.arange(3), np.arange(3, 6)
        att, bias = bias_adjusted_att(pred, y, W, P)
        assert bias == pytest.approx(-0.5)
        assert att == pytest.approx(0.0 - (-0.5))

    def test_level_shift_in_the_outcome_leaves_the_att_alone(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(20)
        pred = y + rng.standard_normal(20) * 0.1
        W, P = np.arange(10), np.arange(10, 20)
        a, _ = bias_adjusted_att(pred, y, W, P)
        b, _ = bias_adjusted_att(pred + 5.0, y + 5.0, W, P)
        assert a == pytest.approx(b)

    def test_a_planted_constant_effect_comes_back(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(30)
        pred = y + rng.standard_normal(30) * 0.05
        W, P = np.arange(15), np.arange(15, 30)
        base, _ = bias_adjusted_att(pred, y, W, P)
        y2 = y.copy()
        y2[P] += 2.5
        shifted, _ = bias_adjusted_att(pred, y2, W, P)
        assert shifted - base == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Layer 1: the diagnostics the paper does not report
# --------------------------------------------------------------------------- #

class TestDiagnostics:
    def test_participation_ratio_is_one_when_errors_are_proportional(self):
        base = np.array([1.0, -2.0, 0.5, 0.9, -1.4])
        R = np.column_stack([base, 2 * base, -3 * base])
        assert participation_ratio(R) == pytest.approx(1.0, abs=1e-8)

    def test_participation_ratio_is_k_when_errors_are_orthogonal(self):
        R = np.eye(4) * 2.0
        assert participation_ratio(R) == pytest.approx(4.0, abs=1e-8)

    def test_participation_ratio_is_bounded(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            R = rng.standard_normal((12, 4))
            assert 1.0 - 1e-9 <= participation_ratio(R) <= 4.0 + 1e-9

    def test_participation_ratio_of_an_all_zero_error_matrix(self):
        """An exact library has no error structure; report 1, not a nan."""
        assert participation_ratio(np.zeros((6, 3))) == pytest.approx(1.0)

    def test_error_matrix_shape_and_content(self):
        pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 3.0, 5.0])
        R = error_matrix(pred, y, np.arange(3))
        np.testing.assert_allclose(R, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))

    def test_error_correlation_is_symmetric_with_unit_diagonal(self):
        rng = np.random.default_rng(2)
        C = error_correlation(rng.standard_normal((20, 3)))
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(C), np.ones(3), atol=1e-12)

    def test_error_correlation_survives_a_constant_column(self):
        """A constant error column has zero variance; the correlation is nan-free."""
        R = np.column_stack([np.array([1.0, -1.0, 2.0]), np.ones(3)])
        C = error_correlation(R)
        assert np.isfinite(C).all()

    def test_flags_a_constant_expert(self):
        R = np.column_stack([np.array([0.1, -0.2, 0.3]), np.full(3, 0.4)])
        flags = flag_degenerate(("good", "flat"), R,
                                ssr=np.array([0.14, 0.48]),
                                weights=np.array([0.6, 0.4]))
        assert "flat" in flags and "constant" in flags["flat"]
        assert "good" not in flags

    def test_flags_a_badly_fitting_expert_that_still_carries_weight(self):
        """The Prop 99 failure: 51x the best expert's loss, 14% of the weight."""
        R = np.column_stack([np.array([0.1, -0.1, 0.05, -0.05]),
                             np.array([3.0, -2.5, 2.8, -3.1])])
        ssr = (R ** 2).sum(axis=0)
        flags = flag_degenerate(("tight", "loose"), R, ssr=ssr,
                                weights=np.array([0.86, 0.14]))
        assert "loose" in flags and "loss" in flags["loose"]

    def test_does_not_flag_a_bad_expert_with_negligible_weight(self):
        R = np.column_stack([np.array([0.1, -0.1, 0.05, -0.05]),
                             np.array([3.0, -2.5, 2.8, -3.1])])
        ssr = (R ** 2).sum(axis=0)
        flags = flag_degenerate(("tight", "loose"), R, ssr=ssr,
                                weights=np.array([0.999, 0.001]))
        assert "loose" not in flags


# --------------------------------------------------------------------------- #
# Layer 1: the expert library
# --------------------------------------------------------------------------- #

class TestExperts:
    @staticmethod
    def _arrays(seed=0, T=60, N=6):
        rng = np.random.default_rng(seed)
        f = rng.standard_normal((T, 2))
        lam = rng.standard_normal((N, 2))
        Yco = 10.0 + f @ lam.T + 0.4 * rng.standard_normal((T, N))
        y = 10.0 + f @ rng.standard_normal(2) + 0.4 * rng.standard_normal(T)
        return Yco, y

    def test_builds_the_papers_four(self):
        Yco, y = self._arrays()
        lib = build_experts(Yco, y, train=slice(0, 30), names=EXPERTS)
        assert lib.names == EXPERTS
        assert lib.predictions.shape == (60, 4)
        assert np.isfinite(lib.predictions).all()

    def test_expert_order_follows_the_requested_order(self):
        Yco, y = self._arrays()
        a = build_experts(Yco, y, slice(0, 30), ("did", "lasso"))
        b = build_experts(Yco, y, slice(0, 30), ("lasso", "did"))
        assert a.names == ("did", "lasso") and b.names == ("lasso", "did")
        np.testing.assert_allclose(a.predictions[:, 0], b.predictions[:, 1])

    def test_lasso_expert_is_seed_independent(self):
        """The defect fix: the penalty's folds are contiguous, not drawn.

        Their ``cv.glmnet(nfolds=5)`` picks the fold assignment from the RNG, so
        ``lambda.min`` lands in one of three places on a 30-period window and the
        effect moves 40 percent with it. Non-shuffled folds also respect time
        order, which a random split does not.
        """
        Yco, y = self._arrays()
        p = [build_experts(Yco, y, slice(0, 30), ("lasso",), seed=s).predictions
             for s in (0, 1, 7, 2026)]
        for q in p[1:]:
            np.testing.assert_allclose(p[0], q, atol=0.0)

    def test_forest_expert_is_reproducible_given_the_seed(self):
        Yco, y = self._arrays()
        kw = dict(train=slice(0, 30), names=("forest",))
        a = build_experts(Yco, y, **kw, seed=5).predictions
        b = build_experts(Yco, y, **kw, seed=5).predictions
        np.testing.assert_allclose(a, b, atol=0.0)

    def test_covariates_reach_the_forest(self):
        """The forest is the only expert with a different information set."""
        Yco, y = self._arrays()
        rng = np.random.default_rng(9)
        Z = rng.standard_normal((60, 3))
        bare = build_experts(Yco, y, slice(0, 30), ("forest",), seed=0)
        with_cov = build_experts(Yco, y, slice(0, 30), ("forest",), seed=0,
                                 covariates=Z)
        assert not np.allclose(bare.predictions, with_cov.predictions)

    def test_did_expert_is_the_parallel_trends_prediction(self):
        """Closed form: mean(y_train) - mean(donor means on train) + donor mean."""
        Yco, y = self._arrays()
        tr = slice(0, 30)
        lib = build_experts(Yco, y, tr, ("did",))
        want = y[tr].mean() - Yco[tr].mean(axis=0).mean() + Yco.mean(axis=1)
        np.testing.assert_allclose(lib.predictions[:, 0], want, atol=1e-10)

    def test_did_expert_is_exact_under_parallel_trends(self):
        rng = np.random.default_rng(11)
        common = rng.standard_normal(40)
        Yco = common[:, None] + np.arange(1.0, 5.0)[None, :]
        y = common + 7.0
        lib = build_experts(Yco, y, slice(0, 20), ("did",))
        np.testing.assert_allclose(lib.predictions[:, 0], y, atol=1e-10)

    def test_predictions_span_every_period_not_just_the_post_window(self):
        Yco, y = self._arrays(T=45)
        lib = build_experts(Yco, y, slice(0, 20), EXPERTS)
        assert lib.predictions.shape[0] == 45

    def test_an_expert_that_cannot_fit_is_dropped_and_recorded(self):
        """A single donor leaves the factor expert nothing to decompose."""
        Yco, y = self._arrays(N=1)
        lib = build_experts(Yco, y, slice(0, 30), EXPERTS)
        assert len(lib.names) >= 1
        assert set(lib.names) | set(lib.dropped) == set(EXPERTS)
        for reason in lib.dropped.values():
            assert reason

    def test_every_expert_failing_is_an_error_not_an_empty_library(self):
        with pytest.raises(MlsynthDataError, match="expert"):
            build_experts(np.zeros((10, 0)), np.zeros(10), slice(0, 5), EXPERTS)

    def test_lasso_records_its_penalty(self):
        Yco, y = self._arrays()
        lib = build_experts(Yco, y, slice(0, 30), ("lasso",))
        assert "alpha" in lib.details["lasso"]
        assert lib.details["lasso"]["alpha"] > 0


# --------------------------------------------------------------------------- #
# Layer 1: the block bootstrap, Algorithm 2
# --------------------------------------------------------------------------- #

class TestBootstrap:
    @staticmethod
    def _setup(seed=0, T=60, K=3):
        rng = np.random.default_rng(seed)
        y = np.cumsum(rng.standard_normal(T)) + 20.0
        pred = y[:, None] + rng.standard_normal((T, K)) * 0.3
        return y, pred

    def test_p_value_is_a_probability(self):
        y, pred = self._setup()
        r = block_bootstrap_test(pred, y, train=slice(0, 30),
                                weight=np.arange(30, 45), post=np.arange(45, 60),
                                eta=1.0, n_boot=200, block=3, seed=0)
        assert 0.0 <= r.p_value <= 1.0

    def test_critical_values_are_ordered_by_level(self):
        y, pred = self._setup()
        r = block_bootstrap_test(pred, y, slice(0, 30), np.arange(30, 45),
                                 np.arange(45, 60), eta=1.0, n_boot=400,
                                 block=3, seed=0, levels=(0.01, 0.05, 0.10, 0.20))
        cv = [r.critical_values[k] for k in (0.01, 0.05, 0.10, 0.20)]
        assert cv == sorted(cv, reverse=True)

    def test_deterministic_given_the_seed(self):
        y, pred = self._setup()
        kw = dict(train=slice(0, 30), weight=np.arange(30, 45),
                  post=np.arange(45, 60), eta=1.0, n_boot=200, block=3)
        a = block_bootstrap_test(pred, y, **kw, seed=11)
        b = block_bootstrap_test(pred, y, **kw, seed=11)
        assert a.p_value == b.p_value
        assert a.critical_values == b.critical_values

    def test_bootstrap_uses_the_configured_eta(self):
        """Their library.R:214 hardcodes eta=1 here while the estimate uses 51.43.

        Measured on the paper's own panel, that inflates the 10 percent critical
        value by 19 percent on one block and 61 percent on another. The critical
        values must move with eta, or the null distribution belongs to a
        differently weighted ensemble than the statistic it is compared against.
        """
        y, pred = self._setup()
        kw = dict(train=slice(0, 30), weight=np.arange(30, 45),
                  post=np.arange(45, 60), n_boot=300, block=3, seed=0)
        lo = block_bootstrap_test(pred, y, **kw, eta=1.0)
        hi = block_bootstrap_test(pred, y, **kw, eta=500.0)
        assert lo.critical_values != hi.critical_values

    def test_a_large_planted_effect_is_rejected(self):
        y, pred = self._setup(seed=2)
        post = np.arange(45, 60)
        y2 = y.copy()
        y2[post] += 30.0
        r = block_bootstrap_test(pred, y2, slice(0, 30), np.arange(30, 45),
                                 post, eta=1.0, n_boot=400, block=3, seed=0)
        assert r.p_value < 0.05
        assert r.statistic > r.critical_values[0.05]

    def test_no_effect_is_not_rejected(self):
        y, pred = self._setup(seed=3)
        r = block_bootstrap_test(pred, y, slice(0, 30), np.arange(30, 45),
                                 np.arange(45, 60), eta=1.0, n_boot=400,
                                 block=3, seed=0)
        assert r.p_value > 0.05

    def test_n_boot_is_honoured(self):
        y, pred = self._setup()
        r = block_bootstrap_test(pred, y, slice(0, 30), np.arange(30, 45),
                                 np.arange(45, 60), eta=1.0, n_boot=137,
                                 block=3, seed=0)
        assert r.n_boot == 137


# --------------------------------------------------------------------------- #
# Layer 4: the assembled estimator
# --------------------------------------------------------------------------- #

class TestSmoke:
    def test_returns_the_contract(self, fitted):
        assert isinstance(fitted, SLResults)
        assert np.isfinite(fitted.effects.att)
        assert fitted.time_series.counterfactual_outcome.shape == (60,)
        assert fitted.time_series.estimated_gap.shape == (60,)

    def test_recovers_the_planted_effect(self, fitted):
        assert fitted.effects.att == pytest.approx(TAU, abs=1.0)

    def test_no_standard_error_is_claimed(self, fitted):
        """The paper supplies a test, not an interval. SE_TT is dead code there."""
        assert fitted.effects.att_std_err is None
        assert fitted.inference.standard_error is None
        assert fitted.inference.ci_lower is None
        assert fitted.inference.ci_upper is None

    def test_the_test_is_reported(self, fitted):
        f = fitted.fit
        assert f.test_statistic >= 0
        assert 0.0 <= f.p_value <= 1.0
        assert 0.05 in f.critical_values
        assert fitted.inference.p_value == pytest.approx(f.p_value)
        assert fitted.inference.method == "sl_block_bootstrap"

    def test_expert_weights_are_on_the_result(self, fitted):
        w = fitted.weights.donor_weights
        assert set(w) == set(fitted.fit.experts)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_diagnostics_are_reported(self, fitted):
        f = fitted.fit
        K = len(f.experts)
        assert 1.0 - 1e-9 <= f.effective_k <= K + 1e-9
        assert 1.0 - 1e-9 <= f.error_participation_ratio <= K + 1e-9
        assert f.error_correlation.shape == (K, K)
        assert isinstance(f.degenerate_experts, dict)

    def test_the_split_is_recorded(self, fitted):
        f = fitted.fit
        assert f.train_periods + f.weight_periods == 40
        assert f.train_periods > 0 and f.weight_periods > 0

    def test_counterfactual_is_the_weighted_expert_combination(self, fitted):
        f = fitted.fit
        w = np.array([f.weights[n] for n in f.experts])
        np.testing.assert_allclose(f.predictions @ w, f.counterfactual, atol=1e-10)

    def test_gap_is_observed_minus_counterfactual(self, fitted):
        np.testing.assert_allclose(
            fitted.time_series.estimated_gap,
            fitted.time_series.observed_outcome - fitted.time_series.counterfactual_outcome,
            atol=1e-12)

    def test_deterministic_across_repeat_fits(self):
        a = SL(_cfg(_panel())).fit()
        b = SL(_cfg(_panel())).fit()
        assert a.effects.att == b.effects.att
        assert a.fit.p_value == b.fit.p_value
        assert a.fit.weights == b.fit.weights


class TestBehaviour:
    def test_eta_zero_makes_it_the_simple_average(self):
        r = SL(_cfg(_panel(), eta=0.0)).fit()
        K = len(r.fit.experts)
        assert r.fit.effective_k == pytest.approx(float(K), abs=1e-6)
        np.testing.assert_allclose(
            r.fit.counterfactual, r.fit.predictions.mean(axis=1), atol=1e-10)

    def test_large_eta_selects_one_expert(self):
        r = SL(_cfg(_panel(), eta=1e8)).fit()
        assert r.fit.effective_k == pytest.approx(1.0, abs=1e-3)

    def test_the_resolved_eta_is_recorded(self):
        r = SL(_cfg(_panel(), eta=7.5)).fit()
        assert r.fit.eta == pytest.approx(7.5)
        auto = SL(_cfg(_panel())).fit()
        assert auto.fit.eta > 0 and np.isfinite(auto.fit.eta)

    def test_a_smaller_library_is_honoured(self):
        r = SL(_cfg(_panel(), experts=["did", "lasso"])).fit()
        assert r.fit.experts == ("did", "lasso")
        assert r.fit.predictions.shape[1] == 2

    def test_post_skip_shortens_the_evaluation_window(self):
        a = SL(_cfg(_panel())).fit()
        b = SL(_cfg(_panel(), post_skip=5)).fit()
        assert b.fit.post_periods == a.fit.post_periods - 5
        assert b.fit.att != a.fit.att

    def test_train_periods_moves_the_split(self):
        r = SL(_cfg(_panel(), train_periods=25)).fit()
        assert r.fit.train_periods == 25 and r.fit.weight_periods == 15

    def test_covariates_are_passed_through(self):
        bare = SL(_cfg(_panel())).fit()
        cov = SL(_cfg(_panel(), covariates=["z"])).fit()
        assert not np.allclose(bare.fit.counterfactual, cov.fit.counterfactual)

    def test_no_interval_is_offered(self):
        """SL ships the paper's test and no interval, so there is no knob for one.

        An interval for this estimate means inverting SL's own statistic over
        candidate effects. ``conformal_att_interval`` is not a substitute: it
        refits a ridge on a donor design, so its interval belongs to another
        estimator's point estimate.
        """
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), interval="conformal"))
        r = SL(_cfg(_panel())).fit()
        assert r.inference.ci_lower is None and r.inference.ci_upper is None
        assert r.inference.method == "sl_block_bootstrap"
        assert "critical_values" in r.inference.details

    def test_a_bigger_effect_gives_a_smaller_p_value(self):
        small = SL(_cfg(_panel(tau=-0.2), n_boot=400)).fit()
        big = SL(_cfg(_panel(tau=-25.0), n_boot=400)).fit()
        assert big.fit.p_value <= small.fit.p_value
        assert big.fit.test_statistic > small.fit.test_statistic


# --------------------------------------------------------------------------- #
# Layer 3: edge cases
# --------------------------------------------------------------------------- #

class TestEdges:
    def test_single_donor(self):
        r = SL(_cfg(_panel(n_donors=1))).fit()
        assert np.isfinite(r.effects.att)
        assert len(r.fit.experts) >= 1

    def test_two_donors(self):
        r = SL(_cfg(_panel(n_donors=2))).fit()
        assert np.isfinite(r.effects.att)

    def test_collinear_donors(self):
        df = _panel(n_donors=4)
        dup = df[df.unit == 1].copy()
        dup["unit"] = 99
        r = SL(_cfg(pd.concat([df, dup], ignore_index=True))).fit()
        assert np.isfinite(r.effects.att)

    def test_a_single_post_period(self):
        r = SL(_cfg(_panel(T=41, T0=40))).fit()
        assert r.fit.post_periods == 1
        assert np.isfinite(r.effects.att)

    def test_the_shortest_workable_pre_window(self):
        r = SL(_cfg(_panel(T=20, T0=8, n_donors=3))).fit()
        assert r.fit.train_periods >= 1 and r.fit.weight_periods >= 1

    def test_block_longer_than_the_window_is_clamped(self):
        r = SL(_cfg(_panel(), block=999)).fit()
        assert 0.0 <= r.fit.p_value <= 1.0

    def test_a_degenerate_expert_is_recorded_not_hidden(self):
        """A panel with one donor drops the factor expert; the reason surfaces."""
        r = SL(_cfg(_panel(n_donors=1))).fit()
        assert set(r.fit.experts) | set(r.fit.dropped_experts) <= set(EXPERTS)
        assert r.method_details.parameters_used["dropped_experts"] == r.fit.dropped_experts


# --------------------------------------------------------------------------- #
# Layer 4: failures, each asserting the translated error
# --------------------------------------------------------------------------- #

class TestFailures:
    def test_empty_expert_library(self):
        with pytest.raises(MlsynthConfigError, match="at least one expert"):
            SLConfig(**_cfg(_panel(), experts=[]))

    def test_duplicate_experts(self):
        with pytest.raises(MlsynthConfigError, match="[Dd]uplicate"):
            SLConfig(**_cfg(_panel(), experts=["did", "did"]))

    def test_unknown_expert(self):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), experts=["xgboost"]))

    def test_train_periods_leaving_no_weighting_window(self):
        with pytest.raises((MlsynthConfigError, MlsynthDataError), match="weight"):
            SL(_cfg(_panel(), train_periods=40)).fit()

    def test_train_periods_beyond_the_pre_window(self):
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SL(_cfg(_panel(), train_periods=95)).fit()

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_train_periods(self, bad):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), train_periods=bad))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range(self, bad):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), alpha=bad))

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_n_boot(self, bad):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), n_boot=bad))

    @pytest.mark.parametrize("bad", [0, -2])
    def test_non_positive_block(self, bad):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), block=bad))

    def test_negative_eta(self):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), eta=-1.0))

    def test_negative_post_skip(self):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), post_skip=-1))

    def test_post_skip_consuming_the_whole_post_window(self):
        with pytest.raises((MlsynthConfigError, MlsynthDataError), match="post"):
            SL(_cfg(_panel(), post_skip=50)).fit()

    def test_covariate_not_in_the_frame(self):
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SL(_cfg(_panel(), covariates=["not_a_column"])).fit()

    def test_no_donors(self):
        df = _panel()
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SL(_cfg(df[df.unit == 0].copy())).fit()

    def test_no_pre_periods(self):
        df = _panel(T0=0)
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SL(_cfg(df)).fit()

    def test_extra_config_key_is_forbidden(self):
        with pytest.raises(Exception):
            SLConfig(**_cfg(_panel(), not_a_field=1))

    def test_never_treated_panel(self):
        df = _panel()
        df["D"] = 0
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            SL(_cfg(df)).fit()


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

class TestPlot:
    def test_returns_a_figure_and_does_not_show_it(self, monkeypatch, fitted):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mlsynth.utils.sl_helpers.plotter import plot_sl

        called = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))
        fig = plot_sl(fitted)
        assert fig is not None and not called
        plt.close(fig)

    def test_display_graphs_true_still_returns(self):
        import matplotlib
        matplotlib.use("Agg")
        r = SL(_cfg(_panel(), display_graphs=True)).fit()
        assert np.isfinite(r.effects.att)


# --------------------------------------------------------------------------- #
# the guards the happy path never reaches
# --------------------------------------------------------------------------- #

class TestGuards:
    def test_duplicate_covariates(self):
        with pytest.raises(MlsynthConfigError, match="[Dd]uplicate covariates"):
            SLConfig(**_cfg(_panel(), covariates=["z", "z"]))

    def test_a_non_finite_prediction_drops_the_expert(self, monkeypatch):
        """The guard exists because an expert can fit and still return nonsense."""
        import mlsynth.utils.sl_helpers.experts as ex

        monkeypatch.setattr(
            ex, "_did", lambda Yco, y, tr, detail: np.full_like(y, np.nan))
        rng = np.random.default_rng(0)
        Yco = rng.standard_normal((40, 4)) + 10.0
        y = rng.standard_normal(40) + 10.0
        lib = ex.build_experts(Yco, y, slice(0, 20), ("did", "lasso"))
        assert "did" in lib.dropped and "non-finite" in lib.dropped["did"]
        assert lib.names == ("lasso",)

    def test_a_covariate_with_a_gap_is_refused(self):
        df = _panel()
        df.loc[df.index[0], "z"] = np.nan
        with pytest.raises(MlsynthDataError, match="incomplete"):
            prepare_sl_inputs(df, unitid="unit", time="time", outcome="y",
                              treat="D", covariates=["z"])

    def test_a_staggered_panel_is_refused(self):
        """dataprep cohorts a multi-treated panel; SL needs the single-treated form."""
        df = _panel()
        df.loc[(df.unit == 1) & (df.time >= 45), "D"] = 1
        with pytest.raises(MlsynthDataError, match="one treated unit"):
            prepare_sl_inputs(df, unitid="unit", time="time", outcome="y",
                              treat="D")

    def test_one_pre_period_cannot_be_split(self):
        """Algorithm 1 needs a period on each side of the split."""
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_sl_inputs(_panel(T=30, T0=1), unitid="unit", time="time",
                              outcome="y", treat="D")

    def test_inputs_report_their_shape(self):
        inp = prepare_sl_inputs(_panel(n_donors=5), unitid="unit", time="time",
                                outcome="y", treat="D")
        assert inp.N == 5 and inp.T == 60 and inp.T0 == 40

    def test_prepopulated_submodels_are_left_alone(self, fitted):
        """Revalidating a result must not overwrite what it already carries."""
        marker = EffectsResults(att=-123.456)
        again = SLResults(inputs=fitted.inputs, fit=fitted.fit, effects=marker)
        assert again.effects.att == pytest.approx(-123.456)
        assert again.time_series is None

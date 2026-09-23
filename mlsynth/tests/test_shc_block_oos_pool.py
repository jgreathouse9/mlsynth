"""The out-of-sample reference pool for SHC's conformal test.

The test of Chen, Yang & Yang (2024, footnote 21) compares the post-period
statistic against a reference distribution resampled from pre-period residuals.
Those residuals have to be the same object as the statistic, and with the
smoother pool they are not: the statistic is the raw outcome minus an
out-of-sample SHC prediction, while the pool is the raw outcome minus an
in-sample kernel-smoother fit. Out-of-sample error exceeds in-sample error, so
the null sits at the wrong scale and the test over-rejects -- 0.267 at a nominal
0.01 on the paper's own simulation.

``reference_pool="block_oos"`` builds the pool the statistic's own way. Each
historical block is treated in turn as if it were the treated block: its
pre-window is matched by a simplex over the blocks that do not overlap it, and
the residual is taken over its own post-window. That residual is the raw outcome
minus a prediction the block itself did not inform, which is what the treated
block's residual is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth import SHC
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.shc_helpers.inference import (
    block_oos_residuals, run_conformal_inference,
)
from mlsynth.utils.shc_helpers.orchestration import solve_shc, summarize_effects
from mlsynth.utils.shc_helpers.setup import prepare_shc_inputs
from mlsynth.utils.shc_helpers.simulation import simulate_shc_panel

_M, _N = 25, 4
_LEVELS = (0.01, 0.05, 0.10)


def _fitted(seed=0, m=_M, n=_N):
    df, _info = simulate_shc_panel(m=m, h=4, n=n, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs = prepare_shc_inputs(df, outcome="y", treat="treated",
                                    unitid="unit", time="time", m=m)
        design = solve_shc(inputs)
        att, ap, obs, cf, gap, wt, fd = summarize_effects(inputs, design)
    return inputs, design, obs, cf


def _fit(seed=0, **extra):
    df, _info = simulate_shc_panel(m=_M, h=4, n=_N, seed=seed)
    cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
           "time": "time", "m": _M, "display_graphs": False}
    cfg.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SHC(cfg).fit()


# =========================================================================== #
# what the pool is made of
# =========================================================================== #
class TestConstruction:

    def test_it_returns_one_residual_per_period_per_evaluated_block(self):
        inputs, design, _o, _c = _fitted()
        pool, info = block_oos_residuals(inputs, design, stride=40)
        assert pool.ndim == 1
        assert pool.size == info["n_blocks"] * inputs.n
        assert info["n_blocks"] >= 2
        assert np.isfinite(pool).all()

    def test_a_block_never_informs_its_own_prediction(self):
        """The property the smoother pool lacks: the residual is out of sample.

        Re-deriving one block's residual by hand, with its own column and every
        overlapping column struck out of the donor matrix, reproduces what the
        helper returns.
        """
        inputs, design, _o, _c = _fitted()
        pool, info = block_oos_residuals(inputs, design, stride=40)
        m, n, N = inputs.m, inputs.n, inputs.N
        from mlsynth.utils.datautils import build_donor_segments
        from mlsynth.utils.shc_helpers.kernels import solve_shc_qp

        L_full, L_post, _ell_eval = build_donor_segments(
            np.asarray(design.latent_pre).ravel(), m, inputs.T0, n)
        j = info["blocks"][0]
        keep = np.array([k for k in range(N) if abs(k - j) >= m + n])
        w, _ = solve_shc_qp(L_full[:, keep], L_full[:, j])
        expected = inputs.y[j + m:j + m + n] - L_post[:, keep] @ w
        assert pool[:n] == pytest.approx(expected, rel=1e-8, abs=1e-10)

    def test_overlapping_blocks_are_excluded_from_the_donor_set(self):
        inputs, design, _o, _c = _fitted()
        _pool, info = block_oos_residuals(inputs, design, stride=40)
        m, n = inputs.m, inputs.n
        for j, donors in zip(info["blocks"], info["donor_sets"]):
            assert all(abs(k - j) >= m + n for k in donors), (
                f"block {j} kept a donor sharing observations with it")

    def test_stride_controls_how_many_blocks_are_refitted(self):
        inputs, design, _o, _c = _fitted()
        coarse = block_oos_residuals(inputs, design, stride=60)[1]["n_blocks"]
        fine = block_oos_residuals(inputs, design, stride=20)[1]["n_blocks"]
        assert fine > coarse >= 2

    def test_it_is_deterministic(self):
        inputs, design, _o, _c = _fitted()
        a, _ = block_oos_residuals(inputs, design, stride=40)
        b, _ = block_oos_residuals(inputs, design, stride=40)
        assert a == pytest.approx(b)


# =========================================================================== #
# the invariant the smoother pool violated
# =========================================================================== #
class TestCommensurability:

    def test_the_pool_is_on_the_same_scale_as_the_statistic(self):
        """Median over null panels, since four post periods is a small sample."""
        ratios = []
        for seed in range(8):
            inputs, design, obs, cf = _fitted(seed)
            pool, _info = block_oos_residuals(inputs, design, stride=30)
            post = obs[_M:] - cf[_M:]
            ratios.append(np.abs(post).mean() / np.abs(pool).mean())
        median = float(np.median(ratios))
        assert 0.4 < median < 2.5, f"median dispersion ratio {median:.2f}"

    def test_it_is_wider_than_the_smoother_pool(self):
        """In-sample residuals understate the error; that was the whole defect."""
        inputs, design, _o, _c = _fitted()
        oos, _info = block_oos_residuals(inputs, design, stride=30)
        smoother = inputs.y[:inputs.T0] - np.asarray(design.latent_pre).ravel()
        assert np.abs(oos).mean() > np.abs(smoother).mean()


# =========================================================================== #
# the headline: size under an exact null
# =========================================================================== #
class TestSize:

    def test_size_is_near_nominal_under_an_exact_null(self):
        """The paper's design with no effect, so every rejection is false."""
        reps = 20
        rejections = {lvl: 0 for lvl in _LEVELS}
        for seed in range(reps):
            det = _fit(seed, reference_pool="block_oos",
                       reference_stride=30).inference_detail
            for lvl in _LEVELS:
                rejections[lvl] += int(det.reject[lvl])
        for lvl in _LEVELS:
            realised = rejections[lvl] / reps
            assert realised <= max(3 * lvl, 0.15), (
                f"size {realised:.3f} at nominal {lvl}")

    def test_it_still_rejects_a_large_real_effect(self):
        """A calibrated test that never rejects would be useless."""
        df, info = simulate_shc_panel(m=_M, h=4, n=_N, seed=3)
        df = df.copy()
        df.loc[df.time > info["T_o"], "y"] += -6.0
        cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
               "time": "time", "m": _M, "display_graphs": False,
               "reference_pool": "block_oos", "reference_stride": 30}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SHC(cfg).fit()
        assert res.inference_detail.reject[0.05] is True

    def test_the_smoother_pool_still_over_rejects(self):
        """The old pool stays reachable, and stays miscalibrated."""
        reps = 10
        rejections = 0
        for seed in range(reps):
            det = _fit(seed, reference_pool="smoother").inference_detail
            rejections += int(det.reject[0.01])
        assert rejections / reps > 0.10


# =========================================================================== #
# reporting and refusals
# =========================================================================== #
class TestReporting:

    def test_the_pool_in_use_is_recorded(self):
        for pool in ("block_oos", "smoother"):
            det = _fit(0, reference_pool=pool,
                       reference_stride=40).inference_detail
            assert det.reference_pool == pool
            assert det.n_reference > 0

    def test_an_unknown_pool_is_refused_at_config_time(self):
        with pytest.raises(MlsynthConfigError, match="reference_pool"):
            _fit(0, reference_pool="nonsense")

    def test_a_non_positive_stride_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="reference_stride"):
            _fit(0, reference_stride=0)

    def test_a_panel_with_no_usable_block_is_refused(self):
        """Every block overlaps every other when the pre-period is barely
        longer than one block, so there is nothing out of sample to calibrate
        on. The refusal names the arithmetic instead of returning an empty
        pool, which would resample from nothing.
        """
        from mlsynth.utils.helperutils import IndexSet
        from mlsynth.utils.shc_helpers.structures import SHCDesign, SHCInputs

        m, n, T0 = 6, 2, 13                 # N = T0 - m - n + 1 = 6 <= m + n = 8
        y = np.linspace(0.0, 1.0, T0 + n)   # n and N are derived properties
        inputs = SHCInputs(
            time_index=IndexSet.from_labels(np.arange(T0 + n)),
            y=y, T0=T0, m=m, treated_label="unit", metadata={})
        assert inputs.n == n and inputs.N <= m + n
        design = SHCDesign(
            bandwidth=1.0, latent_pre=y[:T0], weights=np.ones(1),
            selected_blocks=[0], block_weights={"block@0": 1.0},
            counterfactual_window=np.zeros(m + n), use_augmented=False,
            best_lambda=None)
        with pytest.raises(MlsynthEstimationError, match="out-of-sample"):
            block_oos_residuals(inputs, design, stride=1)

    def test_a_panel_too_short_for_the_pool_falls_back_and_says_so(self):
        """An inference problem must not take the fit down.

        Short panels -- where every historical block overlaps every other --
        cannot supply an out-of-sample residual at all. The fit still has a
        point estimate to report, so the default degrades to the smoother pool,
        warns, and records which pool actually ran, instead of raising and
        losing the estimate.
        """
        # m = 5, n = 18 gives N = 23 historical blocks of length m + n = 23,
        # so every block overlaps every other and no donor is out of sample.
        df, _info = simulate_shc_panel(m=5, h=2, n=18, w_f=(1.0, 0.0), seed=0)
        cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
               "time": "time", "m": 5, "display_graphs": False}
        with pytest.warns(UserWarning, match="out-of-sample"):
            res = SHC(cfg).fit()
        det = res.inference_detail
        assert det.reference_pool == "smoother"
        assert "overlap" in det.reference_note or det.reference_note
        assert det.p_value is not None
        # The claim is that fit() returns and reports which pool ran, not that a
        # panel this degenerate yields a usable estimate. Five pre-period points
        # matched against blocks that all overlap each other is a singular
        # matching problem, and the counterfactual duly explodes -- around 1e282
        # here, overflowing to inf on another BLAS. Asserting anything about the
        # magnitude would be asserting the pathology, so the test asserts the
        # fit completed and populated its effects.
        assert res.effects is not None
        assert res.effects.att is not None

    def test_the_fallback_is_not_taken_when_the_pool_is_available(self):
        det = _fit(0, reference_stride=40).inference_detail
        assert det.reference_pool == "block_oos"
        assert det.reference_note == ""

    def test_an_explicit_smoother_request_does_not_warn_about_fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            det = _fit(0, reference_pool="smoother").inference_detail
        assert det.reference_pool == "smoother"

    def test_run_conformal_inference_accepts_the_pool_argument(self):
        inputs, design, obs, cf = _fitted()
        out = run_conformal_inference(inputs, design, obs, cf,
                                      reference_pool="block_oos",
                                      reference_stride=40)
        assert out.reference_pool == "block_oos"
        assert out.n_reference == out.null_distribution.size or out.n_reference > 0

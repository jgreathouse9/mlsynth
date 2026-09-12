"""The five whys for SHC's conformal test, as tests.

Incident: on real US four-quarter GDP growth the test rejects the sharp null at
the 1% level in 5 of 5 placebo windows with p = 0.0000, one of them with an ATT
of +0.244, and reports a 1% critical value of 0.009 where Chen, Yang & Yang
report 8.517 for the same design (1956Q1-2020Q4, m = 24, n = 4). The point
estimates are unaffected and replicate the paper: S = 11.924 against 11.896,
2020Q2 effect -11.380 against -11.719, N = 229 exactly.

The chain, each stage depending only on the ones to its right::

    p_value, critical_values  <-  null distribution  <-  reference pool
        <-  y_t - latent_pre_t  <-  smooth(y, h)  <-  h from loocv_bandwidth

Links cleared, with the evidence in each test below:

* the observed statistic (scaling, absolute value, window, sign) -- cleared,
  with A2 re-cleared on a mixed-sign design because the COVID panel's four post
  gaps are all negative and there ``|sum|`` and ``sum|.|`` coincide, so the
  check had no power there;
* the resampler (with replacement, ``n`` draws, upper tail, ``P(S* >= S)``) --
  cleared against an independent reimplementation at the same seed.

Two causes, on the two links that were not cleared.

1. The pool and the statistic are different constructions, and this is the
   bottom. The statistic is ``y_t`` minus an out-of-sample SHC prediction; the
   pool is ``y_t`` minus an in-sample smoother fit. Out-of-sample error exceeds
   in-sample error, by a median factor of about two on the paper's own
   simulation, and with ``n = 4`` summed absolute residuals a factor of two
   puts the statistic past the 99th percentile of the null. That is the whole
   of the measured size: 0.267 at the 1% level on a design where the smoother
   does not interpolate and the two scales are within an order of magnitude.

2. On real four-quarter GDP growth ``loocv_bandwidth`` additionally returns the
   smallest value on its grid. The criterion measures leave-one-out prediction,
   and a series whose neighbours predict it almost exactly is best predicted by
   the tightest fit; ``smooth`` has no leave-out, so at that bandwidth it
   interpolates. ``latent_pre`` then correlates 0.9999995 with ``y`` and the
   pool's sd is 0.0027 against the series' 2.291. This is an amplifier, not a
   cause: it takes the critical value from 7.6 -- the most a maximally
   dispersed in-sample pool can give -- down to 0.009, which is 854 of the
   957-fold gap against the paper's 8.517. The paper's own simulation does not
   exhibit it at all.

Confirmation. Cause 1: the failure does not occur without it -- building the
pool the statistic's own way, out-of-sample refits over each historical block's
post-window, takes size to zero. Cause 2: the failure still occurs without it --
fixing the bandwidth at 2.0 leaves size at 0.167 -- so it is a contributing
factor and the ladder does not stop there. Both are faults of omission: nothing
asserted that the reference residuals and the tested residuals are the same
object, and nothing asserted that the selected bandwidth was interior to its
grid.

The size assertions are ``xfail(strict=True)``: they state the behaviour the
test must have, they fail today for the documented reason, and they turn into a
suite failure the moment the construction is corrected and nobody updates them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth import SHC
from mlsynth.utils.shc_helpers.inference import shc_conformal_test
from mlsynth.utils.shc_helpers.kernels import loocv_bandwidth, smooth
from mlsynth.utils.shc_helpers.orchestration import solve_shc, summarize_effects
from mlsynth.utils.shc_helpers.setup import prepare_shc_inputs
from mlsynth.utils.shc_helpers.simulation import simulate_shc_panel

_M, _N, _REPS = 25, 4, 12
_LEVELS = (0.01, 0.05, 0.10)


def _null_panel(seed):
    """The paper's own design with no effect: every rejection is a false one."""
    df, _info = simulate_shc_panel(m=_M, h=4, n=_N, seed=seed)
    return df


def _fit(df, **extra):
    """Fit with the diagnosed pool by default.

    The cause this ladder walks back to was corrected by
    ``reference_pool="block_oos"``, which is now the estimator's default, so the
    rungs name ``"smoother"`` explicitly. The ladder is a record of why that
    default changed, and it keeps failing on the pool it diagnosed.
    """
    cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
           "time": "time", "m": _M, "display_graphs": False,
           "reference_pool": "smoother"}
    cfg.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SHC(cfg).fit()


# =========================================================================== #
# rung 0 -- the incident, reproduced deterministically
# =========================================================================== #
class TestRung0Size:

    def test_the_corrected_pool_fixes_the_size_and_the_diagnosed_one_does_not(self):
        """Rung 0 in its post-fix form: the incident, and the cause, together.

        Switching only the reference pool on the same panels moves the
        rejection rate from grossly over-nominal to near-nominal, which is what
        establishes the pool as the cause instead of merely a correlate of it.
        """
        reps = 8
        counts = {"smoother": 0, "block_oos": 0}
        for seed in range(reps):
            df = _null_panel(seed)
            for pool in counts:
                det = _fit(df, reference_pool=pool,
                           reference_stride=40).inference.details
                counts[pool] += int(det["reject"][0.01])
        smoother = counts["smoother"] / reps
        corrected = counts["block_oos"] / reps
        assert smoother >= 0.25, (
            f"the diagnosed pool should still over-reject grossly, saw "
            f"{smoother:.3f}")
        assert corrected <= 0.15, (
            f"the corrected pool should be near nominal, saw {corrected:.3f}")
        assert corrected < smoother

    def test_the_default_is_the_corrected_pool(self):
        """What the fix changed, asserted where a reader will look for it."""
        df = _null_panel(0)
        cfg = {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit",
               "time": "time", "m": _M, "display_graphs": False,
               "reference_stride": 40}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SHC(cfg).fit()
        assert res.inference.details["reference_pool"] == "block_oos"


# =========================================================================== #
# rung 1 -- which other outputs moved with it
# =========================================================================== #
class TestRung1OtherOutputs:

    def test_the_point_estimates_are_upstream_and_unaffected(self):
        """Blast radius: the pool is computed after the counterfactual.

        The reference pool cannot reach the effect series, so the replication of
        the paper's point estimates is not evidence that inference is sound --
        which is the reason the incident went unnoticed.
        """
        res = _fit(_null_panel(0))
        assert res.effects.att is not None
        assert np.isfinite(res.effects.att)
        assert np.isfinite(np.asarray(res.time_series.estimated_gap)).all()

    def test_the_bandwidth_moves_the_block_selection_too(self):
        """The fault is not confined to inference: the donor set moves with it."""
        df = _null_panel(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = prepare_shc_inputs(df, outcome="y", treat="treated",
                                        unitid="unit", time="time", m=_M)
            counts = []
            for h in (None, 1.0, 2.0):
                design = solve_shc(inputs,
                                   bandwidth_grid=None if h is None else [h])
                counts.append(sum(1 for v in design.block_weights.values()
                                  if v > 1e-8))
        assert len(set(counts)) > 1, (
            f"expected the weighted-block count to depend on the bandwidth, "
            f"saw {counts}")


# =========================================================================== #
# rung 2 -- which step produced the pool
# =========================================================================== #
class TestRung2TheStep:

    def test_loocv_collapses_the_bandwidth_when_the_errors_are_correlated(self):
        """Cause 2, and the assumption it violates.

        The selector minimises leave-one-out prediction error. Under the
        paper's Assumption 1 -- iid errors -- a neighbour carries information
        about the trend and none about the error, so a tight fit predicts badly
        and the criterion smooths: on the paper's own simulation it picks 2.5
        from a 50-point grid. When the errors are serially correlated a
        neighbour also predicts the error, a tight fit wins, and the criterion
        collapses to the smallest bandwidth on offer.

        Real four-quarter GDP growth is an overlapping annual growth rate,
        hence a moving average by construction, so it violates Assumption 1 by
        the definition of the outcome variable -- and LOOCV returns 0.30, the
        third grid point.
        """
        grid = np.linspace(0.1, 5.0, 50)
        rng = np.random.default_rng(0)
        T = 260
        t = np.arange(T, dtype=float)

        iid = np.sin(t / 21.0) * 2.0 + rng.normal(0, 0.7, T)
        ar = np.zeros(T)
        for i in range(1, T):
            ar[i] = 0.86 * ar[i - 1] + rng.normal(0, 1.0)
        ma = np.convolve(rng.normal(0, 1, T + 3), np.ones(4) / 4.0, mode="valid")

        _h_iid, cv_iid = loocv_bandwidth(iid, grid)
        _h_ar, cv_ar = loocv_bandwidth(ar, grid)
        _h_ma, cv_ma = loocv_bandwidth(ma, grid)

        assert int(np.argmin(cv_iid)) > 25, (
            "with iid errors the criterion should smooth, not tighten")
        assert int(np.argmin(cv_ar)) < 5, (
            "with AR(1) errors it should collapse toward the small boundary")
        assert int(np.argmin(cv_ma)) < 10, (
            "an overlapping moving average is the case real GDP growth is in; "
            "it lands at grid index 6 of 50 against the iid case's 49")

    def test_a_collapsed_bandwidth_makes_the_smoother_interpolate(self):
        """The step from a small bandwidth to a degenerate pool.

        ``smooth`` has no leave-out, so the kernel puts essentially all weight
        on the point itself once the bandwidth is small against the unit time
        spacing. The residual pool then carries almost no dispersion, which is
        what takes the reported critical value to 0.009.
        """
        rng = np.random.default_rng(1)
        T = 260
        ar = np.zeros(T)
        for i in range(1, T):
            ar[i] = 0.86 * ar[i - 1] + rng.normal(0, 1.0)

        tight = ar - smooth(ar, 0.3)
        loose = ar - smooth(ar, 2.0)
        assert tight.std(ddof=1) < 0.02 * ar.std(ddof=1), (
            f"expected near-interpolation, residual sd "
            f"{tight.std(ddof=1):.4f} against series sd {ar.std(ddof=1):.4f}")
        assert loose.std(ddof=1) > 20 * tight.std(ddof=1)


# =========================================================================== #
# rung 3 -- the invariant nobody asserted
# =========================================================================== #
class TestRung3Invariant:

    @pytest.mark.xfail(strict=True, reason=(
        "cause 1, on the pool it was diagnosed in: the smoother pool is an "
        "in-sample residual and the statistic an out-of-sample prediction "
        "error, so the post-period residuals are about twice the pool's scale. "
        "With n = 4 summed absolute residuals a factor of two puts the "
        "statistic past the 99th percentile of the null. This stays failing "
        "because it computes the smoother pool directly; the estimator's "
        "default no longer uses it."))
    def test_the_pool_and_the_statistic_have_the_same_scale_under_the_null(self):
        """The permutation argument's precondition, at the tolerance it needs.

        Commensurate to within an order of magnitude is not the requirement.
        Under the null the post-period residuals must be draws from the same
        distribution as the pool, so their mean absolute deviations must agree
        to within sampling error of four observations -- which is wide, but not
        a factor of two systematically in one direction.
        """
        ratios = []
        for seed in range(_REPS):
            df = _null_panel(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = prepare_shc_inputs(df, outcome="y", treat="treated",
                                            unitid="unit", time="time", m=_M)
                design = solve_shc(inputs)
                _a, _ap, obs, cf, _g, _w, _f = summarize_effects(inputs, design)
            pool = inputs.y[:inputs.T0] - np.asarray(design.latent_pre).ravel()
            post = obs[_M:] - cf[_M:]
            ratios.append(np.abs(post).mean() / np.abs(pool).mean())
        median = float(np.median(ratios))
        assert 0.67 < median < 1.5, (
            f"median dispersion ratio {median:.2f} over {_REPS} null panels")

    def test_the_dispersion_gap_is_systematic_and_one_directional(self):
        """The measured version, which is what makes it a fault and not noise.

        Out-of-sample error exceeds in-sample error on essentially every
        replication. A ratio scattered around one would be sampling noise; a
        ratio above one every time is the construction.
        """
        ratios = []
        for seed in range(_REPS):
            df = _null_panel(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = prepare_shc_inputs(df, outcome="y", treat="treated",
                                            unitid="unit", time="time", m=_M)
                design = solve_shc(inputs)
                _a, _ap, obs, cf, _g, _w, _f = summarize_effects(inputs, design)
            pool = inputs.y[:inputs.T0] - np.asarray(design.latent_pre).ravel()
            post = obs[_M:] - cf[_M:]
            ratios.append(np.abs(post).mean() / np.abs(pool).mean())
        ratios = np.asarray(ratios)
        assert (ratios > 1.0).mean() >= 0.8, (
            f"expected the out-of-sample residual to dominate; "
            f"it did on {(ratios > 1.0).mean():.0%} of panels")
        assert np.median(ratios) > 1.5


# =========================================================================== #
# rung 4 -- the contract that was never enforced
# =========================================================================== #
class TestRung4Contract:

    def test_a_degenerate_pool_is_not_refused(self):
        """Nothing stops a pool with no dispersion from producing a p-value.

        A reference distribution concentrated at zero rejects every non-zero
        statistic. The routine reports ``p = 0`` and a critical value of
        essentially zero instead of declining to calibrate.
        """
        out = shc_conformal_test(np.full(200, 1e-9), np.array([1.0, -2.0, 3.0, 1.0]),
                                 num_resamples=200, random_state=0)
        assert out["p_value"] == 0.0
        assert out["critical_values"][0.01] < 1e-6
        assert out["reject"][0.01] is True

    def test_an_empty_pool_is_refused(self):
        """The one degenerate case that is caught."""
        from mlsynth.exceptions import MlsynthDataError
        with pytest.raises(MlsynthDataError):
            shc_conformal_test(np.array([]), np.array([1.0]))

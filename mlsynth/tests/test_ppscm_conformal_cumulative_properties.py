"""Metamorphic invariants of PPSCM's per-unit cumulative conformal bands.

The unit suite pins the bands on fixed panels. These assert what must hold for every
panel the estimator accepts:

* structure -- one band per treated cohort, symmetric about its point, and the point
  is the unit's own summed effect rather than an average or a pooled quantity;
* equivariance -- rescaling the panel rescales points and half-widths together;
* invariance -- relabelling donors leaves the bands alone;
* monotonicity -- a smaller ``alpha`` never narrows a band, and a later first origin
  never adds calibration windows.

The panels are deliberately small: every example runs a pooled QP per origin.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.ppscm_helpers.inference import (
    cumulative_conformal_per_unit, rolling_pooled_block_sums)

_SETTINGS = settings(
    max_examples=8, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@st.composite
def panels(draw):
    """Small staggered-adoption panel with room for a few calibration windows."""
    n_ctrl = draw(st.integers(6, 9))
    n_trt = draw(st.integers(1, 2))
    t0 = draw(st.integers(28, 36))
    horizon = draw(st.integers(2, 4))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    n, T = n_ctrl + n_trt, t0 + horizon
    common = rng.uniform(0.3, 1.3, (n, 2)) @ rng.normal(size=(2, T))
    Xy = rng.normal(scale=2.0, size=n)[:, None] + common + rng.normal(scale=0.3, size=(n, T))
    trt = np.full(n, np.inf)
    trt[:n_trt] = t0
    assume(np.isfinite(Xy).all())
    return Xy, trt, t0, horizon


def _bands(Xy, trt, t0, horizon, alpha=0.1, min_train_frac=0.3):
    return cumulative_conformal_per_unit(
        Xy, trt, t0, horizon, t0, fixedeff=True, time_cohort=False, nu_used=0.5,
        lam=0.0, solver=None, alpha=alpha, horizon=horizon,
        min_train_frac=min_train_frac,
    )


class TestStructure:
    @given(panels())
    @_SETTINGS
    def test_one_symmetric_band_per_treated_cohort(self, panel):
        Xy, trt, t0, horizon = panel
        point, lower, upper, n_scores = _bands(Xy, trt, t0, horizon)
        n_trt = int(np.isfinite(trt).sum())
        assert point.shape == lower.shape == upper.shape == n_scores.shape
        assert 1 <= point.size <= n_trt
        np.testing.assert_allclose(point - lower, upper - point, rtol=1e-9, atol=1e-9)

    @given(panels())
    @_SETTINGS
    def test_each_point_is_that_unit_own_summed_effect(self, panel):
        from mlsynth.utils.ppscm_helpers.engine import run_multisynth
        Xy, trt, t0, horizon = panel
        full = run_multisynth(Xy, trt, t0, horizon, t0, fixedeff=True,
                              time_cohort=False, nu=0.5, lam=0.0, solver=None)
        expected = [float(np.sum(np.asarray(full["tau_rel"][k], float)[:horizon]))
                    for k in range(len(full["groups"]))]
        point, *_ = _bands(Xy, trt, t0, horizon)
        np.testing.assert_allclose(point, expected, rtol=1e-6, atol=1e-8)


class TestEquivariance:
    @given(panels(), st.sampled_from([0.5, 2.0, 3.0]))
    @_SETTINGS
    def test_rescaling_the_panel_rescales_points_and_widths(self, panel, c):
        Xy, trt, t0, horizon = panel
        p0, lo0, hi0, _ = _bands(Xy, trt, t0, horizon)
        p1, lo1, hi1, _ = _bands(c * Xy, trt, t0, horizon)
        np.testing.assert_allclose(p1, c * p0, rtol=1e-5, atol=1e-7)
        finite = np.isfinite(hi0 - lo0)
        if finite.any():
            np.testing.assert_allclose((hi1 - lo1)[finite], c * (hi0 - lo0)[finite],
                                       rtol=1e-5, atol=1e-7)


class TestInvariance:
    @given(panels())
    @_SETTINGS
    def test_relabelling_donors_leaves_the_bands_alone(self, panel):
        Xy, trt, t0, horizon = panel
        donors = np.where(~np.isfinite(trt))[0]
        treated = np.where(np.isfinite(trt))[0]
        perm = np.concatenate([treated, np.random.default_rng(0).permutation(donors)])
        p0, lo0, hi0, _ = _bands(Xy, trt, t0, horizon)
        p1, lo1, hi1, _ = _bands(Xy[perm], trt[perm], t0, horizon)
        np.testing.assert_allclose(p1, p0, rtol=1e-5, atol=1e-7)
        finite = np.isfinite(lo0)
        if finite.any():
            np.testing.assert_allclose(lo1[finite], lo0[finite], rtol=1e-5, atol=1e-6)


class TestMonotonicity:
    @given(panels())
    @_SETTINGS
    def test_smaller_alpha_never_narrows_a_band(self, panel):
        Xy, trt, t0, horizon = panel
        _, lo_t, hi_t, _ = _bands(Xy, trt, t0, horizon, alpha=0.05)
        _, lo_l, hi_l, _ = _bands(Xy, trt, t0, horizon, alpha=0.30)
        assert ((hi_t - lo_t) >= (hi_l - lo_l) - 1e-9).all()

    @given(panels())
    @_SETTINGS
    def test_a_later_first_origin_never_adds_windows(self, panel):
        Xy, trt, t0, horizon = panel
        early = rolling_pooled_block_sums(
            Xy, trt, t0, horizon, t0, fixedeff=True, time_cohort=False, nu_used=0.5,
            lam=0.0, solver=None, horizon=horizon, min_train_frac=0.2)
        late = rolling_pooled_block_sums(
            Xy, trt, t0, horizon, t0, fixedeff=True, time_cohort=False, nu_used=0.5,
            lam=0.0, solver=None, horizon=horizon, min_train_frac=0.7)
        for k in range(min(len(early), len(late))):
            assert late[k].size <= early[k].size

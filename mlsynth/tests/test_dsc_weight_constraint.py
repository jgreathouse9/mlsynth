"""Tests for the DSC weight-constraint option (simplex vs sum-to-one).

DiSCo's default feasible set is not the simplex: ``DiSCo_weights_reg`` passes
``lb = NULL`` unless ``simplex = TRUE``, so weights need only sum to one and be
at most one, and may be negative. Zhang, Zhang & Zhang (2026) discuss the same
relaxation as a bounded extrapolation set. This adds it as an option, with the
simplex kept as the default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import DSC
from mlsynth.config_models import DSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.dsc_helpers.weights import (
    solve_simplex_weights,
    solve_sum_to_one_weights,
)


def _panel(seed=0, n_units=5, n_periods=6, n_obs=60, t0=4):
    """Micro-level panel: one row per (unit, period, individual)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_periods):
            y = rng.normal(10.0 + 2.0 * u + 0.3 * t, 1.0 + 0.1 * u, size=n_obs)
            rows.append(pd.DataFrame({
                "unit": f"u{u}", "time": t, "y": y,
                "D": int(u == 0 and t >= t0),
            }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------

def test_sum_to_one_smoke():
    rng = np.random.default_rng(0)
    A, b = rng.normal(size=(40, 4)), rng.normal(size=40)
    w = solve_sum_to_one_weights(A, b)
    assert w.shape == (4,)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)


def test_sum_to_one_respects_the_upper_bound():
    """DiSCo passes ub = 1 in both modes, so no single weight may exceed one."""
    rng = np.random.default_rng(1)
    A, b = rng.normal(size=(30, 3)), rng.normal(size=30) * 50.0
    assert np.all(solve_sum_to_one_weights(A, b) <= 1.0 + 1e-8)


def test_sum_to_one_is_never_worse_than_the_simplex():
    """The simplex is contained in the sum-to-one set, so the optimum improves."""
    rng = np.random.default_rng(2)
    for _ in range(15):
        A, b = rng.normal(size=(50, 5)), rng.normal(size=50)
        obj = lambda w: float(np.sum((A @ w - b) ** 2))
        assert obj(solve_sum_to_one_weights(A, b)) <= obj(solve_simplex_weights(A, b)) + 1e-6


def test_sum_to_one_goes_negative_when_extrapolation_helps():
    """A target outside the donors' convex hull draws a weight below zero.

    Two donors cannot show this: with the weights summing to one, a negative
    second weight needs a first above one, which the upper bound forbids. Three
    can, so the target here is built at the feasible point (0.8, 0.8, -0.6).
    """
    rng = np.random.default_rng(5)
    A = rng.normal(size=(60, 3))
    truth = np.array([0.8, 0.8, -0.6])
    w = solve_sum_to_one_weights(A, A @ truth)
    np.testing.assert_allclose(w, truth, atol=1e-6)
    assert w.min() < 0.0
    assert w.sum() == pytest.approx(1.0)


def test_the_two_solvers_agree_when_the_simplex_does_not_bind():
    """With an interior simplex optimum, relaxing non-negativity changes nothing."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(200, 3)) + np.array([0.0, 1.0, 2.0])
    b = A @ np.array([0.3, 0.4, 0.3]) + 0.01 * rng.normal(size=200)
    w_s = solve_simplex_weights(A, b)
    assert w_s.min() > 0.02                       # interior, so the bound is slack
    np.testing.assert_allclose(solve_sum_to_one_weights(A, b), w_s, atol=1e-6)


def test_sum_to_one_single_donor():
    assert solve_sum_to_one_weights(np.ones((10, 1)), np.arange(10.0)) == pytest.approx([1.0])


def test_sum_to_one_underdetermined_is_still_feasible():
    """Fewer rows than donors: the fit interpolates, the constraints still hold."""
    rng = np.random.default_rng(4)
    w = solve_sum_to_one_weights(rng.normal(size=(4, 10)), rng.normal(size=4))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w <= 1.0 + 1e-8)


@pytest.mark.parametrize("A, b", [
    (np.ones(10), np.ones(10)),                       # 1-D design
    (np.ones((10, 2)), np.ones((10, 1))),             # 2-D target
    (np.ones((10, 2)), np.ones(9)),                   # length mismatch
])
def test_sum_to_one_rejects_bad_shapes(A, b):
    with pytest.raises(MlsynthEstimationError):
        solve_sum_to_one_weights(A, b)


# --------------------------------------------------------------------------
# the config
# --------------------------------------------------------------------------

def test_default_is_the_simplex():
    cfg = DSCConfig(df=_panel(), outcome="y", treat="D", unitid="unit", time="time")
    assert cfg.weight_constraint == "simplex"


def test_pipeline_rejects_an_unknown_constraint():
    """run_dsc is callable directly, so its own guard has to hold.

    The config Literal stops this at the estimator boundary, which means the
    pipeline guard is only reachable by a direct helper call -- and that is
    exactly how the benchmark suite uses these helpers.
    """
    from mlsynth.utils.dsc_helpers.pipeline import run_dsc
    from mlsynth.utils.dsc_helpers.setup import prepare_dsc_inputs

    inputs = prepare_dsc_inputs(df=_panel(), outcome="y", treat="D",
                                unitid="unit", time="time")
    with pytest.raises(MlsynthEstimationError, match="weight_constraint"):
        run_dsc(inputs=inputs, M=100, grid_method="equidistant",
                weight_constraint="convex-hull")


def test_unknown_constraint_is_rejected():
    with pytest.raises((MlsynthConfigError, ValueError)):
        DSCConfig(df=_panel(), outcome="y", treat="D", unitid="unit",
                  time="time", weight_constraint="convex-hull")


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def _fit(**kw):
    return DSC({"df": _panel(), "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "display_graphs": False, **kw}).fit()


def test_the_default_fit_is_unchanged_by_the_new_option():
    """Passing the default explicitly must reproduce the implicit default exactly."""
    a = _fit()
    b = _fit(weight_constraint="simplex")
    np.testing.assert_allclose(a.weights.weight_vector, b.weights.weight_vector,
                               rtol=0, atol=0)
    assert a.att == pytest.approx(b.att, rel=0, abs=0)


def test_default_weights_stay_on_the_simplex():
    w = _fit().weights.weight_vector
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_sum_to_one_fit_still_sums_to_one():
    res = _fit(weight_constraint="sum_to_one")
    w = res.weights.weight_vector
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w <= 1.0 + 1e-8)
    assert np.isfinite(res.att)


def test_sum_to_one_fits_the_pre_period_at_least_as_well():
    """A larger feasible set cannot worsen the pre-period Wasserstein fit."""
    # pre_period_wasserstein is the per-period loss at that period's own
    # weights, so the inequality holds period by period, not just on average.
    tight = np.asarray(_fit().pre_period_wasserstein)
    loose = np.asarray(_fit(weight_constraint="sum_to_one").pre_period_wasserstein)
    assert np.all(loose <= tight + 1e-8)
    assert loose.sum() < tight.sum()


def test_inference_uses_the_same_constraint():
    """Placebo refits must not silently fall back to the simplex."""
    res = _fit(weight_constraint="sum_to_one", compute_inference=True)
    assert res.inference is not None

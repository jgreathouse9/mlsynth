"""Coverage tests for mlsynth.utils.ctsc_helpers.simulation."""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.ctsc_helpers.simulation import (
    SimulationSummary,
    _lambda1,
    _lambda2,
    generate_model,
    run_simulation,
    twoway_fe_effect,
)


def test_lambda_piecewise_branches():
    # _lambda1 three branches
    assert _lambda1(10) == pytest.approx(1.0)   # t < 20
    assert _lambda1(25) == pytest.approx(2.0)   # 20 <= t < 30
    assert _lambda1(60) == pytest.approx(3.0)   # t >= 30
    # _lambda2 four branches
    assert _lambda2(3) == pytest.approx(2.0)    # t <= 5
    assert _lambda2(10) == pytest.approx(-4.0)  # 5 < t < 20
    assert _lambda2(30) == pytest.approx(6.0)   # 20 <= t < 40
    assert _lambda2(50) == pytest.approx(1.0)   # t >= 40


@pytest.mark.parametrize("model,dims", [(1, (10, 50)), (2, (30, 50)),
                                        (3, (10, 50)), (4, (10, 20))])
def test_generate_model_shapes(model, dims):
    rng = np.random.default_rng(0)
    Y, D, ae = generate_model(model, rng)
    n, T = dims
    assert Y.shape == (n, T)
    assert D.shape == (n, T, 1)
    assert ae == 0.0


def test_generate_model_invalid_raises():
    with pytest.raises(ValueError, match="model must be in"):
        generate_model(5, np.random.default_rng(0))


def test_generate_model_determinism():
    Y1, D1, _ = generate_model(1, np.random.default_rng(3))
    Y2, D2, _ = generate_model(1, np.random.default_rng(3))
    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(D1, D2)


def test_twoway_fe_effect_returns_float():
    rng = np.random.default_rng(1)
    Y, D, _ = generate_model(1, rng)
    val = twoway_fe_effect(Y, D)
    assert isinstance(val, float)


def test_twoway_fe_effect_zero_denominator():
    # Constant treatment -> demeaned Dw is all zeros -> nan branch.
    Y = np.ones((3, 4))
    D = np.ones((3, 4, 1))
    assert np.isnan(twoway_fe_effect(Y, D))


def test_run_simulation_smallest_config():
    summ = run_simulation(4, n_sims=2, seed=0)
    assert isinstance(summ, SimulationSummary)
    assert summ.model == 4
    assert summ.n_sims == 2
    for field in (summ.ctsc_mean_bias, summ.ctsc_mad, summ.ctsc_rmse,
                  summ.fe_mean_bias, summ.fe_mad, summ.fe_rmse):
        assert isinstance(field, float)

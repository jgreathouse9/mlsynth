"""Metamorphic and algebraic properties of the mlSC estimator.

Reference: Bottmer, L., "Synthetic Control with Disaggregated Data" (JMP,
Stanford), Equation 5.2 and the discussion following it.

Two families, neither of which needs an oracle:

* C4 -- metamorphic relations on ``MLSC.fit()``. The paper states that the
  ``sigma_y^2`` factor "ensures the penalty is on the same scale as the loss
  function while the penalty parameter lambda is scale invariant", which is a
  claim about how the output must respond to rescaling the outcome. Location
  and row-order invariance follow from the simplex constraint and from the
  ingestion contract respectively.
* C5 -- algebra of the block penalty matrix. ``build_block`` returns
  ``Q = I - v 1' - 1 v' + (v'v) J``, so for any ``v`` on the simplex it is
  symmetric, positive semi-definite, annihilates ``v``, and has rank
  ``C - 1``. `test_mlsc.py` asserts the first three at a single hand-picked
  ``v = [0.3, 0.2, 0.5]``; asserting them over the domain is what makes them
  claims about the estimator instead of claims about one vector.

The C5 sweeps are deterministic draws, not generated inputs. Adding
``hypothesis`` is a dependency decision (Phase 1 of the plan in
``agents/agents_tests.md``) and belongs on its own branch; each sweep here is
written as a single-draw property function so the conversion is a decorator
swap.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from mlsynth import MLSC
from mlsynth.utils.mlsc_helpers.penalty import (
    build_block,
    build_penalty_matrix,
    build_sqrt_factor,
)

_TOL = 1e-10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panel(seed=0, S=5, C=4, T=24, T0=18) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    Y_sct = rng.standard_normal((S, C, T)) + rng.standard_normal((S, 1, 1)) * 2.0
    pops = rng.uniform(0.5, 2.0, size=(S, C))
    v = pops / pops.sum(axis=1, keepdims=True)
    Y_st = np.einsum("sc,sct->st", v, Y_sct)
    states = [f"state_{s}" for s in range(S)]
    agg = pd.DataFrame([
        {"state": states[s], "year": 2000 + t, "y": float(Y_st[s, t]),
         "treated": int(s == 0 and t >= T0)}
        for s in range(S) for t in range(T)
    ])
    disagg = pd.DataFrame([
        {"county": f"county_{s}_{c}", "state": states[s], "year": 2000 + t,
         "y": float(Y_sct[s, c, t]), "treated": int(s == 0 and t >= T0),
         "pop": float(pops[s, c])}
        for s in range(S) for c in range(C) for t in range(T)
    ])
    return agg, disagg, T0


def _config(df_agg, df_disagg, **overrides):
    cfg = dict(
        df_agg=df_agg, df_disagg=df_disagg, outcome="y", time="year",
        treat="treated", unitid_agg="state", unitid_disagg="county",
        agg_id="state", weight_col="pop", lambda_est="heuristic",
        display_graphs=False,
    )
    cfg.update(overrides)
    return cfg


def _simplex(rng: np.random.Generator, size: int) -> np.ndarray:
    """A draw from the interior of the simplex in ``size`` dimensions."""
    x = rng.gamma(shape=1.0, size=size)
    return x / x.sum()


# ===========================================================================
# C4 -- metamorphic relations on the estimator
# ===========================================================================

@pytest.mark.parametrize("scale", [0.01, 3.0, 250.0])
def test_outcome_rescaling_leaves_the_penalty_parameter_and_weights_fixed(scale):
    """The sigma_y^2 claim, stated as a response to rescaling.

    Multiplying every outcome by ``c`` multiplies the loss by ``c^2`` and,
    because the penalty carries ``sigma_y^2``, multiplies the penalty term by
    ``c^2`` as well. The minimiser is therefore unchanged and the selected
    penalty is scale free, while everything measured in outcome units moves by
    ``c``. Drop ``sigma_y^2`` from the penalty and this breaks.
    """
    agg, disagg, _ = _panel()
    scaled_agg, scaled_disagg = agg.copy(), disagg.copy()
    scaled_agg["y"] *= scale
    scaled_disagg["y"] *= scale

    base = MLSC(_config(agg, disagg)).fit()
    other = MLSC(_config(scaled_agg, scaled_disagg)).fit()

    # Scale free
    np.testing.assert_allclose(other.design.omega, base.design.omega, atol=1e-6)
    np.testing.assert_allclose(other.design.aggregate_weights,
                               base.design.aggregate_weights, atol=1e-6)
    assert other.design.lambda_used == pytest.approx(base.design.lambda_used, rel=1e-8)

    # Carried in outcome units
    assert other.att == pytest.approx(base.att * scale, rel=1e-6)
    assert other.pre_rmse == pytest.approx(base.pre_rmse * scale, rel=1e-6)
    np.testing.assert_allclose(other.paths.counterfactual,
                               base.paths.counterfactual * scale, rtol=1e-6)

    # Variance components are second moments of the outcome
    assert other.design.sigma_y2 == pytest.approx(base.design.sigma_y2 * scale**2, rel=1e-6)
    assert other.design.sigma_eps2 == pytest.approx(base.design.sigma_eps2 * scale**2, rel=1e-6)


@pytest.mark.parametrize("shift", [-40.0, 7.5])
def test_outcome_shifting_leaves_the_weights_and_the_effect_fixed(shift):
    """Adding a constant to every outcome changes nothing that matters.

    The simplex constraint makes the loss shift-invariant: with ``sum omega =
    1`` the constant cancels out of ``Y - X omega``. So the weights, the
    penalty parameter and the ATT are all unchanged, and only the level of the
    counterfactual moves.
    """
    agg, disagg, _ = _panel()
    shifted_agg, shifted_disagg = agg.copy(), disagg.copy()
    shifted_agg["y"] += shift
    shifted_disagg["y"] += shift

    base = MLSC(_config(agg, disagg)).fit()
    other = MLSC(_config(shifted_agg, shifted_disagg)).fit()

    np.testing.assert_allclose(other.design.omega, base.design.omega, atol=1e-6)
    assert other.design.lambda_used == pytest.approx(base.design.lambda_used, rel=1e-8)
    assert other.att == pytest.approx(base.att, abs=1e-6)
    assert other.pre_rmse == pytest.approx(base.pre_rmse, abs=1e-6)
    np.testing.assert_allclose(other.paths.counterfactual,
                               base.paths.counterfactual + shift, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_row_order_of_the_input_frames_is_irrelevant(seed):
    """Ingestion goes through ``dataprep``, so the long frames are sets of
    rows and shuffling them cannot move a single number."""
    agg, disagg, _ = _panel()
    rng = np.random.default_rng(seed)
    shuffled_agg = agg.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    shuffled_disagg = disagg.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))

    base = MLSC(_config(agg, disagg)).fit()
    other = MLSC(_config(shuffled_agg, shuffled_disagg)).fit()

    np.testing.assert_allclose(other.design.omega, base.design.omega, atol=1e-8)
    assert other.att == pytest.approx(base.att, abs=1e-8)
    assert other.donor_weights == pytest.approx(base.donor_weights, abs=1e-8)


# ===========================================================================
# C5 -- algebra of the block penalty matrix
# ===========================================================================

def _check_block_properties(v: np.ndarray) -> None:
    """Every claim `build_block` makes, for one draw of ``v``."""
    C = v.shape[0]
    Q = build_block(v)

    assert Q.shape == (C, C)
    np.testing.assert_allclose(Q, Q.T, atol=_TOL)                    # symmetric
    assert np.linalg.eigvalsh(Q).min() >= -_TOL                      # PSD
    np.testing.assert_allclose(Q @ v, 0.0, atol=1e-12)               # v in kernel

    # The kernel is exactly span{v}: R = I - v 1' drops one dimension, so a
    # weight vector is unpenalised only when it is proportional to population.
    expected_rank = max(C - 1, 0)
    assert np.linalg.matrix_rank(Q, tol=1e-9) == expected_rank

    # Stated as the estimator reads it: the penalty vanishes on proportional
    # weights and is strictly positive off that ray. The probe has to be built
    # by projecting v out, not picked as a basis vector -- at a corner of the
    # simplex such as v = [1, 0] the basis vector *is* v and the assertion
    # would be testing the kernel it meant to avoid.
    for c in (0.3, 1.0, 7.0):
        assert abs(float(c * v @ Q @ (c * v))) < 1e-12
    if C >= 2:
        probe = np.arange(1.0, C + 1.0)
        probe = probe - (probe @ v / (v @ v)) * v          # orthogonal to v
        if np.linalg.norm(probe) > 1e-8:
            assert float(probe @ Q @ probe) > 1e-9


@pytest.mark.parametrize("size", [1, 2, 3, 5, 12, 40])
def test_block_properties_across_block_sizes(size):
    rng = np.random.default_rng(1000 + size)
    for _ in range(25):
        _check_block_properties(_simplex(rng, size))


def test_block_properties_on_degenerate_population_shares():
    """Boundary of the simplex: a share at zero, and all mass on one unit."""
    for v in (
        np.array([1.0]),                       # a block with one disaggregate
        np.array([1.0, 0.0]),                  # a county with no population
        np.array([0.0, 0.0, 1.0]),             # all mass on one county
        np.array([1e-12, 1.0 - 1e-12]),        # near-degenerate
        np.full(6, 1.0 / 6.0),                 # exactly uniform
    ):
        _check_block_properties(v)


@pytest.mark.parametrize("size", [2, 4, 9])
def test_uniform_shares_give_the_centering_matrix(size):
    """`v = 1/C` collapses Q to `I - J/C`, the centering matrix."""
    Q = build_block(np.full(size, 1.0 / size))
    np.testing.assert_allclose(Q, np.eye(size) - np.full((size, size), 1.0 / size),
                               atol=_TOL)


@pytest.mark.parametrize("seed", range(6))
def test_assembly_is_block_diagonal_for_arbitrary_block_structures(seed):
    """`build_penalty_matrix` glues per-block matrices with nothing across."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, size=int(rng.integers(2, 6)))
    v = np.concatenate([_simplex(rng, int(n)) for n in sizes])
    idx = np.repeat(np.arange(len(sizes)), sizes)

    Q = build_penalty_matrix(v, idx)
    assert Q.shape == (v.size, v.size)

    start = 0
    for block, n in enumerate(sizes):
        stop = start + int(n)
        np.testing.assert_allclose(Q[start:stop, start:stop],
                                   build_block(v[start:stop]), atol=_TOL)
        off_block = idx[None, :] != idx[:, None]
        np.testing.assert_allclose(Q[off_block], 0.0, atol=_TOL)
        start = stop

    # The full penalty still annihilates the population vector blockwise.
    np.testing.assert_allclose(Q @ v, 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_square_root_factor_reproduces_the_penalty(seed):
    """`R' R == Q` over arbitrary block structures, not one fixture.

    The solver augments the design with ``sqrt(lambda) R`` instead of forming
    ``Q``, so the two representations agreeing is what makes the augmented
    least-squares path equal the penalised program.
    """
    rng = np.random.default_rng(100 + seed)
    sizes = rng.integers(1, 7, size=int(rng.integers(2, 5)))
    v = np.concatenate([_simplex(rng, int(n)) for n in sizes])
    idx = np.repeat(np.arange(len(sizes)), sizes)

    R = build_sqrt_factor(v, idx)
    np.testing.assert_allclose(R.T @ R, build_penalty_matrix(v, idx), atol=1e-10)

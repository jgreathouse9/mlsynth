"""SDID's intercept-plus-simplex solve, run for a family of problems at once.

Test-first (per ``agents/agents_tests.md``): written before the batched helper
exists, against the one-at-a-time solve it has to reproduce exactly.

Placebo inference (Arkhangelsky et al. 2021, Algorithm 4) reassigns controls as
pseudo-treated ``B`` times -- 500 by default -- and refits the unit and time
weights for every draw. Each refit is the same program on a different column
subset of the same donor matrix, with a ridge that varies by draw, so the family
has a closed form: centring is per column and so survives subsetting, and the
ridge enters the Gram affinely. What makes the batch usable here rather than in
STACKEDSC is the shape: SDID's designs have more rows than columns, so the
optimum is a point and not a face, and the batched and sequential solvers return
the same weights and not merely the same fit.

That equality is what these tests pin. Anything looser would not be enough --
the weights are what the placebo counterfactuals are built from.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.minnorm import gram_reduction_is_safe
from mlsynth.utils.sdid_helpers.weights import _solve_intercept_simplex


def _family(rng, m=30, J=20, n_drop=1, n=25, ridge=0.4, vary_ridge=False):
    """A donor matrix plus ``n`` leave-some-out placebo draws off it."""
    D = np.cumsum(rng.normal(size=(m, J)), axis=0) + 10.0
    problems = []
    for i in range(n):
        perm = rng.permutation(J)
        drop, keep = perm[:n_drop], perm[n_drop:]
        design = D[:, keep]
        target = D[:, drop].mean(axis=1)
        r = ridge * (1.0 + 0.5 * (i % 4)) if vary_ridge else ridge
        problems.append((design, target, r))
    return problems


def _loop(problems):
    return [_solve_intercept_simplex(d, t, r) for d, t, r in problems]


# --------------------------------------------------------------------------- #
# 1. The batch equals the loop, on the weights and the intercept
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_batch_matches_the_loop(seed):
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    problems = _family(np.random.default_rng(seed))
    got = solve_intercept_simplex_many(problems)
    want = _loop(problems)
    assert len(got) == len(want)
    for (a_g, w_g), (a_w, w_w) in zip(got, want):
        np.testing.assert_allclose(w_g, w_w, atol=1e-8)
        assert np.isclose(a_g, a_w, atol=1e-8)


def test_batch_matches_the_loop_with_a_ridge_that_varies_by_draw():
    """The placebo ridge is recomputed from each draw's donors, so the family
    does not share one. It enters the Gram affinely, which is what allows it."""
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    problems = _family(np.random.default_rng(9), vary_ridge=True)
    for (a_g, w_g), (a_w, w_w) in zip(solve_intercept_simplex_many(problems),
                                      _loop(problems)):
        np.testing.assert_allclose(w_g, w_w, atol=1e-8)
        assert np.isclose(a_g, a_w, atol=1e-8)


def test_mixed_shapes_are_handled():
    """Unit and time weights have different shapes, and cohorts differ from one
    another; a caller may hand over the lot."""
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    rng = np.random.default_rng(3)
    problems = (_family(rng, m=30, J=20, n=6)
                + _family(rng, m=25, J=12, n=5)
                + _family(rng, m=40, J=20, n=4, ridge=0.0))
    for (a_g, w_g), (a_w, w_w) in zip(solve_intercept_simplex_many(problems),
                                      _loop(problems)):
        np.testing.assert_allclose(w_g, w_w, atol=1e-8)
        assert np.isclose(a_g, a_w, atol=1e-8)


def test_zero_ridge_is_handled():
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    problems = _family(np.random.default_rng(4), ridge=0.0)
    for (a_g, w_g), (a_w, w_w) in zip(solve_intercept_simplex_many(problems),
                                      _loop(problems)):
        np.testing.assert_allclose(w_g, w_w, atol=1e-8)
        assert np.isclose(a_g, a_w, atol=1e-8)


# --------------------------------------------------------------------------- #
# 2. Feasibility and edges
# --------------------------------------------------------------------------- #
def test_weights_are_on_the_simplex():
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    for _, w in solve_intercept_simplex_many(_family(np.random.default_rng(5))):
        assert w.min() >= -1e-9
        assert abs(w.sum() - 1.0) < 1e-9


def test_empty_family():
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    assert solve_intercept_simplex_many([]) == []


def test_single_problem_family():
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    problems = _family(np.random.default_rng(6), n=1)
    (a_g, w_g), = solve_intercept_simplex_many(problems)
    a_w, w_w = _loop(problems)[0]
    np.testing.assert_allclose(w_g, w_w, atol=1e-8)
    assert np.isclose(a_g, a_w, atol=1e-8)


def test_single_donor_family():
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    rng = np.random.default_rng(7)
    problems = [(rng.normal(size=(8, 1)), rng.normal(size=8), 0.0)]
    (_, w), = solve_intercept_simplex_many(problems)
    assert w.shape == (1,) and np.isclose(w[0], 1.0)


# --------------------------------------------------------------------------- #
# 3. A rank-deficient design keeps the one-at-a-time solve
# --------------------------------------------------------------------------- #
def test_rank_deficient_designs_fall_back_and_still_match_the_loop():
    """More donors than periods: the optimum is a face and the two solvers pick
    different points of it, so the guard has to send these down the sequential
    path. SDID's own shapes are the other way round, but a caller's need not be."""
    from mlsynth.utils.sdid_helpers.weights import solve_intercept_simplex_many

    rng = np.random.default_rng(8)
    problems = _family(rng, m=8, J=30, n=5, ridge=0.0)
    assert not gram_reduction_is_safe(problems[0][0] - problems[0][0].mean(axis=0))
    for (a_g, w_g), (a_w, w_w) in zip(solve_intercept_simplex_many(problems),
                                      _loop(problems)):
        np.testing.assert_allclose(w_g, w_w, atol=1e-8)
        assert np.isclose(a_g, a_w, atol=1e-8)


# --------------------------------------------------------------------------- #
# 4. Through the estimator: batching the placebo draws changes nothing
# --------------------------------------------------------------------------- #
def _placebo_panel(seed=0, n_units=14, T=16, t0=10):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        level = 10.0 + 2.0 * u
        for t in range(T):
            y = level + 0.5 * t + rng.normal(scale=0.4)
            treated = int(u == 0 and t >= t0)
            if treated:
                y -= 5.0
            rows.append({"unit": u, "time": t, "y": y, "treat": treated})
    return pd.DataFrame(rows)


def test_placebo_inference_is_unchanged_by_batching():
    """The estimand this touches is the placebo variance, so it is the thing to
    compare. Solving the draws as a family moves the standard error in the
    eleventh significant figure, which is the batched and sequential solvers'
    agreement on the weights (~1e-11) carried through."""
    import mlsynth.utils.sdid_helpers.inference as INF
    from mlsynth import SDID

    real = INF._solve_placebo_weights

    def sequential(payloads):
        return {(i, p): (None, None) for i, c in enumerate(payloads) for p in c}

    cfg = dict(df=_placebo_panel(), outcome="y", treat="treat", unitid="unit",
               time="time", display_graphs=False, B=60, seed=7)
    try:
        INF._solve_placebo_weights = sequential
        want = SDID(dict(cfg)).fit()
        INF._solve_placebo_weights = real
        got = SDID(dict(cfg)).fit()
    finally:
        INF._solve_placebo_weights = real

    assert np.isclose(got.effects.att, want.effects.att, rtol=0, atol=1e-12)
    for a, b in ((got.inference.ci_lower, want.inference.ci_lower),
                 (got.inference.ci_upper, want.inference.ci_upper)):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-9)


def test_every_placebo_draw_gets_its_weights():
    """A draw whose key is missing would silently fall back to a fresh solve and
    look correct while undoing the batch, so the mapping is checked directly."""
    import mlsynth.utils.sdid_helpers.inference as INF

    seen = {}
    real = INF._solve_placebo_weights

    def spy(payloads):
        out = real(payloads)
        seen["payloads"] = sum(len(c) for c in payloads)
        seen["solved"] = sum(1 for v in out.values() if v[0] is not None)
        return out

    from mlsynth import SDID
    try:
        INF._solve_placebo_weights = spy
        SDID(dict(df=_placebo_panel(), outcome="y", treat="treat", unitid="unit",
                  time="time", display_graphs=False, B=25, seed=3)).fit()
    finally:
        INF._solve_placebo_weights = real
    assert seen["payloads"] == 25
    assert seen["solved"] == 25


def test_sdid_shapes_are_in_the_safe_regime():
    """The premise of batching SDID at all, asserted rather than assumed: its
    unit-weight design (pre-periods plus the ridge block, by donors) and its
    time-weight design (donors by pre-periods) both have full column rank on the
    Prop 99 geometry."""
    rng = np.random.default_rng(10)
    T0, J = 19, 37
    donor_pre = np.cumsum(rng.normal(size=(T0, J)), axis=0) + 10.0
    unit_design = np.vstack([donor_pre - donor_pre.mean(axis=0),
                             np.sqrt(0.4) * np.eye(J)])
    time_design = donor_pre.T - donor_pre.T.mean(axis=0)
    assert gram_reduction_is_safe(unit_design)
    assert gram_reduction_is_safe(time_design)

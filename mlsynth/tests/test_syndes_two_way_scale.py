"""The two-way exact backend does not depend on the outcome's units or level (#802).

The two-way objective reads ``Y`` only through contrasts ``Y u`` whose weights
sum to zero, so scaling ``Y`` by ``s`` scales the objective by ``s^2`` and
adding a level common to every unit changes nothing. The design should follow:
the same treated set at every scale and every level.

Three parts of the backend broke that. The swap search stopped on an absolute
improvement of ``1e-12`` while its cut values scale as ``s^-4``, so from
``s = 1e4`` every restart stopped before its first swap. ``G = Y'Y / T`` carried
the level, so ``cond(M)`` grew with it until the engine refused the instance, and
the refusal named the wrong cause. The inner solver stopped on an absolute duality
gap, loose for small units and unreachable for large ones.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers.exact import (
    solve_synthetic_design_exact,
    solve_synthetic_design_exact_pool,
)
from mlsynth.utils.syndes_helpers.gram import (
    BoundEngine,
    TwoWayProblem,
    maxcut_candidates,
)
from mlsynth.utils.syndes_helpers.partition import (
    solve_partition_batch,
    split_indices,
)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _panel(N=12, T=20, rank=3, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _treated(design):
    return tuple(int(i) for i in design.selected_unit_indices)


def _ill_conditioned():
    """Eight units of rank two over four periods: with ``lam`` near zero the
    problem on contrasts is singular to working precision. Its 56 designs of
    three are priced one by one (#821), so the tests below set a candidate limit
    under that count to reach the refusal that remains."""
    rng = np.random.default_rng(9)
    return rng.standard_normal((4, 2)) @ rng.standard_normal((2, 8))


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def test_a_panel_far_from_zero_gets_the_same_design():
    Y = _panel(seed=1)
    base = solve_synthetic_design_exact(Y, 3)
    far = solve_synthetic_design_exact(Y + 1e8 * Y.std(), 3)
    assert _treated(far) == _treated(base)
    assert far.objective_value == pytest.approx(base.objective_value, rel=1e-6)


# --------------------------------------------------------------------------- #
# the Gram matrix the backend works with
# --------------------------------------------------------------------------- #


def test_the_gram_matrix_agrees_with_the_raw_one_on_every_contrast():
    Y = _panel()
    T, N = Y.shape
    problem = TwoWayProblem.from_panel(Y)
    U = np.random.default_rng(1).standard_normal((50, N))
    U -= U.mean(axis=1, keepdims=True)
    raw = Y.T @ Y / T
    np.testing.assert_allclose(np.einsum("bi,ij,bj->b", U, problem.G, U),
                               np.einsum("bi,ij,bj->b", U, raw, U), rtol=1e-10)


def test_a_common_level_leaves_the_gram_matrix_unchanged():
    Y = _panel()
    G = TwoWayProblem.from_panel(Y).G
    shifted = TwoWayProblem.from_panel(Y + 1e4 * Y.std()).G
    np.testing.assert_allclose(shifted, G, rtol=0, atol=1e-9 * np.abs(G).max())


def test_the_penalty_is_the_uncentred_within_unit_variance():
    Y = _panel()
    expected = float(np.mean(np.var(Y, axis=0, ddof=1)))
    assert TwoWayProblem.from_panel(Y).lam == pytest.approx(expected, rel=1e-12)
    assert TwoWayProblem.from_panel(Y + 1e4 * Y.std()).lam == pytest.approx(
        expected, rel=1e-9)


def test_the_level_direction_sits_inside_the_spectrum_designs_see():
    """No design reads the direction ``1``, so its eigenvalue is placed inside
    the spectrum of ``M`` on contrasts and ``cond(M)`` is that problem's own."""
    Y = _panel() + 1e6 * _panel().std()
    problem = TwoWayProblem.from_panel(Y)
    N = problem.N
    basis = np.linalg.qr(np.eye(N) - 1.0 / N)[0][:, : N - 1]
    contrasts = np.linalg.eigvalsh(basis.T @ problem.M @ basis)
    ones = np.ones(N)
    level = float(ones @ problem.M @ ones) / N
    assert contrasts[0] * (1 - 1e-9) <= level <= contrasts[-1] * (1 + 1e-9)
    assert np.linalg.cond(problem.M) <= contrasts[-1] / contrasts[0] * (1 + 1e-6)


def test_no_penalty_on_a_long_panel_is_solved_exactly():
    """With ``T > N`` and ``lam = 0`` the problem on contrasts is well posed. The
    level direction must not make ``M`` singular."""
    Y = _panel(N=6, T=30, seed=3)
    assert BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=0.0)).reliable
    design = solve_synthetic_design_exact(Y, 2, lam=0.0)

    raw = TwoWayProblem(G=Y.T @ Y / 30, M=Y.T @ Y / 30, lam=0.0, N=6)
    subsets = np.array(list(combinations(range(6), 2)), dtype=np.int64)
    t, c = split_indices(6, subsets)
    best = solve_partition_batch(raw, t, c, max_iter=20_000, tol=1e-14).value.min()
    assert design.objective_value == pytest.approx(float(best), rel=1e-7)


# --------------------------------------------------------------------------- #
# the swap search above the enumeration limit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s", [1e-8, 1e-4, 1e4, 1e8])
def test_the_swap_search_does_not_depend_on_units(s):
    Y = _panel(seed=1)
    ref = maxcut_candidates(BoundEngine.from_problem(TwoWayProblem.from_panel(Y)), 3)
    got = maxcut_candidates(BoundEngine.from_problem(TwoWayProblem.from_panel(Y * s)), 3)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s", [1e-6, 1e6])
def test_the_fallback_design_does_not_depend_on_units(seed, s):
    Y = _panel(seed=seed)
    ref = solve_synthetic_design_exact(Y, 3, candidate_limit=10)
    got = solve_synthetic_design_exact(Y * s, 3, candidate_limit=10)
    assert _treated(got) == _treated(ref)
    assert got.objective_value / s ** 2 == pytest.approx(ref.objective_value, rel=1e-8)


def test_the_fallback_design_does_not_depend_on_a_level():
    Y = _panel(seed=1)
    ref = solve_synthetic_design_exact(Y, 3, candidate_limit=10)
    got = solve_synthetic_design_exact(Y + 1e8 * Y.std(), 3, candidate_limit=10)
    assert _treated(got) == _treated(ref)
    assert got.objective_value == pytest.approx(ref.objective_value, rel=1e-6)


# --------------------------------------------------------------------------- #
# the inner solver's stop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s", [1e-6, 1.0, 1e6])
def test_the_inner_solver_stops_on_a_relative_gap(s):
    problem = TwoWayProblem.from_panel(_panel(seed=0) * s)
    t, c = split_indices(problem.N, np.array([[0, 1, 2], [3, 4, 5]]))
    out = solve_partition_batch(problem, t, c, max_iter=20_000, tol=1e-12)
    assert np.all(out.gap <= 1e-12 * out.value)


# --------------------------------------------------------------------------- #
# failures are named, and reach the caller
# --------------------------------------------------------------------------- #


def test_an_ill_conditioned_instance_past_the_limit_says_so():
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned") as exc:
        solve_synthetic_design_exact(_ill_conditioned(), 3, lam=1e-14,
                                     candidate_limit=10)
    assert "candidate limit" in str(exc.value)


def test_the_pool_reports_an_ill_conditioned_instance():
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned"):
        solve_synthetic_design_exact_pool(_ill_conditioned(), 3, top_K=3, lam=1e-14,
                                          candidate_limit=10)


def test_a_free_size_reports_an_ill_conditioned_instance():
    """Size 1 holds 8 designs, size 2 holds 28, past a limit of 20."""
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned"):
        solve_synthetic_design_exact_pool(_ill_conditioned(), None, top_K=2, lam=1e-14,
                                          candidate_limit=20)


def test_the_estimator_reports_an_ill_conditioned_instance_under_holdout():
    """Forty units at K = 10 hold 8.5e8 designs, past the default candidate
    limit, so the refusal comes before any set is priced."""
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((10, 2)) @ rng.standard_normal((2, 40))
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]), "post": int(t >= 7)}
            for j in range(40) for t in range(10)]
    est = SYNDES({"df": pd.DataFrame(rows), "outcome": "y", "unitid": "unit",
                  "time": "time", "post_col": "post", "K": 10,
                  "mode": "two_way_global", "top_K": 3, "lam": 1e-14})
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned") as exc:
        est.fit()
    assert "unsatisfiable" not in str(exc.value)


def test_the_pool_still_stops_when_the_region_is_exhausted():
    pool = solve_synthetic_design_exact_pool(_panel(N=4, T=10), 2, top_K=10)
    assert len(pool) == 6


def test_a_free_size_skips_the_sizes_restrictions_rule_out():
    design = solve_synthetic_design_exact(
        _panel(N=7, T=12), None, restrictions=DesignRestrictions(forced_in=[0, 1, 2]))
    assert {0, 1, 2} <= set(_treated(design))


def test_an_exhausted_region_is_raised_as_exhaustion():
    """The pool and ``K=None`` stop only on this class, so it is pinned by name.

    The failure is caught and its class compared outside the ``except``: a
    "found no feasible design" error escaping a SYNDES helper is reclassified as
    a skip by ``conftest.py`` whenever MOSEK or GUROBI is absent, which would
    hide exactly this regression.
    """
    from itertools import combinations as _comb

    from mlsynth.utils.syndes_helpers.exact import _NoFeasibleDesign

    every = [np.array(s) for s in _comb(range(4), 2)]
    caught = None
    try:
        solve_synthetic_design_exact(_panel(N=4, T=10), 2, forbidden_sets=every)
    except MlsynthEstimationError as exc:
        caught = exc
    assert type(caught).__name__ == _NoFeasibleDesign.__name__

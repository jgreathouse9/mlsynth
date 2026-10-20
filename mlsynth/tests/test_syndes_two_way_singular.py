"""SYNDES two-way where ``G + lam I`` is singular to working precision (#821).

With ``lam = 0`` on a panel with fewer periods than units, or with a unit that
is an exact combination of others, the closed-form bound needs an inverse that
does not exist. The search then prices every admissible set by projected
gradient, which needs only that ``G + lam I`` be positive semidefinite:

* a set priced at zero, to the rounding of its terms, is optimal, since
  ``f(S) >= 0``, and ends the search;
* otherwise every set is priced, the sets whose lower bound ``value - gap`` is
  still under the incumbent are priced again with more iterations, and the
  design is certified when the incumbent is within ``1e-8`` of the smallest
  lower bound, or within the rounding of zero.

The reference is the MIP with the assignment fixed, solved by CLARABEL
(``optimization._refit``), an independent solver. Its accuracy at ``lam = 0``
is about ``1e-8`` relative, so comparisons are at ``1e-6``.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers.exact import (
    SubsetFilter,
    solve_synthetic_design_exact,
    solve_synthetic_design_exact_pool,
)
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem
from mlsynth.utils.syndes_helpers.optimization import SYNDESSolveWarning, _refit
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _base(N=8, T=14, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


def _collinear():
    """Unit 2 is the mean of units 0 and 1: treating 0 and 1 against 2 fits
    exactly, so the optimum at ``lam = 0`` is zero."""
    Y = _base()
    Y[:, 2] = 0.5 * (Y[:, 0] + Y[:, 1])
    return Y


def _tied():
    """Unit 7 is units 0 + 1 + 2 + 3 less 4 + 5 + 6: one null contrast, with
    four units on each side, which no design of three can use. Singular, with a
    positive optimum at ``K = 3``."""
    Y = _base(seed=1)
    Y[:, 7] = Y[:, :4].sum(axis=1) - Y[:, 4:7].sum(axis=1)
    return Y


def _short():
    """Eight units over three periods."""
    return _base(T=3, seed=2)


def _reference(Y, K, accepts=lambda S: True):
    """``f(S)`` for every admissible set, from CLARABEL."""
    N = Y.shape[1]
    out = {}
    for S in combinations(range(N), K):
        if not accepts(S):
            continue
        D = np.zeros(N)
        D[list(S)] = 1.0
        out[S] = _refit(Y, D, "global_2way", 0.0, None)[1]
    return out


def _treated(design):
    return tuple(int(i) for i in design.selected_unit_indices)


def _singular(Y):
    return not BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=0.0)).reliable


def test_the_panels_are_singular():
    assert _singular(_collinear()) and _singular(_tied()) and _singular(_short())


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


def test_a_singular_panel_gets_a_certified_two_way_design():
    d = solve_synthetic_design_exact(_collinear(), 2, lam=0.0)
    assert d.raw_results["certified"] is True
    assert d.raw_results["lower_bound"] == 0.0
    assert abs(d.objective_value) <= 1e-12


# --------------------------------------------------------------------------- #
# the design is the optimum
# --------------------------------------------------------------------------- #


def test_a_positive_optimum_is_the_reference_one():
    Y = _tied()
    ref = _reference(Y, 3)
    best = min(ref.values())
    assert best > 1e-3
    d = solve_synthetic_design_exact(Y, 3, lam=0.0)
    assert d.raw_results["certified"] is True
    assert d.objective_value == pytest.approx(best, rel=1e-6)
    assert ref[_treated(d)] == pytest.approx(best, rel=1e-6)
    lower = d.raw_results["lower_bound"]
    assert lower <= d.objective_value
    assert d.objective_value - lower <= 1e-8 * d.objective_value


def test_a_short_panel_has_a_zero_optimum():
    d = solve_synthetic_design_exact(_short(), 3, lam=0.0)
    assert d.raw_results["certified"] is True
    assert abs(d.objective_value) <= 1e-12


def test_the_weights_attain_the_objective():
    Y = _tied()
    d = solve_synthetic_design_exact(Y, 3, lam=0.0)
    u = d.treated_weights - d.control_weights
    centred = Y - Y.mean(axis=1, keepdims=True)
    value = float(np.sum((centred @ u) ** 2) / Y.shape[0])
    assert value == pytest.approx(d.objective_value, rel=1e-9)
    assert d.treated_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert d.control_weights.sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# the search's two passes
# --------------------------------------------------------------------------- #


def test_a_zero_ends_the_search(monkeypatch):
    """The first set, (0, 1) against unit 2, fits exactly; with blocks of eight
    the search stops after the first block."""
    import mlsynth.utils.syndes_helpers.exact as ex
    monkeypatch.setattr(ex, "SINGULAR_CHUNK", 8)
    d = solve_synthetic_design_exact(_collinear(), 2, lam=0.0)
    assert d.raw_results["certified"] is True
    assert d.raw_results["n_designs"] == 8


def test_a_small_positive_optimum_is_not_read_as_zero(monkeypatch):
    """At a scale of 1e-6 the optimum is about 1e-14. The zero it is compared
    with scales with ``M``, so every set is priced and the optimum found."""
    import mlsynth.utils.syndes_helpers.exact as ex
    monkeypatch.setattr(ex, "SINGULAR_CHUNK", 8)
    Y = _tied()
    best = min(_reference(Y, 3).values())
    d = solve_synthetic_design_exact(Y * 1e-6, 3, lam=0.0)
    assert d.raw_results["n_designs"] == 56
    assert d.objective_value / 1e-12 == pytest.approx(best, rel=1e-6)


def test_the_open_sets_are_priced_again(monkeypatch):
    """Fifty first-pass iterations leave every gap open; the second pass closes
    the ones whose bound is under the incumbent."""
    import mlsynth.utils.syndes_helpers.exact as ex
    monkeypatch.setattr(ex, "FIRST_PASS_ITERATIONS", 50)
    Y = _tied()
    best = min(_reference(Y, 3).values())
    d = solve_synthetic_design_exact(Y, 3, lam=0.0)
    assert d.raw_results["certified"] is True
    assert d.objective_value == pytest.approx(best, rel=1e-6)


def test_gaps_left_open_are_reported(monkeypatch):
    import mlsynth.utils.syndes_helpers.exact as ex
    monkeypatch.setattr(ex, "FIRST_PASS_ITERATIONS", 50)
    monkeypatch.setattr(ex, "REFINE_ITERATIONS", 50)
    Y = _tied()
    best = min(_reference(Y, 3).values())
    with pytest.warns(SYNDESSolveWarning, match="not a proven optimum"):
        d = solve_synthetic_design_exact(Y, 3, lam=0.0)
    lower = d.raw_results["lower_bound"]
    assert d.raw_results["certified"] is False
    assert d.raw_results["status"] == "feasible"
    assert lower <= best * (1 + 1e-6)
    assert lower < d.objective_value * (1 - 1e-6)


def test_a_certified_search_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESSolveWarning)
        solve_synthetic_design_exact(_tied(), 3, lam=0.0)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def test_two_identical_units():
    """Both units equal their period's mean, so ``M = 0`` and every design
    scores zero: there is no gradient step, and the first design is optimal."""
    Y = _base(N=2, T=5)
    Y[:, 1] = Y[:, 0]
    assert _singular(Y)
    d = solve_synthetic_design_exact(Y, 1, lam=0.0)
    assert d.raw_results["certified"] is True and d.objective_value == 0.0
    assert d.treated_weights.sum() == 1.0 and d.control_weights.sum() == 1.0
    assert np.all(d.control_weights[d.selected_unit_indices] == 0.0)


def test_identical_units_admit_no_design_under_restrictions():
    Y = _base(N=3, T=5)
    Y[:, 1] = Y[:, 0]
    Y[:, 2] = Y[:, 0]
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_synthetic_design_exact(
            Y, 1, lam=0.0, restrictions=DesignRestrictions(forbidden=[0, 1, 2]))


def test_restrictions_that_leave_no_design_are_reported():
    with pytest.raises(MlsynthEstimationError, match="no feasible design"):
        solve_synthetic_design_exact(
            _tied(), 3, lam=0.0,
            restrictions=DesignRestrictions(forbidden=[0, 1, 2, 3, 4, 5]))


def test_a_duplicated_unit():
    Y = _base(N=6, T=14, seed=3)
    Y[:, 5] = Y[:, 0]
    assert _singular(Y)
    d = solve_synthetic_design_exact(Y, 2, lam=0.0)
    assert d.raw_results["certified"] is True
    assert abs(d.objective_value) <= 1e-12


def test_restrictions_hold_on_the_singular_path():
    Y = _tied()
    rules = DesignRestrictions(forbidden=[0, 4], conflict_pairs=[(1, 5)])
    filt = SubsetFilter.build(rules, N=8)
    ref = _reference(Y, 3, accepts=filt.accepts)
    d = solve_synthetic_design_exact(Y, 3, lam=0.0, restrictions=rules)
    assert filt.accepts(_treated(d))
    assert d.objective_value == pytest.approx(min(ref.values()), rel=1e-6)


def test_every_size_is_searched_when_k_is_free():
    d = solve_synthetic_design_exact(_collinear(), None, lam=0.0)
    assert d.raw_results["certified"] is True
    assert abs(d.objective_value) <= 1e-12


def test_the_pool_returns_distinct_designs_in_order():
    pool = solve_synthetic_design_exact_pool(_tied(), 3, top_K=3, lam=0.0)
    assert len({_treated(d) for d in pool}) == 3
    values = [d.objective_value for d in pool]
    assert values == sorted(values)


def test_the_estimator_designs_on_a_singular_panel():
    Y = _collinear()
    T, N = Y.shape
    df = pd.DataFrame([{"unit": j, "time": t, "y": float(Y[t, j]),
                        "post": int(t >= T - 3)} for j in range(N) for t in range(T)])
    res = SYNDES({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                  "post_col": "post", "K": 2, "mode": "two_way_global",
                  "lam": 0.0, "run_inference": False}).fit()
    assert res.design.raw_results["solver"] == "exact"
    assert res.design.raw_results["certified"] is True


# --------------------------------------------------------------------------- #
# failures, and that they reach the caller
# --------------------------------------------------------------------------- #


def test_past_the_pricing_limit_without_a_zero_it_raises(monkeypatch):
    import mlsynth.utils.syndes_helpers.exact as ex
    monkeypatch.setattr(ex, "PRICE_ALL_LIMIT", 10)
    monkeypatch.setattr(ex, "SINGULAR_CHUNK", 8)
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned") as exc:
        solve_synthetic_design_exact(_tied(), 3, lam=0.0)
    assert "mip" not in str(exc.value).lower()


def test_past_the_candidate_limit_it_raises():
    with pytest.raises(MlsynthEstimationError, match="ill-conditioned") as exc:
        solve_synthetic_design_exact(_tied(), 3, lam=0.0, candidate_limit=5)
    assert "mip" not in str(exc.value).lower()


def test_the_pool_reports_the_failure():
    """Caught and checked outside the ``except``, so ``conftest.py``'s
    optional-solver guard cannot reclassify it (#813)."""
    caught = None
    try:
        solve_synthetic_design_exact_pool(_tied(), 3, top_K=2, lam=0.0,
                                          candidate_limit=5)
    except MlsynthEstimationError as exc:
        caught = exc
    assert caught is not None and "ill-conditioned" in str(caught)


def test_a_free_size_reports_the_failure():
    """Sizes 1 and 2 hold 8 and 28 sets; size 3 holds 56, past a limit of 30."""
    caught = None
    try:
        solve_synthetic_design_exact(_tied(), None, lam=0.0, candidate_limit=30)
    except MlsynthEstimationError as exc:
        caught = exc
    assert caught is not None and "ill-conditioned" in str(caught)


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #


@st.composite
def singular_panels(draw):
    N = draw(st.integers(min_value=3, max_value=7))
    T = draw(st.integers(min_value=2, max_value=N))
    K = draw(st.integers(min_value=1, max_value=N - 1))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2 ** 32 - 1)))
    Y = rng.standard_normal((T, 2)) @ rng.standard_normal((2, N)) + 0.3 * rng.standard_normal((T, N))
    if draw(st.booleans()):
        Y[:, -1] = Y[:, 0]
    return Y, K


@settings(max_examples=25, deadline=None)
@given(singular_panels(), st.integers(min_value=-6, max_value=6).map(lambda k: 10.0 ** k))
def test_a_certified_design_is_the_same_in_other_units(case, s):
    Y, K = case
    ref = solve_synthetic_design_exact(Y, K, lam=0.0)
    got = solve_synthetic_design_exact(Y * s, K, lam=0.0)
    assert ref.raw_results["certified"] and got.raw_results["certified"]
    floor = 256 * np.finfo(float).eps * float(
        np.abs(TwoWayProblem.from_panel(Y, lam=0.0).M).max())
    assert abs(got.objective_value / s ** 2 - ref.objective_value) <= max(
        1e-7 * abs(ref.objective_value), 2 * floor)


@settings(max_examples=25, deadline=None)
@given(singular_panels(), st.floats(min_value=-1e6, max_value=1e6,
                                    allow_nan=False, allow_infinity=False))
def test_a_common_level_changes_nothing(case, c):
    Y, K = case
    ref = solve_synthetic_design_exact(Y, K, lam=0.0)
    got = solve_synthetic_design_exact(Y + c, K, lam=0.0)
    assert got.raw_results["certified"]
    floor = 256 * np.finfo(float).eps * float(
        np.abs(TwoWayProblem.from_panel(Y, lam=0.0).M).max())
    assert abs(got.objective_value - ref.objective_value) <= max(
        1e-6 * abs(ref.objective_value), 1e3 * floor)

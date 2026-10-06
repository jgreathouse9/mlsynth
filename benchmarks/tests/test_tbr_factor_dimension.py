r"""The factor-dimension study's own invariants.

The case asserts coverage, which is a statistic. These assert the mechanism
behind it, which is algebra: aggregating gives each group the mean loading of
its members, and the fitted affine relation holds at every period exactly when
those two mean vectors are proportional. At one factor they are scalars and
that is automatic; past one factor it is a coincidence.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.studies.tbr_factor import coverage as cv
from benchmarks.studies.tbr_factor import dgp


# ------------------------------------------------------------------- smoke
def test_a_panel_has_the_shape_dataprep_expects():
    frame = dgp.panel(0, r=2, n_treated=5)
    assert set(frame.columns) == {"geo", "t", "y", "post", "is_treat", "is_ctrl"}
    assert frame["geo"].nunique() == 50
    assert frame["t"].nunique() == 80
    assert len(frame) == 50 * 80
    assert frame["is_treat"].sum() == 5 * 80
    assert (frame["is_treat"] + frame["is_ctrl"] == 1).all()
    assert frame["post"].sum() == 50 * 20
    assert frame["y"].gt(0).all()


def test_one_replication_reports_a_finite_verdict():
    row = cv.one_replication(0, r=1, n_treated=5)
    assert isinstance(row["covered"], bool)
    assert np.isfinite(row["width"]) and row["width"] > 0
    assert np.isfinite(row["estimate"]) and np.isfinite(row["delta2"])


# ------------------------------------------- the mechanism, as exact algebra
@pytest.mark.parametrize("k", [2, 5, 25])
@pytest.mark.parametrize("seed", range(6))
def test_one_factor_makes_the_two_group_loadings_proportional(seed, k):
    """Two scalars are always proportional, so the sine of the angle is zero."""
    assert dgp.panel(seed, 1, k).attrs["collinearity"] == 0.0


@pytest.mark.parametrize("k", [2, 5, 25])
@pytest.mark.parametrize("seed", range(6))
def test_one_factor_makes_the_affine_relation_exact(seed, k):
    """With the noise off, the best affine fit leaves nothing at one factor."""
    assert dgp.noiseless_gap(seed, 1, k) < 1e-12


@pytest.mark.parametrize("r", [2, 3, 5])
@pytest.mark.parametrize("k", [2, 5, 25])
def test_past_one_factor_the_relation_is_not_exact(r, k):
    """And not merely less exact: orders of magnitude away from precision."""
    gaps = [dgp.noiseless_gap(seed, r, k) for seed in range(6)]
    assert min(gaps) > 1e-6, f"r={r}, k={k}: smallest gap {min(gaps):.2e}"


def test_the_gap_and_the_collinearity_move_together():
    """The angle is the quantity that decides the gap, so they rank alike."""
    pairs = [(dgp.panel(s, r, 5).attrs["collinearity"],
              dgp.noiseless_gap(s, r, 5))
             for r in (2, 3, 5, 8) for s in range(6)]
    angles = np.array([p[0] for p in pairs])
    gaps = np.array([p[1] for p in pairs])
    assert np.corrcoef(angles, gaps)[0, 1] > 0.5


def test_averaging_more_geos_does_not_restore_the_relation():
    """The failure is not a small-treated-group artefact."""
    for r in (2, 3):
        for k in (2, 5, 25):
            assert dgp.noiseless_gap(0, r, k) > 1e-6


# ------------------------------------------------- the coverage interval
def test_the_coverage_interval_brackets_an_interior_count():
    lo, hi = cv.beta_interval(30, 60, 0.99)
    assert 0.0 < lo <= 0.5 <= hi < 1.0


@pytest.mark.parametrize("y,n", [(0, 40), (40, 40)])
def test_a_unanimous_count_still_gets_real_bounds(y, n):
    lo, hi = cv.beta_interval(y, n, 0.99)
    assert 0.0 < lo < hi < 1.0


def test_the_prior_is_kermans_and_not_a_uniform_one():
    from scipy import stats
    y, n, level = 9, 30, 0.99
    lo, hi = cv.beta_interval(y, n, level)
    a, b = cv.NEUTRAL_PRIOR + y, cv.NEUTRAL_PRIOR + n - y
    tail = 0.5 * (1 - level)
    assert lo == pytest.approx(stats.beta.ppf(tail, a, b), rel=1e-12)
    assert hi == pytest.approx(stats.beta.ppf(1 - tail, a, b), rel=1e-12)
    assert cv.NEUTRAL_PRIOR == pytest.approx(1.0 / 3.0)


# -------------------------------------------------------- cell independence
def test_a_cells_result_does_not_depend_on_the_request_it_arrived_in():
    """The lesson from the shared-stream defect in the sibling study.

    A cell's draws are derived from its own identity, so asking for one factor
    count alone has to give the same rows as asking for it beside others.
    """
    alone = cv.run_sweep(3, rs=(1,), ks=(5,), seed=0)
    beside = [r for r in cv.run_sweep(3, rs=(1, 2, 3), ks=(2, 5), seed=0)
              if r["r"] == 1 and r["n_treated"] == 5]
    assert len(alone) == len(beside) == 3
    for a, b in zip(alone, beside):
        assert a["estimate"] == pytest.approx(b["estimate"], rel=1e-12)
        assert a["covered"] == b["covered"]


def test_the_sweep_is_reproducible_from_its_seed():
    a = cv.run_sweep(3, rs=(2,), ks=(5,), seed=1)
    b = cv.run_sweep(3, rs=(2,), ks=(5,), seed=1)
    assert [r["estimate"] for r in a] == [r["estimate"] for r in b]


def test_a_different_seed_moves_the_draws():
    a = cv.run_sweep(3, rs=(2,), ks=(5,), seed=1)
    b = cv.run_sweep(3, rs=(2,), ks=(5,), seed=2)
    assert [r["estimate"] for r in a] != [r["estimate"] for r in b]


# -------------------------------------------------------------- edge cases
@pytest.mark.parametrize("r", [0, -1])
def test_a_panel_needs_at_least_one_factor(r):
    with pytest.raises(ValueError, match="at least one factor"):
        dgp.panel(0, r, 5)


@pytest.mark.parametrize("k", [0, 50, 51, -1])
def test_a_panel_needs_both_groups_non_empty(k):
    with pytest.raises(ValueError, match="both groups non-empty"):
        dgp.panel(0, 2, k)


@pytest.mark.parametrize("n_pre", [0, 80, 90])
def test_the_split_has_to_sit_inside_the_panel(n_pre):
    with pytest.raises(ValueError, match="inside the panel"):
        dgp.panel(0, 2, 5, n_pre=n_pre)


def test_collinearity_of_a_degenerate_loading_block_is_not_a_number():
    lam = np.zeros((10, 3))
    assert not np.isfinite(dgp.collinearity(lam, 5))


# --------------------------------------------------------------- properties
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), k=st.integers(1, 49))
def test_one_factor_is_exact_over_the_whole_domain(seed, k):
    """Not just at the fixtures: at one factor the gap is always at precision."""
    assert dgp.noiseless_gap(seed, 1, k) < 1e-10
    assert dgp.panel(seed, 1, k).attrs["collinearity"] == 0.0


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), r=st.integers(1, 6),
       k=st.integers(2, 40))
def test_collinearity_is_a_sine_and_vanishes_exactly_at_one_factor(seed, r, k):
    c = dgp.panel(seed, r, k).attrs["collinearity"]
    assert 0.0 <= c <= 1.0
    assert (c == 0.0) if r == 1 else True


# ---------------------------------------------------------------- summarise
def _rows(spec):
    """Minimal rows: spec maps (r, k) to the covered flags to report."""
    return [{"r": r, "n_treated": k, "seed": i, "covered": c, "width": 1.0 + r,
             "estimate": 0.0, "delta2": 1.0, "collinearity": 0.1 * (r - 1)}
            for (r, k), flags in spec.items() for i, c in enumerate(flags)]


def test_summarise_reports_one_rate_per_factor_count():
    s = cv.summarise(_rows({(1, 5): [True] * 9 + [False],
                            (2, 5): [True] * 5 + [False] * 5}))
    assert s["cov_r1"] == pytest.approx(0.9)
    assert s["cov_r2"] == pytest.approx(0.5)
    assert s["n_per_r"] == 10.0
    assert s["width_r1"] == pytest.approx(2.0)
    assert s["collinearity_r1"] == pytest.approx(0.0)


def test_summarise_brackets_each_rate_with_an_interval():
    s = cv.summarise(_rows({(1, 5): [True] * 18 + [False] * 2}))
    assert s["cov_r1_lo"] <= s["cov_r1"] <= s["cov_r1_hi"]
    assert 0.0 < s["cov_r1_lo"] and s["cov_r1_hi"] < 1.0


def test_summarise_takes_the_best_multifactor_cell_not_the_mean():
    """The claim is an upper bound, so the most favourable cell is the one
    that has to fall short."""
    s = cv.summarise(_rows({(1, 5): [True] * 10,
                            (2, 2): [True] * 2 + [False] * 8,
                            (2, 25): [True] * 7 + [False] * 3}))
    assert s["multifactor_cov_max_over_k"] == pytest.approx(0.7)


def test_summarise_pools_a_factor_count_over_treated_group_sizes():
    s = cv.summarise(_rows({(2, 2): [True] * 5, (2, 25): [False] * 5}))
    assert s["cov_r2"] == pytest.approx(0.5)
    assert s["n_per_r"] == 10.0


def test_summarise_omits_the_multifactor_bound_when_there_is_no_multifactor_cell():
    s = cv.summarise(_rows({(1, 5): [True] * 4}))
    assert "multifactor_cov_max_over_k" not in s

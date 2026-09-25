"""The shared weight-solver layer.

Scope is the quadratic corner of the Cressie-Read family -- gamma = 1 -- over
polyhedral weight sets, which is where an exact finite-termination method
exists. Entropy and empirical likelihood are exponential-cone and are refused
by raising. See `agents/agents_solver.md` for the boundary and its sources.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.weights import (
    WeightConstraint,
    WeightObjective,
    WeightSolution,
    solve_weights,
)

_T0 = 24


@pytest.fixture(scope="module")
def panel():
    """Donors of full column rank, treated unit off the hull by a real margin."""
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 1.0, _T0)
    B = np.column_stack([np.ones(_T0), t, np.sin(3 * t), t ** 2]) @ rng.normal(size=(4, 6))
    B = B * 3.0 + 40.0 + rng.normal(scale=0.4, size=(_T0, 6))
    w = np.zeros(6); w[[0, 2, 4]] = (0.5, 0.3, 0.2)
    return B, B @ w + 2.5 + rng.normal(scale=0.02, size=_T0)


def _ssr(B, A, w, a=0.0):
    return float(np.sum((A - B @ w - a) ** 2))


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------
def test_the_default_constraint_is_the_simplex(panel):
    B, A = panel
    sol = solve_weights(B, A)
    assert isinstance(sol, WeightSolution)
    assert sol.weights.shape == (B.shape[1],)
    assert sol.weights.min() >= 0.0
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# The constraint sets, each doing what it says
# --------------------------------------------------------------------------
def test_simplex_returns_exact_zeros_not_solver_dust(panel):
    """The defect that motivated the layer: a conic solver leaves weights at
    -2e-09 where an exact method returns zeros."""
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint())
    off = sol.weights[sol.weights < 1e-8]
    assert np.all(off == 0.0)


def test_cone_is_nonnegative_and_free_to_exceed_one(panel):
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(sum_to_one=False))
    assert sol.weights.min() >= -1e-12
    assert _ssr(B, A, sol.weights) <= _ssr(B, A, solve_weights(B, A).weights) + 1e-9


def test_affine_sums_to_one_and_may_go_negative(panel):
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(nonneg=False))
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert _ssr(B, A, sol.weights) <= _ssr(B, A, solve_weights(B, A).weights) + 1e-9


def test_free_matches_least_squares(panel):
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(nonneg=False, sum_to_one=False))
    ref, *_ = np.linalg.lstsq(B, A, rcond=None)
    assert sol.weights == pytest.approx(ref, abs=1e-7)


def test_box_respects_its_upper_bound(panel):
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(sum_to_one=False, upper=0.25))
    assert sol.weights.min() >= -1e-9
    assert sol.weights.max() <= 0.25 + 1e-9


def test_the_sets_are_nested_in_the_order_they_restrict(panel):
    """free <= affine <= simplex, and free <= cone <= simplex."""
    B, A = panel
    f = _ssr(B, A, solve_weights(B, A, WeightConstraint(nonneg=False, sum_to_one=False)).weights)
    af = _ssr(B, A, solve_weights(B, A, WeightConstraint(nonneg=False)).weights)
    co = _ssr(B, A, solve_weights(B, A, WeightConstraint(sum_to_one=False)).weights)
    sx = _ssr(B, A, solve_weights(B, A).weights)
    assert f <= af + 1e-9 and af <= sx + 1e-9
    assert f <= co + 1e-9 and co <= sx + 1e-9


# --------------------------------------------------------------------------
# The intercept, which is a field and not a convention
# --------------------------------------------------------------------------
def test_the_intercept_is_free_in_sign(panel):
    """ClusterSC's own implementation puts the intercept inside the simplex,
    bounded in [0, 1] and competing with the donors for the unit budget. A
    treated unit below its donors then cannot be fitted at all."""
    B, A = panel
    sol = solve_weights(B, A - 2 * 2.5, WeightConstraint(intercept=True))
    assert sol.intercept < -1.0
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_shifting_the_target_moves_only_the_intercept(panel):
    B, A = panel
    c = WeightConstraint(intercept=True)
    a0 = solve_weights(B, A, c); a1 = solve_weights(B, A + 13.0, c)
    assert a1.intercept - a0.intercept == pytest.approx(13.0, abs=1e-6)
    assert a1.weights == pytest.approx(a0.weights, abs=1e-6)


def test_the_intercept_zeroes_the_mean_residual(panel):
    """First-order condition for an unconstrained coordinate."""
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(intercept=True))
    assert float(np.mean(A - B @ sol.weights - sol.intercept)) == pytest.approx(0.0, abs=1e-9)


def test_no_intercept_reports_exactly_zero(panel):
    B, A = panel
    assert solve_weights(B, A).intercept == 0.0


# --------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------
def test_ridge_shrinks_toward_uniform_by_default(panel):
    """Liao-Shi-Zheng shrink toward 1_J/J and so do their comparators."""
    B, A = panel
    J = B.shape[1]
    plain = solve_weights(B, A).weights
    heavy = solve_weights(B, A, objective=WeightObjective(ridge=1e6)).weights
    assert heavy == pytest.approx(np.full(J, 1.0 / J), abs=1e-3)
    assert np.abs(heavy - 1.0 / J).sum() < np.abs(plain - 1.0 / J).sum()


def test_ridge_can_shrink_toward_a_named_target(panel):
    B, A = panel
    tgt = np.zeros(B.shape[1]); tgt[1] = 1.0
    got = solve_weights(B, A, objective=WeightObjective(ridge=1e6, toward=tgt)).weights
    assert got == pytest.approx(tgt, abs=1e-3)


# --------------------------------------------------------------------------
# The certificate. Slater holds on the simplex unconditionally, so KKT is
# necessary and sufficient and a residual from the returned weights is a
# complete proof of optimality -- no appeal to what a backend claims.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("con", [
    WeightConstraint(),
    WeightConstraint(sum_to_one=False),
    WeightConstraint(nonneg=False),
    WeightConstraint(nonneg=False, sum_to_one=False),
    WeightConstraint(intercept=True),
])
def test_every_solution_carries_a_small_kkt_residual(panel, con):
    B, A = panel
    assert solve_weights(B, A, con).kkt_residual < 1e-7


def test_a_perturbed_solution_has_a_large_kkt_residual(panel):
    """The certificate must be able to reject, or it certifies nothing."""
    from mlsynth.utils.weights.solve import kkt_residual
    B, A = panel
    sol = solve_weights(B, A)
    bad = sol.weights.copy(); bad[0] += 0.2; bad[1] -= 0.2
    assert kkt_residual(B, A, bad, 0.0, WeightConstraint(), WeightObjective()) > 1e-3


# --------------------------------------------------------------------------
# Uniqueness. rank(B) < J is necessary, not sufficient: the simplex can cut
# out exactly the flat directions.
# --------------------------------------------------------------------------
def test_full_rank_donors_give_a_unique_minimiser(panel):
    B, A = panel
    assert solve_weights(B, A).unique is True


def test_a_duplicated_donor_is_reported_as_not_unique(panel):
    """An exact method returns a *vertex* of the optimal face, so it typically
    puts the whole weight on one twin and leaves the other at zero. The verdict
    must not depend on which of the two the backend happened to pick."""
    B, A = panel
    lead = int(np.argmax(solve_weights(B, A).weights))
    Bd = np.column_stack([B, B[:, lead]])
    sol = solve_weights(Bd, A)
    assert sol.weights[-1] == 0.0                     # the copy is off the support
    assert sol.unique is False                        # and it is still a continuum


def test_moving_weight_between_duplicated_donors_leaves_the_objective_alone(panel):
    """What `unique is False` is claiming, measured directly."""
    B, A = panel
    lead = int(np.argmax(solve_weights(B, A).weights))
    Bd = np.column_stack([B, B[:, lead]])
    w = np.array(solve_weights(Bd, A).weights)
    moved = w.copy(); moved[lead] -= 0.2; moved[-1] += 0.2
    ssr = lambda v: float(np.sum((A - Bd @ v) ** 2))
    assert ssr(moved) == pytest.approx(ssr(w), rel=1e-12)


def test_a_donor_the_optimum_strictly_rejects_does_not_make_it_a_continuum(panel):
    """The test keys on a *zero* reduced gradient at the bound, not on the
    weight being zero. A donor that is off the support because admitting it
    would cost objective leaves the minimiser unique."""
    B, A = panel
    sol = solve_weights(B, A)
    assert sol.weights.min() == 0.0 and sol.unique is True


def test_more_donors_than_periods_need_not_be_non_unique():
    """Proposition 99 is unique at J=38 over T0=19. The rank test alone
    would call it a continuum."""
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "smoking_data.csv")
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    p = dataprep(df, "state", "year", "cigsale", "treat")
    T0 = int(p["pre_periods"])
    B = np.asarray(p["donor_matrix"], float)[:T0]
    A = np.asarray(p["y"], float).ravel()[:T0]
    assert B.shape[1] > T0
    assert np.linalg.matrix_rank(B) < B.shape[1]
    assert solve_weights(B, A).unique is True


# --------------------------------------------------------------------------
# Scale. A conic solver calls the simplex infeasible at 1000x German GDP.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [1e-4, 1.0, 1e3, 1e6])
def test_the_solution_is_equivariant_to_a_common_rescaling(panel, factor):
    B, A = panel
    base = solve_weights(B, A)
    got = solve_weights(B * factor, A * factor)
    assert got.status == "optimal"
    assert got.weights == pytest.approx(base.weights, abs=1e-6)


# --------------------------------------------------------------------------
# Refusal. gamma < 1 is exponential-cone and belongs to cvxpy.
# --------------------------------------------------------------------------
def test_an_entropy_objective_is_refused_by_name():
    with pytest.raises(MlsynthConfigError, match="exponential cone|entropy|gamma"):
        WeightObjective(divergence="entropy")


def test_an_unsupported_constraint_shape_raises_and_names_itself(panel):
    B, A = panel
    with pytest.raises(MlsynthConfigError, match="not covered|upper"):
        solve_weights(B, A, WeightConstraint(upper=0.5))     # box + sum_to_one


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------
def test_a_non_2d_design_raises(panel):
    B, A = panel
    with pytest.raises(MlsynthEstimationError, match="2D"):
        solve_weights(B[:, 0], A)


def test_a_length_mismatch_raises(panel):
    B, A = panel
    with pytest.raises(MlsynthEstimationError, match="mismatch"):
        solve_weights(B, A[:-2])


def test_a_negative_ridge_raises():
    with pytest.raises(MlsynthConfigError, match="ridge"):
        WeightObjective(ridge=-1.0)


# --------------------------------------------------------------------------
# Degenerate and boundary inputs
# --------------------------------------------------------------------------
def test_a_lone_donor_on_the_affine_line_is_forced_to_one(panel):
    B, A = panel
    sol = solve_weights(B[:, :1], A, WeightConstraint(nonneg=False))
    assert sol.weights == pytest.approx([1.0])
    assert sol.kkt_residual < 1e-7


def test_an_all_zero_donor_block_is_reported_as_not_unique():
    B = np.zeros((6, 3))
    sol = solve_weights(B, np.ones(6), WeightConstraint(nonneg=False, sum_to_one=False))
    assert sol.unique is False


def test_a_vertex_solution_with_every_other_donor_off_is_still_unique(panel):
    """No free coordinate left means nothing can move; the face is a point."""
    B, A = panel
    sol = solve_weights(B, B[:, 2], WeightConstraint(sum_to_one=False, upper=1.0))
    assert sol.weights[2] == pytest.approx(1.0, abs=1e-7)
    assert sol.unique is True


def test_no_donors_raises(panel):
    _, A = panel
    with pytest.raises(MlsynthEstimationError, match="no columns"):
        solve_weights(np.empty((A.size, 0)), A)


def test_a_non_finite_entry_raises(panel):
    B, A = panel
    bad = B.copy(); bad[0, 0] = np.nan
    with pytest.raises(MlsynthEstimationError, match="finite"):
        solve_weights(bad, A)


# --------------------------------------------------------------------------
# The constraint and objective records validate themselves
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -0.5, np.inf, np.nan])
def test_a_non_positive_or_infinite_cap_raises(bad):
    with pytest.raises(MlsynthConfigError, match="finite positive cap"):
        WeightConstraint(upper=bad)


def test_a_cap_below_one_without_nonnegativity_is_refused():
    with pytest.raises(MlsynthConfigError, match="does not cover"):
        WeightConstraint(nonneg=False, upper=0.5)


def test_an_unknown_divergence_raises():
    with pytest.raises(MlsynthConfigError, match="Unknown divergence"):
        WeightObjective(divergence="huber")


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan])])
def test_a_degenerate_shrinkage_target_raises(bad):
    with pytest.raises(MlsynthConfigError, match="finite, non-empty"):
        WeightObjective(toward=bad)


def test_a_misshaped_shrinkage_target_raises(panel):
    B, A = panel
    obj = WeightObjective(ridge=1.0, toward=np.ones(3) / 3)
    with pytest.raises(MlsynthConfigError, match="3 entries but there are 6"):
        solve_weights(B, A, objective=obj)


def test_contains_accepts_the_solution_and_rejects_its_violations(panel):
    B, A = panel
    con = WeightConstraint(upper=None)
    w = solve_weights(B, A, con).weights
    assert con.contains(w)
    assert not con.contains(w * 2.0)                                # sum
    assert not WeightConstraint().contains(np.array([1.5, -0.5]))   # sign
    assert not WeightConstraint(sum_to_one=False, upper=0.4).contains(np.array([0.9]))


# --------------------------------------------------------------------------
# The solution object carries its own use
# --------------------------------------------------------------------------
def test_fitted_applies_the_weights_and_the_intercept(panel):
    B, A = panel
    sol = solve_weights(B, A, WeightConstraint(intercept=True))
    assert sol.fitted(B) == pytest.approx(B @ sol.weights + sol.intercept)


def test_the_detail_mapping_names_the_backend_and_the_verdict(panel):
    B, A = panel
    d = solve_weights(B, A).to_dict()
    assert d["solver"] == "simplex:active-set"
    assert d["status"] == "optimal" and d["weights_unique"] is True
    assert d["support_size"] == int((solve_weights(B, A).weights > 0).sum())


def test_the_returned_weights_are_read_only(panel):
    B, A = panel
    with pytest.raises(ValueError):
        solve_weights(B, A).weights[0] = 99.0


# --------------------------------------------------------------------------
# Abadie-Gardeazabal (2003), outcome only. Terrorism in the Basque Country,
# treated 1975, 16 donor regions over 20 pre-periods, no covariates. The
# published weights come from a covariate-matched fit, so the outcome-only
# weights are close and not identical; the ATT is the thing to hold.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def basque():
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "basedata" / "basque_data.csv")
    df = df[df["regionname"] != "Spain (Espana)"]
    df["treat"] = (
        (df["regionname"] == "Basque Country (Pais Vasco)") & (df["year"] >= 1975)
    ).astype(int)
    p = dataprep(df, "regionname", "year", "gdpcap", "treat")
    T0 = int(p["pre_periods"])
    return (
        np.asarray(p["donor_matrix"], float),
        np.asarray(p["y"], float).ravel(),
        T0,
        list(p["donor_names"]),
    )


#: Post-1975 mean gap from the authors' own Synth package (github.com/j-hai/Synth).
BASQUE_ATT = -0.6996


def _att(D, y, T0, sol):
    return float(np.mean(y[T0:] - sol.fitted(D)[T0:]))


def test_basque_simplex_recovers_the_published_donors_and_att(basque):
    D, y, T0, names = basque
    sol = solve_weights(D[:T0], y[:T0])
    carried = {names[j]: round(float(sol.weights[j]), 3) for j in sol.support}
    assert carried["Cataluna"] == pytest.approx(0.826, abs=0.01)
    assert carried["Madrid (Comunidad De)"] == pytest.approx(0.168, abs=0.01)
    assert _att(D, y, T0, sol) == pytest.approx(BASQUE_ATT, abs=0.02)
    assert sol.unique and sol.status == "optimal" and sol.kkt_residual < 1e-12


def test_basque_leaves_thirteen_donors_at_exactly_zero(basque):
    D, y, T0, _ = basque
    sol = solve_weights(D[:T0], y[:T0])
    assert sol.support.size == 3
    assert np.all(sol.weights[sol.weights < 1e-8] == 0.0)


def test_basque_unconstrained_fits_the_pre_period_better_and_the_att_worse(basque):
    """Sixteen donors over twenty pre-periods nearly saturate the design, so
    dropping the hull buys an almost exact pre-fit with no out-of-sample content:
    the ATT flips sign. Amjad's Theorem 4.2.1 carries a rank condition alongside
    the span condition, and this is what its failure looks like."""
    D, y, T0, _ = basque
    sx = solve_weights(D[:T0], y[:T0])
    fr = solve_weights(D[:T0], y[:T0], WeightConstraint(nonneg=False, sum_to_one=False))
    assert fr.objective < sx.objective / 100.0
    assert _att(D, y, T0, fr) > 1.0
    assert abs(_att(D, y, T0, sx) - BASQUE_ATT) < abs(_att(D, y, T0, fr) - BASQUE_ATT)


def test_basque_intercept_improves_the_pre_fit_and_moves_the_att_away(basque):
    """A free intercept is not free: it takes a 0.59 level shift on a series
    near 5, halves Cataluna's weight, and pushes the ATT past the reference."""
    D, y, T0, _ = basque
    sol = solve_weights(D[:T0], y[:T0], WeightConstraint(intercept=True))
    plain = solve_weights(D[:T0], y[:T0])
    assert sol.objective < plain.objective
    assert sol.intercept > 0.5
    assert abs(_att(D, y, T0, sol) - BASQUE_ATT) > abs(_att(D, y, T0, plain) - BASQUE_ATT)


def test_basque_solves_identically_at_a_million_times_scale(basque):
    """CLARABEL declares this same simplex infeasible at 1e6 and leaves a
    2.9e-4 KKT residual at unit scale, with no weight exactly zero."""
    D, y, T0, _ = basque
    base = solve_weights(D[:T0], y[:T0])
    big = solve_weights(D[:T0] * 1e6, y[:T0] * 1e6)
    assert big.status == "optimal"
    assert big.weights == pytest.approx(base.weights, abs=1e-9)


def test_the_origin_is_a_unique_cone_optimum_when_every_donor_is_rejected(panel):
    """A target pointing away from every donor puts the cone's minimiser at the
    origin, with no coordinate free to move and the reduced gradient strictly
    positive everywhere. Nothing can shift, so it is unique."""
    B, _ = panel
    sol = solve_weights(B, -B.sum(axis=1), WeightConstraint(sum_to_one=False))
    assert np.all(sol.weights == 0.0)
    assert sol.support.size == 0
    assert sol.unique is True

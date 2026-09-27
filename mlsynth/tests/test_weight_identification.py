"""Whether a continuum in the weights reaches the quantity being estimated.

`unique` answers a question about the pre-period program. The ATT is a
post-period quantity, and the two come apart: a continuum matters only when the
post-period design fails to annihilate it. Duplicated donors are the case where
it does not matter -- weight moves between the twins and the counterfactual
never notices -- and donors collinear only before treatment are the case where
it decides the answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.weights import WeightConstraint, solve_weights


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
    return np.asarray(p["donor_matrix"], float), np.asarray(p["y"], float).ravel(), T0


@pytest.fixture
def twins_throughout(basque):
    """A donor duplicated in every period: collinear before and after."""
    D, y, T0 = basque
    lead = int(np.argmax(solve_weights(D[:T0], y[:T0]).weights))
    Dd = np.column_stack([D, D[:, lead]])
    return Dd[:T0], y[:T0], Dd[T0:]


@pytest.fixture
def twins_before_only():
    """Two donors identical before treatment that separate after it."""
    rng = np.random.default_rng(2)
    T0, T2, J = 12, 10, 4
    panel = rng.normal(size=(T0 + T2, J)) * 2 + 10
    panel[:T0, 1] = panel[:T0, 0]
    y = panel[:T0] @ np.array([0.3, 0.2, 0.3, 0.2]) + rng.normal(scale=0.01, size=T0)
    return panel[:T0], y, panel[T0:]


# --------------------------------------------------------------------------
# The free directions themselves
# --------------------------------------------------------------------------
def test_a_unique_solution_has_no_free_directions(basque):
    D, y, T0 = basque
    sol = solve_weights(D[:T0], y[:T0])
    assert sol.unique is True
    assert sol.free_directions.shape == (D.shape[1], 0)
    assert sol.free_intercepts.shape == (0,)


def test_unique_is_exactly_the_absence_of_a_free_direction(twins_throughout):
    B, A, _ = twins_throughout
    sol = solve_weights(B, A)
    assert sol.unique is False
    assert sol.free_directions.shape[1] >= 1


def test_a_free_direction_leaves_the_pre_period_fit_alone(twins_throughout):
    """That is what makes it free: the objective does not move along it."""
    B, A, _ = twins_throughout
    sol = solve_weights(B, A)
    d = sol.free_directions[:, 0]
    assert np.abs(B @ d).max() < 1e-9 * np.abs(B).max()


def test_a_free_direction_stays_feasible(twins_throughout):
    B, A, _ = twins_throughout
    sol = solve_weights(B, A)
    d = sol.free_directions[:, 0]
    assert abs(float(d.sum())) < 1e-9                 # stays on the hyperplane
    step = 0.05 * np.sign(d[np.argmax(np.abs(d))])
    moved = np.array(sol.weights) + step * d
    if moved.min() < 0:
        moved = np.array(sol.weights) - step * d      # the other sign is feasible
    assert moved.min() >= -1e-9


# --------------------------------------------------------------------------
# The question the ATT actually asks
# --------------------------------------------------------------------------
def test_duplicated_donors_leave_the_counterfactual_identified(twins_throughout):
    """The continuum is real and the ATT is unaffected: the post-period design
    annihilates the same direction the pre-period one does."""
    B, A, B_post = twins_throughout
    sol = solve_weights(B, A)
    assert sol.unique is False
    assert sol.identifies(B_post) is True


def test_collinearity_only_before_treatment_leaves_the_att_undetermined(twins_before_only):
    B, A, B_post = twins_before_only
    sol = solve_weights(B, A)
    assert sol.unique is False
    assert sol.identifies(B_post) is False


def test_the_verdict_matches_moving_along_the_direction(twins_before_only):
    """`identifies` is a claim about counterfactuals; check it against them."""
    B, A, B_post = twins_before_only
    sol = solve_weights(B, A)
    w = np.array(sol.weights)
    d = sol.free_directions[:, 0]
    step = 0.15 if (w + 0.15 * d).min() >= 0 else -0.15
    alt = w + step * d
    assert alt.min() >= -1e-9 and abs(alt.sum() - 1.0) < 1e-9
    assert B @ alt == pytest.approx(B @ w, abs=1e-10)                 # same fit
    assert np.abs(B_post @ alt - B_post @ w).max() > 0.5             # different ATT


def test_a_unique_solution_identifies_everything(basque):
    """One minimiser, so no linear functional of it can move."""
    D, y, T0 = basque
    sol = solve_weights(D[:T0], y[:T0])
    assert sol.identifies(D[T0:]) is True
    assert sol.identifies(np.random.default_rng(0).normal(size=(5, D.shape[1]))) is True


def test_identification_is_checked_against_the_matrix_it_is_given(twins_before_only):
    """The pre-period design always annihilates its own free directions, so
    asking about it is the question `unique` already answered."""
    B, A, _ = twins_before_only
    sol = solve_weights(B, A)
    assert sol.identifies(B) is True


# --------------------------------------------------------------------------
# The intercept coordinate travels with the directions
# --------------------------------------------------------------------------
def test_the_intercept_component_is_carried(twins_before_only):
    B, A, B_post = twins_before_only
    sol = solve_weights(B, A, WeightConstraint(intercept=True))
    assert sol.free_intercepts.shape[0] == sol.free_directions.shape[1]
    if sol.free_directions.shape[1]:
        shift = B_post @ sol.free_directions + sol.free_intercepts
        expected = bool(np.abs(shift).max() <= 1e-8 * np.abs(B_post).max())
        assert sol.identifies(B_post) is expected


def test_a_level_shifted_copy_of_a_donor_is_the_same_donor(basque):
    """Once the intercept is free, a donor and that donor plus a constant are
    interchangeable: weight moves between them and the intercept takes up the
    difference. The free direction therefore has a nonzero intercept component,
    and the counterfactual is identified only because that component cancels the
    donor part. Reading the direction without it inverts the verdict."""
    D, y, T0 = basque
    lead = int(np.argmax(solve_weights(D[:T0], y[:T0]).weights))
    Dc = np.column_stack([D, D[:, lead] + 5.0])
    sol = solve_weights(Dc[:T0], y[:T0], WeightConstraint(intercept=True))

    assert sol.unique is False
    assert abs(float(sol.free_intercepts[0])) > 0.5          # the level moves
    d = sol.free_directions[:, 0]
    assert d[lead] * d[-1] < 0                               # and it trades the twins

    assert sol.identifies(Dc[T0:]) is True
    # The donor part alone is far from zero; it is the intercept that cancels it.
    assert np.abs(Dc[T0:] @ sol.free_directions).max() > 0.5


def test_a_ridge_penalty_leaves_nothing_free(basque):
    from mlsynth.utils.weights import WeightObjective
    D, y, T0 = basque
    lead = int(np.argmax(solve_weights(D[:T0], y[:T0]).weights))
    Dd = np.column_stack([D, D[:, lead]])
    sol = solve_weights(Dd[:T0], y[:T0], objective=WeightObjective(ridge=1e-3))
    assert sol.unique is True
    assert sol.free_directions.shape[1] == 0
    assert sol.identifies(Dd[T0:]) is True


def test_a_misshaped_matrix_raises(basque):
    from mlsynth.exceptions import MlsynthEstimationError
    D, y, T0 = basque
    sol = solve_weights(D[:T0], y[:T0])
    with pytest.raises(MlsynthEstimationError, match="columns"):
        sol.identifies(D[T0:, :3])

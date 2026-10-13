"""Abadie and L'Hour's penalised SCM on the shared weight solver.

``min ||X1 - X0 w||^2 + lam * (d2' w)`` over the simplex, with ``d2`` the
squared distance from the treated unit to each donor. On the non-negative
orthant that penalty is linear, so the program is the weighted non-negative
lasso and ``solve_simplex_qp`` carries it through its ``linear`` argument.

The method exists to produce a sparse, interpretable donor set: the penalty
prices each donor by how far it sits from the treated unit. An interior-point
solver never returns an exact zero, so the support it reports is a thresholding
decision. On Proposition 99 at ``lam = 1e-3`` it reported all 38 donors as
carrying weight.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.penalized import penalized_weights
from mlsynth.utils.weights import WeightConstraint, WeightObjective, kkt_residual

GRID = [1e-6, 1e-3, 0.1, 1.0, 10.0]


def _panel(name):
    import pandas as pd
    from pathlib import Path
    from mlsynth.utils.datautils import dataprep
    root = Path(__file__).resolve().parents[2]
    if name == "prop99":
        df = pd.read_csv(root / "basedata" / "P99data.csv")
        u, t, o, tr, yr = "state", "year", "cigsale", "California", 1989
    else:
        df = pd.read_csv(root / "basedata" / "basque_data.csv")
        df = df[df["regionname"] != "Spain (Espana)"]
        u, t, o = "regionname", "year", "gdpcap"
        tr, yr = "Basque Country (Pais Vasco)", 1975
    df["treat"] = ((df[u] == tr) & (df[t] >= yr)).astype(int)
    p = dataprep(df, u, t, o, "treat")
    T0 = int(p["pre_periods"])
    return np.asarray(p["y"], float).ravel()[:T0], np.asarray(p["donor_matrix"], float)[:T0]


@pytest.fixture(scope="module")
def prop99():
    return _panel("prop99")


@pytest.fixture(scope="module")
def basque():
    return _panel("basque")


def _program(X1, X0):
    """The normalised program penalized_weights actually solves."""
    s = float(max(np.abs(X0).max(), np.abs(X1).max()))
    x1, x0 = X1 / s, X0 / s
    return x1, x0, np.sum((x1[:, None] - x0) ** 2, axis=0)


def _obj(x1, x0, d2, lam, w):
    return float(np.sum((x1 - x0 @ w) ** 2) + lam * (d2 @ w))


def _cvxpy(x1, x0, d2, lam):
    import cvxpy as cp
    w = cp.Variable(x0.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(x1 - x0 @ w) + float(lam) * (d2 @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.asarray(w.value, float).ravel()


# --------------------------------------------------------------------------
# Differential against the program it replaces
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lam", GRID)
@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_it_matches_the_cvxpy_program(request, panel, lam):
    X1, X0 = request.getfixturevalue(panel)
    x1, x0, d2 = _program(X1, X0)
    got = penalized_weights(X1, X0, lam)
    ref = _cvxpy(x1, x0, d2, lam)
    assert _obj(x1, x0, d2, lam, got) <= _obj(x1, x0, d2, lam, ref) + 1e-9


@pytest.mark.parametrize("lam", GRID)
@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_certificate_holds_with_the_penalty(request, panel, lam):
    X1, X0 = request.getfixturevalue(panel)
    x1, x0, d2 = _program(X1, X0)
    w = penalized_weights(X1, X0, lam)
    res = kkt_residual(x0, x1, w, 0.0, WeightConstraint(),
                       WeightObjective(linear=lam * d2))
    assert res < 1e-8


# --------------------------------------------------------------------------
# The support becomes a fact instead of a threshold
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lam", GRID)
def test_off_support_donors_come_back_at_exactly_zero(prop99, lam):
    w = penalized_weights(*prop99, lam)
    off = w[w < 1e-9]
    assert off.size > 0
    assert np.all(off == 0.0)


def test_the_reported_support_is_no_longer_the_whole_pool(prop99):
    """At lam=1e-3 the interior-point solver reported all 38 donors carrying
    weight. The penalty exists to prevent exactly that."""
    w = penalized_weights(*prop99, 1e-3)
    assert int((w > 0).sum()) < 38


# --------------------------------------------------------------------------
# The penalty does what the method says it does
# --------------------------------------------------------------------------
def test_a_heavier_penalty_moves_weight_towards_the_nearer_donors(prop99):
    X1, X0 = prop99
    _, _, d2 = _program(X1, X0)
    light = penalized_weights(X1, X0, 1e-6)
    heavy = penalized_weights(X1, X0, 10.0)
    assert float(d2 @ heavy) < float(d2 @ light)


def test_a_dominating_penalty_selects_the_nearest_donor(prop99):
    X1, X0 = prop99
    _, _, d2 = _program(X1, X0)
    w = penalized_weights(X1, X0, 1e6)
    assert int(np.argmax(w)) == int(np.argmin(d2))
    assert w.max() > 0.99


@pytest.mark.parametrize("lam", GRID)
@pytest.mark.parametrize("panel", ["prop99", "basque"])
def test_the_weights_stay_on_the_simplex(request, panel, lam):
    w = penalized_weights(*request.getfixturevalue(panel), lam)
    assert w.min() >= 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_a_zero_penalty_is_the_unpenalised_fit(prop99):
    from mlsynth.utils.weights import solve_weights
    X1, X0 = prop99
    x1, x0, _ = _program(X1, X0)
    assert penalized_weights(X1, X0, 0.0) == pytest.approx(
        np.array(solve_weights(x0, x1).weights), abs=1e-8)


# --------------------------------------------------------------------------
# Scale, which is what the normalisation in the function is for
# --------------------------------------------------------------------------
def test_the_fit_is_equivariant_to_a_common_rescaling(prop99):
    """penalized_weights normalises by max|X| before solving, so the answer
    must not move when the panel is handed over in different units."""
    X1, X0 = prop99
    base = penalized_weights(X1, X0, 0.1)
    for f in (1e-3, 1e3, 1e6):
        assert penalized_weights(X1 * f, X0 * f, 0.1) == pytest.approx(base, abs=1e-7)


def test_the_migrated_site_no_longer_builds_a_cvxpy_problem():
    """`penalized_weights` is the site with callers and it is off cvxpy.
    `_simplex_qp` is not: it takes the Gram form, whose linear term carries the
    data fit and not only the penalty, and it has no caller in the library."""
    import inspect
    from mlsynth.utils.bilevel.penalized import penalized_weights, _simplex_qp
    assert "cp.Problem" not in inspect.getsource(penalized_weights)
    assert "cp.Problem" in inspect.getsource(_simplex_qp)

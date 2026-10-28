"""SparseSC's inner simplex QP is centred, certified and returns exact zeros (#934).

``inner.solve_w`` solved ``min w' X0'VX0 w - 2 X1'VX0 w`` with Clarabel in Gram form,
uncentred, with a trace-scaled ridge, and accepted ``Solved`` or ``AlmostSolved``
without checking the answer. On 20,000 solves sampled from a Basque fit (13
predictors, treated from 1975) its KKT residual had median 3.7e-6 and maximum 0.38,
199,478 weights sat in (0, 1e-6), and adding a constant to every predictor moved the
weights by up to 0.78 (agents_solvers.md, rules 2, 4, 5 and 6).

The solve now runs on the centred differences ``V^(1/2) (X0 - X1 1')`` against a zero
target through the active-set ``solve_simplex_qp``: on the simplex the problem is the
same, the level drops out, no Gram matrix is formed, and inactive weights are exact
zeros. Each answer is certified by ``kkt_residual`` on that design and refused above
``KKT_TOL``. On the same 20,000 solves the residual has median 2.4e-17 and maximum
2.8e-10, and a solve warm-started from the previous answer takes 106 microseconds
against Clarabel's 410.

A face of minimisers, about 6 percent of a Basque sweep's solves, does not identify
``w``, and the outer search optimises whichever member it is handed: least norm moved
the Basque fit to one predictor and an ATT of -0.526. The face is resolved by Abadie
and L'Hour's pairwise penalty in its vanishing limit, an LP over the face, with
identical donors sharing their weight; the Basque fit then returns ATT -0.6726, as
with Clarabel.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import mlsynth.utils.sparse_sc_helpers.inner as inner
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.solvers.active_set import solve_simplex_qp, solve_simplex_qp_least_norm
from mlsynth.utils.sparse_sc_helpers.inner import WarmStart, solve_w
from mlsynth.utils.sparse_sc_helpers.objective import outer_loss_and_grad
from mlsynth.utils.weights.solve import KKT_TOL, kkt_residual
from mlsynth.utils.weights.spec import WeightConstraint, WeightObjective


def _problem(seed, P=8, N=12, inside=True):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(P, N))
    X1 = X0 @ rng.dirichlet(np.ones(N)) if inside else rng.normal(size=P) * 2.0
    X1 = X1 + 0.1 * rng.normal(size=P)
    v = np.concatenate([[1.0], rng.uniform(0.0, 2.0, size=P - 1)])
    return v, X1, X0


def _kkt(v, X1, X0, w):
    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    return kkt_residual(B, np.zeros(B.shape[0]), w, 0.0, WeightConstraint(), WeightObjective())


# --------------------------------------------------------------------------- smoke
def test_a_solve_returns_a_certified_simplex_point():
    v, X1, X0 = _problem(0)
    w = solve_w(v, X1, X0)
    assert w.shape == (X0.shape[1],) and w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    assert _kkt(v, X1, X0, w) <= KKT_TOL


# --------------------------------------------------------------------------- unit
def test_inactive_weights_are_exact_zeros():
    v, X1, X0 = _problem(1, inside=False)
    w = solve_w(v, X1, X0)
    assert np.any(w == 0.0)
    assert not np.any((w > 0.0) & (w < 1e-6))


def test_a_unique_optimum_is_the_active_set_solve_of_the_centred_design():
    v, X1, X0 = _problem(2, P=10, N=6, inside=False)
    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    np.testing.assert_array_equal(solve_w(v, X1, X0), solve_simplex_qp(B, np.zeros(B.shape[0])))


def test_a_constant_added_to_a_predictor_does_not_move_the_weights():
    """The issue's table: 1e2, 1e4 and 1e6 standard deviations moved Clarabel's w by up to 0.78."""
    v, X1, X0 = _problem(3)
    w0 = solve_w(v, X1, X0)
    for c in (1e2, 1e4, 1e6):
        shift = c * np.linspace(1.0, 2.0, X0.shape[0])[:, None]
        np.testing.assert_allclose(solve_w(v, X1 + shift[:, 0], X0 + shift), w0, atol=1e-6)


def test_the_warm_start_reaches_the_solver(monkeypatch):
    seen = []
    real = inner.solve_simplex_qp

    def spy(*a, **k):
        seen.append(k.get("warm_start"))
        return real(*a, **k)
    monkeypatch.setattr(inner, "solve_simplex_qp", spy)
    v, X1, X0 = _problem(4)
    ws = np.full(X0.shape[1], 1.0 / X0.shape[1])
    solve_w(v, X1, X0, warm_start=ws)
    assert seen and np.array_equal(seen[-1], ws)


def test_a_warm_start_holder_carries_the_previous_answer():
    v, X1, X0 = _problem(5)
    warm = WarmStart()
    assert warm.w is None
    w = solve_w(v, X1, X0, warm=warm)
    assert np.array_equal(warm.w, w)


def test_the_holders_answer_is_the_next_start(monkeypatch):
    seen = []
    real = inner.solve_simplex_qp

    def spy(*a, **k):
        seen.append(k.get("warm_start"))
        return real(*a, **k)
    monkeypatch.setattr(inner, "solve_simplex_qp", spy)
    v, X1, X0 = _problem(15)
    warm = WarmStart()
    w1 = solve_w(v, X1, X0, warm=warm)
    solve_w(1.01 * v, X1, X0, warm=warm)
    assert seen[0] is None and np.array_equal(seen[1], w1)


def test_the_optimiser_warm_starts_its_inner_solves(monkeypatch):
    """Through ``_minimize_outer``: after the first inner solve, every solve has a start."""
    import mlsynth.utils.sparse_sc_helpers.optimization as optimization
    seen = []
    real = inner.solve_simplex_qp

    def spy(*a, **k):
        seen.append(k.get("warm_start") is not None)
        return real(*a, **k)
    monkeypatch.setattr(inner, "solve_simplex_qp", spy)
    v, X1, X0 = _problem(16, P=5, N=7)
    rng = np.random.default_rng(16)
    Z0 = rng.normal(size=(6, X0.shape[1])); Z1 = rng.normal(size=6)
    for grad in (True, False):
        seen.clear()
        optimization._minimize_outer(x0=v[1:], X1=X1, X0=X0, Z1_outer=Z1, Z0_outer=Z0, lam=0.01,
                                     solver=None, bounds=[(0.0, None)] * (v.size - 1),
                                     max_outer_iter=20, ftol=1e-8, use_analytical_grad=grad)
        assert len(seen) > 2 and not seen[0] and all(seen[1:])


def test_the_gradient_does_not_move_with_a_level():
    """The adjoint's KKT system is centred too: a constant on every predictor leaves it alone."""
    v, X1, X0 = _problem(6, P=6, N=9)
    rng = np.random.default_rng(6)
    Z0 = rng.normal(size=(5, X0.shape[1])); Z1 = Z0 @ rng.dirichlet(np.ones(X0.shape[1]))
    l0, g0 = outer_loss_and_grad(v[1:], X1, X0, Z1, Z0, 0.01)
    shift = 1e4 * np.ones((X0.shape[0], 1))
    l1, g1 = outer_loss_and_grad(v[1:], X1 + shift[:, 0], X0 + shift, Z1, Z0, 0.01)
    assert l1 == pytest.approx(l0, rel=1e-6)
    np.testing.assert_allclose(g1, g0, rtol=1e-4, atol=1e-8 * np.abs(g0).max())


def test_the_gradient_matches_finite_differences():
    v, X1, X0 = _problem(7, P=5, N=7, inside=False)
    rng = np.random.default_rng(7)
    Z0 = rng.normal(size=(6, X0.shape[1])); Z1 = rng.normal(size=6)
    _, g = outer_loss_and_grad(v[1:], X1, X0, Z1, Z0, 0.0)
    h = 1e-6
    for k in range(v.size - 1):
        e = np.zeros(v.size - 1); e[k] = h
        fd = (outer_loss_and_grad(v[1:] + e, X1, X0, Z1, Z0, 0.0)[0]
              - outer_loss_and_grad(v[1:] - e, X1, X0, Z1, Z0, 0.0)[0]) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-3, abs=1e-6)


# --------------------------------------------------------------------------- edge
def test_a_face_resolves_independently_of_donor_order():
    """A face of minimisers: the answer is a function of the panel, not of donor order."""
    v, X1, X0 = _problem(8, P=3, N=10)
    w = solve_w(v, X1, X0)
    perm = np.random.default_rng(8).permutation(X0.shape[1])
    np.testing.assert_allclose(solve_w(v, X1, X0[:, perm]), w[perm], atol=1e-6)


def test_a_single_donor_gets_all_the_weight():
    v, X1, X0 = _problem(9, N=1)
    np.testing.assert_array_equal(solve_w(v, X1, X0), [1.0])


def test_a_treated_unit_equal_to_a_donor_selects_it():
    v, _, X0 = _problem(10)
    w = solve_w(v, X0[:, 3].copy(), X0)
    assert w[3] == pytest.approx(1.0, abs=1e-6)


def test_zero_weight_on_every_predictor_but_an_exact_one():
    """Only the anchor carries weight and it is fitted exactly by two donors: a face, still certified."""
    v, X1, X0 = _problem(11)
    v = np.zeros_like(v); v[0] = 1.0
    X0 = X0.copy(); X0[0, :2] = X1[0]
    w = solve_w(v, X1, X0)
    assert _kkt(v, X1, X0, w) <= KKT_TOL


def test_one_active_donor_pins_w_and_leaves_only_the_penalty_in_the_gradient():
    """A treated unit equal to a donor: w is a vertex, locally constant in v."""
    v, _, X0 = _problem(17, P=5, N=7)
    X1 = X0[:, 2].copy()
    rng = np.random.default_rng(17)
    Z0 = rng.normal(size=(6, X0.shape[1])); Z1 = rng.normal(size=6)
    _, g = outer_loss_and_grad(v[1:], X1, X0, Z1, Z0, 0.03)
    np.testing.assert_allclose(g, 0.03)


def _reference_pairwise(v, X1, X0, w):
    """Brute force: minimise c'w over every minimiser, by an LP in w with the fit pinned."""
    from scipy.optimize import linprog
    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    c = np.sum(B * B, axis=0)
    r = linprog(c / c.max(), A_eq=np.vstack([B, np.ones((1, B.shape[1]))]), b_eq=np.r_[B @ w, 1.0],
                bounds=[(0.0, None)] * B.shape[1], method="highs")
    assert r.status == 0
    return r.x, c


def test_on_a_face_the_closest_donors_are_chosen():
    """The treated unit equals donor 0 and the average of donors 1 and 2: both fit exactly.

    Least norm would split the weight; the pairwise rule (Abadie and L'Hour's penalty in
    its vanishing limit) puts it all on the donor that matches the treated unit itself.
    """
    rng = np.random.default_rng(18)
    P, N = 4, 6
    X0 = rng.normal(size=(P, N))
    X0[:, 0] = 0.5 * (X0[:, 1] + X0[:, 2])
    X1 = X0[:, 0].copy()
    v = np.ones(P)
    w = solve_w(v, X1, X0)
    # the least-norm ridge (1e-8) spreads about that much onto the other donors
    np.testing.assert_allclose(w, np.eye(N)[0], atol=1e-6)
    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    ln = solve_simplex_qp_least_norm(B, np.zeros(P))
    assert ln[0] < 0.99                                      # the rule it replaces did split


@pytest.mark.parametrize("seed", range(6))
def test_on_a_face_the_member_minimises_the_pairwise_discrepancy(seed):
    v, X1, X0 = _problem(100 + seed, P=3, N=10)
    w = solve_w(v, X1, X0)
    ref, c = _reference_pairwise(v, X1, X0, w)
    assert c @ w <= c @ ref * (1.0 + 1e-7) + 1e-12 * c.max()
    assert _kkt(v, X1, X0, w) <= KKT_TOL


def _exact_fit_problem(seed, level=0.0):
    """A treated unit inside the donors' hull, every predictor weighted; odd seeds rescale donors.

    Used to pin witnesses found by direct search against the mutated module.
    """
    r = np.random.default_rng(seed)
    P, N = int(r.integers(1, 4)), int(r.integers(3, 8))
    X0 = r.normal(size=(P, N))
    if seed % 2:
        X0 = X0 * 10.0 ** r.uniform(-3, 3, size=(1, N))
    X0 = X0 + level
    return np.ones(P), X0 @ r.dirichlet(np.ones(N)), X0


@pytest.mark.parametrize("seed, level", [(9675, 0.0), (12, 1e3)])
def test_pinned_faces_reach_the_pairwise_optimum(seed, level):
    """Witnesses found by direct search against the mutated module.

    Seed 9675: discrepancies spanning six orders of magnitude, on which the LP with
    costs divided by their largest entry returned a member that is not the optimum,
    without failing (the weights moved by 0.50). Seed 12 at a level of 1e3: the
    active-set answer leaves a zero-weight donor that can enter at no cost, so a face
    read off the support alone misses the optimum, and an uncentred design turns the
    pairwise cost into the norm of the level instead of the distance to the treated
    unit; each mutation moves the weights by 0.13.
    """
    v, X1, X0 = _exact_fit_problem(seed, level)
    w = solve_w(v, X1, X0)
    ref, c = _reference_pairwise(v, X1, X0, w)
    assert c @ w <= c @ ref * (1.0 + 1e-7) + 1e-12 * c.max()
    assert _kkt(v, X1, X0, w) <= KKT_TOL


def test_identical_donors_share_their_weight_equally():
    """Two donors identical to the treated unit: both cost zero, so the rule has no preference."""
    rng = np.random.default_rng(21)
    X0 = rng.normal(size=(4, 5))
    X0[:, 1] = X0[:, 0]
    v = np.ones(4)
    w = solve_w(v, X0[:, 0].copy(), X0)
    np.testing.assert_allclose(w, [0.5, 0.5, 0.0, 0.0, 0.0], atol=1e-6)   # ridge bias, as above


def test_a_tie_in_the_pairwise_cost_resolves_independently_of_donor_order():
    """Donors 0 and 1 are reflections of each other about the treated unit: same cost, same fit."""
    rng = np.random.default_rng(22)
    X1 = rng.normal(size=3)
    d = rng.normal(size=3)
    X0 = np.column_stack([X1 + d, X1 - d, X1 + 3 * rng.normal(size=3), X1 + 3 * rng.normal(size=3)])
    v = np.ones(3)
    w = solve_w(v, X1, X0)
    np.testing.assert_allclose(w, [0.5, 0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(solve_w(v, X1, X0[:, [1, 0, 3, 2]]), w[[1, 0, 3, 2]], atol=1e-6)


def test_every_donor_equal_to_the_treated_unit_is_still_certified():
    X0 = np.tile(np.arange(1.0, 4.0)[:, None], (1, 4))
    w = solve_w(np.ones(3), X0[:, 0].copy(), X0)
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)


def test_a_unique_optimum_is_left_alone_by_the_tie_break(monkeypatch):
    calls = []
    real = inner.linprog
    monkeypatch.setattr(inner, "linprog", lambda *a, **k: calls.append(1) or real(*a, **k))
    v, X1, X0 = _problem(19, P=10, N=4, inside=False)
    solve_w(v, X1, X0)
    assert not calls


# --------------------------------------------------------------------------- failure
def test_a_certificate_failure_is_raised(monkeypatch):
    v, X1, X0 = _problem(12, inside=False)
    bad = np.zeros(X0.shape[1]); bad[0] = 1.0
    if _kkt(v, X1, X0, bad) <= KKT_TOL:                       # pragma: no cover - seed chosen so it is not optimal
        pytest.skip("vertex happens to be optimal")
    monkeypatch.setattr(inner, "solve_simplex_qp", lambda *a, **k: bad.copy())
    with pytest.raises(MlsynthEstimationError, match="KKT"):
        solve_w(v, X1, X0)


def test_a_tie_break_lp_failure_is_raised(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(inner, "linprog", lambda *a, **k: SimpleNamespace(status=4, message="numerical difficulties"))
    v, X1, X0 = _problem(20, P=3, N=10)
    with pytest.raises(MlsynthEstimationError, match="tie-break"):
        solve_w(v, X1, X0)


@pytest.mark.parametrize("which", ["X0", "X1"])
def test_non_finite_predictors_are_refused(which):
    v, X1, X0 = _problem(13)
    (X0 if which == "X0" else X1)[1] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        solve_w(v, X1, X0)


@pytest.mark.parametrize("bad", [-0.5, np.nan])
def test_an_invalid_v_is_refused(bad):
    v, X1, X0 = _problem(14)
    v = v.copy(); v[2] = bad
    with pytest.raises(MlsynthEstimationError, match="v"):
        solve_w(v, X1, X0)


# --------------------------------------------------------------------------- property
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), P=st.integers(2, 10), N=st.integers(1, 15),
       k=st.integers(0, 6), inside=st.booleans())
@example(seed=0, P=3, N=10, k=6, inside=True)
def test_property_certified_exact_zeros_and_level_free(seed, P, N, k, inside):
    v, X1, X0 = _problem(seed, P=P, N=N, inside=inside)
    w = solve_w(v, X1, X0)
    assert w.min() >= 0.0 and w.sum() == pytest.approx(1.0)
    assert not np.any((w > 0.0) & (w < 1e-12))
    assert _kkt(v, X1, X0, w) <= KKT_TOL
    c = 10.0 ** k
    ws = solve_w(v, X1 + c, X0 + c)
    # Storing X + c rounds each entry by eps c / 2 (rule 11): the shifted problem is the
    # original one perturbed by that much, and its answer is certified on its own data.
    assert _kkt(v, X1 + c, X0 + c, ws) <= KKT_TOL
    B = np.sqrt(v)[:, None] * (X0 - X1[:, None])
    loss = lambda z: float(np.sum((B @ z) ** 2))
    assert loss(ws) <= loss(w) + 1e-9 * max(loss(w), 1.0) + 64 * (np.finfo(float).eps * c) ** 2 * P * float(v.sum())

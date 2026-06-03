"""Unit tests for the self-contained bilevel SCM solver (Malo et al. 2024)."""

import numpy as np
import pytest

from mlsynth.utils.fscm_helpers.bilevel import (
    BilevelProblem,
    BilevelSolution,
    project_simplex,
    simplex_lstsq,
    solve_bilevel,
    lower_level_weights,
)
from mlsynth.utils.fscm_helpers.bilevel.simplex import mspe


# --------------------------------------------------------------------------- #
# project_simplex
# --------------------------------------------------------------------------- #
def test_project_simplex_basic():
    w = project_simplex(np.array([0.2, 0.5, 0.3]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    # Already on the simplex -> unchanged.
    np.testing.assert_allclose(w, [0.2, 0.5, 0.3], atol=1e-9)


def test_project_simplex_negative_and_outside():
    w = project_simplex(np.array([3.0, -1.0, 0.0]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert w[0] == pytest.approx(1.0)  # dominant coordinate absorbs the mass


def test_project_simplex_singleton():
    assert project_simplex(np.array([5.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# simplex_lstsq
# --------------------------------------------------------------------------- #
def test_simplex_lstsq_matches_known_optimum():
    # Target is an exact convex combination of two columns -> recoverable.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.25, 0.75, 1.0])
    w = simplex_lstsq(A, b)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(w, [0.25, 0.75], atol=1e-4)


def test_simplex_lstsq_optimality_vs_grid():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(15, 4))
    b = rng.normal(size=15)
    w = simplex_lstsq(A, b)
    obj = np.sum((A @ w - b) ** 2)
    # No simplex vertex or random simplex point should beat it.
    for _ in range(200):
        wr = rng.dirichlet(np.ones(4))
        assert obj <= np.sum((A @ wr - b) ** 2) + 1e-6


def test_simplex_lstsq_singleton():
    assert simplex_lstsq(np.array([[2.0], [3.0]]), np.array([1.0, 1.0])) == pytest.approx([1.0])


# --------------------------------------------------------------------------- #
# solve_bilevel
# --------------------------------------------------------------------------- #
def test_bilevel_unconstrained_certificate():
    # Treated outcome is an exact convex combo of donors, and a predictor is
    # matched by that same combination -> Stage 1 certifies optimality.
    Y0 = np.array([[1.0, 3.0], [2.0, 0.0], [0.0, 4.0]])
    w_true = np.array([0.5, 0.5])
    y1 = Y0 @ w_true
    X0 = np.array([[10.0, 20.0]])      # predictor matched by w_true -> X1 = 15
    X1 = X0 @ w_true
    sol = solve_bilevel(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0))
    assert isinstance(sol, BilevelSolution)
    assert sol.stage == "unconstrained"
    assert sol.lower_bound == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(sol.W, w_true, atol=1e-3)


def test_bilevel_upper_loss_at_least_lower_bound():
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(12, 6))
    y1 = rng.normal(size=12)
    X0 = rng.normal(size=(3, 6))
    X1 = rng.normal(size=3)
    sol = solve_bilevel(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0))
    assert sol.upper_loss >= sol.lower_bound - 1e-6
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.V.sum() == pytest.approx(1.0, abs=1e-6)


def test_lower_level_weights_fixed_V_on_simplex():
    rng = np.random.default_rng(7)
    Y0 = rng.normal(size=(10, 5))
    y1 = rng.normal(size=10)
    X0 = rng.normal(size=(2, 5))
    X1 = rng.normal(size=2)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    w = lower_level_weights(prob, np.array([1.0, 0.0]))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-9)


def test_bilevel_deterministic():
    rng = np.random.default_rng(9)
    Y0 = rng.normal(size=(14, 7)); y1 = rng.normal(size=14)
    X0 = rng.normal(size=(4, 7)); X1 = rng.normal(size=4)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s1, s2 = solve_bilevel(prob), solve_bilevel(prob)
    np.testing.assert_allclose(s1.V, s2.V)
    np.testing.assert_allclose(s1.W, s2.W)


# --------------------------------------------------------------------------- #
# method dispatch: malo (default) vs mscmt
# --------------------------------------------------------------------------- #
def test_unknown_method_raises():
    prob = BilevelProblem(
        y1_pre=np.zeros(3), Y0_pre=np.eye(3)[:, :2],
        X1=np.zeros(1), X0=np.zeros((1, 2)),
    )
    with pytest.raises(ValueError, match="malo|mscmt"):
        solve_bilevel(prob, method="not-a-method")


def test_mscmt_backend_on_simplex_and_bounded():
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(12, 6)); y1 = rng.normal(size=12)
    X0 = rng.normal(size=(3, 6)); X1 = rng.normal(size=3)
    sol = solve_bilevel(
        BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0),
        method="mscmt", maxiter=50, seed=0,
    )
    assert isinstance(sol, BilevelSolution)
    assert sol.stage in ("mscmt", "mscmt-feasible")
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.V.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(sol.W >= -1e-9)
    assert sol.upper_loss >= sol.lower_bound - 1e-6


def test_mscmt_matches_malo_on_feasible_certificate():
    # When the unconstrained outcome optimum is predictor-feasible, both
    # backends return the exact global solution and must agree.
    Y0 = np.array([[1.0, 3.0], [2.0, 0.0], [0.0, 4.0]])
    w_true = np.array([0.5, 0.5])
    y1 = Y0 @ w_true
    X0 = np.array([[10.0, 20.0]])
    X1 = X0 @ w_true
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s_malo = solve_bilevel(prob, method="malo")
    s_msc = solve_bilevel(prob, method="mscmt")
    assert s_malo.stage == "unconstrained"
    assert s_msc.stage == "mscmt-feasible"
    np.testing.assert_allclose(s_malo.W, s_msc.W, atol=1e-3)


def test_mscmt_deterministic_with_seed():
    rng = np.random.default_rng(11)
    Y0 = rng.normal(size=(14, 7)); y1 = rng.normal(size=14)
    X0 = rng.normal(size=(4, 7)); X1 = rng.normal(size=4)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s1 = solve_bilevel(prob, method="mscmt", seed=3, maxiter=40)
    s2 = solve_bilevel(prob, method="mscmt", seed=3, maxiter=40)
    np.testing.assert_allclose(s1.W, s2.W)
    np.testing.assert_allclose(s1.V, s2.V)


# --------------------------------------------------------------------------- #
# penalized backend (Abadie & L'Hour 2021)
# --------------------------------------------------------------------------- #
from mlsynth.utils.fscm_helpers.bilevel import (  # noqa: E402
    bias_corrected_gaps, solve_penalized,
)


def _toy_prob(seed=0, T=18, J=8, K=3):
    rng = np.random.default_rng(seed)
    Y0 = np.cumsum(rng.normal(size=(T, J)), axis=0) + 5.0
    w = rng.dirichlet(np.ones(J))
    y1 = Y0 @ w + rng.normal(scale=0.1, size=T)
    X0 = rng.normal(size=(K, J))
    X1 = X0 @ w
    return BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)


def test_penalized_on_simplex_and_unique():
    prob = _toy_prob()
    s1 = solve_bilevel(prob, method="penalized", lam=0.1)
    s2 = solve_bilevel(prob, method="penalized", lam=0.1)
    assert s1.stage == "penalized"
    assert s1.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(s1.W >= -1e-9)
    # lambda > 0 => unique => deterministic across runs (Theorem 1).
    np.testing.assert_allclose(s1.W, s2.W, atol=1e-7)


def test_penalized_larger_lambda_is_sparser():
    prob = _toy_prob(seed=2)
    nnz_small = solve_bilevel(prob, method="penalized", lam=1e-3).metadata["n_nonzero"]
    nnz_large = solve_bilevel(prob, method="penalized", lam=10.0).metadata["n_nonzero"]
    assert nnz_large <= nnz_small


def test_penalized_cv_selectors_run():
    prob = _toy_prob(seed=4)
    for cv in ("holdout", "loo"):
        s = solve_bilevel(prob, method="penalized", lam="cv", cv=cv)
        assert s.metadata["lambda_selected_by"] == cv
        assert s.metadata["lambda"] > 0
        assert s.W.sum() == pytest.approx(1.0, abs=1e-6)


def test_penalized_unknown_cv_raises():
    prob = _toy_prob()
    with pytest.raises(ValueError, match="holdout|loo"):
        solve_bilevel(prob, method="penalized", lam="cv", cv="nope")


def test_bias_correction_removes_bias_when_covariates_informative():
    # Outcome is generated *from* the covariates -> the bias correction should
    # collapse the gap of a deliberately imbalanced weight vector toward 0.
    rng = np.random.default_rng(0)
    J, T, K = 12, 20, 3
    X0 = rng.normal(size=(K, J))
    beta = np.array([3.0, -2.0, 1.0])
    Y0 = (X0.T @ beta)[None, :] + rng.normal(scale=0.05, size=(T, J))
    X1 = rng.normal(size=K)
    y1 = (X1 @ beta) + rng.normal(scale=0.05, size=T)
    w = np.full(J, 1.0 / J)                       # imbalanced on purpose
    raw = np.abs(y1 - Y0 @ w).mean()
    bc = np.abs(bias_corrected_gaps(w, X1, X0, y1, Y0, ridge=1e-3)).mean()
    assert bc < 0.5 * raw                          # correction removes most of the bias


# --------------------------------------------------------------------------- #
# Full line/branch coverage of the bilevel package.
#
# The tests below deliberately drive the remaining edge-case branches that the
# behavioural tests above do not reach: power-iteration degeneracies, singleton
# QPs, the mscmt single-predictor path, the Tykhonov improvement step, the
# penalized validation guards, and the small dataclass properties.
# --------------------------------------------------------------------------- #
import numpy as _np  # noqa: E402

from mlsynth.utils.fscm_helpers.bilevel.simplex import (  # noqa: E402
    _lipschitz_constant,
    project_simplex as _project_simplex,
    _EPS as _SIMPLEX_EPS,
)
from mlsynth.utils.fscm_helpers.bilevel.penalized import (  # noqa: E402
    _spectral_radius,
    _simplex_qp,
    _build_block,
    solve_penalized as _solve_penalized,
)
from mlsynth.utils.fscm_helpers.bilevel.stages import (  # noqa: E402
    tykhonov_refine,
    corner_solutions,
    unconstrained_feasibility,
)
from mlsynth.utils.fscm_helpers.bilevel.structure import (  # noqa: E402
    BilevelProblem as _BP,
    BilevelSolution as _BS,
)


class _ZeroRng:
    """A drop-in ``Generator`` whose ``normal`` returns an all-zero vector.

    Used to force the ``||x|| < eps`` degeneracy at the top of the power
    iterations (otherwise unreachable with a real RNG).
    """

    def normal(self, size):
        return _np.zeros(size)


# --------------------------------------------------------------------------- #
# simplex._lipschitz_constant  -- both power-iteration guards
# --------------------------------------------------------------------------- #
def test_lipschitz_zero_initial_vector_returns_one(monkeypatch):
    # ||x|| < eps on the random init  ->  early return 1.0  (simplex.py:49)
    monkeypatch.setattr(_np.random, "default_rng", lambda seed=0: _ZeroRng())
    assert _lipschitz_constant(_np.eye(3)) == pytest.approx(1.0)


def test_lipschitz_zero_matrix_returns_eps():
    # A is all zeros  ->  A'A x == 0  ->  ||y|| < eps  ->  return _EPS  (simplex.py:56)
    val = _lipschitz_constant(_np.zeros((3, 2)))
    assert val == pytest.approx(_SIMPLEX_EPS)


def test_lipschitz_known_value():
    # For A = sqrt(2) * I, A'A = 2 I, lambda_max = 2, so L = 2*2 + eps ~= 4.
    A = _np.sqrt(2.0) * _np.eye(4)
    assert _lipschitz_constant(A) == pytest.approx(4.0, abs=1e-6)


def test_project_simplex_nonunit_radius():
    # z != 1.0 branch: the projection should sum to the requested radius.
    w = _project_simplex(_np.array([0.2, 0.5, 0.3]), z=2.0)
    assert w.sum() == pytest.approx(2.0)
    assert _np.all(w >= 0)


# --------------------------------------------------------------------------- #
# penalized._spectral_radius / _simplex_qp -- guards & singleton
# --------------------------------------------------------------------------- #
def test_spectral_radius_zero_initial_vector_returns_one(monkeypatch):
    # penalized.py:59
    import mlsynth.utils.fscm_helpers.bilevel.penalized as _pen
    monkeypatch.setattr(_pen.np.random, "default_rng", lambda seed=0: _ZeroRng())
    assert _spectral_radius(_np.eye(3)) == pytest.approx(1.0)


def test_spectral_radius_zero_matrix_returns_eps():
    # Q is all zeros  ->  ||y|| < eps  ->  return _EPS  (penalized.py:66)
    assert _spectral_radius(_np.zeros((3, 3))) == pytest.approx(_SIMPLEX_EPS)


def test_spectral_radius_known_value():
    # Largest eigenvalue of diag(1, 5, 2) is 5.
    assert _spectral_radius(_np.diag([1.0, 5.0, 2.0])) == pytest.approx(5.0, abs=1e-6)


def test_simplex_qp_singleton():
    # n == 1 short-circuit  (penalized.py:80)
    w = _simplex_qp(_np.array([[3.0]]), _np.array([-1.0]))
    np.testing.assert_allclose(w, [1.0])


# --------------------------------------------------------------------------- #
# mscmt single-predictor (K == 1) path
# --------------------------------------------------------------------------- #
def test_mscmt_single_predictor_path():
    # One predictor whose target is unmatchable by the outcome optimum, so the
    # feasibility certificate fails and the K == 1 branch runs (mscmt.py:116-124).
    Y0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y1 = np.array([0.4, 0.6, 1.0])               # interior outcome optimum
    X0 = np.array([[1.0, 2.0]])                  # single predictor
    X1 = np.array([100.0])                       # unreachable on the simplex
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    sol = solve_bilevel(prob, method="mscmt")
    assert sol.stage == "mscmt"
    assert sol.V.shape == (1,)
    assert sol.V[0] == pytest.approx(1.0)
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.metadata["backend"] == "mscmt"


# --------------------------------------------------------------------------- #
# stages.tykhonov_refine -- the improvement branch (stages.py:128)
# --------------------------------------------------------------------------- #
def test_tykhonov_refine_improves_from_corner():
    # Seed 25 / this geometry is one where a projected-gradient step on V
    # strictly lowers the outcome loss, exercising the update branch.
    rng = np.random.default_rng(25)
    J, T, K = 5, 10, 3
    Y0 = rng.normal(size=(T, J)); y1 = rng.normal(size=T)
    X0 = rng.normal(size=(K, J)); X1 = rng.normal(size=K)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    V0, _, loss0, _ = corner_solutions(prob)
    V, W, loss, total = tykhonov_refine(
        prob, V0, outer_iters=8, inner_iters=12, lr=0.2, fd_step=1e-2,
    )
    assert total > 0                              # at least one accepted step
    assert not np.allclose(V, V0)                 # V actually moved off the corner
    assert loss <= loss0 + 1e-9                   # and it did not get worse
    assert V.sum() == pytest.approx(1.0, abs=1e-6)
    assert W.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# malo solver -- corner stage (no certificate) and the refine path
# --------------------------------------------------------------------------- #
def test_malo_corner_stage_when_not_certifiable():
    # Multi-predictor random problem: certificate fails, so the solver lands on
    # a corner / refined solution rather than the unconstrained certificate.
    rng = np.random.default_rng(25)
    Y0 = rng.normal(size=(10, 5)); y1 = rng.normal(size=10)
    X0 = rng.normal(size=(3, 5)); X1 = rng.normal(size=3)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    _, _, _, is_opt = unconstrained_feasibility(prob)
    assert is_opt is False                        # confirms we exercise the non-cert path
    sol = solve_bilevel(prob, method="malo")
    assert sol.stage in ("corner", "tykhonov")
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.upper_loss >= sol.lower_bound - 1e-6


# --------------------------------------------------------------------------- #
# penalized backend -- validation guards & remaining option branches
# --------------------------------------------------------------------------- #
def test_penalized_requires_outcomes_or_predictors():
    # Neither matching block selected  ->  ValueError  (penalized.py:248-249)
    prob = _toy_prob()
    with pytest.raises(ValueError, match="outcomes and/or predictors"):
        solve_bilevel(prob, method="penalized",
                      use_outcomes=False, use_predictors=False)


def test_penalized_fixed_float_lambda_path():
    # Numeric lam (not a CV string) takes the explicit-float branch.
    prob = _toy_prob(seed=1)
    sol = solve_bilevel(prob, method="penalized", lam=0.25)
    assert sol.metadata["lambda"] == pytest.approx(0.25)
    assert sol.metadata["lambda_selected_by"] == "fixed"


def test_penalized_predictors_only():
    # use_outcomes=False but predictors present -> predictor-only matching.
    prob = _toy_prob(seed=3)
    sol = solve_bilevel(prob, method="penalized", lam=0.1,
                        use_outcomes=False, use_predictors=True)
    assert sol.stage == "penalized"
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)


def test_penalized_outcomes_only_no_predictors():
    # A problem with zero predictors: lower_loss falls into the n_predictors==0
    # branch and matching is on outcomes alone.
    rng = np.random.default_rng(6)
    Y0 = np.cumsum(rng.normal(size=(16, 7)), axis=0) + 5.0
    w = rng.dirichlet(np.ones(7))
    y1 = Y0 @ w + rng.normal(scale=0.1, size=16)
    prob = _BP(y1_pre=y1, Y0_pre=Y0,
               X1=np.zeros(0), X0=np.zeros((0, 7)))
    assert prob.n_predictors == 0
    sol = solve_bilevel(prob, method="penalized", lam=0.05,
                        use_outcomes=True, use_predictors=False)
    assert sol.stage == "penalized"
    assert sol.lower_loss == pytest.approx(0.0)
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)


def test_penalized_custom_lambda_grid_for_cv():
    # Supplying an explicit lam_grid exercises the non-default grid path.
    prob = _toy_prob(seed=8)
    grid = np.array([1e-3, 1e-2, 1e-1, 1.0])
    sol = solve_bilevel(prob, method="penalized", lam="cv", cv="holdout",
                        lam_grid=grid)
    assert sol.metadata["lambda"] in grid
    assert len(sol.metadata["cv_curve"]) == len(grid)


def test_build_block_shapes():
    # Directly exercise both block sub-paths (outcomes + predictors).
    prob = _toy_prob(seed=2)
    periods = slice(0, prob.Tpre)
    b1, B0 = _build_block(prob, periods, use_outcomes=True, use_predictors=True)
    assert B0.shape[0] == b1.shape[0]
    assert B0.shape[1] == prob.n_donors
    # outcomes-only and predictors-only must each be strictly shorter.
    b_o, _ = _build_block(prob, periods, use_outcomes=True, use_predictors=False)
    b_p, _ = _build_block(prob, periods, use_outcomes=False, use_predictors=True)
    assert b_o.shape[0] + b_p.shape[0] == b1.shape[0]


# --------------------------------------------------------------------------- #
# bias_corrected_gaps -- ridge=0 path
# --------------------------------------------------------------------------- #
def test_bias_corrected_gaps_zero_ridge():
    rng = np.random.default_rng(0)
    J, T, K = 8, 15, 3
    X0 = rng.normal(size=(K, J))
    beta = np.array([2.0, -1.0, 0.5])
    Y0 = (X0.T @ beta)[None, :] + rng.normal(scale=0.05, size=(T, J))
    X1 = rng.normal(size=K)
    y1 = (X1 @ beta) + rng.normal(scale=0.05, size=T)
    w = np.full(J, 1.0 / J)
    out = bias_corrected_gaps(w, X1, X0, y1, Y0, ridge=0.0)
    assert out.shape == (T,)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# dataclass structures -- properties
# --------------------------------------------------------------------------- #
def test_bilevel_problem_properties():
    Y0 = np.zeros((7, 4)); y1 = np.zeros(7)
    X0 = np.zeros((2, 4)); X1 = np.zeros(2)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    assert prob.n_donors == 4
    assert prob.n_predictors == 2
    assert prob.Tpre == 7


def test_bilevel_solution_gap_property():
    sol = _BS(V=np.array([1.0]), W=np.array([1.0]),
              upper_loss=5.0, lower_loss=4.0, lower_bound=3.0, stage="x")
    assert sol.gap == pytest.approx(2.0)          # upper_loss - lower_bound


# --------------------------------------------------------------------------- #
# Remaining solver / stage branch arms.
# --------------------------------------------------------------------------- #
def test_malo_refine_disabled_keeps_corner():
    # refine=False  ->  Stage 3 is skipped entirely (solver.py 129->134).
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(10, 5)); y1 = rng.normal(size=10)
    X0 = rng.normal(size=(3, 5)); X1 = rng.normal(size=3)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    sol = solve_bilevel(prob, method="malo", refine=False)
    assert sol.stage == "corner"
    assert sol.iterations == 0                     # refinement never ran


def test_malo_refine_runs_but_does_not_improve():
    # Seed 5 has a meaningful gap (refine fires) but the default Tykhonov step
    # does not beat the corner, so the result stays "corner" (solver.py 131->134).
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(10, 5)); y1 = rng.normal(size=10)
    X0 = rng.normal(size=(3, 5)); X1 = rng.normal(size=3)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    sol = solve_bilevel(prob, method="malo", refine=True)
    assert sol.stage == "corner"                   # refinement ran, no improvement


def test_tykhonov_inner_loop_runs_to_completion():
    # inner_iters == 1 with an improving first step exits the inner loop
    # normally (no break), exercising the loop-completion arm (stages.py 115->131).
    rng = np.random.default_rng(25)
    Y0 = rng.normal(size=(10, 5)); y1 = rng.normal(size=10)
    X0 = rng.normal(size=(3, 5)); X1 = rng.normal(size=3)
    prob = _BP(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    V0, _, _, _ = corner_solutions(prob)
    V, W, loss, total = tykhonov_refine(
        prob, V0, outer_iters=3, inner_iters=1, lr=0.2, fd_step=1e-2,
    )
    assert total == 3                              # one inner step per outer pass
    assert V.sum() == pytest.approx(1.0, abs=1e-6)

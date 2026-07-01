"""Tests for the shared warm-start + objective-cut MIQP accelerator.

``mlsynth.utils.miqp_accel.solve_warm_cut`` injects two things into cvxpy's SCIP
solve that cvxpy does not forward on its own:

* a *warm start* -- a binary MIP start (an initial incumbent), and
* a valid *objective lower-bound cut* ``c^T x >= L`` -- which raises SCIP's dual
  bound to ``L`` so the existing gap tolerance certifies against a tight bound
  instead of SCIP's loose McCormick relaxation.

The contract these tests pin:

* neither injection changes the proven optimum (both are valid);
* the cut actually raises the reported dual bound to ``L``;
* a too-high (invalid) ``L`` is caught -- the solve falls back to the un-cut
  problem and still returns the true optimum rather than a wrong/infeasible one;
* the warm-start shape guard fires on a mismatch.
"""
import numpy as np
import cvxpy as cp
import pytest

from mlsynth.utils.miqp_accel import solve_warm_cut, AccelInfo
from mlsynth.utils.syndes_helpers.formulation import build_syndes_problem_components
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
from mlsynth.utils.syndes_helpers.certificate import _sdp_moment_bound_two_way


def _make(N=7, K=2, T=10, seed=1):
    """Deterministic small two-way instance; returns (Y, lam, builder)."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, N))
    lam = float(estimate_lambda(Y))

    def build():
        D = cp.Variable(N, boolean=True)
        comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode="global_2way")
        prob = cp.Problem(cp.Minimize(comp.objective), list(comp.constraints))
        return prob, D

    return Y, lam, build


def _plain_opt(build):
    prob, _ = build()
    prob.solve(solver=cp.SCIP)
    return float(prob.value)


class TestSolveWarmCut:
    def test_neutral_no_warm_no_cut_matches_plain(self):
        """With neither injection the objective matches a plain SCIP solve."""
        _, _, build = _make()
        ref = _plain_opt(build)
        prob, D = build()
        prob, info = solve_warm_cut(prob, D, time_limit=30.0)
        assert isinstance(info, AccelInfo)
        assert not info.cut_applied and not info.warm_applied and not info.fell_back
        assert prob.value == pytest.approx(ref, abs=1e-4)

    def test_cut_raises_dual_bound_and_keeps_optimum(self):
        """A valid SDP lower bound, injected as a cut, raises the dual bound to L
        without changing the optimum."""
        Y, lam, build = _make()
        ref = _plain_opt(build)
        L = _sdp_moment_bound_two_way(Y, K=2, lam=lam)
        L_cut = L * 0.99                      # safety-margined, still valid
        prob, D = build()
        prob, info = solve_warm_cut(prob, D, objective_lower_bound=L_cut,
                                    gap_limit=1e-4, time_limit=30.0)
        assert info.cut_applied and not info.fell_back
        # dual bound lifted to (at least) the injected cut
        assert info.dual_bound >= L_cut - 1e-6
        # the returned design still respects the bound and equals the true optimum
        assert prob.value >= L_cut - 1e-6
        assert prob.value == pytest.approx(ref, abs=5e-3)

    def test_warm_start_applied_keeps_optimum(self):
        """A feasible K-hot warm start is accepted and the optimum is unchanged."""
        _, _, build = _make(N=7, K=2)
        ref = _plain_opt(build)
        warm = np.zeros(7); warm[[0, 1]] = 1.0
        prob, D = build()
        prob, info = solve_warm_cut(prob, D, warm_bits=warm, time_limit=30.0)
        assert info.warm_applied
        assert prob.value == pytest.approx(ref, abs=1e-4)

    def test_warm_and_cut_together(self):
        """Warm start and cut compose; both flags set, optimum preserved."""
        Y, lam, build = _make()
        ref = _plain_opt(build)
        L = _sdp_moment_bound_two_way(Y, K=2, lam=lam) * 0.99
        warm = np.zeros(7); warm[[0, 1]] = 1.0
        prob, D = build()
        prob, info = solve_warm_cut(prob, D, warm_bits=warm,
                                    objective_lower_bound=L,
                                    gap_limit=1e-4, time_limit=30.0)
        assert info.cut_applied and info.warm_applied
        assert prob.value == pytest.approx(ref, abs=5e-3)

    def test_too_high_bound_falls_back_to_uncut(self):
        """An invalid (too-high) L makes the cut infeasible; the solve must fall
        back to the un-cut problem and return the true optimum, never a wrong or
        infeasible answer."""
        _, _, build = _make()
        ref = _plain_opt(build)
        prob, D = build()
        prob, info = solve_warm_cut(prob, D, objective_lower_bound=1e6,
                                    time_limit=30.0)
        assert info.fell_back
        assert prob.value == pytest.approx(ref, abs=1e-4)

    def test_warm_shape_mismatch_raises(self):
        _, _, build = _make(N=7, K=2)
        prob, D = build()
        with pytest.raises(ValueError):
            solve_warm_cut(prob, D, warm_bits=np.zeros(5))

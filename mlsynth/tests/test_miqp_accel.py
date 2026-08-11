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
  problem and still returns the true optimum, not a wrong or infeasible one;
* the warm-start shape guard fires on a mismatch.
"""
import numpy as np
import cvxpy as cp
import pytest

from mlsynth.utils.miqp_accel import solve_warm_cut, AccelInfo, _bool_positions
from mlsynth.utils.syndes_helpers.formulation import build_syndes_problem_components
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
from mlsynth.utils.syndes_helpers.certificate import _rayleigh_bound_two_way


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
        """A valid closed-form lower bound, injected as a cut, raises the dual
        bound to L without changing the optimum."""
        Y, lam, build = _make()
        ref = _plain_opt(build)
        L = _rayleigh_bound_two_way(Y, K=2, lam=lam)
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
        L = _rayleigh_bound_two_way(Y, K=2, lam=lam) * 0.99
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


# ----------------------------------------------------------------------
# Which canonical variable each warm bit is written to
# ----------------------------------------------------------------------


def _pinned_through(data, positions, bits):
    """Objective with the boolean variables pinned at ``positions`` to ``bits``.

    Builds the model the accelerator builds, fixes the booleans by bound, and
    solves. What comes back is the objective of whichever design those positions
    actually name.
    """
    from pyscipopt.scip import Model
    from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as _Base

    base = _Base()
    model = Model()
    A, b, c, dims = base._define_data(data)
    variables = base._create_variables(model, data, c)
    base._add_constraints(model, variables, A, b, dims)
    base._set_params(model, False, {}, data, dims)
    for pos, vpos in enumerate(positions):
        model.chgVarLb(variables[vpos], float(bits[pos]))
        model.chgVarUb(variables[vpos], float(bits[pos]))
    model.hideOutput()
    model.optimize()
    return model.getObjVal() if model.getNSols() else None


def _instance(N, K, T, seed=1):
    """A two-way instance plus a fixed design to warm-start from."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, N))
    lam = float(estimate_lambda(Y))
    bits = np.zeros(N)
    bits[:K] = 1.0

    def build():
        D = cp.Variable(N, boolean=True)
        comp = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam,
                                               mode="global_2way")
        return cp.Problem(cp.Minimize(comp.objective), list(comp.constraints)), D

    return Y, lam, bits, build


class TestWarmBitAlignment:
    """A warm bit has to reach the unit the caller named.

    cvxpy hands back ``data[BOOL_IDX]`` as a set, so iterating it yields the hash
    table's order, which agrees with the variable's own order at some sizes and
    not at others. Writing ``warm_bits[j]`` to the j-th element of that iteration
    describes a different design from the one requested, and SCIP is then warm
    started onto a set the caller never asked for. Below N=11 the two orders
    happen to agree, which is why small cases pass while real ones do not.
    """

    SIZES = [(12, 4, 18), (14, 4, 20), (24, 6, 34)]

    def test_cvxpy_hands_back_an_unordered_set(self):
        """The dependency this guards against, pinned so a change is noticed."""
        import cvxpy.settings as s

        _, _, _, build = _instance(14, 4, 20)
        prob, _ = build()
        data, _, _ = prob.get_problem_data(cp.SCIP)
        assert isinstance(data[s.BOOL_IDX], set)

    @pytest.mark.parametrize("N,K,T", SIZES)
    def test_positions_are_the_ascending_block(self, N, K, T):
        """One boolean variable occupies a contiguous ascending run of columns."""
        _, _, _, build = _instance(N, K, T)
        prob, _ = build()
        data, _, _ = prob.get_problem_data(cp.SCIP)
        pos = _bool_positions(data)
        assert pos == sorted(pos)
        assert pos == list(range(min(pos), min(pos) + len(pos)))

    @pytest.mark.parametrize("N,K,T", SIZES)
    def test_warm_bits_name_the_design_the_caller_meant(self, N, K, T):
        """Pinning through the accelerator's positions must equal pinning in cvxpy.

        ``D == bits`` is the caller's intent stated to cvxpy. Fixing the same
        units through the accelerator's own mapping has to reach the same
        objective, whatever order the set iterates in.
        """
        Y, lam, bits, build = _instance(N, K, T)
        prob, D = build()
        intent = cp.Problem(prob.objective, list(prob.constraints) + [D == bits])
        intent.solve(solver=cp.SCIP)

        data, _, _ = build()[0].get_problem_data(cp.SCIP)
        through_mapping = _pinned_through(data, _bool_positions(data), bits)
        assert through_mapping == pytest.approx(intent.value, abs=1e-5)


class TestWarmStartHelps:
    def test_a_warm_start_never_returns_a_worse_design(self):
        """Incumbents only improve, so a start SCIP keeps bounds what it returns.

        Sized so SCIP cannot finish in the budget: that is the regime where the
        start is the only thing standing between the caller and a worse design,
        and where a misaligned one does active harm.
        """
        N, K, T = 60, 6, 40
        Y, lam, _, build = _instance(N, K, T)
        prob, D = build()
        prob, _ = solve_warm_cut(prob, D, time_limit=8.0)
        cold = float(prob.value)
        cold_bits = np.asarray(D.value).round()

        warm_prob, warm_D = build()
        warm_prob, info = solve_warm_cut(warm_prob, warm_D, warm_bits=cold_bits,
                                         time_limit=8.0)
        assert info.warm_applied
        assert float(warm_prob.value) <= cold + 1e-6

    def test_warm_applied_reports_what_scip_stored(self):
        _, _, build = _make(N=7, K=2)
        warm = np.zeros(7); warm[[0, 1]] = 1.0
        prob, D = build()
        _, info = solve_warm_cut(prob, D, warm_bits=warm, time_limit=30.0)
        assert info.warm_applied is True

    def test_warm_applied_is_false_when_scip_declines_the_start(self, monkeypatch):
        """The flag reports SCIP's answer, not that the call was made."""
        import pyscipopt.scip as _scip

        real = _scip.Model

        class Declines(real):
            def addSol(self, sol, free=True):
                real.addSol(self, sol, free=free)
                return False

        monkeypatch.setattr(_scip, "Model", Declines)
        _, _, build = _make(N=7, K=2)
        warm = np.zeros(7); warm[[0, 1]] = 1.0
        prob, D = build()
        _, info = solve_warm_cut(prob, D, warm_bits=warm, time_limit=30.0)
        assert info.warm_applied is False

"""The Graph-SC Monte Carlo shootout: classic SC, the Graph-SC arms, and their
CWZ-debiased versions over four simulation families already in the repository.

``benchmarks/studies/graphsc_mc`` reuses the repository's simulators (the CWZ
calibrated t-test design, the Liao-Shi-Zheng latent-group design, Hsiao-Zhou
DGP6 as Zhang-Wang-Zhang use it, and the Graph-SC paper's own cluster design)
and the CWZ debiased t-test in ``mlsynth.utils.inferutils``. The tests pin the
seams: that each draw is the design it claims, that the exact risk of
Zhang-Wang-Zhang is the expectation it defines, that every weight function uses
only the rows it is handed, that the scaled arms are free of the data's units
and the raw arm is not, and that the debiased rows are the CWZ procedure.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from benchmarks.studies.graphsc_mc import designs as D
from benchmarks.studies.graphsc_mc import methods as M
from benchmarks.studies.graphsc_mc import run as R
from benchmarks.studies.graphsc_replicate import oracle as G


def _panel(seed, T=40, J=12):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    Y0 = F @ rng.normal(size=(2, J)) + rng.normal(size=(T, J))
    y = Y0[:, :3].mean(axis=1) + 0.3 * rng.normal(size=T)
    return y, Y0


class TestTheActiveSetSolver:
    """Eq. (2) over R^J by an active-set search, QR on the final support, a KKT certificate."""

    @staticmethod
    def _problem(seed, J, T0):
        rng = np.random.default_rng(seed)
        X0 = rng.normal(size=(T0, 3)) @ rng.normal(size=(3, J)) + rng.normal(size=(T0, J))
        x1 = X0[:, :2].mean(axis=1) + 0.2 * rng.normal(size=T0)
        L, _ = G.knn_laplacian(X0.T, 3)
        return x1, X0, L

    @settings(max_examples=25, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(4, 40), st.integers(5, 60))
    @example(0, 40, 5)
    def test_it_is_clarabels_optimum_and_certified(self, seed, J, T0):
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(seed, J, T0)
        w, info = S.solve_signed(x1, X0, L, *M.LAM)
        assert info["kkt"] <= 1e-9
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            ref = G.solve_signed(x1, X0, L, *M.LAM)
        f = lambda v: G.objective(v, x1, X0, L, *M.LAM)
        assert f(w) <= f(ref) + 1e-9 * abs(f(ref))

    @settings(max_examples=25, deadline=None)
    @given(st.integers(0, 2**31 - 1))
    def test_a_warm_start_changes_nothing(self, seed):
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(seed, 20, 30)
        w, _ = S.solve_signed(x1, X0, L, *M.LAM)
        w0 = np.random.default_rng(seed).normal(size=20)
        v, info = S.solve_signed(x1, X0, L, *M.LAM, w0=w0)
        assert info["kkt"] <= 1e-9 and np.allclose(v, w, atol=1e-8 * np.abs(w).max())
        assert not info["fallback"]                     # the search found it, not the interior-point rescue

    @settings(max_examples=25, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(4, 25))
    def test_on_a_well_posed_problem_the_fast_path_answers(self, seed, J):
        """More periods than donors: the Gram matrix is definite, so search and QR polish must certify alone."""
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(seed, J, 2 * J + 10)
        w, info = S.solve_signed(x1, X0, L, *M.LAM)
        assert info["status"] == "optimal" and info["polished"] and not info["fallback"]
        assert info["kkt"] <= 1e-9

    @staticmethod
    def _outside_the_hull(seed):
        """Signed weights (-3, 3, 1) on donors at a level up to 1e4: large terms that cancel in ``Hw``."""
        rng = np.random.default_rng(seed)
        J = int(rng.integers(4, 30))
        T0 = int(rng.integers(2 * J, 3 * J + 10))
        lvl = 10.0 ** rng.uniform(0, 4)
        X0 = rng.normal(size=(T0, 3)) @ rng.normal(size=(3, J)) + rng.normal(size=(T0, J)) + lvl
        x1 = X0[:, :3] @ np.array([-3.0, 3.0, 1.0]) + 0.1 * rng.normal(size=T0)
        return x1, X0, G.knn_laplacian(X0.T, 3)[0]

    @settings(max_examples=20, deadline=None)
    @given(st.integers(0, 2**31 - 1))
    @example(67)                         # optimum's residual is 1834 eps of |c| + lambda: needs the |H||w| terms
    def test_the_optimality_test_is_sized_by_its_terms(self, seed):
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._outside_the_hull(seed)
        H, c = S.gram(x1, X0, L, 10.0, 50.0)
        _, _, status = S.active_set(H, c, M.LAM[0])
        assert status == "optimal"

    @staticmethod
    def _exact(x1, X0, L, lam1, lamg, lamsum):
        """The lasso optimum in rational arithmetic, by enumerating every sign pattern (J <= 5)."""
        import itertools
        from fractions import Fraction as Fr

        J = X0.shape[1]
        X = [[Fr(int(v)) for v in row] for row in X0]
        H = [[sum(X[t][i] * X[t][j] for t in range(len(X))) + lamg * Fr(int(L[i, j])) + lamsum
              for j in range(J)] for i in range(J)]
        c = [sum(X[t][i] * Fr(int(x1[t])) for t in range(len(X))) + lamsum for i in range(J)]

        def solve(A, rhs):
            n = len(A)
            m = [row[:] + [r] for row, r in zip(A, rhs)]
            for k in range(n):
                p = next(i for i in range(k, n) if m[i][k] != 0)
                m[k], m[p] = m[p], m[k]
                for i in range(n):
                    if i != k and m[i][k] != 0:
                        q = m[i][k] / m[k][k]
                        m[i] = [a - q * b for a, b in zip(m[i], m[k])]
            return [m[i][n] / m[i][i] for i in range(n)]

        for s in itertools.product((-1, 0, 1), repeat=J):
            A = [i for i in range(J) if s[i]]
            w = [Fr(0)] * J
            if A:
                for i, v in zip(A, solve([[H[i][j] for j in A] for i in A], [c[i] - lam1 * s[i] for i in A])):
                    w[i] = v
            if any(w[i] * s[i] <= 0 for i in A):
                continue
            g = [c[i] - sum(H[i][j] * w[j] for j in range(J)) for i in range(J)]
            if all(abs(g[i]) <= lam1 for i in range(J) if not s[i]):
                return np.array([float(v) for v in w])
        raise AssertionError("no sign pattern satisfies the KKT conditions")

    @settings(max_examples=15, deadline=None)
    @given(st.integers(0, 2**31 - 1))
    @example(0)
    def test_it_is_the_exact_rational_optimum(self, seed):
        """Integer data and a combinatorial Laplacian: the optimum is rational and enumerable.

        The tolerance is an inverse problem's (rule 13): ``64 eps cond(H) max|w|``.
        """
        from fractions import Fraction as Fr

        from benchmarks.studies.graphsc_mc import solver as S

        rng = np.random.default_rng(seed)
        J, T0 = int(rng.integers(2, 6)), int(rng.integers(3, 8))
        X0 = rng.integers(-5, 6, size=(T0, J)).astype(float)
        x1 = rng.integers(-5, 6, size=T0).astype(float)
        A = np.triu(rng.integers(0, 2, size=(J, J)), 1)
        A = A + A.T
        L = np.diag(A.sum(axis=1)) - A
        lam = (Fr(1, 20), Fr(10), Fr(50))
        w_exact = self._exact(x1, X0, L, *lam)
        w, info = S.solve_signed(x1, X0, L.astype(float), *(float(v) for v in lam))
        H, _ = S.gram(x1, X0, L.astype(float), 10.0, 50.0)
        bound = 64 * np.finfo(float).eps * np.linalg.cond(H) * max(np.abs(w_exact).max(), 1e-300)
        assert np.abs(w - w_exact).max() <= bound

    @settings(max_examples=20, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(-30, 30))
    @example(0, -30)
    @example(0, 30)
    def test_the_solve_has_no_units_of_its_own(self, seed, k):
        """Data times 2^k and every lambda times 4^k is the same problem, scaled exactly: the same weights."""
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(seed, 15, 25)
        c = 2.0 ** k
        w, _ = S.solve_signed(x1, X0, L, *M.LAM)
        v, info = S.solve_signed(c * x1, c * X0, L, *(c * c * lam for lam in M.LAM))
        assert not info["fallback"] and np.array_equal(v, w)

    def test_a_singular_reduced_system_says_so(self):
        from benchmarks.studies.graphsc_mc import solver as S

        _, steps, status = S.active_set(np.diag([1.0, 0.0]), np.array([0.0, 5.0]), 1.0)
        assert status == "singular" and steps == 1

    def test_the_polish_keeps_an_empty_support_and_refuses_a_moved_sign(self):
        from benchmarks.studies.graphsc_mc import solver as S

        assert np.array_equal(S._polish(np.zeros(3), np.eye(3), np.ones(3), 0.05), np.zeros(3))
        assert S._polish(np.array([1.0, 1.0]), np.eye(2), np.array([1.0, -1.0]), 0.0) is None

    def test_a_singular_factor_is_refused_not_raised(self):
        from benchmarks.studies.graphsc_mc import solver as S

        rng = np.random.default_rng(0)
        Mx = rng.normal(size=(12, 4))
        Mx[:, 2] = 0.0                                           # R has an exact zero on its diagonal
        assert S._polish(np.array([0.3, -0.2, 0.1, 0.4]), Mx, rng.normal(size=12), 0.05) is None

    def test_an_uncertified_point_is_reported(self, monkeypatch):
        import warnings

        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(2, 10, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            S.solve_signed(x1, X0, L, *M.LAM)                    # a clean solve is silent
        monkeypatch.setattr(S.G, "kkt_residual", lambda *a, **k: 1.0)
        with pytest.warns(S.CertificateWarning, match="no solve certified"):
            S.solve_signed(x1, X0, L, *M.LAM)

    def test_a_capped_search_says_so_and_falls_back(self):
        from benchmarks.studies.graphsc_mc import solver as S

        x1, X0, L = self._problem(1, 30, 40)
        H, c = S.gram(x1, X0, L, 10.0, 50.0)
        _, steps, status = S.active_set(H, c, 0.05, max_iter=1)
        assert status == "cap" and steps == 1
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            w, info = S.solve_signed(x1, X0, L, *M.LAM, max_iter=1)
        assert info["status"] == "cap" and info["fallback"]
        assert G.kkt_residual(w, x1, X0, L, *M.LAM, signed=True) <= 1e-7


class TestHsiaoZhouDesign:
    def test_the_draw_is_the_existing_dgp6_simulator(self):
        from benchmarks.studies.hsiao_zhou_counterfactuals.methods import simulate

        a = D.hz_design(30, 50, 10, np.random.default_rng(5))
        y, Y = simulate("dgp6", 30, 50, 10, np.random.default_rng(5))
        assert np.array_equal(a["y"], y) and np.array_equal(a["Y0"], Y)

    def test_the_covariance_is_zhang_wang_zhangs_closed_form(self):
        s2 = np.random.default_rng(1).uniform(0.5, 2.0, size=8)
        S, b = D.hz_sigma(s2), 1.0
        n = s2.size
        for i in range(n):
            for j in range(n):
                if i == j:
                    want = (1 + b * b) ** 2 * s2[i] + b * b * (
                        (s2[i - 1] if i > 0 else 0.0) + (s2[i + 1] if i < n - 1 else 0.0))
                elif abs(i - j) == 1:
                    want = b * (1 + b * b) * (s2[i] + s2[j])
                elif abs(i - j) == 2:
                    want = b * b * s2[(i + j) // 2]
                else:
                    want = 0.0
                assert np.isclose(S[i, j], want, rtol=1e-12, atol=0.0), (i, j)

    def test_the_risk_is_the_expected_post_period_loss(self):
        """Hold the factors and loadings, redraw only the shocks: the mean loss converges to R(w)."""
        rng = np.random.default_rng(3)
        d = D.hz_design(6, 20, 4, rng)
        mu_post, S = d["mu"][20:], D.hz_sigma(d["sigma2"])
        w = rng.dirichlet(np.ones(6))
        a = np.concatenate([[1.0], -w])
        C = np.linalg.cholesky(S)
        n = 40_000
        e = rng.standard_normal((n, 4, 7)) @ C.T                 # shocks with covariance S
        loss = np.mean(((mu_post + e) @ a) ** 2, axis=1)
        assert abs(loss.mean() - D.hz_risk(w, mu_post, S)) < 4 * loss.std() / np.sqrt(n)

    @settings(max_examples=30, deadline=None)
    @given(st.integers(0, 2**31 - 1))
    @example(0)
    def test_the_minimum_risk_is_the_simplex_minimum(self, seed):
        rng = np.random.default_rng(seed)
        d = D.hz_design(10, 30, 10, rng)
        mu_post, S = d["mu"][30:], D.hz_sigma(d["sigma2"])
        w_opt, r_min = D.hz_min_risk(mu_post, S)
        assert np.all(w_opt >= -1e-12) and abs(w_opt.sum() - 1.0) < 1e-9
        assert abs(D.hz_risk(w_opt, mu_post, S) - r_min) <= 1e-9 * r_min
        for w in rng.dirichlet(np.ones(10), size=50):
            assert D.hz_risk(w, mu_post, S) >= r_min * (1 - 1e-9)

    def test_the_draw_carries_its_risk(self):
        d = D.draw("hz", {"J": 30, "T0": 50}, np.random.default_rng(2))
        assert d.tau == 0.0 and d.T1 == 10 and d.Y0.shape == (60, 30)
        w = np.full(30, 1 / 30)
        assert d.risk(w) >= d.risk_min * (1 - 1e-9)


class TestTheOtherDesigns:
    def test_cwz_target_is_the_systematic_part(self):
        d = D.draw("cwz", {"dgp": 3}, np.random.default_rng(0))
        assert d.y.shape == (46,) and d.Y0.shape == (46, 14) and d.T0 == 30
        assert np.allclose(d.target - d.Y0 @ d.w_true, 2.0)      # DGP 3's constant
        assert d.tau == 0.0

    def test_lsz_target_is_the_oracle_synthetic_control(self):
        d = D.draw("lsz", {"J": 50, "T0": 50, "approximate": False}, np.random.default_rng(4))
        assert d.Y0.shape == (100, 50) and d.T0 == 50 and d.tau == 0.0
        assert np.allclose(d.target, d.Y0 @ d.w_true)
        assert np.all(d.w_true >= 0) and np.isclose(d.w_true.sum(), 1.0)

    def test_the_graph_sc_draw_carries_its_effect(self):
        d = D.draw("gsc", {}, np.random.default_rng(6))
        assert d.tau == 2.0 and d.T0 == 50 and d.Y0.shape == (70, 200)
        assert np.allclose(d.target, d.Y0 @ d.w_true)

    @pytest.mark.parametrize("family", sorted(D.FAMILIES))
    def test_every_family_draws_a_consistent_panel(self, family):
        cell = D.FAMILIES[family]["cells"][0]
        d = D.draw(family, cell, np.random.default_rng(9))
        T = d.T0 + d.T1
        assert d.y.shape == (T,) and d.Y0.shape[0] == T
        assert np.all(np.isfinite(d.y)) and np.all(np.isfinite(d.Y0))

    def test_an_unknown_family_is_refused(self):
        with pytest.raises(ValueError, match="family"):
            D.draw("nope", {}, np.random.default_rng(0))


class TestTheWeights:
    @pytest.mark.parametrize("method", M.SCALED)
    @settings(max_examples=12, deadline=None)
    @given(st.floats(1e-3, 1e3))
    @example(1e-3)
    def test_the_scaled_arms_do_not_see_the_units(self, method, c):
        y, Y0 = _panel(0)
        idx = np.arange(30)
        w1 = M.weight_fn(method, y, Y0, 30, seed=11)(idx)
        w2 = M.weight_fn(method, c * y, c * Y0, 30, seed=11)(idx)
        assert np.allclose(w1, w2, atol=1e-6 * max(1.0, np.abs(w1).max()))

    def test_the_raw_arm_on_a_tiny_scale_ignores_the_data(self):
        """At the paper's fixed lambdas, data of order 1e-3 leave only the penalty."""
        y, Y0 = _panel(1)
        c, idx = 1e-3, np.arange(30)
        w = M.weight_fn("Graph-SC signed, raw units", c * y, c * Y0, 30, seed=0)(idx)
        L, _ = G.knn_laplacian((c * Y0[idx]).T, M.KNN)
        w_pen = G.solve_signed(np.zeros(30), np.zeros((30, 12)), L, *M.LAM)
        assert np.abs(w - w_pen).max() < 0.02
        w_scaled = M.weight_fn("Graph-SC signed", c * y, c * Y0, 30, seed=0)(idx)
        assert np.abs(w_scaled - w_pen).max() > 0.1

    @pytest.mark.parametrize("method", M.METHODS)
    def test_a_weight_uses_only_the_rows_it_is_handed(self, method):
        y, Y0 = _panel(2)
        idx = np.arange(5, 25)
        y2, Y2 = y.copy(), Y0.copy()
        rest = np.setdiff1d(np.arange(40), idx)
        y2[rest] += 50.0
        Y2[rest] *= 3.0
        w1 = M.weight_fn(method, y, Y0, 30, seed=4)(idx)
        w2 = M.weight_fn(method, y2, Y2, 30, seed=4)(idx)
        assert np.allclose(w1, w2, atol=1e-9)

    def test_the_sc_arm_is_the_simplex_fit(self):
        y, Y0 = _panel(3)
        w = M.weight_fn("SC", y, Y0, 30, seed=0)(np.arange(30))
        assert np.all(w >= -1e-12) and np.isclose(w.sum(), 1.0)
        assert np.allclose(w, G.simplex_sc(y[:30], Y0[:30]), atol=1e-10)

    def test_the_control_penalties(self):
        y, Y0 = _panel(4)
        X = Y0[:30] / M.unit_scale(Y0[:30])
        L, A = G.knn_laplacian(X.T, M.KNN)
        Lr, m = M.penalty("Random graph signed", X, seed=8)
        assert m == int(A.sum() // 2)
        Ar = (np.abs(Lr) > 0) & ~np.eye(12, dtype=bool)
        assert int(Ar.sum() // 2) == m and not np.array_equal(Ar, A > 0)
        assert np.array_equal(M.penalty("Ridge signed", X, seed=8)[0], np.eye(12))
        assert np.allclose(M.penalty("Graph-SC signed", X, seed=8)[0], L)
        # The w >= 0 controls carry the same penalties as their signed twins.
        assert np.array_equal(M.penalty("Random graph w>=0", X, seed=8)[0], Lr)
        assert np.array_equal(M.penalty("Ridge w>=0", X, seed=8)[0], np.eye(12))

    @pytest.mark.parametrize("method", [m for m in M.METHODS if m.endswith("w>=0")])
    def test_every_nonnegative_arm_is_the_nonnegative_solve(self, method):
        y, Y0 = _panel(6)
        y = Y0[:, :3] @ np.array([-2.0, 2.0, 1.0])               # outside the hull: the readings differ
        signed = M.weight_fn(method.replace("w>=0", "signed"), y, Y0, 30, seed=3)(np.arange(30))
        assert signed.min() < -1e-3
        w = M.weight_fn(method, y, Y0, 30, seed=3)(np.arange(30))
        X = Y0[:30] / M.unit_scale(Y0[:30])
        P, _ = M.penalty(method, X, seed=3)
        assert np.all(w >= 0)
        assert np.allclose(w, G.solve_exact(y[:30] / M.unit_scale(Y0[:30]), X, P, *M.LAM), atol=1e-12)

    def test_the_gate_has_both_controls_in_both_readings(self):
        for reading in ("signed", "w>=0"):
            assert {f"Random graph {reading}", f"Ridge {reading}", f"Graph-SC {reading}"} <= set(M.METHODS)

    def test_the_oracles_certificate_failures_are_reissued(self):
        import warnings

        from benchmarks.studies.graphsc_mc import solver as S

        def flaky():
            warnings.warn(RuntimeWarning("the simplex QP for classic SC did not converge"))
            warnings.warn(UserWarning("unrelated"))
            return np.ones(2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            M.reported(flaky)
        kinds = [c.category for c in caught]
        assert kinds.count(S.CertificateWarning) == 1 and UserWarning in kinds
        assert R.count_uncertified(caught) == 1

    def test_only_certificate_failures_count(self):
        import warnings

        from benchmarks.studies.graphsc_mc import solver as S

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            np.log(np.zeros(1))                                  # numpy's divide-by-zero RuntimeWarning
            warnings.warn(S.CertificateWarning("x"))
        assert R.count_uncertified(caught) == 1

    def test_unknown_methods_and_flat_panels_are_refused(self):
        y, Y0 = _panel(5)
        with pytest.raises(ValueError, match="method"):
            M.weight_fn("Lasso", y, Y0, 30, seed=0)
        with pytest.raises(ValueError, match="scale"):
            M.weight_fn("Graph-SC signed", y, np.zeros_like(Y0), 30, seed=0)(np.arange(30))
        with pytest.raises(ValueError, match="pre-period"):
            M.weight_fn("SC", y, Y0, 30, seed=0)(np.arange(31))


class TestOneReplication:
    @pytest.fixture(scope="class")
    def rep(self):
        d = D.draw("cwz", {"dgp": 1}, np.random.default_rng(12))
        return d, R.evaluate(d, K=3, alpha=0.1, seed=12)

    def test_every_estimator_reports(self, rep):
        d, out = rep
        assert set(out["plain"]) == set(M.METHODS) == set(out["debiased"])
        assert out["oracle"] is not None and isinstance(out["uncertified"], int)
        for row in list(out["plain"].values()) + list(out["debiased"].values()):
            assert np.isfinite(row["err"])

    def test_the_plain_att_is_the_mean_post_gap(self, rep):
        d, out = rep
        w = M.weight_fn("SC", d.y, d.Y0, d.T0, seed=0)(np.arange(d.T0))
        gap = np.mean(d.y[d.T0:] - d.Y0[d.T0:] @ w)
        assert np.isclose(out["plain"]["SC"]["err"], gap - d.tau, atol=1e-10)

    def test_debiased_sc_is_mlsynths_cwz_t_test(self, rep):
        from mlsynth.utils.inferutils import debiased_sc_ttest

        d, out = rep
        ref = debiased_sc_ttest(d.y, d.Y0, d.T0, d.T1, K=3, alpha=0.1)
        row = out["debiased"]["SC"]
        assert abs(row["err"] - (ref["att"] - d.tau)) < 1e-6
        assert abs(row["length"] - (ref["ci_upper"] - ref["ci_lower"])) < 1e-6
        assert row["covered"] == (ref["ci_lower"] <= d.tau <= ref["ci_upper"])

    def test_the_oracle_row_uses_the_true_weights(self, rep):
        from benchmarks.cwz_common import oracle_ttest

        from scipy.stats import t as student_t

        d, out = rep
        att, se = oracle_ttest(d.y, d.Y0, d.T0, d.T1, d.w_true, 3)
        assert abs(out["oracle"]["err"] - (att - d.tau)) < 1e-12
        assert np.isclose(out["oracle"]["length"], 2 * student_t.ppf(0.95, 2) * se, rtol=1e-12)

    def test_coverage_is_judged_at_the_true_effect(self):
        """Shift the post-period by 100: an interval judged at zero would never cover."""
        d0 = D.draw("cwz", {"dgp": 1}, np.random.default_rng(1))
        shift = 100.0 * (np.arange(d0.y.size) >= d0.T0)
        d = D.Draw(d0.y + shift, d0.Y0, d0.T0, 100.0, None, d0.w_true)
        out = R.evaluate(d, K=3, alpha=0.1, seed=1, methods=("SC",))
        row = out["debiased"]["SC"]
        lo = row["err"] + d.tau - row["length"] / 2
        assert row["covered"] == (lo <= d.tau <= lo + row["length"])
        assert out["oracle"]["covered"] and row["covered"]


class TestTheRunner:
    def test_sc_never_beats_the_simplex_minimum_risk(self):
        d = D.draw("hz", {"J": 30, "T0": 50}, np.random.default_rng(3))
        out = R.evaluate(d, K=5, alpha=0.1, seed=3, methods=("SC",))
        assert out["plain"]["SC"]["risk_ratio"] >= 1 - 1e-9

    def test_main_writes_every_cell(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setattr(R, "OUT", tmp_path / "graphsc_mc.json")
        R.main(["hz"], workers=1, reps=2, dump=str(tmp_path / "reps.pkl"))
        res = json.loads((tmp_path / "graphsc_mc.json").read_text())
        cells = res["hz"]["cells"]
        assert len(cells) == len(D.FAMILIES["hz"]["cells"]) and all(c["n"] == 2 for c in cells)
        assert all(c["plain"]["SC"]["risk_ratio_mean"] >= 1 - 1e-9 and "coverage" in c["oracle"] for c in cells)
        assert res["_meta"]["methods"] == list(M.METHODS) and (tmp_path / "reps.pkl").exists()

    def test_one_replication_is_seeded_by_its_cell_and_index(self):
        f, ci, a = R.one_rep(("hz", 1, 4))
        _, _, b = R.one_rep(("hz", 1, 4))
        _, _, c = R.one_rep(("hz", 1, 5))
        assert (f, ci) == ("hz", 1) and a["plain"]["SC"]["err"] == b["plain"]["SC"]["err"]
        assert a["plain"]["SC"]["err"] != c["plain"]["SC"]["err"]

    def test_a_worker_runs_on_one_blas_thread(self):
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=None):
            R._init_worker()
            assert all(p["num_threads"] == 1 for p in threadpool_info())


class TestSummaries:
    def test_bias_rmse_and_coverage(self):
        reps = [{"plain": {"SC": {"err": e, "sse": s}}, "debiased": {"SC": {"err": e, "covered": c, "length": 2.0}},
                 "oracle": None, "uncertified": 0}
                for e, s, c in ((1.0, 4.0, True), (-1.0, 1.0, False), (3.0, 2.0, True), (1.0, 3.0, True))]
        s = R.summarise(reps, methods=("SC",))
        p, q = s["plain"]["SC"], s["debiased"]["SC"]
        assert np.isclose(p["bias"], 1.0) and np.isclose(p["rmse"], np.sqrt(3.0))
        assert np.isclose(q["coverage"], 0.75) and np.isclose(q["coverage_se"], np.sqrt(0.75 * 0.25 / 4))
        assert np.isclose(q["length"], 2.0)

    def test_the_ratio_to_sc_is_the_median_of_per_replication_ratios(self):
        reps = [{"plain": {"SC": {"err": 0.0, "sse": a}, "X": {"err": 0.0, "sse": b}},
                 "debiased": {}, "oracle": None, "uncertified": 0}
                for a, b in ((1.0, 2.0), (2.0, 1.0), (4.0, 1.0))]
        s = R.summarise(reps, methods=("SC", "X"))
        assert np.isclose(s["plain"]["X"]["sse_ratio_median"], 0.5)

    def test_the_mse_ratio_to_sc_and_its_interval(self):
        rng = np.random.default_rng(0)
        e_sc, e_x = rng.normal(size=200), 0.5 * rng.normal(size=200)
        reps = [{"plain": {"SC": {"err": a}, "X": {"err": b}}, "debiased": {}, "oracle": None, "uncertified": 0}
                for a, b in zip(e_sc, e_x)]
        s = R.summarise(reps, methods=("SC", "X"))["plain"]["X"]
        assert np.isclose(s["mse_ratio"], np.mean(e_x ** 2) / np.mean(e_sc ** 2))
        lo, hi = s["mse_ratio_ci"]
        assert lo < s["mse_ratio"] < hi and hi < 1.0

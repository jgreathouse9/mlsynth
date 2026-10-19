"""The pair-free batched solver in the RRMA study, against HiGHS.

``benchmarks/studies/rrma_ablation/solver_pairfree.py`` solves weighted
Wilcoxon fits with no array carrying a pair dimension and certifies each optimum
through a prefix test on net flows. Two things could be wrong without any
number looking odd: the reduction from pairs to periods (rank scores standing in
for the pairwise objective and subgradient), and the certificate (a prefix test
that missed a violated cut would stop at a vertex that is not optimal). The
properties below compare objectives with HiGHS on the pairwise programme, which
an L1 programme's face of minimisers leaves as the only fair comparison.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from benchmarks.studies.rrma_ablation import highs_reference as HR
from benchmarks.studies.rrma_ablation import oracle as O
from benchmarks.studies.rrma_ablation import solver_pairfree as PF


@st.composite
def _batch(draw):
    K = draw(st.integers(1, 6))
    p = draw(st.integers(1, 5))
    n = draw(st.integers(p + 3, 30))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    X = rng.normal(size=(K, n, p))
    y = np.einsum("knp,p->kn", X, rng.normal(size=p)) + rng.standard_t(2, size=(K, n))
    h = rng.uniform(0.05, 1.0, size=(K, n))
    return y, X, h


def _highs(y, X, h):
    i, j = O.pairs(len(y))
    w, D, r = h[i] * h[j], X[i] - X[j], y[i] - y[j]
    return w, D, r, HR.wlad(r, D, w)


def _obj(w, D, r, b):
    return float(np.sum(w * np.abs(r - D @ b)))


class TestTheReduction:
    @settings(max_examples=40, deadline=None)
    @given(_batch())
    def test_rank_scores_carry_the_pairwise_objective_and_subgradient(self, b):
        y, X, h = b
        rng = np.random.default_rng(0)
        for k in range(len(y)):
            e = y[k] - X[k] @ rng.normal(size=X.shape[2])
            a = PF.rank_scores(h[k:k + 1], np.arange(len(e))[None], e[None])[0]
            i, j = O.pairs(len(e))
            w = h[k, i] * h[k, j]
            assert np.isclose(a @ e, np.sum(w * np.abs(e[i] - e[j])), rtol=1e-12)
            pair_grad = (X[k, i] - X[k, j]).T @ (w * np.sign(e[i] - e[j]))
            assert np.allclose(X[k].T @ a, pair_grad, rtol=1e-10, atol=1e-10)


class TestExactness:
    @settings(max_examples=40, deadline=None)
    @given(_batch(), st.sampled_from(["pairs", "kinetic"]))
    def test_reaches_the_highs_optimum_from_a_crash(self, b, line_search):
        y, X, h = b
        beta, _, _, _ = PF.solve(y, X, h, PF.crash(y, X, h), line_search)
        for k in range(len(y)):
            w, D, r, ref = _highs(y[k], X[k], h[k])
            assert _obj(w, D, r, beta[k]) <= _obj(w, D, r, ref) * (1 + 1e-9) + 1e-12

    @settings(max_examples=25, deadline=None)
    @given(_batch())
    def test_a_warm_start_reaches_the_same_optimum(self, b):
        """The previous window's labelling, plus the new row on its own, is a vertex."""
        y, X, h = b
        _, lab, _, _ = PF.solve(y[:, :-1], X[:, :-1], h[:, :-1], PF.crash(y[:, :-1], X[:, :-1], h[:, :-1]))
        warm = np.concatenate([lab, np.full((len(y), 1), y.shape[1] - 1)], axis=1)
        beta, _, _, _ = PF.solve(y, X, h, warm)
        for k in range(len(y)):
            w, D, r, ref = _highs(y[k], X[k], h[k])
            assert _obj(w, D, r, beta[k]) <= _obj(w, D, r, ref) * (1 + 1e-9) + 1e-12


class TestTheCertificate:
    def test_it_rejects_the_crash_vertex_and_accepts_the_optimum(self):
        """A certificate that always passed would make every crash look optimal."""
        rng = np.random.default_rng(4)
        K, n, p = 8, 40, 3
        X = rng.normal(size=(K, n, p))
        y = np.einsum("knp,p->kn", X, [1.0, -2.0, 0.5]) + rng.standard_t(2, size=(K, n))
        h = rng.uniform(0.2, 1.0, size=(K, n))
        lab0 = PF.crash(y, X, h)
        _, lab, pivots, _ = PF.solve(y, X, h, lab0)
        assert np.all(pivots > 0)

        def worst(labels):
            beta, T, ti, tj = PF.vertex(y, X, labels)
            e = np.take_along_axis(y - np.einsum("knp,kp->kn", X, beta), labels, axis=1)
            a = PF.rank_scores(h, labels, e)
            theta = np.linalg.solve(np.transpose(T, (0, 2, 1)), -np.einsum("knp,kn->kp", X, a)[..., None])[..., 0]
            psi = np.zeros((K, n))
            rows = np.broadcast_to(np.arange(K)[:, None], ti.shape)
            np.add.at(psi, (rows, ti), theta)
            np.add.at(psi, (rows, tj), -theta)
            return PF.certificate(h, labels, psi)[0]

        assert np.all(worst(lab0) > 1e-6)
        assert np.all(worst(lab) <= 1e-9 * (h.sum(axis=1) ** 2 / 4))


# ------------------------------------------------------------ numerical safeguards
#
# Nocedal and Wright (2006), pp. 26-27: an algorithm whose answer depends on the
# units of the data is badly scaled. Russinoff (2022), Lemma 6.45: one rounding
# carries an error relative to the size of its operands, so a residual of data
# sitting at 1e8 is known to 1e8 * eps whatever its own size. These tests pin
# the solver to both: the answer and the certificate do not depend on the units
# of y, X or h, nor on a common level added to y and X.


def _pairwise_highs_objective(y, X, h):
    """The HiGHS optimum of each problem of a batch, on the pairwise programme."""
    out = []
    for k in range(len(y)):
        w, D, r, ref = _highs(y[k], X[k], h[k])
        out.append(_obj(w, D, r, ref))
    return np.array(out)


def _pairwise(y, X, h, beta):
    return np.array([_obj(*_highs_parts(y[k], X[k], h[k]), beta[k]) for k in range(len(y))])


def _highs_parts(y, X, h):
    i, j = O.pairs(len(y))
    return h[i] * h[j], X[i] - X[j], y[i] - y[j]


def _solve_strict(y, X, h, line_search="pairs"):
    """``PF.solve`` with a certificate failure turned into an error."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", PF.CertificateWarning)
        return PF.solve(y, X, h, PF.crash(y, X, h), line_search, return_certified=True)


class TestTheCertificateIsReported:
    def test_a_failed_certificate_warns_and_names_how_many(self, monkeypatch):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(3, 20, 2))
        y = X[..., 0] + rng.normal(size=(3, 20))
        h = np.ones((3, 20))
        monkeypatch.setattr(PF, "certify", lambda *a: np.array([True, False, False]))
        with pytest.warns(PF.CertificateWarning) as rec:
            out = PF.solve(y, X, h, PF.crash(y, X, h), return_certified=True)
        assert [w.message.problems for w in rec] == [2]
        assert out[4].tolist() == [True, False, False]

    def test_a_certified_solve_is_silent(self):
        import warnings

        rng = np.random.default_rng(3)
        X = rng.normal(size=(2, 25, 3))
        y = X @ np.array([1.0, 0.5, -1.0]) + rng.standard_t(2, size=(2, 25))
        h = rng.uniform(0.2, 1.0, size=(2, 25))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PF.solve(y, X, h, PF.crash(y, X, h))

    def test_there_is_no_module_level_counter(self):
        assert not hasattr(PF, "UNCERTIFIED")


class TestUnits:
    @settings(max_examples=30, deadline=None)
    @given(_batch(), st.integers(-8, 8))
    def test_rescaling_y_and_x_changes_nothing(self, b, k):
        y, X, h = b
        s = 10.0 ** k
        beta, _, _, _, cert = _solve_strict(s * y, s * X, h)
        assert cert.all()
        ref = _pairwise_highs_objective(y, X, h)
        assert np.all(_pairwise(y, X, h, beta) <= ref * (1 + 1e-9))

    @settings(max_examples=30, deadline=None)
    @given(_batch(), st.integers(-4, 4))
    def test_rescaling_the_weights_changes_nothing(self, b, k):
        y, X, h = b
        beta, _, _, _, cert = _solve_strict(y, X, h * 10.0 ** k)
        assert cert.all()
        ref = _pairwise_highs_objective(y, X, h)
        assert np.all(_pairwise(y, X, h, beta) <= ref * (1 + 1e-9))

    @settings(max_examples=30, deadline=None)
    @given(_batch(), st.sampled_from([1e2, 1e4, 1e6, 1e8]))
    def test_a_common_level_in_y_and_x_changes_nothing(self, b, c):
        """The pairwise objective reads only differences, so a level ``c`` added to
        every entry of ``y`` and ``X`` leaves the problem unchanged, except that
        the data are then stored to ``eps c / 2``. A residual then errs by up to
        ``eps c (1 + |beta|_1) / 2`` and the objective by up to ``h(.)^2`` times
        that, in either problem, so the solver's point, scored on the unshifted
        data, can exceed the optimum by twice the objective's error. The bound
        is absolute, in the objective's units: a first version made it relative
        to the optimum and failed on a problem whose optimum was 0.07."""
        y, X, h = b
        beta, _, _, _, cert = _solve_strict(y + c, X + c, h)
        assert cert.all()
        ref = _pairwise_highs_objective(y, X, h)
        err = h.sum(axis=1) ** 2 * np.finfo(float).eps * c * (1 + np.abs(beta).sum(axis=1)) / 2
        assert np.all(_pairwise(y, X, h, beta) <= ref * (1 + 1e-9) + 2 * err)


class TestTheCertificateAtEveryScale:
    @pytest.mark.parametrize("s", [1e-6, 1.0, 1e6])
    def test_it_rejects_a_point_one_part_in_a_million_from_the_optimum(self, s):
        """With a floor of 1 under its tolerance the test certified nothing once
        the objective fell below 1: at ``s = 1e-6`` it accepted a point 3e-6 from
        the optimum."""
        rng = np.random.default_rng(0)
        K, n, p = 6, 40, 3
        X = s * rng.normal(size=(K, n, p))
        y = np.einsum("knp,p->kn", X, [1.0, -2.0, 0.5]) + s * rng.standard_t(2, size=(K, n))
        h = rng.uniform(0.2, 1.0, size=(K, n))
        beta, lab, _, _ = PF.solve(y, X, h, PF.crash(y, X, h))
        phi = PF.dual_flows(y + PF.perturbation(y), X, h, lab)
        assert PF.certify(y, X, h, beta, phi).all()
        F = PF.pairwise_objective(y, X, h, beta)
        v = rng.normal(size=beta.shape)
        for t in np.logspace(-9, 0, 40):
            bt = beta + t * v
            off = (PF.pairwise_objective(y, X, h, bt) - F) / F
            ok = PF.certify(y, X, h, bt, phi)
            assert not np.any(ok & (off > 1e-6)), (t, off[ok].max())

    def test_it_rejects_a_dual_that_violates_a_cut(self):
        """``X' phi = 0`` and a zero gap prove nothing unless ``phi`` is the net
        flow of pair flows inside their bounds. A ``phi`` built to meet the first
        two at a vertex that is not optimal must fail on the third."""
        rng = np.random.default_rng(5)
        n, p = 30, 2
        X = rng.normal(size=(1, n, p))
        y = X[..., 0] + rng.standard_t(2, size=(1, n))
        h = rng.uniform(0.3, 1.0, size=(1, n))
        lab0 = PF.crash(y, X, h)
        beta = PF.vertex(y, X, lab0)[0]
        e = y[0] - X[0] @ beta[0]
        a = PF.rank_scores(h, np.arange(n)[None], e[None])[0]
        # w with X'w = -X'a and e'w = 0, so that phi = a + w meets both equalities
        A = np.vstack([X[0].T, e[None]])
        w = np.linalg.lstsq(A, np.r_[-(X[0].T @ a), 0.0], rcond=None)[0]
        phi = (a + w)[None]
        F = PF.pairwise_objective(y, X, h, beta)
        ref = _pairwise_highs_objective(y, X, h)
        assert F[0] > ref[0] * (1 + 1e-3)                       # not optimal
        assert abs(F[0] - phi[0] @ e) <= 1e-9 * F[0]            # zero gap
        assert not PF.certify(y, X, h, beta, phi)[0]


class TestCancellation:
    def test_the_objective_keeps_its_digits_at_a_large_level(self):
        """Every pair's term is ``h_i h_j |e_i - e_j|``; summed as ``a'e`` with
        ``e`` near 1e8 the products cancel to 1.5e-9 relative. Centring ``e``
        first (``a`` sums to zero) leaves only the rounding of the sum."""
        from fractions import Fraction

        rng = np.random.default_rng(11)
        n = 60
        h = rng.uniform(0.5, 1.0, size=(1, n))
        e = rng.normal(size=(1, n)) + 1e8
        X = np.zeros((1, n, 1))
        F = PF.pairwise_objective(e, X, h, np.zeros((1, 1)))[0]
        hf, ef = [Fraction(v) for v in h[0]], [Fraction(v) for v in e[0]]
        exact = sum(hf[i] * hf[j] * abs(ef[i] - ef[j]) for i in range(n) for j in range(i + 1, n))
        assert abs(Fraction(F) - exact) / exact <= 1e-14


class TestThePerturbation:
    @settings(max_examples=60, deadline=None)
    @given(st.integers(2, 200), st.integers(0, 2**31 - 1), st.floats(-1e8, 1e8), st.integers(-6, 6))
    def test_it_is_sized_by_the_spread_and_ignores_the_level(self, n, seed, c, k):
        """Sized by ``max|y|`` the shift at a level of 1e8 was 0.1, as large as the
        data's own variation, and the pivots solved a different problem."""
        y = 10.0 ** k * np.random.default_rng(seed).normal(size=n)
        spread = np.abs(y - y.mean()).max()
        d0, d1 = PF.perturbation(y), PF.perturbation(y + c)
        assert np.abs(d0).max() <= PF.EPS_PERTURB * spread * (1 + 1e-12)
        assert np.abs(d0).max() > 0
        assert np.allclose(d1, d0, rtol=1e-6, atol=PF.EPS_PERTURB * 64 * np.finfo(float).eps * (abs(c) + spread))


class TestInputs:
    def test_an_intercept_column_is_named_as_the_problem(self):
        """Differences cancel a constant column, so ``beta`` is not identified."""
        rng = np.random.default_rng(6)
        X = np.concatenate([np.ones((1, 30, 1)), rng.normal(size=(1, 30, 2))], axis=2)
        y = rng.normal(size=(1, 30))
        with pytest.raises(ValueError, match="rank"):
            PF.crash(y, X, np.ones((1, 30)))

    def test_collinear_columns_are_named_as_the_problem(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1, 30, 1))
        X = np.concatenate([x, 2 * x], axis=2)
        with pytest.raises(ValueError, match="rank"):
            PF.crash(rng.normal(size=(1, 30)), X, np.ones((1, 30)))

    def test_a_crash_that_cannot_find_independent_rows_says_so(self):
        """Rank ``p`` by the numerical-rank test, but the only directions that
        separate the columns sit below the window's 1e-8 independence threshold."""
        rng = np.random.default_rng(8)
        x = rng.normal(size=(1, 30, 1))
        X = np.concatenate([x, x + 1e-11 * rng.normal(size=(1, 30, 1))], axis=2)
        y = x[..., 0] + rng.normal(size=(1, 30))
        with pytest.raises(ValueError, match="independent"):
            PF.crash(y, X, np.ones((1, 30)))

    @pytest.mark.parametrize("bad", [0.0, -0.5, np.nan, np.inf])
    def test_weights_must_be_positive_and_finite(self, bad):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(1, 20, 2))
        y = rng.normal(size=(1, 20))
        h = np.ones((1, 20))
        lab = PF.crash(y, X, h)
        h[0, 4] = bad
        with pytest.raises(ValueError, match="weights"):
            PF.crash(y, X, h)
        with pytest.raises(ValueError, match="weights"):
            PF.solve(y, X, h, lab)


class TestTheLineSearches:
    @settings(max_examples=60, deadline=None)
    @given(st.integers(2, 25), st.integers(0, 2**31 - 1))
    def test_both_stop_at_the_same_breakpoint_and_the_kinetic_one_is_finite(self, n, seed):
        """Each swap of the kinetic search removes one inversion of ``u`` among the
        periods, so it stops within ``n (n - 1) / 2`` rounds whatever the rounding."""
        rng = np.random.default_rng(seed)
        e, u, h = rng.normal(size=(1, n)), rng.normal(size=(1, n)), rng.uniform(0.1, 1, size=(1, n))
        ld = np.arange(n)[None]
        # the slope of the objective at t = 0 along e + t u
        Fp = (PF.rank_scores(h, ld, e, u) * u).sum(axis=1)
        crossing = (e[0][:, None] < e[0][None, :]) & (u[0][:, None] > u[0][None, :])
        assume(Fp[0] < 0 and crossing.any())
        tol = np.zeros(1)
        pp, _ = PF._long_step_pairs(e, u, ld, h, Fp.copy(), tol)
        pk, rounds = PF._long_step_kinetic(e, u, ld, h, Fp.copy(), tol)
        t = lambda pr: (e[0, pr[1]] - e[0, pr[0]]) / (u[0, pr[0]] - u[0, pr[1]])
        assert np.isclose(t(pp[0]), t(pk[0]), rtol=1e-12, atol=1e-12)
        assert rounds <= n * (n - 1) // 2

    @pytest.mark.parametrize("search", ["_long_step_pairs", "_long_step_kinetic"])
    def test_a_slope_that_never_turns_is_an_error(self, search):
        rng = np.random.default_rng(10)
        e, u, h = rng.normal(size=(1, 8)), rng.normal(size=(1, 8)), np.ones((1, 8))
        with pytest.raises(RuntimeError, match="unbounded"):
            getattr(PF, search)(e, u, np.arange(8)[None], h, np.array([-np.inf]), np.zeros(1))


class TestUnidentifiedFits:
    """The pairwise objective reads ``X`` only through differences of rows, so with
    ``n - 1 < p`` periods (or collinear columns) ``beta`` is identified only on the
    row space of the centred design. Algorithm 1 asks for such fits on Basque:
    at ``m = 6`` the last group takes 10 donors, and GWCV fits it on windows of
    7 to 10 pre-periods. ``PF.fit`` solves in the ``r`` identified coordinates
    and returns ``beta = V_r gamma``, the minimum-norm point of the face."""

    @staticmethod
    def _deficient(rng, K=3, n=8, p=10):
        X = rng.normal(size=(K, n, p))
        y = np.einsum("knp,p->kn", X, rng.normal(size=p)) + rng.standard_t(2, size=(K, n))
        return y, X, rng.uniform(0.2, 1.0, size=(K, n))

    def test_it_reaches_the_optimum_and_says_beta_was_not_identified(self):
        rng = np.random.default_rng(12)
        y, X, h = self._deficient(rng)
        with pytest.warns(PF.UnidentifiedFitWarning) as rec:
            beta, _, ident = PF.fit(y, X, h)
        assert sum(w.message.problems for w in rec) == 3 and not ident.any()
        ref = _pairwise_highs_objective(y, X, h)
        got = _pairwise(y, X, h, beta)
        assert np.all(got <= ref * (1 + 1e-9) + 64 * np.finfo(float).eps * h.sum(axis=1) ** 2
                      * (np.abs(y).max(axis=1) + np.abs(X).max(axis=(1, 2)) * np.abs(beta).sum(axis=1)))

    def test_beta_has_no_component_the_data_cannot_see(self):
        rng = np.random.default_rng(13)
        y, X, h = self._deficient(rng, K=2, n=6, p=9)
        with pytest.warns(PF.UnidentifiedFitWarning):
            beta, _, _ = PF.fit(y, X, h)
        for k in range(2):
            Xc = X[k] - X[k].mean(axis=0)
            _, s, Vt = np.linalg.svd(Xc)
            null = Vt[int((s > s[0] * 9 * np.finfo(float).eps).sum()):]
            assert np.abs(null @ beta[k]).max() <= 1e-10 * np.abs(beta[k]).max()

    def test_a_mixed_batch_solves_the_identified_problems_as_before(self):
        rng = np.random.default_rng(14)
        K, n, p = 4, 20, 3
        X = rng.normal(size=(K, n, p))
        X[1, :, 2] = 2.0 * X[1, :, 0]                      # problem 1: collinear columns
        y = np.einsum("knp,p->kn", X, [1.0, -1.0, 0.5]) + rng.standard_t(2, size=(K, n))
        h = rng.uniform(0.2, 1.0, size=(K, n))
        with pytest.warns(PF.UnidentifiedFitWarning) as rec:
            beta, labels, ident = PF.fit(y, X, h)
        assert ident.tolist() == [True, False, True, True]
        assert [w.message.problems for w in rec] == [1]
        keep = [0, 2, 3]
        b_ref = PF.solve(y[keep], X[keep], h[keep], PF.crash(y[keep], X[keep], h[keep]))[0]
        assert np.allclose(_pairwise(y[keep], X[keep], h[keep], beta[keep]),
                           _pairwise(y[keep], X[keep], h[keep], b_ref), rtol=1e-12)
        ref = _pairwise_highs_objective(y[1:2], X[1:2], h[1:2])
        assert _pairwise(y[1:2], X[1:2], h[1:2], beta[1:2])[0] <= ref[0] * (1 + 1e-9)

    def test_a_design_with_no_variation_returns_zero(self):
        X = np.ones((1, 6, 2))
        y = np.arange(6.0)[None]
        with pytest.warns(PF.UnidentifiedFitWarning):
            beta, _, ident = PF.fit(y, X, np.ones((1, 6)))
        assert np.array_equal(beta, np.zeros((1, 2))) and not ident[0]

    def test_windows_that_become_identified_warm_start_correctly(self):
        """``oracle.rank_fits`` over windows from rank-deficient to full rank equals
        a fresh ``PF.fit`` per window."""
        rng = np.random.default_rng(15)
        K, T, p = 3, 16, 6
        X = rng.normal(size=(K, T, p))
        y = X[0] @ rng.normal(size=p) + rng.standard_t(2, size=T)
        hs = [rng.uniform(0.2, 1.0, size=T) for _ in range(K)]
        with pytest.warns(PF.UnidentifiedFitWarning):
            fits = O.rank_fits(y, X, hs, list(range(5, T + 1)))
        H = np.stack(hs)
        for n, (_, beta) in fits.items():
            yy = np.broadcast_to(y[:n], (K, n))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PF.UnidentifiedFitWarning)
                fresh = PF.fit(yy, X[:, :n], H[:, :n])[0]
            assert np.allclose(_pairwise(yy, X[:, :n], H[:, :n], beta),
                               _pairwise(yy, X[:, :n], H[:, :n], fresh), rtol=1e-9, atol=1e-12)

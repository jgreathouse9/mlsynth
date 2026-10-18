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

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

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
    return w, D, r, O.wlad(r, D, w)


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

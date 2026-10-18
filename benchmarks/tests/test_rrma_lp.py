"""The RRMA study's own LP solvers, against HiGHS.

The estimator path of ``benchmarks/studies/rrma_ablation`` solves its linear
programmes with NumPy code written for the study: ``solver_pairfree`` for the
weighted Wilcoxon fits, ``lp.simplex_rank_lad`` for Section 2.4's model weights
(a pairwise L1 loss over the simplex), ``lp.simplex_lad`` for the weights-on-
levels ablation (a plain weighted L1 loss over the simplex). HiGHS stays only in
``highs_reference`` as the independent check these tests compare against, by
objective value: an L1 programme can have a face of minimisers, and any point of
it is correct.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.rrma_ablation import highs_reference as HR
from benchmarks.studies.rrma_ablation import lp as LP
from benchmarks.studies.rrma_ablation import oracle as O

STUDY = Path(__file__).resolve().parents[1] / "studies" / "rrma_ablation"


@st.composite
def _fit(draw, max_p=5):
    p = draw(st.integers(1, max_p))
    n = draw(st.integers(p + 3, 30))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.standard_t(2, size=n)
    h = rng.uniform(0.05, 1.0, size=n)
    return y, X, h


@st.composite
def _weights(draw):
    """Section 2.4's shape: candidate predictions that track the target, plus noise."""
    S = draw(st.integers(1, 6))
    n = draw(st.integers(4, 30))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    y = np.cumsum(rng.normal(size=n))
    P = y[:, None] + rng.normal(scale=rng.uniform(0.1, 2.0, size=S), size=(n, S)) + rng.normal(size=S)
    h = rng.uniform(0.05, 1.0, size=n)
    return y, P, h


def _pair_obj(y, X, h, b):
    i, j = O.pairs(len(y))
    e = y - X @ b
    return float(np.sum(h[i] * h[j] * np.abs(e[i] - e[j])))


def _level_obj(y, X, c, b):
    return float(np.sum(c * np.abs(y - X @ b)))


def _on_simplex(w):
    return np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-10


class TestNoSciPySolver:
    def test_the_oracle_imports_nothing_from_scipy_optimize(self):
        tree = ast.parse((STUDY / "oracle.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("scipy.optimize"), node.module
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("scipy.optimize") for a in node.names)

    def test_the_new_solvers_import_no_scipy(self):
        for name in ("lp.py", "solver_pairfree.py"):
            tree = ast.parse((STUDY / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                    assert not any(m.startswith("scipy") for m in mods), (name, mods)


class TestRankFits:
    @settings(max_examples=40, deadline=None)
    @given(_fit())
    def test_rank_fit_reaches_the_highs_optimum(self, prob):
        y, X, h = prob
        _, beta = O.rank_fit(y, X, h)
        i, j = O.pairs(len(y))
        ref = HR.wlad(y[i] - y[j], X[i] - X[j], h[i] * h[j])
        assert _pair_obj(y, X, h, beta) <= _pair_obj(y, X, h, ref) * (1 + 1e-9) + 1e-12

    @settings(max_examples=25, deadline=None)
    @given(_fit(max_p=4))
    def test_rank_lad_without_constraints_reaches_the_highs_optimum(self, prob):
        y, X, h = prob
        beta = LP.rank_lad(y, X, h)
        i, j = O.pairs(len(y))
        ref = HR.wlad(y[i] - y[j], X[i] - X[j], h[i] * h[j])
        assert _pair_obj(y, X, h, beta) <= _pair_obj(y, X, h, ref) * (1 + 1e-9) + 1e-12


class TestModelWeights:
    @settings(max_examples=60, deadline=None)
    @given(_weights())
    def test_pairwise_weights_reach_the_highs_optimum(self, prob):
        y, P, h = prob
        w = LP.simplex_rank_lad(y, P, h)
        assert _on_simplex(w)
        i, j = O.pairs(len(y))
        ref = HR.simplex_wlad(y[i] - y[j], P[i] - P[j], h[i] * h[j])
        assert _pair_obj(y, P, h, w) <= _pair_obj(y, P, h, ref) * (1 + 1e-9) + 1e-12

    @settings(max_examples=60, deadline=None)
    @given(_weights())
    def test_level_weights_reach_the_highs_optimum(self, prob):
        y, P, c = prob
        w = LP.simplex_lad(y, P, c)
        assert _on_simplex(w)
        ref = HR.simplex_wlad(y, P, c)
        assert _level_obj(y, P, c, w) <= _level_obj(y, P, c, ref) * (1 + 1e-9) + 1e-12

    def test_one_model_gets_all_the_weight(self):
        y = np.arange(5.0)
        P = (y + 1.0)[:, None]
        assert np.array_equal(LP.simplex_rank_lad(y, P, np.ones(5)), [1.0])
        assert np.array_equal(LP.simplex_lad(y, P, np.ones(5)), [1.0])


def _hard(rng):
    """Problems built to be degenerate: rounded data with exact ties, near-duplicate models."""
    S, n = int(rng.integers(1, 9)), int(rng.integers(4, 61))
    y = np.cumsum(rng.normal(size=n))
    kind = int(rng.integers(3))
    if kind == 0:
        P = (y + rng.normal(scale=0.5, size=n))[:, None] + rng.normal(scale=1e-3, size=(n, S))
    elif kind == 1:
        P = np.round(y[:, None] + rng.normal(size=(n, S)), 1)
        y = np.round(y, 1)
    else:
        P = y[:, None] + rng.normal(scale=rng.uniform(0.1, 3, S), size=(n, S)) + rng.normal(size=S)
    h = rng.uniform(0.05, 1, n) if rng.random() < 0.7 else np.ones(n)
    return y, P, h, kind


class TestDegenerateData:
    """Ties in the data put zero-residual pairs outside the basis. Before the
    solvers pivoted on perturbed data, 12 of 6,000 such problems hit the pivot
    limit or a non-descending edge, and one returned 3.6 times the optimum while
    certifying it."""

    def test_rounded_and_near_duplicate_problems(self):
        rng = np.random.default_rng(123)
        before = dict(LP.UNCERTIFIED)
        tied = 0
        for _ in range(400):
            y, P, h, kind = _hard(rng)
            tied += kind == 1
            i, j = O.pairs(len(y))
            w = LP.simplex_rank_lad(y, P, h)
            ref = HR.simplex_wlad(y[i] - y[j], P[i] - P[j], h[i] * h[j])
            f, fr = _pair_obj(y, P, h, w), _pair_obj(y, P, h, ref)
            assert _on_simplex(w) and f <= fr + 1e-9 * (1 + fr)
            w = LP.simplex_lad(y, P, h)
            ref = HR.simplex_wlad(y, P, h)
            f, fr = _level_obj(y, P, h, w), _level_obj(y, P, h, ref)
            assert _on_simplex(w) and f <= fr + 1e-9 * (1 + fr)
        assert tied > 100
        assert LP.UNCERTIFIED == before

    def test_rank_fits_on_rounded_data(self):
        from benchmarks.studies.rrma_ablation import solver_pairfree as PF

        rng = np.random.default_rng(321)
        before = PF.UNCERTIFIED[0]
        for _ in range(150):
            p, n = int(rng.integers(1, 6)), int(rng.integers(8, 50))
            X = np.round(rng.normal(size=(n, p)), 1)
            y = np.round(X @ rng.normal(size=p) + rng.standard_t(2, size=n), 1)
            h = np.ones(n) if rng.random() < 0.5 else rng.uniform(0.1, 1, n)
            _, beta = O.rank_fit(y, X, h)
            i, j = O.pairs(n)
            ref = HR.wlad(y[i] - y[j], X[i] - X[j], h[i] * h[j])
            f, fr = _pair_obj(y, X, h, beta), _pair_obj(y, X, h, ref)
            assert f <= fr + 1e-9 * (1 + fr)
        assert PF.UNCERTIFIED[0] == before


class TestTheOracleUsesThem:
    def test_gwcv_batched_matches_one_fit_at_a_time(self):
        """The warm-started batched sweep and a fresh fit per window agree."""
        from benchmarks.studies.rrma_ablation import simulate as S

        rng = np.random.default_rng(S.seed_for("b", 60, 25, 0))
        y, X = S.draw("b", 60, 30, 25, rng)
        T1 = 60
        groups = [np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([0, 5, 9, 20])]
        hs = [O.mallows_or_ones(X[:T1][:, g])[0] for g in groups]
        P = O.gwcv_predictions(y, X, T1, groups, O.rank_fit, hs)
        val = O.validation_window(T1)
        for k, t in enumerate(val[::7]):
            for s, g in enumerate(groups):
                a, b = O.rank_fit(y[:t], X[:t][:, g], hs[s][:t])
                assert abs(P[list(val).index(t), s] - (a + X[t, g] @ b)) < 1e-8

    def test_mscm_matches_scipy_nnls(self):
        from scipy.optimize import nnls

        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, 12)).cumsum(axis=0)
        y = X[:, :3] @ [0.5, 0.3, 0.2] + rng.normal(size=40)
        a, b = O.mscm_fit(y, X)
        Xc, yc = X - X.mean(0), y - y.mean()
        ref, _ = nnls(Xc, yc)
        assert np.all(b >= 0)
        assert np.sum((yc - Xc @ b) ** 2) <= np.sum((yc - Xc @ ref) ** 2) * (1 + 1e-9) + 1e-12

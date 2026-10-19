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
import warnings
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from benchmarks.studies.rrma_ablation import highs_reference as HR
from benchmarks.studies.rrma_ablation import lp as LP
from benchmarks.studies.rrma_ablation import oracle as O
from benchmarks.studies.rrma_ablation.solver_pairfree import CertificateWarning

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


@pytest.mark.filterwarnings("error::benchmarks.studies.rrma_ablation.solver_pairfree.CertificateWarning")
class TestDegenerateData:
    """Ties in the data put zero-residual pairs outside the basis. Before the
    solvers pivoted on perturbed data, 12 of 6,000 such problems hit the pivot
    limit or a non-descending edge, and one returned 3.6 times the optimum while
    certifying it."""

    def test_rounded_and_near_duplicate_problems(self):
        rng = np.random.default_rng(123)
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

    def test_rank_fits_on_rounded_data(self):
        rng = np.random.default_rng(321)
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


def _strict(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error", CertificateWarning)
        return fn(*args)


class TestTheCertificateIsReported:
    """A failed certificate used to bump a module counter nobody read."""

    def test_rank_lad_warns(self, monkeypatch):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 2))
        y = X[:, 0] + rng.normal(size=20)
        monkeypatch.setattr(LP, "_gap_ok", lambda *a, **k: False)
        with pytest.warns(CertificateWarning, match="rank_lad"):
            LP.rank_lad(y, X, np.ones(20))

    def test_lad_warns(self, monkeypatch):
        rng = np.random.default_rng(2)
        y, P = rng.normal(size=20), rng.normal(size=(20, 3))
        monkeypatch.setattr(LP, "_gap_ok", lambda *a, **k: False)
        with pytest.warns(CertificateWarning, match="lad"):
            LP.simplex_lad(y, P, np.ones(20))

    def test_there_is_no_module_level_counter(self):
        assert not hasattr(LP, "UNCERTIFIED")


def _witness_shared_rate_scale():
    """With one floored tolerance for both kinds of pivot rate, weights scaled by
    1e-6 stop ``simplex_lad`` 3e-5 from the optimum here, uncertified. Found by
    searching against the mutant at the 229th seed, which is why 30 random draws
    caught it in one mutation run and missed it in the next."""
    rng = np.random.default_rng(228)
    S, n = int(rng.integers(2, 7)), int(rng.integers(4, 31))
    y = np.cumsum(rng.normal(size=n))
    P = y[:, None] + rng.normal(scale=rng.uniform(0.1, 2.0, size=S), size=(n, S)) + rng.normal(size=S)
    return y, P, rng.uniform(0.05, 1.0, size=n)


def _floor(y, P, h, pair, level=0.0):
    """The rounding floor of an objective: 64 eps times the size of its terms.

    An exact fit has optimum zero, computed as rounding noise, so a comparison
    relative to the optimum alone means nothing there; the floor is built from
    the data the terms are computed from, plus any level ``c`` they carried.
    """
    W = h.sum() ** 2 if pair else h.sum()
    return 64 * np.finfo(float).eps * W * (np.abs(y).max() + np.abs(P).max() + level)


class TestUnits:
    """Nocedal and Wright (2006), pp. 26-27: the answer must not depend on units.
    With ``max(1, .)`` under every tolerance, data at a scale of 1e-7 stopped
    0.3 to 0.6 percent from the optimum, weights scaled by 1e-6 stopped
    ``simplex_lad`` 19 percent from it, and data at a level of 1e8 (where the
    perturbation, sized by ``max|y|``, was 0.1) 18 to 26 percent from it."""

    @staticmethod
    def _check(y, P, h, ys, Ps, hs, level=0.0):
        i, j = O.pairs(len(y))
        w = _strict(LP.simplex_rank_lad, ys, Ps, hs)
        fr = _pair_obj(y, P, h, HR.simplex_wlad(y[i] - y[j], P[i] - P[j], h[i] * h[j]))
        assert _on_simplex(w)
        assert _pair_obj(y, P, h, w) <= fr * (1 + 1e-9) + _floor(y, P, h, True, level)
        w = _strict(LP.simplex_lad, ys, Ps, hs)
        fr = _level_obj(y, P, h, HR.simplex_wlad(y, P, h))
        assert _on_simplex(w)
        assert _level_obj(y, P, h, w) <= fr * (1 + 1e-9) + _floor(y, P, h, False, level)

    @settings(max_examples=30, deadline=None)
    @given(_weights(), st.integers(-8, 8))
    def test_rescaling_changes_nothing(self, b, k):
        y, P, h = b
        self._check(y, P, h, 10.0 ** k * y, 10.0 ** k * P, h)

    @settings(max_examples=30, deadline=None)
    @given(_weights(), st.integers(-6, 6))
    @example(_witness_shared_rate_scale(), -6)
    def test_rescaling_the_weights_changes_nothing(self, b, k):
        y, P, h = b
        self._check(y, P, h, y, P, 10.0 ** k * h)

    @settings(max_examples=30, deadline=None)
    @given(_weights(), st.sampled_from([1e2, 1e4, 1e6, 1e8]))
    def test_a_common_level_changes_nothing(self, b, c):
        """On the simplex ``(y + c) - (P + c) w = y - P w``, so both programmes are
        unchanged; the data are stored to ``c * eps``, which enters the floor."""
        y, P, h = b
        self._check(y, P, h, y + c, P + c, h, level=c)


class TestTheGapTestAtEveryScale:
    """The certificates read ``max(1, F)`` under their gap tolerance, so once the
    objective fell below 1 they certified points far from the optimum. At a scale
    of 1e-7 the objectives are about 1e-5: with the optimal dual held fixed,
    points 1e-6 to 1e-4 worse than the optimum have gaps between the scaled
    tolerance (about 1e-13) and a floored one (1e-9), and must fail. A stubbed
    pivot loop cannot test this: a vertex that is not optimal fails on its
    multipliers or its cuts before the gap decides."""

    @staticmethod
    def _problem():
        rng = np.random.default_rng(31)
        n, S = 25, 4
        y = 1e-7 * np.cumsum(rng.normal(size=n))
        P = y[:, None] + 1e-7 * rng.normal(scale=[0.3, 0.6, 1.0, 1.5], size=(n, S))
        return y, P, rng.uniform(0.2, 1.0, size=n)

    @staticmethod
    def _worse(F, w):
        S = len(w)
        out = []
        for j in range(S):
            for t in np.logspace(-8, -1, 50):
                b = (1 - t) * w + t * np.eye(S)[j]
                if 1e-6 < (F(b) - F(w)) / F(w) < 1e-4:
                    out.append(b)
        assert len(out) >= 5
        return out

    def test_rank_lad(self):
        from benchmarks.studies.rrma_ablation import solver_pairfree as PF

        y, P, h = self._problem()
        n, S = P.shape
        yc, Pc = PF._centre(y, P)
        yp = yc + PF.perturbation(yc)
        G, g, E, f = np.eye(S), np.zeros(S), np.ones((1, S)), np.ones(1)
        F = lambda b: _pair_obj(yc, Pc, h, b)
        s0 = int(np.argmin([F(np.eye(S)[s]) for s in range(S)]))
        labels, active = LP._rank_lad_pivot(yp, Pc, h, G, g, E, f, np.arange(n),
                                            [k for k in range(S) if k != s0], 5000)
        w = LP._rank_vertex(yc, Pc, G, g, E, f, labels, active)
        dual = LP._rank_dual(yp, Pc, h, G, g, E, f, labels, active)
        ok = lambda b: LP._rank_certified(yc, Pc, h, G, g, E, f, b, active, *dual)
        assert ok(w)
        assert not any(ok(b) for b in self._worse(F, w))

    def test_lad(self):
        from benchmarks.studies.rrma_ablation import solver_pairfree as PF

        y, P, c = self._problem()
        n, S = P.shape
        yp = y + PF.perturbation(y)
        G, g, E, f = np.eye(S), np.zeros(S), np.ones((1, S)), np.ones(1)
        F = lambda b: _level_obj(y, P, c, b)
        s0 = int(np.argmin([F(np.eye(S)[s]) for s in range(S)]))
        B, active = LP._lad_pivot(yp, P, c, G, g, E, f, [], [k for k in range(S) if k != s0], 5000)
        M = np.vstack([P[B], G[active], E])
        w = np.linalg.solve(M, np.r_[y[B], g[active], f])
        dual = LP._lad_dual(yp, P, c, G, g, E, f, B, active)
        ok = lambda b: LP._lad_certified(y, P, c, G, g, E, f, b, active, *dual)
        assert ok(w)
        assert not any(ok(b) for b in self._worse(F, w))


class TestANonOptimalVertexIsReportedAtEveryScale:
    """End to end: stub the pivot loop so it returns its starting vertex (the best
    single model, not the optimum here) and the solver must warn, at any scale.
    At a scale of 1e-12 the gap is far below 1, so a tolerance floored at 1
    passes it; that floor sat in the call to ``_gap_ok``, not inside it. The
    reference optimum comes from the unit-scale problem (both objectives are
    homogeneous of degree one in the scale): at 1e-12, HiGHS's own absolute
    tolerances stop it at the starting vertex."""

    @staticmethod
    def _problem(scale):
        rng = np.random.default_rng(41)
        n, S = 30, 4
        y = np.cumsum(rng.normal(size=n))
        P = y[:, None] + rng.normal(scale=[0.5, 0.8, 1.0, 1.4], size=(n, S))
        return scale * y, scale * P, rng.uniform(0.2, 1.0, size=n)

    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
    def test_rank_lad(self, monkeypatch, scale):
        y, P, h = self._problem(scale)
        y1, P1, _ = self._problem(1.0)
        i, j = O.pairs(len(y))
        start = np.eye(P.shape[1])[np.argmin([_pair_obj(y, P, h, e) for e in np.eye(P.shape[1])])]
        best = scale * _pair_obj(y1, P1, h, HR.simplex_wlad(y1[i] - y1[j], P1[i] - P1[j], h[i] * h[j]))
        assert _pair_obj(y, P, h, start) > best * (1 + 1e-3)          # the start is not optimal
        monkeypatch.setattr(LP, "_rank_lad_pivot", lambda *a: (a[7], a[8]))
        with pytest.warns(CertificateWarning, match="rank_lad"):
            LP.simplex_rank_lad(y, P, h)

    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
    def test_lad(self, monkeypatch, scale):
        y, P, h = self._problem(scale)
        y1, P1, _ = self._problem(1.0)
        start = np.eye(P.shape[1])[np.argmin([_level_obj(y, P, h, e) for e in np.eye(P.shape[1])])]
        best = scale * _level_obj(y1, P1, h, HR.simplex_wlad(y1, P1, h))
        assert _level_obj(y, P, h, start) > best * (1 + 1e-3)
        monkeypatch.setattr(LP, "_lad_pivot", lambda *a: (a[7], a[8]))
        with pytest.warns(CertificateWarning, match="lad"):
            LP.simplex_lad(y, P, h)


class TestTheRatioTest:
    """Russinoff (2022), Lemma 6.45: a slack computed from operands of size ``m``
    is known to about ``m * eps``. A constraint one rounding inside its boundary
    is on it, and a constraint violated by more than the tolerance is an error,
    not a slack to clip to zero."""

    def test_a_slack_inside_the_rounding_is_zero(self):
        G = np.array([[1.0, 0.0], [0.0, 1.0]])
        g = np.zeros(2)
        beta = np.array([0.7, 3 * np.finfo(float).eps * 0.7])
        t, k = LP._first_hit(G, g, beta, np.array([1.0, -1.0]), [0, 1])
        assert (t, k) == (0.0, 1)

    def test_a_violated_inactive_inequality_is_an_error(self):
        G, g = np.eye(2), np.zeros(2)
        with pytest.raises(RuntimeError, match="violated"):
            LP._first_hit(G, g, np.array([0.5, -1e-6]), np.array([1.0, -1.0]), [0, 1])


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

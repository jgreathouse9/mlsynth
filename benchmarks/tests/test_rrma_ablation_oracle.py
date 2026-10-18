"""The RRMA study's oracle: the HiGHS reference and the Basque R reference.

The study is ``benchmarks/studies/rrma_ablation``, a replication spike for Lv &
Li (2026), arXiv:2610.15553v1. Its estimator path solves every linear programme
with its own NumPy code (``solver_pairfree``, ``lp``); HiGHS is kept in
``highs_reference`` as the check the solver tests compare against, so the
reference itself carries assertions here.

``highs_reference.wlad`` and ``simplex_wlad`` solve the dual of a weighted
least-absolute-deviation programme and read the coefficients off the dual
multipliers. The sign of those multipliers is a convention of the solver, and
the negated reading returns a point whose objective is two to fourteen times the
optimum, so the tests compare against the primal programme by objective, not by
coefficient: an L1 programme can have a face of minimisers, and any point of the
face is correct.

The Basque m = 2 path is the one place Algorithm 1 is deterministic on the
paper's panel. It is pinned against ``benchmarks/reference/rrma_basque``, an
independent R implementation calling ``MASS::cov.rob``, ``quantreg::rq`` and
``lpSolve``, so the study's own solvers are checked against R's as well as
against HiGHS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import linprog

from benchmarks.studies.rrma_ablation import highs_reference as HR
from benchmarks.studies.rrma_ablation import oracle as O

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "benchmarks" / "reference" / "rrma_basque" / "reference.json"


def _primal_lad(y, D, c, simplex=False):
    """The primal programme, slow and literal: the objective the dual must reach."""
    n, p = D.shape
    cost = np.r_[np.zeros(p), c, c]
    A_eq = np.hstack([D, np.eye(n), -np.eye(n)])
    b_eq = y
    bounds = [(0, None) if simplex else (None, None)] * p + [(0, None)] * (2 * n)
    if simplex:
        A_eq = np.vstack([A_eq, np.r_[np.ones(p), np.zeros(2 * n)]])
        b_eq = np.r_[y, 1.0]
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.fun


def _objective(y, D, c, b):
    return float(np.sum(c * np.abs(y - D @ b)))


@st.composite
def _lad_problem(draw):
    n = draw(st.integers(4, 30))
    p = draw(st.integers(1, 4))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n, p))
    y = D @ rng.normal(size=p) + rng.standard_t(2, size=n)
    c = rng.uniform(0.05, 1.0, size=n)
    return y, D, c


class TestTheDualProgrammes:
    """Dual multipliers, read with the right sign, reach the primal optimum."""

    @settings(max_examples=60, deadline=None)
    @given(_lad_problem())
    def test_wlad_reaches_the_primal_optimum(self, prob):
        y, D, c = prob
        b = HR.wlad(y, D, c)
        best = _primal_lad(y, D, c)
        assert _objective(y, D, c, b) <= best * (1 + 1e-9) + 1e-9

    @settings(max_examples=60, deadline=None)
    @given(_lad_problem())
    def test_simplex_wlad_is_on_the_simplex_and_optimal(self, prob):
        y, D, c = prob
        w = HR.simplex_wlad(y, D, c)
        assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
        best = _primal_lad(y, D, c, simplex=True)
        assert _objective(y, D, c, w) <= best * (1 + 1e-7) + 1e-7

    def test_the_negated_reading_is_not_optimal(self):
        """The defect a sign slip would ship: the mirror point is far from optimal."""
        rng = np.random.default_rng(3)
        D = rng.normal(size=(40, 3))
        y = D @ np.array([1.0, -2.0, 0.5]) + rng.standard_t(3, size=40)
        c = np.ones(40)
        b = HR.wlad(y, D, c)
        assert _objective(y, D, c, -b) > 1.5 * _objective(y, D, c, b)


class TestWhenHiGHSFails:
    """HiGHS can return an unrecognised status (15, model status Unknown) on one
    programme among tens of thousands; the first 500-replication run died on it.
    The reference retries, and the estimator path does not call HiGHS at all."""

    def _problem(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(30, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_t(3, size=30)
        h = rng.uniform(0.3, 1.0, size=30)
        return y, X, h

    def test_the_reference_retries_with_another_highs_method(self, monkeypatch):
        y, X, h = self._problem()
        i, j = O.pairs(len(y))
        r, D, c = y[i] - y[j], X[i] - X[j], h[i] * h[j]
        best = HR.wlad(r, D, c)
        real, calls = HR.linprog, []

        def flaky(*args, **kwargs):
            calls.append(kwargs.get("method"))
            res = real(*args, **kwargs)
            if len(calls) == 1:
                res.status = 4
            return res

        monkeypatch.setattr(HR, "linprog", flaky)
        before = HR.LP_FALLBACKS["retry"]
        b = HR.wlad(r, D, c)
        assert len(calls) == 2 and calls[1] != calls[0]
        assert HR.LP_FALLBACKS["retry"] == before + 1
        assert _objective(r, D, c, b) <= _objective(r, D, c, best) * (1 + 1e-9)

    def test_rank_fit_does_not_need_highs(self, monkeypatch):
        y, X, h = self._problem()
        i, j = O.pairs(len(y))
        r, D, c = y[i] - y[j], X[i] - X[j], h[i] * h[j]
        best = HR.wlad(r, D, c)

        def broken(*args, **kwargs):
            raise AssertionError("the estimator path called HiGHS")

        monkeypatch.setattr(HR, "linprog", broken)
        alpha, beta = O.rank_fit(y, X, h)
        assert _objective(r, D, c, beta) <= _objective(r, D, c, best) * (1 + 1e-9) + 1e-12
        assert abs(alpha - float(np.median(y - X @ beta))) < 1e-12


class TestCandidateModels:
    """Equation (14) and the fallback Section 4.1 does not cover."""

    def test_the_last_group_takes_the_remainder(self):
        groups = O.partition(np.arange(11), 3)
        assert [len(g) for g in groups] == [3, 3, 5]
        assert np.array_equal(np.concatenate(groups), np.arange(11))

    def test_fewer_survivors_than_the_group_size_gives_no_groups(self):
        assert O.partition(np.arange(2), 3) == []

    def test_candidate_sets_fall_back_to_one_model(self):
        sets = O.candidate_sets(np.array([4, 1]), 6, range(3, 11))
        assert list(sets) == [2] and np.array_equal(sets[2][0], [4, 1])
        sets = O.candidate_sets(np.array([], dtype=int), 6, range(3, 11))
        assert np.array_equal(sets[6][0], np.arange(6))


@pytest.mark.skipif(not REFERENCE.exists(), reason="R reference not captured")
class TestBasqueAgainstR:
    """The m = 2 path, every intermediate, against the R implementation."""

    @pytest.fixture(scope="class")
    def port(self):
        from benchmarks.studies.rrma_ablation.run import basque

        y, X, T1, _ = basque("regionno")
        path = O.candidate_paths(y, X, T1, [2])[2]
        choice = O.choose({2: path}, y, T1, O.weigh_pairwise)
        return y, X, T1, path, choice

    @pytest.fixture(scope="class")
    def ref(self):
        return json.loads(REFERENCE.read_text())

    def test_the_candidate_models_are_rs(self, port, ref):
        assert [g.tolist() for g in port[3].groups] == ref["groups"]

    def test_mallows_weights_match_cov_rob(self, port, ref):
        assert np.max(np.abs(np.asarray(port[3].hs) - ref["h"])) < 1e-9

    def test_cross_validated_predictions_match(self, port, ref):
        assert np.max(np.abs(port[3].P - np.asarray(ref["P"]))) < 1e-9

    def test_model_weights_and_effect_match(self, port, ref):
        y, X, T1, path, choice = port
        assert np.max(np.abs(choice.w - ref["w"])) < 1e-9
        att = float(np.mean(y[T1:] - choice.counterfactual[T1:]))
        assert abs(att - ref["att"]) < 1e-9

"""The RRMA study's distribution functions, against R and against SciPy.

``benchmarks/studies/rrma_ablation/special.py`` replaces the three
``scipy.stats`` calls the oracle made: the chi-square quantile (``cov.rob``'s
cut and the Mallows weights), the normal quantile (equation (15)) and the
Kendall test (the donor screen of Section 4.1). Each is checked against R's own
function, captured once in ``benchmarks/reference/rrma_special`` (``qnorm``,
``qchisq``, ``pchisq``, ``cor.test``), and against SciPy, which the tests may
import and the estimator path may not.

R computes the upper tail of the exact Kendall distribution as ``1 - P``, which
loses every digit once the p-value falls near ``1e-16`` (it reported
``-4.4e-16`` for a Basque donor). The study sums the tail directly in exact
integer arithmetic, so the comparison with R stops where R's own value stops
being accurate, and below that the exact distribution is checked against
enumerating every permutation.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from benchmarks.studies.rrma_ablation import special as SP

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "benchmarks" / "studies" / "rrma_ablation"
REF = ROOT / "benchmarks" / "reference" / "rrma_special" / "reference.json"


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


class TestNoSciPyAnywhere:
    @pytest.mark.parametrize("name", ["oracle.py", "lp.py", "solver_pairfree.py", "special.py"])
    def test_the_estimator_path_imports_no_scipy(self, name):
        tree = ast.parse((STUDY / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(m.startswith("scipy") for m in mods), (name, mods)


@pytest.mark.skipif(not REF.exists(), reason="R reference not captured")
class TestAgainstR:
    @pytest.fixture(scope="class")
    def ref(self):
        return json.loads(REF.read_text())

    def test_qnorm(self, ref):
        worst = max(_rel(SP.norm_ppf(p), q) for p, q in ref["qnorm"])
        assert worst <= 4e-16

    def test_qchisq(self, ref):
        worst = max(_rel(SP.chi2_ppf(p, df), q) for p, df, q in ref["qchisq"])
        assert worst <= 1e-13

    def test_pchisq_both_tails(self, ref):
        for x, df, lower, upper in ref["pchisq"]:
            assert _rel(SP.chi2_cdf(x, df), lower) <= 1e-13
            assert _rel(SP.chi2_sf(x, df), upper) <= 1e-13

    def test_cor_test_kendall(self, ref):
        """R's exact branch forms its tail as ``1 - P``, accurate to a few ulps of 1
        in absolute terms: the measured gap is 5.5 eps at worst, so the bound is
        8 eps absolute. The normal branch agrees to 0.5 eps; the bound is 1e-14
        relative."""
        eps = np.finfo(float).eps
        exact = normal = 0
        for case in ref["kendall"]:
            p = SP.kendall_pvalue(np.array(case["x"]), np.array(case["y"]))
            if case["method"] == "exact":
                assert abs(p - case["p"]) <= 8 * eps, case["name"]
                exact += 1
            else:
                assert _rel(p, case["p"]) <= 1e-14, case["name"]
                normal += 1
        assert exact >= 30 and normal >= 10

    def test_the_basque_screen_order_is_rs(self, ref):
        """R's exact tails are quantised near 1e-15 (one is negative); the order survives."""
        cases = [c for c in ref["kendall"] if c["name"].startswith("basque")]
        ours = [SP.kendall_pvalue(np.array(c["x"]), np.array(c["y"])) for c in cases]
        r_order = np.argsort([c["p"] for c in cases], kind="stable")
        assert np.array_equal(np.argsort(ours, kind="stable"), r_order)
        assert min(ours) > 0


class TestNormal:
    @settings(max_examples=300, deadline=None)
    @given(st.floats(1e-300, float(np.nextafter(1.0, 0.0))))
    def test_matches_scipy(self, p):
        from scipy.stats import norm

        assert _rel(SP.norm_ppf(p), norm.ppf(p)) <= 1e-14 or abs(SP.norm_ppf(p)) < 1e-300

    def test_symmetry_and_the_tail(self):
        for p in (1e-15, 1e-10, 1e-5, 0.025, 0.3):
            # Symmetry is a statement about a representable u: 1 - u is exact for
            # u >= 1/2 (Sterbenz), while the double nearest 1 - p can move a 1e-15
            # tail by 10 percent.
            u = 1.0 - p
            assert _rel(SP.norm_ppf(u), -SP.norm_ppf(1.0 - u)) <= 1e-14
        eps = np.finfo(float).eps
        for p in (1e-300, 1e-100, 1e-20, 1e-5, 0.025, 0.3):
            # The upper tail recovers p without passing through 1 - p. The round
            # trip is ill-conditioned: d log Phi(-z) / d log z = -z^2 roughly, so a
            # quantile right to an ulp or two moves the tail by about z^2 ulps.
            z = SP.norm_ppf(p)
            assert _rel(SP.norm_sf(-z), p) <= 8 * eps * (1 + z * z)

    def test_rejects_probabilities_outside_the_open_interval(self):
        for p in (0.0, 1.0, -0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                SP.norm_ppf(p)


class TestChiSquare:
    @settings(max_examples=300, deadline=None)
    @given(st.floats(1e-12, 1 - 1e-12), st.integers(1, 80))
    def test_quantile_matches_scipy(self, q, df):
        from scipy.stats import chi2

        assert _rel(SP.chi2_ppf(q, df), chi2.ppf(q, df)) <= 1e-11

    @settings(max_examples=300, deadline=None)
    @given(st.floats(1e-12, 1 - 1e-12), st.integers(1, 80))
    def test_quantile_inverts_the_cdf_in_its_smaller_tail(self, q, df):
        x = SP.chi2_ppf(q, df)
        if q <= 0.5:
            assert _rel(SP.chi2_cdf(x, df), q) <= 1e-12
        else:
            assert _rel(SP.chi2_sf(x, df), 1 - q) <= 1e-12

    def test_rejects_bad_arguments(self):
        for q, df in ((0.0, 2), (1.0, 2), (0.5, 0), (0.5, -1)):
            with pytest.raises(ValueError):
                SP.chi2_ppf(q, df)


class TestTheQuantileIteration:
    """Nocedal and Wright (2006), p. 615 (differencing similar numbers) and p. 62
    (a safeguarded step stays clear of the bracket's ends; the iteration stops on
    a relative change near machine precision)."""

    @pytest.fixture
    def counted(self, monkeypatch):
        n = {"calls": 0}
        cdf, sf = SP.chi2_cdf, SP.chi2_sf

        def c(x, df):
            n["calls"] += 1
            return cdf(x, df)

        def s(x, df):
            n["calls"] += 1
            return sf(x, df)

        monkeypatch.setattr(SP, "chi2_cdf", c)
        monkeypatch.setattr(SP, "chi2_sf", s)
        return n

    def test_newton_does_the_work(self, counted):
        """Newton on ``log f - log q`` stalled: two logs near -18.4 agree to their
        last bit when ``f`` and ``q`` differ by 2e-15, the step came out 0, and
        bisection took 55 more evaluations at ``q = 1e-8``, ``df = 2``. The grid's
        worst is now 6; the bound is 10."""
        worst = 0
        for df in (0.5, 1, 1.5, 2, 3, 5, 8, 20, 80, 300):
            for q in (1e-300, 1e-100, 1e-20, 1e-8, 1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 0.975,
                      1 - 1e-8, 1 - 1e-15):
                counted["calls"] = 0
                SP.chi2_ppf(q, df)
                worst = max(worst, counted["calls"])
        assert worst <= 10

    def test_a_misjudged_slope_still_converges(self, monkeypatch):
        """A derivative 1000 times too large makes every Newton step 1000 times too
        short. Inside the bracket each one is accepted; the safeguard bisects
        whenever two steps fail to halve the bracket."""
        logpdf = SP._chi2_logpdf
        monkeypatch.setattr(SP, "_chi2_logpdf", lambda x, df: logpdf(x, df) + math.log(1000.0))
        for q, df in ((0.3, 4), (0.95, 2), (1e-6, 3)):
            x = SP.chi2_ppf(q, df)
            tail = SP.chi2_cdf(x, df) if q <= 0.5 else SP.chi2_sf(x, df)
            assert _rel(tail, min(q, 1 - q)) <= 1e-12

    @settings(max_examples=300, deadline=None)
    @given(st.floats(5e-324, float(np.nextafter(1.0, 0.0))))
    # the counterexamples the property found, kept so they do not depend on the database
    @example(3.4209794462655953e-23)
    @example(1.4718929084089382e-52)
    @example(1.6961562603034342e-305)
    @example(3.041717799945633e-308)
    @example(0.86328125)
    @example(0.8395644640923099)
    def test_two_degrees_of_freedom_in_closed_form(self, q):
        """``chi2_2`` is exponential with mean 2, so ``x = -2 log1p(-q)``, down to
        subnormal ``q`` where the derivative ratio ``pdf / P`` overflows.

        A quantile is as accurate as its tail divided by ``kappa = P / (x pdf)``
        (the smaller tail), its condition number. Over 15,000 ``q`` from 1e-323
        to ``1 - 1e-16`` the worst error was ``6.0 eps (1 + kappa)`` relative, at
        ``q = 0.84`` where ``Q = 1 - P`` multiplies the series' error by
        ``P / Q``, and zero in the subnormal range; the bound is 12, plus one
        subnormal step. Before the fixes it failed by ``1.4e-37`` at
        ``q = 3.4e-23`` (Newton stalled on ``log f - log q``) and by 188 eps at
        the edge of the normal range (the prefactor through logs)."""
        ref = -2.0 * math.log1p(-q)
        x = SP.chi2_ppf(q, 2)
        kappa = min(q, 1 - q) / (ref * math.exp(-ref / 2) / 2) if ref > 0 else 1.0
        assert abs(x - ref) <= 12 * np.finfo(float).eps * (1 + kappa) * ref + 5e-324


def _brute_tail(n, q):
    """P(T >= q) for T the number of concordant pairs of a random permutation, by enumeration."""
    hits = total = 0
    for perm in itertools.permutations(range(n)):
        conc = sum(perm[i] < perm[j] for i in range(n) for j in range(i + 1, n))
        hits += conc >= q
        total += 1
    return Fraction(hits, total)


class TestKendall:
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
    def test_exact_counts_equal_enumeration(self, n):
        T0 = n * (n - 1) // 2
        for q in range(T0 + 1):
            assert SP.kendall_upper_tail(q, n) == _brute_tail(n, q)

    def test_the_far_tail_keeps_its_digits(self):
        """n = 20, all pairs concordant: P = 1/20!, which 1 - P cannot represent."""
        n = 20
        p = SP.kendall_pvalue(np.arange(n, dtype=float), np.arange(n, dtype=float))
        assert _rel(p, 2 / math.factorial(n)) <= 1e-15

    @settings(max_examples=150, deadline=None)
    @given(st.integers(5, 45), st.integers(0, 2**31 - 1))
    def test_exact_branch_matches_scipy(self, n, seed):
        from scipy.stats import kendalltau

        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=n), rng.normal(size=n)
        ref = kendalltau(x, y, method="exact").pvalue
        p = SP.kendall_pvalue(x, y)
        assert 0.0 <= p <= 1.0
        assert _rel(p, ref) <= 1e-9 or ref < 1e-12

    @settings(max_examples=150, deadline=None)
    @given(st.integers(5, 120), st.integers(0, 2**31 - 1))
    def test_normal_branch_with_ties_matches_scipy(self, n, seed):
        from scipy.stats import kendalltau

        rng = np.random.default_rng(seed)
        k = max(2, n // 3)                       # fewer values than observations: ties for sure
        x, y = rng.integers(0, k, size=n).astype(float), rng.integers(0, k, size=n).astype(float)
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return
        ref = kendalltau(x, y, method="asymptotic").pvalue
        assert _rel(SP.kendall_pvalue(x, y), ref) <= 1e-10 or ref < 1e-300

    def test_the_oracle_screens_with_it(self):
        from benchmarks.studies.rrma_ablation import oracle as O

        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 6)).cumsum(axis=0)
        y = X[:, 0] + rng.normal(size=30)
        p = O.kendall_pvalues(y, X)
        assert np.array_equal(p, [SP.kendall_pvalue(y, X[:, j]) for j in range(6)])

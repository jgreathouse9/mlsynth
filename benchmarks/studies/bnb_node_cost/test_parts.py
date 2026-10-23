"""Checks of the candidates for a cheaper node (#844).

Two candidates claim to compute the library's numbers bit for bit, faster:
the cut matrix by ``bincount`` and the rounding of a dyadic matrix by
integer division. The checks are that they do, over their input domain. The
third, Rump's floating-point test, claims a different but rigorous bound on
the smallest eigenvalue; the check is in exact arithmetic, that the matrix
less the bound is positive semidefinite.
"""
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from mlsynth.utils.marex_helpers import branch as marex_branch
from mlsynth.utils.marex_helpers.certify import Dyadic
from mlsynth.utils.syndes_helpers import branch

from . import parts


def _cuts(rng, size, count):
    T = branch._triples(size)
    pick = rng.choice(len(T), size=min(count, len(T)), replace=True)
    t = rng.integers(1, 5, size=len(pick))
    return branch.Cuts(T[pick, 0].copy(), T[pick, 1].copy(), T[pick, 2].copy(), t)


def _psd_exact(M) -> bool:
    """Whether the rational symmetric ``M`` is positive semidefinite, by
    exact symmetric elimination: every pivot nonnegative, and a zero pivot
    with a zero row."""
    M = [list(row) for row in M]
    n = len(M)
    for k in range(n):
        p = M[k][k]
        if p < 0:
            return False
        if p == 0:
            if any(M[k][j] != 0 for j in range(k + 1, n)):
                return False
            continue
        for i in range(k + 1, n):
            f = M[i][k] / p
            for j in range(k + 1, n):
                M[i][j] -= f * M[k][j]
    return True


# ----------------------------------------------------------------------
# the cut matrix
# ----------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(st.integers(3, 30), st.integers(0, 600), st.integers(0, 10 ** 6))
def test_property_the_cut_matrix_by_bincount_is_the_librarys_bit_for_bit(size, count, seed):
    """Repeated cuts included: the same cell summed in the same order."""
    rng = np.random.default_rng(seed)
    cuts = _cuts(rng, size, count)
    w = rng.standard_normal(len(cuts)) * 10.0 ** rng.integers(-8, 8, size=len(cuts))
    got = parts.cut_matrix(size, cuts, w)
    assert np.array_equal(got, branch._cut_matrix(size, cuts, w))


def test_no_cuts_is_the_zero_matrix():
    got = parts.cut_matrix(5, branch.Cuts.empty(), np.zeros(0))
    assert got.shape == (5, 5) and not got.any()


# ----------------------------------------------------------------------
# the dual evaluation
# ----------------------------------------------------------------------


def _state(rng, n, count):
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2
    b = rng.standard_normal(n)
    y = rng.standard_normal(2 * n + 2)
    cuts = _cuts(rng, n + 1, count)
    v = np.abs(rng.standard_normal(len(cuts)))
    v[rng.random(len(v)) < 0.3] = 0.0
    return float(rng.standard_normal()), b, A, int(rng.integers(-n, n + 1)), y, cuts, v, \
        float(10.0 ** rng.uniform(-5, -1))


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 30), st.integers(0, 400), st.integers(0, 10 ** 6))
def test_property_the_lean_evaluation_is_the_librarys_bit_for_bit(n, count, seed):
    args = _state(np.random.default_rng(seed), n, count)
    want = branch._theta(*args)
    got = parts.theta(*args)
    assert got[0] == want[0] and got[1] == want[1]
    for g, w in zip(got[2:], want[2:]):
        assert np.array_equal(g, w)


def test_the_lean_evaluation_without_cuts_is_the_librarys():
    args = _state(np.random.default_rng(1), 6, 0)
    want, got = branch._theta(*args), parts.theta(*args)
    assert got[3].shape == (0,) and np.array_equal(got[4], want[4])


def test_a_lift_with_the_lean_parts_is_the_librarys_bit_for_bit():
    """The lift's bound, multipliers and evaluations are unchanged when its
    evaluation and cut matrix are swapped for the lean ones."""
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 14)) + 0.3 * rng.standard_normal((6, 14))
    out = parts.lean_lift(Y, 5, depths=(0, 3))
    assert [r["fixed"] for r in out] == [0, 3]
    for r in out:
        assert r["identical"] and r["evaluations"] > 0
        assert r["seconds"]["library"] > 0 and r["seconds"]["lean"] > 0


# ----------------------------------------------------------------------
# rounding a dyadic matrix
# ----------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 6), st.integers(0, 1200), st.integers(1, 2400), st.integers(0, 10 ** 6))
def test_property_rounding_by_division_is_the_librarys_bit_for_bit(n, e, bits, seed):
    """Every size of numerator and exponent, overflow to an infinity and
    underflow to a subnormal or zero included."""
    rng = np.random.default_rng(seed)
    N = np.array([int(rng.integers(0, 2 ** 62)) << max(bits - 62, 0) for _ in range(n * n)],
                 dtype=object) * np.array(rng.choice([-1, 1], size=n * n), dtype=object)
    D = Dyadic(N.reshape(n, n), e)
    want = marex_branch._float(D)
    got = parts.to_float(D)
    assert np.array_equal(got, want)
    assert np.array_equal(np.signbit(got), np.signbit(want))


def test_rounding_overflows_to_signed_infinities():
    D = Dyadic(np.array([[1 << 1100, -(1 << 1100)], [3, 0]], dtype=object), 2)
    got = parts.to_float(D)
    assert got[0, 0] == np.inf and got[0, 1] == -np.inf and got[1, 0] == 0.75


# ----------------------------------------------------------------------
# Rump's floating-point test
# ----------------------------------------------------------------------


def _symmetric_dyadic(rng, n, e):
    M = rng.integers(-2 ** 40, 2 ** 40, size=(n, n)).astype(object)
    M = M + M.T
    return Dyadic(M, e)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 7), st.integers(0, 80), st.integers(0, 10 ** 6))
def test_property_rumps_bound_is_rigorous(n, e, seed):
    """``D - bound I`` is positive semidefinite in exact arithmetic."""
    D = _symmetric_dyadic(np.random.default_rng(seed), n, e)
    low = parts.rump_lower(D)
    assert low is not None
    exact = [[Fraction(int(D.N[i, j]), 1 << D.e) - (low if i == j else 0)
              for j in range(n)] for i in range(n)]
    assert _psd_exact(exact)


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 7), st.integers(0, 80), st.integers(0, 10 ** 6))
def test_property_rumps_bound_is_as_tight_as_the_librarys_to_rounding(n, e, seed):
    D = _symmetric_dyadic(np.random.default_rng(seed), n, e)
    rump, lib = parts.rump_lower(D), marex_branch.lambda_min_lower(D)
    lam = float(np.linalg.eigvalsh(marex_branch._float(D))[0])
    scale = float(np.abs(marex_branch._float(D)).sum(axis=1).max())
    assert abs(float(rump) - lam) <= 1e-9 * scale
    assert abs(float(rump) - float(lib)) <= 1e-9 * scale


def test_a_singular_matrix_gets_a_bound_at_or_under_zero():
    D = Dyadic(np.array([[1, 1], [1, 1]], dtype=object), 0)
    low = parts.rump_lower(D)
    assert low <= 0 and low > -1e-9


def test_a_matrix_rounding_to_an_infinity_gets_no_bound():
    D = Dyadic(np.array([[1 << 1100, 0], [0, 1]], dtype=object), 0)
    assert parts.rump_lower(D) is None


def test_a_matrix_whose_norm_overflows_gets_no_bound():
    D = Dyadic(np.array([[1 << 700, 0], [0, 1 << 700]], dtype=object), 0)
    assert parts.rump_lower(D) is None


def test_no_factor_in_the_tries_is_no_bound(monkeypatch):
    def never(*a, **k):
        raise np.linalg.LinAlgError("not positive definite")
    monkeypatch.setattr(parts.np.linalg, "cholesky", never)
    D = _symmetric_dyadic(np.random.default_rng(0), 4, 3)
    assert parts.rump_lower(D, tries=3) is None


# ----------------------------------------------------------------------
# the measurements
# ----------------------------------------------------------------------


def test_the_overhead_of_an_iteration_is_measured():
    out = parts.lbfgsb_overhead(20, iterations=30)
    assert out["iterations"] > 0 and out["seconds_per_evaluation"] > 0


def test_the_candidates_at_nodes_agree_where_they_claim_to(monkeypatch):
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((6, 3)) @ rng.standard_normal((3, 12)) + 0.3 * rng.standard_normal((6, 12))
    rows = parts.candidates(Y, 4, depths=(0, 2), repeats=2)
    assert [r["fixed"] for r in rows] == [0, 2]
    for r in rows:
        assert r["evaluations"] > 0
        assert r["cut_matrix_identical"] and r["to_float_identical"]
        assert r["rump"] is not None and r["library"] is not None
        assert abs(r["rump"] - r["library"]) <= 1e-9 * max(1.0, abs(r["library"]))
        for k in ("cut_matrix", "cut_matrix_bincount", "to_float", "to_float_division",
                  "lambda_min_lower", "rump_lower", "certify"):
            assert r["seconds"][k] > 0, k


def test_the_runner_appends_rows(tmp_path, monkeypatch):
    import json
    from benchmarks.studies.syndes_global_bnb import run
    monkeypatch.setattr(parts, "RESULTS", tmp_path)
    rng = np.random.default_rng(4)
    monkeypatch.setitem(run.PANELS, "tiny", lambda N: rng.standard_normal((6, N)))
    monkeypatch.setattr(parts, "candidates", lambda Y, K: [{"units": Y.shape[1], "K": K}])
    monkeypatch.setattr(parts, "lean_lift", lambda Y, K: [{"lean": K}])
    assert parts.main(["candidates", "tiny", "12", "4"]) == 0
    assert parts.main(["lean", "tiny", "12", "5"]) == 0
    assert json.loads(open(tmp_path / "parts_lean.jsonl").readline()) == {"lean": 5, "panel": "tiny"}
    assert parts.main(["overhead", "12"]) == 0
    row = json.loads(open(tmp_path / "parts_candidates.jsonl").readline())
    assert row == {"units": 12, "K": 4, "panel": "tiny"}
    assert json.loads(open(tmp_path / "parts_overhead.jsonl").readline())["variables"] == 12
    with pytest.raises(ValueError, match="no mode"):
        parts.main(["walk"])


def test_the_comparison_sees_every_output_of_a_lift():
    rng = np.random.default_rng(3)
    cuts = _cuts(rng, 5, 4)
    m = branch.Multipliers(np.ones(5), np.zeros(5), cuts, np.ones(len(cuts)))
    assert parts._same((1.0, m, 7), (1.0, m, 7))
    for other in (branch.Multipliers(np.ones(5) * 2, np.zeros(5), cuts, np.ones(len(cuts))),
                  branch.Multipliers(np.ones(5), np.ones(5), cuts, np.ones(len(cuts))),
                  branch.Multipliers(np.ones(5), np.zeros(5), cuts, np.zeros(len(cuts))),
                  branch.Multipliers(np.ones(5), np.zeros(5), cuts.take(slice(1, None)),
                                     np.ones(len(cuts) - 1))):
        assert not parts._same((1.0, m, 7), (1.0, other, 7))
    assert not parts._same((1.0, m, 7), (1.5, m, 7))
    assert not parts._same((1.0, m, 7), (1.0, m, 8))

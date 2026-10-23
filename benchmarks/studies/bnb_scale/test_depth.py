"""Checks of the depth instruments (#858).

The instruments read nodes the two-way search visits, so the checks are that
the nodes lie on the path to the design, that the spectrum reports the
structure the module docstring derives, and that a certificate, whenever it
claims one, is a point of the lift's relaxation at the trust region's value.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.bnb_scale import depth
from mlsynth.utils.syndes_helpers import branch
from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem


def _shock(N, T, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1)) + rng.standard_normal((T, N))


def _R(N, T, seed):
    return BoundEngine.from_problem(TwoWayProblem.from_panel(_shock(N, T, seed))).R


def _node(N, T, seed, fixed):
    """The first ``fixed // 2`` units in, the next ones out, at ``K = N / 2``:
    ``(c, b, A, s)``."""
    I, O = tuple(range(fixed // 2)), tuple(range(fixed // 2, fixed))
    F = tuple(range(fixed, N))
    c, b, A = branch.node_parts(_R(N, T, seed), I, O, F)
    return c, b, A, 2 * (N // 2 - len(I)) - len(F)


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ------------------------------------------------------------------ the nodes
class TestNodes:

    def test_they_lie_on_the_path_to_the_design(self):
        design = [1, 4, 6, 9]
        for I, O, F in depth.nodes(design, 12, 5, reps=4, seed=3):
            assert len(I) == 2 and len(O) == 3
            assert set(I) <= set(design) and not set(O) & set(design)
            assert sorted(I + O + F) == list(range(12))

    def test_the_root_comes_once(self):
        assert depth.nodes([0, 1], 6, 0, reps=5) == [((), (), tuple(range(6)))]

    def test_a_seed_fixes_them(self):
        a = depth.nodes([0, 2, 5], 10, 4, reps=3, seed=7)
        assert a == depth.nodes([0, 2, 5], 10, 4, reps=3, seed=7)
        assert a != depth.nodes([0, 2, 5], 10, 4, reps=3, seed=8)

    @pytest.mark.parametrize("fixed", [-1, 9, 12])
    def test_more_than_the_design_or_its_complement_holds_is_refused(self, fixed):
        with pytest.raises(ValueError, match="cannot fix"):
            depth.nodes([0, 1, 2], 8, fixed)


# --------------------------------------------------------------- the spectrum
class TestSpectrum:

    def test_the_top_eigenspace_is_the_excess(self):
        """``R = (alpha/lam) I - W``, ``W`` of rank ``T + 1``: on the plane the
        top eigenvalue holds ``n - T - 1`` directions, at the root and below."""
        for fixed in (0, 4, 8):
            c, b, A, s = _node(24, 5, 0, fixed)
            assert depth.spectrum(c, b, A, s)["multiplicity"] == len(b) - 5 - 1

    def test_fixing_units_leaves_the_top_eigenvalue(self):
        tops = [depth.spectrum(*_node(24, 5, 0, d))["top"] for d in (0, 4, 8)]
        assert max(tops) - min(tops) <= 1e-9 * abs(tops[0])

    def test_a_matrix_with_no_structure_has_a_simple_top_reached_by_its_linear_term(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((8, 8))
        out = depth.spectrum(1.0, rng.standard_normal(8), M + M.T, 2)
        assert out["multiplicity"] == 1 and out["reach"] > 1e-3

    def test_the_root_has_no_linear_term(self):
        assert depth.spectrum(*_node(16, 4, 1, 0))["reach"] == 0.0

    @pytest.mark.parametrize("n", [0, 1])
    def test_a_node_with_one_free_unit_has_no_spectrum(self, n):
        out = depth.spectrum(0.0, np.zeros(n), np.zeros((n, n)), 0)
        assert out["multiplicity"] == 0 and np.isnan(out["top"])


@settings(max_examples=20, deadline=None)
@given(N=st.integers(10, 24), T=st.integers(2, 6), seed=st.integers(0, 10 ** 6),
       fixed=st.integers(1, 6))
def test_property_the_fixed_units_never_reach_the_top_eigenspace(N, T, seed, fixed):
    """The linear term the fixed units add lies in ``W``'s row space, which the
    top eigenspace avoids, so its share there is rounding."""
    c, b, A, s = _node(N, T, seed, fixed)
    out = depth.spectrum(c, b, A, s)
    if out["multiplicity"] >= 2:
        assert out["reach"] < 1e-9


# ------------------------------------------------------------ the certificate
@pytest.fixture(scope="module")
def certified():
    """A root and a node with four units fixed that the fit certifies."""
    return [_node(24, 3, 0, 0), _node(20, 3, 1, 4)]


class TestCertificate:

    def test_it_is_a_point_of_the_lifts_relaxation(self, certified):
        for c, b, A, s in certified:
            cert = depth.certificate(c, b, A, s)
            assert cert.reason == "certified"
            Y, n = cert.Y, len(b)
            assert np.linalg.eigvalsh(Y)[0] >= -1e-9
            np.testing.assert_allclose(np.diag(Y), 1.0, atol=1e-8)
            assert Y[0, 1:].sum() == pytest.approx(s, abs=1e-7 * n)
            np.testing.assert_allclose(Y[1:, 1:] @ np.ones(n), s * Y[0, 1:], atol=1e-7 * n)
            assert not len(branch.separate(Y, gap=1e-9))

    def test_it_has_the_trust_regions_value(self, certified):
        for c, b, A, s in certified:
            cert = depth.certificate(c, b, A, s)
            Q = np.block([[np.array([[c]]), b[None, :]], [b[:, None], A]])
            assert cert.value == pytest.approx(float(np.sum(Q * cert.Y)), rel=1e-12)
            assert _rel(cert.value, branch.trust_region(c, b, A, s)[0]) < 1e-9

    def test_no_lift_passes_it(self, certified):
        for c, b, A, s in certified:
            U = branch.lift(c, b, A, s)[0]
            assert U >= depth.certificate(c, b, A, s).value * (1 - 1e-9)

    @pytest.mark.parametrize("node, reason", [
        ((12, 4, 0, 6), "simple"),
        ((12, 3, 2, 6), "easy"),
        ((12, 3, 0, 2), "not_psd"),
        ((12, 8, 3, 0), "fit"),
        ((12, 3, 0, 0), "triangle"),
    ])
    def test_each_failed_check_is_named(self, node, reason):
        cert = depth.certificate(*_node(*node))
        assert cert.reason == reason and cert.value is None and cert.Y is None

    def test_a_linear_term_in_the_top_eigenspace_is_named(self):
        """A flat quadratic part puts every direction of the plane in the top
        eigenspace, and a generic linear term reaches it."""
        b = np.random.default_rng(0).standard_normal(6)
        assert depth.certificate(0.0, b, 2.0 * np.eye(6), 0).reason == "reached"

    def test_two_free_units_are_too_few(self):
        assert depth.certificate(0.0, np.ones(2), np.eye(2), 0).reason == "small"


class TestProjectedFit:
    """Dykstra's projections search for the semidefinite fit the least-norm
    solve misses, and report the same checks after."""

    def test_it_finds_a_semidefinite_fit_the_least_norm_one_misses(self):
        node = _node(12, 3, 0, 2)
        assert depth.certificate(*node).reason == "not_psd"
        assert depth.certificate(*node, fit="projection").reason == "triangle"

    def test_where_it_finds_none_the_fit_stays_not_semidefinite(self):
        node = _node(12, 3, 0, 6)
        assert depth.certificate(*node, fit="projection").reason == "not_psd"

    def test_one_iteration_is_too_few(self):
        assert depth.certificate(*_node(12, 3, 0, 2), fit="projection",
                                 iterations=1).reason == "not_psd"

    def test_it_keeps_a_least_norm_certificate(self, certified):
        for c, b, A, s in certified:
            least = depth.certificate(c, b, A, s)
            projected = depth.certificate(c, b, A, s, fit="projection")
            assert projected.reason == "certified"
            np.testing.assert_allclose(projected.Y, least.Y)

    def test_an_unknown_fit_is_refused(self):
        with pytest.raises(ValueError, match="no fit named"):
            depth.certificate(*_node(12, 3, 0, 2), fit="sdp")

    def test_a_row_reports_it_only_where_the_least_norm_fit_failed_semidefiniteness(self):
        R = _R(12, 3, 0)
        missed = depth.depth_row(R, 3, 6, (0,), (1,), tuple(range(2, 12)), projection=True)
        assert missed["certificate"] == "not_psd" and missed["projection"] == "triangle"
        assert missed["projection_seconds"] >= 0
        R = _R(24, 3, 0)
        settled = depth.depth_row(R, 3, 12, (), (), tuple(range(24)), projection=True)
        assert settled["certificate"] == "certified" and "projection" not in settled


@settings(max_examples=20, deadline=None)
@given(N=st.integers(14, 26), T=st.integers(2, 5), seed=st.integers(0, 10 ** 6),
       fixed=st.integers(0, 4))
def test_property_a_certificate_is_feasible_at_the_trust_regions_value(N, T, seed, fixed):
    c, b, A, s = _node(N, T, seed, fixed)
    cert = depth.certificate(c, b, A, s, fit="projection" if seed % 2 else "least_norm")
    assert cert.reason in depth.REASONS
    if cert.reason == "certified":
        n = len(b)
        assert np.linalg.eigvalsh(cert.Y)[0] >= -1e-9
        np.testing.assert_allclose(np.diag(cert.Y), 1.0, atol=1e-8)
        assert not len(branch.separate(cert.Y, gap=1e-9))
        assert _rel(cert.value, branch.trust_region(c, b, A, s)[0]) < 1e-9
        assert cert.Y[0, 1:].sum() == pytest.approx(s, abs=1e-7 * n)


def test_the_node_closure_lies_between_the_maximum_and_the_trust_region():
    """At a node with units fixed, where ``c`` is far from zero: the closure is
    a relaxation, so it bounds every completion, and it is tighter than the
    trust region, which relaxes further."""
    from itertools import combinations
    from benchmarks.studies.bnb_scale.root_gap import closure
    N, K, T = 12, 6, 4
    R = _R(N, T, 5)
    I, O, F = (0, 1), (2, 3), tuple(range(4, N))
    c, b, A = branch.node_parts(R, I, O, F)
    s = 2 * (K - len(I)) - len(F)
    top = -np.inf
    for extra in combinations(F, K - len(I)):
        sigma = -np.ones(N)
        sigma[list(I) + list(extra)] = 1.0
        top = max(top, float(sigma @ R @ sigma))
    U = closure(c, b, A, s)["U"]
    assert abs(c) > 2 * _scale_of(b, A)
    assert top * (1 - 1e-7) <= U <= branch.trust_region(c, b, A, s)[0] * (1 + 1e-7)


def _scale_of(b, A):
    return max(float(np.abs(b).max()), float(np.abs(A).max()))


# ------------------------------------------------------------------- the rows
class TestDepthRows:

    @pytest.fixture(scope="class")
    def rows(self):
        Y = _shock(16, 4, 2)
        return depth.depth_table(Y, 8, [0, 3, 5, 7, 9, 11, 12, 14], fixed=(0, 4, 13),
                                 reps=2, closure_at=(4,))

    def test_a_count_that_leaves_fewer_than_four_free_is_skipped(self, rows):
        assert [r["fixed"] for r in rows] == [0, 4, 4]

    def test_they_describe_the_node(self, rows):
        for r in rows:
            assert r["free"] == 16 - r["fixed"] and r["excess"] == r["free"] - 4 - 1
            assert r["certificate"] in depth.REASONS
            assert r["evaluations"] >= 0 and r["lift_seconds"] >= 0

    def test_the_lift_never_loosens_the_trust_region(self, rows):
        assert all(r["gain"] >= -1e-12 for r in rows)

    def test_the_closure_bounds_what_the_lift_reaches(self, rows):
        """The closure is the relaxation's optimum and the lift a bound on it,
        so the closure gains at least what the lift does."""
        closed = [r for r in rows if "closure" in r]
        assert len(closed) == 1 and closed[0]["rep"] == 0
        assert closed[0]["closure_gain"] >= closed[0]["gain"] - 1e-7


def test_the_runner_records_the_depth_rows(tmp_path, monkeypatch):
    """``depth`` at 12 units reads the design ``syndes`` found and appends a
    row per node, the closure on the counts asked for."""
    import json
    from benchmarks.studies.bnb_scale import run
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    args = ["--kinds", "bls", "--sizes", "12"]
    assert run.main(["syndes", *args]) == 0
    assert run.main(["depth", *args, "--fixed", "0", "4", "--reps", "2",
                     "--closure-at", "4", "--projection"]) == 0
    rows = [json.loads(line) for line in open(tmp_path / "depth.jsonl")]
    assert [r["fixed"] for r in rows] == [0, 4, 4]
    assert all(r["kind"] == "bls" and "load_before" in r for r in rows)
    assert ["closure" in r for r in rows] == [False, True, False]
    assert all(("projection" in r) == (r["certificate"] == "not_psd") for r in rows)

"""The hull's simplex QP in one compiled loop (#1068, option 1).

The hull stage (hull.ControlBound.piece, hull.Band) calls the library's capped simplex
QP, mlsynth.utils.solvers.active_set.solve_simplex_qp, tens of thousands of times a
run, about 0.3 ms a call and mostly the interpreter: on bed s15_k3, 33,390 calls and
47% of the bed's time. fastqp.solve is that solver's loop compiled with numba for the
hull's calls (upper and warm_start, nothing else), operation by operation: the same
LAPACK dgelsy on the same bytes, the products through BLAS with the transpose numpy
uses for each layout, numpy's pairwise sums. Its weights are the library's bit for
bit, so the hull's bounds and every decision taken on them are unchanged; any other
call, a unit scale too near its rounding boundary to reproduce, or a machine without
numba, goes to the library."""
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.solvers import active_set

from mlsynth.utils.tbr_helpers.design.exact import fastqp, hull
from . import beds
from .brute import geo_panel

SETTINGS = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
LIB = active_set.solve_simplex_qp

pytestmark = pytest.mark.skipif(not fastqp.AVAILABLE, reason="the fast extra (numba) is not installed")


def _both(B, A, **kw):
    """(library weights or error, its warnings), (fastqp's, its warnings)."""
    out = []
    for fn in (LIB, fastqp.solve):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                w = fn(B, A, **kw)
            except ValueError as e:
                w = e
        out.append((w, [(type(x.message), str(x.message)) for x in seen]))
    return out


def _same(B, A, **kw):
    """The same weights to the bit and the same warnings, or the same error: caps
    that round to a sum below one (six caps of 1/6) are refused by both."""
    (w0, m0), (w1, m1) = _both(B, A, **kw)
    if isinstance(w0, ValueError):
        assert isinstance(w1, ValueError) and str(w1) == str(w0)
        return
    assert w1.dtype == w0.dtype and w1.shape == w0.shape
    assert w1.tobytes() == w0.tobytes()
    assert m1 == m0


def _spy_library(monkeypatch):
    calls = []
    monkeypatch.setattr(fastqp, "_library", lambda *a, **k: calls.append(1) or LIB(*a, **k))
    return calls


@st.composite
def _problems(draw):
    m = draw(st.integers(2, 30))
    J = draw(st.one_of(st.integers(1, 40), st.integers(80, 99)))
    seed = draw(st.integers(0, 10_000))
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((m, J)) * draw(st.sampled_from([1e-3, 1.0, 37.0]))
    if draw(st.booleans()):
        B = np.asfortranarray(B)
    A = rng.standard_normal(m) * draw(st.sampled_from([0.5, 1.0, 10.0]))
    kw = {}
    cap = draw(st.sampled_from(["none", "scalar", "vector", "loose"]))
    if cap == "scalar":
        kw["upper"] = max(1.0 / J, draw(st.floats(0.05, 0.9)))
    elif cap == "vector":
        u = rng.uniform(0.0, 0.6, J)
        kw["upper"] = u if u.sum() >= 1.0 else u + (1.0 - u.sum()) / J + 1e-3
    elif cap == "loose":
        kw["upper"] = 1.5
    start = draw(st.sampled_from(["none", "feasible", "infeasible", "vertex"]))
    if start == "vertex" and isinstance(kw.get("upper"), np.ndarray):
        u, w0, left = kw["upper"], np.zeros(J), 1.0   # a vertex of the capped simplex: caps filled in a random order
        for j in rng.permutation(J):
            w0[j] = min(u[j], left); left -= w0[j]
            if left <= 0.0:
                break
        if abs(w0.sum() - 1.0) <= 1e-9:
            kw["warm_start"] = w0
    elif start == "feasible":
        w0 = rng.uniform(0, 1, J); w0 /= w0.sum()
        kw["warm_start"] = w0
    elif start == "infeasible":
        kw["warm_start"] = rng.standard_normal(J)
    return B, A, kw


# ---------------------------------------------------------------- smoke


def test_smoke_a_hull_piece_is_the_librarys():
    y, X = geo_panel(4, n=20, p=14)[:, 0], geo_panel(4, n=20, p=14)[:, 1:]
    cb = hull.ControlBound(y, X)
    _same(cb._Z, cb.y, upper=0.2)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(_problems())
def test_property_the_weights_are_the_librarys_bit_for_bit(problem):
    """Over C and F layouts, scales, pools that take the FISTA seed (80 and more),
    no cap, a scalar or vector cap (binding or loose) and every kind of start: the
    same weights to the bit, and the same warnings."""
    B, A, kw = problem
    _same(B, A, **kw)


@SETTINGS
@given(st.integers(2, 25), st.integers(2, 12), st.integers(0, 10_000), st.booleans())
def test_property_the_certificate_is_the_librarys(J, m, seed, fortran):
    """The kernel's KKT error at any point of the capped simplex, some weights at
    zero and some at their caps, against active_set.simplex_kkt_error: the same
    double. At the solver's own points the capped term seldom decides the warning,
    so the certificate is pinned where every term can."""
    r = np.random.default_rng(seed)
    B = r.standard_normal((m, J)) * 10.0 ** r.uniform(-3, 3)
    if fortran:
        B = np.asfortranarray(B)
    A = r.standard_normal(m)
    up = r.uniform(0.05, 0.6, J)
    if up.sum() < 1.0:
        up = up + (1.0 - up.sum()) / J + 1e-3
    w, left = np.zeros(J), 1.0
    for j in r.permutation(J):                   # fill some caps, leave a part, some zeros
        take = min(up[j], left) if r.random() < 0.6 else min(up[j], left) * r.random()
        w[j] = take; left -= take
    w[r.integers(J)] += max(left, 0.0) if w.sum() < 1.0 else 0.0
    S, t = (B, False) if B.flags.c_contiguous else (B.T, True)
    DS, _, _, _, _ = fastqp._prepare(S, t, A)
    for upper, has in ((up, True), (fastqp._EMPTY, False)):
        got = fastqp._kkt(DS, t, w, m, J, upper, has)
        want = active_set.simplex_kkt_error(B, A, w, None, upper if has else None)
        assert np.float64(got).tobytes() == np.float64(want).tobytes()


@SETTINGS
@given(st.integers(0, 6000), st.integers(0, 10_000))
def test_property_the_pairwise_sum_is_numpys(n, seed):
    a = np.random.default_rng(seed).standard_normal(n) * 1e3
    assert np.float64(fastqp._sum(a)).tobytes() == np.float64(np.sum(a)).tobytes()


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 500), st.integers(8, 16))
def test_property_the_hull_bounds_are_the_librarys(seed, p):
    """ControlBound pieces and Band ends on a geo panel, through fastqp and through
    the library: the same floats and the same minimisers."""
    M = geo_panel(seed, n=20, p=p)
    y, X = M[:, 0], M[:, 1:]
    m = max(1, (p - 1) // 2)
    out = {}
    for fast in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fastqp, "AVAILABLE", fast)
            cb, band = hull.ControlBound(y, X), hull.Band(y, X)
            vals = [cb.piece(0.1, 2.0, m), cb.point, cb.piece(0.5, 0.9, m), cb.point]
            vals += list(band._at(0.7 * m)) + list(band._at(1.3 * m))
            out[fast] = [np.asarray(v, dtype=float).tobytes() if v is not None else None for v in vals]
    assert out[True] == out[False]


# ---------------------------------------------------------------- the beds


def test_bed_groups_are_the_same_with_and_without_the_kernel(monkeypatch):
    fields = ("certified", "controls", "nodes", "full", "shor", "open", "rounded")
    fast = [beds.run_group("s15_k3", i) for i in range(4)]
    monkeypatch.setattr(fastqp, "AVAILABLE", False)
    slow = [beds.run_group("s15_k3", i) for i in range(4)]
    for a, b in zip(fast, slow):
        assert all(getattr(a, f) == getattr(b, f) for f in fields)


def test_the_hull_calls_the_kernel(monkeypatch):
    seen, real = [], fastqp._core
    monkeypatch.setattr(fastqp, "_core", lambda *a: seen.append(1) or real(*a))
    lib = _spy_library(monkeypatch)
    beds.run_group("s15_k3", 0)
    assert len(seen) > 100 and len(lib) < len(seen) / 100


# ---------------------------------------------------------------- edges


def test_one_donor():
    _same(np.ones((3, 1)), np.zeros(3))


def test_a_zero_target_and_a_zero_column():
    B = np.zeros((4, 3)); B[:, 0] = 1.0
    _same(B, np.zeros(4))


def test_caps_that_pin_a_vertex():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((6, 5))
    _same(B, rng.standard_normal(6), upper=np.array([0.5, 0.5, 0.0, 0.0, 0.0]))


def test_a_vertex_start_is_left_for_a_better_vertex():
    """Started at the vertex (0.5, 0.5, 0) of three unit donors capped at one half,
    with the target near the third: the optimum (0.25, 0.25, 0.5) is reached only
    through the pair release at the vertex."""
    B, A, up = np.eye(3), np.array([0.1, 0.1, 0.8]), np.full(3, 0.5)
    _same(B, A, upper=up, warm_start=np.array([0.5, 0.5, 0.0]))
    assert fastqp.solve(B, A, upper=up, warm_start=np.array([0.5, 0.5, 0.0]))[2] > 0.4


@pytest.mark.parametrize("delta", [1e-15, 4e-16, -1e-15])
def test_a_vertex_whose_release_is_inside_the_rounding_margin(delta):
    """Donor 3 is donor 1 scaled by 1 + delta: at the vertex (0.5, 0.5, 0) their
    gradients differ by rounding only, inside the release margin, and the loop
    stays where the library stays."""
    rng = np.random.default_rng(8)
    C = rng.standard_normal((6, 2))
    B = np.column_stack([C, C[:, 0] * (1.0 + delta)])
    A = 0.5 * C[:, 0] + 0.5 * C[:, 1] + 0.01 * rng.standard_normal(6)
    _same(B, A, upper=np.full(3, 0.5), warm_start=np.array([0.5, 0.5, 0.0]))


def test_least_squares_points_on_their_caps():
    """Targets built from two donors at weight one half each, caps of one half on
    them: the free least-squares point lands on the caps, at rounding above them on
    some seeds, and is clipped where the library clips it."""
    for seed in range(300):
        r = np.random.default_rng(seed)
        C = r.standard_normal((8, 3))
        B = np.column_stack([C, r.standard_normal((8, 2))])
        _same(B, 0.5 * C[:, 0] + 0.5 * C[:, 1], upper=np.array([0.5, 0.5, 0.9, 0.9, 0.9]))


def _extreme(seed):
    """Columns scaled from 1e-8 to 1e8, a near-duplicate pair on half the seeds, and
    a vector cap on half: inputs where the library's certificate can refuse."""
    r = np.random.default_rng(seed)
    m, J = int(r.integers(2, 10)), int(r.integers(2, 25))
    B = r.standard_normal((m, J)) * (10.0 ** r.uniform(-8, 8, J))
    if r.random() < 0.5:
        k = int(r.integers(0, J))
        B[:, (k + 1) % J] = B[:, k] * (1 + 1e-14)
    A = r.standard_normal(m) * 10.0 ** r.uniform(-6, 6)
    kw = {}
    if r.random() < 0.5:
        u = r.uniform(0, 0.6, J)
        kw["upper"] = u if u.sum() >= 1 else u + (1 - u.sum()) / J + 1e-3
    return B, A, kw


@pytest.mark.parametrize("seed", [5, 236, 442, 1796, 2210, 2758, 3309, 3998, 4098, 4295])
def test_an_uncertified_solve_warns_as_the_library_does(seed):
    """Ten of the first 4,300 extreme inputs the library's certificate refuses (three
    capped): the same weights and the same SimplexQPUncertifiedWarning, KKT error
    included."""
    B, A, kw = _extreme(seed)
    (w0, m0), _ = _both(B, A, **kw)
    assert any(c is active_set.SimplexQPUncertifiedWarning for c, _ in m0)
    _same(B, A, **kw)


@pytest.mark.parametrize("upper", [1.0, 1.5, "ones", "mixed"])
@pytest.mark.parametrize("seed", range(6))
def test_loose_caps_are_the_uncapped_program(upper, seed):
    """Caps of one or more cannot bind on the simplex, and the library solves the
    uncapped program for them, bit for bit."""
    r = np.random.default_rng(seed)
    J = 3 + seed
    B, A = r.standard_normal((7, J)), r.standard_normal(7)
    if upper == "ones":
        upper = np.ones(J)
    elif upper == "mixed":
        upper = 1.0 + r.uniform(0, 2, J)
    _same(B, A, upper=upper)
    assert fastqp.solve(B, A, upper=upper).tobytes() == LIB(B, A).tobytes()


def _cold_pool(seed):
    r = np.random.default_rng(seed)
    J, m = int(r.integers(80, 100)), int(r.integers(2, 30))
    B = r.standard_normal((m, J)) * r.choice([1e-3, 1.0, 37.0])
    A = r.standard_normal(m) * r.choice([0.5, 1.0, 10.0])
    kw = {"upper": max(1.0 / J * 1.0000001, r.uniform(0.05, 0.9))} if seed % 3 == 1 else {}
    return B, A, kw


@pytest.mark.parametrize("seed", [62, 76, 198, 240, 383, 399])
def test_a_cold_large_pool_is_seeded_as_the_library_seeds_it(seed):
    """No warm start on 80 or more donors: the library seeds the loop with FISTA, and
    so does the kernel, with the library's seed. Most pools end on the same free set
    from the uniform start too; these six of the first 400 do not, so they tell the
    seed's absence apart."""
    B, A, kw = _cold_pool(seed)
    _same(B, A, **kw)


def test_a_warm_start_off_the_simplex_is_ignored():
    rng = np.random.default_rng(5)
    B = rng.standard_normal((6, 5))
    _same(B, rng.standard_normal(6), warm_start=np.full(5, 0.5))


def test_a_scale_at_its_rounding_boundary_goes_to_the_library(monkeypatch):
    """The unit scale is a power of two picked from an einsum the kernel does not
    reproduce bit for bit; at the boundary (here sqrt of the mean energy exactly 1)
    the call goes to the library."""
    lib = _spy_library(monkeypatch)
    B, A = np.array([[1.0, -1.0]]), np.array([0.0])
    _same(B, A)
    assert lib


# ---------------------------------------------------------------- what the kernel does not take


@pytest.mark.parametrize("kw", [dict(linear=np.array([0.1, 0.0, 0.2])), dict(max_iter=5), dict(return_info=True),
                                dict(accelerate=False), dict(tol=1e-6)])
def test_other_calls_go_to_the_library(monkeypatch, kw):
    lib = _spy_library(monkeypatch)
    rng = np.random.default_rng(1)
    B, A = rng.standard_normal((5, 3)), rng.standard_normal(5)
    got, want = fastqp.solve(B, A, **kw), LIB(B, A, **kw)
    assert lib
    if isinstance(want, tuple):
        assert got[0].tobytes() == want[0].tobytes() and got[1] == want[1]
    else:
        assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("make", [lambda B: B[:, ::2], lambda B: B.astype(np.float32), lambda B: B.tolist()])
def test_other_storage_goes_to_the_library(monkeypatch, make):
    lib = _spy_library(monkeypatch)
    rng = np.random.default_rng(2)
    B, A = make(rng.standard_normal((5, 6))), rng.standard_normal(5)
    assert fastqp.solve(B, A).tobytes() == LIB(B, A).tobytes() and lib


def test_without_numba_every_call_goes_to_the_library(monkeypatch):
    monkeypatch.setattr(fastqp, "AVAILABLE", False)
    lib = _spy_library(monkeypatch)
    rng = np.random.default_rng(4)
    B, A = rng.standard_normal((5, 4)), rng.standard_normal(5)
    assert fastqp.solve(B, A, upper=0.4).tobytes() == LIB(B, A, upper=0.4).tobytes() and lib


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("upper", [np.array([0.5, -0.1, 0.6]), np.array([0.5, np.inf, 0.6]),
                                   np.array([0.2, 0.2, 0.2]), np.array([0.5, 0.5])])
def test_an_invalid_cap_raises_as_the_library_does(upper):
    rng = np.random.default_rng(6)
    B, A = rng.standard_normal((4, 3)), rng.standard_normal(4)
    with pytest.raises(ValueError) as want:
        LIB(B, A, upper=upper)
    with pytest.raises(ValueError) as got:
        fastqp.solve(B, A, upper=upper)
    assert str(got.value) == str(want.value)


@pytest.mark.parametrize("B,A", [(np.ones(3), np.ones(3)), (np.ones((3, 2)), np.ones(4)), (np.ones((3, 0)), np.ones(3))])
def test_malformed_data_raises_as_the_library_does(B, A):
    with pytest.raises(ValueError) as want:
        LIB(B, A)
    with pytest.raises(ValueError) as got:
        fastqp.solve(B, A)
    assert str(got.value) == str(want.value)


def test_importing_the_kernel_links_nothing_and_the_first_solve_links_dgelsy():
    """Registering dgelsy with LLVM is process-wide state, so the module does it on its
    first compiled solve, not at import: in a fresh process the symbol is absent after
    the import, present after one solve, and that solve's weights are the library's."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import numpy as np, llvmlite.binding as llvm\n"
        "from mlsynth.utils.solvers import active_set\n"
        "from mlsynth.utils.tbr_helpers.design.exact import fastqp\n"
        "name = 'mlsynth_tbr_fastqp_dgelsy'\n"
        "assert not llvm.address_of_symbol(name), 'linked at import'\n"
        "r = np.random.default_rng(3); B = r.standard_normal((20, 6)); A = r.standard_normal(20)\n"
        "w = fastqp.solve(B, A, upper=0.5)\n"
        "assert llvm.address_of_symbol(name), 'not linked by the solve'\n"
        "assert w.tobytes() == active_set.solve_simplex_qp(B, A, upper=0.5).tobytes()\n"
    )
    root = Path(__file__).resolve().parents[3]
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]

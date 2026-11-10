"""The relaxation's rows built by index arithmetic, the dict table built on demand (#965).

Rows keeps its table of dicts as the definition: the exact certificate (_slack),
holds() and the tests read it. The solvers and the float certificate read only the
counts, the right-hand sides, the rows that carry c and the sparse matrices, and
those are now built directly from the row families' index patterns, with the dicts
built the first time something reads them. The tests pin the two paths equal: every
count, right-hand side and row index, and the matrices bit for bit at any scale of
the c rows."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from .brute import panel
from .certificate import ExactData
from .slopefree import Rows

SETTINGS = settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _from_tables(R, c_scale):
    """The matrices from the dict table, as Rows.matrices built them before (the
    reference)."""
    n = R.n

    def build(rows, scale_rows=()):
        data, ri, ci = [], [], []
        scale = set(scale_rows)
        for k, r in enumerate(rows):
            f = 1.0 / c_scale if k in scale else 1.0
            for (a, b), v in r.items():
                data.append(float(v) * f); ri.append(k); ci.append(a + n * b)
        return sp.csr_matrix((data, (ri, ci)), shape=(len(rows), n * n))
    return build(R.eq), build(R.ineq, R._tables()[4])


def _same(M, N):
    M, N = sp.csr_matrix(M), sp.csr_matrix(N)
    M.sort_indices(); N.sort_indices()
    return (M.shape == N.shape and np.array_equal(M.indptr, N.indptr) and np.array_equal(M.indices, N.indices)
            and np.array_equal(M.data, N.data))


def _check(R):
    eq, beq, ineq, bin_, c_rows, t_rows = R._tables()
    assert (R.n_eq, R.n_in) == (len(eq), len(ineq))
    assert list(R.beq) == beq and list(R.bin) == bin_
    assert list(R.c_rows) == c_rows and tuple(R.triangle_rows) == t_rows
    for cs in (1.0, 3.7, 2.0 ** -40):
        Me, Mi = R.matrices(cs)
        Re, Ri = _from_tables(R, cs)
        assert _same(Me, Re) and _same(Mi, Ri), cs


def _c(D):
    return D.c


# ---------------------------------------------------------------- smoke


def test_smoke_the_rows_build_without_the_table():
    """Construction and the matrices leave the dict table unbuilt."""
    D = ExactData(*panel(0, p=6))
    R = Rows(D.c, 3, cuts=[{0, 1, 2}], face=True, box=True)
    Me, Mi = R.matrices(1.0)
    assert Me.shape == (R.n_eq, R.n ** 2) and Mi.shape == (R.n_in, R.n ** 2)
    assert R._table is None


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("m", [None, 1, 3, 6])
@pytest.mark.parametrize("level", ["full", "shor"])
@pytest.mark.parametrize("box", [False, True])
def test_the_fast_rows_are_the_table(m, level, box):
    D = ExactData(*panel(1, p=6))
    R = Rows(D.c, m, cuts=[{0, 2}, {1, 3, 5}], triangles=[(0, (0, 1, 2)), (1, (1, 3, 4))],
             fixes=((2, 1), (5, 0)) if m else (), level=level, box=box)
    _check(R)


def test_the_table_is_built_once_on_demand():
    D = ExactData(*panel(0, p=5))
    R = Rows(D.c, 2)
    first = R.eq
    assert R._table is not None and R.eq is first and R.ineq is R._tables()[2]


def test_holds_reads_the_table():
    """holds() checks a design's lifted matrix against every row, as before."""
    D = ExactData(*panel(0, p=4))
    R = Rows(D.c, 2, cuts=[{0, 1}])
    u = [Q(0), Q(1), Q(1), Q(0)]
    z = [Q(1)] + u
    Z = np.array([[a * b for b in z] for a in z], dtype=object)
    c_u = sum((D.c[j] * u[j] for j in range(4)), Q(0))
    assert R.holds(Z) == (c_u >= 0)


# ---------------------------------------------------------------- properties


@st.composite
def _rows(draw):
    p = draw(st.integers(1, 7))
    c = [Q(draw(st.integers(-5, 5)), draw(st.sampled_from([1, 2, 3, 7]))) for _ in range(p)]
    m = draw(st.one_of(st.none(), st.integers(0, p)))
    level = draw(st.sampled_from(["full", "shor"]))
    cuts = draw(st.lists(st.sets(st.integers(0, p - 1), min_size=1), max_size=3))
    triangles = []
    if p >= 3:
        trip = list(itertools.permutations(range(p), 3))
        triangles = draw(st.lists(st.tuples(st.integers(0, 1), st.sampled_from(trip)), max_size=4, unique=True))
    fixes = []
    if m is not None:
        js = draw(st.lists(st.integers(0, p - 1), unique=True, max_size=p))
        fixes = [(j, draw(st.integers(0, 1))) for j in js]
    box = draw(st.booleans())
    return Rows(c, m, cuts=cuts, triangles=triangles, fixes=fixes, level=level, box=box)


@SETTINGS
@given(R=_rows())
def test_property_the_fast_rows_are_the_table(R):
    """For every size (and none), level, cut set, triangle set, fixing and box, with
    c holding zeros, negatives and non-dyadic rationals: the counts, right-hand
    sides, c rows and triangle range are the table's, and the matrices are the
    table's bit for bit at every scale of the c rows."""
    _check(R)


# ---------------------------------------------------------------- edge cases


def test_a_single_control():
    _check(Rows([Q(3)], 1, box=True))
    _check(Rows([Q(3)], None))


def test_a_zero_in_c_drops_its_coefficients():
    """A zero coefficient is not stored, in the table or the matrices."""
    R = Rows([Q(0), Q(2), Q(-1)], 2)
    _check(R)
    assert all(v != 0 for r in R.ineq for v in r.values())


def test_size_zero_drops_the_size_coefficient():
    _check(Rows([Q(1), Q(2), Q(3)], 0))


def test_every_control_fixed_has_no_box_rows():
    R = Rows([Q(1), Q(2), Q(3)], 2, fixes=((0, 1), (1, 1), (2, 0)), box=True)
    _check(R)
    assert R.n_in == Rows([Q(1), Q(2), Q(3)], 2, fixes=((0, 1), (1, 1), (2, 0))).n_in


# ---------------------------------------------------------------- failure


def test_an_unknown_level_is_refused():
    with pytest.raises(ValueError, match="level"):
        Rows([Q(1), Q(2)], 1, level="half")

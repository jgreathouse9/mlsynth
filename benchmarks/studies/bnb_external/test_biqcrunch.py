"""Checks of the #847 BiqCrunch bridge: the 0/1 form is the two-way core, the
rounding stays inside its stated error, the BC file states the problem it
claims, the output is read back, and (where the program is built) BiqCrunch
returns the enumeration's optimum."""
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from . import biqcrunch as bc


def _sym(rng, n):
    a = rng.normal(size=(n, n))
    return a + a.T


def _sigma(z):
    return 2.0 * np.asarray(z, dtype=float) - 1.0


sizes = st.integers(3, 9).flatmap(
    lambda n: st.tuples(st.just(n), st.integers(1, n - 1), st.integers(0, 10 ** 6)))


@given(sizes)
@settings(max_examples=100, deadline=None)
def test_the_binary_form_is_the_core(case):
    """``sigma'R sigma`` at ``sigma = 2z - 1`` is the 0/1 form's value, at
    every set of the size and at every other."""
    n, K, seed = case
    rng = np.random.default_rng(seed)
    R = _sym(rng, n)
    Q, lin, const = bc.binary_form(R)
    for z in itertools.product((0, 1), repeat=n):
        z = np.array(z)
        s = _sigma(z)
        got = z @ np.triu(Q, 1) @ z + lin @ z + const
        assert got == pytest.approx(s @ R @ s, rel=1e-12, abs=1e-9)


@given(sizes, st.floats(1.0, 1e6))
@settings(max_examples=100, deadline=None)
def test_the_rounding_stays_inside_its_bound(case, scale):
    """Each coefficient is off by at most half a unit, so a set of ``K`` units,
    with ``K(K-1)/2`` pairs and ``K`` linear terms, is off by at most
    ``(K(K-1)/2 + K) / 2`` units: ``rounding_error(K, scale)`` in ``sigma'R sigma``."""
    n, K, seed = case
    rng = np.random.default_rng(seed)
    R = _sym(rng, n)
    inst = bc.Instance.build(R, K, scale=scale)
    for S in itertools.combinations(range(n), K):
        z = np.zeros(n, dtype=int)
        z[list(S)] = 1
        s = _sigma(z)
        assert abs(inst.core_value(inst.integer_value(z)) - s @ R @ s) <= (
            inst.rounding_error() * (1 + 1e-9))


@pytest.mark.parametrize("n, K", [(40, 10), (100, 10), (40, 2)])
def test_the_scale_fits_biqcrunchs_integers(n, K):
    """BiqCrunch keeps its values in a C ``int``: by default no set of size
    ``K`` can reach ``VALUE_CEILING``, eight times under 2**31, and the
    coefficients keep at least a million units of resolution."""
    rng = np.random.default_rng(0)
    inst = bc.Instance.build(1e5 * _sym(rng, n), K)
    top = max(np.abs(inst.C).max(), np.abs(inst.d).max())
    assert (K * (K - 1) / 2 + K) * top <= bc.VALUE_CEILING * (1 + 1e-9)
    assert bc.VALUE_CEILING <= 2 ** 31 / 8 and top >= 1e6


def _read_bc(text):
    """The BC file back as (sense, rhs, matrices): a reader written from the
    manual's grammar (sec. 2.2), independent of the writer."""
    lines = [l for l in text.splitlines() if l and l[0] not in "#;*"]
    sense = int(lines[0].split()[0])
    m = int(lines[1].split()[0])
    blocks = int(lines[2].split()[0])
    size = int(lines[3].split(",")[0])
    rhs = [float(v) for v in lines[4].split()]
    mats = [np.zeros((size, size)) for _ in range(m + 1)]
    for line in lines[5:]:
        k, b, i, j, v = line.split()
        assert b == "1"
        i, j = int(i) - 1, int(j) - 1
        mats[int(k)][i, j] += float(v)
        if i != j:
            mats[int(k)][j, i] += float(v)
    return sense, blocks, rhs, mats


def _lifted(z):
    x = np.append(z, 1.0)
    return np.outer(x, x)


@pytest.mark.parametrize("products", [True, False])
def test_the_bc_file_states_the_problem(products):
    """Read back with a reader written from the manual: the objective at every
    set of the size is the integer value, and the constraints hold exactly at
    the sets of the size, and fail at every other set."""
    rng = np.random.default_rng(3)
    n, K = 7, 3
    inst = bc.Instance.build(_sym(rng, n), K, scale=1000.0)
    sense, blocks, rhs, mats = _read_bc(inst.bc(products=products))
    assert sense == 1 and blocks == 1
    assert len(rhs) == len(mats) - 1 == (1 + n if products else 1)
    for z in itertools.product((0, 1), repeat=n):
        z = np.array(z, dtype=float)
        X = _lifted(z)
        held = all(abs(np.sum(M * X) - r) < 1e-9 for M, r in zip(mats[1:], rhs))
        assert held == (z.sum() == K)
        if held:
            assert np.sum(mats[0] * X) == pytest.approx(inst.integer_value(z), abs=1e-9)


def test_the_output_is_read_back():
    finished = """Nodes = 3
Root node bound = 47.43000
Maximum value = 43
Solution = { 1 2 3 }
CPU time = 0.0045 s
"""
    got = bc.parse(finished)
    assert got == {"nodes": 3, "bound": 47.43, "value": 43, "solution": [0, 1, 2],
                   "cpu": 0.0045, "stopped": False, "gap_percent": 0.0}
    stopped = """Nodes = 812
Root node bound = 1203.50000
Best value = 1190
Current bound = 1203.50000
Gap = 1.13%
CPU time = 60.0012 s
"""
    got = bc.parse(stopped)
    assert got["stopped"] and got["value"] == 1190 and got["solution"] is None
    assert got["bound"] == 1203.5 and got["gap_percent"] == 1.13 and got["nodes"] == 812


def test_an_output_without_a_summary_is_refused():
    with pytest.raises(ValueError):
        bc.parse("Node 0 Feasible solution 43\n")


def test_the_parameters_are_written_as_biqcrunch_reads_them():
    text = bc.params(time_limit=60, soln_value=1189, root=True)
    lines = dict(l.split(" = ") for l in text.splitlines())
    assert lines == {"time_limit": "60", "soln_value_provided": "1",
                     "soln_value": "1189", "root": "1"}
    assert bc.params() == ""


BINARY = os.environ.get("BIQCRUNCH")


@pytest.mark.skipif(not BINARY or not Path(BINARY).exists(),
                    reason="BIQCRUNCH (the built generic binary) is not set")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biqcrunch_returns_the_enumerations_optimum(seed, tmp_path):
    rng = np.random.default_rng(seed)
    n, K = 12, 4
    R = _sym(rng, n)
    inst = bc.Instance.build(R, K)
    out = bc.run(inst, binary=BINARY, workdir=tmp_path)
    best = max(itertools.combinations(range(n), K),
               key=lambda S: _sigma(np.isin(np.arange(n), S)) @ R
               @ _sigma(np.isin(np.arange(n), S)))
    assert not out["stopped"] and tuple(out["solution"]) == best
    z = np.isin(np.arange(n), best).astype(float)
    assert out["value"] == inst.integer_value(z)

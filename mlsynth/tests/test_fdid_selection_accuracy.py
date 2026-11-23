"""FDID's forward selection against exact rational arithmetic.

Each step adds the donor whose running mean gives the treated unit's pre-period the best DID fit (Li, "Frontiers: A
simple forward difference-in-differences method", Marketing Science 43 (2024) 267-279). The residual sum of squares
of that fit was computed by expanding ||y_c - (S_c + x_j)/(k+1)||^2 into ss_tot + ss_X - 2 cross: the cross-product
form, whose error grows with the square of the conditioning (Golub and Van Loan, Matrix Computations, 4th ed., section
5.3.7; CLAUDE.md invariant 9). On donor pools with a common factor of 1e6 the R^2 path erred by up to 1.2e-4 of
1 - R^2, and with near-duplicate donors at a factor of 1e8 one pick differed from the exact one.

The oracle selects in exact arithmetic on the panel's doubles, ties to the first index, as Li's Fun_FDID.R does with
which.max.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.fdid_helpers.estimation import forward_did_select

#: R^2 near 1 is a double next to 1, which holds it to within ulp(1); closer than that no computation can report it.
_ULP1 = Fr(2.0 ** -52)


def _exact(y, X, T0):
    """Forward selection in exact arithmetic: (sequence, ss_res path, optimal prefix)."""
    T0 = int(T0)
    yv = [Fr(float(v)) for v in y[:T0]]
    ym = sum(yv) / T0
    yc = [v - ym for v in yv]
    N = X.shape[1]
    cols = [[Fr(float(X[t, j])) for t in range(T0)] for j in range(N)]
    S = [Fr(0)] * T0
    seq, path = [], []
    left = list(range(N))
    for k in range(N):
        best = bj = None
        for j in left:
            m = [(S[t] + cols[j][t]) / (k + 1) for t in range(T0)]
            mm = sum(m) / T0
            ssr = sum((yc[t] - (m[t] - mm)) ** 2 for t in range(T0))
            if best is None or ssr < best:
                best, bj = ssr, j
        seq.append(bj)
        path.append(best)
        left.remove(bj)
        S = [S[t] + cols[bj][t] for t in range(T0)]
    kbest = min(range(N), key=lambda i: (path[i], i))
    sst = sum(v * v for v in yc)
    return seq, path, seq[:kbest + 1], sst


def _run(y, X, T0):
    out = forward_did_select(y, X, T0, list(range(X.shape[1])), verbose=True)["FDID"]
    seq = [int(s["selected_idx"]) for s in out["intermediary"]]
    return seq, out


def _pool(seed, N, T, F, LEV, EPS, k=3):
    """Donors sharing a common factor of amplitude F at levels LEV with noise EPS; the treated unit is the mean of the
    first k donors plus a level, so the best fits have R^2 near 1."""
    rng = np.random.default_rng(seed)
    f = np.sin(np.arange(T) * 2 * np.pi / 12) + 0.3 * np.arange(T) / T
    load = 1 + 0.01 * rng.normal(size=N)
    X = LEV * (1 + rng.uniform(size=N)) + F * np.outer(f, load) + EPS * rng.normal(size=(T, N))
    y = X[:, :min(k, N)].mean(axis=1) + 5.0 + EPS * 0.1 * rng.normal(size=T)
    return y, X


def _pairs(seed, N, T, F, LEV, EPS, gap):
    """Donors in near-identical pairs: loadings within ``gap`` of each other, so successive picks nearly tie."""
    rng = np.random.default_rng(seed)
    f = np.sin(np.arange(T) * 2 * np.pi / 12) + 0.3 * np.arange(T) / T
    base = 1 + 0.01 * rng.normal(size=(N + 1) // 2)
    load = np.repeat(base, 2)[:N] + gap * rng.normal(size=N)
    X = LEV * (1 + rng.uniform(size=N)) + F * np.outer(f, load) + EPS * rng.normal(size=(T, N))
    y = X[:, :3].mean(axis=1) + EPS * 0.1 * rng.normal(size=T)
    return y, X


def test_a_near_duplicate_pick_is_the_exact_one():
    """Regression: with donors in near-identical pairs at a common factor of 1e8, the expanded form picked donor 1 at
    step 2 where exact arithmetic picks donor 0. Every pick is now the exact one."""
    y, X = _pairs(0, 12, 30, 1e8, 1e9, 1e-3, 1e-11)
    seq, out = _run(y, X, 24)
    ex_seq, _, ex_opt, _ = _exact(y, X, 24)
    assert seq == ex_seq
    assert out["selected_controls"] == ex_opt


@pytest.mark.parametrize("seed", range(6))
def test_the_r2_path_is_accurate_on_a_large_common_factor(seed):
    """Unit: with a common factor of 1e6 the R^2 path is within 1e-6 of the exact one, relative to 1 - R^2; the
    expanded form erred by up to 1.2e-4 over these six pools."""
    y, X = _pool(seed, 16, 36, 1e6, 1e7, 1e-2)
    seq, out = _run(y, X, 30)
    ex_seq, path, ex_opt, sst = _exact(y, X, 30)
    assert seq == ex_seq
    for got, ssr in zip(out["R2_at_each_step"], path):
        r2 = 1 - ssr / sst
        assert abs(Fr(float(got)) - r2) <= Fr(1e-6) * (1 - r2) + _ULP1


def test_exact_duplicate_donors_tie_to_the_first_index():
    """Edge: two identical donors give exactly equal fits; the pick is the lower index, as which.max picks it."""
    rng = np.random.default_rng(11)
    base = rng.standard_normal((9, 5)) * 1e3 + 1e6
    X = np.column_stack([base, base[:, 1], base[:, 3]])
    y = base[:, 1] + 0.1 * rng.standard_normal(9)
    seq, out = _run(y, X, 6)
    ex_seq, _, ex_opt, _ = _exact(y, X, 6)
    assert seq == ex_seq
    assert out["selected_controls"] == ex_opt


@pytest.mark.parametrize("N", [1, 2])
def test_tiny_pools(N):
    """Edge: one or two donors; the selection is the exact one."""
    y, X = _pool(3, N, 10, 1e3, 1e4, 1.0)
    seq, out = _run(y, X, 8)
    ex_seq, _, ex_opt, _ = _exact(y, X, 8)
    assert seq == ex_seq and out["selected_controls"] == ex_opt


def test_a_constant_treated_pre_period():
    """Edge: a constant treated pre-period has ss_tot 0; the selection still follows the exact residuals."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((8, 5)) * 1e4 + 1e7
    y = np.r_[np.full(6, 3.0), [4.0, 5.0]]
    seq, out = _run(y, X, 6)
    ex_seq, _, ex_opt, _ = _exact(y, X, 6)
    assert seq == ex_seq and out["selected_controls"] == ex_opt
    assert np.all(np.isfinite(out["R2_at_each_step"]))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), N=st.integers(1, 7), T0=st.integers(3, 10), F=st.sampled_from([0.0, 1e3, 1e7]),
       LEV=st.sampled_from([0.0, 1e5, 1e9]), EPS=st.sampled_from([1e-4, 1.0]),
       gap=st.sampled_from([None, 1e-9, 1e-12]))
def test_property_the_selection_is_the_exact_one(seed, N, T0, F, LEV, EPS, gap):
    """Property: over donor pools with large common factors, large levels, small noise and near-duplicate pairs, every
    pick and the optimal prefix are those of exact arithmetic, and the R^2 path is within 1e-6 of the exact one,
    relative to 1 - R^2."""
    T = T0 + 2
    y, X = _pool(seed, N, T, F, LEV, EPS) if gap is None else _pairs(seed, N, T, F, LEV, EPS, gap)
    seq, out = _run(y, X, T0)
    ex_seq, path, ex_opt, sst = _exact(y, X, T0)
    assert seq == ex_seq
    assert out["selected_controls"] == ex_opt
    if sst > 0:
        for got, ssr in zip(out["R2_at_each_step"], path):
            r2 = 1 - ssr / sst
            assert abs(Fr(float(got)) - r2) <= Fr(1e-6) * (1 - r2) + _ULP1


def test_a_non_finite_treated_pre_period_is_reported():
    """Failure: a NaN in the treated unit's pre-period leaves no comparable fit and raises, as the search always did."""
    y, X = _pool(1, 5, 10, 1.0, 0.0, 1.0)
    y[2] = np.nan
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        forward_did_select(y, X, 8, list(range(5)))


def test_a_donor_with_a_non_finite_pre_period_is_never_a_candidate():
    """Edge: a donor with a NaN pre-period is never selected, as the search always left it, and the selection over
    the others is the exact one; the search used to end by re-selecting a chosen donor at R^2 = -inf."""
    y, X = _pool(1, 6, 10, 1e3, 1e4, 1.0)
    X[3, 4] = np.nan
    seq, out = _run(y, X, 8)
    keep = [0, 1, 2, 3, 5]
    ex_seq, _, ex_opt, _ = _exact(y, X[:, keep], 8)
    assert seq == [keep[i] for i in ex_seq]
    assert out["selected_controls"] == [keep[i] for i in ex_opt]
    assert 4 not in seq and np.all(np.isfinite(out["R2_at_each_step"]))


def test_no_finite_donor_is_reported():
    """Failure: every donor with a non-finite pre-period leaves nothing to select, and raises."""
    y, X = _pool(1, 3, 10, 1.0, 0.0, 1.0)
    X[0, :] = np.nan
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        forward_did_select(y, X, 8, list(range(3)))

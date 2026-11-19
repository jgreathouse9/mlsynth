"""The certified search under the relative key: power is the only criterion.

certify_treatment ranks a design (T, C) by the library's relative key,
level_T / required_impact (mlsynth.utils.tbr_helpers.design.objective.score_split,
objective="relative"): no assumption gates and no correlation term. Every design
shares the window, so the required impact is one constant times sqrt(RSS), and
designs compare on level_T^2 / RSS. Within a treatment group the best design is the
least-RSS one of either slope sign; across groups a group can beat the incumbent
(level*, RSS*) only below the ceiling RSS* (level_T / level*)^2 / (1 + g)^2. The
oracle is enumeration of every design (brute.relative_optimum).
"""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import score_split

from . import fastfloor, treatment
from .brute import mirror_panel, raw_panel, relative_optimum
from .treatment import (beats, certify_treatment, geo_sums, group_level, relative_bar,
                        relative_ceiling, rss_floor)

N_TEST = 6
REL = 1e-9


def key_of(M, T, C):
    return score_split(M[:, sorted(T)].sum(axis=1), M[:, sorted(C)].sum(axis=1),
                       objective="relative", n_test=N_TEST).key[0]


def slope_of(M, T, C):
    y = M[:, sorted(T)].sum(axis=1); x = M[:, sorted(C)].sum(axis=1)
    return float(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------------ smoke


def test_smoke():
    M = raw_panel(0)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert out.certified
    assert len(out.treatment) == 2 and out.controls
    assert not set(out.treatment) & set(out.controls)
    assert np.isfinite(out.key) and out.key > 0
    assert out.key == pytest.approx(key_of(M, out.treatment, out.controls), rel=REL)


# ------------------------------------------------------------------ the optimum


@pytest.mark.parametrize("seed,k", [(0, 1), (1, 2), (2, 2), (3, 1), (4, 2), (5, 2)])
def test_the_design_is_the_relative_optimum(seed, k):
    M = raw_panel(seed)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=3, blank_periods=0)
    best, T, C = relative_optimum(M, k, N_TEST)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)


@pytest.mark.parametrize("seed", [103, 104, 105, 108, 109])
def test_a_negative_slope_optimum_is_found(seed):
    """These panels' optimum has a negative slope (asserted): the search covers it by
    certifying the group on -y beside y."""
    M = mirror_panel(seed, p=7)
    best, T, C = relative_optimum(M, 2, N_TEST)
    assert slope_of(M, T, C) < 0
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert out.key == pytest.approx(best, rel=REL)
    assert out.slope == -1


@pytest.mark.parametrize("seed", [103, 104, 105, 108, 109])
def test_a_poor_start_still_reaches_a_negative_slope_optimum(seed):
    """From a start group far from the optimum and no pass, only the certificates on
    -y can find the negative-slope design: a search over positive slopes alone
    reports a worse one."""
    M = mirror_panel(seed, p=7)
    best, _, _ = relative_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=1, pass_restarts=0, start=[3, 4], blank_periods=0)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)


def test_the_slope_reported_is_the_designs():
    for seed in (0, 1, 103):
        M = raw_panel(seed) if seed < 100 else mirror_panel(seed, p=7)
        out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
        assert out.slope == (1 if slope_of(M, out.treatment, out.controls) > 0 else -1)


def test_the_rss_and_level_are_the_designs():
    M = raw_panel(2)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    y = M[:, out.treatment].sum(axis=1); x = M[:, out.controls].sum(axis=1)
    b, a = np.polyfit(x, y, 1)
    assert float(out.rss) == pytest.approx(float(((y - a - b * x) ** 2).sum()), rel=1e-7)
    assert out.level == sum(Q(float(v)) for j in out.treatment for v in M[:, j].tolist()) / len(y)


def test_the_gates_are_reported_and_do_not_rank():
    """The four gates are read at the certified design and reported; the design is
    the key's optimum whether it passes them or not."""
    from .gates import gates_of

    hit = 0
    for seed in range(6):
        M = raw_panel(seed, n=30)
        out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
        y = M[:, out.treatment].sum(axis=1)
        assert out.gates == gates_of(y, M, out.controls, N_TEST)
        assert set(out.gates) == {"corr_test", "aa_test", "bb_test", "dw_test"}
        hit += not all(out.gates.values())
        assert out.key == pytest.approx(relative_optimum(M, 2, N_TEST)[0], rel=REL)
    assert hit > 0                                   # some optimum fails a gate: the gates did not rank


def test_the_gap_holds():
    """At gap g the optimum's key is at most (1 + g) times the reported design's."""
    for seed in range(4):
        M = raw_panel(seed)
        best, _, _ = relative_optimum(M, 2, N_TEST)
        for g in (Q(1, 100), Q(1, 4)):
            out = certify_treatment(M, 2, n_test=N_TEST, restarts=1, pass_restarts=0, start=[0, 1], gap=g, blank_periods=0)
            assert out.certified
            assert best <= (1 + float(g)) * out.key * (1 + REL)


def test_workers_give_the_sequential_design():
    for seed in (1, 4, 104):
        M = raw_panel(seed, p=7) if seed < 100 else mirror_panel(seed, p=7)
        one = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
        four = certify_treatment(M, 2, n_test=N_TEST, restarts=3, n_jobs=4, blank_periods=0)
        assert (one.treatment, one.controls, one.key) == (four.treatment, four.controls, four.key)


def test_an_exact_fit_is_never_reported():
    """The library scores a design whose regression fits exactly at -inf: geo 1 is
    an affine copy of geo 0, so ({1}, {0}) has RSS zero, and the search reports the
    best of the other designs."""
    M = raw_panel(3, p=5)
    M[:, 1] = 2.0 * M[:, 0] + 4.0
    assert key_of(M, [1], [0]) == -np.inf
    out = certify_treatment(M, 1, n_test=N_TEST, restarts=3, blank_periods=0)
    best, _, _ = relative_optimum(M, 1, N_TEST)
    assert out.certified and np.isfinite(out.key)
    assert out.key == pytest.approx(best, rel=REL)


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), p=st.integers(4, 6), k=st.integers(1, 2), mirror=st.booleans())
def test_the_certified_key_is_the_enumerated_optimum(seed, p, k, mirror):
    M = mirror_panel(seed, p=p) if mirror else raw_panel(seed, p=p)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=2, blank_periods=0)
    best, _, _ = relative_optimum(M, k, N_TEST)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)


# ------------------------------------------------------------------ the ceiling


def test_the_ceiling():
    bar = relative_bar(Q(50), Q(10), Q(0))
    assert relative_ceiling(Q(20), bar) == Q(50) * (Q(20) / Q(10)) ** 2
    bar = relative_bar(Q(50), Q(10), Q(1, 10))
    assert relative_ceiling(Q(20), bar) == Q(50) * (Q(20) / Q(10)) ** 2 / Q(11, 10) ** 2
    assert relative_ceiling(Q(20), None) is None


def test_beats_is_strict():
    assert beats(Q(2), Q(3), Q(1), Q(1))                  # 4 / 3 > 1 / 1
    assert not beats(Q(2), Q(4), Q(1), Q(1))              # equal keys: the incumbent stays
    assert not beats(Q(1), Q(2), Q(1), Q(1))


@settings(max_examples=200, deadline=None)
@given(rss=st.fractions(min_value=Q(1, 1000), max_value=1000), lev=st.fractions(min_value=Q(1, 100), max_value=100),
       rss_s=st.fractions(min_value=Q(1, 1000), max_value=1000), lev_s=st.fractions(min_value=Q(1, 100), max_value=100))
def test_below_the_ceiling_is_beating(rss, lev, rss_s, lev_s):
    """At g = 0 a design is below its group's ceiling exactly when it beats the
    incumbent."""
    c = relative_ceiling(lev, relative_bar(rss_s, lev_s, Q(0)))
    assert (rss < c) == beats(lev, rss, lev_s, rss_s)


def test_levels_are_exact():
    M = raw_panel(1)
    S = geo_sums(M)
    assert S == [sum(Q(float(v)) for v in M[:, j].tolist()) for j in range(M.shape[1])]
    assert group_level(S, (0, 2), M.shape[0]) == (S[0] + S[2]) / M.shape[0]


# ------------------------------------------------------------------ the floors


@pytest.mark.parametrize("seed", [0, 1, 103])
def test_the_negative_floor_bounds_every_negative_slope_design(seed):
    """rss_floor(..., sign=-1) is at most the RSS of every design of the group whose
    slope is negative: such a design is a positive-slope design for -y."""
    M = raw_panel(seed) if seed < 100 else mirror_panel(seed, p=6)
    Mc = M - M.mean(axis=0)
    lam = treatment.gram_floor(Mc)
    from itertools import combinations
    for T in combinations(range(M.shape[1]), 2):
        y, X, _ = treatment._split(Mc, T)
        f = float(rss_floor(Mc, T, lam, sign=-1))
        for r in range(1, X.shape[1] + 1):
            for C in combinations(range(X.shape[1]), r):
                x = X[:, list(C)].sum(axis=1)
                b = (y @ x) / (x @ x)
                if b < 0:
                    assert f <= float(((y - b * x) ** 2).sum()) * (1 + 1e-12)


@pytest.mark.skipif(not fastfloor.AVAILABLE, reason="numba is not installed")
def test_the_kernel_floors_each_sign_as_rss_floor_does():
    M = raw_panel(4, p=7)
    Mc = M - M.mean(axis=0)
    lam = treatment.gram_floor(Mc)
    groups = np.array([(0, 1), (2, 5), (3, 6)], dtype=np.int64)
    for sign in (1, -1):
        fl, lo, hi = fastfloor.floors(np.ascontiguousarray(Mc), np.ascontiguousarray(Mc.T @ Mc), groups,
                                      float(lam), float(sign))
        for f, T in zip(fl, groups.tolist()):
            ref = float(rss_floor(Mc, T, lam, sign=sign))
            assert max(f, 0.0) == pytest.approx(ref, rel=1e-9, abs=1e-9)


# ------------------------------------------------------------------ refusals


def test_a_geo_whose_level_is_not_positive_is_refused():
    M = raw_panel(0)
    M[:, 3] -= M[:, 3].mean()                          # mean zero
    M[:, 4] -= M[:, 4].mean() + 1.0                    # mean -1
    with pytest.raises(MlsynthDataError, match=r"3.*4|level"):
        certify_treatment(M, 2, n_test=N_TEST, blank_periods=0)


def test_a_centred_panel_is_refused():
    M = raw_panel(0)
    with pytest.raises(MlsynthDataError):
        certify_treatment(M - M.mean(axis=0), 2, n_test=N_TEST, blank_periods=0)

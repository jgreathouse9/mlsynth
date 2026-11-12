"""Properties of ``objective="relative"``, over the input domain.

Each invariant the relative key promises, asserted by ``hypothesis`` over
random panels and designs, not only at the fixtures in
``test_tbr_relative_objective.py``:

* the key is unit-free: rescaling both aggregates by one positive constant
  leaves it unchanged, which the reference's power term is not;
* it reads the treated aggregate's level and not the control's: shifting the
  control aggregate by a constant leaves it unchanged;
* for a fixed treatment group it orders control groups exactly as the
  required impact does, so control matching is unchanged in kind;
* the reported relative MDE is the key's reciprocal over the test length;
* the batched screen agrees with the direct fit, design for design;
* the local search never ends below its start and returns the direct key.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.tbr_helpers.design.batched import design_cache, score_designs
from mlsynth.utils.tbr_helpers.design.local import local_search
from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.design.search import ROLES, greedy_search

ALL = frozenset(ROLES)
_SETTINGS = dict(max_examples=40, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow])


def _panel(seed, n_geos, n_periods):
    rng = np.random.default_rng(seed)
    size = 100.0 * rng.lognormal(0.0, 0.8, n_geos)
    common = 1.0 + 0.6 * rng.normal(size=(n_periods, 1))
    idio = 1.0 + 0.12 * rng.normal(size=(n_periods, n_geos))
    return size * (0.5 * common + 0.5 * idio)


def _split(Y, T, C):
    return Y[:, sorted(T)].sum(axis=1), Y[:, sorted(C)].sum(axis=1)


def _rel(y, x, n_test):
    return score_split(y, x, objective="relative", n_test=n_test)


panels = st.tuples(st.integers(0, 10_000), st.integers(5, 12),
                   st.integers(16, 50))


@settings(**_SETTINGS)
@given(panels, st.floats(1e-3, 1e3), st.integers(1, 6))
def test_the_key_is_unit_free(panel, scale, n_test):
    seed, N, n = panel
    Y = _panel(seed, N, n)
    y, x = _split(Y, [0], range(1, N))
    a = _rel(y, x, n_test)
    b = _rel(scale * y, scale * x, n_test)
    assert b.value == pytest.approx(a.value, rel=1e-9)
    assert b.detail["relative_mde"] == pytest.approx(a.detail["relative_mde"],
                                                     rel=1e-9)


@settings(**_SETTINGS)
@given(panels, st.floats(-1e4, 1e4), st.integers(1, 6))
def test_the_level_is_the_treated_aggregates(panel, shift, n_test):
    seed, N, n = panel
    Y = _panel(seed, N, n)
    y, x = _split(Y, [0, 1], range(2, N))
    a = _rel(y, x, n_test)
    b = _rel(y, x + shift, n_test)
    assert b.value == pytest.approx(a.value, rel=1e-7)
    assert b.detail["level"] == pytest.approx(float(np.mean(y)), rel=1e-12)


@settings(**_SETTINGS)
@given(panels, st.integers(1, 6), st.data())
def test_for_a_fixed_treatment_group_controls_rank_by_required_impact(
        panel, n_test, data):
    seed, N, n = panel
    Y = _panel(seed, N, n)
    k = data.draw(st.integers(1, N // 2))
    T = list(range(k))
    rest = list(range(k, N))
    subsets = st.lists(st.sampled_from(rest), min_size=1, unique=True)
    C1, C2 = data.draw(subsets), data.draw(subsets)
    s1 = _rel(*_split(Y, T, C1), n_test)
    s2 = _rel(*_split(Y, T, C2), n_test)
    assert s1.detail["level"] == s2.detail["level"]
    i1, i2 = s1.detail["required_impact"], s2.detail["required_impact"]
    if i1 < i2:
        assert s1.key >= s2.key
    elif i2 < i1:
        assert s2.key >= s1.key
    else:
        assert s1.key == s2.key


@settings(**_SETTINGS)
@given(panels, st.integers(1, 6))
def test_the_relative_mde_is_the_keys_reciprocal_over_the_test(panel, n_test):
    seed, N, n = panel
    s = _rel(*_split(_panel(seed, N, n), [N - 1], range(N - 1)), n_test)
    assert s.detail["relative_mde"] * s.value * n_test == pytest.approx(1.0,
                                                                        rel=1e-12)


@settings(**_SETTINGS)
@given(panels, st.integers(1, 4), st.data())
def test_the_batched_screen_agrees_with_the_direct_fit(panel, n_test, data):
    seed, N, n = panel
    Y = _panel(seed, N, n)
    rng = np.random.default_rng(data.draw(st.integers(0, 10_000)))
    designs = []
    for _ in range(12):
        perm = rng.permutation(N)
        k = int(rng.integers(1, N // 2 + 1))
        c = int(rng.integers(1, N - k + 1))
        designs.append((perm[:k].tolist(), perm[k:k + c].tolist()))
    UT = np.zeros((len(designs), N))
    UC = np.zeros((len(designs), N))
    for r, (T, C) in enumerate(designs):
        UT[r, T] = 1.0
        UC[r, C] = 1.0
    keys = score_designs(design_cache(Y, n_test=n_test), UT, UC,
                         objective="relative")
    for row, (T, C) in zip(keys, designs):
        assert row[0] == pytest.approx(_rel(*_split(Y, T, C), n_test).value,
                                       rel=1e-8)


@settings(max_examples=12, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 10_000), st.integers(6, 9), st.integers(0, 2))
def test_the_local_search_never_ends_below_its_start(seed, N, restarts):
    Y = _panel(seed, N, 30)
    roles = [ALL] * N
    g = greedy_search(Y, roles, max_treatment_size=2, n_test=5,
                      objective="relative")[-1]
    out = local_search(Y, roles, g, n_test=5, restarts=restarts, seed=seed,
                       objective="relative")
    assert out.score.key >= g.score.key
    assert out.score.key == _rel(*_split(Y, out.treatment, out.control), 5).key

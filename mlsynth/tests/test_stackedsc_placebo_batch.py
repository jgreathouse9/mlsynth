"""The placebo layer's refits, solved as leave-one-out families.

Test-first (per ``agents/agents_tests.md``).

Casting each donor in a pool as treated in turn and refitting against the rest
is a leave-one-out family over one matrix, in both donor pools the estimator
offers. Under ``donors-only`` that matrix is the cohort's design and the family
is shared by every unit in the cohort. Under ``permutation`` the treated unit
joins each pool, so the matrix is the design with that unit's column appended
and the family is one per treated unit -- its targets are the donors, and the
appended column is a donor to all of them and a target to none, which is what
``n_targets`` short of the column count expresses.

This is where STACKEDSC's time is: 566 treated units against 39 donors is 22,074
programs under the default pool, roughly 1.2 minutes on the Wiltshire panel.

The p-values are rank statistics over these fits, so the batch has to reproduce
the one-at-a-time solve and not merely tie with it. What these tests pin is that
nothing observable moves: the same paths, the same fit count, the same p-values.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


def _staggered(n_treated=4, n_donors=7, T=12, cohorts=(6, 8), seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    rows = []
    for j in range(n_donors):
        load, base = rng.normal(size=2), 100.0 + 20.0 * rng.random()
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "adopt": np.nan,
                         "y": float(base + F[t] @ load + rng.normal() * 0.05)})
    for i in range(n_treated):
        g = cohorts[i % len(cohorts)]
        load, base = rng.normal(size=2), 100.0 + 20.0 * rng.random()
        for t in range(T):
            rows.append({"unit": f"x{i}", "t": t, "adopt": float(g),
                         "y": float(base + F[t] @ load + rng.normal() * 0.05)})
    d = pd.DataFrame(rows)
    d["treat"] = ((~d.adopt.isna()) & (d.t >= d.adopt)).astype(int)
    return d


def _fit(df=None, **kw):
    from mlsynth import STACKEDSC

    df = _staggered() if df is None else df
    cfg = dict(df=df, unitid="unit", time="t", outcome="y", treat="treat",
               inference="placebo", n_placebo_samples=200, seed=0)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STACKEDSC(cfg).fit()


@pytest.fixture
def unbatched(monkeypatch):
    """Force the one-at-a-time path, so a run can be compared against it."""
    import mlsynth.utils.stackedsc_helpers.inference as INF

    monkeypatch.setattr(INF, "_placebo_weights",
                        lambda *a, **k: None, raising=True)
    return None


# --------------------------------------------------------------------------- #
# 1. Nothing observable moves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pool", ["permutation", "donors-only"])
def test_paths_match_the_one_at_a_time_solve(pool, monkeypatch):
    import mlsynth.utils.stackedsc_helpers.inference as INF

    batched = _fit(placebo_donor_pool=pool)
    monkeypatch.setattr(INF, "_placebo_weights", lambda *a, **k: None)
    looped = _fit(placebo_donor_pool=pool)

    assert set(batched.placebo.paths) == set(looped.placebo.paths)
    for unit, path in batched.placebo.paths.items():
        np.testing.assert_allclose(np.asarray(path, float),
                                   np.asarray(looped.placebo.paths[unit], float),
                                   atol=1e-9)


@pytest.mark.parametrize("pool", ["permutation", "donors-only"])
def test_the_p_values_are_unchanged(pool, monkeypatch):
    import mlsynth.utils.stackedsc_helpers.inference as INF

    batched = _fit(placebo_donor_pool=pool)
    monkeypatch.setattr(INF, "_placebo_weights", lambda *a, **k: None)
    looped = _fit(placebo_donor_pool=pool)

    # The RMSPE ranking is the statistic the layer exists for, and a rank can
    # move on an arbitrarily small perturbation, so it is compared exactly.
    assert batched.placebo.rmspe_p == looped.placebo.rmspe_p
    assert set(batched.placebo.rmspe_actual) == set(looped.placebo.rmspe_actual)
    for e, v in batched.placebo.rmspe_actual.items():
        assert abs(v - looped.placebo.rmspe_actual[e]) < 1e-9

    for attr in ("p_variance", "ci_lower", "ci_upper", "se"):
        np.testing.assert_allclose(
            np.asarray(getattr(batched.placebo, attr), float),
            np.asarray(getattr(looped.placebo, attr), float), atol=1e-9)
    for attr in ("att_se", "att_p_value", "att_ci_lower", "att_ci_upper"):
        assert abs(getattr(batched.placebo, attr)
                   - getattr(looped.placebo, attr)) < 1e-9


def test_the_fit_count_still_counts_programs_not_solver_calls():
    """``n_fits`` reports the programs the placebo layer poses. Solving them
    together does not change how many there are."""
    res = _fit()
    assert res.placebo.n_fits == 4 * 7
    shared = _fit(placebo_donor_pool="donors-only")
    assert shared.placebo.n_fits == 2 * 7


# --------------------------------------------------------------------------- #
# 2. The properties the placebo layer is built on survive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pool", ["permutation", "donors-only"])
def test_every_placebo_is_indexed_to_its_own_base_period(pool):
    """A pseudo-treated donor's gap at e = -1 is zero by construction, the same
    identity the real treated units satisfy."""
    res = _fit(placebo_donor_pool=pool)
    e = np.asarray(res.placebo.horizons)
    at_base = np.flatnonzero(e == -1)
    if at_base.size:
        for path in res.placebo.paths.values():
            assert np.max(np.abs(np.asarray(path, float)[:, at_base[0]])) < 1e-9


def test_dropping_the_treated_unit_makes_the_path_a_cohort_property():
    """Under ``donors-only`` every unit in a cohort faces the same design and
    the same targets, so the batch is shared and the paths coincide."""
    res = _fit(placebo_donor_pool="donors-only")
    a = np.asarray(res.placebo.paths["x0"], float)
    b = np.asarray(res.placebo.paths["x2"], float)      # same cohort
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_the_treated_unit_is_a_donor_to_its_placebos_but_never_a_target():
    """Under ``permutation`` the pool has one more column than there are rows to
    report, which is what the family's target border expresses."""
    res = _fit(placebo_donor_pool="permutation")
    for path in res.placebo.paths.values():
        assert np.asarray(path, float).shape[0] == 7


# --------------------------------------------------------------------------- #
# 3. A binding predicate and other edges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pool", ["permutation", "donors-only"])
def test_a_binding_predicate_matches_the_one_at_a_time_solve(pool, monkeypatch):
    import mlsynth.utils.stackedsc_helpers.inference as INF

    pred = lambda unit, donor: donor != "d0"
    batched = _fit(placebo_donor_pool=pool, donor_predicate=pred)
    monkeypatch.setattr(INF, "_placebo_weights", lambda *a, **k: None)
    looped = _fit(placebo_donor_pool=pool, donor_predicate=pred)
    for unit, path in batched.placebo.paths.items():
        np.testing.assert_allclose(np.asarray(path, float),
                                   np.asarray(looped.placebo.paths[unit], float),
                                   atol=1e-9)
        assert np.asarray(path, float).shape[0] == 6


def test_a_pool_of_two_donors_is_the_smallest_that_admits_a_placebo(monkeypatch):
    import mlsynth.utils.stackedsc_helpers.inference as INF

    pred = lambda unit, donor: donor in ("d0", "d1")
    batched = _fit(donor_predicate=pred)
    monkeypatch.setattr(INF, "_placebo_weights", lambda *a, **k: None)
    looped = _fit(donor_predicate=pred)
    for unit, path in batched.placebo.paths.items():
        np.testing.assert_allclose(np.asarray(path, float),
                                   np.asarray(looped.placebo.paths[unit], float),
                                   atol=1e-9)


def test_covariate_matching_still_matches_the_one_at_a_time_solve(monkeypatch):
    """With predictors the design is sqrt(V)-scaled and badly conditioned, so
    most members fail certification and are re-solved. The answer is the same
    either way."""
    import mlsynth.utils.stackedsc_helpers.inference as INF

    df = _staggered()
    rng = np.random.default_rng(1)
    df["z"] = rng.normal(size=len(df)) * 1e4

    batched = _fit(df=df, covariates=["z"])
    monkeypatch.setattr(INF, "_placebo_weights", lambda *a, **k: None)
    looped = _fit(df=df, covariates=["z"])
    for unit, path in batched.placebo.paths.items():
        np.testing.assert_allclose(np.asarray(path, float),
                                   np.asarray(looped.placebo.paths[unit], float),
                                   atol=1e-9)


def test_a_pool_too_small_for_a_placebo_is_still_reported():
    """One donor leaves nothing to fit a placebo against, and the batch must not
    swallow the error the loop raised."""
    from mlsynth.exceptions import MlsynthDataError

    with pytest.raises(MlsynthDataError, match="donor"):
        _fit(donor_predicate=lambda unit, donor: donor == "d0")

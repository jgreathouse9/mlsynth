"""Typed returns from ``dataprep`` (issue #1138, part 2).

``dataprep`` returns a :class:`~mlsynth.utils.prepared.SinglePanel` when one unit is treated and a
:class:`~mlsynth.utils.prepared.CohortPanel` when several are, each cohort a
:class:`~mlsynth.utils.prepared.Cohort`. Each type checks that its pieces agree when it is built (the outcome
vector and the donor matrix have a row per period, the pre- and post-periods add up to the total, a name per donor
column, the covariate and predictor blocks shaped to the units), and raises ``MlsynthDataError`` when they do not.

The 60 library callers read the result by string key today. Each type is also a read-only mapping with exactly the
keys the dict had, in the same order, so callers migrate one at a time.
"""
from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.datautils import dataprep
from mlsynth.utils.prepared import Cohort, CohortPanel, CovariateBlock, PredictorBlock, SinglePanel

SINGLE_KEYS = ["treated_unit_name", "Ywide", "y", "donor_names", "donor_matrix", "total_periods", "pre_periods",
               "post_periods", "time_labels"]
COHORT_KEYS = ["Ywide", "cohorts", "time_labels"]
COHORT_ENTRY_KEYS = ["treated_units", "y", "donor_names", "donor_matrix", "total_periods", "pre_periods",
                     "post_periods"]
COVARIATE_KEYS = ["covariate_matrix", "covariate_names", "covariate_means", "covariate_scales"]
SPLIT_KEYS = ["donor_covariate_matrix", "treated_covariate_matrix"]
PREDICTOR_KEYS = ["predictor_matrix", "predictor_block_sizes", "f_weights", "X_population_mean"]


def _panel(n_units=4, T=6, starts=None, seed=0, covariates=False):
    """A balanced long panel. ``starts`` maps unit position to its first treated period (0-based)."""
    starts = {0: T // 2} if starts is None else starts
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        level = rng.normal() * 5
        for t in range(T):
            row = {"unit": f"u{u}", "time": 2000 + t, "y": level + rng.normal(),
                   "d": int(u in starts and t >= starts[u])}
            if covariates:
                row["x1"] = rng.normal()
                row["x2"] = float(u)
            rows.append(row)
    return pd.DataFrame(rows)


def _prep(df, **kw):
    return dataprep(df, "unit", "time", "y", "d", **kw)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_one_treated_unit_gives_a_single_panel():
    p = _prep(_panel())
    assert isinstance(p, SinglePanel) and isinstance(p, Mapping)
    assert p.treated_unit_name == "u0"
    assert p.y.shape == (6,) and p.donor_matrix.shape == (6, 3)
    assert (p.pre_periods, p.post_periods, p.total_periods) == (3, 3, 6)
    assert p.covariates is None and p.predictors is None


def test_smoke_several_treated_units_give_a_cohort_panel():
    p = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 3, 2: 5}))
    assert isinstance(p, CohortPanel) and isinstance(p, Mapping)
    assert sorted(p.cohorts) == [2003, 2005]
    for c in p.cohorts.values():
        assert isinstance(c, Cohort) and isinstance(c, Mapping)
    assert p.cohorts[2003].treated_units == ["u0", "u1"]
    assert p.cohorts[2003].y.shape == (8, 2)
    assert list(p.cohorts[2005].donor_names) == ["u3", "u4"]


# --- unit: the typed fields --------------------------------------------------------------------------------------

def test_the_single_panel_fields_are_the_panel_read_directly():
    df = _panel(seed=3)
    p = _prep(df)
    wide = df.pivot(index="time", columns="unit", values="y")
    np.testing.assert_array_equal(p.y, wide["u0"].to_numpy())
    np.testing.assert_array_equal(p.donor_matrix, wide[["u1", "u2", "u3"]].to_numpy())
    assert list(p.donor_names) == ["u1", "u2", "u3"]
    assert list(p.time_labels) == list(range(2000, 2006))
    pd.testing.assert_frame_equal(p.Ywide, wide, check_names=False)


def test_the_covariate_block_splits_donors_then_the_treated_unit():
    p = _prep(_panel(covariates=True), covariates=["x1", "x2"])
    cov = p.covariates
    assert isinstance(cov, CovariateBlock)
    assert cov.matrix.shape == (4, 2) and cov.names == ("x1", "x2")
    np.testing.assert_array_equal(cov.donor_matrix, cov.matrix[:3])
    np.testing.assert_array_equal(cov.treated_matrix, cov.matrix[3:])
    assert p.predictors is None


def test_the_predictor_block_stacks_the_pre_period_outcomes_on_the_covariates():
    p = _prep(_panel(covariates=True), covariates=["x1", "x2"], marex=True)
    pred = p.predictors
    assert isinstance(pred, PredictorBlock)
    assert pred.block_sizes == (3, 2) and pred.matrix.shape == (5, 4)
    np.testing.assert_allclose(pred.f_weights, np.full(4, 0.25))
    np.testing.assert_allclose(pred.population_mean, pred.matrix @ pred.f_weights)


def test_marex_without_covariates_gives_a_predictor_block_of_outcomes_only():
    p = _prep(_panel(), marex=True)
    assert p.covariates is None
    assert p.predictors.block_sizes == (3, 0) and p.predictors.matrix.shape == (3, 4)


def test_a_cohort_panel_carries_its_covariate_block_without_the_split():
    p = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5}, covariates=True), covariates=["x1"])
    assert p.covariates.matrix.shape == (5, 1)
    assert p.covariates.donor_matrix is None and p.covariates.treated_matrix is None


# --- unit: the mapping view ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw,extra", [
    ({}, []),
    (dict(covariates=["x1", "x2"]), COVARIATE_KEYS + SPLIT_KEYS),
    (dict(covariates=["x1", "x2"], marex=True), COVARIATE_KEYS + SPLIT_KEYS + PREDICTOR_KEYS),
    (dict(marex=True), PREDICTOR_KEYS),
])
def test_the_single_panel_view_has_the_dicts_keys_in_order(kw, extra):
    p = _prep(_panel(covariates=True), **kw)
    assert list(p) == SINGLE_KEYS + extra
    assert len(p) == len(SINGLE_KEYS + extra)
    assert list(p.keys()) == list(dict(p).keys())


@pytest.mark.parametrize("kw,extra", [
    ({}, []),
    (dict(covariates=["x1"]), COVARIATE_KEYS),
    (dict(covariates=["x1"], marex=True), COVARIATE_KEYS + PREDICTOR_KEYS),
    (dict(marex=True), PREDICTOR_KEYS),
])
def test_the_cohort_panel_view_has_the_dicts_keys_in_order(kw, extra):
    p = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5}, covariates=True), **kw)
    assert list(p) == COHORT_KEYS + extra
    for c in p["cohorts"].values():
        assert list(c) == COHORT_ENTRY_KEYS


def test_the_view_returns_the_fields_themselves():
    p = _prep(_panel(covariates=True), covariates=["x1", "x2"], marex=True)
    for key in SINGLE_KEYS:
        assert p[key] is getattr(p, key)
    assert p["covariate_matrix"] is p.covariates.matrix
    assert p["covariate_names"] is p.covariates.names
    assert p["covariate_means"] is p.covariates.means
    assert p["covariate_scales"] is p.covariates.scales
    assert p["donor_covariate_matrix"] is p.covariates.donor_matrix
    assert p["treated_covariate_matrix"] is p.covariates.treated_matrix
    assert p["predictor_matrix"] is p.predictors.matrix
    assert p["predictor_block_sizes"] is p.predictors.block_sizes
    assert p["f_weights"] is p.predictors.f_weights
    assert p["X_population_mean"] is p.predictors.population_mean
    q = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5}))
    assert q["cohorts"] is q.cohorts and q["Ywide"] is q.Ywide and q["time_labels"] is q.time_labels
    c = q.cohorts[2003]
    for key in COHORT_ENTRY_KEYS:
        assert c[key] is getattr(c, key)


def test_membership_get_and_missing_keys_behave_as_the_dict_did():
    p = _prep(_panel())
    assert "y" in p and "cohorts" not in p and "covariate_matrix" not in p
    assert p.get("cohorts") is None and p.get("cohorts", 7) == 7 and p.get("pre_periods") == 3
    with pytest.raises(KeyError):
        p["cohorts"]
    q = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5}))
    assert "cohorts" in q and "y" not in q
    with pytest.raises(KeyError):
        q["y"]


def test_the_types_are_read_only():
    p = _prep(_panel())
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.pre_periods = 4
    with pytest.raises(TypeError):
        p["pre_periods"] = 4
    c = _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5})).cohorts[2003]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.pre_periods = 1


def test_the_types_pickle():
    for p in (_prep(_panel(covariates=True), covariates=["x1"], marex=True),
              _prep(_panel(n_units=5, T=8, starts={0: 3, 1: 5}))):
        q = pickle.loads(pickle.dumps(p))
        assert type(q) is type(p) and list(q) == list(p)
        pd.testing.assert_frame_equal(q.Ywide, p.Ywide)


# --- edge ---------------------------------------------------------------------------------------------------------

def test_no_donors_when_allowed_gives_an_empty_donor_block():
    df = _panel(n_units=1, T=4, starts={0: 2})
    p = _prep(df, allow_no_donors=True)
    assert isinstance(p, SinglePanel)
    assert p.donor_matrix.shape == (4, 0) and len(p.donor_names) == 0


def test_a_single_donor_and_a_single_post_period():
    p = _prep(_panel(n_units=2, T=4, starts={0: 3}))
    assert p.donor_matrix.shape == (4, 1) and (p.pre_periods, p.post_periods) == (3, 1)


def test_treatment_in_the_first_period_is_still_refused():
    with pytest.raises(MlsynthDataError):
        _prep(_panel(starts={0: 0}))


# --- failure: the types check their own consistency ----------------------------------------------------------------

def _single(**over):
    T, J = 5, 2
    names = pd.Index(["a", "b"])
    base = dict(treated_unit_name="t", Ywide=pd.DataFrame(np.zeros((T, J + 1)), columns=["a", "b", "t"]),
                y=np.zeros(T), donor_names=names, donor_matrix=np.zeros((T, J)), total_periods=T, pre_periods=3,
                post_periods=2, time_labels=pd.Index(range(T)))
    base.update(over)
    return SinglePanel(**base)


def _cohort(**over):
    T = 5
    base = dict(treated_units=["t"], y=np.zeros((T, 1)), donor_names=["a", "b"], donor_matrix=np.zeros((T, 2)),
                total_periods=T, pre_periods=3, post_periods=2)
    base.update(over)
    return Cohort(**base)


def test_a_consistent_hand_built_single_panel_is_accepted():
    assert _single().total_periods == 5


@pytest.mark.parametrize("over,match", [
    (dict(pre_periods=2), "pre_periods"),
    (dict(post_periods=3), "pre_periods"),
    (dict(pre_periods=-1, post_periods=6), "negative"),
    (dict(y=np.zeros(4)), "y"),
    (dict(y=np.zeros((5, 1))), "y"),
    (dict(donor_matrix=np.zeros((4, 2))), "donor_matrix"),
    (dict(donor_matrix=np.zeros(5)), "donor_matrix"),
    (dict(donor_names=pd.Index(["a"])), "donor_names"),
    (dict(time_labels=pd.Index(range(4))), "time_labels"),
    (dict(Ywide=pd.DataFrame(np.zeros((4, 3)))), "Ywide"),
])
def test_an_inconsistent_single_panel_is_refused(over, match):
    with pytest.raises(MlsynthDataError, match=match):
        _single(**over)


@pytest.mark.parametrize("over,match", [
    (dict(pre_periods=1), "pre_periods"),
    (dict(y=np.zeros((5, 2))), "y"),
    (dict(y=np.zeros(5)), "y"),
    (dict(donor_matrix=np.zeros((5, 3))), "donor_matrix"),
    (dict(donor_matrix=np.zeros((4, 2))), "donor_matrix"),
])
def test_an_inconsistent_cohort_is_refused(over, match):
    with pytest.raises(MlsynthDataError, match=match):
        _cohort(**over)


def test_a_cohort_panel_refuses_a_cohort_of_another_length():
    T = 5
    ok = dict(Ywide=pd.DataFrame(np.zeros((T, 3))), cohorts={3: _cohort()}, time_labels=pd.Index(range(T)))
    assert CohortPanel(**ok).cohorts[3].total_periods == 5
    with pytest.raises(MlsynthDataError, match="cohort"):
        CohortPanel(**{**ok, "cohorts": {3: _cohort(), 4: _cohort(total_periods=4, pre_periods=2, post_periods=2,
                                                                   y=np.zeros((4, 1)), donor_matrix=np.zeros((4, 2)))}})
    with pytest.raises(MlsynthDataError, match="time_labels"):
        CohortPanel(**{**ok, "time_labels": pd.Index(range(4))})


@pytest.mark.parametrize("over,match", [
    (dict(names=("x1",)), "names"),
    (dict(means=np.zeros(1)), "means"),
    (dict(scales=np.zeros(3)), "scales"),
    (dict(donor_matrix=np.zeros((3, 2))), "donor_matrix"),
    (dict(treated_matrix=np.zeros((2, 2))), "treated_matrix"),
])
def test_an_inconsistent_covariate_block_is_refused(over, match):
    base = dict(matrix=np.zeros((3, 2)), names=("x1", "x2"), means=np.zeros(2), scales=np.ones(2),
                donor_matrix=np.zeros((2, 2)), treated_matrix=np.zeros((1, 2)))
    with pytest.raises(MlsynthDataError, match=match):
        CovariateBlock(**{**base, **over})


def test_a_covariate_block_for_the_wrong_number_of_units_is_refused():
    cov = CovariateBlock(matrix=np.zeros((4, 1)), names=("x",), means=np.zeros(1), scales=np.ones(1),
                         donor_matrix=np.zeros((3, 1)), treated_matrix=np.zeros((1, 1)))
    with pytest.raises(MlsynthDataError, match="covariate"):
        _single(covariates=cov)


@pytest.mark.parametrize("over,match", [
    (dict(block_sizes=(3, 1)), "block_sizes"),
    (dict(f_weights=np.zeros(2)), "f_weights"),
    (dict(population_mean=np.zeros(4)), "population_mean"),
])
def test_an_inconsistent_predictor_block_is_refused(over, match):
    base = dict(matrix=np.zeros((5, 3)), block_sizes=(3, 2), f_weights=np.full(3, 1 / 3),
                population_mean=np.zeros(5))
    with pytest.raises(MlsynthDataError, match=match):
        PredictorBlock(**{**base, **over})


def test_a_predictor_block_for_the_wrong_number_of_units_is_refused():
    pred = PredictorBlock(matrix=np.zeros((3, 2)), block_sizes=(3, 0), f_weights=np.full(2, 0.5),
                          population_mean=np.zeros(3))
    with pytest.raises(MlsynthDataError, match="predictor"):
        _single(predictors=pred)


# --- property -----------------------------------------------------------------------------------------------------

@st.composite
def _panels(draw):
    n = draw(st.integers(1, 7))
    T = draw(st.integers(2, 10))
    treated = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=n, unique=True))
    starts = {u: draw(st.integers(1, T - 1)) for u in treated}
    seed = draw(st.integers(0, 10**6))
    return n, T, starts, seed


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(spec=_panels(), covariates=st.booleans(), marex=st.booleans(), shuffle=st.booleans())
def test_property_the_prepared_panel_agrees_with_the_panel(spec, covariates, marex, shuffle):
    """Property: over balanced panels (any number of units, periods, treated units and start times, rows in any
    order, with or without covariates and the predictor block), the typed result is the panel read directly: each
    outcome block is the wide panel's columns for its units, the periods add up, and the view holds exactly the
    dict's keys, each value the field itself."""
    n, T, starts, seed = spec
    df = _panel(n_units=n, T=T, starts=starts, seed=seed, covariates=True)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed)
    kw = dict(covariates=["x1", "x2"] if covariates else None, marex=marex)
    no_donors = len(starts) == n
    if len(starts) == 1 and no_donors:
        p = _prep(df, allow_no_donors=True, **kw)
    else:
        p = _prep(df, **kw)
    wide = df.pivot(index="time", columns="unit", values="y").sort_index()
    extra_c = COVARIATE_KEYS if covariates else []
    extra_p = PREDICTOR_KEYS if marex else []
    if len(starts) == 1:
        (u, s), = starts.items()
        assert isinstance(p, SinglePanel)
        assert p.treated_unit_name == f"u{u}"
        np.testing.assert_array_equal(p.y, wide[f"u{u}"].to_numpy())
        np.testing.assert_array_equal(p.donor_matrix, wide[list(p.donor_names)].to_numpy())
        assert sorted(p.donor_names) == sorted(c for c in wide.columns if c != f"u{u}")
        assert (p.pre_periods, p.post_periods, p.total_periods) == (s, T - s, T)
        assert list(p) == SINGLE_KEYS + extra_c + (SPLIT_KEYS if covariates else []) + extra_p
    else:
        assert isinstance(p, CohortPanel)
        assert list(p) == COHORT_KEYS + extra_c + extra_p
        assert sorted(p.cohorts) == sorted({2000 + s for s in starts.values()})
        for start, c in p.cohorts.items():
            s = start - 2000
            assert sorted(c.treated_units) == sorted(f"u{u}" for u, v in starts.items() if v == s)
            np.testing.assert_array_equal(c.y, wide[list(c.treated_units)].to_numpy())
            np.testing.assert_array_equal(c.donor_matrix, wide[list(c.donor_names)].to_numpy())
            assert sorted(c.donor_names) == sorted(f"u{v}" for v in range(n) if v not in starts)
            assert (c.pre_periods, c.post_periods, c.total_periods) == (s, T - s, T)
            assert list(c) == COHORT_ENTRY_KEYS
    assert list(p.time_labels) == list(range(2000, 2000 + T))
    for key in p:
        assert p[key] is p.get(key)


@settings(max_examples=80, deadline=None)
@given(T=st.integers(1, 9), J=st.integers(0, 5), pre=st.integers(-2, 10), post=st.integers(-2, 10),
       dy=st.integers(-1, 1), dd=st.integers(-1, 1), dn=st.integers(-1, 1), dl=st.integers(-1, 1))
def test_property_a_single_panel_is_built_exactly_when_its_pieces_agree(T, J, pre, post, dy, dd, dn, dl):
    """Property: over every combination of piece sizes, including the invalid ones, construction succeeds exactly
    when the periods add up and are non-negative and every piece has the panel's length, and raises
    MlsynthDataError otherwise."""
    names = pd.Index([f"d{j}" for j in range(max(J + dn, 0))])
    kw = dict(treated_unit_name="t", Ywide=pd.DataFrame(np.zeros((T, J + 1))), y=np.zeros(max(T + dy, 0)),
              donor_names=names, donor_matrix=np.zeros((max(T + dd, 0), J)), total_periods=T, pre_periods=pre,
              post_periods=post, time_labels=pd.Index(range(max(T + dl, 0))))
    ok = (pre >= 0 and post >= 0 and pre + post == T and kw["y"].shape == (T,) and len(names) == J
          and kw["donor_matrix"].shape[0] == T and len(kw["time_labels"]) == T)
    if ok:
        assert SinglePanel(**kw).total_periods == T
    else:
        with pytest.raises(MlsynthDataError):
            SinglePanel(**kw)

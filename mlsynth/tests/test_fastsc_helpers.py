import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.fast_scm_helpers import fast_scm_setup as setup_mod
from mlsynth.utils.fast_scm_helpers.structure import (
    Identification,
    IndexSet,
    Inference,
    Losses,
    PredictionVectors,
    SEDCandidate,
    WeightVectors,
)


class _DummySolution:
    label = "cand"


class _DataErr(Exception):
    pass


def _candidate():
    return SEDCandidate(
        identification=Identification(solution=_DummySolution(), treated_idx=np.array([0])),
        weights=WeightVectors(treated=np.array([1.0]), control=np.array([0.0, 1.0])),
        predictions=PredictionVectors(
            synthetic_treated=np.array([0.0, 0.0]),
            synthetic_control=np.array([0.0, 0.0]),
            effects=np.array([0.0, 0.0]),
            residuals_E=np.array([0.0]),
            residuals_B=np.array([0.0]),
        ),
        losses=Losses(0, 0, 0, 0, 0, 0, 0),
        inference=Inference(),
    )


@pytest.fixture
def panel_df():
    return pd.DataFrame(
        {
            "time": [1, 1, 2, 2, 3, 3],
            "unit": ["A", "B", "A", "B", "A", "B"],
            "y": [1.0, 3.0, 2.0, 4.0, 10.0, 7.0],
            "z1": [10, 20, 30, 40, 50, 60],
            "z2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "is_post": [0, 0, 0, 0, 1, 1],
            "eligible": [1, 0, 1, 0, 1, 0],
            "w": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    "post_col, expected_pre, expected_post",
    [("is_post", 4, 2), (None, 6, 0)],
)
def test_prepare_working_df_with_and_without_post_col(panel_df, post_col, expected_pre, expected_post):
    pre_df, post_df = setup_mod._prepare_working_df(panel_df, post_col)
    assert len(pre_df) == expected_pre
    assert len(post_df) == expected_post

def test_prepare_working_df_raises_when_no_pre_rows(monkeypatch, panel_df):
    monkeypatch.setattr(setup_mod, "MlsynthDataError", _DataErr, raising=False)
    all_post = panel_df.assign(is_post=1)
    with pytest.raises(_DataErr, match="No pre-period data"):
        setup_mod._prepare_working_df(all_post, "is_post")


def test_candidate_mask_and_matrix_builders(panel_df):
    unit_idx = IndexSet.from_labels(["A", "B", "C"])
    mask = setup_mod.build_candidate_mask(panel_df, "eligible", unit_idx, "unit")
    np.testing.assert_array_equal(mask, np.array([True, False, False]))

    Y = setup_mod.build_Y_matrix(panel_df[panel_df["is_post"] == 0], "y", "time", "unit", unit_idx)
    assert Y.shape == (2, 3)
    assert np.isnan(Y[:, 2]).all()

    Z = setup_mod.build_Z_matrix(panel_df[panel_df["is_post"] == 0], ["z1", "z2"], "time", "unit", unit_idx)
    assert Z.shape == (4, 3)
    Z_none = setup_mod.build_Z_matrix(panel_df, [], "time", "unit", unit_idx)
    assert Z_none is None


@pytest.mark.parametrize(
    "weight_col, expected",
    [("w", np.array([2 / 3, 1 / 3])), (None, np.array([0.5, 0.5]))],
)
def test_build_f_vector_weighted_and_uniform(panel_df, weight_col, expected):
    unit_idx = IndexSet.from_labels(["A", "B"])
    result = setup_mod.build_f_vector(panel_df, weight_col, "unit", unit_idx)
    np.testing.assert_allclose(result, expected)

def test_prepare_experiment_inputs_smoke_and_edges():
    Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = np.array([[5.0], [6.0]])

    X, f, cidx, T, N = setup_mod.prepare_experiment_inputs(Y, Z=Z, candidate_mask=np.array([True]), m=1)
    assert (T, N) == (2, 3)
    np.testing.assert_array_equal(cidx, np.array([0]))
    assert np.isclose(f.sum(), 1.0)


@pytest.mark.parametrize(
    "candidate_mask, error_match",
    [
        (np.array([True, False, True, False]), "candidate_mask length"),
        (np.array([False, False, False]), "Not enough candidate units"),
    ],
)
def test_prepare_experiment_inputs_edge_errors(candidate_mask, error_match):
    Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = np.array([[5.0], [6.0]])
    with pytest.raises(ValueError, match=error_match):
        setup_mod.prepare_experiment_inputs(Y, Z=Z, candidate_mask=candidate_mask, m=1)


def test_split_periods_and_build_x_tilde():
    post_df = pd.DataFrame({"time": [3, 3, 4, 4], "unit": ["A", "B", "A", "B"]})
    E_idx, B_idx, post_idx = setup_mod.split_periods(T0=4, frac_E=0.5, post_df=post_df, time_col="time")
    np.testing.assert_array_equal(E_idx, np.array([0, 1]))
    np.testing.assert_array_equal(B_idx, np.array([2, 3]))
    np.testing.assert_array_equal(post_idx, np.array([4, 5]))

    X = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 4.0]])
    f = np.array([0.5, 0.5])
    XE, G = setup_mod.build_X_tilde(X, f, idx=np.array([0, 1]), J=2)
    assert XE.shape == (2, 2)
    assert G.shape == (2, 2)
    assert np.isfinite(XE).all()


def test_run_post_intervention_updates_smoke(monkeypatch):
    cand = _candidate()
    unit_idx = IndexSet.from_labels(["A", "B"])
    Y_pre = np.array([[1.0, 2.0], [1.5, 2.5]])
    post_df = pd.DataFrame(
        {
            "time": [3, 3],
            "unit": ["A", "B"],
            "y": [2.0, 3.0],
        }
    )
    post_idx = np.array([2])

    def _fake_post(candidate, post_idx, n_perms, seed):
        candidate.inference.p_value = 0.25
        return candidate

    def _fake_ci(candidate, post_idx, alpha, seed):
        candidate.inference.ci_lower = -1.0
        candidate.inference.ci_upper = 1.0
        return candidate

    monkeypatch.setattr(setup_mod, "compute_post_inference", _fake_post)
    monkeypatch.setattr(setup_mod, "compute_moving_block_conformal_ci", _fake_ci)

    y_pop, updated = setup_mod._run_post_intervention_updates(
        candidate_results=[cand],
        Y_pre=Y_pre,
        post_df=post_df,
        post_idx=post_idx,
        unit_index=unit_idx,
        unitid="unit",
        time="time",
        outcome="y",
        n_sims=50,
        alpha=0.1,
        seed=123,
    )

    assert y_pop.shape == (3,)
    assert updated[0].inference.ate is not None
    assert updated[0].inference.p_value == 0.25
    assert updated[0].inference.ci_lower == -1.0
    assert updated[0].inference.ci_upper == 1.0


def test_run_post_intervention_updates_returns_early_when_no_post():
    cand = _candidate()
    Y_pre = np.array([[1.0, 2.0], [1.5, 2.5]])
    y_pop, updated = setup_mod._run_post_intervention_updates(
        candidate_results=[cand],
        Y_pre=Y_pre,
        post_df=pd.DataFrame(),
        post_idx=np.array([], dtype=int),
        unit_index=IndexSet.from_labels(["A", "B"]),
        unitid="unit",
        time="time",
        outcome="y",
        n_sims=50,
        alpha=0.1,
        seed=123,
    )
    np.testing.assert_allclose(y_pop, Y_pre.mean(axis=1))
    assert updated[0].inference.p_value is None

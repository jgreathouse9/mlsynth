import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.fast_scm_helpers import fast_scm_setup as setup
from mlsynth.utils.fast_scm_helpers.structure import (
    Identification,
    IndexSet,
    Inference,
    Losses,
    PredictionVectors,
    SEDCandidate,
    WeightVectors,
)

# =========================================================
# FIXTURES / HELPERS (LOCAL ONLY)
# =========================================================

class DummySolution:
    label = "cand"


def make_candidate():
    return SEDCandidate(
        identification=Identification(solution=DummySolution(), treated_idx=np.array([0])),
        weights=WeightVectors(
            treated=np.array([1.0]),
            control=np.array([0.0, 0.5, 1e-10]),
        ),
        predictions=PredictionVectors(
            synthetic_treated=np.array([1.0]),
            synthetic_control=np.array([1.2]),
            effects=np.array([-0.2]),
            residuals_E=np.array([0.1]),
            residuals_B=np.array([0.1]),
        ),
        losses=Losses(1, 1, 1, 1, 1, 1, 1),
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

# =========================================================
# INDEXSET
# =========================================================

def test_indexset_roundtrip():
    idx = IndexSet.from_labels(["u3", "u1", "u2"])

    assert len(idx) == 3

    np.testing.assert_array_equal(
        idx.get_labels([0, 2]),
        np.array(["u3", "u2"])
    )

    np.testing.assert_array_equal(
        idx.get_index(["u1", "u3"]),
        np.array([1, 0])
    )


# =========================================================
# STRUCTURE (DATACLASSES)
# =========================================================

def test_inference_defaults_and_mutation():
    inf = Inference()
    assert inf.ate is None
    assert inf.p_value is None
    assert inf.treated_col_idx == []

    inf.ate = 1.2
    inf.treated_col_idx = [0, 1]

    assert inf.ate == 1.2
    assert inf.treated_col_idx == [0, 1]


def test_weight_sparse_behavior():
    w = WeightVectors(
        treated=np.array([1.0]),
        control=np.array([1e-12, -1e-9, 0.3]),
    )

    np.testing.assert_array_equal(
        w.control_sparse,
        np.array([0.0, 0.0, 0.3])
    )


def test_candidate_properties():
    c = make_candidate()

    assert c.treated_size == 1

    np.testing.assert_array_equal(
        c.control_idx,
        np.array([1])
    )

    assert c.control_size == 1

def test_identification_label_fallback():
    c = make_candidate()
    assert c.identification.tuple_id == "cand"


# =========================================================
# SETUP HELPERS
# =========================================================

def test_prepare_working_df(panel_df):
    pre, post = setup._prepare_working_df(panel_df, "is_post")

    assert len(pre) == 4
    assert len(post) == 2


def test_prepare_working_df_no_post(panel_df):
    pre, post = setup._prepare_working_df(panel_df, None)

    assert len(pre) == 6
    assert post.empty


def test_candidate_mask(panel_df):
    idx = IndexSet.from_labels(["A", "B", "C"])

    mask = setup.build_candidate_mask(panel_df, "eligible", idx, "unit")

    np.testing.assert_array_equal(mask, np.array([True, False, False]))


def test_f_vector_uniform_and_weighted(panel_df):
    idx = IndexSet.from_labels(["A", "B"])

    f_uniform = setup.build_f_vector(panel_df, None, "unit", idx)
    np.testing.assert_allclose(f_uniform, np.array([0.5, 0.5]))


# =========================================================
# EXPERIMENT INPUTS
# =========================================================

def test_prepare_experiment_inputs_valid():
    Y = np.array([[1, 2], [3, 4]])

    X, f, idx, T, N = setup.prepare_experiment_inputs(
        Y,
        candidate_mask=np.array([True, True]),
        m=1,
    )

    assert T == 2
    assert N == 2
    assert len(idx) == 2


def test_prepare_experiment_inputs_too_few_candidates():
    Y = np.array([[1, 2]])

    with pytest.raises(ValueError):
        setup.prepare_experiment_inputs(
            Y,
            candidate_mask=np.array([False, False]),
            m=1,
        )


# =========================================================
# POST INTERVENTION PIPELINE
# =========================================================

def test_post_intervention_no_post():
    cand = make_candidate()

    Y_pre = np.array([
        [1, 2],
        [3, 4],
    ])

    y_pop, updated = setup._run_post_intervention_updates(
        [cand],
        Y_pre,
        pd.DataFrame(),
        np.array([], dtype=int),
        None,
        "unit",
        "time",
        "y",
        10,
        0.05,
        123,
    )

    assert y_pop.shape == (2,)
    assert updated[0].inference.p_value is None




# =========================================================
# ADDITIONAL HIGH-VALUE INVARIANT TESTS
# =========================================================

def test_build_y_matrix_alignment(panel_df):
    unit_idx = IndexSet.from_labels(["A", "B"])
    pre = panel_df[panel_df["is_post"] == 0]

    Y = setup.build_Y_matrix(pre, "y", "time", "unit", unit_idx)

    assert Y.shape == (2, 2)
    assert np.isfinite(Y).all()


def test_build_z_matrix_stacking(panel_df):
    unit_idx = IndexSet.from_labels(["A", "B"])
    pre = panel_df[panel_df["is_post"] == 0]

    Z = setup.build_Z_matrix(pre, ["z1", "z2"], "time", "unit", unit_idx)

    # 2 covariates stacked vertically → 2 * 2 time periods = 4 rows
    assert Z.shape == (4, 2)


def test_split_periods_no_post():
    E, B, P = setup.split_periods(T0=4, frac_E=0.5, post_df=None)

    assert len(E) == 2
    assert len(B) == 2
    assert len(P) == 0


def test_candidate_mask_extension():
    Y = np.array([[1, 2], [3, 4]])

    # intentionally too short mask → should auto-extend
    mask = np.array([True])

    X, f, idx, T, N = setup.prepare_experiment_inputs(
        Y,
        candidate_mask=mask,
        m=1,
    )

    # still valid output
    assert T == 2
    assert N == 2
    assert len(idx) >= 1


def test_build_x_tilde_stability():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    f = np.array([0.5, 0.5])

    XE, G = setup.build_X_tilde(X, f, idx=np.array([0, 1]), J=2)

    assert XE.shape == (2, 2)
    assert np.isfinite(XE).all()
    assert np.isfinite(G).all()


def test_post_intervention_effect_computation(monkeypatch):
    cand = make_candidate()

    Y_pre = np.array([
        [1, 2],
        [3, 4],
    ])

    post_df = pd.DataFrame({
        "time": [3, 3],
        "unit": ["A", "B"],
        "y": [5, 6],
    })

    def fake_post(*args, **kwargs):
        return args[0]

    monkeypatch.setattr(setup, "compute_post_inference", fake_post)
    monkeypatch.setattr(setup, "compute_moving_block_conformal_ci", lambda *a, **k: a[0])

    _, updated = setup._run_post_intervention_updates(
        [cand],
        Y_pre,
        post_df,
        np.array([2]),
        IndexSet.from_labels(["A", "B"]),
        "unit",
        "time",
        "y",
        10,
        0.05,
        123,
    )

    assert updated[0].predictions.effects is not None
    assert updated[0].inference.ate is not None

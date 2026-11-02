"""Held-out pre-treatment outcomes score designs without fitting them."""

from copy import deepcopy
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.tests.test_sensitivity import config, study


def validate(specs=(), base=None, **kwargs):
    options = {"validation_windows": [range(4, 8)], **kwargs}
    return study(specs, base=base, **options).fit()


def test_native_training_fit_and_full_refit_agree():
    cfg = config()
    result = validate(base=cfg)
    fold_cfg = deepcopy(cfg)
    fold_cfg["df"] = cfg["df"].query("time < 8").copy()
    fold_cfg["df"]["treated"] = (
        (fold_cfg["df"].unit == "u0") & (fold_cfg["df"].time >= 4)
    ).astype(int)
    native = VanillaSC(fold_cfg).fit()
    assert result.validation_fits["baseline"]["fold_1"].weights == native.weights
    expected = native.time_series.counterfactual_outcome[4:]
    np.testing.assert_allclose(result.validation_paths.synthetic, expected)
    error = result.validation_paths.gap.to_numpy()
    assert result.validation.rmse.iloc[0] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert result.summary.validation_rmse.iloc[0] == result.validation.rmse.iloc[0]
    full = study(base=cfg).fit()
    pd.testing.assert_frame_equal(result.summary[full.summary.columns], full.summary)
    pd.testing.assert_frame_equal(result.paths, full.paths)
    pd.testing.assert_frame_equal(result.weights, full.weights)
    assert result.metadata["treatment_time"] == 8
    assert result.baseline["df"].equals(cfg["df"])
    assert result.validation_registry["baseline"]["fold_1"]["fit_window"] == (0, 3)
    assert result.validation_registry["baseline"]["fold_1"]["inference"] is False


@settings(max_examples=6, deadline=None)
@given(st.floats(min_value=1, max_value=100, allow_nan=False))
def test_heldout_outcomes_do_not_change_training_weights(delta):
    cfg = config()
    original = validate(base=cfg)
    cfg["df"].loc[(cfg["df"].unit == "u0") & cfg["df"].time.between(4, 7), "y"] += delta
    changed = validate(base=cfg)
    pd.testing.assert_frame_equal(
        original.validation_weights, changed.validation_weights
    )
    np.testing.assert_array_equal(
        original.validation_paths.synthetic, changed.validation_paths.synthetic
    )
    assert (
        changed.validation.rmse.iloc[0] > original.validation.rmse.iloc[0] + delta / 2
    )


def test_covariate_windows_and_predictor_tuning_stop_before_validation():
    cfg = config()
    cfg.update(
        backend="malo", covariates=["x", "lag_y"], covariate_windows={"lag_y": (1, 7)}
    )
    cfg["df"]["lag_y"] = cfg["df"].y
    original = validate(base=cfg)
    cfg["df"].loc[cfg["df"].time >= 4, ["x", "lag_y"]] = np.nan
    changed = validate(base=cfg)
    assert (changed.validation.status != "failed").all()
    pd.testing.assert_frame_equal(
        original.validation_weights, changed.validation_weights
    )
    assert changed.validation_registry["baseline"]["fold_1"]["covariate_windows"] == {
        "x": (0, 3),
        "lag_y": (1, 3),
    }
    old_v = original.validation_fits["baseline"]["fold_1"].additional_outputs[
        "solver_diagnostics"
    ]
    new_v = changed.validation_fits["baseline"]["fold_1"].additional_outputs[
        "solver_diagnostics"
    ]
    assert repr(old_v) == repr(new_v)


@settings(max_examples=5, deadline=None)
@given(st.sampled_from([np.nan, -100.0, 1000.0]))
def test_actual_post_outcomes_never_change_validation(value):
    cfg = config()
    original = validate(base=cfg)
    cfg["df"].loc[cfg["df"].time >= 8, "y"] = value
    changed = validate(base=cfg)
    pd.testing.assert_frame_equal(original.validation, changed.validation)
    pd.testing.assert_frame_equal(original.validation_paths, changed.validation_paths)
    pd.testing.assert_frame_equal(
        original.validation_weights, changed.validation_weights
    )


def test_rolling_folds_pool_errors_and_retain_failed_attempts():
    cfg = config()
    cfg["df"].loc[(cfg["df"].unit == "u0") & (cfg["df"].time == 3), "y"] += 3
    result = validate(
        [{"name": "late", "fit_window": (4, 7)}],
        cfg,
        validation_windows=[[3], [4, 5, 6, 7]],
    )
    paths = result.validation_paths.query("spec_id == 'baseline'")
    assert result.summary.validation_rmse.iloc[0] == pytest.approx(
        np.sqrt(np.mean(paths.gap**2))
    )
    assert result.validation.status.tolist() == ["ok", "ok", "failed", "failed"]
    assert np.isnan(result.summary.validation_rmse.iloc[1])
    assert np.isfinite(result.summary.att.iloc[1])
    assert result.validation.query("status == 'failed'").message.str.len().gt(0).all()
    assert result.validation_registry["late"]["fold_1"]["fit_window"] == ()


def test_failed_fold_never_yields_partial_aggregate():
    result = validate(
        [{"name": "late", "fit_window": (4, 7)}], validation_windows=[[3], [6, 7]]
    )
    assert result.validation.query("spec_id == 'late'").status.tolist() == [
        "failed",
        "ok",
    ]
    assert np.isnan(result.summary.validation_rmse.iloc[1])


@pytest.mark.parametrize(
    "windows", [[], [[]], [[0]], [[7, 8]], [[3, 3]], [[99]], [[3, 4], [4, 5]]]
)
def test_invalid_windows_raise_before_any_fit(windows):
    with patch.object(VanillaSC, "fit") as fit:
        with pytest.raises(MlsynthConfigError, match="validation"):
            validate(validation_windows=windows)
        fit.assert_not_called()


def test_invalid_original_designs_cannot_be_repaired_by_clipping():
    result = validate(
        [{"name": "leaky", "fit_window": (0, 10)}, {"name": "identity", "outcome": "x"}]
    )
    assert result.validation.status.tolist() == ["ok", "failed", "failed"]


def test_empty_covariate_window_is_a_fold_failure():
    cfg = config()
    cfg.update(backend="malo", covariates=["x"], covariate_windows={"x": (6, 7)})
    result = validate(base=cfg)
    assert result.validation.status.tolist() == ["failed"]
    assert np.isnan(result.summary.validation_rmse.iloc[0])
    assert result.validation_registry["baseline"]["fold_1"]["covariate_windows"] == {
        "x": ()
    }


def test_declared_threshold_keeps_all_results_and_controls_subset_summary():
    result = validate(validation_max_rmse=1)
    assert result.summary.validation_pass.iloc[0]
    assert result.dispersion(validation_only=True)["attempted"] == 1
    cfg = config()
    cfg["df"].loc[(cfg["df"].unit == "u0") & (cfg["df"].time == 7), "y"] += 10
    result = validate(base=cfg, validation_max_rmse=1)
    assert not result.summary.validation_pass.iloc[0]
    assert result.dispersion()["attempted"] == 1
    assert result.dispersion(validation_only=True)["attempted"] == 0
    with pytest.raises(MlsynthConfigError, match="validation_max_rmse"):
        validate().dispersion(validation_only=True)
    for limit in [-1, np.inf, np.nan]:
        with pytest.raises(MlsynthConfigError, match="validation_max_rmse"):
            validate(validation_max_rmse=limit)
    with pytest.raises(MlsynthConfigError, match="validation"):
        validate(validation_windows=None, validation_max_rmse=1)


def test_plot_contains_valid_points_and_names_failed_designs():
    result = validate([{"name": "late", "fit_window": (6, 7)}], validation_max_rmse=1)
    fig = result.plot_validation()
    assert len(fig.axes[0].collections[0].get_offsets()) == 1
    assert any("late" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)
    with pytest.raises(MlsynthConfigError, match="validation"):
        study().fit().plot_validation()


def test_validation_plot_without_threshold_and_with_all_failures():
    for result in (validate(), validate(excluded_donors=["u1", "u2", "u3"])):
        fig = result.plot_validation()
        assert len(fig.axes[0].lines) == 1
        if result.validation.status.iloc[0] == "failed":
            assert not fig.axes[0].collections
            assert any("baseline" in text.get_text() for text in fig.axes[0].texts)
        else:
            assert len(fig.axes[0].collections) == 1
        plt.close(fig)


def test_fold_warnings_preserve_finite_scores_and_predictor_diagnostics():
    original = VanillaSC.fit

    def warned(model):
        result = original(model)
        if model.config.df.time.max() < 8:
            result = result.model_copy(
                update={
                    "additional_outputs": {
                        **result.additional_outputs,
                        "solver_warnings": [
                            {
                                "warning_type": "RuntimeWarning",
                                "message": "not converged",
                            }
                        ],
                    }
                }
            )
        return result

    with patch.object(VanillaSC, "fit", warned):
        result = validate()
    assert result.summary.status.tolist() == ["ok"]
    assert result.validation.status.tolist() == ["warning"]
    assert result.summary.validation_status.tolist() == ["warning"]
    assert result.validation_warnings.message.tolist() == ["not converged"]
    assert np.isfinite(result.summary.validation_rmse.iloc[0])
    fig = result.plot_validation()
    assert len(fig.axes[0].collections) == 1
    plt.close(fig)


def test_fitting_exclusions_are_intersected_with_training_time():
    cfg = config()
    cfg.update(fit_window=(None, 6), fit_exclude=[1, 5])
    result = validate(base=cfg)
    assert result.validation.status.tolist() == ["ok"]
    effective = result.validation_registry["baseline"]["fold_1"]
    assert effective["fit_exclude"] == [1]
    assert effective["fit_window"] == (0, 3)
    native_cfg = deepcopy(cfg)
    native_cfg.update(fit_window=(0, 3), fit_exclude=[1])
    native_cfg["df"] = cfg["df"].query("time < 8").copy()
    native_cfg["df"]["treated"] = (
        (native_cfg["df"].unit == "u0") & (native_cfg["df"].time >= 4)
    ).astype(int)
    native = VanillaSC(native_cfg).fit()
    assert result.validation_fits["baseline"]["fold_1"].weights == native.weights


@settings(max_examples=8, deadline=None)
@given(st.lists(st.integers(2, 7), min_size=1, unique=True))
def test_all_declared_scoring_periods_are_used_once(periods):
    result = validate(validation_windows=[periods])
    assert result.validation_paths.time.tolist() == sorted(periods)
    assert result.validation.n_validation.iloc[0] == len(periods)
    assert result.summary.validation_rmse.iloc[0] == pytest.approx(
        np.sqrt(np.mean(result.validation_paths.gap**2))
    )
    assert result.validation_registry["baseline"]["fold_1"]["fit_window"][1] < min(
        periods
    )


@settings(max_examples=6, deadline=None)
@given(st.permutations(["u1", "u2", "u3"]), st.booleans())
def test_donor_and_window_order_do_not_change_validation(donors, reverse):
    windows = [[3, 4], [6, 7]]
    reference = validate(validation_windows=windows)
    cfg = config()
    cfg["donors"] = donors
    result = validate(
        base=cfg, validation_windows=windows[::-1] if reverse else windows
    )
    pd.testing.assert_frame_equal(
        result.validation_weights, reference.validation_weights
    )
    pd.testing.assert_frame_equal(result.validation_paths, reference.validation_paths)

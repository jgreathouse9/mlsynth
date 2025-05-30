import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import os
import matplotlib.pyplot as plt # For mocking and type hints

from mlsynth.utils.resultutils import plot_estimates, effects, SDID_plot
from mlsynth.exceptions import MlsynthDataError, MlsynthPlottingError

# --- Fixtures for plot_estimates ---

@pytest.fixture
def plot_estimates_sample_data() -> dict:
    """Provides sample data for plot_estimates tests."""
    T = 20
    pre_periods = 10
    y = np.random.rand(T)
    cf1 = y * 0.8 + np.random.rand(T) * 0.1
    cf2 = y * 0.7 + np.random.rand(T) * 0.15
    
    time_index = pd.date_range(start="2000-01-01", periods=T, freq="MS")
    processed_data_dict_mock = {
        "Ywide": pd.DataFrame(index=time_index),
        "pre_periods": pre_periods
    }
    uncertainty_intervals_array_mock = np.column_stack((cf1 * 0.9, cf1 * 1.1))

    return {
        "processed_data_dict": processed_data_dict_mock,
        "time_axis_label": "Month",
        "unit_identifier_column_name": "TestUnit",
        "outcome_variable_label": "Sales",
        "treatment_name_label": "Ad Campaign",
        "treated_unit_name": "Store A",
        "observed_outcome_series": y,
        "counterfactual_series_list": [cf1, cf2],
        "estimation_method_name": "TestSCM",
        "treated_series_color": "black",
        "counterfactual_series_colors": ["blue", "green"],
        "counterfactual_names": ["CF Model 1", "CF Model 2"],
        "uncertainty_intervals_array": uncertainty_intervals_array_mock
    }

# --- Tests for plot_estimates ---

@patch("mlsynth.utils.resultutils.plt.show")
def test_plot_estimates_smoke(mock_plt_show, plot_estimates_sample_data):
    """Smoke test for plot_estimates function."""
    data = plot_estimates_sample_data
    plot_estimates(
        processed_data_dict=data["processed_data_dict"],
        time_axis_label=data["time_axis_label"],
        unit_identifier_column_name=data["unit_identifier_column_name"],
        outcome_variable_label=data["outcome_variable_label"],
        treatment_name_label=data["treatment_name_label"],
        treated_unit_name=data["treated_unit_name"],
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_series_list=data["counterfactual_series_list"],
        estimation_method_name=data["estimation_method_name"],
        treated_series_color=data["treated_series_color"],
        counterfactual_series_colors=data["counterfactual_series_colors"],
        counterfactual_names=data["counterfactual_names"],
        save_plot_config=False,
        uncertainty_intervals_array=None
    )
    mock_plt_show.assert_called_once()

@patch("mlsynth.utils.resultutils.plt.show")
@patch("mlsynth.utils.resultutils.plt.savefig")
def test_plot_estimates_save_true(mock_savefig, mock_plt_show, plot_estimates_sample_data):
    """Test plot_estimates with save_plot_config=True."""
    data = plot_estimates_sample_data
    plot_estimates(
        processed_data_dict=data["processed_data_dict"],
        time_axis_label=data["time_axis_label"],
        unit_identifier_column_name=data["unit_identifier_column_name"],
        outcome_variable_label=data["outcome_variable_label"],
        treatment_name_label=data["treatment_name_label"],
        treated_unit_name=data["treated_unit_name"],
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_series_list=[data["counterfactual_series_list"][0]],
        estimation_method_name=data["estimation_method_name"],
        treated_series_color=data["treated_series_color"],
        counterfactual_series_colors=[data["counterfactual_series_colors"][0]],
        save_plot_config=True
    )
    expected_filename = os.path.join(os.getcwd(), f"{data['estimation_method_name']}_{data['treated_unit_name']}.png")
    mock_savefig.assert_called_once_with(expected_filename)
    mock_plt_show.assert_not_called() # Should not be called when save_plot_config is True (boolean)

@patch("mlsynth.utils.resultutils.plt.show")
@patch("mlsynth.utils.resultutils.plt.savefig")
def test_plot_estimates_save_dict(mock_savefig, mock_plt_show, plot_estimates_sample_data, tmp_path):
    """Test plot_estimates with save_plot_config as a dictionary."""
    data = plot_estimates_sample_data
    # Use tmp_path for test directory
    test_plot_dir = tmp_path / "test_plots"
    save_config = {"filename": "custom_plot", "extension": "pdf", "directory": str(test_plot_dir), "display": False}
    
    plot_estimates(
        processed_data_dict=data["processed_data_dict"],
        time_axis_label=data["time_axis_label"],
        unit_identifier_column_name=data["unit_identifier_column_name"],
        outcome_variable_label=data["outcome_variable_label"],
        treatment_name_label=data["treatment_name_label"],
        treated_unit_name=data["treated_unit_name"],
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_series_list=[data["counterfactual_series_list"][0]],
        estimation_method_name=data["estimation_method_name"],
        treated_series_color=data["treated_series_color"],
        counterfactual_series_colors=[data["counterfactual_series_colors"][0]],
        save_plot_config=save_config
    )
    expected_filepath = test_plot_dir / f"{save_config['filename']}.{save_config['extension']}"
    mock_savefig.assert_called_once_with(str(expected_filepath))
    mock_plt_show.assert_not_called()


@patch("mlsynth.utils.resultutils.plt.show")
def test_plot_estimates_with_uncertainty_intervals(mock_plt_show, plot_estimates_sample_data):
    """Test plot_estimates with uncertainty intervals."""
    data = plot_estimates_sample_data
    plot_estimates(
        processed_data_dict=data["processed_data_dict"],
        time_axis_label=data["time_axis_label"],
        unit_identifier_column_name=data["unit_identifier_column_name"],
        outcome_variable_label=data["outcome_variable_label"],
        treatment_name_label=data["treatment_name_label"],
        treated_unit_name=data["treated_unit_name"],
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_series_list=data["counterfactual_series_list"],
        estimation_method_name=data["estimation_method_name"],
        treated_series_color=data["treated_series_color"],
        counterfactual_series_colors=data["counterfactual_series_colors"],
        uncertainty_intervals_array=data["uncertainty_intervals_array"]
    )
    mock_plt_show.assert_called_once()

def test_plot_estimates_missing_processed_data_keys(plot_estimates_sample_data):
    data = plot_estimates_sample_data.copy()
    del data["processed_data_dict"]["Ywide"]
    with pytest.raises(MlsynthDataError, match="processed_data_dict is missing required keys"):
        plot_estimates(**{k:v for k,v in data.items() if k != "uncertainty_intervals_array"})

def test_plot_estimates_invalid_uncvectors_shape(plot_estimates_sample_data):
    data = plot_estimates_sample_data.copy()
    data["uncertainty_intervals_array"] = np.random.rand(len(data["observed_outcome_series"]), 3) # 3 columns instead of 2
    with pytest.raises(MlsynthDataError, match="uncertainty_intervals_array must be a 2D NumPy array with 2 columns"):
        plot_estimates(**data)

    data["uncertainty_intervals_array"] = np.random.rand(len(data["observed_outcome_series"])-1, 2) # Wrong number of rows
    with pytest.raises(MlsynthDataError, match="uncertainty_intervals_array must have the same number of rows"):
        plot_estimates(**data)


@patch("mlsynth.utils.resultutils.plt.show")
@patch("mlsynth.utils.resultutils.plt.savefig", side_effect=OSError("Disk full"))
def test_plot_estimates_savefig_oserror(mock_savefig, mock_plt_show, plot_estimates_sample_data, tmp_path):
    data = plot_estimates_sample_data.copy()
    test_plot_dir = tmp_path / "test_plots_fail"
    save_config = {"filename": "fail_plot", "extension": "png", "directory": str(test_plot_dir), "display": True}
    data["save_plot_config"] = save_config
    
    with pytest.raises(MlsynthPlottingError, match="Failed to save plot"):
        plot_estimates(**data)
    mock_savefig.assert_called_once()
    # mock_plt_show should still be called if display is True, even if save fails
    # However, the current implementation in resultutils.py calls plt.close() after savefig,
    # so plt.show() might not be reached if savefig raises an error before it.
    # The test confirms the MlsynthPlottingError is raised.


# --- Fixtures for effects.calculate ---

@pytest.fixture
def effects_calculate_sample_data() -> dict:
    """Provides sample data for effects.calculate tests."""
    T = 20
    t1 = 10 # Pre-treatment periods
    num_post = T - t1
    y = np.arange(T, dtype=float) * 2.0
    y_counterfactual = y * 0.9 + np.random.normal(0, 0.5, T)
    y_counterfactual[t1:] = y[t1:]*0.7 # Make post-treatment different
    return {
        "observed_outcome_series": y,
        "counterfactual_outcome_series": y_counterfactual,
        "num_pre_treatment_periods": t1,
        "num_actual_post_periods": num_post
    }

# --- Tests for effects.calculate ---

def test_effects_calculate_smoke(effects_calculate_sample_data):
    """Smoke test for effects.calculate method."""
    data = effects_calculate_sample_data
    effects_dict, fit_dict, vector_dict = effects.calculate(
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_outcome_series=data["counterfactual_outcome_series"],
        num_pre_treatment_periods=data["num_pre_treatment_periods"],
        num_actual_post_periods=data["num_actual_post_periods"]
    )
    
    assert isinstance(effects_dict, dict)
    assert isinstance(fit_dict, dict)
    assert isinstance(vector_dict, dict)

    assert "ATT" in effects_dict and isinstance(effects_dict["ATT"], (float, np.floating))
    assert "T0 RMSE" in fit_dict and isinstance(fit_dict["T0 RMSE"], (float, np.floating))
    assert "R-Squared" in fit_dict 
    assert "Observed Unit" in vector_dict and isinstance(vector_dict["Observed Unit"], np.ndarray)
    assert "Counterfactual" in vector_dict and isinstance(vector_dict["Counterfactual"], np.ndarray)
    assert "Gap" in vector_dict and isinstance(vector_dict["Gap"], np.ndarray)
    
    assert vector_dict["Observed Unit"].shape == (len(data["observed_outcome_series"]), 1)
    assert vector_dict["Counterfactual"].shape == (len(data["counterfactual_outcome_series"]), 1)
    assert vector_dict["Gap"].shape == (len(data["observed_outcome_series"]), 2)


def test_effects_calculate_no_post_periods(effects_calculate_sample_data):
    data = effects_calculate_sample_data
    effects_dict, fit_dict, _ = effects.calculate(
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_outcome_series=data["counterfactual_outcome_series"],
        num_pre_treatment_periods=data["num_pre_treatment_periods"],
        num_actual_post_periods=0 
    )
    assert np.isnan(effects_dict["ATT"])
    assert np.isnan(effects_dict["Percent ATT"])
    assert np.isnan(effects_dict["SATT"])
    assert np.isnan(effects_dict["TTE"])
    assert fit_dict["Post-Periods"] == 0
    assert np.isnan(fit_dict["T1 RMSE"])


def test_effects_calculate_no_pre_periods(effects_calculate_sample_data):
    data = effects_calculate_sample_data
    effects_dict, fit_dict, _ = effects.calculate(
        observed_outcome_series=data["observed_outcome_series"],
        counterfactual_outcome_series=data["counterfactual_outcome_series"],
        num_pre_treatment_periods=0,
        num_actual_post_periods=data["num_actual_post_periods"]
    )
    assert np.isnan(fit_dict["T0 RMSE"])
    assert np.isnan(fit_dict["R-Squared"])
    if data["num_actual_post_periods"] > 0:
        assert not np.isnan(effects_dict["ATT"])
    else:
        assert np.isnan(effects_dict["ATT"])


# --- Fixtures for SDID_plot ---

@pytest.fixture
def sdid_plot_sample_data() -> dict:
    """Provides sample data for SDID_plot tests."""
    results = {
        "pooled_estimates": {
            "-2": {"tau": -0.1, "ci": (-0.3, 0.1)},
            "-1": {"tau": 0.05, "ci": (-0.15, 0.25)},
            "0": {"tau": 0.5, "ci": (0.2, 0.8)},
            "1": {"tau": 0.6, "ci": (0.3, 0.9)},
        }
    }
    return {"sdid_results_dict": results}

# --- Tests for SDID_plot ---

@patch("mlsynth.utils.resultutils.plt.show")
def test_SDID_plot_smoke(mock_plt_show, sdid_plot_sample_data):
    """Smoke test for SDID_plot function."""
    data = sdid_plot_sample_data
    SDID_plot(sdid_results_dict=data["sdid_results_dict"])
    mock_plt_show.assert_called_once()

def test_SDID_plot_missing_pooled_estimates():
    with pytest.raises(MlsynthDataError, match="sdid_results_dict is missing the required 'pooled_estimates' key."):
        SDID_plot(sdid_results_dict={})

def test_SDID_plot_pooled_estimates_not_dict():
    with pytest.raises(MlsynthDataError, match="'pooled_estimates' must be a dictionary."):
        SDID_plot(sdid_results_dict={"pooled_estimates": "not_a_dict"})

def test_SDID_plot_event_time_key_not_int():
    data = {"pooled_estimates": {"event_A": {"tau": 0.1, "ci": [0, 0.2]}}}
    with pytest.raises(MlsynthDataError, match="Event time key 'event_A' cannot be converted to an integer."):
        SDID_plot(sdid_results_dict=data)

def test_SDID_plot_event_time_value_not_dict():
    data = {"pooled_estimates": {"-1": "not_a_dict"}}
    with pytest.raises(MlsynthDataError, match="Value for event time key '-1' must be a dictionary."):
        SDID_plot(sdid_results_dict=data)

def test_SDID_plot_missing_tau_or_ci():
    data_no_tau = {"pooled_estimates": {"-1": {"ci": [0, 0.2]}}}
    with pytest.raises(MlsynthDataError, match="Estimate data for event time '-1' is missing required keys \\('tau' or 'ci'\\)."):
        SDID_plot(sdid_results_dict=data_no_tau)
    
    data_no_ci = {"pooled_estimates": {"-1": {"tau": 0.1}}}
    with pytest.raises(MlsynthDataError, match="Estimate data for event time '-1' is missing required keys \\('tau' or 'ci'\\)."):
        SDID_plot(sdid_results_dict=data_no_ci)

def test_SDID_plot_malformed_ci():
    data_ci_not_list = {"pooled_estimates": {"-1": {"tau": 0.1, "ci": "not_a_list"}}}
    with pytest.raises(MlsynthDataError, match="Confidence interval 'ci' for event time '-1' must be a list or tuple of two numbers."):
        SDID_plot(sdid_results_dict=data_ci_not_list)

    data_ci_wrong_len = {"pooled_estimates": {"-1": {"tau": 0.1, "ci": [0.1]}}}
    with pytest.raises(MlsynthDataError, match="Confidence interval 'ci' for event time '-1' must be a list or tuple of two numbers."):
        SDID_plot(sdid_results_dict=data_ci_wrong_len)

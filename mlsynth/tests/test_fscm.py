import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor  # Added import
from pydantic import ValidationError

from mlsynth import FSCM
from mlsynth.config_models import FSCMConfig, BaseEstimatorResults
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthConfigError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from mlsynth.utils.estutils import fsSCM

from mlsynth.utils.selector_helpers import (
_fscm_evaluate_candidates, _fscm_pick_best_candidate, _fscm_extract_weights,
_fscm_inner
)


# Base configuration dictionary containing all parameters used in tests.
# We will extract Pydantic-valid fields from this for FSCMConfig instantiation.
FSCM_FULL_TEST_CONFIG_BASE: Dict[str, Any] = {
    "outcome": "Y",
    "treat": "treated_indicator_programmatic",
    "unitid": "unit_id",
    "time": "time_id",
    "counterfactual_color": ["blue"],
    "treated_color": "green",
    "display_graphs": False,
    "save": False,
    # FSCM-specific parameters (not in BaseEstimatorConfig/current FSCMConfig)
    "lambda_": 0.1,  # Note: Pydantic model might use 'lambda_val'
    "omega": "balanced",  # Note: Pydantic model might use 'omega_val'
    "cv_lambda": False,
    "cv_omega": False,
    "min_lambda": 1e-5,
    "max_lambda": 1e5,
    "grid_lambda": 10,
    "min_omega": 0,
    "max_omega": 1,
    "grid_omega": 10,
    "cv_folds": 5,
    "model_type": "conformal",  # This seems to be for conformal prediction, maybe not FSCM core
    "level": 0.95,  # For conformal prediction
    "seed": 12345,
    "parallel": False,
    "verbose": False,
}

# Fields that are part of BaseEstimatorConfig (and thus FSCMConfig)
FSCM_PYDANTIC_MODEL_FIELDS = [
    "df", "outcome", "treat", "unitid", "time",
    "display_graphs", "save", "counterfactual_color", "treated_color"
]


def _get_pydantic_config_dict(full_config: Dict[str, Any], df_fixture: pd.DataFrame) -> Dict[str, Any]:
    """Helper to extract Pydantic-valid fields and add the DataFrame."""
    pydantic_dict = {
        k: v for k, v in full_config.items() if k in FSCM_PYDANTIC_MODEL_FIELDS
    }
    pydantic_dict["df"] = df_fixture
    return pydantic_dict


@pytest.fixture
def basic_panel_data_with_treatment():
    """Provides a very basic panel dataset with a treatment column for smoke testing."""
    data_dict = {
        'unit_id': ['1', '1', '1', '1', '2', '2', '2', '2', '3', '3', '3', '3'],  # Changed to strings
        'time_id': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        'Y': [10, 11, 15, 16, 9, 10, 11, 12, 12, 13, 14, 15],
        'X1': [5, 6, 7, 8, 4, 5, 6, 7, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data_dict)

    # Add treatment column as expected by FSCM (via config['treat'])
    # Unit 1 is treated starting from time_id = 3
    treatment_col_name = FSCM_FULL_TEST_CONFIG_BASE["treat"]
    df[treatment_col_name] = 0
    df.loc[(df['unit_id'] == '1') & (df['time_id'] >= 3), treatment_col_name] = 1  # Corrected to string comparison
    return df


def test_fscm_creation(basic_panel_data_with_treatment: pd.DataFrame):
    """Test that the FSCM estimator can be instantiated."""
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)

    try:
        config_obj = FSCMConfig(**pydantic_dict)
        estimator = FSCM(config=config_obj)  # Assumes FSCM.__init__ now takes FSCMConfig
        assert estimator is not None, "FSCM estimator should be created."
        assert estimator.outcome == "Y", "Outcome attribute should be set from config."
        assert estimator.treat == FSCM_FULL_TEST_CONFIG_BASE["treat"]
        assert not estimator.display_graphs, "display_graphs should be False for tests."
    except Exception as e:
        pytest.fail(f"FSCM instantiation failed: {e}")


def test_fscm_fit_smoke_modern(basic_panel_data_with_treatment: pd.DataFrame):
    """Modern smoke test for FSCM.fit() with minimal mocks for fsSCM and result builder."""
    from mlsynth.estimators.fscm import FSCM
    from mlsynth.config_models import FSCMConfig, BaseEstimatorResults
    from unittest.mock import patch, MagicMock

    # Prepare config and estimator
    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    # Patch fsSCM and _build_fscm_results to control outputs
    with patch('mlsynth.utils.estutils.fsSCM') as mock_fsSCM, \
         patch('mlsynth.utils.resultutils._build_fscm_results') as mock_build_results:

        # Minimal plausible return from fsSCM
        mock_fsSCM.return_value = {
            'Convex SCM': {'weight_dict': {'d2': 0.5, 'd3': 0.5}},
            'Forward SCM': {'weights': {'d2': 0.6, 'd3': 0.4}},
            'Forward Augmented SCM': {'weight_dict': {'d2': 0.7, 'd3': 0.3}}
        }

        # Mock result builder returns a BaseEstimatorResults with minimal sub-methods
        mocked_results = BaseEstimatorResults(
            sub_method_results={
                'Convex SCM': MagicMock(),
                'Forward SCM': MagicMock(),
                'Forward Augmented SCM': MagicMock()
            },
            additional_outputs={'Ywide': 'mocked', 'donor_matrix': 'mocked'},
            raw_results={'dummy': 'results'}
        )
        mock_build_results.return_value = mocked_results

        # Run fit
        output = estimator.fit()

        # Check top-level output
        assert hasattr(output, 'results')
        assert hasattr(output, 'prepped_data')
        assert isinstance(output.results, dict)
        assert isinstance(output.prepped_data, dict)

        # Check that each sub-method is a BaseEstimatorResults
        for method in ['Convex SCM', 'Forward SCM', 'Forward Augmented SCM']:
            sub_res = output.results[method]
            assert isinstance(sub_res, BaseEstimatorResults)


def test_fscm_init_missing_dataframe_leads_to_config_error():
    """Test FSCMConfig instantiation fails if DataFrame 'df' is missing."""
    config_dict_no_df = FSCM_FULL_TEST_CONFIG_BASE.copy()
    # "df" is deliberately not added via _get_pydantic_config_dict or explicitly

    pydantic_dict_attempt = {
        k: v for k, v in config_dict_no_df.items() if k in FSCM_PYDANTIC_MODEL_FIELDS and k != "df"
    }
    with pytest.raises(ValidationError):  # df is required by BaseEstimatorConfig
        FSCMConfig(**pydantic_dict_attempt)


@patch('mlsynth.estimators.fscm.fsSCM')
def test_fscm_fit_fscm_method_raises_error(mock_fscm_method, basic_panel_data_with_treatment: pd.DataFrame):
    from mlsynth.estimators.fscm import FSCM
    from mlsynth.config_models import FSCMConfig
    from mlsynth.exceptions import MlsynthEstimationError

    pydantic_dict = _get_pydantic_config_dict(FSCM_FULL_TEST_CONFIG_BASE, basic_panel_data_with_treatment)
    config_obj = FSCMConfig(**pydantic_dict)
    estimator = FSCM(config=config_obj)

    mock_fscm_method.side_effect = MlsynthEstimationError("Internal fsSCM failure")

    with pytest.raises(MlsynthEstimationError, match="Internal fsSCM failure"):
        estimator.fit()

# Mock donor solution object
class MockSCoptSolution:
    def __init__(self, weights, value=1.0):
        self.solution = MagicMock()
        self.solution.primal_vars = {'w': np.array(weights)}
        self.value = value


# --- Test _extract_weights ---
def test_extract_weights():
    mock_solution = MockSCoptSolution(weights=[0.2, 0.8])
    weights = _fscm_extract_weights(mock_solution)
    assert np.allclose(weights, [0.2, 0.8])


# --- Test fscm_inner with mocked Opt.SCopt ---
@patch('mlsynth.utils.estutils.Opt.SCopt')
def test_fscm_inner(mock_scopt):
    mock_scopt.return_value = MockSCoptSolution(weights=[0.3, 0.7], value=2.0)

    y = np.array([10, 12])
    Y = np.array([[10, 11], [12, 13]])
    T0 = 2
    candidate_indices = [0, 1]
    donor_names = ['A', 'B']

    mse, weights = _fscm_inner(y, Y, T0, candidate_indices, donor_names)
    assert np.isclose(mse, (2.0 ** 2) / T0)
    assert np.allclose(weights, [0.3, 0.7])




@patch('mlsynth.utils.selector_helpers._fscm_inner')
def test_evaluate_candidates(mock_inner):
    from unittest.mock import call
    """
    Test that _fscm_evaluate_candidates correctly loops over remaining donors
    and collects MSE + weights from _fscm_inner.
    """

    # Two calls: remaining = [1, 2]
    mock_inner.side_effect = [
        (0.5, np.array([1.0, 0.0])),   # candidate_subset = [0, 1]
        (0.2, np.array([0.0, 1.0]))    # candidate_subset = [0, 2]
    ]

    # y has T0=2 periods
    y = np.array([10, 12])

    # Y has THREE donors (columns 0, 1, 2)
    Y = np.array([
        [10, 11, 12],
        [12, 13, 14]
    ])

    T0 = 2
    selected = [0]
    remaining = [1, 2]
    donor_names = ['A', 'B', 'C']

    from mlsynth.utils.selector_helpers import _fscm_evaluate_candidates

    mse_list, weights_list = _fscm_evaluate_candidates(
        y, Y, T0, selected, remaining, donor_names
    )

    # correct outputs from side_effect
    assert mse_list == [0.5, 0.2]
    assert len(weights_list) == 2
    assert np.allclose(weights_list[0], np.array([1.0, 0.0]))
    assert np.allclose(weights_list[1], np.array([0.0, 1.0]))

    # ensure _fscm_inner was called with correct subsets
    expected_calls = [
        call(y, Y, T0, [0, 1], donor_names),
        call(y, Y, T0, [0, 2], donor_names),
    ]

    mock_inner.assert_has_calls(expected_calls, any_order=False)





# --- Test pick_best_candidate ---
def test_pick_best_candidate():
    mse_list = [0.5, 0.2, 0.8]
    weights_list = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    remaining = [0, 1, 2]
    selected = []
    donor_names = ['A', 'B', 'C']

    best_candidate, best_mse, best_weights = _fscm_pick_best_candidate(mse_list, weights_list, remaining, selected,
                                                                 donor_names)

    assert best_candidate == 1
    assert best_mse == 0.2
    assert np.allclose(best_weights, [0.0, 1.0])
    assert selected == [1]
    assert remaining == [0, 2]


import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mlsynth.utils.estutils import tune_affine_lambda


def make_fake_solution(weights, status="optimal"):
    sol = MagicMock()
    sol.solution.status = status
    sol.solution.primal_vars = {"w": np.array(weights)}
    return sol


@patch("mlsynth.utils.estutils.Opt.SCopt")
def test_tune_affine_lambda_selects_best_lambda(mock_SCopt):
    w_true = np.array([0.7, 0.3])
    Y = np.array([[1, 2],
                  [2, 0],
                  [3, 1],
                  [4, 3]], dtype=float)
    y = Y @ w_true

    donor_names = ["A", "B"]
    w_convex = np.array([0.5, 0.5])

    lambdas = [0.1, 1.0, 10.0]
    fake_weights = {
        0.1: [0.69, 0.31],   # best
        1.0: [0.6, 0.4],
        10.0: [1.0, 0.0]
    }

    def side_effect(*args, **kwargs):
        lam = kwargs["lambda_penalty"]
        return make_fake_solution(fake_weights[lam])

    mock_SCopt.side_effect = side_effect

    best_lambda, results = tune_affine_lambda(
        y=y,
        Y=Y,
        T0=4,
        donor_names=donor_names,
        w_convex=w_convex,
        lambda_grid=lambdas,
        train_frac=0.5
    )

    assert best_lambda == 0.1
    assert set(results.keys()) == set(lambdas)
    assert results[best_lambda] == min(results.values())


@patch("mlsynth.utils.estutils.Opt.SCopt")
def test_tune_affine_lambda_raises_if_all_infeasible(mock_SCopt):
    mock_SCopt.return_value = make_fake_solution(
        weights=[0.0, 0.0], status="infeasible"
    )

    y = np.array([1, 2, 3])
    Y = np.random.randn(3, 2)

    with pytest.raises(RuntimeError):
        tune_affine_lambda(
            y=y,
            Y=Y,
            T0=3,
            donor_names=["A", "B"],
            w_convex=np.array([0.5, 0.5]),
            lambda_grid=[0.1, 1.0]
        )

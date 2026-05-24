from mlsynth.exceptions import MlsynthEstimationError, MlsynthDataError, MlsynthConfigError
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict
from unittest.mock import patch
from pydantic import ValidationError

from mlsynth.estimators.proximal import PROXIMAL
from mlsynth.config_models import PROXIMALConfig
from mlsynth.utils.proximal_helpers.structures import (
    PROXIMALResults,
    ProximalMethodFit,
)


@pytest.fixture
def sample_proximal_data(request: Any) -> pd.DataFrame:
    """Sample panel for PROXIMAL tests.

    Parameterize with ``{"with_surrogates": True}`` to add surrogate proxy
    columns.
    """
    include_surrogate_data = hasattr(request, "param") and request.param.get("with_surrogates", False)

    n_total_units = 5  # Unit 1 treated, 2/3 donors, 4/5 potential surrogates
    n_periods = 10
    treatment_start_period = 7  # 6 pre-periods, 4 post-periods

    units = np.repeat(np.arange(1, n_total_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_total_units)

    np.random.seed(789)
    outcomes = []
    donor_proxy_data = []
    surrogate_specific_proxy_data = []

    for i in range(n_total_units):
        base_trend = np.linspace(start=10 + i * 2, stop=25 + i * 2, num=n_periods)
        noise = np.random.normal(0, 0.7, n_periods)
        outcomes.extend(base_trend + noise)
        donor_proxy_data.extend(np.random.rand(n_periods) * 5 + base_trend * 0.3)
        if include_surrogate_data:
            surrogate_specific_proxy_data.extend(np.random.rand(n_periods) * 3 + base_trend * 0.2)

    data = {
        "UnitIdentifier": units,
        "TimeIdx": times,
        "OutcomeValue": outcomes,
        "IsTreated": np.zeros(n_total_units * n_periods, dtype=int),
        "DonorProxyVar1": donor_proxy_data,
    }
    if include_surrogate_data:
        data["SurrogateSpecificProxyVar1"] = surrogate_specific_proxy_data

    df = pd.DataFrame(data)
    df.loc[(df["UnitIdentifier"] == 1) & (df["TimeIdx"] >= treatment_start_period), "IsTreated"] = 1
    return df


# --- Construction ---

def test_proximal_creation(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data,
        "outcome": "OutcomeValue",
        "treat": "IsTreated",
        "unitid": "UnitIdentifier",
        "time": "TimeIdx",
        "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]},
        "surrogates": [],
        "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    assert estimator is not None
    assert estimator.outcome == "OutcomeValue"
    assert estimator.donors == [2, 3]
    assert estimator.vars == {"donorproxies": ["DonorProxyVar1"]}
    assert not estimator.display_graphs


def test_proximal_creation_from_dict(sample_proximal_data: pd.DataFrame) -> None:
    """A dict config is accepted and converted internally."""
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data,
        "outcome": "OutcomeValue",
        "treat": "IsTreated",
        "unitid": "UnitIdentifier",
        "time": "TimeIdx",
        "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]},
        "surrogates": [],
        "display_graphs": False,
    }
    estimator = PROXIMAL(config_dict)
    assert isinstance(estimator.config, PROXIMALConfig)


# --- Helpers ---

def _validate_method_fit(fit: ProximalMethodFit, name: str, T: int) -> None:
    assert isinstance(fit, ProximalMethodFit)
    assert fit.name == name
    assert isinstance(fit.counterfactual, np.ndarray) and fit.counterfactual.shape == (T,)
    assert isinstance(fit.gap, np.ndarray) and fit.gap.shape == (T,)
    assert isinstance(fit.time_varying_effect, np.ndarray) and fit.time_varying_effect.shape == (T,)
    assert isinstance(fit.att, float) and np.isfinite(fit.att)
    assert fit.att_se is None or isinstance(fit.att_se, float)
    assert isinstance(fit.pre_rmse, float)
    assert isinstance(fit.post_rmse, float)
    assert isinstance(fit.alpha_weights, np.ndarray)
    assert isinstance(fit.donor_weights, dict)
    lo, hi = fit.ci
    assert lo <= fit.att <= hi or not np.isfinite(lo)


# --- Smoke: PI only ---

def test_proximal_fit_smoke_pi_only(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data,
        "outcome": "OutcomeValue",
        "treat": "IsTreated",
        "unitid": "UnitIdentifier",
        "time": "TimeIdx",
        "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]},
        "surrogates": [],
        "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    results = estimator.fit()

    assert isinstance(results, PROXIMALResults)
    assert results.mode == "proximal"
    assert results.pi is not None
    assert results.pis is None and results.pipost is None
    assert list(results.methods.keys()) == ["PI"]

    n_periods = sample_proximal_data["TimeIdx"].nunique()
    _validate_method_fit(results.pi, "PI", n_periods)

    # Convenience accessors forward to PI.
    assert results.att == results.pi.att
    assert results.att_se == results.pi.att_se
    np.testing.assert_array_equal(results.counterfactual, results.pi.counterfactual)
    assert results.donor_weights == results.pi.donor_weights
    assert set(results.donor_weights.keys()) == {2, 3}
    assert results.att_by_method() == {"PI": results.pi.att}


# --- Smoke: with surrogates ---

@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_fit_smoke_with_surrogates(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data,
        "outcome": "OutcomeValue",
        "treat": "IsTreated",
        "unitid": "UnitIdentifier",
        "time": "TimeIdx",
        "donors": [2, 3],
        "surrogates": [4, 5],
        "vars": {
            "donorproxies": ["DonorProxyVar1"],
            "surrogatevars": ["SurrogateSpecificProxyVar1"],
        },
        "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    results = estimator.fit()

    assert isinstance(results, PROXIMALResults)
    assert results.pi is not None and results.pis is not None and results.pipost is not None
    assert list(results.methods.keys()) == ["PI", "PIS", "PIPost"]

    n_periods = sample_proximal_data["TimeIdx"].nunique()
    for name, fit in results.methods.items():
        _validate_method_fit(fit, name, n_periods)

    by_method = results.att_by_method()
    assert set(by_method) == {"PI", "PIS", "PIPost"}
    ses = results.se_by_method()
    assert set(ses) == {"PI", "PIS", "PIPost"}
    cis = results.ci_by_method()
    assert set(cis) == {"PI", "PIS", "PIPost"}


# --- Config validation ---

def test_proximal_empty_donors_rejected(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [],
        "vars": {"donorproxies": ["DonorProxyVar1"]}, "surrogates": [], "display_graphs": False,
    }
    with pytest.raises(ValidationError):
        PROXIMALConfig(**config_dict)


def test_proximal_missing_donorproxies_key_rejected(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "vars": {}, "surrogates": [], "display_graphs": False,
    }
    with pytest.raises(MlsynthConfigError, match="must contain a non-empty list for 'donorproxies'."):
        PROXIMALConfig(**config_dict)


def test_proximal_empty_donorproxies_list_rejected(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "vars": {"donorproxies": []}, "surrogates": [], "display_graphs": False,
    }
    with pytest.raises(MlsynthConfigError, match="must contain a non-empty list for 'donorproxies'."):
        PROXIMALConfig(**config_dict)


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_surrogates_missing_vars_keys_rejected(sample_proximal_data: pd.DataFrame) -> None:
    base: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "surrogates": [4, 5], "display_graphs": False,
    }
    with pytest.raises(MlsynthConfigError, match="'surrogatevars' when surrogates are provided."):
        PROXIMALConfig(**{**base, "vars": {"donorproxies": ["DonorProxyVar1"]}})
    with pytest.raises(MlsynthConfigError, match="'surrogatevars' when surrogates are provided."):
        PROXIMALConfig(**{**base, "vars": {"donorproxies": ["DonorProxyVar1"], "surrogatevars": []}})
    with pytest.raises(MlsynthConfigError, match="must contain a non-empty list for 'donorproxies'."):
        PROXIMALConfig(**{**base, "vars": {"surrogatevars": ["SurrogateSpecificProxyVar1"]}})


# --- Fit-time data errors ---

def test_proximal_donorproxy_col_missing_in_df(sample_proximal_data: pd.DataFrame) -> None:
    df_missing = sample_proximal_data.drop(columns=["DonorProxyVar1"])
    config_dict: Dict[str, Any] = {
        "df": df_missing, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]}, "surrogates": [], "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    with pytest.raises(MlsynthEstimationError, match="DonorProxyVar1"):
        estimator.fit()


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_surrogate_col_missing_in_df(sample_proximal_data: pd.DataFrame) -> None:
    df_missing = sample_proximal_data.drop(columns=["SurrogateSpecificProxyVar1"])
    config_dict: Dict[str, Any] = {
        "df": df_missing, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3], "surrogates": [4, 5],
        "vars": {"donorproxies": ["DonorProxyVar1"], "surrogatevars": ["SurrogateSpecificProxyVar1"]},
        "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    with pytest.raises(MlsynthEstimationError, match="SurrogateSpecificProxyVar1"):
        estimator.fit()


def test_proximal_no_valid_donors_raises(sample_proximal_data: pd.DataFrame) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [10, 11],
        "vars": {"donorproxies": ["DonorProxyVar1"]}, "surrogates": [], "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    with pytest.raises(MlsynthDataError, match="donor units are present"):
        estimator.fit()


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": False}], indirect=True)
def test_proximal_insufficient_pre_periods_pi_only(sample_proximal_data: pd.DataFrame) -> None:
    df_short = sample_proximal_data.copy()
    df_short["IsTreated"] = 0
    df_short.loc[(df_short["UnitIdentifier"] == 1) & (df_short["TimeIdx"] >= 2), "IsTreated"] = 1
    config_dict: Dict[str, Any] = {
        "df": df_short, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]}, "surrogates": [], "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    with pytest.raises(MlsynthEstimationError, match=r"(Singular matrix|Not enough pre-treatment)"):
        estimator.fit()


@pytest.mark.parametrize(
    "sample_proximal_data, nan_column",
    [
        ({"with_surrogates": False}, "OutcomeValue"),
        ({"with_surrogates": False}, "DonorProxyVar1"),
        ({"with_surrogates": True}, "OutcomeValue"),
        ({"with_surrogates": True}, "DonorProxyVar1"),
        ({"with_surrogates": True}, "SurrogateSpecificProxyVar1"),
    ],
    indirect=["sample_proximal_data"],
)
def test_proximal_fit_with_nans(sample_proximal_data: pd.DataFrame, nan_column: str) -> None:
    """balance() imputes NaNs at the start of fit, so estimation proceeds."""
    df_with_nans = sample_proximal_data.copy()
    idx_to_nan = df_with_nans[df_with_nans["UnitIdentifier"] == 2].index[0]
    df_with_nans.loc[idx_to_nan, nan_column] = np.nan

    config_vars = {"donorproxies": ["DonorProxyVar1"]}
    has_surr = "SurrogateSpecificProxyVar1" in df_with_nans.columns
    if has_surr:
        config_vars["surrogatevars"] = ["SurrogateSpecificProxyVar1"]

    config_dict: Dict[str, Any] = {
        "df": df_with_nans, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "surrogates": [4, 5] if has_surr else [],
        "vars": config_vars, "display_graphs": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    results = estimator.fit()
    assert isinstance(results, PROXIMALResults)
    assert results.pi is not None
    if has_surr:
        assert results.pis is not None and results.pipost is not None


# --- Plotting ---

@pytest.mark.parametrize("display_graphs_flag", [True, False])
@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": False}], indirect=True)
@patch("mlsynth.estimators.proximal.plot_proximal")
def test_proximal_plotting_pi_only(
    mock_plot: Any, sample_proximal_data: pd.DataFrame, display_graphs_flag: bool
) -> None:
    config_dict: Dict[str, Any] = {
        "df": sample_proximal_data, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "vars": {"donorproxies": ["DonorProxyVar1"]}, "surrogates": [],
        "display_graphs": display_graphs_flag, "save": False,
    }
    estimator = PROXIMAL(PROXIMALConfig(**config_dict))
    results = estimator.fit()

    if display_graphs_flag:
        mock_plot.assert_called_once()
        (called_with,) = mock_plot.call_args.args
        assert called_with is results
    else:
        mock_plot.assert_not_called()

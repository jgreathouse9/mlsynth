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
    outcomes, donor_proxy_data, surrogate_specific_proxy_data = [], [], []
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


def _base(df, **extra):
    cfg = {
        "df": df, "outcome": "OutcomeValue", "treat": "IsTreated",
        "unitid": "UnitIdentifier", "time": "TimeIdx", "donors": [2, 3],
        "display_graphs": False,
    }
    cfg.update(extra)
    return cfg


# --- Construction ---

def test_proximal_creation(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    assert estimator.outcome == "OutcomeValue"
    assert estimator.donors == [2, 3]
    assert estimator.methods == ["PI"]


def test_proximal_creation_from_dict(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]}))
    assert isinstance(estimator.config, PROXIMALConfig)


# --- Helpers ---

def _validate_method_fit(fit: ProximalMethodFit, name: str, T: int) -> None:
    assert isinstance(fit, ProximalMethodFit)
    assert fit.name == name
    assert fit.counterfactual.shape == (T,)
    assert fit.gap.shape == (T,)
    assert fit.time_varying_effect.shape == (T,)
    assert isinstance(fit.att, float) and np.isfinite(fit.att)
    assert fit.att_se is None or isinstance(fit.att_se, float)
    assert isinstance(fit.pre_rmse, float) and isinstance(fit.post_rmse, float)
    assert isinstance(fit.alpha_weights, np.ndarray)
    assert isinstance(fit.donor_weights, dict)


# --- Method selection: exactly what's asked runs ---

def test_proximal_pi_only(sample_proximal_data: pd.DataFrame) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]}))).fit()
    assert isinstance(results, PROXIMALResults) and results.mode == "proximal"
    assert list(results.methods.keys()) == ["PI"]
    assert results.pis is None and results.pipost is None and results.spsc is None
    n = sample_proximal_data["TimeIdx"].nunique()
    _validate_method_fit(results.pi, "PI", n)
    assert results.att == results.pi.att
    assert set(results.donor_weights.keys()) == {2, 3}


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_all_methods(sample_proximal_data: pd.DataFrame) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI", "PIS", "PIPost", "SPSC"],
        surrogates=[4, 5],
        vars={"donorproxies": ["DonorProxyVar1"], "surrogatevars": ["SurrogateSpecificProxyVar1"]}))).fit()
    assert list(results.methods.keys()) == ["PI", "PIS", "PIPost", "SPSC"]
    n = sample_proximal_data["TimeIdx"].nunique()
    for name, fit in results.methods.items():
        _validate_method_fit(fit, name, n)
    assert set(results.att_by_method()) == {"PI", "PIS", "PIPost", "SPSC"}


def test_proximal_selected_variant_is_first_method(sample_proximal_data: pd.DataFrame) -> None:
    """With PI not requested, convenience accessors forward to the first method."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=False))).fit()
    assert results.pi is None
    assert results.att == results.spsc.att


# --- SPSC: single proxy, no proxies required ---

@pytest.mark.parametrize("detrend", [False, True])
def test_proximal_spsc_no_proxies(sample_proximal_data: pd.DataFrame, detrend: bool) -> None:
    """Methods=['SPSC'] runs with only donors -- no donorproxies/surrogates."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=detrend))).fit()
    assert list(results.methods.keys()) == ["SPSC"]
    fit = results.spsc
    assert np.isfinite(fit.att)
    assert fit.metadata["variant"] == ("SPSC-DT" if detrend else "SPSC-NoDT")
    assert fit.metadata["detrend"] is detrend


def test_proximal_spsc_conformal(sample_proximal_data: pd.DataFrame) -> None:
    """Conformal intervals attach to the SPSC fit when requested."""
    n = sample_proximal_data["TimeIdx"].nunique()
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["SPSC"], spsc_detrend=False,
        spsc_conformal=True, spsc_conformal_periods=[7, 8]))).fit()
    cf = results.spsc.metadata.get("conformal")
    assert cf is not None
    assert list(cf["periods"]) == [7, 8]
    assert cf["lower"].shape == (2,) and cf["upper"].shape == (2,)
    assert np.all(cf["upper"] >= cf["lower"])


# --- DR + PIPW (doubly robust proximal) ---

def test_proximal_dr_pipw_path(sample_proximal_data: pd.DataFrame) -> None:
    """Methods=['DR','PIPW'] run off donors + donorproxies (W, Z)."""
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["DR", "PIPW"],
        vars={"donorproxies": ["DonorProxyVar1"]}))).fit()
    assert list(results.methods.keys()) == ["DR", "PIPW"]
    assert results.dr is not None and results.pipw is not None
    n = sample_proximal_data["TimeIdx"].nunique()
    assert isinstance(results.dr.att, float)
    assert results.dr.counterfactual.shape == (n,) and np.all(np.isfinite(results.dr.counterfactual))
    # PIPW is a weighting estimator: no imputed counterfactual trajectory.
    assert np.all(np.isnan(results.pipw.counterfactual))
    assert isinstance(results.pipw.att, float)


def test_proximal_dr_requires_donorproxies(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["DR"], vars={}))
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PIPW"], vars={}))


# --- Config validation (input consistency per requested method) ---

def test_proximal_methods_required(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        PROXIMALConfig(**_base(sample_proximal_data, vars={"donorproxies": ["DonorProxyVar1"]}))


def test_proximal_unknown_method_rejected(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="Unknown PROXIMAL method"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["NOPE"]))


def test_proximal_empty_donors_rejected(sample_proximal_data: pd.DataFrame) -> None:
    cfg = _base(sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})
    cfg["donors"] = []
    with pytest.raises(ValidationError):
        PROXIMALConfig(**cfg)


def test_proximal_pi_requires_donorproxies(sample_proximal_data: pd.DataFrame) -> None:
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI"], vars={}))
    with pytest.raises(MlsynthConfigError, match="donorproxies"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI"], vars={"donorproxies": []}))


def test_proximal_spsc_needs_no_proxies(sample_proximal_data: pd.DataFrame) -> None:
    """SPSC-only must NOT require donorproxies/surrogatevars."""
    cfg = PROXIMALConfig(**_base(sample_proximal_data, methods=["SPSC"]))
    assert cfg.methods == ["SPSC"]


@pytest.mark.parametrize("sample_proximal_data", [{"with_surrogates": True}], indirect=True)
def test_proximal_pis_requires_surrogates(sample_proximal_data: pd.DataFrame) -> None:
    # surrogatevars missing
    with pytest.raises(MlsynthConfigError, match="surrogatevars"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PI", "PIS", "PIPost"],
                               surrogates=[4, 5], vars={"donorproxies": ["DonorProxyVar1"]}))
    # surrogates list empty
    with pytest.raises(MlsynthConfigError, match="surrogates"):
        PROXIMALConfig(**_base(sample_proximal_data, methods=["PIS"], surrogates=[],
                               vars={"donorproxies": ["DonorProxyVar1"], "surrogatevars": ["SurrogateSpecificProxyVar1"]}))


# --- Fit-time data errors ---

def test_proximal_donorproxy_col_missing_in_df(sample_proximal_data: pd.DataFrame) -> None:
    df_missing = sample_proximal_data.drop(columns=["DonorProxyVar1"])
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        df_missing, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthEstimationError, match="DonorProxyVar1"):
        estimator.fit()


def test_proximal_no_valid_donors_raises(sample_proximal_data: pd.DataFrame) -> None:
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], donors=[10, 11],
        vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthDataError, match="donor units are present"):
        estimator.fit()


def test_proximal_insufficient_pre_periods(sample_proximal_data: pd.DataFrame) -> None:
    df_short = sample_proximal_data.copy()
    df_short["IsTreated"] = 0
    df_short.loc[(df_short["UnitIdentifier"] == 1) & (df_short["TimeIdx"] >= 2), "IsTreated"] = 1
    estimator = PROXIMAL(PROXIMALConfig(**_base(
        df_short, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]})))
    with pytest.raises(MlsynthEstimationError, match=r"(Singular matrix|Not enough pre-treatment)"):
        estimator.fit()


@pytest.mark.parametrize(
    "sample_proximal_data, nan_column",
    [
        ({"with_surrogates": False}, "OutcomeValue"),
        ({"with_surrogates": False}, "DonorProxyVar1"),
        ({"with_surrogates": True}, "SurrogateSpecificProxyVar1"),
    ],
    indirect=["sample_proximal_data"],
)
def test_proximal_fit_with_nans(sample_proximal_data: pd.DataFrame, nan_column: str) -> None:
    """Balance() imputes NaNs at the start of fit, so estimation proceeds."""
    df_with_nans = sample_proximal_data.copy()
    idx_to_nan = df_with_nans[df_with_nans["UnitIdentifier"] == 2].index[0]
    df_with_nans.loc[idx_to_nan, nan_column] = np.nan

    has_surr = "SurrogateSpecificProxyVar1" in df_with_nans.columns
    vars_ = {"donorproxies": ["DonorProxyVar1"]}
    if has_surr:
        vars_["surrogatevars"] = ["SurrogateSpecificProxyVar1"]
    methods = ["PI", "PIS", "PIPost"] if has_surr else ["PI"]

    results = PROXIMAL(PROXIMALConfig(**_base(
        df_with_nans, methods=methods, surrogates=[4, 5] if has_surr else [], vars=vars_))).fit()
    assert results.pi is not None
    if has_surr:
        assert results.pis is not None and results.pipost is not None


# --- Plotting ---

@pytest.mark.parametrize("display_graphs_flag", [True, False])
@patch("mlsynth.estimators.proximal.plot_proximal")
def test_proximal_plotting(mock_plot: Any, sample_proximal_data: pd.DataFrame, display_graphs_flag: bool) -> None:
    results = PROXIMAL(PROXIMALConfig(**_base(
        sample_proximal_data, methods=["PI"], vars={"donorproxies": ["DonorProxyVar1"]},
        display_graphs=display_graphs_flag, save=False))).fit()
    if display_graphs_flag:
        mock_plot.assert_called_once()
        (called_with,) = mock_plot.call_args.args
        assert called_with is results
    else:
        mock_plot.assert_not_called()

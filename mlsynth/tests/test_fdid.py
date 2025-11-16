import pytest
import numpy as np
import pandas as pd
from typing import Any, Dict
from mlsynth.estimators.fdid import FDIDOutput
from mlsynth.estimators.fdid import FDID, FDIDConfig, FDIDOutput
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError


@pytest.fixture
def sample_fdid_data() -> pd.DataFrame:
    """Create a small balanced panel dataset for FDID testing."""
    data = {
        "unit": ["T"] * 4 + ["C1"] * 4 + ["C2"] * 4,
        "time": [1, 2, 3, 4] * 3,
        "y": [10, 12, 20, 22, 8, 9, 10, 11, 9, 10, 11, 12],
        "treated_indicator": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)


def test_fdid_creation(sample_fdid_data: pd.DataFrame):
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    assert isinstance(estimator, FDID)


def test_fdid_fit_smoke(sample_fdid_data: pd.DataFrame):
    """Smoke test for FDID fit method."""
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    results = estimator.fit()

    # FDID.fit now returns FDIDOutput
    assert isinstance(results, FDIDOutput)
    assert "FDID" in results.results
    assert "DID" in results.results


def test_fdid_fit_insufficient_periods(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with insufficient pre or post periods."""
    df = sample_fdid_data.copy()
    # Make pre-period only 1 row for treated
    df.loc[(df["unit"] == "T") & (df["time"] == 1), "treated_indicator"] = 0
    df.loc[(df["unit"] == "T") & (df["time"] == 2), "treated_indicator"] = 1

    config = FDIDConfig(
        df=df,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)

    # Expect MlsynthEstimationError due to all-NaN slice in selector
    with pytest.raises(MlsynthEstimationError, match="All-NaN slice encountered"):
        estimator.fit()


def test_fdid_fit_insufficient_donors(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with too few or no donor units."""
    # 1. No donor units
    df_no_donors = sample_fdid_data[sample_fdid_data["unit"] == "T"].copy()
    config_no_donors = FDIDConfig(
        df=df_no_donors,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator_no_donors = FDID(config=config_no_donors)
    with pytest.raises(MlsynthDataError, match="No donor units found"):
        estimator_no_donors.fit()

    # 2. Only one donor unit
    df_one_donor = sample_fdid_data[sample_fdid_data["unit"].isin(["T", "C1"])].copy()
    config_one_donor = FDIDConfig(
        df=df_one_donor,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator_one_donor = FDID(config=config_one_donor)
    results = estimator_one_donor.fit()
    assert isinstance(results, FDIDOutput)
    assert "FDID" in results.results
    assert "DID" in results.results


def test_fdid_fit_nan_in_outcome(sample_fdid_data: pd.DataFrame):
    """Test FDID fit with NaN values in the outcome variable."""
    df_nan = sample_fdid_data.copy()
    df_nan.loc[(df_nan["unit"] == "T") & (df_nan["time"] == 1), "y"] = np.nan

    config = FDIDConfig(
        df=df_nan,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)

    with pytest.raises(MlsynthEstimationError, match="All-NaN slice encountered"):
        estimator.fit()


def test_fdid_results_structure_and_types(sample_fdid_data: pd.DataFrame):
    """Check that FDID.fit returns correct structure and types."""
    config = FDIDConfig(
        df=sample_fdid_data,
        unitid="unit",
        time="time",
        outcome="y",
        treat="treated_indicator",
        display_graphs=False,
    )
    estimator = FDID(config=config)
    results = estimator.fit()

    assert isinstance(results, FDIDOutput)
    for method_name, method_result in results.results.items():
        # Each result should have effects, time_series, weights, and inference
        assert hasattr(method_result, "effects")
        assert hasattr(method_result, "time_series")
        assert hasattr(method_result, "weights")
        assert hasattr(method_result, "inference")

# tests/test_marex.py
import pytest
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from pydantic import ValidationError  # Keep this import

def test_initialization_valid_config(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "cluster": "Region",  # Use column name as per schema
        "design": "eq11",
        "m_eq": 1,
        "lambda1": 0.2,
        "lambda2": 0.2
    }
    try:
        marex = MAREX(config=MAREXConfig(**config_data))
        assert marex.df is not None
        assert marex.outcome == "Y_obs"
        assert marex.T0 == 104
        assert marex.clusters is not None  # Check that clusters is derived
    except ValidationError as e:
        assert False, f"Initialization failed with ValidationError: {e}"

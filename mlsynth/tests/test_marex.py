# tests/test_marex.py
import pytest
from mlsynth import MAREX
from mlsynth.config_models import MAREXConfig
from pydantic import ValidationError  # Add this import

def test_initialization_valid_config(curacao_sim_data):
    config_data = {
        "df": curacao_sim_data["df"],
        "outcome": "Y_obs",
        "unitid": "town",
        "time": "time",
        "T0": 104,
        "clusters": curacao_sim_data["clusters"],  # Use the clusters array from fixture
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
        assert marex.clusters is not None  # Check clusters is set (array, not string)
    except ValidationError as e:
        assert False, f"Initialization failed with ValidationError: {e}"

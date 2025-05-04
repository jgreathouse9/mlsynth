import pytest
import numpy as np
import pandas as pd
from mlsynth.utils.datautils import dataprep, balance, treatlogic

def test_treat_single_unit():
    treatment_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1]  # Only unit 2 is treated
    ])
    result = treatlogic(treatment_matrix)
    assert result["Num Treated Units"] == 1
    assert result["Treated Index"] == [2]
    assert result["Pre Periods"] == 1
    assert result["Post Periods"] == 2


def test_treat_multiple_units():
    treatment_matrix = np.array([
        [0, 0, 0],  # Unit 0 - never treated
        [0, 1, 1],  # Unit 1 - treated
        [0, 1, 1]   # Unit 2 - treated
    ])
    result = treatlogic(treatment_matrix)
    assert result["Num Treated Units"] == 2
    assert set(result["Treated Index"]) == {1, 2}


def test_treat_no_treated_units():
    treatment_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # No treated units
    with pytest.raises(AssertionError):
        treatlogic(treatment_matrix)

def test_treat_invalid_type():
    with pytest.raises(TypeError):
        treatlogic("invalid_input")  # Passing a non-NumPy array

def test_treat_division_by_zero():
    treatment_matrix = np.array([[1, 0], [1, 0], [0, 0]])  # Zero treatment in the second unit
    with pytest.raises(ZeroDivisionError):
        treatlogic(treatment_matrix)

# Test `dataprep`
def test_dataprep_single_treated_unit():
    df = pd.DataFrame({
        'unitid': [0, 0, 1, 1],
        'time': [0, 1, 0, 1],
        'outcome': [10, 20, 30, 40],
        'treat': [0, 1, 0, 0]
    })
    result = dataprep(df, 'unitid', 'time', 'outcome', 'treat')
    assert "treated_unit_name" in result
    assert result["treated_unit_name"] == 1
    assert result["total_periods"] == 2
    assert result["pre_periods"] == 1
    assert result["post_periods"] == 1

def test_dataprep_multiple_treated_units():
    df = pd.DataFrame({
        'unitid': [0, 0, 1, 1, 2, 2],
        'time': [0, 1, 0, 1, 0, 1],
        'outcome': [10, 20, 30, 40, 50, 60],
        'treat': [1, 1, 0, 1, 0, 0]
    })
    result = dataprep(df, 'unitid', 'time', 'outcome', 'treat')
    assert "cohorts" in result
    assert len(result["cohorts"]) > 0

# Test `balance`
def test_balance_valid():
    df = pd.DataFrame({
        'unit': [1, 1, 2, 2],
        'time': [0, 1, 0, 1],
        'outcome': [10, 20, 30, 40]
    })
    try:
        balance(df, 'unit', 'time')
    except ValueError:
        pytest.fail("ValueError raised unexpectedly")

def test_balance_invalid():
    df = pd.DataFrame({
        'unit': [1, 1, 2, 2, 2],
        'time': [0, 1, 0, 1, 2],
        'outcome': [10, 20, 30, 40, 50]
    })
    with pytest.raises(ValueError, match="The panel is not strongly balanced"):
        balance(df, 'unit', 'time')

def test_balance_duplicate_observations():
    df = pd.DataFrame({
        'unit': [1, 1, 2, 2, 2],
        'time': [0, 1, 0, 1, 1],
        'outcome': [10, 20, 30, 40, 50]
    })
    with pytest.raises(ValueError, match="Duplicate observations found"):
        balance(df, 'unit', 'time')

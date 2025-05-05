import pytest
import numpy as np
import pandas as pd
from mlsynth.utils.datautils import dataprep, balance, logictreat

# === Test Data Helpers ===

def make_single_treated_df():
    """Creates a panel with a single treated unit"""
    data = {
        'unit': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
        'time': [1, 2, 3, 4, 5] * 3,
        'outcome': list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35)),
        'treat': [0, 0, 1, 1, 1] + [0] * 5 + [0] * 5
    }
    return pd.DataFrame(data)


def make_multiple_treated_df():
    """Creates a panel with multiple treated units"""
    df = make_single_treated_df()
    df.loc[df['unit'] == 'C', 'treat'] = [0, 0, 1, 1, 1]
    return df


# === logictreat Tests ===

def test_logictreat_single_unit():
    matrix = np.array([
        [0],
        [0],
        [1],
        [1],
        [1]
    ])
    result = logictreat(matrix)
    assert result['Num Treated Units'] == 1
    assert result['Pre Periods'] == 2
    assert result['Post Periods'] == 3
    assert result['Total Periods'] == 5


def test_logictreat_multiple_units():
    matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    result = logictreat(matrix)
    assert result['Num Treated Units'] == 3
    assert np.array_equal(result['Treated Index'], np.array([0, 1, 2]))
    assert result['Total Periods'] == 5


def test_logictreat_unsustained_treatment_raises():
    matrix = np.array([
        [0],
        [1],
        [0]
    ])
    with pytest.raises(AssertionError, match="Treatment is not sustained"):
        logictreat(matrix)


# === dataprep Tests ===

def test_dataprep_single_unit():
    df = make_single_treated_df()
    result = dataprep(df, unitid='unit', time='time', outcome='outcome', treat='treat')
    assert result['treated_unit_name'] == 'A'
    assert result['pre_periods'] == 2
    assert result['post_periods'] == 3
    assert result['donor_matrix'].shape == (5, 2)


def test_dataprep_multiple_units():
    data = {
        'unit': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
        'time': [1, 2, 3, 4, 5] * 3,
        'outcome': list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35)),
        'treat': [0]*5 + [0, 0, 1, 1, 1] + [0, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    result = dataprep(df, unitid='unit', time='time', outcome='outcome', treat='treat')
    
    assert 'cohorts' in result
    assert len(result['cohorts']) == 2  # B treated at t=3, C at t=2
    assert sorted(result['cohorts'].keys()) == [2, 3]



# === balance Tests ===

def test_balance_valid():
    df = make_single_treated_df()
    balance(df, unit_col='unit', time_col='time')  # Should not raise


def test_balance_unbalanced_raises():
    df = make_single_treated_df().drop(index=0)  # Drop a row to unbalance
    with pytest.raises(ValueError, match="The panel is not strongly balanced."):
        balance(df, unit_col='unit', time_col='time')


def test_balance_duplicates_raises():
    df = make_single_treated_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate observations found"):
        balance(df, unit_col='unit', time_col='time')

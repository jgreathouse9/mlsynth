import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from mlsynth.utils.datautils import dataprep, balance, logictreat, clean_surrogates2, proxy_dataprep
from mlsynth.exceptions import MlsynthDataError

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

def test_logictreat_invalid_input_type():
    """Test logictreat with non-numpy array input."""
    with pytest.raises(MlsynthDataError, match="treatment_matrix must be a NumPy array"):
        logictreat("not_a_numpy_array")

def test_logictreat_no_treated_units():
    """Test logictreat with no treated units in the matrix."""
    matrix = np.array([
        [0, 0],
        [0, 0]
    ])
    with pytest.raises(MlsynthDataError, match="No treated units found \\(zero treated observations with value 1\\)"): # Updated match string
        logictreat(matrix)

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

def test_logictreat_treatment_from_start_single_unit():
    """Test logictreat for a single unit treated from the first period."""
    matrix = np.array([
        [1],
        [1],
        [1]
    ])
    result = logictreat(matrix)
    assert result['Num Treated Units'] == 1
    assert result['Pre Periods'] == 0
    assert result['Post Periods'] == 3 # All periods are post-treatment (including start)
    assert result['Total Periods'] == 3
    assert np.array_equal(result['Treated Index'], np.array([0]))

def test_logictreat_multiple_units_detailed():
    """Test logictreat with multiple units and varying start times, checking detailed period counts."""
    matrix = np.array([
        # U0 U1 U2 (U1 never treated)
        [0, 0, 0], # Time 0
        [0, 0, 0], # Time 1
        [1, 0, 0], # Time 2 (U0 treated)
        [1, 0, 1], # Time 3 (U2 treated)
        [1, 0, 1]  # Time 4
    ])
    result = logictreat(matrix)
    assert result['Num Treated Units'] == 2
    assert np.array_equal(result['Treated Index'], np.array([0, 2])) # U0 and U2
    assert result['Total Periods'] == 5
    
    # Check per-unit periods for treated units
    # U0 (index 0) treated at time index 2: Pre=2, Post=3 (2,3,4)
    # U2 (index 2) treated at time index 3: Pre=3, Post=2 (3,4)
    assert np.array_equal(result['First Treat Periods'], np.array([2, 3]))
    assert np.array_equal(result['Pre Periods by Unit'], np.array([2, 3]))
    assert np.array_equal(result['Post Periods by Unit'], np.array([3, 2])) # TotalPeriods - PrePeriodsByUnit

def test_logictreat_multiple_units(): # Original test, can be kept or merged/removed
    matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 1], # U0, U2 treat at time index 2
        [1, 0, 1],
        [1, 1, 1]  # U1 treats at time index 4
    ])
    # Corrected expectation based on how logictreat works:
    # U0 treated at index 2. Pre=2, Post=3
    # U1 treated at index 4. Pre=4, Post=1
    # U2 treated at index 2. Pre=2, Post=3
    result = logictreat(matrix)
    assert result['Num Treated Units'] == 3
    assert np.array_equal(result['Treated Index'], np.array([0, 1, 2]))
    assert result['Total Periods'] == 5
    assert np.array_equal(result['First Treat Periods'], np.array([2, 4, 2]))
    assert np.array_equal(result['Pre Periods by Unit'], np.array([2, 4, 2]))
    assert np.array_equal(result['Post Periods by Unit'], np.array([3, 1, 3]))


def test_logictreat_unsustained_treatment_raises():
    matrix = np.array([
        [0],
        [1],
        [0]
    ])
    with pytest.raises(MlsynthDataError, match="Treatment is not sustained for the treated unit."): # Updated match string
        logictreat(matrix)


# === dataprep Tests ===

def test_dataprep_single_unit():
    df = make_single_treated_df()
    result = dataprep(df, unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')
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

    result = dataprep(df, unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')
    
    assert 'cohorts' in result
    assert len(result['cohorts']) == 2  # B treated at t=3, C at t=2
    assert sorted(result['cohorts'].keys()) == [2, 3]

    # Detailed check for one cohort (C treated at time index 1, actual time 2)
    # In the provided data for test_dataprep_multiple_units:
    # 'treat': [0]*5 (A) + [0, 0, 1, 1, 1] (B, treated at time index 2 of its own series, which is time 3 overall)
    #          + [0, 1, 1, 1, 1] (C, treated at time index 1 of its own series, which is time 2 overall)
    # So cohorts are at time 2 (unit C) and time 3 (unit B).
    # Let's check cohort for time 2 (unit C)
    # Time periods are 1-indexed in the original df, so time=2 is index 1.
    # dataprep uses 0-indexed time for cohort keys if it's from Ywide.index.get_loc(treat_time_val)
    # The keys in result['cohorts'] are the actual time values from the time column.
    
    cohort_c_data = result['cohorts'][2] # Unit C treated at time 2
    assert cohort_c_data['treated_units'] == ['C']
    assert cohort_c_data['pre_periods'] == 1 # Time 1 is pre
    assert cohort_c_data['post_periods'] == 4 # Times 2,3,4,5 are post
    assert cohort_c_data['total_periods'] == 5
    assert cohort_c_data['y'].shape == (5,1) # 5 time periods, 1 unit
    assert cohort_c_data['donor_matrix'].shape == (5,2) # Units A and B are donors for C

    cohort_b_data = result['cohorts'][3] # Unit B treated at time 3
    assert cohort_b_data['treated_units'] == ['B']
    assert cohort_b_data['pre_periods'] == 2 # Times 1,2 are pre
    assert cohort_b_data['post_periods'] == 3 # Times 3,4,5 are post
    assert cohort_b_data['y'].shape == (5,1)
    assert cohort_b_data['donor_matrix'].shape == (5,2) # Units A and C are donors for B


def test_dataprep_no_donors():
    """Test dataprep when no donor units are available."""
    df_no_donors = make_single_treated_df()
    df_no_donors = df_no_donors[df_no_donors['unit'] == 'A'] # Keep only treated unit
    with pytest.raises(MlsynthDataError, match="No donor units found after pivoting and selecting."):
        dataprep(df_no_donors, unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')

def test_dataprep_no_pre_periods():
    """Test dataprep when the treated unit has no pre-treatment periods."""
    df_no_pre = make_single_treated_df()
    df_no_pre.loc[df_no_pre['unit'] == 'A', 'treat'] = 1 # Treat unit A from the start
    with pytest.raises(MlsynthDataError, match="Not enough pre-treatment periods \\(0 pre-periods found\\)."):
        dataprep(df_no_pre, unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')

def test_dataprep_missing_columns():
    """Test dataprep with missing essential columns in DataFrame."""
    df = make_single_treated_df()
    
    with pytest.raises(KeyError, match="'missing_outcome'"):
        dataprep(df.drop(columns=['outcome']), unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='missing_outcome', treatment_indicator_column_name='treat')
    
    with pytest.raises(KeyError, match="'missing_treat'"):
        dataprep(df.drop(columns=['treat']), unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='missing_treat')

    # Test for missing unitid or time column (pivot will fail)
    with pytest.raises(KeyError): # Actual error message might vary (e.g. "['unitid'] not in index")
        dataprep(df.drop(columns=['unit']), unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')
    
    with pytest.raises(KeyError):
        dataprep(df.drop(columns=['time']), unit_id_column_name='unit', time_period_column_name='time', outcome_column_name='outcome', treatment_indicator_column_name='treat')


# === balance Tests ===

def test_balance_valid():
    df = make_single_treated_df()
    balance(df, unit_id_column_name='unit', time_period_column_name='time')  # Should not raise


def test_balance_unbalanced_raises():
    df = make_single_treated_df().drop(index=0)  # Drop a row to unbalance
    # The error message in balance() for this case is "The panel is not strongly balanced. Not all units have observations for all time periods."
    # or "The panel is not strongly balanced. Units have different numbers of observations."
    # Using a more general match for "The panel is not strongly balanced."
    with pytest.raises(MlsynthDataError, match="The panel is not strongly balanced."):
        balance(df, unit_id_column_name='unit', time_period_column_name='time')


def test_balance_duplicates_raises():
    df = make_single_treated_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(MlsynthDataError, match="Duplicate observations found"):
        balance(df, unit_id_column_name='unit', time_period_column_name='time')


# === Proximal/Special Dataprep tests ===


def test_clean_surrogates2_basic():
    # Simulate input matrices
    X = np.array([[1, 2], [2, 3], [3, 4]])
    Z0 = np.array([[1], [2], [3]])
    W = np.array([[2], [3], [4]])
    T0 = 2

    cleaned = clean_surrogates2(X, Z0, W, T0)

    assert cleaned.shape == X.shape
    # Check that output is numeric and finite
    assert np.isfinite(cleaned).all()

def test_clean_surrogates2_singular_matrix():
    """Test clean_surrogates2 when the matrix for np.linalg.solve is singular."""
    # Make Z0W_pre singular. This can happen if T0 < number of covariates or collinearity.
    # Example: T0 = 1, Z0 and W have 1 covariate each. Z0W_pre will be scalar.
    # If Z0_aug[:T0] or W_aug[:T0] leads to a non-invertible Z0W_pre
    X = np.array([[1, 2], [2, 3], [3, 4]]).astype(float)
    Z0 = np.array([[1], [1], [1]]).astype(float) # Collinear columns if T0 is small
    W = np.array([[1], [1], [1]]).astype(float)
    T0 = 1 # With T0=1, Z0_aug[:T0].T @ W_aug[:T0] becomes a 1x1 matrix [Z0[0]*W[0]]
           # If Z0[0]*W[0] is 0, it's singular for solve. Let's make it non-zero but make Z0W_pre singular later.

    # To make Z0W_pre singular with more covariates:
    # Let Z0_aug and W_aug be (T0, K)
    # Z0W_pre is (K, K). We need K > T0 for guaranteed rank deficiency.
    # Or perfect collinearity for K <= T0.
    
    # Scenario: T0 = 1, K = 2. Z0W_pre will be 2x2 but rank 1.
    Z0_singular = np.array([[1, 1], [2, 2], [3, 3]]).astype(float) # Collinear columns
    W_singular = np.array([[1, 1], [2, 2], [3, 3]]).astype(float)
    T0_small = 1
    
    with pytest.raises(np.linalg.LinAlgError): # Expect "Singular matrix"
        clean_surrogates2(X, Z0_singular, W_singular, T0_small)

    # Scenario: T0 = 2, K = 2, but columns are collinear
    Z0_collinear = np.array([[1, 2], [1, 2], [3, 4]]).astype(float)
    W_collinear = np.array([[1, 2], [1, 2], [3, 4]]).astype(float)
    T0_sufficient_rows_but_collinear = 2
    with pytest.raises(np.linalg.LinAlgError):
        clean_surrogates2(X, Z0_collinear, W_collinear, T0_sufficient_rows_but_collinear)


def test_clean_surrogates2_with_cy():
    X = np.random.rand(5, 2)
    Z0 = np.random.rand(5, 2)
    W = np.random.rand(5, 2)
    Cy = np.random.rand(5, 1)
    T0 = 3

    cleaned = clean_surrogates2(X, Z0, W, T0, Cy)

    assert cleaned.shape == X.shape
    assert np.isfinite(cleaned).all()

def test_proxy_dataprep_shapes():
    df = pd.DataFrame({
        'Artist': ['A', 'A', 'B', 'B'],
        'Date': [1, 2, 1, 2],
        'proxy1': [10, 11, 20, 21],
        'proxy2': [30, 31, 40, 41],
    })

    proxy_vars = {
        'donorproxies': ['proxy1'],
        'surrogatevars': ['proxy2']
    }

    surrogate_units = ['A', 'B']

    X, Z1 = proxy_dataprep(df, surrogate_units, proxy_variable_column_names_map=proxy_vars, unit_id_column_name='Artist', time_period_column_name='Date')

    assert X.shape == (2, 2)
    assert Z1.shape == (2, 2)
    assert_array_almost_equal(X, np.array([[10, 20], [11, 21]]))
    assert_array_almost_equal(Z1, np.array([[30, 40], [31, 41]]))


def test_proxy_dataprep_invalid_proxy_vars():
    df = pd.DataFrame({
        'Artist': ['A', 'A', 'B', 'B'],
        'Date': [1, 2, 1, 2],
        'proxy1': [10, 11, 20, 21],
        'proxy2': [30, 31, 40, 41],
    })
    surrogate_units = ['A', 'B']

    # Missing 'donorproxies'
    proxy_vars_missing_donor = {'surrogatevars': ['proxy2']}
    with pytest.raises(KeyError, match="'donorproxies'"):
        proxy_dataprep(df, surrogate_units, proxy_variable_column_names_map=proxy_vars_missing_donor, unit_id_column_name='Artist', time_period_column_name='Date')

    # Missing 'surrogatevars'
    proxy_vars_missing_surrogate = {'donorproxies': ['proxy1']}
    with pytest.raises(KeyError, match="'surrogatevars'"):
        proxy_dataprep(df, surrogate_units, proxy_variable_column_names_map=proxy_vars_missing_surrogate, unit_id_column_name='Artist', time_period_column_name='Date')

    # Empty list for 'donorproxies'
    proxy_vars_empty_donor_list = {'donorproxies': [], 'surrogatevars': ['proxy2']}
    with pytest.raises(IndexError): # Accessing [0] on empty list
        proxy_dataprep(df, surrogate_units, proxy_variable_column_names_map=proxy_vars_empty_donor_list, unit_id_column_name='Artist', time_period_column_name='Date')

    # Empty list for 'surrogatevars'
    proxy_vars_empty_surrogate_list = {'donorproxies': ['proxy1'], 'surrogatevars': []}
    with pytest.raises(IndexError): # Accessing [0] on empty list
        proxy_dataprep(df, surrogate_units, proxy_variable_column_names_map=proxy_vars_empty_surrogate_list, unit_id_column_name='Artist', time_period_column_name='Date')


def test_proxy_dataprep_empty_surrogate_units():
    df = pd.DataFrame({
        'Artist': ['A', 'A', 'B', 'B'],
        'Date': [1, 2, 1, 2],
        'proxy1': [10, 11, 20, 21],
        'proxy2': [30, 31, 40, 41],
    })
    proxy_vars = {
        'donorproxies': ['proxy1'],
        'surrogatevars': ['proxy2']
    }
    empty_surrogate_units = []
    # Expect pivot to result in empty DataFrame, then .to_numpy() on empty DF.
    # This should result in empty arrays, not an error.
    X, Z1 = proxy_dataprep(df, empty_surrogate_units, proxy_variable_column_names_map=proxy_vars, unit_id_column_name='Artist', time_period_column_name='Date')
    assert X.shape == (0,0) # Empty DataFrame pivot results in (0,0) array
    assert Z1.shape == (0,0) # Empty DataFrame pivot results in (0,0) array

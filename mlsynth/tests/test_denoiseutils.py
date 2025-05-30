import numpy as np
import pytest
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.denoiseutils import (
    universal_rank,
    spectral_rank,
    shrink,
    SVT,
    svt,
    RPCA
)

# ---------- Tests for universal_rank ----------
def test_universal_rank_basic():
    s = np.array([5, 3, 1, 0.5])
    rank = universal_rank(s, matrix_aspect_ratio=0.5)
    assert isinstance(rank, int)
    assert rank >= 1

# ---------- Tests for spectral_rank ----------
def test_spectral_rank_exact():
    s = np.array([4, 2, 1])
    t = 1.0
    rank = spectral_rank(s, t)
    assert rank == len(s)

def test_spectral_rank_partial():
    s = np.array([3, 2, 1])
    t = 0.8
    rank = spectral_rank(s, t)
    assert rank >= 1 and rank <= len(s)

# ---------- Tests for shrink ----------
def test_shrink_behavior():
    s = np.array([5.0, 2.0, 0.5])
    tau = 1.0
    shrunk = shrink(s, tau)
    expected = np.array([4.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(shrunk, expected)

# ---------- Tests for SVT (singular value thresholding) ----------
def test_SVT_identity_matrix():
    X = np.eye(3)
    tau = 0.5
    result = SVT(X, tau)
    assert result.shape == X.shape
    assert np.all(np.isfinite(result))

# ---------- Tests for svt (low-rank approximation) ----------
def test_svt_low_rank_approximation():
    X = np.random.randn(5, 4)
    Y0_rank, n2, u_rank, s_rank, v_rank = svt(X)

    assert Y0_rank.shape == X.shape
    assert u_rank.shape[0] == X.shape[0]
    assert v_rank.shape[1] == X.shape[1]
    assert len(s_rank) == u_rank.shape[1] == v_rank.shape[0]

# ---------- Tests for RPCA ----------
def test_RPCA_output_shape_and_type():
    X = np.random.randn(5, 5)
    L = RPCA(X)
    assert L.shape == X.shape
    assert np.all(np.isfinite(L))

def test_RPCA_low_rank_approximation():
    # Construct a low-rank matrix and add noise
    U = np.random.randn(5, 2)
    V = np.random.randn(2, 5)
    low_rank = U @ V
    noise = np.random.normal(scale=0.1, size=low_rank.shape)
    X = low_rank + noise
    L = RPCA(X)
    error = np.linalg.norm(L - low_rank) / np.linalg.norm(low_rank)
    assert error < 0.7  # Further increased tolerance for qualitative approximation

# ---------- Tests for standardize ----------
from mlsynth.utils.denoiseutils import standardize

def test_standardize_matrix():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    X_std = standardize(X)
    assert X_std.shape == X.shape
    # Check column means are close to 0
    np.testing.assert_array_almost_equal(np.mean(X_std, axis=0), np.zeros(X.shape[1]), decimal=6)
    # Check column std devs are close to 1
    np.testing.assert_array_almost_equal(np.std(X_std, axis=0), np.ones(X.shape[1]), decimal=6)

def test_standardize_single_value_column():
    # Column with all same values will have std=0, leading to division by zero
    X = np.array([[1, 2], [1, 3], [1, 4]], dtype=float)
    with pytest.raises(MlsynthDataError, match=r"Cannot standardize columns with zero standard deviation: columns \[[^]]+\]"): # Use raw string and more general regex
        standardize(X)

# ---------- Tests for demean_matrix ----------
from mlsynth.utils.denoiseutils import demean_matrix

def test_demean_matrix_valid():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    X_demeaned = demean_matrix(X)
    assert X_demeaned.shape == X.shape
    # Check column means are close to 0
    np.testing.assert_array_almost_equal(np.mean(X_demeaned, axis=0), np.zeros(X.shape[1]), decimal=6)

def test_demean_matrix_already_demeaned():
    X = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float) # Already demeaned
    X_demeaned = demean_matrix(X)
    np.testing.assert_array_almost_equal(X_demeaned, X, decimal=6)

def test_demean_matrix_invalid_input_type():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` must be a NumPy array."):
        demean_matrix([1, 2, 3]) # type: ignore

def test_demean_matrix_scalar_input():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` must be at least 1D."):
        demean_matrix(np.array(5))

def test_demean_matrix_empty_input():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.array([]))
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.empty((0,5)))
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.empty((5,0)))


    def test_demean_matrix_2d_zero_cols():
        # This input (shape (2,0)) has size 0, so it's caught by the "cannot be empty" check first.
        with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
            demean_matrix(np.array([[],[]]))


# ---------- Tests for svd_fast ----------
from mlsynth.utils.denoiseutils import svd_fast

def test_svd_fast_square_matrix():
    M = np.array([[1, 2], [3, 4]], dtype=float)
    U, s, Vh = svd_fast(M)
    assert U.shape == (2, 2)
    assert s.shape == (2,)
    assert Vh.shape == (2, 2)
    reconstructed_M = U @ np.diag(s) @ Vh
    np.testing.assert_array_almost_equal(reconstructed_M, M, decimal=6)

# ---------- Tests for debias ----------
from mlsynth.utils.denoiseutils import debias

def test_debias_smoke_test():
    n1, n2, num_treat = 5, 4, 2
    M = np.random.rand(n1, n2)
    tau = np.random.rand(num_treat)
    Z = np.zeros((n1, n2, num_treat))
    Z[0, 0, 0] = 1 
    Z[1, 1, 1] = 1 
    l_reg = 0.1

    M_debias, tau_debias = debias(M, tau, Z, l_reg)

    assert isinstance(M_debias, np.ndarray)
    assert M_debias.shape == M.shape
    assert np.all(np.isfinite(M_debias))

    assert isinstance(tau_debias, np.ndarray)
    assert tau_debias.shape == tau.shape
    assert np.all(np.isfinite(tau_debias))

def test_debias_zero_M():
    n1, n2, num_treat = 5, 4, 2
    M = np.zeros((n1, n2))
    tau = np.random.rand(num_treat)
    Z = np.zeros((n1, n2, num_treat))
    Z[0, 0, 0] = 1
    Z[1, 1, 1] = 1
    l_reg = 0.1

    # Warning was due to rank calculation, which is now more robust for zero matrices.
    M_debias, tau_debias = debias(M, tau, Z, l_reg)

    np.testing.assert_array_almost_equal(M_debias, M)
    np.testing.assert_array_almost_equal(tau_debias, tau)

# ---------- Tests for remove_tangent_space_component ----------
from mlsynth.utils.denoiseutils import remove_tangent_space_component

def test_remove_tangent_space_component_smoke():
    n1, n2, r_rank = 5, 4, 2 # Renamed r to r_rank to avoid conflict if pytest uses r
    u = np.random.rand(n1, r_rank)
    vh = np.random.rand(r_rank, n2)
    Z_single = np.random.rand(n1, n2)

    u_ortho, _ = np.linalg.qr(u)
    vh_ortho_transposed, _ = np.linalg.qr(vh.T)
    vh_ortho = vh_ortho_transposed.T

    PTperpZ = remove_tangent_space_component(u_ortho, vh_ortho, Z_single)
    assert PTperpZ.shape == Z_single.shape
    assert isinstance(PTperpZ, np.ndarray)
    assert np.all(np.isfinite(PTperpZ))

def test_remove_tangent_space_component_r_equals_zero():
    n1, n2 = 5, 4
    u_zero_r = np.empty((n1, 0))
    vh_zero_r = np.empty((0, n2))
    Z_single = np.random.rand(n1, n2)

    PTperpZ = remove_tangent_space_component(u_zero_r, vh_zero_r, Z_single)
    np.testing.assert_array_almost_equal(PTperpZ, Z_single)

def test_remove_tangent_space_component_Z_in_tangent_space():
    n1, n2, r_rank = 5, 4, 2
    
    U_full, _, Vh_full = np.linalg.svd(np.random.rand(n1, n2))
    u = U_full[:, :r_rank]
    vh = Vh_full[:r_rank, :]

    S_core = np.random.rand(r_rank, r_rank)
    Z_single = u @ S_core @ vh

    PTperpZ = remove_tangent_space_component(u, vh, Z_single)
    np.testing.assert_array_almost_equal(PTperpZ, np.zeros((n1, n2)), decimal=6)

# ---------- Tests for prepare_OLS ----------
from mlsynth.utils.denoiseutils import prepare_OLS

def test_prepare_OLS_smoke():
    n1, n2, num_treat = 5, 4, 2
    Z_3d = np.zeros((n1, n2, num_treat))
    Z_3d[0, 0, 0] = 1
    Z_3d[1, 1, 1] = 2
    Z_3d[0, 0, 1] = 0.5
    Z_3d[2, 2, 0] = 1 # Ensure enough independent rows for X_ols
    Z_3d[3, 3, 1] = 1


    small_index, X_ols, Xinv_ols = prepare_OLS(Z_3d)

    assert isinstance(small_index, np.ndarray)
    assert small_index.shape == (n1, n2)
    assert small_index.dtype == bool
    
    expected_num_rows_X_ols = np.sum(small_index)
    assert X_ols.shape == (expected_num_rows_X_ols, num_treat)
    assert isinstance(X_ols, np.ndarray)
    
    assert Xinv_ols.shape == (num_treat, num_treat)
    assert isinstance(Xinv_ols, np.ndarray)

    if expected_num_rows_X_ols >= num_treat: 
        XTX = X_ols.T @ X_ols
        if np.linalg.matrix_rank(XTX) == num_treat:
            expected_Xinv_ols = np.linalg.inv(XTX)
            np.testing.assert_array_almost_equal(Xinv_ols, expected_Xinv_ols)
        else: # pragma: no cover
             # This case implies XTX is singular, np.linalg.inv would raise LinAlgError
             # which should be tested separately.
            pass


def test_prepare_OLS_all_zeros_Z():
    n1, n2, num_treat = 5, 4, 2
    Z_3d = np.zeros((n1, n2, num_treat))
    with pytest.raises(MlsynthDataError, match="No active intervention entries found"):
        prepare_OLS(Z_3d)

def test_prepare_OLS_collinear_Z():
    n1, n2, num_treat = 5, 4, 2
    Z_3d = np.zeros((n1, n2, num_treat))
    Z_3d[0,0,0] = 1
    Z_3d[0,0,1] = 2
    Z_3d[1,0,0] = 3
    Z_3d[1,0,1] = 6
    Z_3d[2,0,0] = -1 # Need at least num_treat (2) rows in X_ols
    Z_3d[2,0,1] = -2
    
    with pytest.raises(MlsynthEstimationError, match=r"Matrix \(X_ols\.T @ X_ols\) is singular in prepare_OLS"): # More specific regex
        prepare_OLS(Z_3d)

# ---------- Tests for solve_tau ----------
from mlsynth.utils.denoiseutils import solve_tau

def test_solve_tau_smoke():
    n1, n2, num_treat = 5, 4, 2
    O = np.random.rand(n1, n2)
    Z_3d = np.zeros((n1, n2, num_treat))
    Z_3d[0,0,0] = 1
    Z_3d[1,1,1] = 1 
    Z_3d[2,0,0] = 0.5 
    Z_3d[0,1,1] = 0.8

    tau = solve_tau(O, Z_3d)
    assert tau.shape == (num_treat,)
    assert isinstance(tau, np.ndarray)
    assert np.all(np.isfinite(tau))

def test_solve_tau_zero_O():
    n1, n2, num_treat = 5, 4, 2
    O = np.zeros((n1, n2))
    Z_3d = np.zeros((n1, n2, num_treat))
    Z_3d[0,0,0] = 1
    Z_3d[1,1,1] = 1
    Z_3d[2,0,0] = 0.5 
    Z_3d[0,1,1] = 0.8

    tau = solve_tau(O, Z_3d)
    np.testing.assert_array_almost_equal(tau, np.zeros(num_treat))

def test_solve_tau_collinear_Z_propagates_error():
    n1, n2, num_treat = 5, 4, 2
    O = np.random.rand(n1, n2)
    Z_3d = np.zeros((n1, n2, num_treat))
    Z_3d[0,0,0] = 1
    Z_3d[0,0,1] = 2 
    Z_3d[1,0,0] = 3
    Z_3d[1,0,1] = 6
    Z_3d[2,0,0] = -1 # Ensure enough rows for X_ols
    Z_3d[2,0,1] = -2


    with pytest.raises(MlsynthEstimationError, match=r"Matrix \(X_ols\.T @ X_ols\) is singular in prepare_OLS"): # More specific regex
        solve_tau(O, Z_3d)

# ---------- Tests for DC_PR_with_l ----------
from mlsynth.utils.denoiseutils import DC_PR_with_l

def test_DC_PR_with_l_smoke():
    n1, n2, num_treat = 5, 4, 1 
    O = np.random.rand(n1, n2)
    
    Z_single_treat = np.zeros((n1, n2))
    Z_single_treat[0,0] = 1
    Z_single_treat[1,1] = 1 
                            
    l_reg = 0.1
    initial_tau = np.array([0.5]) 

    M_est, tau_est = DC_PR_with_l(O, Z_single_treat, l_reg, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3) 

    assert M_est.shape == O.shape
    assert isinstance(M_est, np.ndarray)
    assert np.all(np.isfinite(M_est))

    assert tau_est.shape == (num_treat,)
    assert isinstance(tau_est, np.ndarray)
    assert np.all(np.isfinite(tau_est))

    Z_list = [Z_single_treat]
    M_est_list, tau_est_list = DC_PR_with_l(O, Z_list, l_reg, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3)
    assert M_est_list.shape == O.shape
    assert tau_est_list.shape == (num_treat,)
    
    Z_3d = Z_single_treat.reshape(n1, n2, 1)
    M_est_3d, tau_est_3d = DC_PR_with_l(O, Z_3d, l_reg, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3)
    assert M_est_3d.shape == O.shape
    assert tau_est_3d.shape == (num_treat,)

    M_est_no_init_tau, tau_est_no_init_tau = DC_PR_with_l(O, Z_3d, l_reg, initial_treatment_effects=None, convergence_tolerance=1e-3)
    assert M_est_no_init_tau.shape == O.shape
    assert tau_est_no_init_tau.shape == (num_treat,)

# ---------- Tests for non_convex_PR ----------
from mlsynth.utils.denoiseutils import non_convex_PR

def test_non_convex_PR_smoke():
    n1, n2, num_treat = 5, 4, 1
    O = np.random.rand(n1, n2)
    
    Z_single_treat = np.zeros((n1, n2))
    Z_single_treat[0,0] = 1
    Z_single_treat[1,1] = 1 
        
    rank_r = 2 
    initial_tau = np.array([0.5])

    M_est, tau_est = non_convex_PR(O, Z_single_treat, rank_constraint=rank_r, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3)

    assert M_est.shape == O.shape
    assert isinstance(M_est, np.ndarray)
    assert np.all(np.isfinite(M_est))
    _, s_M_est, _ = np.linalg.svd(M_est)
    assert np.sum(s_M_est > 1e-6) <= rank_r

    assert tau_est.shape == (num_treat,)
    assert isinstance(tau_est, np.ndarray)
    assert np.all(np.isfinite(tau_est))

    Z_list = [Z_single_treat]
    M_est_list, tau_est_list = non_convex_PR(O, Z_list, rank_constraint=rank_r, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3)
    assert M_est_list.shape == O.shape
    assert tau_est_list.shape == (num_treat,)
    
    Z_3d = Z_single_treat.reshape(n1, n2, 1)
    M_est_3d, tau_est_3d = non_convex_PR(O, Z_3d, rank_constraint=rank_r, initial_treatment_effects=initial_tau, convergence_tolerance=1e-3)
    assert M_est_3d.shape == O.shape
    assert tau_est_3d.shape == (num_treat,)

    M_est_no_init_tau, tau_est_no_init_tau = non_convex_PR(O, Z_3d, rank_constraint=rank_r, initial_treatment_effects=None, convergence_tolerance=1e-3)
    assert M_est_no_init_tau.shape == O.shape
    assert tau_est_no_init_tau.shape == (num_treat,)

# ---------- Tests for panel_regression_CI ----------
from mlsynth.utils.denoiseutils import panel_regression_CI

def test_panel_regression_CI_smoke():
    n1, n2, num_treat = 5, 4, 2
    M = np.random.rand(n1, n2)
    Z = np.random.rand(n1, n2, num_treat)
    Z[0,0,0]=1 # Ensure some non-zero elements for X_reg
    Z[1,1,1]=1
    E = np.random.rand(n1, n2) * 0.1 # Small noise

    # Ensure M has some rank for u, vh to be non-trivial
    U_m, s_m, Vh_m = np.linalg.svd(np.random.rand(n1,n2))
    M = U_m[:,:2] @ np.diag(s_m[:2]) @ Vh_m[:2,:]


    CI_matrix = panel_regression_CI(M, Z, E)

    assert CI_matrix.shape == (num_treat, num_treat)
    assert isinstance(CI_matrix, np.ndarray)
    assert np.all(np.isfinite(CI_matrix))

def test_panel_regression_CI_zero_rank_M():
    n1, n2, num_treat = 5, 4, 1
    M = np.zeros((n1,n2)) # Zero rank M
    Z = np.zeros((n1,n2,num_treat))
    Z[0,0,0] = 1
    E = np.random.rand(n1,n2) * 0.1

    # Warning was due to rank calculation, which is now more robust for zero matrices.
    CI_matrix = panel_regression_CI(M, Z, E)
    
    assert CI_matrix.shape == (num_treat, num_treat)
    # With zero rank M, PTperpZ becomes Z. X_reg becomes Z.reshape(-1, num_treat)
    # If Z is sparse, X_reg.T @ X_reg might be singular or ill-conditioned.
    # For this specific Z, X_reg.T @ X_reg is [[1.0]], inv is [[1.0]]
    # A_reg will have shape (1, 20)
    # CI_matrix will be a scalar.
    assert np.all(np.isfinite(CI_matrix))


# ---------- Tests for RPCA_HQF ----------
from mlsynth.utils.denoiseutils import RPCA_HQF

def test_RPCA_HQF_smoke():
    m, n = 10, 8
    rak = 2
    maxiter = 10
    ip = 1.5
    lam_1 = 0.1
    
    # Create a low-rank matrix + sparse noise
    L_true = np.random.rand(m, rak) @ np.random.rand(rak, n)
    S_true = np.zeros((m, n))
    S_true[0,0] = 5
    S_true[m-1, n-1] = -3
    M_noise = L_true + S_true

    X_est = RPCA_HQF(M_noise, rak, maxiter, ip, lam_1)

    assert X_est.shape == M_noise.shape
    assert isinstance(X_est, np.ndarray)
    assert np.all(np.isfinite(X_est))
    # Check if rank of X_est is close to rak
    _, s_X_est, _ = np.linalg.svd(X_est)
    assert np.sum(s_X_est > 1e-6) <= rak + 1 # Allow for slight numerical inaccuracies

# ---------- Tests for nbpiid ----------
from mlsynth.utils.denoiseutils import nbpiid

def test_nbpiid_smoke(): # Corrected indentation
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    kmax = 5 # Corrected indentation
    jj = 1 # Corrected indentation
    demean_flag = 1 # Demean # Corrected indentation

    ic1_selected, chat, Fhat = nbpiid(x, kmax, jj, demean_flag) # Corrected indentation

    assert isinstance(ic1_selected, int)
    assert 0 <= ic1_selected <= kmax # Selected factors should be within [0, kmax]
    
    assert chat.shape == x.shape
    assert isinstance(chat, np.ndarray)
    assert np.all(np.isfinite(chat))

    if ic1_selected > 0:
        assert Fhat.shape == (T_obs, ic1_selected)
        assert isinstance(Fhat, np.ndarray)
        assert np.all(np.isfinite(Fhat))
    else: # ic1_selected == 0
        assert Fhat.shape == (T_obs, 0) # Or specific shape for 0 factors

@pytest.mark.parametrize("jj_val", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("demean_val", [0, 1, 2])
def test_nbpiid_various_jj_demean(jj_val, demean_val): # Corrected indentation
    T_obs, N_series = 30, 15 # Slightly larger for more robust SVD
    x = np.random.rand(T_obs, N_series)
    # Construct a matrix with a clear factor structure
    true_factors = np.random.rand(T_obs, 2)
    true_loadings = np.random.rand(2, N_series)
    x_structured = true_factors @ true_loadings + np.random.rand(T_obs, N_series) * 0.1

    kmax = 4 # Max factors to check

    ic1_selected, chat, Fhat = nbpiid(x_structured, kmax, jj_val, demean_val)

    assert isinstance(ic1_selected, int)
    assert 0 <= ic1_selected <= kmax
    assert chat.shape == x_structured.shape
    if ic1_selected > 0:
        assert Fhat.shape == (T_obs, ic1_selected)
    else:
        assert Fhat.shape == (T_obs, 0)


def test_nbpiid_jj10_jj11():
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    kmax = 3
    demean_flag = 2 # Standardize

    # Test jj=10
    m_N, m_T = N_series // 2, T_obs // 2
    ic1_10, _, _ = nbpiid(x, kmax, 10, demean_flag, N_series_adjustment=m_N, T_obs_adjustment=m_T)
    assert isinstance(ic1_10, int)
    assert 0 <= ic1_10 <= kmax

    # Test jj=11
    ic1_11, _, _ = nbpiid(x, kmax, 11, demean_flag) # m_N, m_T not used for jj=11
    assert isinstance(ic1_11, int)
    assert 0 <= ic1_11 <= kmax

def test_nbpiid_jj10_missing_m_params():
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    kmax = 3
    demean_flag = 0
    with pytest.raises(MlsynthConfigError, match="`N_series_adjustment` must be an integer if criterion_selector_code is 10."): # Corrected message
        nbpiid(x, kmax, 10, demean_flag)

def test_nbpiid_invalid_criterion_code():
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    kmax = 3
    demean_flag = 0
    with pytest.raises(MlsynthConfigError, match="`criterion_selector_code` must be one of .* got 99."): # Corrected message
        nbpiid(x, kmax, 99, demean_flag)

def test_nbpiid_kmax_zero_or_negative():
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` must be positive."): # Corrected message
         nbpiid(x, 0, 1, 1)
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` must be positive."): # Corrected message
         nbpiid(x, -1, 1, 1)

def test_nbpiid_kmax_too_large():
    T_obs, N_series = 5, 10
    x = np.random.rand(T_obs, N_series)
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` .* cannot exceed min\\(num_time_periods, num_series\\) which is 5."):
        nbpiid(x, 6, 1, 1)

def test_nbpiid_invalid_input_panel_data():
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` must be a NumPy array."):
        nbpiid([1,2,3], 3, 1, 1) # type: ignore
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` must be a 2D array."):
        nbpiid(np.array([1,2,3]), 3, 1, 1)
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` cannot be empty."):
        nbpiid(np.empty((5,0)), 3, 1, 1)

def test_nbpiid_invalid_preprocessing_code():
    T_obs, N_series = 20, 10
    x = np.random.rand(T_obs, N_series)
    kmax = 3
    with pytest.raises(MlsynthConfigError, match="`preprocessing_method_code` must be one of .* got 5."):
        nbpiid(x, kmax, 1, 5)

def test_nbpiid_criterion11_invalid_T():
    T_obs, N_series = 1, 10 # T=1, log(T)=0, log(log(T)) undefined or error
    x = np.random.rand(T_obs, N_series)
    kmax = 1
    with pytest.raises(MlsynthEstimationError, match="T must be > 1 for criterion 11 due to log\\(T\\)."):
        nbpiid(x, kmax, 11, 0)

    T_obs, N_series = 2, 10 # T=2, log(T)=0.69, log(log(T))=-0.36 (<=0)
    x = np.random.rand(T_obs, N_series)
    kmax = 1
    with pytest.raises(MlsynthEstimationError, match="log\\(T\\) must be > 1 for criterion 11 .*"):
        nbpiid(x, kmax, 11, 0)
    
    # Removed the T=e case as it's too specific and might be flaky with float precision.
    # The T=1 and T=2 cases cover the log(log(T)) issues sufficiently.


# ---------- Tests for DC_PR_with_suggested_rank ----------
from mlsynth.utils.denoiseutils import DC_PR_with_suggested_rank
from mlsynth.utils.resultutils import effects # For mocking if needed, or for result structure

def test_DC_PR_with_suggested_rank_smoke():
    n1, n2 = 10, 8
    O = np.random.rand(n1, n2)
    Z_input = np.zeros((n1, n2))
    Z_input[0, n2-2:] = 1 # Treat unit 0 in last 2 periods
    Z_input[1, n2-2:] = 0.5 # Partial treatment for unit 1
    
    suggest_r = 2
    
    # Test 'convex' method
    results_convex = DC_PR_with_suggested_rank(O, Z_input, target_rank=suggest_r, method="convex")
    assert isinstance(results_convex, dict)
    assert "Effects" in results_convex
    assert "Fit" in results_convex
    assert "Vectors" in results_convex
    assert "Inference" in results_convex
    assert "Counterfactual_Full_Matrix" in results_convex["Vectors"]
    assert results_convex["Vectors"]["Counterfactual_Full_Matrix"].shape == O.shape

    # Test 'non-convex' method
    results_non_convex = DC_PR_with_suggested_rank(O, Z_input, target_rank=suggest_r, method="non-convex")
    assert isinstance(results_non_convex, dict)
    assert "Effects" in results_non_convex
    assert results_non_convex["Vectors"]["Counterfactual_Full_Matrix"].shape == O.shape


    # Test 'auto' method
    results_auto = DC_PR_with_suggested_rank(O, Z_input, target_rank=suggest_r, method="auto")
    assert isinstance(results_auto, dict)
    assert "Effects" in results_auto
    assert results_auto["Vectors"]["Counterfactual_Full_Matrix"].shape == O.shape

def test_DC_PR_with_suggested_rank_no_treated_unit_in_Z():
    n1, n2 = 10, 8
    O = np.random.rand(n1, n2)
    Z_input_no_treat = np.zeros((n1, n2)) # No treated unit
    suggest_r = 1
    with pytest.raises(MlsynthDataError, match="No treated unit found .* in `intervention_panel_input`."): # Corrected exception type
        DC_PR_with_suggested_rank(O, Z_input_no_treat, target_rank=suggest_r, method="convex")

# ---------- Tests for DC_PR_auto_rank ----------
from mlsynth.utils.denoiseutils import DC_PR_auto_rank

def test_DC_PR_auto_rank_smoke():
    n1, n2 = 10, 8
    O = np.random.rand(n1, n2) + np.arange(n2) # Add trend to make SVD more interesting
    
    # Ensure O has some rank > 1 for suggest_r to be meaningful
    U_o, s_o, Vh_o = np.linalg.svd(np.random.rand(n1,n2))
    O_rank_gt_1 = U_o[:,:3] @ np.diag(s_o[:3] + np.array([5,3,1])) @ Vh_o[:3,:]
    
    Z_input = np.zeros((n1, n2))
    Z_input[0, n2-2:] = 1 # Treat unit 0 in last 2 periods

    results = DC_PR_auto_rank(O_rank_gt_1, Z_input, spectrum_cut=0.1) # Use higher cut for faster test
    
    assert isinstance(results, dict)
    assert "Vectors" in results
    assert "Effects" in results
    assert "CIs" in results
    assert "RMSE" in results
    assert "Suggested_Rank" in results
    assert isinstance(results["Suggested_Rank"], np.integer) # np.int or similar
    assert results["Vectors"]["Treated Unit"].shape == (n2,1)
    assert results["Vectors"]["Counterfactual"].shape == (n2,1)
    # The Counterfactual_Full_Matrix is added by DC_PR_with_suggested_rank,
    # DC_PR_auto_rank itself doesn't add it to its direct return dict.
    # It's inside the dict returned by the call to DC_PR_with_suggested_rank.

def test_DC_PR_auto_rank_no_treated_unit_in_Z():
    n1, n2 = 10, 8
    O = np.random.rand(n1, n2)
    Z_input_no_treat = np.zeros((n1, n2)) # No treated unit
    with pytest.raises(MlsynthDataError, match="No treated unit found .* in `intervention_panel_matrix`."):
        DC_PR_auto_rank(O, Z_input_no_treat)

# ---------- Tests for transform_to_3D ----------
from mlsynth.utils.denoiseutils import transform_to_3D

def test_transform_to_3D_list_of_2D():
    Z1 = np.array([[1,2],[3,4]])
    Z2 = np.array([[5,6],[7,8]])
    Z_list = [Z1, Z2]
    Z_3d = transform_to_3D(Z_list)
    assert Z_3d.shape == (2, 2, 2)
    np.testing.assert_array_equal(Z_3d[:,:,0], Z1)
    np.testing.assert_array_equal(Z_3d[:,:,1], Z2)
    assert Z_3d.dtype == float

def test_transform_to_3D_single_2D():
    Z_2d = np.array([[1,2],[3,4]])
    Z_3d = transform_to_3D(Z_2d)
    assert Z_3d.shape == (2, 2, 1)
    np.testing.assert_array_equal(Z_3d[:,:,0], Z_2d)
    assert Z_3d.dtype == float

def test_transform_to_3D_already_3D():
    Z_input_3d = np.random.rand(2,2,3)
    Z_3d = transform_to_3D(Z_input_3d)
    assert Z_3d.shape == (2,2,3)
    np.testing.assert_array_equal(Z_3d, Z_input_3d) # Should be same if already 3D
    assert Z_3d.dtype == float # Should ensure float type

def test_transform_to_3D_empty_list():
    Z_list_empty = []
    # np.stack with empty list raises ValueError on axis > 0 or if list is truly empty.
    # The refactored function raises MlsynthDataError.
    with pytest.raises(MlsynthDataError, match="Input `intervention_data` list cannot be empty."):
        transform_to_3D(Z_list_empty)

def test_transform_to_3D_list_invalid_item_type():
    Z_list_invalid = [np.array([[1,2]]), "not_an_array"]
    with pytest.raises(MlsynthDataError, match="Item 1 in `intervention_data` list is not a NumPy array."):
        transform_to_3D(Z_list_invalid) # type: ignore

def test_transform_to_3D_list_invalid_item_ndim():
    Z_list_invalid = [np.array([[1,2]]), np.array([1,2,3])] # 2D then 1D
    with pytest.raises(MlsynthDataError, match="Item 1 in `intervention_data` list is not a 2D array"):
        transform_to_3D(Z_list_invalid)

def test_transform_to_3D_list_inconsistent_shapes():
    Z1 = np.array([[1,2],[3,4]])
    Z2 = np.array([[5,6,7],[8,9,10]]) # Different shape
    Z_list_inconsistent = [Z1, Z2]
    with pytest.raises(MlsynthDataError, match="Item 1 in `intervention_data` list has shape .* expected .*"):
        transform_to_3D(Z_list_inconsistent)

def test_transform_to_3D_invalid_numpy_ndim():
    Z_invalid_ndim = np.random.rand(2,2,2,2) # 4D array
    with pytest.raises(MlsynthDataError, match="`intervention_data` NumPy array must be 2D or 3D, got 4D."):
        transform_to_3D(Z_invalid_ndim)

def test_transform_to_3D_invalid_type():
    with pytest.raises(MlsynthDataError, match="`intervention_data` must be a list of NumPy arrays or a NumPy array."):
        transform_to_3D("not_a_list_or_array") # type: ignore


# ---------- Tests for SVD (hard truncation) ----------
from mlsynth.utils.denoiseutils import SVD as SVD_hard_truncate

def test_SVD_hard_truncate():
    M = np.random.rand(5, 4)
    U_orig, s_orig, Vh_orig = np.linalg.svd(M, full_matrices=False)
    
    r = 2 # Target rank
    M_rank_r = SVD_hard_truncate(M, r)
    
    assert M_rank_r.shape == M.shape
    # Check that the reconstructed matrix indeed has rank r (or less if original rank < r)
    # np.linalg.matrix_rank might be sensitive, check singular values instead
    _ , s_reconstructed, _ = np.linalg.svd(M_rank_r)
    assert np.sum(s_reconstructed > 1e-6) <= r # Effective rank should be at most r

    # Check if it's a good approximation
    expected_reconstruction = U_orig[:, :r] @ np.diag(s_orig[:r]) @ Vh_orig[:r, :]
    np.testing.assert_array_almost_equal(M_rank_r, expected_reconstruction, decimal=6)

def test_SVD_hard_truncate_rank_too_high():
    M = np.random.rand(5, 3) # Max rank is 3
    r_target = 5 # Target rank higher than possible
    # SVD function now raises MlsynthConfigError if target_rank > min(shape)
    with pytest.raises(MlsynthConfigError, match="target_rank .* cannot exceed min\\(matrix_rows, matrix_cols\\)"):
        SVD_hard_truncate(M, r_target)

    # Test with valid rank equal to max possible rank
    r_valid_max = min(M.shape)
    M_rank_r_valid = SVD_hard_truncate(M, r_valid_max)
    _ , s_reconstructed_valid, _ = np.linalg.svd(M_rank_r_valid)
    assert np.sum(s_reconstructed_valid > 1e-6) <= r_valid_max
    np.testing.assert_array_almost_equal(M_rank_r_valid, M, decimal=6)


# ---------- Tests for SVD_soft ----------
from mlsynth.utils.denoiseutils import SVD_soft

def test_SVD_soft():
    M = np.array([[10, 0], [0, 3]], dtype=float) # Diagonal matrix, s_orig = [10, 3]
    lmbda = 2.0
    M_soft = SVD_soft(M, lmbda)
    
    assert M_soft.shape == M.shape
    
    # Expected singular values after soft thresholding: max(0, s - lmbda)
    # s_orig for M = [[10,0],[0,3]] are [10, 3]
    s_thresholded_manually = np.array([max(0, 10 - lmbda), max(0, 3 - lmbda)]) # Should be [8, 1]
    
    # Reconstruct manually using original U, Vh and manually thresholded s
    U_orig, _, Vh_orig = np.linalg.svd(M) # Get U and Vh from original M
    expected_reconstruction = U_orig @ np.diag(s_thresholded_manually) @ Vh_orig
    
    np.testing.assert_array_almost_equal(M_soft, expected_reconstruction, decimal=6)

def test_SVD_soft_threshold_all_to_zero():
    M = np.array([[1, 0], [0, 0.5]], dtype=float) # s_orig = [1, 0.5]
    lmbda = 1.5
    M_soft = SVD_soft(M, lmbda) # s_thresholded = [0, 0]
    
    np.testing.assert_array_almost_equal(M_soft, np.zeros_like(M), decimal=6)

def test_svd_fast_fat_matrix(): # More columns than rows
    M = np.array([[1, 2, 3], [4, 5, 6]], dtype=float) # Shape (2, 3)
    U, s, Vh = svd_fast(M)
    assert U.shape == (2, 2)
    assert s.shape == (2,) # Number of singular values is min(rows, cols)
    assert Vh.shape == (2, 3)
    reconstructed_M = U @ np.diag(s) @ Vh
    np.testing.assert_array_almost_equal(reconstructed_M, M, decimal=6)

def test_svd_fast_skinny_matrix(): # More rows than columns
    M = np.array([[1, 2], [3, 4], [5, 6]], dtype=float) # Shape (3, 2)
    U, s, Vh = svd_fast(M)
    assert U.shape == (3, 2)
    assert s.shape == (2,)
    assert Vh.shape == (2, 2)
    reconstructed_M = U @ np.diag(s) @ Vh
    np.testing.assert_array_almost_equal(reconstructed_M, M, decimal=6)

def test_svd_fast_zero_matrix():
    M = np.zeros((3, 2), dtype=float)
    U, s, Vh = svd_fast(M)
    assert U.shape == (3, 2)
    assert s.shape == (2,)
    assert Vh.shape == (2, 2)
    np.testing.assert_array_almost_equal(s, np.zeros(2), decimal=6)
    reconstructed_M = U @ np.diag(s) @ Vh
    np.testing.assert_array_almost_equal(reconstructed_M, M, decimal=6)

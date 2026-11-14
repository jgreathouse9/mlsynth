"""The unregularised PCR weights are the truncated SVD solution at the HSVT rank.

ClusterSC's Algorithm 2 Step 3 (Rho et al. 2025), and Robust Synthetic Control
at ``eta = 0`` (Amjad, Shah & Shen 2018), regress the treated pre-period ``x`` on
the rank-``r`` HSVT reconstruction ``M`` of the donor pre-period and take the
minimum-norm solution

    f = sum_{i <= r} (u_i' x / s_i) v_i,

the truncated SVD (TSVD) solution of Golub & Van Loan (2013, Sec. 5.5.4) and
Bjorck (1996, Sec. 2.7.2).

``M`` has rank ``r`` in exact arithmetic only. Formed in floating point it is
``M + E``, with ``E`` of the size of the rounding (up to 18 eps * s_1 measured on
random panels), and ``M + E`` generically has full rank. The pseudoinverse is not
continuous where the rank changes:

    if rank(A + E) != rank(A), then ||(A + E)^+ - A^+||_2 >= 1 / ||E||_2

Wedin (1973) proves it: matrices in the acute case have equal rank (Lemma 7.5,
p. 229), so a change of rank leaves the acute case, and outside it
``||B^+ - A^+||_2 >= 1 / ||B - A||_2`` (Theorem 7.2, p. 229); his Section 3
(p. 219) gives the two-by-two example. The statement is Bjorck (1996, Sec.
1.4.2, Theorem 1.4.1, p. 26) and Golub & Van Loan (2013, Sec. 5.5.3). The
upper side is of the same order: for any ``E``,
``||B^+ - A^+||_F <= max(||A^+||_2^2, ||B^+||_2^2) ||E||_F`` (Meng & Zheng 2010,
Theorem 2.1, p. 958), and a kept rounding direction makes
``||B^+||_2 = 1 / s_{r+1} ~ 1 / ||E||``. When the rank does not change the
perturbation is bounded by ``||A^+||_2 ||B^+||_2 ||E||_F`` instead (Meng & Zheng
2010, Theorem 2.2, p. 960; Wedin 1973, Theorem 4.1; Bjorck 2015, Theorem
2.2.5), which is why holding the rank at ``r`` is stable. Bjorck (1996, Sec.
2.7) draws the consequence these tests
hold the solver to: the rank of A has to be determined explicitly before its
pseudoinverse is computed numerically, and overestimating it gives a solution
of very large norm, because components belonging to numerically zero singular
values are included. The tolerance that decides a numerical rank has to reflect
the error level in the matrix (Bjorck 1996, Definition 2.7.1; Bjorck 2015,
Sec. 2.4.1, Definition 2.4.1; Golub & Van Loan 2013, Sec. 5.4.1, eq. 5.4.5).

``solve_ols`` used to return ``numpy.linalg.pinv`` of the reconstruction times
``x``. That decides the rank a second time, at numpy's default tolerance of
``1e-15 * s_1`` (about 4.5 eps * s_1), which is below the rounding of a
reconstructed matrix. On ``rsc_rank_condition_mc``'s panel (1600 pre-periods, 99
donors, ``r = 25``) it kept a rounding direction at ``1.27e-15 * s_1`` and the
weights reached 1.9e12, which is Wedin's ``1 / ||E||`` times the target's
component along that direction. Whether a rounding direction clears ``1e-15``
depends on the BLAS kernel, so the same commit passed on CI and failed on an
AVX-512 machine.

The authors' implementation (``jehangiramjad/tslib`` at ``3e50bc1``:
``SVDModel._computeWeights`` with ``tsUtils.pInverseMatrixFromSVD``) builds the
pseudoinverse from the top-``r`` singular factors, so the rank is the one the
method fixed. These tests hold ``solve_ols`` to that construction.

References
----------
Bjorck, A. (1996). *Numerical Methods for Least Squares Problems*. SIAM.

Bjorck, A. (2015). *Numerical Methods in Matrix Computations*. Texts in Applied
Mathematics 59, Springer.

Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns
Hopkins University Press.

Meng, L. & Zheng, B. (2010). The optimal perturbation bounds of the
Moore-Penrose inverse under the Frobenius norm. *Linear Algebra and its
Applications* 432, 956-963. doi:10.1016/j.laa.2009.10.009.

Wedin, P.-A. (1973). Perturbation theory for pseudo-inverses. *BIT* 13, 217-232.
doi:10.1007/BF01933494.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.pcr.frequentist import solve_ols
from mlsynth.utils.clustersc_helpers.pcr.pipeline import run_pcr
from mlsynth.utils.clustersc_helpers.simulation import simulate_rsc_panel
from mlsynth.utils.pcr.core import hsvt

EPS = np.finfo(float).eps

# numpy.linalg.pinv's default tolerance, relative to the largest singular value.
NUMPY_PINV_RCOND = 1e-15


def _orthonormal(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return q


def _design(rng, T0, J, head, tail=()):
    """A ``(T0, J)`` design with known singular triples: ``head``, then ``tail``."""
    s = np.concatenate([np.asarray(head, float), np.asarray(tail, float)])
    U, V = _orthonormal(rng, T0, s.size), _orthonormal(rng, J, s.size)
    return (U * s) @ V.T, U, s, V


def _tsvd(U, s, V, x, r):
    """Golub & Van Loan's ``x_r`` (Sec. 5.5.4), from known factors."""
    return V[:, :r] @ ((U[:, :r].T @ x) / s[:r])


def _tsvd_of(X, x, r):
    """The same solution from the SVD of the raw design ``X``."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return _tsvd(U, s, Vt.T, x, r)


def _assert_close(got, want, rel=1e-9):
    gap = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert gap <= rel, (
        f"weights of norm {np.linalg.norm(got):.3e} against the TSVD solution's "
        f"{np.linalg.norm(want):.3e}: relative gap {gap:.1e} > {rel:.0e}")


@pytest.fixture(scope="module")
def rsc_cell():
    """``rsc_rank_condition_mc``'s failing cell: target 0, 1600 pre-periods, 99 donors."""
    panel = simulate_rsc_panel(N=100, T=2000, T0=1600, noise=1.0, seed=0)
    Y = panel.observed
    return Y[1:].T, Y[0], panel.T0


# ---------------------------------------------------------------------------
# Wedin's 1/||E||, planted where no BLAS kernel can move it
# ---------------------------------------------------------------------------

def test_the_mechanism_is_bounded_on_both_sides():
    """numpy's pinv of a rank-changing perturbation moves by order ``1 / ||E||``.

    ``A`` has rank 5; ``B = A + E`` adds a sixth singular value at
    ``1e-14 * s_1``, which numpy's default tolerance keeps. ``A`` and ``B`` are
    not in the acute case (Wedin 1973, Lemma 7.5), so
    ``||B^+ - A^+||_2 >= 1 / ||E||_2`` (Theorem 7.2), and for any ``E``
    ``||B^+ - A^+||_F <= max(||A^+||_2^2, ||B^+||_2^2) ||E||_F`` (Meng & Zheng
    2010, Theorem 2.1). The pseudo-inverse built from ``B``'s top five factors
    keeps the rank, so the equal-rank bound (Meng & Zheng 2010, Theorem 2.2)
    leaves it within rounding of ``A^+``. The slack of a tenth allows for the
    rounding of a pseudo-inverse whose condition number is 1e14.
    """
    rng = np.random.default_rng(12)
    head = np.array([10.0, 6.0, 4.0, 2.5, 1.5])
    U, V = _orthonormal(rng, 400, 6), _orthonormal(rng, 60, 6)
    A = (U[:, :5] * head) @ V[:, :5].T
    A_pinv = (V[:, :5] / head) @ U[:, :5].T
    B = A + (1e-14 * head[0]) * np.outer(U[:, 5], V[:, 5])
    E = B - A
    P = np.linalg.pinv(B)
    assert np.linalg.norm(P - A_pinv, 2) >= 0.9 / np.linalg.norm(E, 2)
    upper = max(np.linalg.norm(A_pinv, 2), np.linalg.norm(P, 2)) ** 2
    assert (np.linalg.norm(P - A_pinv, "fro")
            <= 1.1 * upper * np.linalg.norm(E, "fro"))
    Ub, sb, Vbt = np.linalg.svd(B, full_matrices=False)
    P5 = (Vbt[:5].T / sb[:5]) @ Ub[:, :5].T
    assert (np.linalg.norm(P5 - A_pinv, "fro")
            <= 1e-10 * np.linalg.norm(A_pinv, "fro"))


def test_a_direction_under_the_rounding_floor_is_not_inverted():
    """A sixth singular value at ``1e-14 * s_1`` carries no information.

    It sits ten times above numpy's default pinv tolerance and under the
    rounding floor ``max(T0, J) * eps * s_1 = 8.9e-14 * s_1`` of a 400 x 60
    design, with room on both sides for the few ``eps * s_1`` that forming the
    design adds. Inverting it adds ``(u_6' x / s_6) v_6``, of norm about 1e13
    (Wedin 1973, Theorem 7.2; Bjorck 1996, Theorem 1.4.1).
    """
    rng = np.random.default_rng(0)
    head = [10.0, 6.0, 4.0, 2.5, 1.5]
    A, U, s, V = _design(rng, 400, 60, head, [1e-14 * head[0]])
    x = rng.standard_normal(400)
    _assert_close(solve_ols(A, x), _tsvd(U, s, V, x, 5))


@pytest.mark.parametrize("tail", [1e-15, 1e-12, 1e-6])
def test_given_the_rank_only_the_top_r_factors_enter(tail):
    """With ``rank``, the solution is truncated at ``r``, whatever lies below.

    The pipeline passes HSVT's rank, so no tolerance decides it a second time
    (Bjorck 1996, Sec. 2.7). A trailing value of ``1e-6 * s_1`` is far above any
    rounding floor and is still excluded, because the caller fixed the rank.
    """
    rng = np.random.default_rng(1)
    head = [10.0, 6.0, 4.0]
    A, U, s, V = _design(rng, 120, 30, head, [tail * 10.0, tail * 5.0])
    x = rng.standard_normal(120)
    _assert_close(solve_ols(A, x, rank=3), _tsvd(U, s, V, x, 3))


# ---------------------------------------------------------------------------
# The incident
# ---------------------------------------------------------------------------

def test_the_rsc_rank_condition_cell(rsc_cell):
    """At ``r = 25`` pinv kept a direction at ``1.27e-15 * s_1`` and returned 1.9e12.

    That was on OpenBLAS's SkylakeX kernel; its Haswell kernel happens to leave
    the same direction under ``1e-15``. The TSVD solution depends on neither.
    """
    X, y, T0 = rsc_cell
    want = _tsvd_of(X[:T0], y[:T0], 25)
    M = hsvt(X[:T0], rank=25)[0]
    _assert_close(solve_ols(M, y[:T0]), want, rel=1e-8)
    _assert_close(solve_ols(M, y[:T0], rank=25), want, rel=1e-8)
    assert np.linalg.norm(want) < 1.0


def test_the_pipeline_returns_the_tsvd_weights(rsc_cell):
    """``run_pcr`` passes HSVT's rank through to the weight solve."""
    X, y, T0 = rsc_cell
    names = [f"d{j}" for j in range(X.shape[1])]
    fit, _ = run_pcr(y, X, names, T0, rank=25, project_denoised=True,
                     compute_shen_ci=False)
    w = np.array([fit.donor_weights[n] for n in names])
    _assert_close(w, _tsvd_of(X[:T0], y[:T0], 25), rel=1e-8)
    assert np.all(np.isfinite(fit.counterfactual))
    assert fit.pre_rmse < 2.0


def test_smoke_clustersc_pcr_ols_returns_the_tsvd_weights():
    """End to end through ``CLUSTERSC`` and ``dataprep``, as the RSC case calls it."""
    from mlsynth import CLUSTERSC

    panel = simulate_rsc_panel(N=20, T=80, T0=60, noise=0.5, seed=3)
    X, T0 = panel.observed, panel.T0
    wide = pd.DataFrame(X.T)
    long = (wide.reset_index()
            .melt(id_vars="index", var_name="unit", value_name="y")
            .rename(columns={"index": "time"}))
    long["treat"] = ((long["unit"] == 0) & (long["time"] >= T0)).astype(int)
    res = CLUSTERSC({
        "df": long, "outcome": "y", "treat": "treat", "unitid": "unit",
        "time": "time", "method": "pcr", "clustering": False,
        "pcr_objective": "OLS", "rank": 6, "rank_method": "fixed",
        "project_denoised": True, "compute_shen_ci": False,
        "display_graphs": False,
    }).fit()
    weights = res.weights.donor_weights
    w = np.array([weights[str(j)] for j in range(1, 20)])
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(np.asarray(res.counterfactual, dtype=float)))
    _assert_close(w, _tsvd_of(X[1:, :T0].T, X[0, :T0], 6), rel=1e-8)


# ---------------------------------------------------------------------------
# Rung 3: the weights are a continuous function of the design
# ---------------------------------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), T0=st.integers(100, 300),
       J=st.integers(12, 60), r=st.integers(1, 8), where=st.floats(0.0, 1.0))
def test_weights_do_not_move_under_the_rounding_floor(seed, T0, J, r, where):
    """Trailing singular values the arithmetic cannot tell from zero change nothing.

    They are planted anywhere between four times numpy's pinv tolerance and a
    quarter of the rounding floor ``max(T0, J) * eps * s_1`` -- a perturbation
    under the error level of the design (Bjorck 1996, Definition 2.7.1). The
    weights stay at the TSVD solution with and without the rank given. A rank
    decided at ``1e-15 * s_1`` inverts them and moves the weights by about
    ``1 / ||E||`` (Wedin 1973, Theorem 7.2).
    """
    rng = np.random.default_rng(seed)
    head = np.sort(rng.uniform(1.0, 10.0, r))[::-1]
    lo, hi = 4 * NUMPY_PINV_RCOND, max(T0, J) * EPS / 4
    level = (lo + where * (hi - lo)) * head[0]
    n_tail = min(3, min(T0, J) - r)
    A, U, s, V = _design(rng, T0, J, head, level * rng.uniform(0.5, 1.0, n_tail))
    x = rng.standard_normal(T0)
    want = _tsvd(U, s, V, x, r)
    _assert_close(solve_ols(A, x), want, rel=1e-8)
    _assert_close(solve_ols(A, x, rank=r), want, rel=1e-8)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), T0=st.integers(150, 400),
       J=st.integers(40, 100), r=st.integers(10, 39))
def test_run_pcr_weights_are_the_tsvd_solution_of_the_donors(seed, T0, J, r):
    """The pipeline's weights are the TSVD solution of the raw donor pre-period.

    A differential between two computations of one estimator: ``run_pcr``
    reconstructs the donors at rank ``r`` and solves on the reconstruction; the
    oracle truncates the SVD of the raw donors, as tslib does. The domain is
    where the fault lives: on OpenBLAS's SkylakeX kernel numpy's pinv kept a
    rounding direction in 59 of 120 random panels of this size, against none of
    120 with ``T0`` 60-200, ``J`` 20-60 and ``r`` up to 19.
    """
    rng = np.random.default_rng(seed)
    T = T0 + 10
    X = (rng.standard_normal((T, 4)) @ rng.standard_normal((4, J))
         + rng.standard_normal((T, J)))
    y = X[:, :3].mean(axis=1) + 0.1 * rng.standard_normal(T)
    names = [f"d{j}" for j in range(J)]
    fit, _ = run_pcr(y, X, names, T0, rank=r, compute_shen_ci=False)
    w = np.array([fit.donor_weights[n] for n in names])
    _assert_close(w, _tsvd_of(X[:T0], y[:T0], r), rel=1e-8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_rank_one():
    rng = np.random.default_rng(2)
    A, U, s, V = _design(rng, 40, 10, [5.0], [1e-14 * 5.0])
    x = rng.standard_normal(40)
    _assert_close(solve_ols(A, x, rank=1), _tsvd(U, s, V, x, 1))


def test_a_full_rank_design_is_ordinary_least_squares():
    rng = np.random.default_rng(3)
    X, x = rng.standard_normal((50, 8)), rng.standard_normal(50)
    want = np.linalg.lstsq(X, x, rcond=None)[0]
    _assert_close(solve_ols(X, x), want, rel=1e-10)
    _assert_close(solve_ols(X, x, rank=8), want, rel=1e-10)


def test_a_rank_above_the_design_adds_no_directions():
    """Asking for more directions than the design has adds rounding, not signal."""
    rng = np.random.default_rng(4)
    A, U, s, V = _design(rng, 30, 6, [3.0, 2.0])       # exact rank 2, six columns
    x = rng.standard_normal(30)
    _assert_close(solve_ols(A, x, rank=6), _tsvd(U, s, V, x, 2))
    _assert_close(solve_ols(A, x, rank=50), _tsvd(U, s, V, x, 2))


def test_a_single_donor():
    rng = np.random.default_rng(5)
    m = rng.standard_normal((25, 1))
    x = 2.0 * m[:, 0] + 0.1 * rng.standard_normal(25)
    want = np.array([m[:, 0] @ x / (m[:, 0] @ m[:, 0])])
    _assert_close(solve_ols(m, x, rank=1), want, rel=1e-12)


def test_more_donors_than_pre_periods():
    """J > T0, the usual synthetic-control shape; the tail sits under 40 * eps."""
    rng = np.random.default_rng(6)
    head = [9.0, 5.0, 3.0, 2.0]
    A, U, s, V = _design(rng, 12, 40, head, [3e-15 * head[0]])
    x = rng.standard_normal(12)
    want = _tsvd(U, s, V, x, 4)
    _assert_close(solve_ols(A, x), want)
    _assert_close(solve_ols(A, x, rank=4), want)


def test_duplicate_donors_share_the_weight():
    """Collinear donors: the minimum-norm solution splits the weight equally."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((40, 5))
    X = np.column_stack([base, base[:, 0]])
    x = rng.standard_normal(40)
    w = solve_ols(X, x, rank=6)
    assert w[0] == pytest.approx(w[5], rel=1e-9)
    _assert_close(w, np.linalg.lstsq(X, x, rcond=None)[0], rel=1e-9)


def test_no_pre_periods_are_refused():
    with pytest.raises(MlsynthEstimationError, match="pre-period"):
        solve_ols(np.empty((0, 4)), np.empty(0))


def test_treatment_at_the_first_period_is_refused():
    rng = np.random.default_rng(8)
    X, y = rng.standard_normal((20, 5)), rng.standard_normal(20)
    with pytest.raises(MlsynthEstimationError, match="T0"):
        run_pcr(y, X, [f"d{j}" for j in range(5)], 0, rank=2,
                compute_shen_ci=False)


# ---------------------------------------------------------------------------
# Failures are reported
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [0, -1, 2.5, True, "3"])
def test_a_rank_that_is_not_a_positive_integer_is_refused(rank):
    rng = np.random.default_rng(9)
    A, x = rng.standard_normal((20, 4)), rng.standard_normal(20)
    with pytest.raises(MlsynthEstimationError, match="rank"):
        solve_ols(A, x, rank=rank)


def test_a_numpy_integer_rank_is_accepted():
    rng = np.random.default_rng(10)
    A, U, s, V = _design(rng, 30, 8, [4.0, 2.0, 1.0])
    x = rng.standard_normal(30)
    _assert_close(solve_ols(A, x, rank=np.int64(3)), _tsvd(U, s, V, x, 3))


@pytest.mark.parametrize("where", ["design", "target"])
def test_non_finite_input_is_refused(where):
    rng = np.random.default_rng(11)
    A, x = rng.standard_normal((20, 4)), rng.standard_normal(20)
    if where == "design":
        A[3, 1] = np.nan
    else:
        x[5] = np.inf
    with pytest.raises(MlsynthEstimationError, match="finite"):
        solve_ols(A, x)


def test_a_zero_design_is_reported_not_returned_as_zero_weights():
    """No weight vector is identified; pinv answered zeros and said nothing."""
    with pytest.raises(MlsynthEstimationError, match="numerical rank 0"):
        solve_ols(np.zeros((10, 3)), np.ones(10))

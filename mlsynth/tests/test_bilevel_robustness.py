"""Robustness / failure-mode tests for the bilevel SCM package.

Where ``test_fscm_bilevel.py`` proves every line *runs* on well-formed input,
this suite proves the package *fails loudly and correctly* on malformed,
degenerate, or pathological input:

* :class:`MlsynthDataError` from ``BilevelProblem`` validation (shapes,
  finiteness, empty donors, label mismatch);
* ``ValueError`` guards (non-positive simplex radius, negative penalty,
  predictor-free malo/mscmt, CV on too-short panels);
* ``RuntimeWarning`` diagnostics that were previously silent (FISTA / DE
  non-convergence, boundary-``lambda`` CV selection, a blown-up bilevel gap).
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.bilevel import (
    BilevelProblem,
    project_simplex,
    simplex_lstsq,
    solve_bilevel,
    penalized_weights,
)
from mlsynth.utils.bilevel.penalized import _simplex_qp
from mlsynth.utils.bilevel.stages import warn_on_gap


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _valid_kwargs(Tpre=10, J=5, K=3, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        y1_pre=rng.normal(size=Tpre),
        Y0_pre=rng.normal(size=(Tpre, J)),
        X1=rng.normal(size=K),
        X0=rng.normal(size=(K, J)),
    )


# --------------------------------------------------------------------------- #
# BilevelProblem validation  (MlsynthDataError)
# --------------------------------------------------------------------------- #
def test_valid_problem_constructs_and_coerces():
    # Lists / integer arrays are coerced to float ndarrays.
    prob = BilevelProblem(
        y1_pre=[1, 2, 3], Y0_pre=[[1, 2], [3, 4], [5, 6]],
        X1=[1], X0=[[1, 2]],
    )
    assert prob.y1_pre.dtype == float
    assert prob.Y0_pre.dtype == float
    assert (prob.n_donors, prob.n_predictors, prob.Tpre) == (2, 1, 3)


def test_zero_predictors_is_allowed():
    # K == 0 is legal at construction (only the penalized backend uses it).
    prob = BilevelProblem(
        y1_pre=np.zeros(5), Y0_pre=np.zeros((5, 3)),
        X1=np.zeros(0), X0=np.zeros((0, 3)),
    )
    assert prob.n_predictors == 0


def test_period_mismatch_rejected():
    kw = _valid_kwargs()
    kw["Y0_pre"] = np.zeros((kw["y1_pre"].shape[0] + 1, 5))
    with pytest.raises(MlsynthDataError, match="period mismatch"):
        BilevelProblem(**kw)


def test_predictor_count_mismatch_rejected():
    kw = _valid_kwargs(K=3)
    kw["X0"] = np.zeros((4, 5))                       # 4 rows vs X1's 3
    with pytest.raises(MlsynthDataError, match="predictor mismatch"):
        BilevelProblem(**kw)


def test_donor_count_mismatch_rejected():
    kw = _valid_kwargs(J=5, K=3)
    kw["X0"] = np.zeros((3, 4))                       # 4 donor cols vs Y0's 5
    with pytest.raises(MlsynthDataError, match="donor mismatch"):
        BilevelProblem(**kw)


def test_empty_donors_rejected():
    with pytest.raises(MlsynthDataError, match="at least one donor"):
        BilevelProblem(y1_pre=np.zeros(5), Y0_pre=np.zeros((5, 0)),
                       X1=np.zeros(2), X0=np.zeros((2, 0)))


def test_empty_periods_rejected():
    with pytest.raises(MlsynthDataError, match="at least one pre-period"):
        BilevelProblem(y1_pre=np.zeros(0), Y0_pre=np.zeros((0, 3)),
                       X1=np.zeros(2), X0=np.zeros((2, 3)))


@pytest.mark.parametrize("bad", ["y1_pre", "Y0_pre", "X1", "X0"])
def test_nan_rejected(bad):
    kw = _valid_kwargs()
    arr = np.array(kw[bad], dtype=float)
    arr.flat[0] = np.nan
    kw[bad] = arr
    with pytest.raises(MlsynthDataError, match="NaN or infinite"):
        BilevelProblem(**kw)


def test_inf_rejected():
    kw = _valid_kwargs()
    kw["y1_pre"][0] = np.inf
    with pytest.raises(MlsynthDataError, match="NaN or infinite"):
        BilevelProblem(**kw)


def test_wrong_dimensionality_rejected():
    base = _valid_kwargs()
    with pytest.raises(MlsynthDataError, match="y1_pre must be 1-D"):
        BilevelProblem(**{**base, "y1_pre": np.zeros((10, 1))})
    with pytest.raises(MlsynthDataError, match="Y0_pre must be 2-D"):
        BilevelProblem(**{**base, "Y0_pre": np.zeros(10)})
    with pytest.raises(MlsynthDataError, match="X1 must be 1-D"):
        BilevelProblem(**{**base, "X1": np.zeros((3, 1))})
    with pytest.raises(MlsynthDataError, match="X0 must be 2-D"):
        BilevelProblem(**{**base, "X0": np.zeros(3)})


def test_predictor_names_length_mismatch_rejected():
    kw = _valid_kwargs(K=3)
    with pytest.raises(MlsynthDataError, match="predictor_names"):
        BilevelProblem(predictor_names=["a", "b"], **kw)   # 2 labels for 3 predictors


# --------------------------------------------------------------------------- #
# project_simplex non-positive radius
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("z", [0.0, -1.0])
def test_project_simplex_nonpositive_radius_rejected(z):
    with pytest.raises(ValueError, match="radius z must be positive"):
        project_simplex(np.array([1.0, 2.0]), z=z)


# --------------------------------------------------------------------------- #
# Negative penalty
# --------------------------------------------------------------------------- #
def test_penalized_weights_negative_lambda_rejected():
    with pytest.raises(ValueError, match="lambda must be non-negative"):
        penalized_weights(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), lam=-0.5)


def test_solve_bilevel_negative_lambda_rejected():
    prob = BilevelProblem(**_valid_kwargs())
    with pytest.raises(ValueError, match="lambda must be non-negative"):
        solve_bilevel(prob, method="penalized", lam=-1.0)


# --------------------------------------------------------------------------- #
# Predictor-free malo / mscmt are rejected (penalized is the right backend)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["malo", "mscmt"])
def test_predictor_free_backend_rejected(method):
    prob = BilevelProblem(y1_pre=np.zeros(6), Y0_pre=np.zeros((6, 3)),
                          X1=np.zeros(0), X0=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="needs at least one predictor"):
        solve_bilevel(prob, method=method)


# --------------------------------------------------------------------------- #
# CV on too-short panels
# --------------------------------------------------------------------------- #
def test_cv_on_short_panel_rejected():
    prob = BilevelProblem(y1_pre=np.zeros(3), Y0_pre=np.ones((3, 4)),
                          X1=np.zeros(2), X0=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="pre-periods"):
        solve_bilevel(prob, method="penalized", lam="cv", cv="holdout")


# --------------------------------------------------------------------------- #
# Convergence warnings (previously silent)
# --------------------------------------------------------------------------- #
def test_simplex_lstsq_warns_on_nonconvergence():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 4)); b = rng.normal(size=12)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        simplex_lstsq(A, b, max_iter=1, warn=True)


def test_simplex_lstsq_silent_by_default():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 4)); b = rng.normal(size=12)
    with warnings_as_errors():
        simplex_lstsq(A, b, max_iter=1)               # warn=False -> no warning


def test_simplex_qp_warns_on_nonconvergence():
    rng = np.random.default_rng(1)
    X0 = rng.normal(size=(3, 5)); Q = X0.T @ X0; c = rng.normal(size=5)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        _simplex_qp(Q, c, max_iter=1, warn=True)


def test_penalized_weights_warns_on_nonconvergence():
    rng = np.random.default_rng(2)
    X0 = rng.normal(size=(3, 6)); X1 = rng.normal(size=3)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        penalized_weights(X1, X0, lam=0.1, max_iter=1, warn=True)


def test_mscmt_warns_on_de_nonconvergence():
    prob = BilevelProblem(**_valid_kwargs(Tpre=12, J=6, K=3, seed=5))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        solve_bilevel(prob, method="mscmt", maxiter=1, polish=False, seed=0)


# --------------------------------------------------------------------------- #
# Boundary-lambda CV warning
# --------------------------------------------------------------------------- #
def test_cv_boundary_lambda_warns():
    rng = np.random.default_rng(4)
    Y0 = np.cumsum(rng.normal(size=(18, 8)), axis=0) + 5.0
    w = rng.dirichlet(np.ones(8)); y1 = Y0 @ w + rng.normal(scale=0.1, size=18)
    X0 = rng.normal(size=(3, 8)); X1 = X0 @ w
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    # A two-point grid forces the selection onto an endpoint either way.
    with pytest.warns(RuntimeWarning, match="edge of lam_grid"):
        solve_bilevel(prob, method="penalized", lam="cv", cv="holdout",
                      lam_grid=np.array([1e-3, 1e2]))


# --------------------------------------------------------------------------- #
# Bilevel-gap warning  (weakly-identified predictors)
# --------------------------------------------------------------------------- #
def test_warn_on_gap_fires_for_large_gap():
    with pytest.warns(RuntimeWarning, match="optimality gap"):
        warn_on_gap(gap=100.0, lower_bound=1.0, factor=10.0)


def test_warn_on_gap_silent_for_small_gap():
    with warnings_as_errors():
        warn_on_gap(gap=2.0, lower_bound=1.0, factor=10.0)       # 2x < 10x


def test_warn_on_gap_silent_for_trivial_lower_bound():
    with warnings_as_errors():
        warn_on_gap(gap=100.0, lower_bound=1e-15, factor=10.0)   # bound ~ 0 -> skip


# --------------------------------------------------------------------------- #
# small context manager: assert *no* warning is emitted
# --------------------------------------------------------------------------- #
import contextlib
import warnings as _warnings


@contextlib.contextmanager
def warnings_as_errors():
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        yield

"""BEAST's weighted-least-squares Lasso (``beast_helpers.estimator._fista_wls``).

The immunizing outcome model calls it about 12,600 times on the Proposition 99
panel, and each call runs a few hundred FISTA iterations, so the per-iteration cost
is what one BEAST fit costs. These tests pin the iterates to the textbook form
below: the same updates and the same stopping rule, so the coefficients are equal
bit for bit, and the objective of the previous iterate is not computed twice."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.beast_helpers import estimator as est
from mlsynth.utils.beast_helpers.estimator import _fista_wls


def _reference(y, X, W, lam, nopen, max_iter=200000, tol=1e-6):
    """FISTA as first written: both objectives evaluated at every iteration."""
    sw = np.sqrt(W)
    yw, Xw = sw * y, X * sw[:, None]
    n = X.shape[0]
    eta = 1.0 / np.max(2.0 * np.linalg.eigvalsh(Xw.T @ Xw) / n)
    nopen = np.asarray(nopen)
    pen = np.array([j for j in range(X.shape[1]) if j not in set(nopen.tolist())])
    beta = np.zeros(X.shape[1])
    v = beta.copy()
    theta = 1.0

    def obj(b):
        l1 = np.sum(np.abs(b[pen])) if pen.size else 0.0
        return float(np.mean((yw - Xw @ b) ** 2) + lam * l1)

    for _ in range(max_iter):
        th_o, theta = theta, (1.0 + np.sqrt(1.0 + 4.0 * theta ** 2)) / 2.0
        delta = (1.0 - th_o) / theta
        b_o = beta
        g = -2.0 * (yw - Xw @ v) @ Xw / n
        beta = est._soft(v - eta * g, lam * eta, nopen)
        v = (1.0 - delta) * beta + delta * b_o
        if abs(obj(beta) - obj(b_o)) < tol:
            break
    return beta


def _problem(seed, n=40, p=6, scale=1.0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.normal(size=(n, p - 1))]
    y = X @ np.r_[1.0, rng.normal(size=p - 1) * (rng.random(p - 1) < 0.5)] + 0.1 * rng.normal(size=n)
    W = np.exp(rng.normal(size=n)) * scale
    return y, X, W


def _iterations(monkeypatch, *args, **kw):
    """The FISTA iterations of a call: one soft-thresholding each."""
    calls = []
    real = est._soft
    monkeypatch.setattr(est, "_soft", lambda *a: (calls.append(1), real(*a))[1])
    _fista_wls(*args, **kw)
    return len(calls)


# ---------------------------------------------------------------- smoke


def test_smoke_a_finite_solution_of_the_right_shape():
    y, X, W = _problem(0)
    b = _fista_wls(y, X, W, 0.05, nopen=[0])
    assert b.shape == (X.shape[1],) and np.all(np.isfinite(b))


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("lam", [0.0, 0.01, 0.2])
def test_the_iterates_are_the_reference_bit_for_bit(seed, lam):
    y, X, W = _problem(seed)
    np.testing.assert_array_equal(_fista_wls(y, X, W, lam, nopen=[0]),
                                  _reference(y, X, W, lam, nopen=[0]))


def test_the_objective_is_evaluated_once_per_iteration(monkeypatch):
    """The previous iterate's objective is carried, not recomputed: the residuals and
    the penalty are each reduced once per iteration, and once more for the start."""
    y, X, W = _problem(1)
    iters = _iterations(monkeypatch, y, X, W, 0.05, [0])
    reduced = []
    real = np.add.reduce
    monkeypatch.setattr(est, "_reduce", lambda a: (reduced.append(1), real(a))[1])
    _fista_wls(y, X, W, 0.05, nopen=[0])
    assert iters > 5 and len(reduced) == 2 * (iters + 1)   # residuals and penalty, per iterate and the start


def test_a_large_penalty_keeps_the_penalised_coefficients_at_zero():
    y, X, W = _problem(2)
    b = _fista_wls(y, X, W, 1e6, nopen=[0])
    assert np.all(b[1:] == 0.0) and b[0] != 0.0


def test_no_penalty_solves_weighted_least_squares():
    y, X, W = _problem(3)
    b = _fista_wls(y, X, W, 0.0, nopen=[0], tol=1e-14)
    ols = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
    np.testing.assert_allclose(b, ols, atol=1e-5)


# ---------------------------------------------------------------- edge


def test_every_column_unpenalised():
    y, X, W = _problem(4, p=3)
    np.testing.assert_array_equal(_fista_wls(y, X, W, 0.5, nopen=[0, 1, 2]),
                                  _reference(y, X, W, 0.5, nopen=[0, 1, 2]))


def test_a_single_column():
    y, X, W = _problem(5, p=1)
    np.testing.assert_array_equal(_fista_wls(y, X, W, 0.1, nopen=[0]),
                                  _reference(y, X, W, 0.1, nopen=[0]))


def test_a_zero_outcome_stops_after_one_iteration(monkeypatch):
    _, X, W = _problem(6)
    assert _iterations(monkeypatch, np.zeros(X.shape[0]), X, W, 0.1, [0]) == 1


def test_collinear_columns():
    y, X, W = _problem(7)
    X = np.c_[X, X[:, 1]]
    np.testing.assert_array_equal(_fista_wls(y, X, W, 0.05, nopen=[0]),
                                  _reference(y, X, W, 0.05, nopen=[0]))


def test_the_iteration_cap_is_kept(monkeypatch):
    y, X, W = _problem(8)
    assert _iterations(monkeypatch, y, X, W, 0.05, [0], max_iter=3, tol=0.0) == 3
    np.testing.assert_array_equal(_fista_wls(y, X, W, 0.05, nopen=[0], max_iter=3, tol=0.0),
                                  _reference(y, X, W, 0.05, nopen=[0], max_iter=3, tol=0.0))


# ---------------------------------------------------------------- failure


def test_a_non_finite_weight_is_not_hidden():
    """The helper does no validation (BEASTConfig and the pipeline do): a NaN weight
    gives NaN coefficients, the same as the reference, not a plausible number."""
    y, X, W = _problem(9)
    W[3] = np.nan
    with np.errstate(invalid="ignore"):
        with pytest.raises(np.linalg.LinAlgError):
            _reference(y, X, W, 0.05, nopen=[0])
        with pytest.raises(np.linalg.LinAlgError):
            _fista_wls(y, X, W, 0.05, nopen=[0])


# ---------------------------------------------------------------- property


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(3, 30), p=st.integers(1, 6),
       lam=st.sampled_from([0.0, 1e-3, 0.05, 1.0]), scale=st.sampled_from([1e-3, 1.0, 1e3]),
       tol=st.sampled_from([1e-6, 1e-9]))
def test_property_equal_to_the_reference_bit_for_bit(seed, n, p, lam, scale, tol):
    y, X, W = _problem(seed, n=n, p=p, scale=scale)
    np.testing.assert_array_equal(_fista_wls(y, X, W, lam, nopen=[0], max_iter=5000, tol=tol),
                                  _reference(y, X, W, lam, nopen=[0], max_iter=5000, tol=tol))

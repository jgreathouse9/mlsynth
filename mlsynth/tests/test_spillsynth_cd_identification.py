"""Every inversion of A' M A on the Cao-Dowd path names Assumption 1(d).

The point estimate inverts ``A' M A`` through ``_invert_AMA``, which raises a
typed error naming the failed identification condition. Three later steps
inverted the same matrix with a bare ``np.linalg.inv``: the inference sandwich
``G_matrix``, the ``kappa_A`` specification test and the pure-donor
sensitivity. On a design at the edge of singular, which side a solve lands on
is a property of the LAPACK build, so the first step could pass and a later one
raise ``LinAlgError``; the estimator then reported "estimation failed (Singular
matrix)", with no word of why (#960, CI on Python 3.10 to 3.12).
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.spillsynth_helpers.cd.inference import G_matrix, kappa_A_test
from mlsynth.utils.spillsynth_helpers.cd.sensitivity import pure_donor_sensitivity


def _design(seed=0, N=6, T0=12, T1=3):
    rng = np.random.default_rng(seed)
    B = np.abs(rng.normal(size=(N, N)))
    np.fill_diagonal(B, 0.0)
    B /= B.sum(axis=1, keepdims=True)
    A = np.zeros((N, 2))
    A[0, 0], A[1, 1] = 1.0, 1.0
    Y_pre, Y_post = rng.normal(size=(N, T0)), rng.normal(size=(N, T1))
    return B, A, Y_pre, Y_post, np.zeros(N), np.zeros((N, T1))


def _unidentified(A):
    """A spillover column that is all zeros: A' M A is exactly singular."""
    A = A.copy()
    A[:, 1] = 0.0
    return A


def _calls(B, A, Y_pre, Y_post, a, alpha):
    return {
        "G_matrix": lambda: G_matrix(A, B),
        "kappa_A_test": lambda: kappa_A_test(Y_post=Y_post, alpha_hat=alpha,
                                             Y_pre=Y_pre, a=a, B=B, A=A),
        "pure_donor_sensitivity": lambda: pure_donor_sensitivity(Y_pre, B=B, A=A),
    }


@pytest.mark.parametrize("step", ["G_matrix", "kappa_A_test", "pure_donor_sensitivity"])
def test_identified_structure_runs(step):
    """Smoke: an identified A gives finite output at every step."""
    B, A, *rest = _design()
    out = _calls(B, A, *rest)[step]()
    if step == "G_matrix":
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("step", ["G_matrix", "kappa_A_test", "pure_donor_sensitivity"])
def test_a_singular_structure_names_assumption_1d(step):
    """Failure: the error is typed and says which condition failed."""
    B, A, *rest = _design()
    with pytest.raises(MlsynthEstimationError, match=r"Assumption 1\(d\)"):
        _calls(B, _unidentified(A), *rest)[step]()


def test_the_sandwich_is_unchanged_where_identified():
    """Unit: routing through the named inverse leaves the value as it was."""
    B, A, *_ = _design(3)
    I_B = np.eye(B.shape[0]) - B
    M = I_B.T @ I_B
    expected = A @ np.linalg.inv(A.T @ M @ A) @ A.T @ I_B.T
    np.testing.assert_allclose(G_matrix(A, B), expected, rtol=1e-12, atol=1e-14)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), N=st.integers(3, 9))
def test_every_step_reports_a_zero_column(seed, N):
    """Property: whatever the panel, a structure with a zero column is
    reported by name at every step, never as a bare LinAlgError."""
    B, A, *rest = _design(seed, N=N)
    for step, call in _calls(B, _unidentified(A), *rest).items():
        with pytest.raises(MlsynthEstimationError, match=r"Assumption 1\(d\)"):
            call()

"""Generative property tests for the modified-BIC LASSO penalty rule.

Reference: Shi and Huang (2023), Remark 4 cont.; ``fsPDA``'s ``lasso.BIC``. The
rule scores each penalty on a grid by

    IC(lambda) = log(sigma^2(lambda))
                 + H log(log N) log(T1)/T1 k(lambda)

and keeps the minimiser, fitting the path as
``glmnet(..., standardize = TRUE, intercept = FALSE)``.

Why this layer. ``test_pda_lasso_mbic.py`` asserts the rule at one panel, which
is where its pinned ``glmnet`` value has to live -- a generated panel carries no
known truth to compare against. What that file cannot do is establish the claims
below over the input domain, and those claims are the ones a fault of omission
hides in: they are true of every panel, so a single fixture satisfying them says
little.

Three claims carry the rule's meaning.

Scale equivariance is the whole content of ``standardize = TRUE``. Multiplying a
donor's column by a positive constant is a change of units, and the point of
standardising is that the answer does not depend on whether a series is measured
in dollars or thousands of dollars. The selected penalty, the selected donors and
the counterfactual all have to come back unchanged, with only that donor's
coefficient rescaled. Drop the standardisation and the penalty starts charging
more for donors that happen to be recorded in small units.

Permutation equivariance says the donor pool is a set. Column order comes from
however the panel was pivoted, so it cannot reach the answer.

Monotonicity in ``H`` is the knob the paper tunes. The charge per donor is
non-negative at every pool size, so raising ``H`` can only raise the criterion,
and it can never make a denser fit look better than it did.

The generators keep the donor columns away from degeneracy. A column with no
spread has no scale to divide by, so the rule leaves it alone and scale
equivariance has nothing to say about it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.pda_helpers.lasso import (
    MBIC_CONST,
    fit_lasso,
    lasso_mbic_alpha,
    mbic_criterion,
)

_entry = st.floats(min_value=-5.0, max_value=5.0,
                   allow_nan=False, allow_infinity=False)
_penalty = st.floats(min_value=0.01, max_value=1.0,
                     allow_nan=False, allow_infinity=False)
_const = st.floats(min_value=0.05, max_value=20.0,
                   allow_nan=False, allow_infinity=False)

#: Small enough that a property runs in milliseconds; the rule refits the path
#: once per grid point, so the grid is trimmed too.
_GRID = np.array([0.05, 0.2, 0.5, 1.0])

_SETTINGS = settings(max_examples=40, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])


@st.composite
def panels(draw, min_donors: int = 2, max_donors: int = 5):
    """``(y, X, T0)`` with every donor column carrying some spread."""
    N = draw(st.integers(min_value=min_donors, max_value=max_donors))
    T0 = draw(st.integers(min_value=4, max_value=10))
    T = T0 + draw(st.integers(min_value=1, max_value=4))
    flat = draw(st.lists(_entry, min_size=T * N, max_size=T * N))
    X = np.asarray(flat, dtype=float).reshape(T, N)
    assume(np.std(X[:T0], axis=0).min() > 1e-3)
    y = np.asarray(draw(st.lists(_entry, min_size=T, max_size=T)), dtype=float)
    assume(np.abs(y).max() > 1e-6)
    return y, X, T0


@st.composite
def positive_scales(draw, n: int):
    return np.asarray(
        draw(st.lists(st.floats(min_value=0.25, max_value=4.0,
                                allow_nan=False, allow_infinity=False),
                      min_size=n, max_size=n)), dtype=float)


class TestScaleEquivariance:
    """``standardize = TRUE`` makes the answer independent of donor units."""

    @given(panels(), st.data())
    @_SETTINGS
    def test_rescaling_donors_leaves_the_selected_penalty_alone(self, panel, data):
        y, X, T0 = panel
        scales = data.draw(positive_scales(X.shape[1]))
        assert (lasso_mbic_alpha(y, X * scales, T0, grid=_GRID)
                == lasso_mbic_alpha(y, X, T0, grid=_GRID))

    @given(panels(), _penalty, st.data())
    @_SETTINGS
    def test_rescaling_donors_rescales_only_the_coefficients(self, panel, alpha,
                                                             data):
        y, X, T0 = panel
        scales = data.draw(positive_scales(X.shape[1]))
        beta, _, cf, support = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                         standardize=True)
        beta_s, _, cf_s, support_s = fit_lasso(y, X * scales, T0, alpha=alpha,
                                               intercept=False, standardize=True)
        assert np.array_equal(support, support_s)
        assert np.allclose(beta_s * scales, beta, atol=1e-7)
        assert np.allclose(cf_s, cf, atol=1e-7)

    @given(panels(), _penalty, st.data())
    @_SETTINGS
    def test_the_criterion_is_unchanged_by_donor_units(self, panel, alpha, data):
        """To rounding: the criterion is a log of a mean of squares, so the two
        orders of operation differ in the last bit even when the fit does not.
        The selected penalty above is compared exactly, since a grid argmin is
        discrete and a rounding difference has to be large enough to change which
        point wins before it can change the answer."""
        y, X, T0 = panel
        scales = data.draw(positive_scales(X.shape[1]))
        assert mbic_criterion(y, X * scales, T0, alpha) == pytest.approx(
            mbic_criterion(y, X, T0, alpha), rel=1e-12, abs=1e-12)


class TestPermutationEquivariance:
    """The donor pool is a set; column order comes from the pivot."""

    @given(panels(), st.data())
    @_SETTINGS
    def test_reordering_donors_leaves_the_selected_penalty_alone(self, panel, data):
        y, X, T0 = panel
        perm = data.draw(st.permutations(range(X.shape[1])))
        perm = np.asarray(list(perm), dtype=int)
        assert (lasso_mbic_alpha(y, X[:, perm], T0, grid=_GRID)
                == lasso_mbic_alpha(y, X, T0, grid=_GRID))

    @given(panels(), _penalty, st.data())
    @_SETTINGS
    def test_reordering_donors_permutes_the_coefficients(self, panel, alpha, data):
        y, X, T0 = panel
        perm = np.asarray(list(data.draw(st.permutations(range(X.shape[1])))),
                          dtype=int)
        beta, _, cf, _ = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                   standardize=True)
        beta_p, _, cf_p, _ = fit_lasso(y, X[:, perm], T0, alpha=alpha,
                                       intercept=False, standardize=True)
        assert np.allclose(beta_p, beta[perm], atol=1e-7)
        assert np.allclose(cf_p, cf, atol=1e-7)


class TestTheConstantIsACharge:
    """``H`` scales a non-negative per-donor charge at every pool size."""

    @given(panels(), _penalty, _const, _const)
    @_SETTINGS
    def test_the_criterion_is_nondecreasing_in_the_constant(self, panel, alpha,
                                                            c1, c2):
        y, X, T0 = panel
        lo, hi = sorted((c1, c2))
        assert mbic_criterion(y, X, T0, alpha, const=hi) >= \
            mbic_criterion(y, X, T0, alpha, const=lo) - 1e-12

    @given(panels(), _penalty, _const)
    @_SETTINGS
    def test_the_criterion_never_falls_below_its_fit_term(self, panel, alpha,
                                                          const):
        """The charge is non-negative, so the criterion sits above log MSE."""
        y, X, T0 = panel
        beta, _, _, _ = fit_lasso(y, X, T0, alpha=alpha, intercept=False,
                                  standardize=True)
        mse = float(np.mean((y[:T0] - X[:T0] @ beta) ** 2))
        log_mse = np.log(max(mse, np.finfo(float).tiny))
        assert mbic_criterion(y, X, T0, alpha, const=const) >= log_mse - 1e-12


class TestTheSelectionIsTheArgmin:
    @given(panels(), _const)
    @_SETTINGS
    def test_no_grid_point_scores_better_than_the_one_returned(self, panel, const):
        y, X, T0 = panel
        alpha = lasso_mbic_alpha(y, X, T0, const=const, grid=_GRID)
        assert alpha in set(_GRID.tolist())
        best = mbic_criterion(y, X, T0, alpha, const=const)
        for a in _GRID:
            assert mbic_criterion(y, X, T0, float(a), const=const) >= best - 1e-12

    @given(panels())
    @_SETTINGS
    def test_the_default_constant_is_the_one_the_package_ships(self, panel):
        y, X, T0 = panel
        assert (lasso_mbic_alpha(y, X, T0, grid=_GRID)
                == lasso_mbic_alpha(y, X, T0, const=MBIC_CONST, grid=_GRID))

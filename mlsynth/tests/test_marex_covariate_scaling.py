"""MAREX covariate scaling: standardise each predictor, then weight it.

Vives-i-Bastida (2022, Algorithm 1) centres and scales every predictor and
then weights it through a diagonal H, with the covariates at a fraction of
the outcomes' weight. ``_augment_fit`` used to apply ``covariate_weight``
before standardising, and the standardisation divided it back out, so under
``standardize=True`` every weight gave the same design (issue #758).

The invariants pinned here:

* under ``standardize=True`` each non-constant outcome column has unit
  cross-unit standard deviation and each non-constant covariate column has
  standard deviation ``covariate_weight``;
* the covariate block is linear in the weight, and the outcome block does
  not depend on it;
* rescaling a covariate's units leaves the standardised predictors unchanged;
* weight 0 gives the design without covariates;
* without standardisation the predictors are ``[Y, covariate_weight * Z]``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.optimization import _augment_fit

warnings.filterwarnings("ignore")


def _block(seed, n=8, t=5, r=2, cov_scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t)), cov_scale * rng.normal(size=(n, r))


def _panel(seed, n=10, T=14, T0=12):
    """A panel whose covariates disagree with the outcomes about which units
    look like the population, so the covariate weight decides the design."""
    rng = np.random.default_rng(seed)
    level = rng.normal(size=n)
    Y = level[:, None] + rng.normal(scale=0.3, size=(n, T))
    Z = rng.normal(size=(n, 2)) * np.array([1.0, 1000.0])
    rows = [dict(unit=f"u{j}", t=t, y=Y[j, t], z1=Z[j, 0], z2=Z[j, 1])
            for j in range(n) for t in range(T)]
    return pd.DataFrame(rows), T0


def _design(df, T0, **kw):
    cfg = dict(df=df, outcome="y", unitid="unit", time="t", T0=T0,
               blank_periods=0, m_eq=2, backend="exact", inference=False)
    cfg.update(kw)
    res = MAREX(cfg).fit()
    return sorted(map(str, res.selected_units)), res


# ----------------------------------------------------------------------
# Unit: the predictor matrix
# ----------------------------------------------------------------------

class TestStandardizeThenWeight:

    def test_covariate_columns_have_sd_equal_to_the_weight(self):
        Y, Z = _block(0)
        for w in (0.1, 1.0, 10.0):
            X = _augment_fit(Y, Z, w, standardize=True)
            np.testing.assert_allclose(X[:, :Y.shape[1]].std(axis=0), 1.0)
            np.testing.assert_allclose(X[:, Y.shape[1]:].std(axis=0), w)

    def test_the_weight_changes_the_standardised_predictors(self):
        Y, Z = _block(1)
        a = _augment_fit(Y, Z, 0.1, standardize=True)
        b = _augment_fit(Y, Z, 10.0, standardize=True)
        assert not np.allclose(a, b)

    def test_unit_weight_is_the_previous_standardisation(self):
        """At weight 1 the order of the two steps does not matter, so every
        existing standardised design with the default weight is unchanged."""
        Y, Z = _block(2)
        X = np.hstack([Y, Z])
        np.testing.assert_allclose(_augment_fit(Y, Z, 1.0, standardize=True),
                                   X / X.std(axis=0))

    def test_without_standardize_the_weight_scales_the_covariates(self):
        Y, Z = _block(3)
        np.testing.assert_allclose(_augment_fit(Y, Z, 0.3),
                                   np.hstack([Y, 0.3 * Z]))

    def test_no_covariates_ignores_the_weight(self):
        Y, _ = _block(4)
        np.testing.assert_allclose(_augment_fit(Y, None, 7.0, standardize=True),
                                   Y / Y.std(axis=0))

    def test_constant_covariate_stays_finite_and_constant(self):
        Y, Z = _block(5)
        Z[:, 0] = 3.0
        X = _augment_fit(Y, Z, 0.5, standardize=True)
        assert np.isfinite(X).all()
        assert np.ptp(X[:, Y.shape[1]]) == 0.0

    def test_column_constant_up_to_rounding_is_not_blown_up(self):
        """228.3822942812345 repeated seven times has a computed standard
        deviation of 2.8e-14, not 0; the old ``sd == 0`` guard divided by it
        and produced a column of order 1e16 (found by the linearity
        property)."""
        Y = np.zeros((7, 1))
        Z = np.full((7, 1), 228.3822942812345)
        assert 0.0 < Z.std() < 1e-12
        X = _augment_fit(Y, Z, 3.0, standardize=True)
        np.testing.assert_allclose(X[:, 1], 3.0 * 228.3822942812345)

    def test_one_dimensional_covariate(self):
        Y, Z = _block(6)
        X = _augment_fit(Y, Z[:, 0], 0.25, standardize=True)
        assert X.shape == (Y.shape[0], Y.shape[1] + 1)
        assert X[:, -1].std() == pytest.approx(0.25)

    def test_inputs_are_not_modified(self):
        Y, Z = _block(7)
        Y0, Z0 = Y.copy(), Z.copy()
        _augment_fit(Y, Z, 0.1, standardize=True)
        np.testing.assert_array_equal(Y, Y0)
        np.testing.assert_array_equal(Z, Z0)


# ----------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------

_finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _blocks(draw):
    n = draw(st.integers(3, 8))
    t = draw(st.integers(1, 5))
    r = draw(st.integers(1, 3))
    Y = draw(arrays(float, (n, t), elements=_finite))
    Z = draw(arrays(float, (n, r), elements=_finite))
    return Y, Z


class TestProperties:

    @settings(max_examples=60, deadline=None)
    @given(_blocks(), st.floats(0.0, 100.0))
    def test_covariate_block_is_linear_in_the_weight(self, blocks, w):
        Y, Z = blocks
        T = Y.shape[1]
        one = _augment_fit(Y, Z, 1.0, standardize=True)
        X = _augment_fit(Y, Z, w, standardize=True)
        np.testing.assert_allclose(X[:, :T], one[:, :T])
        np.testing.assert_allclose(X[:, T:], w * one[:, T:], atol=1e-9)

    @settings(max_examples=60, deadline=None)
    @given(_blocks(), st.floats(1e-3, 1e3), st.floats(0.0, 10.0))
    def test_standardised_predictors_do_not_depend_on_covariate_units(
            self, blocks, a, w):
        """Compared after centring each column: the design weights sum to one,
        so a column's level never enters the objective, and a column constant
        across units is left in its own units."""
        Y, Z = blocks
        X = _augment_fit(Y, Z, w, standardize=True)
        Xa = _augment_fit(Y, a * Z, w, standardize=True)
        np.testing.assert_allclose(Xa - Xa.mean(axis=0), X - X.mean(axis=0),
                                   atol=1e-7)

    @settings(max_examples=60, deadline=None)
    @given(_blocks(), st.floats(1e-2, 100.0))
    def test_non_constant_columns_have_the_assigned_scale(self, blocks, w):
        Y, Z = blocks
        X = _augment_fit(Y, Z, w, standardize=True)
        scale = np.r_[np.ones(Y.shape[1]), np.full(Z.shape[1], w)]
        raw = np.hstack([Y, Z]).std(axis=0)
        live = raw > 1e-9 * (1.0 + np.abs(np.hstack([Y, Z])).max(axis=0))
        np.testing.assert_allclose(X.std(axis=0)[live], scale[live], rtol=1e-6)


# ----------------------------------------------------------------------
# End to end: the design responds to the weight
# ----------------------------------------------------------------------

class TestDesign:

    def test_zero_weight_is_the_design_without_covariates(self):
        df, T0 = _panel(0)
        bare, r0 = _design(df, T0, standardize=True)
        zero, r1 = _design(df, T0, standardize=True,
                           covariates=["z1", "z2"], covariate_weight=0.0)
        assert bare == zero
        w0 = r0.design_weights.donor_weights
        w1 = r1.design_weights.donor_weights
        for k in set(w0) | set(w1):
            assert w0.get(k, 0.0) == pytest.approx(w1.get(k, 0.0), abs=1e-8)

    def test_the_weight_moves_the_standardised_design(self):
        """Under standardisation a small and a large covariate weight lead to
        different designs on some panel; before the fix they never could."""
        moved = False
        for seed in range(12):
            df, T0 = _panel(seed)
            small, _ = _design(df, T0, standardize=True,
                               covariates=["z1", "z2"], covariate_weight=0.01)
            large, _ = _design(df, T0, standardize=True,
                               covariates=["z1", "z2"], covariate_weight=10.0)
            if small != large:
                moved = True
                break
        assert moved

    def test_small_weight_approaches_the_outcome_only_design(self):
        df, T0 = _panel(1)
        bare, _ = _design(df, T0, standardize=True)
        tiny, _ = _design(df, T0, standardize=True,
                          covariates=["z1", "z2"], covariate_weight=1e-6)
        assert tiny == bare

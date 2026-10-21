"""MAREX per-covariate weights: the diagonal H of Vives-i-Bastida (2022).

Vives-i-Bastida's program (1) measures the design's fit in a weighted norm
``||x||_H^2 = x'Hx`` with ``H`` diagonal, so a researcher can say that one
predictor matters more than another; his UBI design gives every outcome the
same weight and the covariates a tenth of it. ``covariate_weight`` was one
scalar for every covariate (issue #760). It now also takes a mapping from
covariate name to weight, applied after standardisation (#758).

The invariants pinned here:

* a mapping scales each covariate column by its own weight;
* a scalar is the mapping that gives every covariate that value;
* the mapping is matched by name, so the order of ``covariates`` does not
  change the design;
* weight 0 on a covariate is the design without that covariate;
* a mapping that misses a covariate, names an unknown one, or carries a
  negative weight is refused, and so is a negative scalar.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.optimization import (
    _augment_fit,
    resolve_covariate_weight,
)

warnings.filterwarnings("ignore")


def _panel(seed, n=10, T=14):
    rng = np.random.default_rng(seed)
    level = rng.normal(size=n)
    Y = level[:, None] + rng.normal(scale=0.3, size=(n, T))
    Z = rng.normal(size=(n, 3)) * np.array([1.0, 50.0, 0.01])
    rows = [dict(unit=f"u{j}", t=t, y=Y[j, t], a=Z[j, 0], b=Z[j, 1], c=Z[j, 2])
            for j in range(n) for t in range(T)]
    return pd.DataFrame(rows)


def _cfg(df, **kw):
    cfg = dict(df=df, outcome="y", unitid="unit", time="t", T0=12,
               blank_periods=0, m_eq=2, backend="exact", inference=False,
               standardize=True)
    cfg.update(kw)
    return cfg


def _design(df, **kw):
    res = MAREX(_cfg(df, **kw)).fit()
    w = res.design_weights.donor_weights
    return (sorted(map(str, res.selected_units)),
            {k: round(v, 9) for k, v in w.items() if abs(v) > 1e-10},
            round(float(res.globres.objective), 9))


# ----------------------------------------------------------------------
# Unit: resolving and applying the weights
# ----------------------------------------------------------------------

class TestResolve:

    def test_a_mapping_is_ordered_by_the_covariates(self):
        got = resolve_covariate_weight({"b": 2.0, "a": 0.5}, ("a", "b"))
        np.testing.assert_array_equal(got, [0.5, 2.0])

    def test_a_scalar_passes_through(self):
        assert resolve_covariate_weight(0.3, ("a", "b")) == 0.3

    def test_a_mapping_with_a_missing_name_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="b"):
            resolve_covariate_weight({"a": 1.0}, ("a", "b"))


class TestApply:

    def test_each_column_takes_its_own_weight(self):
        rng = np.random.default_rng(0)
        Y, Z = rng.normal(size=(8, 4)), rng.normal(size=(8, 3)) * 40
        w = np.array([0.1, 1.0, 3.0])
        X = _augment_fit(Y, Z, w, standardize=True)
        np.testing.assert_allclose(X[:, 4:].std(axis=0), w)
        np.testing.assert_allclose(X[:, :4].std(axis=0), 1.0)

    def test_without_standardize_the_columns_are_scaled(self):
        rng = np.random.default_rng(1)
        Y, Z = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
        w = np.array([0.5, 2.0])
        np.testing.assert_allclose(_augment_fit(Y, Z, w),
                                   np.hstack([Y, Z * w]))

    def test_a_vector_of_the_wrong_length_is_refused(self):
        rng = np.random.default_rng(2)
        Y, Z = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
        with pytest.raises(ValueError, match="covariate_weight"):
            _augment_fit(Y, Z, np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------

_finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _blocks(draw):
    n = draw(st.integers(3, 8))
    Y = draw(arrays(float, (n, draw(st.integers(1, 4))), elements=_finite))
    Z = draw(arrays(float, (n, draw(st.integers(1, 3))), elements=_finite))
    return Y, Z


class TestProperties:

    @settings(max_examples=60, deadline=None)
    @given(_blocks(), st.floats(0.0, 50.0), st.booleans())
    def test_a_scalar_is_the_uniform_vector(self, blocks, w, std):
        Y, Z = blocks
        np.testing.assert_allclose(
            _augment_fit(Y, Z, w, standardize=std),
            _augment_fit(Y, Z, np.full(Z.shape[1], w), standardize=std))

    @settings(max_examples=60, deadline=None)
    @given(_blocks(), st.data())
    def test_each_covariate_block_is_linear_in_its_own_weight(self, blocks,
                                                              data):
        Y, Z = blocks
        R, T = Z.shape[1], Y.shape[1]
        w = np.array(data.draw(st.lists(st.floats(0.0, 20.0), min_size=R,
                                        max_size=R)))
        X = _augment_fit(Y, Z, w, standardize=True)
        one = _augment_fit(Y, Z, 1.0, standardize=True)
        np.testing.assert_allclose(X[:, :T], one[:, :T])
        np.testing.assert_allclose(X[:, T:], one[:, T:] * w, atol=1e-9)


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------

class TestDesign:

    def test_zero_weight_drops_the_covariate(self):
        df = _panel(0)
        with_zero = _design(df, covariates=["a", "b", "c"],
                            covariate_weight={"a": 0.4, "b": 0.0, "c": 2.0})
        dropped = _design(df, covariates=["a", "c"],
                          covariate_weight={"a": 0.4, "c": 2.0})
        assert with_zero == dropped

    def test_the_mapping_follows_names_not_positions(self):
        df = _panel(1)
        weights = {"a": 0.1, "b": 3.0, "c": 1.0}
        one = _design(df, covariates=["a", "b", "c"], covariate_weight=weights)
        two = _design(df, covariates=["c", "a", "b"], covariate_weight=weights)
        assert one == two

    def test_a_uniform_mapping_is_the_scalar(self):
        df = _panel(2)
        a = _design(df, covariates=["a", "b"], covariate_weight=0.7)
        b = _design(df, covariates=["a", "b"],
                    covariate_weight={"a": 0.7, "b": 0.7})
        assert a == b

    def test_unequal_weights_move_the_design_somewhere(self):
        moved = False
        for seed in range(10):
            df = _panel(seed)
            a = _design(df, covariates=["a", "b"],
                        covariate_weight={"a": 5.0, "b": 0.01})
            b = _design(df, covariates=["a", "b"],
                        covariate_weight={"a": 0.01, "b": 5.0})
            if a[0] != b[0]:
                moved = True
                break
        assert moved

    def test_the_solution_pool_takes_a_mapping(self):
        """``top_K > 1`` re-solves through its own path, on a held-out blank
        tail, so its designs differ from the primary one. It must receive the
        resolved weights: weight 0 on a covariate gives the pool built
        without that covariate."""
        df = _panel(6)

        def pool(**kw):
            res = MAREX(_cfg(df, top_K=2, **kw)).fit()
            return [(sorted(map(str, d["markets"])),
                     round(float(d["mip_objective"]), 9)) for d in res.pool]

        assert (pool(covariates=["a", "b"],
                     covariate_weight={"a": 0.0, "b": 2.0})
                == pool(covariates=["b"], covariate_weight={"b": 2.0}))

    def test_the_default_is_unchanged(self):
        df = _panel(3)
        assert (_design(df, covariates=["a", "b"])
                == _design(df, covariates=["a", "b"], covariate_weight=1.0))

    def test_the_mip_backend_takes_a_mapping(self):
        df = _panel(4)
        exact = _design(df, covariates=["a", "b"],
                        covariate_weight={"a": 0.2, "b": 2.0})
        mip = _design(df, covariates=["a", "b"], backend="mip",
                      covariate_weight={"a": 0.2, "b": 2.0})
        assert exact[0] == mip[0]
        assert exact[2] == pytest.approx(mip[2], rel=1e-5)


# ----------------------------------------------------------------------
# Failure
# ----------------------------------------------------------------------

class TestRefusals:

    @pytest.mark.parametrize("weight,match", [
        ({"a": 1.0}, "b"),
        ({"a": 1.0, "b": 1.0, "zz": 1.0}, "zz"),
        ({"a": 1.0, "b": -0.5}, ">= 0"),
        (-0.1, ">= 0"),
    ])
    def test_bad_weights_are_refused(self, weight, match):
        with pytest.raises(MlsynthConfigError, match=match):
            MAREXConfig(**_cfg(_panel(5), covariates=["a", "b"],
                               covariate_weight=weight))

    def test_a_mapping_needs_covariates(self):
        with pytest.raises(MlsynthConfigError, match="covariates"):
            MAREXConfig(**_cfg(_panel(5), covariate_weight={"a": 1.0}))

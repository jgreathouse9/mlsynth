"""Numeric guards must accept numpy scalars, not just Python floats and ints.

``isinstance(x, (float, int))`` looks like a test for "is this a number". It is
not. ``np.float64`` happens to subclass Python ``float`` and passes; ``np.float32``
does not subclass it and fails. So a guard written that way is silently
type-width dependent, and the width is decided by whatever produced the data.

That matters because pandas reads Stata ``.dta`` floats as ``float32``. A user
loading a replication package the obvious way gets their panel rejected, with a
message naming an internal quantity they never set:

    MlsynthConfigError: regularization_parameter_zeta must be a non-negative
    float or int.

The value is fine. Only the predicate is wrong. These tests pin the predicate at
every site that had it, and pin that the *value* checks around it still bite --
the fix must widen what counts as a number without loosening what counts as a
valid one.
"""
from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


def _panel(dtype):
    """Small block panel: 6 units, 12 periods, one treated from t=9."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        base = 10.0 + i
        for t in range(1, 13):
            y = base + 0.3 * t + rng.normal(0, 0.2) - (2.0 if (i == 0 and t >= 9) else 0.0)
            rows.append((f"u{i}", t, y, int(i == 0 and t >= 9)))
    df = pd.DataFrame(rows, columns=["unit", "t", "y", "treat"])
    df["y"] = df["y"].astype(dtype)
    return df


def _fit(df):
    import warnings

    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDID({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                     "time": "t", "vce": "noinference",
                     "display_graphs": False}).fit()


class TestSDIDAcceptsFloat32Panels:
    """The reachable bug: a float32 outcome column crashed the fit."""

    def test_a_float32_panel_fits(self):
        assert np.isfinite(_fit(_panel("float32")).effects.att)

    def test_float32_and_float64_agree(self):
        """Not merely 'does not crash' -- the two must be the same estimate.

        float32 carries ~7 significant digits, so the tolerance is a float32
        round-trip, not machine epsilon.
        """
        a = _fit(_panel("float32")).effects.att
        b = _fit(_panel("float32").assign(
            y=lambda d: d["y"].astype("float64"))).effects.att
        assert a == pytest.approx(b, rel=1e-5)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
    def test_every_reasonable_dtype_fits(self, dtype):
        df = _panel("float64")
        if dtype == "int64":
            df["y"] = (df["y"] * 100).round().astype("int64")
        else:
            df["y"] = df["y"].astype(dtype)
        assert np.isfinite(_fit(df).effects.att)


class TestTheZetaGuardStillBites:
    """Widening the type must not loosen the value check."""

    @pytest.mark.parametrize("bad", [np.float32(-1.0), np.float64(-1.0), -1.0])
    def test_a_negative_zeta_is_still_rejected(self, bad):
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.sdid_helpers.weights import unit_weights

        Y = np.random.default_rng(0).normal(size=(8, 3))
        with pytest.raises(MlsynthConfigError, match="non-negative"):
            unit_weights(Y, Y[:, 0], bad)

    @pytest.mark.parametrize("good", [np.float32(1.5), np.float64(1.5),
                                      np.int64(2), 1.5, 2])
    def test_numeric_scalars_are_accepted(self, good):
        from mlsynth.utils.sdid_helpers.weights import unit_weights

        rng = np.random.default_rng(1)
        Y = rng.normal(size=(8, 3))
        intercept, w = unit_weights(Y, Y[:, 0], good)
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("bad", [float("nan"), np.float32("nan"),
                                     float("inf")])
    def test_a_non_finite_zeta_is_rejected(self, bad):
        """Closed while fixing the type predicate, not before it.

        ``isinstance(nan, float)`` is True and ``nan < 0`` is False, so a NaN
        penalty passed the old guard untouched and propagated silently into
        every weight. Widening the predicate is the moment to notice that the
        value check never covered this.
        """
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.sdid_helpers.weights import unit_weights

        Y = np.random.default_rng(0).normal(size=(8, 3))
        with pytest.raises(MlsynthConfigError):
            unit_weights(Y, Y[:, 0], bad)

    @pytest.mark.parametrize("bad", ["1.0", None, [1.0]])
    def test_non_numbers_are_still_rejected(self, bad):
        """The predicate widens to numbers, not to anything at all."""
        from mlsynth.exceptions import MlsynthConfigError
        from mlsynth.utils.sdid_helpers.weights import unit_weights

        Y = np.random.default_rng(0).normal(size=(8, 3))
        with pytest.raises(MlsynthConfigError):
            unit_weights(Y, Y[:, 0], bad)


class TestTheOtherSitesWithTheSamePredicate:
    """Same mistake, three other places. Fixed together so the class is closed."""

    @pytest.mark.parametrize("budget", [np.float32(9.0), np.float64(9.0),
                                        np.int64(9), 9.0])
    def test_marex_accepts_a_numpy_budget(self, budget):
        """``isinstance(budget, (int, float))`` is a *branch* here, so a numpy
        scalar fell through to 'budget must be a scalar or dict' -- which it is."""
        from mlsynth.utils.marex_helpers.formulation import validate_costs_budget

        costs, per_cluster = validate_costs_budget(
            costs=np.ones(4), budget=budget, N=4, K=2,
            cluster_labels=["a", "b"])
        assert costs.shape == (4,)
        assert set(per_cluster) == {"a", "b"}
        assert all(v == pytest.approx(4.5) for v in per_cluster.values())

    @pytest.mark.parametrize("val", [np.float32(0.5), np.float64(0.5), 0.5])
    def test_clustersc_posterior_accepts_numpy_scalars(self, val):
        from mlsynth.utils.clustersc_helpers.pcr.bayesian.posterior import BayesSCM

        rng = np.random.default_rng(3)
        donors = rng.normal(size=(10, 3))
        target = donors @ np.array([0.5, 0.3, 0.2])
        w, cf, *_ = BayesSCM(donors, target, val, val)
        assert np.isfinite(w).all()

    @pytest.mark.parametrize("grid", [[np.float32(0.5), np.float32(1.0)],
                                      np.array([0.5, 1.0], dtype="float32")])
    def test_shc_accepts_a_numpy_bandwidth_grid(self, grid):
        from mlsynth.utils.shc_helpers.config import SHCConfig

        df = _panel("float64")
        cfg = SHCConfig(df=df, outcome="y", treat="treat", unitid="unit",
                        time="t", bandwidth_grid=list(grid))
        assert len(cfg.bandwidth_grid) == 2

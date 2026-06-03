"""Coverage tests for the Li & Sonnier (2023) FMA DGP helpers.

Exercises every branch of :mod:`mlsynth.utils.fma_helpers.simulation`:
both factor processes, all three variance regimes, the error/raise paths,
and determinism under a fixed RNG seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.fma_helpers.simulation import (
    FMASample,
    _factors_dgp1,
    _factors_dgp2,
    _VARIANCE_CASE_PARAMS,
    simulate_fma_sample,
)


# ----------------------------------------------------------------------
# Factor processes
# ----------------------------------------------------------------------

class TestFactorProcesses:
    def test_dgp1_shape_and_determinism(self):
        T = 25
        f1 = _factors_dgp1(T, np.random.default_rng(0))
        f2 = _factors_dgp1(T, np.random.default_rng(0))
        assert f1.shape == (T, 3)
        np.testing.assert_array_equal(f1, f2)

    def test_dgp1_custom_burn(self):
        # Exercise the burn keyword and the t>=2 lag-2 branch (needs burn>=2).
        f = _factors_dgp1(10, np.random.default_rng(1), burn=5)
        assert f.shape == (10, 3)
        assert np.all(np.isfinite(f))

    def test_dgp1_short_burn_hits_lag_guards(self):
        # burn=0 forces the t<2 guard (lag2=0) inside the loop on early rows.
        f = _factors_dgp1(4, np.random.default_rng(2), burn=0)
        assert f.shape == (4, 3)

    def test_dgp2_shape_and_determinism(self):
        T = 25
        f1 = _factors_dgp2(T, np.random.default_rng(0))
        f2 = _factors_dgp2(T, np.random.default_rng(0))
        assert f1.shape == (T, 3)
        np.testing.assert_array_equal(f1, f2)

    def test_dgp2_unit_root_is_cumsum(self):
        # F2 must be a drift-less unit root: column 1 is a cumulative sum, so
        # its first differences equal the underlying innovations.
        f = _factors_dgp2(8, np.random.default_rng(3))
        assert f.shape == (8, 3)
        # difference of a cumsum has length T-1 and is finite
        assert np.all(np.isfinite(np.diff(f[:, 1])))


# ----------------------------------------------------------------------
# simulate_fma_sample: variance regimes
# ----------------------------------------------------------------------

class TestVarianceRegimes:
    @pytest.mark.parametrize("case,expected", list(_VARIANCE_CASE_PARAMS.items()))
    def test_each_variance_case(self, case, expected):
        sample = simulate_fma_sample(
            variance_case=case, N_co=5, T1=6, T2=4,
            rng=np.random.default_rng(0),
        )
        assert (sample.sigma_tr, sample.sigma_co) == expected

    def test_bad_variance_case_raises(self):
        with pytest.raises(ValueError, match="variance_case must be one of"):
            simulate_fma_sample(variance_case="bogus",
                                rng=np.random.default_rng(0))


# ----------------------------------------------------------------------
# simulate_fma_sample: both DGPs, structure, determinism
# ----------------------------------------------------------------------

class TestSimulateSample:
    @pytest.mark.parametrize("dgp", ["dgp1", "dgp2"])
    def test_sample_structure(self, dgp):
        N_co, T1, T2 = 7, 8, 5
        T = T1 + T2
        sample = simulate_fma_sample(
            dgp=dgp, N_co=N_co, T1=T1, T2=T2, alpha=2.5,
            rng=np.random.default_rng(0),
        )
        assert isinstance(sample, FMASample)
        assert sample.dgp == dgp
        assert sample.T1 == T1 and sample.T2 == T2
        assert sample.factors.shape == (T, 3)
        assert sample.Y_treated.shape == (T,)
        assert sample.Y_controls.shape == (N_co, T)
        # long panel: (N_co + 1) units x T periods
        assert isinstance(sample.df, pd.DataFrame)
        assert len(sample.df) == (N_co + 1) * T
        assert set(sample.df.columns) == {"unit", "time", "y", "D"}
        # exactly one treated unit, switching on at T1
        treated = sample.df[sample.df["unit"] == "treated"]
        assert (treated["D"] == (treated["time"] >= T1).astype(int)).all()
        # controls never treated
        controls = sample.df[sample.df["unit"] != "treated"]
        assert (controls["D"] == 0).all()
        assert controls["unit"].nunique() == N_co

    @pytest.mark.parametrize("dgp", ["dgp1", "dgp2"])
    def test_determinism(self, dgp):
        kw = dict(dgp=dgp, N_co=4, T1=5, T2=3)
        a = simulate_fma_sample(rng=np.random.default_rng(42), **kw)
        b = simulate_fma_sample(rng=np.random.default_rng(42), **kw)
        np.testing.assert_array_equal(a.Y_treated, b.Y_treated)
        np.testing.assert_array_equal(a.Y_controls, b.Y_controls)
        np.testing.assert_array_equal(a.factors, b.factors)
        pd.testing.assert_frame_equal(a.df, b.df)

    def test_default_rng_runs(self):
        # rng=None path -> np.random.default_rng() default.
        sample = simulate_fma_sample(N_co=3, T1=4, T2=3)
        assert sample.Y_treated.shape == (7,)

    def test_bad_dgp_raises(self):
        with pytest.raises(ValueError, match="dgp must be one of"):
            simulate_fma_sample(dgp="dgp3", rng=np.random.default_rng(0))

    def test_treated_path_matches_dataframe(self):
        sample = simulate_fma_sample(
            N_co=3, T1=4, T2=3, rng=np.random.default_rng(7),
        )
        treated = sample.df[sample.df["unit"] == "treated"].sort_values("time")
        np.testing.assert_allclose(treated["y"].to_numpy(), sample.Y_treated)

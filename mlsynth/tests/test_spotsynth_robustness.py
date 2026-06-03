"""Robustness tests for the SPOTSYNTH subsystem.

Mirrors the bilevel robustness template: every guard added to the SPOTSYNTH
source is asserted here with ``pytest.raises`` / ``pytest.warns``, and the
happy path is checked to emit NO warning via a ``warnings_as_errors()``
context manager.

Guards under test
-----------------
* ``setup.prepare_spotsynth_inputs`` -- missing columns, NaN outcome/treat,
  duplicate (unit, time) rows, too few units, unbalanced panel, no treated
  unit, multiple treated units, treated-at-earliest, too few pre/post periods.
* ``screen.spillover_screen`` -- bad D shape/finiteness, T0 out of range,
  donor_names length mismatch, unknown selection, unknown forecast,
  (near-)constant donor warning, S2-empties-pool warning.
* ``bayes.bayesian_simplex_sc`` -- bad y/D shapes, finiteness, T0 range.
* ``debias.proximal_debias`` -- bad matrix dims, T0 range.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.spotsynth_helpers import (
    prepare_spotsynth_inputs,
    proximal_debias,
    simulate_spillover_panel,
    spillover_screen,
)
from mlsynth.utils.spotsynth_helpers.bayes import bayesian_simplex_sc


@contextlib.contextmanager
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _panel(**kw):
    df, _ = simulate_spillover_panel(**kw)
    return df


# ----------------------------------------------------------------------
# setup.py guards
# ----------------------------------------------------------------------

class TestSetupGuards:
    def test_missing_columns(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0).drop(columns=["Y"])
        with pytest.raises(MlsynthDataError, match="Missing columns"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_nan_outcome(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df.loc[0, "Y"] = np.nan
        with pytest.raises(MlsynthDataError, match="Outcome column contains NaN"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_nan_treat(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df["treated"] = df["treated"].astype(float)
        df.loc[0, "treated"] = np.nan
        with pytest.raises(MlsynthDataError, match="Treatment column contains NaN"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_duplicate_unit_time(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(MlsynthDataError, match="duplicate"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_too_few_units(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df = df[df["unit"].isin(["target", "d0"])]
        with pytest.raises(MlsynthDataError, match="at least 2 donors"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_unbalanced(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0).iloc[1:]
        with pytest.raises(MlsynthDataError, match="unbalanced"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_no_treated_unit(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df["treated"] = 0
        with pytest.raises(MlsynthDataError, match="No treated unit"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_multiple_treated_units(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df.loc[(df["unit"] == "d0") & (df["time"] >= 18), "treated"] = 1
        with pytest.raises(MlsynthDataError, match="single treated unit"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_treated_at_earliest_period(self):
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df.loc[(df["unit"] == "target"), "treated"] = 1
        with pytest.raises(MlsynthDataError, match="earliest period"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_too_few_pre_periods(self):
        # treatment turns on at t=1 -> only 1 pre-period.
        df = _panel(n_donors=6, T0=18, n_post=5, seed=0)
        df["treated"] = ((df["unit"] == "target") & (df["time"] >= 1)).astype(int)
        with pytest.raises(MlsynthDataError, match="pre-intervention"):
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")


# ----------------------------------------------------------------------
# screen.py guards
# ----------------------------------------------------------------------

class TestScreenGuards:
    def _D(self, seed=0):
        rng = np.random.default_rng(seed)
        T, n = 30, 6
        return rng.normal(size=(T, n)), 20, [f"d{j}" for j in range(n)]

    def test_bad_ndim(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            spillover_screen(np.zeros(10), 6, ["a"])

    def test_no_donor_columns(self):
        with pytest.raises(MlsynthDataError, match="no donor columns"):
            spillover_screen(np.zeros((10, 0)), 6, [])

    def test_t0_out_of_range(self):
        D, _, names = self._D()
        with pytest.raises(MlsynthDataError, match="T0 must satisfy"):
            spillover_screen(D, 0, names)
        with pytest.raises(MlsynthDataError, match="T0 must satisfy"):
            spillover_screen(D, D.shape[0], names)

    def test_donor_names_length_mismatch(self):
        D, T0, _ = self._D()
        with pytest.raises(MlsynthDataError, match="donor_names has length"):
            spillover_screen(D, T0, ["d0", "d1"])

    def test_non_finite_D(self):
        D, T0, names = self._D()
        D[0, 0] = np.inf
        with pytest.raises(MlsynthDataError, match="non-finite"):
            spillover_screen(D, T0, names)

    def test_unknown_selection(self):
        D, T0, names = self._D()
        with pytest.raises(ValueError, match="Unknown selection"):
            spillover_screen(D, T0, names, selection="bogus")

    def test_unknown_forecast(self):
        D, T0, names = self._D()
        with pytest.raises(ValueError, match="Unknown forecast"):
            spillover_screen(D, T0, names, forecast="bogus")

    def test_constant_donor_warns(self):
        D, T0, names = self._D()
        D[:, 0] = 1.0  # constant donor -> zero pre-variance
        with pytest.warns(RuntimeWarning, match="near-.constant"):
            spillover_screen(D, T0, names, selection="all")

    def test_s2_empties_pool_warns(self):
        df = _panel(n_donors=8, T0=20, n_post=6, sigma_x=0.3, seed=4)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        with pytest.warns(RuntimeWarning, match="fewer than 2 donors"):
            spillover_screen(inp.D, inp.T0, inp.donor_names,
                             selection="S2", forecast="lag", ppi=0.01)


# ----------------------------------------------------------------------
# bayes.py guards (no MCMC: validation happens before the sampler)
# ----------------------------------------------------------------------

class TestBayesGuards:
    def test_bad_D_ndim(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            bayesian_simplex_sc(np.arange(10.0), np.arange(10.0), 6)

    def test_y_D_axis_mismatch(self):
        with pytest.raises(MlsynthDataError, match="share the T axis"):
            bayesian_simplex_sc(np.arange(8.0), np.zeros((10, 3)), 6)

    def test_no_donor_columns(self):
        with pytest.raises(MlsynthDataError, match="at least one donor"):
            bayesian_simplex_sc(np.arange(10.0), np.zeros((10, 0)), 6)

    def test_t0_out_of_range(self):
        with pytest.raises(MlsynthDataError, match="T0 must satisfy"):
            bayesian_simplex_sc(np.arange(10.0), np.zeros((10, 3)), 0)

    def test_non_finite(self):
        D = np.zeros((10, 3))
        D[0, 0] = np.nan
        with pytest.raises(MlsynthDataError, match="non-finite"):
            bayesian_simplex_sc(np.arange(10.0), D, 6)


# ----------------------------------------------------------------------
# debias.py guards
# ----------------------------------------------------------------------

class TestDebiasGuards:
    def test_bad_dims(self):
        with pytest.raises(MlsynthDataError, match="2-D"):
            proximal_debias(np.arange(10.0), np.arange(10.0),
                            np.zeros((10, 2)), 6)

    def test_t0_out_of_range(self):
        with pytest.raises(MlsynthDataError, match="T0 must satisfy"):
            proximal_debias(np.arange(10.0), np.zeros((10, 2)),
                            np.zeros((10, 3)), 0)


# ----------------------------------------------------------------------
# Happy path emits NO warning
# ----------------------------------------------------------------------

class TestHappyPathNoWarnings:
    def test_screen_no_warning(self):
        df = _panel(n_donors=10, T0=24, n_post=6, sigma_x=0.3, seed=0)
        inp = prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")
        with warnings_as_errors():
            spillover_screen(inp.D, inp.T0, inp.donor_names,
                             selection="S1", forecast="loo", n_donors=4)
            spillover_screen(inp.D, inp.T0, inp.donor_names, selection="all")

    def test_setup_no_warning(self):
        df = _panel(n_donors=10, T0=24, n_post=6, seed=0)
        with warnings_as_errors():
            prepare_spotsynth_inputs(df, "Y", "treated", "unit", "time")

    def test_debias_no_warning(self):
        rng = np.random.default_rng(0)
        T, T0, k = 30, 22, 3
        X = rng.normal(size=(T, k))
        y = X @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.1, T)
        Z = rng.normal(size=(T, 6))
        with warnings_as_errors():
            proximal_debias(y, X, Z, T0)

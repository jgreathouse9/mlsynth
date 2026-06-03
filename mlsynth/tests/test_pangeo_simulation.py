"""Coverage tests for mlsynth.utils.pangeo_helpers.simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.pangeo_helpers.simulation import make_seasonal_sales_panel


def test_default_rw_panel():
    df = make_seasonal_sales_panel(units_per_arm=3, arms=("A", "B"), T=20,
                                   season_period=12, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"unit", "time", "sales", "arm"}
    assert df["arm"].nunique() == 2
    assert df["unit"].nunique() == 6
    assert len(df) == 6 * 20


@pytest.mark.parametrize("factor", ["rw", "iid", "ar1"])
def test_factor_processes(factor):
    df = make_seasonal_sales_panel(units_per_arm=2, arms=("A",), T=15,
                                   season_period=12, factor=factor, seed=1)
    assert len(df) == 2 * 15
    assert df["arm"].unique().tolist() == ["A"]


def test_unknown_factor_raises():
    with pytest.raises(ValueError, match="unknown factor process"):
        make_seasonal_sales_panel(units_per_arm=2, arms=("A",), T=10,
                                  factor="bogus", seed=0)


def test_covariates_columns():
    df = make_seasonal_sales_panel(units_per_arm=2, arms=("A",), T=10,
                                   covariates=True, seed=0)
    assert "population" in df.columns
    assert "income" in df.columns


def test_n_post_adds_post_col():
    df = make_seasonal_sales_panel(units_per_arm=2, arms=("A",), T=8,
                                   n_post=4, season_period=12, seed=0)
    assert "post_col" in df.columns
    assert df["time"].max() == 11  # T + n_post - 1
    assert df[df["time"] >= 8]["post_col"].eq(1).all()
    assert df[df["time"] < 8]["post_col"].eq(0).all()


def test_determinism():
    a = make_seasonal_sales_panel(units_per_arm=2, arms=("A", "B"), T=10,
                                  season_period=12, seed=42)
    b = make_seasonal_sales_panel(units_per_arm=2, arms=("A", "B"), T=10,
                                  season_period=12, seed=42)
    pd.testing.assert_frame_equal(a, b)

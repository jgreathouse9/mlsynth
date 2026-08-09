"""The GEOX name, and the SDIDGEO names it replaces.

The estimator was called SDIDGEO while synthetic DiD was the only thing that
could score a design. It now dispatches over engines, so that name says which
engine is running instead of what the estimator does, and the design harness is
what the module actually is. GEOX is the name; the SDIDGEO spellings stay as
aliases so existing scripts keep running.

The aliases are bindings to the same objects, not subclasses. A subclass would
compare unequal under ``is``, would appear separately in ``isinstance`` chains,
and would let the two drift apart. Identity is what makes them one thing under
two spellings, so every alias test here asserts identity.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import mlsynth


@pytest.fixture(scope="module")
def panel():
    """A small factor-model geo panel: 10 markets, 24 periods."""
    rng = np.random.default_rng(0)
    n_units, n_periods = 10, 24
    factors = rng.normal(size=(n_periods, 2))
    loadings = rng.normal(size=(n_units, 2)) + 3.0
    y = 100.0 + factors @ loadings.T + rng.normal(scale=0.5,
                                                  size=(n_periods, n_units))
    return pd.DataFrame({
        "unit": np.repeat([f"m{i}" for i in range(n_units)], n_periods),
        "time": np.tile(np.arange(n_periods), n_units),
        "y": y.T.ravel(),
    })


# --- the new name, at every level of the surface -------------------------


def test_geox_is_exported_from_the_package():
    from mlsynth import GEOX

    assert GEOX is mlsynth.GEOX
    assert "GEOX" in mlsynth.__all__


def test_geox_config_resolves_through_config_models():
    from mlsynth.config_models import GEOXConfig
    from mlsynth.utils.geox_helpers.config import GEOXConfig as Direct

    assert GEOXConfig is Direct


def test_geox_results_lives_in_the_helper_package():
    from mlsynth.config_models import DesignResult
    from mlsynth.utils.geox_helpers.structures import GEOXResults

    assert issubclass(GEOXResults, DesignResult)


def test_geox_plot_is_exported():
    from mlsynth import plot_geox_design

    assert callable(plot_geox_design)
    assert "plot_geox_design" in mlsynth.__all__


@pytest.mark.parametrize("module", [
    "aggregate", "batch", "candidates", "config", "constraints", "engine",
    "engines", "feasibility", "orchestration", "plotter", "realize",
    "shaping", "simulate", "structures", "windows",
])
def test_helper_modules_import_under_the_new_path(module):
    importlib.import_module(f"mlsynth.utils.geox_helpers.{module}")


@pytest.mark.parametrize("engine", ["sdid", "augsynth"])
def test_engine_subpackages_move_with_the_helper_package(engine):
    mod = importlib.import_module(f"mlsynth.utils.geox_helpers.engines.{engine}")
    assert mod.ENGINE.name == engine


def test_feasibility_audit_renamed():
    from mlsynth.utils.geox_helpers.feasibility import audit_geox_feasibility

    assert callable(audit_geox_feasibility)


# --- the old spellings, still bound to the same objects ------------------


def test_sdidgeo_estimator_alias_is_the_same_class():
    from mlsynth import GEOX, SDIDGEO

    assert SDIDGEO is GEOX
    assert "SDIDGEO" in mlsynth.__all__


def test_sdidgeo_config_alias_is_the_same_class():
    from mlsynth.config_models import GEOXConfig, SDIDGEOConfig

    assert SDIDGEOConfig is GEOXConfig


def test_sdidgeo_config_alias_resolves_from_the_helper_module_too():
    from mlsynth.utils.geox_helpers.config import GEOXConfig, SDIDGEOConfig

    assert SDIDGEOConfig is GEOXConfig


def test_sdidgeo_results_alias_is_the_same_class():
    from mlsynth.utils.geox_helpers.structures import (
        GEOXResults,
        SDIDGEOResults,
    )

    assert SDIDGEOResults is GEOXResults


def test_sdidgeo_plot_alias_is_the_same_function():
    from mlsynth import plot_geox_design, plot_sdidgeo_design

    assert plot_sdidgeo_design is plot_geox_design


def test_reaching_an_alias_does_not_warn():
    """The rename is documented, not enforced at runtime.

    ``SDIDGEOConfig`` is not in ``config_models.__dict__``, so every access runs
    the module ``__getattr__`` -- this is a live check of the path, not of a
    cached name. The module is deliberately not reloaded: reloading rebinds
    ``GEOXResults`` to a fresh class object while the estimator holds the
    original, and every later ``isinstance`` in this file would fail.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import mlsynth.config_models as cm

        cm.SDIDGEOConfig
        mlsynth.SDIDGEO
        mlsynth.plot_sdidgeo_design

    assert caught == []


# --- the old spellings are gone from the tree ----------------------------


@pytest.mark.parametrize("path", [
    "mlsynth.estimators.sdidgeo",
    "mlsynth.utils.sdidgeo_helpers",
    "mlsynth.utils.sdidgeo_helpers.orchestration",
])
def test_old_module_paths_are_not_kept(path):
    """Class aliases cover the names a user writes; module aliases do not.

    Keeping ``mlsynth.utils.sdidgeo_helpers`` importable would mean keeping
    every submodule importable under both paths, doubling the surface for an
    import path no documented example uses.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


# --- the aliases actually run --------------------------------------------


def test_a_design_built_through_the_old_names_runs(panel):
    """End to end on the old spellings: config in, design out."""
    from mlsynth import SDIDGEO
    from mlsynth.config_models import SDIDGEOConfig
    from mlsynth.utils.geox_helpers.structures import GEOXResults

    design = SDIDGEO(SDIDGEOConfig(
        df=panel, unitid="unit", time="time", outcome="y",
        treatment_size=2, durations=[3], effect_sizes=[-0.2, 0.0, 0.2],
        n_backtests=2, n_draws=20, seed=0, n_jobs=1,
    )).fit()

    assert isinstance(design, GEOXResults)
    assert len(design.selected_units) == 2


def test_the_two_spellings_give_the_same_design(panel):
    """Same inputs, either spelling, identical answer."""
    from mlsynth import GEOX, SDIDGEO
    from mlsynth.config_models import GEOXConfig, SDIDGEOConfig

    kwargs = dict(
        df=panel, unitid="unit", time="time", outcome="y",
        treatment_size=2, durations=[3], effect_sizes=[-0.2, 0.0, 0.2],
        n_backtests=2, n_draws=20, seed=0, n_jobs=1,
    )
    new = GEOX(GEOXConfig(**kwargs)).fit()
    old = SDIDGEO(SDIDGEOConfig(**kwargs)).fit()

    assert new.selected_units == old.selected_units
    assert (new.metadata["winner_mde_planning"]
            == old.metadata["winner_mde_planning"])
    pd.testing.assert_frame_equal(new.power, old.power)

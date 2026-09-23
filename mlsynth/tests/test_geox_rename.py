"""The GEOX name, and the absence of the SDIDGEO names it replaces.

The estimator was called SDIDGEO while synthetic DiD was the only thing that
could score a design. It now dispatches over engines, so that name says which
engine is running instead of what the estimator does, and the design harness is
what the module actually is.

The old spellings are gone, not aliased. An alias is a second public name
that never stops being supported, and it costs more than the line that defines
it: the index generator counted it as a second estimator, and every later reader
has to work out whether the two names differ. Removing them makes the failure a
clean ``ImportError`` at the import line, which says what happened and where.

These tests pin both halves -- every new name resolves, and every old one is
absent -- so a partially-applied rename fails here instead of shipping.
"""

import importlib

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


def test_the_export_name_is_the_class_name():
    """No alias hiding behind the export.

    ``gen_llms_txt.py`` walks ``__all__`` for classes and indexes each one, so a
    class exported under a second name appears twice in the agent-facing index
    with the same description.
    """
    assert mlsynth.GEOX.__name__ == "GEOX"


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
    "shaping", "similarity", "simulate", "structures", "windows",
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


def test_the_estimator_runs_under_the_new_name(panel):
    """End to end: config in, design out."""
    from mlsynth import GEOX
    from mlsynth.config_models import GEOXConfig
    from mlsynth.utils.geox_helpers.structures import GEOXResults

    design = GEOX(GEOXConfig(
        df=panel, unitid="unit", time="time", outcome="y",
        treatment_size=2, durations=[3], effect_sizes=[-0.2, 0.0, 0.2],
        n_backtests=2, n_draws=20, seed=0, n_jobs=1,
    )).fit()

    assert isinstance(design, GEOXResults)
    assert len(design.selected_units) == 2


# --- the old names are gone ----------------------------------------------


@pytest.mark.parametrize("name", [
    "SDIDGEO", "SDIDGEOConfig", "SDIDGEOResults", "plot_sdidgeo_design",
])
def test_old_names_are_absent_from_the_package(name):
    assert not hasattr(mlsynth, name)
    assert name not in mlsynth.__all__


@pytest.mark.parametrize("name", ["SDIDGEOConfig", "SDIDGEOResults"])
def test_old_names_are_absent_from_config_models(name):
    """The lazy config re-export must not resolve them either.

    ``config_models`` resolves relocated config classes through a module
    ``__getattr__``, which answers for any name in its table. An entry left
    behind there would keep ``from mlsynth.config_models import SDIDGEOConfig``
    working long after the class stopped existing under that name.
    """
    import mlsynth.config_models as cm

    with pytest.raises(AttributeError):
        getattr(cm, name)
    assert name not in cm._RELOCATED_CONFIGS


@pytest.mark.parametrize("module, name", [
    ("mlsynth.estimators.geox", "SDIDGEO"),
    ("mlsynth.utils.geox_helpers.config", "SDIDGEOConfig"),
    ("mlsynth.utils.geox_helpers.structures", "SDIDGEOResults"),
    ("mlsynth.utils.geox_helpers.plotter", "plot_sdidgeo_design"),
])
def test_old_names_are_absent_from_their_defining_modules(module, name):
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize("path", [
    "mlsynth.estimators.sdidgeo",
    "mlsynth.utils.sdidgeo_helpers",
    "mlsynth.utils.sdidgeo_helpers.orchestration",
    "mlsynth.utils.sdidgeo_helpers.engines",
])
def test_old_module_paths_are_gone(path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


def _old_name_hits(suffix, exclude=()):
    """Tracked files with the given suffix that contain the old name."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True,
    ).stdout.split("\0")

    hits = []
    for rel in tracked:
        if not rel or not rel.endswith(suffix) or rel in exclude:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):     # binary or unreadable
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "sdidgeo" in line.lower():
                hits.append(f"{rel}:{i}: {line.strip()}")
    return hits


def test_no_python_source_mentions_the_old_name():
    """A grep, as a test.

    In Python the old name can only be a binding, an import, or a docstring
    describing one, and all three are things this rename removed. Any hit is a
    missed reference or an alias someone put back. This file is exempt because
    naming the old spellings is how it asserts their absence.
    """
    hits = _old_name_hits(".py", exclude={"mlsynth/tests/test_geox_rename.py"})
    assert hits == [], "old name still in Python source:\n" + "\n".join(hits)


def test_only_the_migration_note_mentions_the_old_name_in_docs():
    """Prose may name the old spelling; exactly one page should.

    A reader arriving with ``SDIDGEO`` in their script needs the page to say
    what happened to it, so ``docs/geox.rst`` names it deliberately. A hit
    anywhere else is a page that was not updated.
    """
    hits = _old_name_hits(".rst", exclude={"docs/geox.rst"})
    assert hits == [], "old name outside the migration note:\n" + "\n".join(hits)

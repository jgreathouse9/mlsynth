"""TBR's engine under TBRMM, and the absence of the standalone helper package.

TBR is the analysis model a TBRMM design is scored against. Au's objective is
TBR's own posterior scale read backwards, so every candidate split TBRMM
evaluates is a TBR pretest fit, and nothing else in the library fits one. The
helpers therefore live under the dispatcher that drives them,
``mlsynth/utils/tbrmm_helpers/engine/``, which is the layout
``spillsynth_helpers/sar/`` and ``geox_helpers/engines/`` already use.

TBR stays an exported estimator with its own documentation page. An advertiser
whose groups were fixed by someone else runs TBR and never runs a search, so
removing the public class would remove a method people use. What moves is where
its helpers sit, not whether it is reachable.

The old spellings are gone, not aliased, following ``test_geox_rename.py``: an
alias is a second import path that never stops being supported, and a reader
who finds both has to work out whether they differ. A removed path fails as an
``ImportError`` at the import line, which says what happened and where.

Two invariants here outlive the move itself. The engine is reached only from
TBR's own estimator module and from TBRMM's helpers, which is the claim that
makes it an engine instead of a second library; and the two group-flag readers
are no longer imported across package boundaries, which is the specific reach
``tbrmm_helpers/pipeline.py`` performed before the move.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import numpy as np
import pandas as pd
import pytest

import mlsynth

ENGINE = "mlsynth.utils.tbrmm_helpers.engine"
ENGINE_MODULES = ("config", "pipeline", "plotter", "posterior", "setup",
                  "structures")

#: Where the engine may be imported from. TBR's thin estimator class builds its
#: config, pipeline and results from it; TBRMM's helpers score candidate splits
#: with it. Any third importer means the engine is being used as a library and
#: the containment claim in the module docstring has stopped holding.
ENGINE_IMPORTERS = ("mlsynth/estimators/tbr.py",
                    "mlsynth/utils/tbrmm_helpers/")

#: The group-flag readers that ``tbrmm_helpers/pipeline.py`` reached across
#: packages for before the move.
GROUP_FLAG_READERS = ("_binary_unit_flag", "_block_flag_start")

PACKAGE_ROOT = pathlib.Path(mlsynth.__file__).resolve().parent


# --------------------------------------------------------------------------- #
# panel builder
# --------------------------------------------------------------------------- #
def geo_panel(n_control=4, n_treat=3, T=20, T0=14, alpha=5.0, beta=1.5,
              seed=0):
    """A geo panel whose group aggregates satisfy the pretest relation exactly.

    The treatment sum is defined as ``alpha + beta * (control sum)``, so with no
    noise and no lift the cumulative effect is zero by construction. That is
    enough for a smoke test: the point here is that the moved engine still
    runs and still returns finite numbers, not what those numbers are, which
    ``test_tbr.py`` asserts against the paper's closed forms.
    """
    rng = np.random.default_rng(seed)
    ctl = rng.uniform(10.0, 30.0, size=(T, n_control))
    target = alpha + beta * ctl.sum(axis=1)
    trt = np.tile((target / n_treat)[:, None], (1, n_treat))

    rows = []
    for block, tag, arr in (("c", "control", ctl), ("t", "treat", trt)):
        for j in range(arr.shape[1]):
            for t in range(T):
                rows.append(dict(geo=f"{block}{j}", date=t,
                                 sales=float(arr[t, j]),
                                 is_control=int(tag == "control"),
                                 D=int(tag == "treat" and t >= T0)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return geo_panel()


# --------------------------------------------------------------------------- #
# import resolution, so a relative import is checked as what it resolves to
# --------------------------------------------------------------------------- #
def _module_name(path: pathlib.Path) -> str:
    """The dotted name of a file inside the installed package."""
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, module: str) -> str:
    """The absolute dotted module an ``ImportFrom`` names.

    ``node.level`` is the number of leading dots. A dot count of ``n`` strips
    ``n - 1`` trailing components from the importing module's package, so the
    same statement text resolves differently depending on how deep the file
    sits -- which is the detail that makes a textual grep the wrong instrument
    for this check.
    """
    if not node.level:
        return node.module or ""
    package = module.rsplit(".", 1)[0] if "." in module else module
    parts = package.split(".")
    if node.level > 1:
        parts = parts[: -(node.level - 1)]
    base = ".".join(parts)
    return f"{base}.{node.module}" if node.module else base


def _source_files():
    """Every module in the package except the tests, which import freely."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(PACKAGE_ROOT).parts:
            continue
        yield path


def _import_sites():
    """``(repo-relative path, resolved module, imported names)`` per statement."""
    for path in _source_files():
        module = _module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = str(path.relative_to(PACKAGE_ROOT.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = tuple(a.name for a in node.names)
                yield rel, _resolve(node, module), names
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    yield rel, alias.name, ()


# --------------------------------------------------------------------------- #
# the new home resolves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ENGINE_MODULES)
def test_every_engine_module_resolves_at_its_new_home(name):
    assert importlib.import_module(f"{ENGINE}.{name}") is not None


ENGINE_EXPORTS = ("TBRConfig", "TBRFit", "CumulativeEffect", "IROASResult",
                  "TBRResults")


def test_the_engine_package_exports_the_result_models():
    engine = importlib.import_module(ENGINE)
    for name in ENGINE_EXPORTS:
        assert hasattr(engine, name), name


def test_the_dispatcher_surfaces_the_engine_names():
    """``tbrmm_helpers`` re-exports what its engine exports.

    ``spillsynth_helpers/__init__.py`` lifts ``run_cd``, ``SARFit`` and the rest
    out of its method subpackages the same way, so one import reaches the whole
    family. The import also proves the package initialises without a cycle,
    which is the failure a dispatcher importing its own engine could introduce.
    """
    helpers = importlib.import_module("mlsynth.utils.tbrmm_helpers")
    for name in ENGINE_EXPORTS + ("TBRMMConfig", "TBRMMDesign",
                                  "TBRMMResults"):
        assert name in helpers.__all__, name
        assert hasattr(helpers, name), name


# --------------------------------------------------------------------------- #
# the old home is absent
# --------------------------------------------------------------------------- #
def test_the_standalone_helper_package_is_gone():
    """Not aliased to the engine, absent.

    A shim here would leave two import paths for one module, and the benchmark
    dependency map records paths, so both would have to be maintained in it.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mlsynth.utils.tbr_helpers")


@pytest.mark.parametrize("name", ENGINE_MODULES)
def test_no_module_survives_at_the_old_path(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"mlsynth.utils.tbr_helpers.{name}")


def test_no_source_file_mentions_the_old_package():
    """The move is applied everywhere or it fails here.

    A partially-applied move leaves a working import in one file and a broken
    one in another, and the broken one surfaces only when that path runs.
    """
    offenders = [rel for rel, module, _ in _import_sites()
                 if module.startswith("mlsynth.utils.tbr_helpers")]
    assert offenders == []


# --------------------------------------------------------------------------- #
# TBR is still a public estimator
# --------------------------------------------------------------------------- #
def test_tbr_is_still_exported_from_the_package():
    from mlsynth import TBR

    assert TBR is mlsynth.TBR
    assert "TBR" in mlsynth.__all__
    assert mlsynth.TBR.__name__ == "TBR"


def test_tbr_config_still_resolves_from_config_models():
    """The PEP 562 hook in ``config_models`` keeps the documented import path.

    ``from mlsynth.config_models import TBRConfig`` is what every example and
    every docs page uses, so it survives the relocation of the class it points
    at. This is invariant 1 in ``CLAUDE.md``.
    """
    from mlsynth.config_models import TBRConfig

    engine_config = importlib.import_module(f"{ENGINE}.config")
    assert TBRConfig is engine_config.TBRConfig


def test_the_relocation_map_points_at_the_engine():
    from mlsynth.config_models import _RELOCATED_CONFIGS

    assert _RELOCATED_CONFIGS["TBRConfig"] == f"{ENGINE}.config"


def test_tbr_still_fits_through_the_moved_engine(panel):
    from mlsynth import TBR
    from mlsynth.config_models import TBRConfig

    result = TBR(TBRConfig(df=panel, unitid="geo", time="date",
                           outcome="sales", treat="D",
                           control_col="is_control",
                           display_graphs=False)).fit()
    assert np.isfinite(result.effects.additional_effects["cumulative_effect"])
    assert np.isfinite(result.effects.att)


def test_tbrmm_still_fits_through_the_moved_engine(panel):
    from mlsynth import TBRMM
    from mlsynth.config_models import TBRMMConfig

    result = TBRMM(TBRMMConfig(df=panel, unitid="geo", time="date",
                               outcome="sales", max_treatment_size=2,
                               n_test=4)).fit()
    assert result.designs
    assert all(np.isfinite(d.objective_value) for d in result.designs)


# --------------------------------------------------------------------------- #
# containment: the engine is an engine, not a second library
# --------------------------------------------------------------------------- #
def test_the_engine_is_reached_only_from_its_allowlist():
    """Who may import the engine, asserted over the whole package.

    TBRMM is the only caller that fits a TBR pretest, and TBR's own estimator
    class is the public door onto the same code. A third importer would mean
    some other estimator had taken a dependency on the analysis model of a geo
    experiment, which is the coupling this layout exists to make visible.
    """
    reaches = {rel for rel, module, _ in _import_sites()
               if module.startswith(ENGINE)}
    unexpected = {rel for rel in reaches
                  if not rel.startswith(ENGINE_IMPORTERS)}
    assert unexpected == set(), (
        f"these modules import the TBR engine but are not in the allowlist: "
        f"{sorted(unexpected)}")


def test_the_group_flag_readers_are_no_longer_a_cross_package_reach():
    """``_binary_unit_flag`` and ``_block_flag_start`` stay inside the family.

    Before the move ``tbrmm_helpers/pipeline.py`` imported both from
    ``tbr_helpers.setup`` -- two private names crossing a package boundary,
    which is what said the two packages were one thing filed as two.
    """
    for rel, module, names in _import_sites():
        if not any(n in GROUP_FLAG_READERS for n in names):
            continue
        assert module.startswith("mlsynth.utils.tbrmm_helpers"), (
            f"{rel} imports {names} from {module}, outside the TBRMM family")


def test_the_engine_does_not_import_the_dispatcher():
    """The dependency runs one way.

    An engine that imports the search that drives it is a cycle, and the search
    is what is allowed to change when a different design objective is added.
    """
    for rel, module, _ in _import_sites():
        if not rel.startswith("mlsynth/utils/tbrmm_helpers/engine/"):
            continue
        assert not module.startswith("mlsynth.utils.tbrmm_helpers.search")
        assert not module.startswith("mlsynth.utils.tbrmm_helpers.objective")
        assert module != "mlsynth.utils.tbrmm_helpers.pipeline"

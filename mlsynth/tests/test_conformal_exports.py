"""The conformal package's export list must match what it actually re-exports.

``mlsynth/utils/conformal/__init__.py`` is a hand-curated surface: every module
in the package adds an import line and a name to ``__all__``, in two regions that
every contributor edits. Two branches adding modules therefore conflict there by
construction, and a conflict is resolved by hand under time pressure.

The two halves of that edit fail differently. Dropping an import line is caught
immediately -- nine test modules do ``from mlsynth.utils.conformal import ...``
and an absent name raises. Dropping the matching ``__all__`` entry is caught by
nothing: the direct import still resolves, so the package keeps working while its
declared surface silently shrinks, and only a downstream ``import *`` or a reader
trusting ``__all__`` ever finds out.

These tests close that half. They assert the list and the namespace agree in both
directions, so a resolution that keeps one and loses the other fails here instead
of later somewhere else.
"""
import importlib
import pkgutil
from types import ModuleType

import pytest

import mlsynth.utils.conformal as conformal

#: Names bound by the module machinery rather than exported on purpose.
_MACHINERY = {"annotations"}


def _public_namespace(package):
    """Public names the package binds that are not submodules of itself."""
    return {
        name for name, value in vars(package).items()
        if not name.startswith("_")
        and name not in _MACHINERY
        and not isinstance(value, ModuleType)
    }


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_the_package_declares_an_export_list():
    assert isinstance(conformal.__all__, list)
    assert conformal.__all__


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_every_declared_name_exists():
    """``__all__`` must not promise a name the package cannot supply -- the half
    of a botched merge resolution that keeps the list entry and loses the
    import."""
    missing = [n for n in conformal.__all__ if not hasattr(conformal, n)]
    assert not missing, f"declared in __all__ but not bound: {sorted(missing)}"


def test_every_re_exported_name_is_declared():
    """The half nothing else catches: an import line kept, its ``__all__`` entry
    lost. The package keeps working and its declared surface silently shrinks."""
    undeclared = _public_namespace(conformal) - set(conformal.__all__)
    assert not undeclared, (
        f"re-exported but absent from __all__: {sorted(undeclared)}")


def test_the_export_list_has_no_duplicates():
    """A duplicate is the other signature of a merge that kept both sides of a
    conflicting region."""
    seen = sorted(conformal.__all__)
    assert len(seen) == len(set(seen)), (
        f"duplicated in __all__: "
        f"{sorted({n for n in seen if seen.count(n) > 1})}")


def test_every_submodule_still_imports():
    """A resolution can also drop a module's import line entirely, which leaves
    the file importable and the module unreachable through the package."""
    for info in pkgutil.iter_modules(conformal.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{conformal.__name__}.{info.name}")


@pytest.mark.parametrize("name", sorted(conformal.__all__))
def test_each_declared_name_is_importable_by_itself(name):
    """``from mlsynth.utils.conformal import <name>`` is how every caller reaches
    these, so each one is exercised as a caller would."""
    module = importlib.import_module("mlsynth.utils.conformal")
    assert getattr(module, name) is not None

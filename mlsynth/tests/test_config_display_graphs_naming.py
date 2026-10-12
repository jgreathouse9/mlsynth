"""Every config spells the plot toggle the same way.

``BaseEstimatorConfig`` names it ``display_graphs``, and 72 of the 79 configs
exported from :mod:`mlsynth.config_models` followed. Five did not. The four
``BaseMAREXConfig`` children each declared a singular ``display_graph`` of their
own, because that base defines no graph field for them to inherit, and SIV
carried both spellings at once with opposite defaults.

Since every config is ``extra="forbid"``, a caller who learned
``display_graphs=False`` on one estimator got ``MlsynthConfigError`` on those
four, with an error naming the field as forbidden instead of naming the right
one. SIV's collision was worse than a typo trap: two different toggles one
character apart, the plural defaulting to on, so silencing the singular left
the main plot drawn.

These tests pin the name so a sixth variant cannot arrive unnoticed. The
singular is spelled by concatenation below so that a future sweep over this
repository cannot rewrite the thing being asserted, which is how the first
draft of this file was broken.
"""

from __future__ import annotations

import pytest

import mlsynth.config_models as cm

PLURAL = "display_graphs"
SINGULAR = "display_" + "graph"          # deliberately not a literal
assert SINGULAR != PLURAL, "the two spellings must differ for these tests"


def _exported_configs():
    """Every ``*Config`` reachable from :mod:`mlsynth.config_models`.

    The per-estimator configs are lazily re-exported from their helper
    packages, so they are resolved through the relocation registry, not read
    off the module's ``__dict__``.
    """
    out = {}
    for name in sorted(getattr(cm, "_RELOCATED_CONFIGS", {})):
        if not name.endswith("Config"):
            continue
        try:
            out[name] = getattr(cm, name)
        except Exception:  # pragma: no cover - an unimportable config is its
            continue       # own failure, owned by that estimator's suite
    for name, obj in vars(cm).items():
        if name.endswith("Config") and hasattr(obj, "model_fields"):
            out[name] = obj
    return out


def test_the_registry_resolves_enough_configs_to_be_a_real_sweep():
    """Guard the guard: a registry that resolved nothing would pass silently."""
    configs = _exported_configs()
    assert len(configs) >= 70, f"only {len(configs)} configs resolved"


def test_no_config_declares_the_singular_spelling():
    """The invariant, stated over every config at once."""
    offenders = sorted(
        name for name, cls in _exported_configs().items()
        if SINGULAR in cls.model_fields
    )
    assert offenders == [], (
        f"these configs declare {SINGULAR!r} where the library uses "
        f"{PLURAL!r}: {offenders}"
    )


@pytest.mark.parametrize(
    "name", ["LEXSCMConfig", "MAREXConfig", "SPCDConfig", "SYNDESConfig"])
def test_the_four_design_configs_expose_the_plural(name):
    """The four that had the singular now carry the library-wide name."""
    fields = getattr(cm, name).model_fields
    assert PLURAL in fields
    assert SINGULAR not in fields


def test_siv_keeps_two_toggles_under_two_names():
    """SIV genuinely has two of them, so they need two names.

    ``display_graphs`` is the library-wide toggle it inherits;
    ``display_event_study`` is its own, for the event-study coefficient plot.
    Collapsing them into one name would drop one of the two figures.
    """
    fields = cm.SIVConfig.model_fields
    assert PLURAL in fields
    assert "display_event_study" in fields
    assert SINGULAR not in fields
    assert fields["display_event_study"].default is False


def test_every_toggle_is_a_documented_bool():
    """A plot toggle may default either way, but it is a bool and it is
    described, because the description is what a user reads in the autodoc."""
    for name, cls in _exported_configs().items():
        if PLURAL not in cls.model_fields:
            continue
        fd = cls.model_fields[PLURAL]
        assert isinstance(fd.default, bool), f"{name}.{PLURAL} default"
        assert fd.description, f"{name}.{PLURAL} has no description"

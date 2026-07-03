"""PANGEO conforms to the DesignResult two-family result contract.

``PangeoResults`` is migrated from a bespoke frozen dataclass onto
:class:`~mlsynth.config_models.DesignResult`: it is a design (chooses the
treatment/control assignment before any intervention) that *resolves* to an
:class:`~mlsynth.config_models.EffectResult` via ``report`` once post-period
outcomes exist. These pin the same read-contract the shared
``test_result_contract`` applies to the design family, kept PANGEO-local so it
does not depend on the other design estimators' fixtures. Backward-compatible
field access (``arm_designs``, ``time_labels``, ``effects``, ``power``,
``metadata``) is also pinned so the migration is non-breaking.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth import PANGEO
from mlsynth.config_models import DesignResult, EffectResult
from mlsynth.config_models import MlsynthResult
from mlsynth.utils.pangeo_helpers.simulation import make_seasonal_sales_panel


def _design_only():
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A",), T=40, seed=0)
    return PANGEO({"df": df, "outcome": "sales", "arm": "arm", "unitid": "unit",
                   "time": "time", "max_supergeo_size": 2, "fast": True,
                   "compute_power": False, "display_graphs": False}).fit()


def _realized():
    df = make_seasonal_sales_panel(
        units_per_arm=6, arms=("A",), T=40, n_post=10, seed=0)
    return PANGEO({"df": df, "outcome": "sales", "arm": "arm", "unitid": "unit",
                   "time": "time", "post_col": "post_col",
                   "max_supergeo_size": 2, "fast": True,
                   "compute_power": False, "display_graphs": False}).fit()


class TestDesignFamilyMembership:
    def test_is_design_result_not_effect(self):
        res = _design_only()
        assert isinstance(res, MlsynthResult)
        assert isinstance(res, DesignResult)
        assert not isinstance(res, EffectResult)

    def test_design_only_has_assignment_and_selected_units(self):
        res = _design_only()
        assert res.assignment is not None and len(res.assignment) > 0
        assert res.selected_units is not None and len(res.selected_units) > 0
        # selected_units are exactly the treated units in the assignment
        treated = {u for u, s in res.assignment.items() if s == "treatment"}
        assert set(res.selected_units) == treated
        # a design-only fit has not resolved to a report yet
        assert res.report is None


class TestResolvesToEffectReport:
    def test_report_is_effect_result_with_flat_contract(self):
        res = _realized()
        report = res.report
        assert isinstance(report, EffectResult)
        assert report.effects is not None and report.effects.att is not None
        assert isinstance(report.att, float)
        assert report.att == pytest.approx(report.effects.att)
        cf = np.asarray(report.counterfactual)
        gap = np.asarray(report.gap)
        assert cf.ndim == 1 and cf.shape == gap.shape
        ci = report.att_ci
        assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])

    def test_report_matches_rich_effects_program(self):
        res = _realized()
        assert res.report.att == pytest.approx(res.effects.program.att)


class TestBackwardCompatibleFields:
    def test_pangeo_specific_fields_still_present(self):
        res = _realized()
        assert set(res.arm_designs) == {"A"}
        assert res.max_supergeo_size == 2
        assert np.asarray(res.time_labels).ndim == 1
        assert res.effects is not None
        # rich design-based inference still reachable
        assert "program" in res.effects.randomization

    def test_metadata_and_power_roundtrip_through_model_copy(self):
        # auto-Q uses model_copy to attach the sweep/power; the fields survive.
        df = make_seasonal_sales_panel(units_per_arm=6, arms=("A",), T=40, seed=1)
        res = PANGEO({"df": df, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": None, "fast": True,
                      "display_graphs": False}).fit()
        assert res.metadata["q_auto_selected"] is True
        assert "q_sweep" in res.metadata
        assert res.power is not None

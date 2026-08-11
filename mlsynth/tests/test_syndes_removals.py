"""What SYNDES no longer accepts, and what it now does by default.

The two-way global problem stopped being a mixed-integer program: naming the
treated set removes every binary, so the design is found by searching treated
sets. Three things followed, and this module pins all three.

``backend`` now defaults to ``"exact"`` for ``two_way_global``. ``"mip"`` stays
selectable, and the other two modes are unaffected -- one-way pins the treated
weights and per-unit carries an ``(N, N)`` weight matrix, so neither reduces to
a search over treated sets.

The annealed relaxation is gone. It existed to dodge the MIP's cost on problems
the solver could not finish, and the search finishes those directly, so the mode
and its five helper modules are removed, not left as a slower second path.

The two-way MIP accelerator is gone with it. Its warm start and SDP objective
cut only ever applied to the two-way branch-and-bound, which is no longer the
default route, and the certificate it fed now uses the closed-form Rayleigh
bound.

Donor-side restrictions are refused for two-way. They tie the control weights to
which units are treated, so a design stops being scoreable from its treated set
alone -- the one restriction the search cannot express.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units=9, T=16, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    L = rng.standard_normal((3, n_units))
    Y = F @ L + 0.3 * rng.standard_normal((T, n_units))
    return pd.DataFrame([
        {"unit": f"u{j}", "time": t, "y": float(Y[t, j]),
         "post": int(t >= T - n_post)}
        for j in range(n_units) for t in range(T)
    ])


def _cfg(**over):
    cfg = {"df": _panel(), "outcome": "y", "unitid": "unit", "time": "time",
           "K": 3, "mode": "two_way_global", "post_col": "post",
           "run_inference": False}
    cfg.update(over)
    return cfg


# ----------------------------------------------------------------------
# the new default
# ----------------------------------------------------------------------


class TestDefaultBackend:
    def test_two_way_defaults_to_the_search(self):
        design = SYNDES(_cfg()).fit().design
        assert design.raw_results["solver"] == "exact"

    def test_the_mip_is_still_selectable(self):
        design = SYNDES(_cfg(backend="mip")).fit().design
        assert design.raw_results["solver"] != "exact"

    def test_both_backends_still_agree_when_the_mip_proves_optimality(self):
        exact = SYNDES(_cfg()).fit().design
        mip = SYNDES(_cfg(backend="mip", gap_limit=0.0, time_limit=600.0)).fit().design
        assert (tuple(exact.selected_unit_indices)
                == tuple(mip.selected_unit_indices))

    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_other_modes_still_use_the_mip(self, mode):
        design = SYNDES(_cfg(mode=mode)).fit().design
        assert design.raw_results["solver"] != "exact"

    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_other_modes_reject_the_search(self, mode):
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(mode=mode, backend="exact"))


# ----------------------------------------------------------------------
# the annealed relaxation
# ----------------------------------------------------------------------


class TestAnnealedModeRemoved:
    def test_the_mode_is_rejected(self):
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(mode="two_way_global_annealed"))

    @pytest.mark.parametrize("field", ["relaxed_max_iter", "relaxed_decay"])
    def test_its_config_fields_are_rejected(self, field):
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(**{field: 5}))

    @pytest.mark.parametrize("module", [
        "relaxed_solver", "relaxed_structures", "relaxed_formulation",
        "relaxed_annealing", "relaxed_initialization",
    ])
    def test_its_helper_modules_are_gone(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"mlsynth.utils.syndes_helpers.{module}")

    def test_the_relaxed_result_container_is_not_exported(self):
        import mlsynth
        assert not hasattr(mlsynth, "RelaxedSolverResults")

    def test_the_relaxed_permutation_test_is_gone(self):
        from mlsynth.utils.syndes_helpers import inference
        assert not hasattr(inference, "permutation_test_relaxed_global")

    def test_the_relaxed_plotter_is_gone(self):
        from mlsynth.utils.syndes_helpers import plotter
        assert not hasattr(plotter, "plot_relaxed_design")


# ----------------------------------------------------------------------
# the two-way MIP accelerator
# ----------------------------------------------------------------------


class TestAcceleratorRemoved:
    @pytest.mark.parametrize("field", [
        "accelerate", "accel_min_tuples", "accel_safety_margin",
        "certify_sdp_n_max",
    ])
    def test_its_config_fields_are_rejected(self, field):
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(**{field: 1}))

    def test_the_module_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mlsynth.utils.syndes_helpers.accelerate")

    def test_the_sdp_moment_bound_is_gone(self):
        from mlsynth.utils.syndes_helpers import certificate
        assert not hasattr(certificate, "_sdp_moment_bound_two_way")

    def test_the_shared_warm_cut_helper_survives(self):
        """Other estimators use it, so only the SYNDES-specific wrapper went."""
        from mlsynth.utils import miqp_accel
        assert hasattr(miqp_accel, "solve_warm_cut")

    def test_certify_still_works_on_the_new_default(self):
        res = SYNDES(_cfg(certify=True)).fit()
        assert res.certificate is not None
        assert res.certificate.method == "rayleigh"
        assert res.certificate.lower_bound <= float(res.design.objective_value) + 1e-6


# ----------------------------------------------------------------------
# donor-side restrictions
# ----------------------------------------------------------------------


class TestDonorRestrictionsRefusedForTwoWay:
    def _adjacency(self):
        labels = [f"u{i}" for i in range(9)]
        return pd.DataFrame(np.eye(9), index=labels, columns=labels)

    def test_config_rejects_a_donor_exclusion_matrix(self):
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(donor_exclusion=self._adjacency()))

    def test_config_rejects_a_donor_region_column(self):
        df = _panel()
        df["region"] = ["north" if u in ("u0", "u1") else "south" for u in df["unit"]]
        with pytest.raises(MlsynthConfigError):
            SYNDES(_cfg(df=df, donor_region_col="region"))

    def test_the_constraint_builder_refuses_two_way(self):
        from mlsynth.utils.syndes_helpers.restrictions import donor_constraints
        with pytest.raises(MlsynthConfigError):
            donor_constraints("global_2way", {}, None, [(0, 4)])

    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_other_modes_keep_donor_rules(self, mode):
        df = _panel()
        df["region"] = ["north" if u in ("u0", "u1", "u2") else "south"
                        for u in df["unit"]]
        res = SYNDES(_cfg(df=df, mode=mode, donor_region_col="region")).fit()
        assert int(res.design.assignment.sum()) == 3


# ----------------------------------------------------------------------
# what stayed
# ----------------------------------------------------------------------


class TestSurvivingKnobs:
    @pytest.mark.parametrize("field,value", [("gap_limit", 0.01),
                                             ("time_limit", 30.0)])
    def test_solver_limits_are_still_accepted(self, field, value):
        """They describe branch-and-bound, which the other modes still do."""
        res = SYNDES(_cfg(backend="mip", **{field: value})).fit()
        assert int(res.design.assignment.sum()) == 3

    def test_pool_and_selection_still_run_on_the_default(self):
        res = SYNDES(_cfg(top_K=3, selection="holdout", holdout_frac=0.3)).fit()
        assert int(res.design.assignment.sum()) == 3

    def test_k_none_still_runs_on_the_default(self):
        res = SYNDES(_cfg(K=None)).fit()
        assert 1 <= int(res.design.assignment.sum()) <= 8


class TestDefaultResolvesPerMode:
    """The default is ``"exact"``, which only two-way can honour.

    Resolving it per mode is what keeps the other two working without the user
    naming a backend; asking for it explicitly there is still an error.
    """

    @pytest.mark.parametrize("mode", ["one_way_global", "per_unit"])
    def test_other_modes_silently_keep_the_mip(self, mode):
        assert SYNDES(_cfg(mode=mode)).config.backend == "mip"

    def test_two_way_keeps_the_search(self):
        assert SYNDES(_cfg()).config.backend == "exact"

    def test_an_explicit_mip_is_honoured_on_two_way(self):
        assert SYNDES(_cfg(backend="mip")).config.backend == "mip"

"""What SYNDES does when a lower bound is unavailable.

The certificate's bounds go absent for two different reasons, and both are
exercised here.

The continuous relaxation is a convex solve that can fail to converge, as is the
SDP / moment lift the two-way MIP accelerator still uses as an objective cut:
that one lifts ``x = [w; q; D]`` to a ``(3N+1)`` moment matrix and hands it to
SCS under a fixed iteration cap. When a solve returns without an objective
value, CVXPY leaves ``.value`` at ``None``.

The two-way certificate itself no longer solves anything -- it is the closed-form
Rayleigh bound on the Gram matrix, so it cannot fail to converge. It goes absent
only when that Gram matrix is too ill-conditioned to invert reliably, which
happens as ``lam`` approaches zero on a panel with fewer pre-periods than units.

The bound is a diagnostic and a solver speed-up, never a correctness
requirement: ``solve_synthetic_design`` takes ``objective_lower_bound`` as
optional, so a missing bound costs solve time and nothing else. These tests pin
that a failed bound degrades to "no bound" -- reported through a warning and
through the certificate's own ``None`` contract -- instead of ending the fit.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.config_models import SYNDESConfig
from mlsynth.utils.syndes_helpers import accelerate as accel_mod
from mlsynth.utils.syndes_helpers import certificate as cert_mod
from mlsynth.utils.syndes_helpers.accelerate import two_way_accel_inputs
from mlsynth.utils.syndes_helpers.certificate import (
    SYNDESCertificate,
    _continuous_relaxation_bound,
    _sdp_moment_bound_two_way,
    syndes_certificate,
)
from mlsynth.utils.syndes_helpers.optimization import (
    estimate_lambda,
    solve_synthetic_design,
)

TWO_WAY = "global_2way"


def _Y(N=12, T=8, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    Lam = rng.standard_normal((N, 3))
    return (Lam @ F.T).T + 0.3 * rng.standard_normal((T, N))


def _panel(n_units=12, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, 3))
    L = rng.standard_normal((3, n_units))
    Y = F @ L + 0.3 * rng.standard_normal((T, n_units))
    return pd.DataFrame([
        {"unit": f"u{j}", "time": t, "y": Y[t, j], "post": int(t >= T - n_post)}
        for j in range(n_units) for t in range(T)
    ])


@pytest.fixture
def unsolved(monkeypatch):
    """Make every CVXPY solve a no-op, so objectives keep ``value is None``."""
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)


@pytest.fixture
def bound_fails(monkeypatch):
    """Make only the two-way SDP bound report failure."""
    monkeypatch.setattr(accel_mod, "_sdp_moment_bound_two_way",
                        lambda *a, **k: None)
    monkeypatch.setattr(cert_mod, "_sdp_moment_bound_two_way",
                        lambda *a, **k: None)


class TestBoundHelpersReturnNone:
    """A bound that does not solve is ``None``, not a raised TypeError."""

    def test_sdp_moment_bound(self, unsolved):
        assert _sdp_moment_bound_two_way(_Y(), 3, 1.0) is None

    def test_continuous_relaxation_bound(self, unsolved):
        assert _continuous_relaxation_bound(_Y(), 3, TWO_WAY, 1.0) is None

    @pytest.mark.parametrize("mode", ["global_equal_weights", TWO_WAY, "per_unit"])
    def test_no_mode_raises(self, unsolved, mode):
        assert _continuous_relaxation_bound(_Y(), 3, mode, 1.0) is None

    def test_solved_bound_is_still_a_float(self):
        lb = _sdp_moment_bound_two_way(_Y(), 3, 1.0)
        assert isinstance(lb, float) and np.isfinite(lb)


def _ill_conditioned_Y(N=10, T=4, seed=0):
    """Rank-deficient Gram: ``rank(G) <= T < N``, so ``G + lam I`` is near-singular."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 2)) @ rng.standard_normal((2, N))


class TestCertificateReportsMissingBound:
    """``SYNDESCertificate`` already documents ``lower_bound`` as optional."""

    def test_fields_are_none_and_uncertified(self, unsolved):
        c = syndes_certificate(_Y(), 3, "global_equal_weights", 1.0)
        assert isinstance(c, SYNDESCertificate)
        assert c.lower_bound is None
        assert c.optimality_gap is None
        assert c.certified is False

    def test_note_says_the_bound_failed(self, unsolved):
        c = syndes_certificate(_Y(), 3, "global_equal_weights", 1.0)
        assert c.note, "a missing bound must be explained in the note"
        assert "converge" in c.note.lower() or "fail" in c.note.lower()

    @pytest.mark.parametrize("mode", ["global_equal_weights", "per_unit"])
    def test_solve_backed_modes_degrade(self, unsolved, mode):
        c = syndes_certificate(_Y(), 3, mode, 1.0)
        assert c.lower_bound is None and c.optimality_gap is None

    def test_two_way_needs_no_solver(self, unsolved):
        """The Rayleigh bound is pure linear algebra, so a dead solver is moot."""
        Y = _Y()
        c = syndes_certificate(Y, 3, TWO_WAY, 1.0)
        assert c.method == "rayleigh"
        assert c.lower_bound is not None and c.certified is True

    def test_two_way_degrades_when_the_gram_is_ill_conditioned(self):
        c = syndes_certificate(_ill_conditioned_Y(), 3, TWO_WAY, 1.0, lam=1e-14)
        assert c.method == "rayleigh"
        assert c.lower_bound is None and c.optimality_gap is None
        assert c.certified is False
        assert "ill-conditioned" in c.note

    def test_ill_conditioned_note_says_what_to_do(self):
        c = syndes_certificate(_ill_conditioned_Y(), 3, TWO_WAY, 1.0, lam=1e-14)
        assert "lam" in c.note and "pre-treatment" in c.note

    def test_successful_certificate_unchanged(self):
        Y = _Y()
        inc = float(solve_synthetic_design(Y, K=3, mode=TWO_WAY).objective_value)
        c = syndes_certificate(Y, 3, TWO_WAY, inc)
        assert c.lower_bound is not None
        assert c.optimality_gap is not None
        assert c.certified is True


class TestAcceleratorDropsTheCut:
    """The warm start survives a failed bound; only the cut is dropped."""

    def test_returns_none_for_the_cut(self, bound_fails):
        Y = _Y(N=20)
        with pytest.warns(UserWarning, match="lower bound"):
            warm_D, cut = two_way_accel_inputs(Y, 4, float(estimate_lambda(Y)))
        assert cut is None
        assert warm_D.shape == (20,)
        assert int(warm_D.sum()) == 4

    def test_cut_present_when_the_bound_solves(self):
        Y = _Y(N=14)
        warm_D, cut = two_way_accel_inputs(Y, 4, float(estimate_lambda(Y)))
        assert isinstance(cut, float) and np.isfinite(cut)
        assert int(warm_D.sum()) == 4

    def test_solver_accepts_a_missing_cut(self):
        """``objective_lower_bound=None`` is already a supported call."""
        Y = _Y(N=10)
        d = solve_synthetic_design(Y, K=3, mode=TWO_WAY,
                                   warm_start_D=None, objective_lower_bound=None)
        assert d.objective_value is not None


class TestFitSurvivesBoundFailure:
    """End to end: the design still comes back."""

    def test_fit_returns_a_design(self, bound_fails):
        df = _panel(n_units=12)
        cfg = SYNDESConfig(df=df, outcome="y", unitid="unit", time="time",
                           K=3, post_col="post", mode="two_way_global",
                           accelerate=True, accel_min_tuples=1,
                           display_graph=False)
        with pytest.warns(UserWarning, match="lower bound"):
            res = SYNDES(cfg).fit()
        assert len(np.asarray(res.design.selected_unit_labels)) == 3

    def test_accelerated_design_matches_the_unaccelerated_one(self, bound_fails):
        """Dropping the cut changes runtime, not the optimum."""
        df = _panel(n_units=10)
        common = dict(df=df, outcome="y", unitid="unit", time="time", K=3,
                      post_col="post", mode="two_way_global", display_graph=False)
        with pytest.warns(UserWarning, match="lower bound"):
            fast = SYNDES(SYNDESConfig(**common, accelerate=True,
                                       accel_min_tuples=1)).fit()
        plain = SYNDES(SYNDESConfig(**common, accelerate=False)).fit()
        assert sorted(map(str, np.asarray(fast.design.selected_unit_labels))) == \
            sorted(map(str, np.asarray(plain.design.selected_unit_labels)))

"""Random restarts for the SparseSC outer V-solve.

The outer V-objective is non-convex, and a face of the donor simplex on
which only a few donors are active is a stationary point of it almost for
free: w*(v) has |A| - 1 degrees of freedom there, so the envelope gradient
carries no information about the donors that are out, and at |A| = 1 the
gradient is identically zero. A single cold start from ``default_v20``
therefore settles at whichever such point it is nearest, which on the
augmented Vives California specification (40 predictors, 38 donors) is a
two-donor point with a training loss of 77.42 against the 1.45 the paper's
own stored V attains.

Deterministic heuristic starts do not fix it -- ``1``, ``0.1 * 1`` and the
warm start from the neighbouring lambda all sit in the same region. Random
log-normal restarts around the cold init do: five of them take the k=40
California ATT from -29.04 to -18.64 against the paper's -18.2.

Layers: smoke, invariants of the restart set, edge cases, failure modes,
and a regression panel on which a cold start is measurably worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SparseSC
from mlsynth.config_models import SparseSCConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.sparse_sc_helpers import optimization
from mlsynth.utils.sparse_sc_helpers.objective import outer_loss, selection_mse
from mlsynth.utils.sparse_sc_helpers.optimization import (
    default_lambda_grid,
    sweep_lambda,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _arrays(seed: int = 0, N: int = 8, P: int = 10, T: int = 20,
            T0_total: int = 14, R: int = 3):
    """Factor panel returned as the array contract ``sweep_lambda`` takes.

    Two of the predictors are pre-treatment outcome means, so the predictor
    block genuinely carries information about the outcome path -- without
    that, every V is as good as every other and the test has no power to
    separate a good critical point from a bad one.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, R))
    L = rng.normal(size=(N + 1, R))
    Y = F @ L.T + 0.35 * rng.normal(size=(T, N + 1))
    Y1, Y0 = Y[:, 0], Y[:, 1:]
    X = rng.normal(size=(P, N + 1))
    X[0] = Y[:T0_total].mean(axis=0)
    X[1] = Y[: T0_total // 2].mean(axis=0)
    sd = X.std(axis=1, ddof=1)
    sd[sd == 0] = 1.0
    X = X / sd[:, None]
    return Y1, Y0, X[:, 0], X[:, 1:], T0_total


def _long_panel(seed: int = 0, N_donors: int = 6, T: int = 20, T0: int = 14,
                P: int = 4, true_effect: float = -2.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_units = N_donors + 1
    F = rng.standard_normal((T, P))
    Lambda = rng.standard_normal((n_units, P))
    Y = F @ Lambda.T + 0.3 * rng.standard_normal((T, n_units))
    Y[T0:, 0] += true_effect
    records = []
    for u in range(n_units):
        covs = {f"p{p}": float(Lambda[u, p]) for p in range(P)}
        for t in range(T):
            records.append({"unit": f"unit_{u}", "year": 2000 + t,
                            "y": float(Y[t, u]),
                            "tr": int(u == 0 and t >= T0), **covs})
    return pd.DataFrame(records)


COVS = ["p0", "p1", "p2", "p3"]
GRID = np.concatenate([[0.0], np.logspace(-4, 0, 4)])
ONE = np.array([0.01])      # single point: no cross-lambda coupling


def _sweep(seed_panel: int, restarts: int, restart_seed: int = 0, **kw):
    Y1, Y0, X1, X0, T0_total = _arrays(seed_panel)
    grid = kw.pop("grid", GRID)
    return sweep_lambda(
        X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
        lambda_grid=grid, outer_loss_window="training",
        use_analytical_grad=True, robust=False,
        outer_restarts=restarts, outer_restart_seed=restart_seed, **kw)


# ---------------------------------------------------------------------------
# Layer 0: smoke
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_a_restarted_sweep_returns_the_documented_contract(self):
        optv, lam, grid, outer, val, v_path = _sweep(0, restarts=4)
        P = _arrays(0)[3].shape[0]
        assert optv.shape == (P,)
        assert optv[0] == pytest.approx(1.0)
        assert (optv >= 0).all()
        assert np.isfinite(optv).all()
        assert lam in set(grid.tolist())
        assert v_path.shape == (grid.size, P)
        assert np.isfinite(outer).all() and np.isfinite(val).all()

    def test_a_restarted_fit_produces_a_finite_att(self):
        cfg = SparseSCConfig(df=_long_panel(), outcome="y", treat="tr",
                             unitid="unit", time="year", covariates=COVS,
                             outer_restarts=3, display_graphs=False)
        res = SparseSC(cfg).fit()
        assert np.isfinite(res.effects.att)
        assert res.method_details.parameters_used["outer_restarts"] == 3


# ---------------------------------------------------------------------------
# Layer 1: invariants of the restart set
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_restarts_never_raise_the_outer_objective_at_one_lambda(self):
        """Per solve, the cold start stays in the candidate set, so the
        best-of cannot be worse than the cold start alone.

        Asserted on a one-point grid on purpose. Across a grid the guarantee
        does NOT hold: a better solution at lambda_i becomes the champion and
        the warm start for lambda_{i+1}, so a strictly better solve upstream
        can move a later solve into a different basin. That coupling is a
        property of the sweep, not a defect, and the sweep still minimises
        validation MSE over the whole grid.
        """
        for panel in range(4):
            _, _, _, cold, _, _ = _sweep(panel, restarts=0, grid=ONE)
            _, _, _, rs, _, _ = _sweep(panel, restarts=6, restart_seed=7,
                                       grid=ONE)
            assert rs[0] <= cold[0] + 1e-9

    def test_the_same_seed_gives_the_same_answer(self):
        a = _sweep(1, restarts=5, restart_seed=99)
        b = _sweep(1, restarts=5, restart_seed=99)
        np.testing.assert_allclose(a[0], b[0], rtol=0, atol=0)
        assert a[1] == b[1]

    def test_the_seed_actually_reaches_the_restart_draws(self):
        """Guards against a seed that is accepted and then ignored: two seeds
        must be able to produce different candidate sets, so on some panel
        they produce different paths."""
        differed = False
        for panel_seed in range(6):
            a = _sweep(panel_seed, restarts=6, restart_seed=1)
            b = _sweep(panel_seed, restarts=6, restart_seed=2)
            if not np.allclose(a[5], b[5], rtol=1e-10, atol=1e-12):
                differed = True
                break
        assert differed, "outer_restart_seed had no effect on any panel"

    def test_more_restarts_never_hurt_at_one_lambda(self):
        """Draws come off one stream per grid point, so eight restarts see
        the two a two-restart run saw. Same one-point-grid caveat as above."""
        for panel in range(4):
            _, _, _, o2, _, _ = _sweep(panel, restarts=2, restart_seed=5,
                                       grid=ONE)
            _, _, _, o8, _, _ = _sweep(panel, restarts=8, restart_seed=5,
                                       grid=ONE)
            assert o8[0] <= o2[0] + 1e-9

    def test_restarts_do_not_disturb_the_anchor_or_the_bounds(self):
        optv, _, _, _, _, v_path = _sweep(2, restarts=5, restart_seed=3)
        assert optv[0] == pytest.approx(1.0)
        assert np.allclose(v_path[:, 0], 1.0)
        assert (v_path >= 0).all()


# ---------------------------------------------------------------------------
# Layer 2: edge cases
# ---------------------------------------------------------------------------

class TestEdges:
    def test_zero_restarts_reproduces_the_unrestarted_sweep_exactly(self):
        Y1, Y0, X1, X0, T0_total = _arrays(5)
        base = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False)
        with_zero = _sweep(5, restarts=0)
        np.testing.assert_allclose(base[0], with_zero[0], rtol=0, atol=0)
        np.testing.assert_allclose(base[3], with_zero[3], rtol=0, atol=0)

    def test_a_single_free_coordinate_is_handled(self):
        """P = 2 leaves one free V-weight once the anchor is pinned."""
        Y1, Y0, X1, X0, T0_total = _arrays(6, P=2)
        optv, lam, _, outer, _, _ = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False,
            outer_restarts=4, outer_restart_seed=0)
        assert optv.shape == (2,)
        assert np.isfinite(outer).all()

    def test_a_single_donor_does_not_poison_the_restart_draws(self):
        """With one donor ``ddof=1`` leaves the predictor sd undefined, so
        ``default_v20`` used to return all-NaN and every restart draw, being
        ``v20 * exp(...)``, inherited it."""
        Y1, Y0, X1, X0, T0_total = _arrays(7, N=1)
        optv, _, _, outer, _, _ = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False,
            outer_restarts=4, outer_restart_seed=0)
        assert np.isfinite(optv).all()
        assert np.isfinite(outer).all()

    def test_collinear_predictors_do_not_break_the_restarts(self):
        Y1, Y0, X1, X0, T0_total = _arrays(8)
        X0 = X0.copy(); X1 = X1.copy()
        X0[3] = X0[2]; X1[3] = X1[2]          # exact duplicate predictor row
        optv, _, _, outer, _, _ = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False,
            outer_restarts=5, outer_restart_seed=0)
        assert np.isfinite(optv).all()
        assert np.isfinite(outer).all()

    def test_restarts_compose_with_the_robust_continuation_pass(self):
        """The backward pass draws its own restarts and accepts a candidate
        only when it lowers that grid point's outer objective, so on a single
        lambda the two together are still no worse than the pass alone."""
        for panel in (9, 10):
            Y1, Y0, X1, X0, T0_total = _arrays(panel)
            kw = dict(X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total,
                      T0_train=10, lambda_grid=ONE,
                      outer_loss_window="training",
                      use_analytical_grad=True, robust=True)
            plain = sweep_lambda(**kw)
            both = sweep_lambda(**kw, outer_restarts=5, outer_restart_seed=0)
            assert both[3][0] <= plain[3][0] + 1e-9

    def test_restarts_and_the_robust_pass_together_keep_the_contract(self):
        Y1, Y0, X1, X0, T0_total = _arrays(9)
        optv, lam, grid, outer, val, v_path = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=True,
            outer_restarts=4, outer_restart_seed=0)
        assert np.isfinite(outer).all() and np.isfinite(val).all()
        assert np.allclose(v_path[:, 0], 1.0)
        assert (v_path >= 0).all()
        assert lam in set(grid.tolist())


# ---------------------------------------------------------------------------
# Layer 2b: the reported objective must belong to the reported V
# ---------------------------------------------------------------------------

class TestReportedObjectiveMatchesReportedV:
    """``scipy.optimize.minimize`` can return ``fun`` and ``x`` from different
    iterates when L-BFGS-B ends in ABNORMAL_TERMINATION_IN_LNSRCH (status 2),
    which on a kinked objective like this one is common, not exotic: on the
    Vives k=40 specification it happens at 10 of 26 grid points under finite
    differences and 19 of 26 under the analytic gradient, with gaps up to 1818.

    The sweep used ``res.fun`` to rank candidate starts, to fill
    ``outer_curve`` and to decide whether the backward pass accepts a
    candidate. A number that belongs to a different point cannot do any of
    those jobs.
    """

    @pytest.mark.parametrize("panel", [0, 3, 11])
    @pytest.mark.parametrize("restarts", [0, 4])
    def test_outer_curve_is_the_objective_at_the_reported_v(self, panel, restarts):
        Y1, Y0, X1, X0, T0_total = _arrays(panel)
        T0_train = 10
        _, _, grid, outer, _, v_path = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=T0_train,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False,
            outer_restarts=restarts, outer_restart_seed=0)
        Z1, Z0 = Y1[:T0_train], Y0[:T0_train]
        for i, lam in enumerate(grid):
            recomputed = outer_loss(v_path[i, 1:], X1, X0, Z1, Z0, float(lam))
            assert outer[i] == pytest.approx(recomputed, rel=1e-6, abs=1e-8), (
                f"grid point {i} (lambda={lam:g}) reports {outer[i]:.6f} but the "
                f"V it returned scores {recomputed:.6f}")

    @pytest.mark.parametrize("panel", [0, 3, 11])
    def test_each_grid_point_keeps_the_best_candidate_it_tried(self, panel,
                                                               monkeypatch):
        """Ranking starts on a stale ``fun`` can keep the worse one.

        The contract is a statement about one solve: among the starts a grid
        point was given, the one kept is the one with the lowest objective
        recomputed at its own returned point. Spying on ``_minimize_outer``
        records every candidate; ``lam`` identifies the grid point, since the
        grid is strictly increasing and ``robust=False`` visits each once.

        The power comes from ``res.fun`` disagreeing with that objective often
        enough to change the argmin. Over the 29 solves this sweep performs it
        disagrees at 7 of them on panel 3 (worst gap 144,672 against a true
        10.06) and at 17 on panel 11, twice reporting BELOW the truth, which is
        the direction that wins a comparison it should lose. Panel 0 sees no
        disagreement and is kept as the case where the two agree.
        """
        Y1, Y0, X1, X0, T0_total = _arrays(panel)
        T0_train = 10
        Z1, Z0 = Y1[:T0_train], Y0[:T0_train]

        tried: dict = {}
        real = optimization._minimize_outer

        def spy(**kw):
            res = real(**kw)
            x = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
            tried.setdefault(float(kw["lam"]), []).append(x)
            return res

        monkeypatch.setattr(optimization, "_minimize_outer", spy)
        _, _, grid, _, _, v_path = sweep_lambda(
            X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=T0_train,
            lambda_grid=GRID, outer_loss_window="training",
            use_analytical_grad=True, robust=False,
            outer_restarts=4, outer_restart_seed=0)

        for i, lam in enumerate(grid):
            candidates = tried[float(lam)]
            assert len(candidates) > 1, (
                f"grid point {i} tried {len(candidates)} start(s); the ranking "
                f"rule is untested with fewer than two")
            f_kept = outer_loss(v_path[i, 1:], X1, X0, Z1, Z0, float(lam))
            for j, x in enumerate(candidates):
                f_cand = outer_loss(x, X1, X0, Z1, Z0, float(lam))
                if not np.isfinite(f_cand):
                    continue
                assert f_kept <= f_cand + 1e-8, (
                    f"grid point {i} (lambda={lam:g}) kept a V scoring "
                    f"{f_kept:.6f} over candidate {j}, which scores {f_cand:.6f}")

    @pytest.mark.parametrize("panel", [3, 11])
    def test_restarts_cannot_lose_ground_at_any_single_lambda(self, panel):
        """Adding restarts is monotone only where the candidate set nests.

        An earlier version asserted that one sweep with restarts returns a V no
        worse than a sweep without them at EVERY point of a shared grid, and
        that is false by construction. ``sweep_lambda`` seeds grid point ``i``
        from the running validation-MSE champion, which has already diverged
        between the two runs, so from ``i = 1`` on they are handed different
        candidate sets, not nested ones. Measured on panel 3, the base start
        lists are identical at ``i = 0`` and differ at all four later points.
        A restart run can then land in a worse basin with nothing wrong. That
        assertion passed on one machine, where both chains re-converged to the
        same V at the last two grid points, and failed on CI for Python 3.10,
        3.11 and 3.12 alike.

        Running each lambda as its own one-point sweep removes the coupling
        without giving up any lambda. Every lambda is then the first grid
        point: the champion is ``None`` and there is no previous solution, so
        ``_build_starts`` returns ``[v20_cold]`` in both runs and the restart
        run's candidates are that list plus the draws. ``_solve`` keeps the
        argmin of the objective recomputed at each returned point, so the
        comparison is guaranteed at every lambda.

        Keeping the whole grid is what gives this power. Restricting it to the
        first point of the shared grid does not: that point is ``lambda = 0``,
        the one lambda with no L1 term and so no kink, where ``res.fun`` agrees
        with the recomputed objective to 6e-12. Against
        ``the-reported-objective-comes-from-a-different-iterate`` a
        first-point-only assertion passed 12 of 12 times over four panels and
        three seeds; per-lambda catches it on panel 3 at ``lambda = 1`` and on
        panel 11 at ``lambda = 1e-4``.
        """
        Y1, Y0, X1, X0, T0_total = _arrays(panel)
        T0_train = 10
        Z1, Z0 = Y1[:T0_train], Y0[:T0_train]

        def solve(lam, restarts):
            _, _, _, _, _, v = sweep_lambda(
                X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total,
                T0_train=T0_train, lambda_grid=np.array([lam], dtype=float),
                outer_loss_window="training", use_analytical_grad=True,
                robust=False, outer_restarts=restarts, outer_restart_seed=1)
            return outer_loss(v[0, 1:], X1, X0, Z1, Z0, float(lam))

        for lam in GRID:
            f_cold = solve(float(lam), 0)
            f_restarted = solve(float(lam), 8)
            assert f_restarted <= f_cold + 1e-8, (
                f"lambda={lam:g}: the restart run's candidates are the cold "
                f"run's plus the draws, so its V cannot score worse; it scores "
                f"{f_restarted:.6f} against {f_cold:.6f}")


# ---------------------------------------------------------------------------
# Layer 3: the regression -- panels where a cold start is measurably worse
# ---------------------------------------------------------------------------

class TestRegression:
    """Without restarts the sweep settles wherever ``default_v20`` leads.

    The cuts pinned here are measured against the objective recomputed at the
    returned V. An earlier version of this file compared ``res.fun`` instead
    and reported cuts of 86% to 98% on four panels; two of those were the
    stale-``res.fun`` defect inflating the cold-start number, not the restarts
    doing work. Re-measured, seed 11 cuts 47.3% (not 97.9%), seed 34 cuts
    94.9%, seed 53 cuts 85.9%, and seed 19 -- previously reported at 93.5% --
    shows no improvement at all and has been dropped.

    The cold solution is not always the low-|A| one: on these panels it
    carries more active donors than the restarted solution, not fewer, so what
    the restarts buy is a different basin, not a particular donor count.
    """

    PANELS = (11, 34, 53)

    @pytest.mark.parametrize("panel", PANELS)
    def test_restarts_reach_a_materially_better_critical_point(self, panel):
        _, _, _, cold, _, _ = _sweep(panel, restarts=0, grid=ONE)
        _, _, _, rs, _, _ = _sweep(panel, restarts=8, restart_seed=0, grid=ONE)
        assert rs[0] < cold[0]
        # measured cuts are 47%, 86% and 95%; assert 25%, about half the
        # smallest, so the test survives a different BLAS kernel without
        # going vacuous
        assert rs[0] <= 0.75 * cold[0]

    def test_the_cold_start_is_what_is_being_escaped(self):
        """The restarted solve must beat the cold start's own critical point,
        not merely differ from it -- otherwise the draws are just noise."""
        worse = 0
        for panel in self.PANELS:
            _, _, _, cold, _, _ = _sweep(panel, restarts=0, grid=ONE)
            _, _, _, rs, _, _ = _sweep(panel, restarts=8, restart_seed=0,
                                       grid=ONE)
            worse += int(cold[0] > rs[0] * 1.5)
        assert worse == len(self.PANELS)


# ---------------------------------------------------------------------------
# Layer 4: failure modes -- reported, not swallowed
# ---------------------------------------------------------------------------

class TestFailures:
    """The estimator translates pydantic's ValidationError into
    MlsynthConfigError; constructing the config class directly does not, so
    each of these is asserted on the path a caller actually takes."""

    @staticmethod
    def _cfg(**kw):
        base = {"df": _long_panel(), "outcome": "y", "treat": "tr",
                "unitid": "unit", "time": "year", "covariates": COVS,
                "display_graphs": False}
        base.update(kw)
        return base

    def test_a_negative_restart_count_is_rejected(self):
        with pytest.raises(MlsynthConfigError):
            SparseSC(self._cfg(outer_restarts=-1))

    def test_a_non_integer_restart_count_is_rejected(self):
        with pytest.raises(MlsynthConfigError):
            SparseSC(self._cfg(outer_restarts=2.5))

    def test_an_unknown_restart_field_is_rejected(self):
        with pytest.raises(MlsynthConfigError):
            SparseSC(self._cfg(outer_restart=3))

    def test_a_negative_restart_seed_is_rejected(self):
        with pytest.raises(MlsynthConfigError):
            SparseSC(self._cfg(outer_restart_seed=-5))

    def test_a_negative_restart_count_is_rejected_at_the_sweep(self):
        Y1, Y0, X1, X0, T0_total = _arrays(0)
        with pytest.raises(ValueError):
            sweep_lambda(
                X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0_total, T0_train=10,
                lambda_grid=GRID, outer_loss_window="training",
                outer_restarts=-3)

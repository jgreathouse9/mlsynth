"""Every cold entry into the simplex solver seeds itself, SDID's two included.

The seed used to be a FISTA pass computed in ``ridge_augment.simplex_qp``, so
being accelerated was a property of that one entry point. Of the thirteen call
sites in the library, twelve never supplied a warm start -- MEDSC, SCD, COMPSC,
StackedSC, mlSC, the proximal over-identified weights, the ``minnorm`` fallbacks,
and SDID's two simplex programs -- so they called ``solve_simplex_qp`` cold.
Starting cold means starting from the uniform point, and the active set then
sheds one donor per pivot until only the support is left: work proportional to
the pool, not to the answer.

That is what made SDID slow. On a 101x120 panel its two programs took 117 inner
least-squares solves between them, while ``VanillaSC`` on the same panel took 31
through the accelerated door.

The seed is now :func:`~mlsynth.utils.solvers.accelerate.priced_seed`, which
prices the columns at the uniform point and keeps the ``SEED_KEEP`` whose reduced
gradient is most negative, capped at ``m + 1``. Its gate is that budget and not
the pool's width: a pool no wider than the cap has nothing to price away and
takes the untouched cold path, and everything wider is pruned. So the fixtures
below are sized against ``SEED_KEEP`` and not against a donor floor.

The gate now lives in the solver. These tests pin the consequences: SDID's
weights are unchanged to the last bit, its pivot count collapses, and the gate
still respects the two cases that must skip it.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.utils.solvers import active_set
from mlsynth.utils.solvers.accelerate import SEED_KEEP
from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.sdid_helpers.weights import _solve_intercept_simplex


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(n_donors=90, n_periods=40, n_pre=30, effect=-2.0, seed=3):
    """A factor-structure panel with one treated unit."""
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    factors = np.cumsum(rng.standard_normal((n_periods, 3)) * 0.3, axis=0)
    loadings = rng.uniform(0.2, 1.2, (n_units, 3))
    outcome = (loadings @ factors.T
               + rng.standard_normal((n_units, n_periods)) * 0.4 + 10.0)
    treat = np.zeros((n_units, n_periods), dtype=int)
    treat[0, n_pre:] = 1
    outcome[0, n_pre:] += effect
    return pd.DataFrame(
        [{"id": f"u{i:03d}", "time": t + 1, "y": outcome[i, t], "d": treat[i, t]}
         for i in range(n_units) for t in range(n_periods)]
    )


def _fit(df, **overrides):
    config = {"df": df, "outcome": "y", "treat": "d", "unitid": "id",
              "time": "time", "display_graphs": False, "vce": "noinference"}
    config.update(overrides)
    return SDID(config).fit()


def _cold(monkeypatch):
    """Decline every seed, restoring the pre-change cold path."""
    monkeypatch.setattr(active_set, "priced_seed", lambda B, A, **kw: None)


class _Spy:
    """Counts the seeds the solver computes for itself, with their shapes."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = active_set.priced_seed

        def spy(design, target, **kwargs):
            self.calls.append(np.asarray(design).shape)
            return real(design, target, **kwargs)

        monkeypatch.setattr(active_set, "priced_seed", spy)

    def __len__(self):
        return len(self.calls)


# --------------------------------------------------------------------------- #
# 1. smoke
# --------------------------------------------------------------------------- #
class TestSmoke:
    def test_wide_panel_fits(self):
        result = _fit(_panel())
        assert np.isfinite(result.effects.att)
        assert len(result.weights.donor_weights) == 90

    def test_the_panel_is_wide_enough_to_engage_the_gate(self):
        """The premise of the file: 90 donors is wider than the seed's budget, so
        there is something to price away."""
        assert 90 > SEED_KEEP


# --------------------------------------------------------------------------- #
# 2. the binding invariant -- the seam changes cost, not weights
# --------------------------------------------------------------------------- #
class TestAnswerUnchanged:
    def test_att_and_weights_identical_to_the_cold_path(self, monkeypatch):
        seeded = _fit(_panel())
        _cold(monkeypatch)
        cold = _fit(_panel())

        assert seeded.effects.att == pytest.approx(cold.effects.att, rel=0, abs=1e-12)
        for name in ("donor_weights", "time_weights", "unit_weights"):
            got, want = getattr(seeded.weights, name), getattr(cold.weights, name)
            if want is None:
                assert got is None
                continue
            assert set(got) == set(want)
            np.testing.assert_allclose(
                [got[k] for k in want], [want[k] for k in want], rtol=0, atol=1e-12)

    def test_counterfactual_series_identical(self, monkeypatch):
        seeded = _fit(_panel())
        _cold(monkeypatch)
        cold = _fit(_panel())
        np.testing.assert_allclose(
            np.asarray(seeded.time_series.counterfactual_outcome, float),
            np.asarray(cold.time_series.counterfactual_outcome, float),
            rtol=0, atol=1e-10)

    def test_intercept_program_agrees_with_its_own_cold_path(self, monkeypatch):
        """Straight at SDID's solver call, ridge on and off."""
        rng = np.random.default_rng(11)
        design = rng.standard_normal((60, 120))
        target = design @ rng.dirichlet(np.ones(120)) + rng.standard_normal(60) * 0.1
        for ridge in (0.0, 0.75):
            intercept_a, weights_a = _solve_intercept_simplex(design, target, ridge)
            _cold(monkeypatch)
            intercept_b, weights_b = _solve_intercept_simplex(design, target, ridge)
            monkeypatch.undo()
            np.testing.assert_allclose(weights_a, weights_b, rtol=0, atol=1e-12)
            assert intercept_a == pytest.approx(intercept_b, rel=0, abs=1e-10)


# --------------------------------------------------------------------------- #
# 3. the work the seam removes
# --------------------------------------------------------------------------- #
class TestWorkReduction:
    def test_sdid_takes_more_pivots_and_less_time(self, monkeypatch):
        """The direction is inverted here, and the reason is the ridge.

        SDID stacks ``sqrt(ridge) I`` beneath its design, and a ridge exists to
        spread weight, so its optima are dense: on this panel the support is 65
        of 90 donors on one program and 14 of 30 on the other. A seed keeping
        ``SEED_KEEP`` columns therefore undershoots badly, and the release rule
        lets exactly one donor back per pivot, so the pivot count rises. What
        does not rise is the cost, because an add-back pivot solves on a small
        free set:

            cold        41 pivots   16.2 ms
            k=4         73 pivots   15.0 ms
            k=8         67 pivots   14.2 ms
            k=16        59 pivots   14.1 ms
            k=64        37 pivots   12.1 ms

        So the pivot count is a proxy that stops tracking the cost on a dense
        optimum, and this test asserts the measured direction of both instead of
        the one that reads better. ``test_cold_work_scales_with_the_pool_and_
        seeded_work_does_not`` keeps the pivot-reduction claim where it holds --
        an unridged pool, where the optimum is sparse.
        """
        pivots = {"seeded": [], "cold": []}
        real = solve_simplex_qp

        def recording(bucket):
            def wrapper(B, A, **kwargs):
                w, info = real(B, A, **{**kwargs, "return_info": True})
                bucket.append(info["pivots"])
                return (w, info) if kwargs.get("return_info") else w
            return wrapper

        import mlsynth.utils.sdid_helpers.weights as sdid_weights

        monkeypatch.setattr(sdid_weights, "solve_simplex_qp",
                            recording(pivots["seeded"]))
        seeded = _fit(_panel())

        _cold(monkeypatch)
        monkeypatch.setattr(sdid_weights, "solve_simplex_qp",
                            recording(pivots["cold"]))
        cold = _fit(_panel())

        assert len(pivots["seeded"]) == len(pivots["cold"]) > 0
        assert sum(pivots["seeded"]) > sum(pivots["cold"]), (
            "on a ridged program the seed undershoots a dense optimum, so it "
            "should take more pivots; fewer would mean the ridge stopped "
            "spreading the weight"
        )
        assert seeded.effects.att == pytest.approx(cold.effects.att, rel=0, abs=1e-12)

    def test_the_ridged_programs_optima_really_are_dense(self):
        """Power for the test above: it explains itself by the support size, so
        the support size is asserted. A sparse optimum here would make the pivot
        direction above an unexplained accident."""
        import mlsynth.utils.sdid_helpers.weights as sdid_weights

        seen = []
        real = solve_simplex_qp

        def rec(B, A, **kwargs):
            w = real(B, A, **kwargs)
            arr = w[0] if isinstance(w, tuple) else w
            seen.append((B.shape[1], int((np.asarray(arr) > 1e-9).sum())))
            return w

        sdid_weights.solve_simplex_qp = rec
        try:
            _fit(_panel())
        finally:
            sdid_weights.solve_simplex_qp = real
        assert seen, "no simplex program was solved"
        # Both programs are dense as a fraction of their pool -- 65 of 90 and 14
        # of 30 on this panel -- which is what a ridge does. Only the first
        # exceeds a budget of 16, and it is the one that drives the pivot count:
        # undershooting a 65-donor support by 49 costs one add-back apiece.
        for J, support in seen:
            assert support >= 0.4 * J, (
                f"support {support} of {J} is sparse, so this program is not "
                f"the dense case the pivot direction above rests on"
            )
        assert max(s for _, s in seen) > SEED_KEEP, (
            f"no program's support exceeds the seed budget {SEED_KEEP}, so "
            f"nothing here undershoots and the pivot count should not rise"
        )

    @pytest.mark.parametrize("J", [100, 160, 240])
    def test_cold_work_scales_with_the_pool_and_seeded_work_does_not(self, J):
        """The invariant nobody had written down. Cold pivots track ``J``;
        seeded pivots track the support, which is far smaller."""
        rng = np.random.default_rng(J)
        factors = rng.standard_normal((2 * J, 2))
        loadings = np.zeros((J, 2))
        loadings[: J // 2, 0] = 1.0
        loadings[J // 2:, 1] = 1.0
        B = factors @ loadings.T + rng.standard_normal((2 * J, J))
        A = factors @ np.array([1.0, 1.0]) + rng.standard_normal(2 * J)

        w_cold, cold = solve_simplex_qp(B, A, return_info=True, accelerate=False)
        w_seed, seeded = solve_simplex_qp(B, A, return_info=True)
        support = int((w_cold > 1e-9).sum())

        assert cold["pivots"] >= J // 2, "cold path should walk most of the pool"
        assert seeded["pivots"] <= support
        np.testing.assert_allclose(w_seed, w_cold, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 4. the gate
# --------------------------------------------------------------------------- #
class TestGate:
    def test_engages_on_a_wide_pool(self, monkeypatch):
        spy = _Spy(monkeypatch)
        _fit(_panel())
        assert len(spy) > 0

    def test_declined_on_a_pool_no_wider_than_the_budget(self, monkeypatch):
        """``priced_seed`` is called and returns ``None``: with ``SEED_KEEP``
        donors there is nothing to price away, so the solve starts uniform. The
        fixture is sized off the constant, so retuning it cannot make this pass
        for the wrong reason."""
        spy = _Spy(monkeypatch)
        _fit(_panel(n_donors=SEED_KEEP, n_periods=30, n_pre=20))
        assert len(spy) > 0, "the seed is consulted even when it declines"
        from mlsynth.utils.solvers.accelerate import priced_seed as real_seed
        rng = np.random.default_rng(0)
        B = rng.normal(size=(20, SEED_KEEP))
        assert real_seed(B, rng.normal(size=20)) is None

    def test_skipped_when_the_caller_already_warm_starts(self, monkeypatch):
        """The placebo fallback chains the previous draw's solution; a chained
        seed must not be thrown away to recompute one."""
        spy = _Spy(monkeypatch)
        rng = np.random.default_rng(4)
        design = rng.standard_normal((50, 120))
        target = rng.standard_normal(50)
        _solve_intercept_simplex(design, target, 0.0,
                                 warm_start=np.full(120, 1.0 / 120))
        assert len(spy) == 0

    def test_gate_reads_the_design_the_solver_sees(self, monkeypatch):
        """SDID stacks the ridge block *beneath* the design, so it adds rows and
        not donors: the seed is computed on the augmented design."""
        spy = _Spy(monkeypatch)
        rng = np.random.default_rng(6)
        design = rng.standard_normal((40, 100))
        target = rng.standard_normal(40)
        _solve_intercept_simplex(design, target, 2.0)
        assert spy.calls == [(140, 100)]

    def test_accelerate_false_forces_the_cold_path(self, monkeypatch):
        spy = _Spy(monkeypatch)
        rng = np.random.default_rng(7)
        B = rng.standard_normal((150, 120))
        A = rng.standard_normal(150)
        solve_simplex_qp(B, A, accelerate=False)
        assert len(spy) == 0


# --------------------------------------------------------------------------- #
# 5. edges
# --------------------------------------------------------------------------- #
class TestEdges:
    def test_single_donor(self, monkeypatch):
        spy = _Spy(monkeypatch)
        rng = np.random.default_rng(8)
        _, weights = _solve_intercept_simplex(
            rng.standard_normal((12, 1)), rng.standard_normal(12), 0.0)
        np.testing.assert_allclose(weights, np.ones(1))
        assert len(spy) == 0

    def test_collinear_donors_still_agree_with_the_cold_path(self, monkeypatch):
        rng = np.random.default_rng(12)
        core = rng.standard_normal((40, 45))
        design = np.hstack([core, core])              # every donor duplicated
        target = rng.standard_normal(40)
        _, seeded = _solve_intercept_simplex(design, target, 0.0)
        _cold(monkeypatch)
        _, cold = _solve_intercept_simplex(design, target, 0.0)
        assert (np.sum((design @ seeded - target) ** 2)
                == pytest.approx(np.sum((design @ cold - target) ** 2),
                                 rel=0, abs=1e-9))

    def test_more_donors_than_rows(self):
        rng = np.random.default_rng(13)
        _, weights = _solve_intercept_simplex(
            rng.standard_normal((15, 120)), rng.standard_normal(15), 0.0)
        assert weights.min() >= -1e-12
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_all_zero_donors(self):
        """A degenerate design makes the FISTA Lipschitz bound collapse; the
        seed falls back to uniform and the solve still returns a simplex point."""
        weights = solve_simplex_qp(np.zeros((20, 100)), np.arange(20.0))
        assert weights.min() >= 0.0
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_placebo_inference_unchanged(self, monkeypatch):
        """The batched placebo path routes around the solver; it must still land
        on the same numbers."""
        df = _panel(n_donors=90, n_periods=24, n_pre=18)
        seeded = _fit(df, vce="placebo")
        _cold(monkeypatch)
        cold = _fit(df, vce="placebo")
        assert seeded.effects.att == pytest.approx(cold.effects.att, rel=0, abs=1e-12)
        assert (seeded.inference.standard_error
                == pytest.approx(cold.inference.standard_error, rel=0, abs=1e-10))


# --------------------------------------------------------------------------- #
# 6. failure -- bad input is reported before any seeding work
# --------------------------------------------------------------------------- #
class TestFailures:
    def test_empty_design_raises(self):
        with pytest.raises(ValueError, match="at least one donor"):
            solve_simplex_qp(np.zeros((10, 0)), np.zeros(10))

    def test_length_mismatch_raises(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="must equal"):
            solve_simplex_qp(rng.standard_normal((10, 100)), np.zeros(9))

    def test_mismatch_raises_before_the_seed_is_computed(self, monkeypatch):
        spy = _Spy(monkeypatch)
        rng = np.random.default_rng(14)
        with pytest.raises(ValueError):
            solve_simplex_qp(rng.standard_normal((10, 100)), np.zeros(9))
        assert len(spy) == 0

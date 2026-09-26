"""The synthetic stand-ins for the Hsiao and Zhou study's two smoking panels.

The study was measured on the panels shipped with the paper, which are not in
this repository, so its two empirical arms were unrunnable for anyone without
them. The generators under test replace those panels with ones drawn to behave
the same way, so the arms run anywhere and the paper's data becomes an option
instead of a prerequisite. The turnout arm needs no stand-in: that panel is
already here as `basedata/xu_edr_turnout.parquet`.

What the tests hold them to is behaviour, not values. The consumption panel has
to fall over time, carry a factor structure, and include one near-constant
regressor with a large coefficient -- that last is not decoration. It is the
feature of `lnincome` that made Bai's PCA2 iteration stall (#647), and a
stand-in without it would let the study's own regression test pass for the
wrong reason, so the test asserts a margin the coefficient's size has to earn. The expenditure panel has to be wide relative to its pre-period,
which is the rank-deficient regime its arm exists to exercise.

Nothing here pins a number that came from any real panel.

Levels: smoke, unit invariants, edge, failure.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.studies.hsiao_zhou_counterfactuals import standin


class TestConsumptionPanel:
    """The Proposition 99 analogue: 39 units, 31 periods, treated at 19."""

    @pytest.fixture(scope="class")
    def panel(self):
        return standin.consumption_panel(seed=0)

    def test_it_has_the_shape_the_arm_expects(self, panel):
        assert set(panel.columns) == {"state", "year", "cigsale", "lnincome",
                                      "EduAttain", "Poverty"}
        assert panel.state.nunique() == 39
        assert panel.year.nunique() == 31
        assert len(panel) == 39 * 31
        assert panel.state.min() == 1, "unit 1 is the treated one"

    def test_the_outcome_falls_over_the_window(self, panel):
        early = panel[panel.year < panel.year.min() + 5].cigsale.mean()
        late = panel[panel.year > panel.year.max() - 5].cigsale.mean()
        assert late < early, "consumption has to decline, as the real one does"
        assert 0.25 < late / early < 0.85

    def test_one_regressor_is_near_constant_and_the_others_are_not(self, panel):
        """The lnincome analogue. Its flatness is what stalls PCA2."""
        cv = {c: panel[c].std() / abs(panel[c].mean())
              for c in ("lnincome", "EduAttain", "Poverty")}
        assert cv["lnincome"] < 0.08, "lnincome must be nearly flat"
        assert cv["EduAttain"] > 2 * cv["lnincome"]
        assert cv["Poverty"] > 2 * cv["lnincome"]

    def test_it_carries_a_factor_structure(self, panel):
        wide = panel.pivot(index="year", columns="state", values="cigsale")
        Y = wide.to_numpy()
        Yc = Y - Y.mean(axis=0)
        s = np.linalg.svd(Yc, compute_uv=False)
        share = float((s[:2] ** 2).sum() / (s ** 2).sum())
        assert share > 0.45, "two factors must carry a real share of the panel"
        assert share < 0.995, "and must not be the whole of it"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_bai_step_stalls_on_it_by_a_margin(self, seed):
        """The point of the near-constant regressor, asserted end to end.

        PCA2 from a zero start is the defect #647 fixed. On a panel without a
        weakly identified regressor it converges fine and the study's guard
        would pass vacuously, so the stand-in is a fair substitute only if PCA2
        still loses on it.

        The assertion is a margin, not a strict inequality. Two schemes that
        converge to the same point differ in the last bits, and which way that
        falls depends on the BLAS: the first version of this test compared two
        equal objectives and failed in CI on one platform and passed on
        another. The margin is what separates a real gap from that. Measured
        over six seeds the excess is 31 percent at worst, matching the 18
        percent the real panel shows, so 5 percent has room and no ambiguity.
        """
        from benchmarks.studies.hsiao_zhou_counterfactuals.empirics import (
            _beta_bai_pca2_from_zero, bai_objective, beta_bai, build)
        panel = standin.consumption_panel(seed=seed)
        Y, X, _, T0 = build(panel, "state", "year", "cigsale",
                            ["lnincome", "EduAttain", "Poverty"], 1,
                            int(panel.year.min()) + 19)
        Yco, Xco = Y[:, 1:], X[:, 1:, :]
        good, _, _ = beta_bai(Yco, Xco, r=2)
        bad = _beta_bai_pca2_from_zero(Yco, Xco, r=2)
        reached = bai_objective(Yco, Xco, good, 2)
        stalled = bai_objective(Yco, Xco, bad, 2)
        assert (stalled - reached) / reached > 0.05, (
            f"PCA2 must stall by a clear margin on the stand-in; got "
            f"{(stalled - reached) / reached:.2e}. A near-zero gap means the "
            f"near-constant regressor has stopped carrying the defect.")

    def test_it_is_reproducible_from_its_seed(self):
        import pandas as pd
        pd.testing.assert_frame_equal(standin.consumption_panel(seed=3),
                                      standin.consumption_panel(seed=3))
        assert not standin.consumption_panel(seed=3).cigsale.equals(
            standin.consumption_panel(seed=4).cigsale)


class TestExpenditurePanel:
    """The healthcare-expenditure analogue: 39 units, 21 periods, treated at 9."""

    @pytest.fixture(scope="class")
    def panel(self):
        return standin.expenditure_panel(seed=0)

    def test_it_has_the_shape_the_arm_expects(self, panel):
        assert set(panel.columns) == {"state", "year", "lnhexpense", "lnincome"}
        assert panel.state.nunique() == 39
        assert panel.year.nunique() == 21
        assert panel.state.min() == 1

    def test_the_donor_pool_is_wider_than_the_pre_period(self, panel):
        """38 controls against 9 pre-periods: the rank-deficient regime."""
        pre = panel[panel.year < panel.year.min() + 9].year.nunique()
        assert pre == 9
        assert panel.state.nunique() - 1 > 4 * pre

    def test_the_outcome_rises_and_is_on_a_log_scale(self, panel):
        early = panel[panel.year == panel.year.min()].lnhexpense.mean()
        late = panel[panel.year == panel.year.max()].lnhexpense.mean()
        assert late > early
        assert 7.0 < early < 11.0 and 7.0 < late < 11.0

    def test_it_is_reproducible_from_its_seed(self):
        import pandas as pd
        pd.testing.assert_frame_equal(standin.expenditure_panel(seed=1),
                                      standin.expenditure_panel(seed=1))


class TestItIsNotRealData:
    def test_no_state_names_or_real_years_are_used(self):
        p = standin.consumption_panel(seed=0)
        assert p.state.dtype.kind in "iu", "units are integers, not state names"
        assert p.year.min() == 0, "periods are integers from zero, not calendar years"

    def test_the_generators_refuse_a_degenerate_size(self):
        with pytest.raises(ValueError):
            standin.consumption_panel(n_units=1)
        with pytest.raises(ValueError):
            standin.expenditure_panel(n_periods=2)

"""MAREX covariates are aggregated over the fitting window only.

Abadie and Zhao split the pre-period into fitting periods E, on which the
design is chosen, and blank periods B, whose outcomes stay out of the choice
so that their gaps form the placebo distribution for inference. The outcomes
were matched on E, but each covariate was averaged over every pre-period, E
and B together, so a time-varying covariate carried blank-window data into
the design (issue #759). Vives-i-Bastida (2022) averages the covariates over
the fitting years only.

The invariant pinned here: the design is a function of the data in the
fitting window. Changing a covariate inside the blank window or the
post-period leaves the selected units, the weights and the objective as they
were; changing it inside the fitting window is seen.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import MAREX
from mlsynth.utils.marex_helpers.setup import prepare_marex_panel

warnings.filterwarnings("ignore")

N, T, T0 = 10, 20, 16


def _arrays(seed):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, 1)) + rng.normal(scale=0.5, size=(N, T)).cumsum(1)
    Z = rng.normal(size=(N, T))
    return Y, Z, rng


def _frame(Y, Z):
    rows = [dict(unit=f"u{j}", t=t, y=Y[j, t], z=Z[j, t], c=float(j % 3))
            for j in range(Y.shape[0]) for t in range(Y.shape[1])]
    return pd.DataFrame(rows)


def _fit(df, blanks, covariates=("z",), **kw):
    cfg = dict(df=df, outcome="y", unitid="unit", time="t", T0=T0,
               blank_periods=blanks, m_eq=2, covariates=list(covariates),
               backend="exact", inference=False)
    cfg.update(kw)
    return MAREX(cfg).fit()


def _summary(res):
    w = res.design_weights.donor_weights
    return (sorted(map(str, res.selected_units)),
            {k: round(v, 10) for k, v in w.items() if abs(v) > 1e-12})


def _panel(df, blanks):
    return prepare_marex_panel(df=df, outcome="y", unitid="unit", time="t",
                               cluster=None, T0=T0, inference=True,
                               blank_periods=blanks, T_post=T - T0,
                               covariates=["z"])


# ----------------------------------------------------------------------
# Unit: the aggregation window
# ----------------------------------------------------------------------

class TestAggregationWindow:

    def test_covariate_is_the_mean_over_the_fitting_window(self):
        Y, Z, _ = _arrays(0)
        p = _panel(_frame(Y, Z), blanks=4)
        np.testing.assert_allclose(p.covariates[:, 0], Z[:, :T0 - 4].mean(1))

    def test_no_blank_window_keeps_the_whole_pre_period(self):
        """With no blank periods the fitting window is the pre-period, so the
        aggregation, and every design built on it, is unchanged."""
        Y, Z, _ = _arrays(1)
        p = _panel(_frame(Y, Z), blanks=0)
        np.testing.assert_allclose(p.covariates[:, 0], Z[:, :T0].mean(1))

    def test_one_fitting_period(self):
        Y, Z, _ = _arrays(2)
        p = _panel(_frame(Y, Z), blanks=T0 - 1)
        np.testing.assert_allclose(p.covariates[:, 0], Z[:, 0])

    def test_time_invariant_covariate_is_its_value(self):
        Y, Z, _ = _arrays(3)
        p = prepare_marex_panel(df=_frame(Y, Z), outcome="y", unitid="unit",
                                time="t", cluster=None, T0=T0, inference=True,
                                blank_periods=4, T_post=T - T0,
                                covariates=["c"])
        np.testing.assert_allclose(p.covariates[:, 0], np.arange(N) % 3)

    def test_too_many_blank_periods_still_raises(self):
        Y, Z, _ = _arrays(4)
        with pytest.raises(ValueError, match="blank_periods"):
            _panel(_frame(Y, Z), blanks=T0)


# ----------------------------------------------------------------------
# End to end: the design ignores data outside the fitting window
# ----------------------------------------------------------------------

class TestDesignWindow:

    def test_blank_window_covariates_do_not_move_the_design(self):
        """Seed 4 is the reproducer from the issue: before the fix, changing
        the covariate inside the blank window changed the treated pair from
        (u0, u6) to (u2, u4)."""
        Y, Z, rng = _arrays(4)
        Z2 = Z.copy()
        Z2[:, T0 - 4:T0] = rng.normal(scale=20, size=(N, 4))
        a = _fit(_frame(Y, Z), blanks=4)
        b = _fit(_frame(Y, Z2), blanks=4)
        assert _summary(a) == _summary(b)

    def test_post_period_covariates_do_not_move_the_design(self):
        Y, Z, rng = _arrays(5)
        Z2 = Z.copy()
        Z2[:, T0:] = rng.normal(scale=20, size=(N, T - T0))
        assert (_summary(_fit(_frame(Y, Z), blanks=4))
                == _summary(_fit(_frame(Y, Z2), blanks=4)))

    def test_fitting_window_covariates_are_seen(self):
        """The check above has power: the same perturbation inside the
        fitting window changes the predictors the design is chosen on."""
        Y, Z, rng = _arrays(6)
        Z2 = Z.copy()
        Z2[:, :4] = rng.normal(scale=20, size=(N, 4))
        a = _panel(_frame(Y, Z), blanks=4).covariates
        b = _panel(_frame(Y, Z2), blanks=4).covariates
        assert not np.allclose(a, b)

    @settings(max_examples=12, deadline=None)
    @given(seed=st.integers(0, 10_000), blanks=st.integers(1, 6),
           scale=st.floats(0.1, 50.0))
    def test_design_is_a_function_of_the_fitting_window(self, seed, blanks,
                                                         scale):
        Y, Z, rng = _arrays(seed)
        Z2 = Z.copy()
        Z2[:, T0 - blanks:] = scale * rng.normal(size=(N, T - T0 + blanks))
        a = _fit(_frame(Y, Z), blanks=blanks)
        b = _fit(_frame(Y, Z2), blanks=blanks)
        assert _summary(a) == _summary(b)

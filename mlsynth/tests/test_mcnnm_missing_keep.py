"""MC-NNM on a panel with missing outcome cells (#1146).

MC-NNM (Athey, Bayati, Doudchenko, Imbens and Khosravi, "Matrix Completion
Methods for Causal Panel Data Models", JASA 116 (2021)) fits its low-rank
matrix and fixed effects on the observed set ``O`` only (eq. 4.3) and imputes
every cell outside it. The treated cells are outside ``O`` by construction; a
missing outcome cell belongs outside ``O`` for the same reason. So the setup
asks ``dataprep`` to keep the cell (``missing_outcomes="keep"``) and builds the
mask from both the treatment and the ``observed`` mask, and the pipeline hands
the engine a matrix whose entries outside ``O`` are never read as values.

A missing outcome of a treated unit after its adoption is refused, with the
unit and period named: the effect there is the treated outcome minus the
imputed one, and neither the paper (which observes every cell's realized
outcome, section 2) nor ``MCPanel`` (a completion routine with no effect
estimand) defines it without the treated outcome.

What is pinned here:

* a complete panel gives the inputs and the fit it gave before the opt-in;
* a missing cell is ``NaN`` in ``Y`` and ``0`` in the mask; the engine gets a
  finite matrix and a mask that excludes the cell, and a finite value stored
  behind the mask changes nothing;
* on a low-rank panel with a planted effect the ATT stays near the truth as up
  to 30% of the cells are removed at random;
* a missing treated post-treatment cell is refused; any other missing cell is
  accepted;
* an infinite outcome, an absent row and a missing treatment are still refused.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import MCNNM
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.mcnnm_helpers import pipeline as PIPE
from mlsynth.utils.mcnnm_helpers import setup as SETUP
from mlsynth.utils.mcnnm_helpers.pipeline import run_mcnnm
from mlsynth.utils.mcnnm_helpers.setup import prepare_mcnnm_inputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")
FAST = dict(n_lambda=6, n_folds=3)


def _panel(n_units=8, n_periods=12, adopt=None, seed=0, shuffle=True):
    adopt = {0: 6} if adopt is None else adopt
    rng = np.random.default_rng(seed)
    rows = [dict(unit=u, time=2000 + t,
                 y=rng.normal() + u + 0.2 * t + 1.5 * int(u in adopt and t >= adopt[u]),
                 d=int(u in adopt and t >= adopt[u]))
            for u in range(n_units) for t in range(n_periods)]
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed) if shuffle else df


def _plant(df, cells, value=np.nan):
    out = df.copy()
    for u, t in cells:
        out.loc[(out.unit == u) & (out.time == t), "y"] = value
    return out


def _fit(df, **kw):
    cfg = dict(FAST) if "lam" not in kw else {}
    cfg.update(kw)
    return MCNNM(dict(df=df, display_graphs=False, **COLS, **cfg)).fit()


def _low_rank(seed, n_units=24, n_periods=24, t0=16, n_treated=2, tau=2.0):
    """Rank-2 factor panel plus small noise; the first ``n_treated`` units
    gain ``tau`` from period ``t0`` on."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_units, 2))
    V = rng.normal(size=(n_periods, 2))
    Y0 = 5.0 + U @ V.T + 0.1 * rng.normal(size=(n_units, n_periods))
    D = np.zeros((n_units, n_periods), dtype=int)
    D[:n_treated, t0:] = 1
    Y = Y0 + tau * D
    return pd.DataFrame(dict(unit=np.repeat(np.arange(n_units), n_periods),
                             time=np.tile(np.arange(n_periods), n_units),
                             y=Y.ravel(), d=D.ravel()))


def _drop_at_random(df, frac, seed):
    """NaN a ``frac`` share of the cells that are not treated post-treatment."""
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(df["d"].to_numpy() == 0)
    k = int(round(frac * cand.size))
    out = df.copy()
    out.iloc[rng.choice(cand, k, replace=False), out.columns.get_loc("y")] = np.nan
    return out


def _same_fit(a, b):
    assert a.att == b.att
    assert a.best_lambda == b.best_lambda
    np.testing.assert_array_equal(a.counterfactual_matrix, b.counterfactual_matrix)
    np.testing.assert_array_equal(a.L, b.L)
    assert list(a.event_study) == list(b.event_study)
    np.testing.assert_array_equal(list(a.event_study.values()),
                                  list(b.event_study.values()))
    assert a.weights.donor_weights == b.weights.donor_weights


# --------------------------------------------------------------------------- #
# routing: the opt-in and the mask
# --------------------------------------------------------------------------- #
class TestOptIn:
    def test_setup_asks_dataprep_to_keep_missing_outcomes(self, monkeypatch):
        seen = []
        real = SETUP.dataprep

        def spy(*a, **k):
            seen.append(k.get("missing_outcomes"))
            return real(*a, **k)

        monkeypatch.setattr(SETUP, "dataprep", spy)
        prepare_mcnnm_inputs(_panel(), **COLS)
        assert seen == ["keep"]

    def test_a_missing_cell_is_nan_in_y_and_outside_the_mask(self):
        cells = [(3, 2004), (0, 2002), (5, 2010)]
        inputs = prepare_mcnnm_inputs(_plant(_panel(), cells), **COLS)
        t = list(inputs.time_labels)
        for u, yr in cells:
            i = inputs.unit_names.index(u)
            assert np.isnan(inputs.Y[i, t.index(yr)])
            assert inputs.mask[i, t.index(yr)] == 0
        np.testing.assert_array_equal(
            inputs.mask, ((inputs.D == 0) & ~np.isnan(inputs.Y)).astype(float))

    def test_a_complete_panel_masks_only_the_treated_cells(self):
        inputs = prepare_mcnnm_inputs(_panel(adopt={0: 5, 1: 8}), **COLS)
        np.testing.assert_array_equal(inputs.mask, 1.0 - inputs.D)


class TestTheEngineSeesTheMask:
    @pytest.mark.parametrize("lam", [None, 0.5])
    def test_the_engine_gets_a_finite_matrix_and_the_mask(self, monkeypatch, lam):
        calls = []
        for name in ("mcnnm_fit", "mcnnm_cv"):
            real = getattr(PIPE, name)

            def spy(Y, mask, *a, _real=real, **k):
                calls.append((np.array(Y), np.array(mask)))
                return _real(Y, mask, *a, **k)

            monkeypatch.setattr(PIPE, name, spy)
        df = _plant(_panel(), [(3, 2004), (6, 2011)])
        inputs = prepare_mcnnm_inputs(df, **COLS)
        run_mcnnm(inputs, lam=lam, n_lam=4, n_folds=2, inference=True)
        assert len(calls) >= 2                  # the main fit and the jackknife refits
        main_Y, main_mask = calls[0]
        assert np.isfinite(main_Y).all()
        np.testing.assert_array_equal(main_mask, inputs.mask)
        missing = np.isnan(inputs.Y)
        assert (main_mask[missing] == 0).all()
        for Y, _ in calls[1:]:
            assert np.isfinite(Y).all()


class TestFitWithMissingCells:
    def test_smoke_missing_control_cells(self):
        res = _fit(_plant(_panel(), [(3, 2004), (5, 2010), (7, 2001)]))
        assert np.isfinite(res.att)
        assert res.metadata["missing_outcome_cells"] == 3
        assert res.metadata["n_missing"] == 6 + 3       # treated cells plus missing ones

    def test_a_complete_panel_reports_no_missing_cell(self):
        assert _fit(_panel(), lam=0.5).metadata["missing_outcome_cells"] == 0

    def test_a_missing_untreated_cell_is_imputed(self):
        res = _fit(_plant(_panel(), [(3, 2004)]), lam=0.5)
        i = res.inputs.unit_names.index(3)
        j = list(res.inputs.time_labels).index(2004)
        assert np.isfinite(res.counterfactual_matrix[i, j])
        assert np.isnan(res.effects_matrix[i, j])
        assert np.isfinite(res.counterfactual_matrix).all()

    def test_staggered_adoption_with_missing_cells(self):
        df = _plant(_panel(adopt={0: 5, 1: 8}), [(1, 2006), (4, 2009), (6, 2000)])
        res = _fit(df, lam=0.5)
        assert np.isfinite(res.att)
        assert all(np.isfinite(v) for v in res.cohort_att.values())

    def test_jackknife_runs_with_missing_cells(self):
        res = _fit(_plant(_panel(), [(3, 2004), (5, 2010)]), lam=0.5, inference=True)
        assert np.isfinite(res.inference_jackknife.se)
        assert res.inference_jackknife.n_jackknife == 7

    def test_the_event_study_skips_a_missing_treated_pre_cell(self):
        # unit 0 adopts in 2006 and is missing in 2002, relative time -4
        res = _fit(_plant(_panel(), [(0, 2002)]), lam=0.5)
        assert np.isnan(res.event_study[-4])
        assert all(np.isfinite(v) for e, v in res.event_study.items() if e != -4)
        # the treated path has a missing period, so its pre-period fit is undefined
        assert np.isnan(res.fit_diagnostics.rmse_pre)
        assert np.isfinite(res.att)
        # with a second treated unit, relative time -4 is that unit's gap alone
        res = _fit(_plant(_panel(adopt={0: 6, 1: 6}), [(0, 2002)]), lam=0.5)
        i = res.inputs.unit_names.index(1)
        gap = res.inputs.Y[i, 2] - res.counterfactual_matrix[i, 2]
        assert res.event_study[-4] == pytest.approx(gap, abs=1e-12)


class TestMaskReachesTheMethod:
    """A finite value stored in a masked cell never enters the fit."""

    @pytest.mark.parametrize("fill", [0.0, 1e6, -37.5])
    @pytest.mark.parametrize("lam", [None, 0.5])
    def test_a_finite_value_behind_the_mask_changes_nothing(self, fill, lam):
        df = _plant(_panel(adopt={0: 6, 1: 9}), [(3, 2004), (0, 2001), (6, 2011)])
        inputs = prepare_mcnnm_inputs(df, **COLS)
        Y = inputs.Y.copy()
        Y[np.isnan(Y)] = fill
        kw = dict(lam=lam, n_lam=4, n_folds=2, inference=True)
        a = run_mcnnm(inputs, **kw)
        b = run_mcnnm(dataclasses.replace(inputs, Y=Y), **kw)
        _same_fit(a, b)
        assert a.inference_jackknife.se == b.inference_jackknife.se


# --------------------------------------------------------------------------- #
# the treated post-treatment rule
# --------------------------------------------------------------------------- #
class TestTreatedPostCell:
    def test_refused_naming_the_unit_and_period(self):
        with pytest.raises(MlsynthDataError, match=r"treated unit '0'.*time 2008"):
            prepare_mcnnm_inputs(_plant(_panel(), [(0, 2008)]), **COLS)

    def test_the_estimator_reports_it(self):
        df = _plant(_panel(), [(0, 2011), (0, 2007)])
        with pytest.raises(MlsynthDataError, match=r"2 treated post-treatment cell.*time 2007"):
            _fit(df)

    def test_the_adoption_period_itself_counts(self):
        with pytest.raises(MlsynthDataError, match=r"treated post-treatment.*time 2006"):
            prepare_mcnnm_inputs(_plant(_panel(), [(0, 2006)]), **COLS)

    def test_staggered_the_unit_s_own_adoption_decides(self):
        df = _panel(adopt={0: 5, 1: 8})
        prepare_mcnnm_inputs(_plant(df, [(1, 2006)]), **COLS)
        with pytest.raises(MlsynthDataError, match=r"treated unit '1'.*time 2008"):
            prepare_mcnnm_inputs(_plant(df, [(1, 2008)]), **COLS)


class TestStillRefused:
    def test_an_infinite_outcome(self):
        with pytest.raises(MlsynthDataError, match="infinite"):
            prepare_mcnnm_inputs(_plant(_panel(), [(3, 2004)], np.inf), **COLS)

    def test_an_absent_row(self):
        df = _panel(shuffle=False)
        df = df[~((df.unit == 4) & (df.time == 2003))]
        with pytest.raises(MlsynthDataError, match="balanced"):
            _fit(df)

    def test_a_missing_treatment(self):
        df = _panel(shuffle=False).astype({"d": float})
        df.loc[(df.unit == 3) & (df.time == 2004), "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="Missing treatment values"):
            _fit(df)


# --------------------------------------------------------------------------- #
# planted missingness on a low-rank panel with a known effect
# --------------------------------------------------------------------------- #
# Mean absolute ATT error over ten seeds (tau = 2, threshold pinned at 0.5) for
# each share of non-treated-post cells removed at random. The fit gives about
# 0.04-0.05 at every share; the bounds are about 1.5 times that.
SIM_BOUNDS = {0.0: 0.06, 0.10: 0.07, 0.20: 0.08, 0.30: 0.08}


@pytest.mark.parametrize("frac", sorted(SIM_BOUNDS))
def test_planted_missingness_stays_near_the_truth(frac):
    errs = []
    for seed in range(10):
        df = _drop_at_random(_low_rank(seed), frac, 100 + seed)
        res = _fit(df, lam=0.5)
        assert res.metadata["missing_outcome_cells"] == int(df.y.isna().sum())
        errs.append(abs(res.att - 2.0))
    assert np.mean(errs) <= SIM_BOUNDS[frac], (frac, np.round(errs, 3))


# --------------------------------------------------------------------------- #
# properties over random panels with random missing cells
# --------------------------------------------------------------------------- #
@st.composite
def missing_panels(draw):
    n_units = draw(st.integers(4, 8))
    n_periods = draw(st.integers(4, 9))
    n_treated = draw(st.integers(1, n_units - 3))
    adopt = {u: draw(st.integers(2, n_periods - 1)) for u in range(n_treated)}
    df = _panel(n_units=n_units, n_periods=n_periods, adopt=adopt,
                seed=draw(st.integers(0, 10_000)))
    k = draw(st.integers(0, 4))
    cells = draw(st.lists(st.integers(0, len(df) - 1), min_size=k, max_size=k, unique=True))
    out = df.copy()
    out.iloc[cells, out.columns.get_loc("y")] = np.nan
    return out


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=missing_panels(), fill=st.floats(-1e3, 1e3))
def test_property_missing_cells_leave_the_fit_or_the_treated_post_cell_is_refused(df, fill):
    if (df.y.isna() & (df.d == 1)).any():
        with pytest.raises(MlsynthDataError, match="treated post-treatment"):
            prepare_mcnnm_inputs(df, **COLS)
        return
    inputs = prepare_mcnnm_inputs(df, **COLS)
    missing = np.isnan(inputs.Y)
    np.testing.assert_array_equal(inputs.mask, ((inputs.D == 0) & ~missing).astype(float))
    a = run_mcnnm(inputs, lam=0.5)
    Y = inputs.Y.copy()
    Y[missing] = fill
    b = run_mcnnm(dataclasses.replace(inputs, Y=Y), lam=0.5)
    _same_fit(a, b)
    assert np.isfinite(a.att)
    assert np.isfinite(a.counterfactual_matrix).all()
    assert a.metadata["missing_outcome_cells"] == int(missing.sum())

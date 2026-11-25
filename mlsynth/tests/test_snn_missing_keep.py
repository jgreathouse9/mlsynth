"""SNN on a panel with missing outcome cells (#1146).

SNN is a matrix-completion method (Agarwal, Dahleh, Shah and Shen, "Causal
Matrix Completion", 2021): it imputes every missing entry of a partially
observed matrix from a fully observed anchor block, and in the panel setting the
treated post-treatment cells are among the missing entries. A missing outcome
cell is one more missing entry, so the setup asks ``dataprep`` to keep it
(``missing_outcomes="keep"``) and hands the ``observed`` mask to the pipeline,
which marks the cell missing before the completion runs.

The one case the method does not define is a missing outcome of a treated unit
after its adoption: the effect at that cell is its observed outcome minus the
imputed ``Y(0)``, and the observed outcome is not there. Neither the paper nor
the reference implementation (``deshen24/syntheticNN``, a generic completion
routine with no effect estimand) says what to do with it, so the setup refuses
it and names the unit and period.

What is pinned here:

* a complete panel gives the inputs and the fit it gave before the opt-in;
* a missing cell stays NaN in ``Y``, is ``False`` in ``observed``, and never
  enters the fit as a value: the pipeline reads the mask, not the NaN;
* on a low-rank panel with a planted effect the ATT stays near the truth as
  cells are removed at random;
* a missing treated post-treatment cell is refused; a missing cell anywhere
  else is accepted;
* an infinite outcome, an absent row and a missing treatment are still refused.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth import SNN
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.snn_helpers import setup as SETUP
from mlsynth.utils.snn_helpers.completion import snn_donor_weights
from mlsynth.utils.snn_helpers.pipeline import run_snn
from mlsynth.utils.snn_helpers.setup import prepare_snn_inputs

COLS = dict(outcome="y", treat="d", unitid="unit", time="time")


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
    return SNN(dict(df=df, display_graphs=False, **COLS, **kw)).fit()


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
        prepare_snn_inputs(_panel(), **COLS)
        assert seen == ["keep"]

    def test_a_complete_panel_has_an_all_true_mask(self):
        inputs = prepare_snn_inputs(_panel(adopt={0: 5, 1: 8}), **COLS)
        assert inputs.observed.dtype == bool
        assert inputs.observed.shape == inputs.Y.shape
        assert inputs.observed.all()

    def test_a_missing_cell_is_nan_in_y_and_false_in_the_mask(self):
        df = _plant(_panel(), [(3, 2004), (0, 2002), (5, 2010)])
        inputs = prepare_snn_inputs(df, **COLS)
        t = list(inputs.time_labels)
        for u, yr in [(3, 2004), (0, 2002), (5, 2010)]:
            i = inputs.unit_names.index(u)
            assert np.isnan(inputs.Y[i, t.index(yr)])
            assert not inputs.observed[i, t.index(yr)]
        assert inputs.observed.sum() == inputs.Y.size - 3
        np.testing.assert_array_equal(inputs.observed, ~np.isnan(inputs.Y))


# --------------------------------------------------------------------------- #
# the fit with missing cells
# --------------------------------------------------------------------------- #
class TestFitWithMissingCells:
    def test_smoke_missing_control_cells(self):
        df = _plant(_panel(), [(3, 2004), (5, 2010), (7, 2001)])
        res = _fit(df)
        assert np.isfinite(res.att)
        assert res.metadata["missing_outcome_cells"] == 3

    def test_a_complete_panel_reports_no_missing_cell(self):
        assert _fit(_panel()).metadata["missing_outcome_cells"] == 0

    def test_a_missing_treated_pre_period_cell_is_accepted(self):
        res = _fit(_plant(_panel(), [(0, 2002)]))
        assert np.isfinite(res.att)
        assert all(np.isfinite(v) for v in res.att_by_period.values())

    def test_a_missing_untreated_cell_gets_an_imputation(self):
        df = _plant(_panel(), [(3, 2004)])
        res = _fit(df)
        i = res.inputs.unit_names.index(3)
        j = list(res.inputs.time_labels).index(2004)
        assert res.feasible[i, j]
        assert np.isfinite(res.counterfactual_matrix[i, j])
        # Only treated cells carry an effect.
        assert np.isnan(res.effects_matrix[i, j])

    def test_observed_untreated_cells_keep_their_values(self):
        df = _plant(_panel(), [(3, 2004)])
        res = _fit(df)
        keep = res.inputs.observed & (res.inputs.D == 0)
        np.testing.assert_array_equal(res.counterfactual_matrix[keep], res.inputs.Y[keep])

    def test_staggered_adoption_with_missing_cells(self):
        df = _plant(_panel(adopt={0: 5, 1: 8}), [(1, 2006), (4, 2009), (6, 2000)])
        res = _fit(df)
        assert np.isfinite(res.att)
        # unit 1 is missing in 2006, before its own adoption in 2008
        assert res.metadata["missing_outcome_cells"] == 3

    def test_jackknife_runs_with_missing_cells(self):
        res = _fit(_plant(_panel(), [(3, 2004), (5, 2010)]), inference=True)
        assert np.isfinite(res.inference_jackknife.se)
        assert res.inference_jackknife.n_jackknife == 7

    def test_weights_reproduce_the_first_post_period_counterfactual(self):
        # The donor weights are the PCR coefficients of the first post-treatment
        # cell. A missing pre-period cell of the treated unit must not move them
        # onto that pre-period cell's anchor block.
        df = _plant(_panel(), [(0, 2002), (4, 2006)])
        res = _fit(df, clip=False)
        inp = res.inputs
        i = int(inp.treated_idx[0])
        w = res.weights.donor_weights
        row = {str(n): k for k, n in enumerate(inp.unit_names)}
        donors = [row[n] for n in w]
        assert len(donors) == 6
        assert row["4"] not in donors                    # missing at the first post period
        pred = sum(w[n] * inp.Y[d, inp.T0] for n, d in zip(w, donors))
        assert pred == pytest.approx(res.counterfactual_matrix[i, inp.T0], abs=1e-9)


class TestDonorWeightsColumn:
    def _matrix(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 8))
        X[0, 5:] = np.nan       # treated row
        X[0, 1] = np.nan        # a missing pre-period cell of the treated row
        X[3, 5] = np.nan        # a donor missing at the first post period
        return X, (~np.isnan(X)).astype(float)

    def test_the_default_column_is_the_row_s_first_missing_cell(self):
        X, mask = self._matrix()
        a = snn_donor_weights(X, mask, 0)
        b = snn_donor_weights(X, mask, 0, column=1)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_the_column_chooses_the_anchor_rows(self):
        X, mask = self._matrix()
        AR, _ = snn_donor_weights(X, mask, 0, column=5)
        assert 3 not in AR.tolist() and AR.size == 4
        assert 3 in snn_donor_weights(X, mask, 0)[0].tolist()

    def test_a_row_with_no_missing_cell_has_no_weights(self):
        X, mask = self._matrix()
        AR, w = snn_donor_weights(X, mask, 1)
        assert AR.size == 0 and w.size == 0


class TestMaskReachesTheMethod:
    """A masked cell's stored value never enters the fit."""

    @pytest.mark.parametrize("fill", [0.0, 1e6, -37.5])
    def test_a_finite_value_behind_the_mask_changes_nothing(self, fill):
        df = _plant(_panel(adopt={0: 6, 1: 9}), [(3, 2004), (0, 2001), (6, 2011)])
        inputs = prepare_snn_inputs(df, **COLS)
        Y = inputs.Y.copy()
        Y[~inputs.observed] = fill
        filled = dataclasses.replace(inputs, Y=Y)
        a = run_snn(inputs, inference=True)
        b = run_snn(filled, inference=True)
        assert a.att == b.att
        np.testing.assert_array_equal(a.feasible, b.feasible)
        masked = ~inputs.observed
        np.testing.assert_array_equal(a.counterfactual_matrix[~masked],
                                      b.counterfactual_matrix[~masked])
        np.testing.assert_array_equal(a.counterfactual_matrix[masked],
                                      b.counterfactual_matrix[masked])
        assert a.weights.donor_weights == b.weights.donor_weights
        assert a.inference_jackknife.se == b.inference_jackknife.se


# --------------------------------------------------------------------------- #
# the treated post-treatment rule
# --------------------------------------------------------------------------- #
class TestTreatedPostCell:
    def test_refused_naming_the_unit_and_period(self):
        df = _plant(_panel(), [(0, 2008)])
        with pytest.raises(MlsynthDataError, match=r"treated unit '0'.*time 2008"):
            prepare_snn_inputs(df, **COLS)

    def test_the_estimator_reports_it(self):
        df = _plant(_panel(), [(0, 2011), (0, 2007)])
        with pytest.raises(MlsynthDataError, match=r"2 treated post-treatment cell.*time 2007"):
            _fit(df)

    def test_the_adoption_period_itself_counts(self):
        with pytest.raises(MlsynthDataError, match="time 2006"):
            prepare_snn_inputs(_plant(_panel(), [(0, 2006)]), **COLS)

    def test_staggered_the_unit_s_own_adoption_decides(self):
        df = _panel(adopt={0: 5, 1: 8})
        # 2006 is after unit 0's adoption but before unit 1's: accepted for unit 1
        prepare_snn_inputs(_plant(df, [(1, 2006)]), **COLS)
        with pytest.raises(MlsynthDataError, match=r"treated unit '1'.*time 2008"):
            prepare_snn_inputs(_plant(df, [(1, 2008)]), **COLS)


# --------------------------------------------------------------------------- #
# what is still refused under the opt-in
# --------------------------------------------------------------------------- #
class TestStillRefused:
    def test_an_infinite_outcome(self):
        with pytest.raises(MlsynthDataError, match="infinite"):
            prepare_snn_inputs(_plant(_panel(), [(3, 2004)], np.inf), **COLS)

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
# Mean absolute ATT error over ten seeds (tau = 2) for each share of
# non-treated-post cells removed at random. SNN's greedy anchor search keeps a
# smaller fully observed block as cells go missing, so the error grows with the
# share; the bound at each share is about 1.5 times what the fit gives.
SIM_BOUNDS = {0.0: 0.06, 0.05: 0.1, 0.10: 0.25, 0.20: 0.8}


@pytest.mark.parametrize("frac", sorted(SIM_BOUNDS))
def test_planted_missingness_stays_near_the_truth(frac):
    errs = []
    for seed in range(10):
        df = _drop_at_random(_low_rank(seed), frac, 100 + seed)
        res = _fit(df)
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
def test_property_missing_cells_are_masked_or_the_treated_post_cell_refused(df, fill):
    hit = (df.y.isna() & (df.d == 1)).any()
    if hit:
        with pytest.raises(MlsynthDataError, match="treated post-treatment"):
            prepare_snn_inputs(df, **COLS)
        return
    inputs = prepare_snn_inputs(df, **COLS)
    np.testing.assert_array_equal(inputs.observed, ~np.isnan(inputs.Y))
    assert int((~inputs.observed).sum()) == int(df.y.isna().sum())
    a = run_snn(inputs)
    Y = inputs.Y.copy()
    Y[~inputs.observed] = fill
    b = run_snn(dataclasses.replace(inputs, Y=Y))
    assert (a.att == b.att) or (np.isnan(a.att) and np.isnan(b.att))
    np.testing.assert_array_equal(a.counterfactual_matrix, b.counterfactual_matrix)
    # observed untreated cells keep their values; effects live on treated cells only
    keep = inputs.observed & (inputs.D == 0)
    np.testing.assert_array_equal(a.counterfactual_matrix[keep], inputs.Y[keep])
    assert np.isnan(a.effects_matrix[inputs.D == 0]).all()

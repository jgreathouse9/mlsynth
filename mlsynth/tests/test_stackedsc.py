"""STACKEDSC: stacked-in-event-time bias-corrected SCM (Wiltshire 2023).

One synthetic control per treated unit against the never-treated donors, with
each unit and its donors indexed to 100 at that unit's own final pre-treatment
period, then averaged on a common event clock under user-supplied weights.

Three things about this estimator drive the tests below.

The normalisation is not cosmetic. Rescaling unit ``j`` and every donor to
``b_j`` and imposing ``sum(w) = 1`` is algebraically the same as fitting on
levels under the constraint ``sum_j v_j b_j = b_1`` -- the synthetic control
must reproduce the treated unit's base-period level exactly. That is why the
paper's printed gap at ``e = -1`` is exactly zero, and it is the sharpest
available check that the base year is right.

The base year belongs to the cohort, not the unit. Units adopting in the same
period share ``T0i``, so the donor block takes as many distinct values as there
are adoption times -- six on the Walmart panel, not 566. A common base year
across cohorts would be a different (and wrong) estimator, and the event-time
path is what detects it.

Aggregation weights are part of the specification. Wiltshire uses 1990
population "so each tau_e is not overly-affected by large percentage effects in
small counties" (fn 28), and reports that equal weighting gives the same pattern
with different magnitudes. An estimator that silently equal-weights is answering
a different question.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def _staggered(n_treated=6, n_donors=8, T=12, cohorts=(6, 8), effect=0.0,
               seed=0):
    """Staggered adoption with a common factor, so donors can span the treated.

    ``effect`` is a proportional post-treatment shift on the treated units, so
    the true normalised gap is known up to noise.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    rows = []
    for j in range(n_donors):
        load = rng.normal(size=2)
        base = 100.0 + 20.0 * rng.random()
        for t in range(T):
            rows.append({"unit": f"d{j}", "t": t, "adopt": np.nan,
                         "y": float(base + F[t] @ load + rng.normal() * 0.05),
                         "w": 1.0})
    for i in range(n_treated):
        g = cohorts[i % len(cohorts)]
        load = rng.normal(size=2)
        base = 100.0 + 20.0 * rng.random()
        for t in range(T):
            y = base + F[t] @ load + rng.normal() * 0.05
            if t >= g:
                y *= (1.0 + effect)
            rows.append({"unit": f"x{i}", "t": t, "adopt": float(g),
                         "y": float(y), "w": float(1 + i)})
    d = pd.DataFrame(rows)
    d["treat"] = ((~d.adopt.isna()) & (d.t >= d.adopt)).astype(int)
    return d


def _fit(df=None, **over):
    from mlsynth import STACKEDSC

    cfg = {"df": _staggered() if df is None else df, "outcome": "y",
           "treat": "treat", "unitid": "unit", "time": "t",
           "display_graphs": False}
    cfg.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STACKEDSC(cfg).fit()


# ---------------------------------------------------------------------------
class TestSmoke:
    def test_it_fits_and_returns_the_contract_type(self):
        from mlsynth.config_models import EffectResult

        assert isinstance(_fit(), EffectResult)

    def test_the_headline_quantities_are_finite(self):
        res = _fit()
        assert np.isfinite(res.effects.att)
        assert np.isfinite(np.asarray(res.event_study.tau, float)).any()

    def test_it_reports_one_fit_per_treated_unit(self):
        res = _fit()
        assert len(res.per_unit) == 6


# ---------------------------------------------------------------------------
class TestTheNormalisation:
    """The constraint the base-year indexing imposes, and its consequences."""

    def test_the_gap_at_the_base_period_is_exactly_zero(self):
        """Every series is 100 at its own T0i and the weights sum to one, so
        the synthetic reproduces the treated level there identically. A wrong
        base year breaks this immediately; nothing else in the output does.
        """
        res = _fit()
        e = np.asarray(res.event_study.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        assert abs(float(tau[e == -1][0])) < 1e-9

    def test_each_cohort_uses_its_own_base_year(self):
        """Two cohorts adopting at different times must be indexed to different
        periods. Indexing both to a common year is a different estimator, and
        it shows up as a nonzero gap at e = -1 for at least one cohort."""
        res = _fit(df=_staggered(cohorts=(5, 9)))
        for u in res.per_unit.values():
            e = np.asarray(u.horizons, int)
            g = np.asarray(u.tau, float)
            assert abs(float(g[e == -1][0])) < 1e-9, u.label

    def test_the_effects_are_in_percent_of_the_base_level(self):
        """A 10 percent post shift on every treated unit should read as about
        10, not as a level difference in the outcome's own units."""
        res = _fit(df=_staggered(effect=0.10, seed=3))
        e = np.asarray(res.event_study.horizons, int)
        tau = np.asarray(res.event_study.tau, float)
        post = tau[e >= 0]
        assert 7.0 < float(np.mean(post)) < 13.0


# ---------------------------------------------------------------------------
class TestAggregationWeights:
    def test_equal_weighting_is_the_default(self):
        res = _fit()
        assert res.method_details.parameters_used["agg_weights"] is None

    def test_a_supplied_weight_column_changes_the_average(self):
        plain = _fit()
        weighted = _fit(agg_weights="w")
        assert plain.effects.att != pytest.approx(weighted.effects.att,
                                                  rel=1e-9)

    def test_a_degenerate_weight_reproduces_the_unit_it_selects(self):
        """Putting all the mass on one treated unit must return that unit's own
        path -- the aggregation is a weighted mean, so this is exact."""
        df = _staggered()
        df["only"] = (df.unit == "x2").astype(float)
        res = _fit(df=df, agg_weights="only")
        target = _fit(df=df).per_unit["x2"]
        np.testing.assert_allclose(
            np.asarray(res.event_study.tau, float),
            np.asarray(target.tau, float), atol=1e-8)


# ---------------------------------------------------------------------------
class TestBiasCorrection:
    def test_it_is_off_by_default(self):
        from mlsynth.utils.stackedsc_helpers.config import STACKEDSCConfig

        assert STACKEDSCConfig.model_fields["bias_correct"].default is False

    def test_it_moves_the_estimate_when_predictors_are_supplied(self):
        df = _staggered()
        df["x1"] = df.groupby("unit").y.transform("mean")
        plain = _fit(df=df, covariates=["x1"])
        corrected = _fit(df=df, covariates=["x1"], bias_correct=True)
        assert plain.effects.att != pytest.approx(corrected.effects.att,
                                                  rel=1e-9)


# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_a_single_treated_unit_works(self):
        res = _fit(df=_staggered(n_treated=1, cohorts=(7,)))
        assert len(res.per_unit) == 1
        assert np.isfinite(res.effects.att)

    def test_a_single_donor_takes_all_the_weight(self):
        res = _fit(df=_staggered(n_donors=1))
        for u in res.per_unit.values():
            assert sum(u.donor_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_every_cohort_is_represented(self):
        res = _fit(df=_staggered(n_treated=6, cohorts=(5, 7, 9)))
        assert {u.adoption_time for u in res.per_unit.values()} == {5, 7, 9}


# ---------------------------------------------------------------------------
class TestFailuresAreReported:
    def test_no_never_treated_donors_is_an_error(self):
        from mlsynth.exceptions import MlsynthDataError

        df = _staggered(n_donors=0)
        with pytest.raises((MlsynthDataError, ValueError)):
            _fit(df=df)

    def test_a_missing_weight_column_is_an_error(self):
        from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

        with pytest.raises((MlsynthConfigError, MlsynthDataError, ValueError),
                           match="nope"):
            _fit(agg_weights="nope")

    def test_a_weight_that_varies_within_a_unit_is_an_error(self):
        """An aggregation weight is a property of the treated unit. If it moves
        over time there is no single value to weight by, and silently taking
        the first would be a guess."""
        from mlsynth.exceptions import MlsynthDataError

        df = _staggered()
        df.loc[df.unit == "x0", "w"] = np.arange((df.unit == "x0").sum())
        with pytest.raises((MlsynthDataError, ValueError)):
            _fit(df=df, agg_weights="w")

    def test_bias_correction_without_covariates_is_an_error(self):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises((MlsynthConfigError, ValueError), match="covariat"):
            _fit(bias_correct=True)

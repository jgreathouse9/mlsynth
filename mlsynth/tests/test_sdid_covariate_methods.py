"""How SDID uses covariates: two methods, named explicitly.

There are two defensible ways to give SDID covariates, and mlsynth now supports
both. They are different estimators and the point of this file is that they can
never be confused for one another.

``adjust`` -- Kranz (2022), the ``xsynthdid`` two-step
    Fit ``outcome ~ covariates | unit + time`` on the untreated rows, subtract
    ``X @ beta`` from the outcome across the whole panel, then run ordinary
    SDID. The weight problem never sees a covariate. Shipped in #309.

``match`` -- de Brabander, Juodis & Miyazato Szini (2025), eqs. 11-12
    Covariates enter *inside* the unit-weight problem, ADH-style: the treated
    unit's demeaned pre-treatment outcomes are stacked with its covariates, the
    donors likewise, and a diagonal ``V`` weighting the stacked rows is chosen
    by nested optimisation against the pre-treatment outcome fit. The outcome is
    never modified.

Because ``SDIDConfig.covariates`` already meant ``adjust`` before this change,
the collision risk is real and immediate: anyone on #309 passing
``covariates=["x"]`` must not silently get a different estimator. The chosen
guard is structural rather than defensive -- covariates are now a dict keyed by
method, so the method cannot be omitted, and the old list form is a type error
at construction rather than a changed result.

The dict also makes the combined case expressible, which a flat selector could
not: ``{"adjust": ["seasonal"], "match": ["gdp_pc"]}`` residualises the outcome
for seasonality while matching donors on income. Those are different jobs and
there is no reason to force a choice.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

_GOLD = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
         / "sdid_kranz")


@pytest.fixture(scope="module")
def panel():
    """The #309 fixture: xsynthdid's own worked example, with a covariate."""
    p = _GOLD / "panel.csv"
    if not p.exists():  # pragma: no cover - gold is committed
        pytest.skip(f"missing {p}")
    return pd.read_csv(p)


def _att(df, **extra):
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID({
            "df": df, "outcome": "y", "treat": "treat_exp", "unitid": "i",
            "time": "t", "vce": "noinference", "display_graphs": False,
            **extra}).fit().effects.att)


class TestTheOldFormIsRejectedLoudly:
    """A list is no longer a valid value, and that is the point.

    Silently reinterpreting ``covariates=["x"]`` would change which estimator
    runs for every existing caller. A type error at construction is the cheapest
    possible way to make that impossible.
    """

    def test_a_bare_list_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError):
            _att(panel, covariates=["x"])

    def test_the_error_explains_the_dict_form(self, panel):
        """A caller upgrading from #309 should learn the fix from the message."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError) as exc:
            _att(panel, covariates=["x"])
        msg = str(exc.value)
        assert "adjust" in msg and "match" in msg

    def test_none_still_means_no_covariates(self, panel):
        assert _att(panel, covariates=None) == _att(panel)


class TestAdjustReproducesTheKranzBehaviour:
    """``{"adjust": [...]}`` is exactly what ``covariates=[...]`` used to do."""

    def test_it_matches_the_frozen_gold(self, panel):
        """Pinned against the #309 reference, not merely against itself."""
        txt = (_GOLD / "gold.txt").read_text().splitlines()
        gold = {k.strip(): v.strip()
                for k, v in (l.split(":", 1) for l in txt if ":" in l)}
        got = _att(panel, covariates={"adjust": ["x"]})
        assert got == pytest.approx(
            float(gold["tau_adjusted_converged"]), abs=1e-6)

    def test_it_equals_adjusting_the_outcome_by_hand(self, panel):
        """Independent of any gold: the two routes must coincide exactly."""
        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            by_hand = float(SDID({
                "df": panel, "outcome": "y.adj", "treat": "treat_exp",
                "unitid": "i", "time": "t", "vce": "noinference",
                "display_graphs": False}).fit().effects.att)
        assert _att(panel, covariates={"adjust": ["x"]}) == pytest.approx(
            by_hand, abs=1e-9)


class TestMatchIsADifferentEstimator:
    """``{"match": [...]}`` leaves the outcome alone and moves the weights."""

    def test_it_does_not_equal_adjust(self, panel):
        """The load-bearing separation. If these ever coincide, one branch is
        not doing what its name says."""
        a = _att(panel, covariates={"adjust": ["x"]})
        m = _att(panel, covariates={"match": ["x"]})
        assert abs(a - m) > 1e-6

    def test_it_does_not_equal_plain_sdid(self, panel):
        """Matching on a covariate must actually change the donor weights."""
        assert abs(_att(panel, covariates={"match": ["x"]}) - _att(panel)) > 1e-6

    def test_the_outcome_is_untouched(self, panel):
        """``match`` changes weights, not the outcome. Asserted through the
        reported observed series, which must still be the raw ``y``."""
        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({
                "df": panel, "outcome": "y", "treat": "treat_exp", "unitid": "i",
                "time": "t", "vce": "noinference", "display_graphs": False,
                "covariates": {"match": ["x"]}}).fit()
        obs = np.asarray(res.time_series.observed_outcome, dtype=float)
        treated = panel[panel.i.isin(panel.loc[panel.treat_exp == 1, "i"].unique())]
        expected = treated.groupby("t").y.mean().to_numpy()
        np.testing.assert_allclose(obs, expected, rtol=0, atol=1e-9)

    def test_the_weights_stay_on_the_simplex(self, panel):
        from mlsynth import SDID

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDID({
                "df": panel, "outcome": "y", "treat": "treat_exp", "unitid": "i",
                "time": "t", "vce": "noinference", "display_graphs": False,
                "covariates": {"match": ["x"]}}).fit()
        w = np.array(list(res.donor_weights.values()), dtype=float)
        assert w.min() >= -1e-9
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_a_covariate_that_carries_no_information_barely_moves_it(self, panel):
        """A constant covariate is uninformative for matching, so the estimate
        should stay close to plain SDID. Guards against the stacked rows
        swamping the outcome rows through a scaling bug."""
        d = panel.copy()
        d["const"] = 1.0
        assert _att(d, covariates={"match": ["const"]}) == pytest.approx(
            _att(d), abs=0.5)


class TestBothTogether:
    """The case the dict exists to make expressible."""

    def test_adjust_and_match_can_be_combined(self, panel):
        d = panel.copy()
        d["z"] = d.x * 0.5 + d.t * 0.01
        got = _att(d, covariates={"adjust": ["x"], "match": ["z"]})
        assert np.isfinite(got)

    def test_the_combination_differs_from_either_alone(self, panel):
        d = panel.copy()
        d["z"] = d.x * 0.5 + d.t * 0.01
        both = _att(d, covariates={"adjust": ["x"], "match": ["z"]})
        assert abs(both - _att(d, covariates={"adjust": ["x"]})) > 1e-6
        assert abs(both - _att(d, covariates={"match": ["z"]})) > 1e-6

    def test_a_column_under_both_keys_is_refused(self, panel):
        """Adjusting the outcome for a variable and then matching donors on it
        double-counts it. Refused rather than silently allowed, because the
        alternative reading (deliberate) is defensible enough that it must be a
        recorded decision rather than an accident."""
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="both"):
            _att(panel, covariates={"adjust": ["x"], "match": ["x"]})


class TestTheConfigRejectsBadInputEarly:
    """Everything catchable is caught at construction."""

    def test_unknown_key(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError) as exc:
            _att(panel, covariates={"residualise": ["x"]})
        assert "adjust" in str(exc.value) and "match" in str(exc.value)

    def test_empty_dict(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="empty"):
            _att(panel, covariates={})

    @pytest.mark.parametrize("key", ["adjust", "match"])
    def test_empty_list_under_a_key(self, panel, key):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="empty"):
            _att(panel, covariates={key: []})

    @pytest.mark.parametrize("key", ["adjust", "match"])
    def test_unknown_column(self, panel, key):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="ghost"):
            _att(panel, covariates={key: ["ghost"]})

    @pytest.mark.parametrize("key", ["adjust", "match"])
    def test_non_numeric_column(self, panel, key):
        from mlsynth.exceptions import MlsynthConfigError

        d = panel.copy()
        d["label"] = "a"
        with pytest.raises(MlsynthConfigError, match="numeric"):
            _att(d, covariates={key: ["label"]})

    @pytest.mark.parametrize("key", ["adjust", "match"])
    def test_the_outcome_cannot_be_a_covariate(self, panel, key):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="outcome"):
            _att(panel, covariates={key: ["y"]})

    @pytest.mark.parametrize("key", ["adjust", "match"])
    def test_the_treatment_cannot_be_a_covariate(self, panel, key):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="treatment"):
            _att(panel, covariates={key: ["treat_exp"]})

    def test_subgroup_is_still_refused_for_both_methods(self, panel):
        """SC-DDD collapses the panel over the subgroup dimension, which leaves
        no (unit, time) panel for either method. The #309 refusal must hold for
        ``match`` too, not just the branch it was written for."""
        from mlsynth.exceptions import MlsynthConfigError

        d = panel.copy()
        d["band"] = np.where(d.index % 2 == 0, "young", "old")
        for cov in ({"adjust": ["x"]}, {"match": ["x"]}):
            with pytest.raises(MlsynthConfigError, match="subgroup"):
                _att(d, covariates=cov, subgroup="band", target_subgroup="young")


class TestTheDefaultIsUnchanged:
    """No covariates at all must be bit-identical to before."""

    def test_omitting_covariates_is_bit_identical(self, panel):
        assert _att(panel) == _att(panel, covariates=None)

    def test_plain_sdid_still_matches_the_frozen_gold(self, panel):
        txt = (_GOLD / "gold.txt").read_text().splitlines()
        gold = {k.strip(): v.strip()
                for k, v in (l.split(":", 1) for l in txt if ":" in l)}
        assert _att(panel) == pytest.approx(
            float(gold["tau_unadjusted_converged"]), abs=1e-6)

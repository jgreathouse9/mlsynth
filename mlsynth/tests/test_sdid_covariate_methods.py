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

Two panels, for a reason. Most assertions run on the ``#309`` xsdid.mc fixture,
which is where ``adjust`` is pinned against frozen gold. Anything asking whether
covariates *bind* runs on the Brexit panel instead, because the xsdid fixture
violates the convex-hull condition and makes those assertions vacuous --
``TestWhyThoseTestsMovedToBrexit`` asserts that rather than leaving it as a
claim.

This file and ``test_sdid_match_seam.py`` are complementary, not redundant, and
mutation testing shows it: disabling the nested ``V`` search survives here and
is killed there, while removing the outcome demeaning is killed here and
survives there (the seam bundle hands over already-demeaned matrices). Neither
file alone covers the solver.
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


_BREXIT = (Path(__file__).resolve().parents[2] / "benchmarks" / "reference"
           / "brabander_sdid_match" / "brexit_panel.csv")


@pytest.fixture(scope="module")
def brexit():
    """The paper's own panel: 24 OECD countries, 1995Q1-2020Q4, UK treated 2016Q3.

    Needed because the xsdid.mc fixture above cannot test whether covariates
    bind. There the treated unit sits outside the donor convex hull -- demeaned
    donors span [40.962, 57.561] at the last pre-period, the treated is at
    39.968 -- so the simplex solution is pinned at a vertex and V is irrelevant
    whatever the matching scheme. That is the paper's own section 2.2 condition
    failing, a property of the fixture rather than of the estimator, and it
    makes every "the covariate changes something" assertion vacuous.

    Exported by benchmarks/reference/brabander_sdid_match/reference.R so both
    sides read one file.
    """
    if not _BREXIT.exists():  # pragma: no cover - gold is committed
        pytest.skip(f"missing {_BREXIT}")
    return pd.read_csv(_BREXIT)


def _batt(df, **extra):
    """ATT on the Brexit panel."""
    from mlsynth import SDID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(SDID({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "country",
            "time": "t", "vce": "noinference", "display_graphs": False,
            **extra}).fit().effects.att)


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

    def test_a_constant_covariate_carries_no_information(self, panel):
        """A covariate with no cross-donor variation cannot inform matching.

        On the simplex ``1 - sum(w) == 0``, so a constant row contributes
        nothing to the objective whatever weight ``V`` gives it. Two different
        constants must therefore be interchangeable -- which is the scale-free
        way to say "uninformative", and does not require comparing against a
        different matching scheme.
        """
        d = panel.copy()
        d["c1"] = 1.0
        d["c2"] = 7.5
        assert _att(d, covariates={"match": ["c1"]}) == pytest.approx(
            _att(d, covariates={"match": ["c2"]}), abs=1e-9)

    def test_a_real_covariate_is_not_interchangeable_with_a_constant(self, brexit):
        """The complement, and the test that would catch the covariate row
        being silently dropped: a real covariate varies across donors, so it
        must move the weights somewhere a constant does not.

        On the Brexit panel rather than the fixture above, because this is the
        assertion the convex-hull failure makes vacuous there.
        """
        d = brexit.copy()
        d["c1"] = 1.0
        assert abs(_batt(d, covariates={"match": ["real_con"]})
                   - _batt(d, covariates={"match": ["c1"]})) > 1e-6


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


class TestTheMatchingScheme:
    """How many pre-treatment outcome rows enter the matching problem.

    This is not a tuning knob -- it decides whether covariates can do anything
    at all. Kaul, Klossner, Pfeifer & Schieler (2022) prove that when every
    pre-treatment outcome is among the predictors, the nested-``V`` search
    drives the covariate rows to zero weight and the covariates are irrelevant.
    The paper adopts that result (p. 1629) and answers it by reporting three
    schemes: all pre-treatment periods, half of them, and the last one only.

    So ``match_pre_periods`` defaults to ``"last"``: the only scheme under which
    covariates demonstrably bind, and therefore the only default that does not
    ship a silent no-op.
    """

    def test_all_periods_makes_covariates_irrelevant(self, panel):
        """Kaul's theorem, pinned as behaviour.

        Not an accident of this panel and not something to fix -- it is what
        the nested-``V`` problem implies. Recorded so that a future change
        making covariates 'work' under ``"all"`` is recognised as a departure
        from the theory rather than an improvement.
        """
        d = panel.copy()
        d["c1"] = 1.0
        a = _att(d, covariates={"match": ["x"]}, match_pre_periods="all")
        b = _att(d, covariates={"match": ["c1"]}, match_pre_periods="all")
        assert a == pytest.approx(b, abs=1e-9)

    def test_last_period_makes_covariates_bind(self, brexit):
        """The contrast that justifies the default.

        The reference measures the covariates' share of V at 79.8 percent under
        this scheme against 2.4 percent under "all"; this is the estimator-level
        consequence.
        """
        d = brexit.copy()
        d["c1"] = 1.0
        a = _batt(d, covariates={"match": ["real_con"]}, match_pre_periods="last")
        b = _batt(d, covariates={"match": ["c1"]}, match_pre_periods="last")
        assert abs(a - b) > 1e-6

    def test_the_default_is_last(self, panel):
        assert _att(panel, covariates={"match": ["x"]}) == pytest.approx(
            _att(panel, covariates={"match": ["x"]}, match_pre_periods="last"),
            abs=1e-12)

    @pytest.mark.parametrize("scheme", ["all", "half", "last", 1, 3])
    def test_every_scheme_produces_a_finite_simplex_fit(self, panel, scheme):
        assert np.isfinite(
            _att(panel, covariates={"match": ["x"]}, match_pre_periods=scheme))

    def test_an_integer_selects_that_many_final_pre_periods(self, panel):
        """``1`` and ``"last"`` name the same scheme, so they must agree."""
        assert _att(panel, covariates={"match": ["x"]}, match_pre_periods=1) == (
            pytest.approx(_att(panel, covariates={"match": ["x"]},
                               match_pre_periods="last"), abs=1e-12))

    def test_more_periods_weakens_the_covariate(self, brexit):
        """The mechanism behind Kaul's result, made visible at the estimator.

        The reference shows it in V directly -- 79.8 percent of the weight on
        the covariates with one outcome row, 8.7 with the later half, 2.4 with
        all 86. Here the same gradient is read off the estimate: the further a
        real covariate can move the ATT away from an uninformative one shrinks
        as outcome rows are added.
        """
        d = brexit.copy()
        d["c1"] = 1.0
        gaps = []
        for n in (1, 20, 80):
            gaps.append(abs(
                _batt(d, covariates={"match": ["real_con"]}, match_pre_periods=n)
                - _batt(d, covariates={"match": ["c1"]}, match_pre_periods=n)))
        assert gaps[0] > gaps[-1]


class TestTheSchemeIsOnlyMeaningfulWithMatch:
    """It cannot be set where it would do nothing."""

    def test_setting_it_without_match_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="match"):
            _att(panel, match_pre_periods="last")

    def test_setting_it_with_only_adjust_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="match"):
            _att(panel, covariates={"adjust": ["x"]}, match_pre_periods="last")

    def test_an_unknown_scheme_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError):
            _att(panel, covariates={"match": ["x"]}, match_pre_periods="most")

    def test_a_non_positive_integer_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError):
            _att(panel, covariates={"match": ["x"]}, match_pre_periods=0)

    def test_more_periods_than_the_panel_has_raises(self, panel):
        from mlsynth.exceptions import MlsynthConfigError

        with pytest.raises(MlsynthConfigError, match="pre-treatment"):
            _att(panel, covariates={"match": ["x"]}, match_pre_periods=999)


class TestWhyThoseTestsMovedToBrexit:
    """The convex-hull failure that makes the xsdid.mc fixture unusable here.

    Asserted rather than left as a comment, because it is the whole reason three
    tests above read a different panel. If the fixture ever changes so that the
    treated unit falls inside the hull, these fail and the tests can move back.
    """

    def test_the_treated_unit_is_outside_the_donor_hull(self, panel):
        Y = panel.pivot(index="t", columns="i", values="y")
        treated = sorted(panel.loc[panel.treat_exp == 1, "i"].unique())
        donors = [c for c in Y.columns if c not in treated]
        pre = Y.index[:15]
        dm = Y.loc[pre] - Y.loc[pre].mean(axis=0)
        row = dm.loc[pre[-1], donors].to_numpy()
        t = float(dm.loc[pre[-1], treated].to_numpy().mean())
        assert not (row.min() <= t <= row.max())

    def test_the_brexit_panel_does_not_have_that_problem(self, brexit):
        Y = brexit.pivot(index="t", columns="country", values="y")
        treated = sorted(brexit.loc[brexit.treat == 1, "country"].unique())
        donors = [c for c in Y.columns if c not in treated]
        pre = [t for t in Y.index if t < 227]
        dm = Y.loc[pre] - Y.loc[pre].mean(axis=0)
        row = dm.loc[pre[-1], donors].to_numpy()
        t = float(dm.loc[pre[-1], treated].to_numpy().mean())
        assert row.min() <= t <= row.max()

    def test_the_covariate_is_inert_on_the_degenerate_fixture(self, panel):
        """The consequence, stated as behaviour.

        Not a defect to fix: no convex combination can reach a treated value
        below every donor, so V has nothing to work with. Pinned so that a
        future change appearing to make covariates bind here is recognised as a
        departure from the simplex constraint rather than an improvement.
        """
        d = panel.copy()
        d["c1"] = 1.0
        assert _att(d, covariates={"match": ["x"]}) == pytest.approx(
            _att(d, covariates={"match": ["c1"]}), abs=1e-9)
